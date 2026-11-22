"""Web interface: create projects, review/edit plans, run them, watch live,
and answer the agent when it's stuck. Standard library only.

One run at a time (there's one GPU). Live updates use long-polling on
/api/events, which works through any proxy and needs no extra packages.
"""

from __future__ import annotations

import json
import os
import queue
import re
import threading
import traceback
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .config import Settings
from .contract import DONE, PENDING, SKIPPED, Contract
from .log import RunLog

STATIC_DIR = Path(__file__).parent / "static"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class EventBus:
    """Numbered, bounded event history that clients long-poll with ?after=<id>."""

    def __init__(self, size: int = 3000):
        self._events: deque[dict] = deque(maxlen=size)
        self._next_id = 1
        self._cond = threading.Condition()

    def publish(self, record: dict) -> None:
        with self._cond:
            self._events.append({**record, "id": self._next_id})
            self._next_id += 1
            self._cond.notify_all()

    def since(self, after: int, timeout: float = 25.0) -> tuple[list[dict], int]:
        with self._cond:
            if self._next_id - 1 <= after:
                self._cond.wait(timeout)
            events = [e for e in self._events if e["id"] > after]
            return events, self._next_id - 1


class Job:
    """One supervisor run in a background thread."""

    def __init__(self, app: App, project: str):
        self.app = app
        self.project = project
        self.status = "starting"  # starting, running, stopping, finished, stopped, failed
        self.message = ""
        self.escalation: dict | None = None
        self._answers: queue.Queue[tuple[str, str]] = queue.Queue()
        self.supervisor = None
        self.thread = threading.Thread(target=self._run, name=f"job-{project}", daemon=True)

    def to_dict(self) -> dict:
        return {
            "project": self.project,
            "status": self.status,
            "message": self.message,
            "escalation": self.escalation,
            "active": self.status in ("starting", "running", "stopping"),
        }

    def _set(self, status: str, message: str = "") -> None:
        self.status, self.message = status, message
        self.app.bus.publish({"kind": "job", "project": self.project, **self.to_dict()})

    def _run(self) -> None:
        from .runner import Supervisor

        path = self.app.project_path(self.project)
        try:
            contract = Contract.load(path)
            log = RunLog(path)
            log.add_listener(lambda rec: self.app.bus.publish({**rec, "project": self.project}))
            self.supervisor = Supervisor(self.app.settings, path, contract, log, on_escalate=self._escalation)
            self._set("running")
            ok = self.supervisor.run()
            if self.status == "stopping" or not ok and self.supervisor._stop.is_set():
                self._set("stopped", "Stopped. Progress is saved; press Run to resume.")
            elif ok:
                self._set("finished", "All milestones done.")
            else:
                self._set("stopped", "Run ended before all milestones were done.")
        except Exception as exc:
            traceback.print_exc()
            self._set("failed", f"{type(exc).__name__}: {exc}")

    def _escalation(self, milestone, decision, check) -> tuple[str, str]:
        """Runs in the job thread: show the question in the UI and wait for an answer."""
        self.escalation = {
            "milestone": milestone.title,
            "reason": decision.reason,
            "agent_text": decision.message if decision.message != decision.reason else "",
            "check": milestone.check,
            "check_output": check.output[-3000:],
        }
        self.app.bus.publish({"kind": "job", "project": self.project, **self.to_dict()})
        try:
            return self._answers.get()
        finally:
            self.escalation = None
            self.app.bus.publish({"kind": "job", "project": self.project, **self.to_dict()})

    def answer(self, action: str, text: str) -> None:
        if self.escalation is None:
            raise ApiError(HTTPStatus.CONFLICT, "the agent is not waiting for an answer")
        if action not in ("guidance", "skip", "stop"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "action must be guidance, skip or stop")
        if action == "guidance" and not text.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "write some guidance first")
        if action == "stop":
            self.status = "stopping"
        self._answers.put((action, text.strip()))

    def stop(self) -> None:
        self._set("stopping", "Stopping after the agent's current step...")
        if self.supervisor is not None:
            self.supervisor.request_stop()
        if self.escalation is not None:
            self._answers.put(("stop", ""))


class App:
    def __init__(self, settings: Settings, projects_root: Path, goal_file: Path | None = None):
        self.settings = settings
        self.root = Path(projects_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.goal_file = goal_file
        self.bus = EventBus()
        self.job: Job | None = None
        self._lock = threading.Lock()
        self._planning: set[str] = set()

    # --- helpers -------------------------------------------------------

    def project_path(self, name: str) -> Path:
        if not NAME_RE.match(name or ""):
            raise ApiError(HTTPStatus.BAD_REQUEST, "project names may use letters, digits, '.', '_' and '-'")
        return self.root / name

    def _require_contract(self, name: str) -> Path:
        path = self.project_path(name)
        if not Contract.path_for(path).exists():
            raise ApiError(HTTPStatus.NOT_FOUND, f"no project named {name!r}")
        return path

    def _active_job_for(self, name: str) -> bool:
        return self.job is not None and self.job.project == name and self.job.to_dict()["active"]

    def default_goal(self) -> str:
        if self.goal_file is None:
            return ""
        from .cli import read_goal

        return read_goal(self.goal_file)

    # --- API operations ----------------------------------------------------

    def state(self) -> dict:
        projects = []
        for path in sorted(p for p in self.root.iterdir() if p.is_dir()):
            if not Contract.path_for(path).exists():
                continue
            try:
                c = Contract.load(path)
            except Exception as exc:
                projects.append({"name": path.name, "error": str(exc)})
                continue
            projects.append(
                {
                    "name": path.name,
                    "goal": c.goal,
                    "done": sum(m.status == DONE for m in c.milestones),
                    "skipped": sum(m.status == SKIPPED for m in c.milestones),
                    "total": len(c.milestones),
                    "mtime": Contract.path_for(path).stat().st_mtime,
                }
            )
        return {
            "projects": projects,
            "job": self.job.to_dict() if self.job else None,
            "planning": sorted(self._planning),
            "model": self.settings.model,
            "default_goal": self.default_goal(),
            "last_event": self.bus.since(10**12, timeout=0)[1],
        }

    def project(self, name: str) -> dict:
        path = self._require_contract(name)
        contract = Contract.load(path)
        log = RunLog(path, echo=False).read_all()[-400:]
        return {"name": name, "contract": _contract_dict(contract), "log": log, "running": self._active_job_for(name)}

    def create(self, name: str, goal: str) -> dict:
        from .planner import plan

        path = self.project_path(name)
        goal = (goal or "").strip()
        if not goal:
            raise ApiError(HTTPStatus.BAD_REQUEST, "describe what to build")
        if Contract.path_for(path).exists():
            raise ApiError(HTTPStatus.CONFLICT, f"project {name!r} already exists")
        with self._lock:
            if name in self._planning:
                raise ApiError(HTTPStatus.CONFLICT, "already planning this project")
            self._planning.add(name)
        self.bus.publish({"kind": "planning", "project": name, "active": True})
        try:
            contract = plan(goal, self.settings.ollama_url, self.settings.model, timeout=self.settings.llm_timeout)
        except Exception as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"planning failed: {exc}") from exc
        finally:
            with self._lock:
                self._planning.discard(name)
            self.bus.publish({"kind": "planning", "project": name, "active": False})
        path.mkdir(parents=True, exist_ok=True)
        contract.save(path)
        self.bus.publish({"kind": "contract", "project": name})
        return self.project(name)

    def save(self, name: str, data: dict) -> dict:
        path = self._require_contract(name)
        if self._active_job_for(name):
            raise ApiError(HTTPStatus.CONFLICT, "stop the run before editing the plan")
        try:
            contract = Contract.from_dict(data)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        for m in contract.milestones:
            if m.status not in (PENDING, DONE, SKIPPED):
                raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown status {m.status!r}")
        contract.save(path)
        self.bus.publish({"kind": "contract", "project": name})
        return self.project(name)

    def start(self, name: str) -> dict:
        self._require_contract(name)
        with self._lock:
            if self.job is not None and self.job.to_dict()["active"]:
                raise ApiError(HTTPStatus.CONFLICT, f"'{self.job.project}' is already running; stop it first")
            if not Contract.load(self.project_path(name)).pending():
                raise ApiError(HTTPStatus.CONFLICT, "nothing to do: no pending milestones")
            self.job = Job(self, name)
            self.job.thread.start()
        return self.job.to_dict()

    def stop(self) -> dict:
        if self.job is None or not self.job.to_dict()["active"]:
            raise ApiError(HTTPStatus.CONFLICT, "nothing is running")
        self.job.stop()
        return self.job.to_dict()

    def answer(self, action: str, text: str) -> dict:
        if self.job is None:
            raise ApiError(HTTPStatus.CONFLICT, "nothing is running")
        self.job.answer(action, text)
        return self.job.to_dict()


def _contract_dict(contract: Contract) -> dict:
    from dataclasses import asdict

    return asdict(contract)


def make_handler(app: App, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "openhands-supervisor"

        def log_message(self, fmt, *args):  # quiet: the long-poll would flood the console
            pass

        # --- plumbing --------------------------------------------------

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 1_000_000:
                raise ApiError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError as exc:
                raise ApiError(HTTPStatus.BAD_REQUEST, "invalid JSON") from exc
            if not isinstance(data, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "expected a JSON object")
            return data

        def _authorized(self) -> bool:
            return not token or self.headers.get("X-Token") == token

        def _dispatch(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if method == "GET" and parts in ([], ["index.html"]):
                    body = (STATIC_DIR / "index.html").read_bytes()
                    return self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                if not parts or parts[0] != "api":
                    raise ApiError(HTTPStatus.NOT_FOUND, "not found")
                if not self._authorized():
                    raise ApiError(HTTPStatus.UNAUTHORIZED, "token required")
                route = parts[1:]

                if method == "GET" and route == ["state"]:
                    return self._json(HTTPStatus.OK, app.state())
                if method == "GET" and route == ["events"]:
                    query = parse_qs(url.query)
                    after = int((query.get("after") or ["0"])[0])
                    events, last = app.bus.since(after)
                    return self._json(HTTPStatus.OK, {"events": events, "last": last})
                if method == "POST" and route == ["projects"]:
                    body = self._body()
                    return self._json(HTTPStatus.CREATED, app.create(str(body.get("name", "")), str(body.get("goal", ""))))
                if len(route) == 2 and route[0] == "projects":
                    if method == "GET":
                        return self._json(HTTPStatus.OK, app.project(route[1]))
                    if method == "PUT":
                        return self._json(HTTPStatus.OK, app.save(route[1], self._body().get("contract") or {}))
                if method == "POST" and len(route) == 3 and route[0] == "projects" and route[2] == "run":
                    return self._json(HTTPStatus.OK, app.start(route[1]))
                if method == "POST" and route == ["stop"]:
                    return self._json(HTTPStatus.OK, app.stop())
                if method == "POST" and route == ["answer"]:
                    body = self._body()
                    return self._json(HTTPStatus.OK, app.answer(str(body.get("action", "")), str(body.get("text", ""))))
                raise ApiError(HTTPStatus.NOT_FOUND, "not found")
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # never kill the server thread
                traceback.print_exc()
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_PUT(self):
            self._dispatch("PUT")

    return Handler


def serve(settings: Settings, projects_root: Path, host: str, port: int, goal_file: Path | None = None) -> None:
    token = os.environ.get("WEB_TOKEN") or None
    app = App(settings, projects_root, goal_file)
    server = ThreadingHTTPServer((host, port), make_handler(app, token))
    server.daemon_threads = True
    shown = "localhost" if host in ("0.0.0.0", "") else host
    print(f"Supervisor web UI on http://{shown}:{port}  (projects in {app.root})", flush=True)
    if token:
        print("WEB_TOKEN is set: the page will ask for it once.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if app.job is not None and app.job.to_dict()["active"]:
            print("Stopping the running job...", flush=True)
            app.job.stop()
            app.job.thread.join(timeout=60)
        server.server_close()
