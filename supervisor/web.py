"""Web interface: create projects, review/edit plans, run them, watch live,
and answer the agent when it's stuck. Standard library only.

One run at a time (there's one GPU). Live updates use long-polling on
/api/events, which works through any proxy and needs no extra packages.
"""

from __future__ import annotations

import hmac
import json
import os
import queue
import re
import shutil
import tempfile
import threading
import zipfile
import time
import traceback
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import urllib.parse
from urllib.parse import parse_qs, urlparse

from .config import Settings
from .contract import DONE, PENDING, SKIPPED, STATE_DIR, Contract
from .log import RunLog
from .review import load_report

STATIC_DIR = Path(__file__).parent / "static"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
FILE_VIEW_LIMIT = 512 * 1024


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
            last = self._next_id - 1
            if after == last:  # nothing new yet (after > last: we restarted, answer now)
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
        self.started_at = time.time()
        self.ended_at: float | None = None
        self._answers: queue.Queue[tuple[str, str]] = queue.Queue()
        self._lock = threading.Lock()
        self._stop_requested = False
        self.supervisor = None
        self.thread = threading.Thread(target=self._run, name=f"job-{project}", daemon=True)

    def to_dict(self) -> dict:
        return {
            "project": self.project,
            "status": self.status,
            "message": self.message,
            "escalation": self.escalation,
            "active": self.status in ("starting", "running", "stopping"),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "milestone": getattr(self.supervisor, "_milestone_n", 0) or None,
        }

    def _publish(self) -> None:
        self.app.bus.publish({"kind": "job", "project": self.project, "ts": _now(), **self.to_dict()})

    def _set(self, status: str, message: str = "") -> None:
        self.status, self.message = status, message
        if status in ("finished", "stopped", "failed"):
            self.ended_at = time.time()
        self._publish()

    def _run(self) -> None:
        from .runner import Supervisor

        path = self.app.project_path(self.project)
        try:
            contract = Contract.load(path)
            log = RunLog(path)
            log.add_listener(lambda rec: self.app.bus.publish({**rec, "project": self.project}))
            supervisor = Supervisor(self.app.settings, path, contract, log, on_escalate=self._escalation)
            with self._lock:
                self.supervisor = supervisor
                stop_early = self._stop_requested
            if stop_early:  # Stop was pressed while we were still starting
                supervisor.request_stop()
            else:
                self._set("running")
            ok = supervisor.run()
            infra = getattr(supervisor, "infra_error", "")
            if infra:  # the sandbox wasn't usable: say that, not "the project failed"
                self._set("failed", f"Sandbox problem; nothing was judged. {infra[:600]}")
            elif self._stop_requested or (not ok and supervisor._stop.is_set()):
                self._set("stopped", "Stopped. Progress is saved; press Run to resume.")
            elif ok:
                self._set("finished", "All milestones done." + _review_summary(path))
            else:
                c = supervisor.contract if hasattr(supervisor, "contract") else contract
                skipped = sum(m.status == SKIPPED for m in c.milestones)
                pending = len(c.pending())
                parts = [f"{len(c.completed())}/{len(c.milestones)} milestones done"]
                if skipped:
                    parts.append(f"{skipped} skipped")
                if pending:
                    parts.append(f"{pending} still pending")
                self._set("finished" if not pending else "stopped", "Run ended: " + ", ".join(parts) + "." + _review_summary(path))
        except Exception as exc:
            traceback.print_exc()
            self._set("failed", f"{type(exc).__name__}: {exc}")

    def _escalation(self, milestone, decision, check) -> tuple[str, str]:
        """Runs in the job thread: show the question in the UI and wait for an answer."""
        if self._stop_requested:  # Stop was pressed during the last check: don't ask
            return ("stop", "")
        while True:  # drop answers left over from an earlier question (double clicks)
            try:
                self._answers.get_nowait()
            except queue.Empty:
                break
        self.escalation = {
            "milestone": milestone.title,
            "reason": decision.reason,
            "agent_text": decision.message if decision.message != decision.reason else "",
            "check": milestone.check,
            "check_output": check.output[-3000:],
            "restarts": getattr(milestone, "restarts", 0),
            "since": time.time(),
        }
        self._publish()
        try:
            return self._answers.get()
        finally:
            self.escalation = None
            self._publish()

    def answer(self, action: str, text: str) -> None:
        if action not in ("guidance", "skip", "stop"):
            raise ApiError(HTTPStatus.BAD_REQUEST, "action must be guidance, skip or stop")
        if action == "guidance" and not text.strip():
            raise ApiError(HTTPStatus.BAD_REQUEST, "write some guidance first")
        with self._lock:
            # One answer per question: a second click (or a second browser tab)
            # must not leave an answer queued for the next question.
            if self.escalation is None or self.escalation.get("answered"):
                raise ApiError(HTTPStatus.CONFLICT, "the agent is not waiting for an answer")
            self.escalation["answered"] = action
        if action == "stop":
            self._stop_requested = True
            self._set("stopping", "Stopping...")
        self._answers.put((action, text.strip()))

    def stop(self) -> None:
        with self._lock:
            self._stop_requested = True
            supervisor = self.supervisor
        self._set("stopping", "Stopping after the agent's current step...")
        if supervisor is not None:
            supervisor.request_stop()
        escalation = self.escalation
        if escalation is not None and not escalation.get("answered"):
            escalation["answered"] = "stop"
            self._answers.put(("stop", ""))


class App:
    def __init__(self, settings: Settings, projects_root: Path, goal_file: Path | None = None,
                 config_path: Path | None = None):
        from .settings_page import OVERRIDES_FILE

        self.settings = settings
        self.root = Path(projects_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Settings changed in the UI live next to the projects (supervisor.toml
        # is often mounted read-only) and apply on top of it.
        self.config_path = config_path
        self.overrides_path = self.root / OVERRIDES_FILE
        if config_path is not None:
            self.settings = Settings.load(config_path, self.overrides_path)
        self.goal_file = goal_file
        self.bus = EventBus()
        self.job: Job | None = None
        self._lock = threading.Lock()
        # name -> token of the request planning it; cancelling drops the entry, and
        # only the request holding the current token may clear it or save a plan.
        self._planning: dict[str, object] = {}
        self._chatting: set[str] = set()

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

    def _live(self, name: str) -> dict | None:
        """What the agent is writing right now, for a page that opens mid-reply."""
        if not self._active_job_for(name) or self.job.supervisor is None:
            return None
        live = getattr(self.job.supervisor, "live", None)
        return live.snapshot() if live is not None else None

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
        run_log = RunLog(path, echo=False)
        return {
            "name": name,
            "contract": _contract_dict(contract),
            "log": run_log.read_all()[-400:],
            "chat": run_log.read_chat(600),
            "review": load_report(path),
            "live": self._live(name),
            "context": getattr(self.job.supervisor, "context", None) if self._active_job_for(name) else None,
            "final_review": self.settings.final_review,
            "running": self._active_job_for(name),
        }

    def files(self, name: str) -> dict:
        """Project files (build output and our state excluded), plus when the
        supervisor last saw each one change, from the turn records."""
        from .progress import _iter_files

        root = self._require_contract(name)
        files = []
        for p in _iter_files(root):
            try:
                st = p.stat()
            except OSError:
                continue
            files.append({"path": p.relative_to(root).as_posix(), "size": st.st_size, "mtime": st.st_mtime})
            if len(files) >= 3000:
                break
        files.sort(key=lambda f: f["path"])
        touched: dict[str, str] = {}
        for rec in RunLog(root, echo=False).read_all():
            if rec.get("kind") == "turn":
                for changed in rec.get("changed") or []:
                    touched[changed] = rec.get("ts", "")
        return {"files": files, "touched": touched}

    def assist_goal(self, data: dict) -> dict:
        """One round of shaping a new project's goal: a better draft, plus
        clarifying questions. Stateless: the page sends the whole Q&A so far."""
        from .planner import assist_goal

        idea = str(data.get("idea", "")).strip()
        if not idea:
            raise ApiError(HTTPStatus.BAD_REQUEST, "describe what you want to build first")
        answers = [a for a in (data.get("answers") or []) if isinstance(a, dict)][:40]
        try:
            return assist_goal(idea, answers, str(data.get("goal", "")), self.settings.ollama_url,
                               self.settings.model, timeout=self.settings.llm_timeout)
        except Exception as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"the assistant failed: {exc}") from exc

    def draft_milestone(self, name: str, data: dict) -> dict:
        """Ask the model for one new milestone (with a check command) from a
        plain description. Nothing is saved: the page adds it to the plan for
        you to review. `contract` lets it see unsaved edits on the page."""
        from .planner import draft_milestone
        from .progress import file_tree

        path = self._require_contract(name)
        description = str(data.get("description", "")).strip()
        title = str(data.get("title", "")).strip()
        instructions = str(data.get("instructions", "")).strip()
        report = str(data.get("report", "")).strip()
        logs = str(data.get("logs", ""))[-200_000:]
        if not (description or title or instructions or report or logs.strip()):
            raise ApiError(HTTPStatus.BAD_REQUEST, "describe what the milestone should do")
        try:
            contract = Contract.from_dict(data["contract"]) if data.get("contract") else Contract.load(path)
        except (ValueError, TypeError, AttributeError, KeyError):
            contract = Contract.load(path)  # half-edited plan on the page: fall back to the saved one
        try:
            return draft_milestone(contract, description, file_tree(path), self.settings.ollama_url,
                                   self.settings.model, title=title, instructions=instructions,
                                   timeout=self.settings.llm_timeout, report=report, logs=logs)
        except Exception as exc:
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"drafting failed: {exc}") from exc

    # --- ideas chat -----------------------------------------------------------

    def ideas_stream(self, chat_id: str, text: str):
        """Yields NDJSON-able events for one message in an ideas chat (see ideas.py)."""
        from .ideas import IdeaStore, Tools, chat_turn

        text = (text or "").strip()
        if not text:
            raise ApiError(HTTPStatus.BAD_REQUEST, "write a message first")
        store = IdeaStore(self.root)
        try:
            chat = store.new() if chat_id == "new" else store.get(chat_id)
        except KeyError as exc:
            raise ApiError(HTTPStatus.NOT_FOUND, "no such chat") from exc
        with self._lock:
            if chat["id"] in self._chatting:
                raise ApiError(HTTPStatus.CONFLICT, "this chat is still answering")
            self._chatting.add(chat["id"])

        def events():
            try:
                yield {"type": "chat", "id": chat["id"]}
                busy = self.job is not None and self.job.to_dict()["active"]
                tools = Tools(self, allow_web=self.settings.chat_web)
                turn = chat_turn(chat, text, self.settings, tools, busy=busy)
                try:
                    yield from turn
                finally:
                    turn.close()
                    store.save(chat)  # also after Stop: the partial reply is kept
                yield {"type": "done", "title": chat["title"]}
            finally:
                with self._lock:
                    self._chatting.discard(chat["id"])

        return events()

    def cancel_planning(self, name: str) -> dict:
        """Forget a project that's still being planned. The model call can't be
        interrupted, so it finishes in the background and its plan is thrown away."""
        with self._lock:
            if self._planning.pop(name, None) is None:
                raise ApiError(HTTPStatus.NOT_FOUND, f"{name!r} isn't being planned")
        self.bus.publish({"kind": "planning", "project": name, "active": False})
        return {"cancelled": name}

    def rename(self, name: str, new_name: str) -> dict:
        path = self._require_contract(name)
        new_path = self.project_path(new_name)
        with self._lock:
            if self._active_job_for(name):
                raise ApiError(HTTPStatus.CONFLICT, "stop the run before renaming the project")
            if name in self._planning or new_name in self._planning:
                raise ApiError(HTTPStatus.CONFLICT, "wait for planning to finish")
            if new_path.exists() or new_path.is_symlink():
                raise ApiError(HTTPStatus.CONFLICT, f"there's already a folder named {new_name!r}")
            if path.is_symlink() or path.resolve().parent != self.root:
                raise ApiError(HTTPStatus.BAD_REQUEST, "refusing to rename outside the projects folder")
            os.rename(path, new_path)
            if self.job is not None and self.job.project == name:
                self.job.project = new_name  # finished/stopped status follows the project
        self.bus.publish({"kind": "contract", "project": name, "renamed_to": new_name})
        return {"renamed": name, "to": new_name}

    def download_file(self, name: str, rel: str) -> Path:
        root = self._require_contract(name).resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "no such file")
        return target

    def archive(self, name: str):
        """The project as a .zip in a temp file (rewound), without build output,
        downloaded dependencies or the supervisor's state. Returns (file, size)."""
        from .progress import IGNORED_DIRS

        skip = IGNORED_DIRS - {".git"}  # keep the history if the agent made one
        root = self._require_contract(name).resolve()
        out = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for folder, dirs, files in os.walk(root):
                dirs[:] = sorted(d for d in dirs if d not in skip and not os.path.islink(os.path.join(folder, d)))
                for f in sorted(files):
                    full = os.path.join(folder, f)
                    if os.path.islink(full):
                        continue  # could point outside the project
                    try:
                        zf.write(full, os.path.join(name, os.path.relpath(full, root)))
                    except OSError:
                        continue  # unreadable (e.g. owned by the sandbox's user): skip it
        size = out.tell()
        out.seek(0)
        return out, size

    def delete(self, name: str, keep_code: bool) -> dict:
        """Remove a project from the supervisor (its .supervisor folder), or,
        unless keep_code, the whole project folder with its code."""
        path = self._require_contract(name)
        with self._lock:
            if self._active_job_for(name):
                raise ApiError(HTTPStatus.CONFLICT, "stop the run before deleting the project")
            if name in self._planning:
                raise ApiError(HTTPStatus.CONFLICT, "wait for planning to finish")
            # Never follow a symlink out of the projects folder.
            if path.is_symlink() or path.resolve().parent != self.root:
                raise ApiError(HTTPStatus.BAD_REQUEST, "refusing to delete outside the projects folder")
            target = path / STATE_DIR if keep_code else path
            failed: list[str] = []
            shutil.rmtree(target, onexc=lambda fn, p, exc: failed.append(p))
            if self.job is not None and self.job.project == name:
                self.job = None  # its finished/stopped status refers to a project that's gone
        if failed:
            # Usually files the sandbox created as another user. Say exactly what's left.
            raise ApiError(HTTPStatus.INTERNAL_SERVER_ERROR,
                           f"deleted all but {len(failed)} file(s) it may not remove, e.g. {failed[0]}. "
                           f"Remove them on the server, e.g. sudo rm -rf '{target}'")
        self.bus.publish({"kind": "contract", "project": name, "deleted": True})
        return {"deleted": name, "kept_code": keep_code}

    def settings_view(self) -> dict:
        from . import settings_page
        from .modelinfo import _get

        data = settings_page.view(self.config_path, self.overrides_path)
        data["running"] = self.job is not None and self.job.to_dict()["active"]
        try:  # the models Ollama has, for the model field's suggestions
            tags = _get(self.settings.ollama_url.rstrip("/") + "/api/tags", 3)
            data["models"] = sorted(m.get("name", "") for m in tags.get("models", []) if m.get("name"))
        except Exception:
            data["models"] = []
        return data

    def save_settings(self, values: dict) -> dict:
        from . import settings_page

        if not isinstance(values, dict) or not values:
            raise ApiError(HTTPStatus.BAD_REQUEST, "nothing to save")
        try:
            self.settings = settings_page.save(self.config_path, self.overrides_path, values)
        except ValueError as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        return self.settings_view()

    def reset_settings(self) -> dict:
        from . import settings_page

        self.settings = settings_page.reset(self.config_path, self.overrides_path)
        return self.settings_view()

    def stats(self, name: str) -> dict:
        from . import stats

        path = self._require_contract(name)
        data = stats.compute(Contract.load(path), RunLog(path, echo=False).read_all())
        data["max_input_tokens"] = self.settings.max_input_tokens
        return data

    def file(self, name: str, rel: str) -> dict:
        root = self._require_contract(name).resolve()
        target = (root / rel).resolve()
        if root not in target.parents or not target.is_file():
            raise ApiError(HTTPStatus.NOT_FOUND, "no such file")
        size = target.stat().st_size
        data = target.read_bytes()[:FILE_VIEW_LIMIT]
        if b"\x00" in data[:8000]:
            return {"path": rel, "size": size, "binary": True, "text": ""}
        return {
            "path": rel,
            "size": size,
            "binary": False,
            "truncated": size > FILE_VIEW_LIMIT,
            "text": data.decode("utf-8", errors="replace"),
        }

    def create(self, name: str, goal: str) -> dict:
        from .planner import plan

        path = self.project_path(name)
        goal = (goal or "").strip()
        if not goal:
            raise ApiError(HTTPStatus.BAD_REQUEST, "describe what to build")
        if Contract.path_for(path).exists():
            raise ApiError(HTTPStatus.CONFLICT, f"project {name!r} already exists")
        token = object()
        with self._lock:
            if name in self._planning:
                raise ApiError(HTTPStatus.CONFLICT, "already planning this project")
            self._planning[name] = token
        self.bus.publish({"kind": "planning", "project": name, "active": True})
        try:
            contract = plan(goal, self.settings.ollama_url, self.settings.model, timeout=self.settings.llm_timeout)
        except Exception as exc:
            if self._planning.get(name) is not token:
                raise ApiError(HTTPStatus.CONFLICT, "planning was cancelled") from exc
            raise ApiError(HTTPStatus.BAD_GATEWAY, f"planning failed: {exc}") from exc
        finally:
            with self._lock:
                cancelled = self._planning.get(name) is not token
                if not cancelled:
                    del self._planning[name]
            if not cancelled:
                self.bus.publish({"kind": "planning", "project": name, "active": False})
        if cancelled:
            raise ApiError(HTTPStatus.CONFLICT, "planning was cancelled")
        if Contract.path_for(path).exists():  # created meanwhile (CLI, another tab)
            raise ApiError(HTTPStatus.CONFLICT, f"project {name!r} was created while planning")
        path.mkdir(parents=True, exist_ok=True)
        contract.save(path)
        self.bus.publish({"kind": "contract", "project": name})
        return self.project(name)

    def save(self, name: str, data: dict) -> dict:
        path = self._require_contract(name)
        try:
            contract = Contract.from_dict(data)
        except (ValueError, TypeError, AttributeError) as exc:
            raise ApiError(HTTPStatus.BAD_REQUEST, str(exc)) from exc
        for m in contract.milestones:
            if m.status not in (PENDING, DONE, SKIPPED):
                raise ApiError(HTTPStatus.BAD_REQUEST, f"unknown status {m.status!r}")
        with self._lock:  # same lock as start(): a run can't begin mid-save
            if self._active_job_for(name):
                raise ApiError(HTTPStatus.CONFLICT, "stop the run before editing the plan")
            contract.save(path)
        self.bus.publish({"kind": "contract", "project": name})
        return self.project(name)

    def start(self, name: str) -> dict:
        self._require_contract(name)
        with self._lock:
            if self.job is not None and self.job.to_dict()["active"]:
                raise ApiError(HTTPStatus.CONFLICT, f"'{self.job.project}' is already running; stop it first")
            # With nothing pending, Run does just the final review.
            if not Contract.load(self.project_path(name)).pending() and not self.settings.final_review:
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


def _review_summary(path: Path) -> str:
    report = load_report(path)
    if not report:
        return ""
    if report.get("verdict") == "error":
        return " The final review could not run (see the review card)."
    if report.get("verdict") == "environment":
        return f" Final review: {len(report.get('environment') or [])} check(s) couldn't be judged (environment problem, not a regression)."
    if report.get("verdict") == "regressions":
        return f" Final review: {len(report.get('regressions') or [])} milestone(s) broken again."
    reqs = report.get("requirements") or []
    passed = sum(r.get("passed") is True for r in reqs)
    return f" Final review: {passed}/{len(reqs)} requirements pass."


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


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

        def _file(self, f, size: int, filename: str, content_type: str = "application/octet-stream") -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "download"
            quoted = urllib.parse.quote(filename)
            self.send_header("Content-Disposition", f"attachment; filename=\"{safe}\"; filename*=UTF-8''{quoted}")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            shutil.copyfileobj(f, self.wfile, 256 * 1024)

        def _stream(self, events) -> None:
            """One JSON object per line, flushed as it comes; the connection
            closes at the end (HTTP/1.0). If the page goes away (Stop), writing
            fails and closing `events` stops the model."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")  # no buffering behind nginx
            self.end_headers()
            try:
                for event in events:
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                traceback.print_exc()
                try:
                    self.wfile.write((json.dumps({"type": "error", "text": f"{type(exc).__name__}: {exc}"}) + "\n").encode())
                except OSError:
                    pass
            finally:
                events.close()

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
            if not token:
                return True
            given = self.headers.get("X-Token") or ""
            return hmac.compare_digest(given.encode(), token.encode())

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

                if route == ["settings"] and method == "GET":
                    return self._json(HTTPStatus.OK, app.settings_view())
                if route == ["settings"] and method == "PUT":
                    return self._json(HTTPStatus.OK, app.save_settings(self._body().get("values")))
                if route == ["settings", "reset"] and method == "POST":
                    return self._json(HTTPStatus.OK, app.reset_settings())
                if method == "GET" and route == ["model"]:
                    from . import modelinfo

                    fresh = (parse_qs(url.query).get("fresh") or ["0"])[0] == "1"
                    return self._json(HTTPStatus.OK, modelinfo.fetch(app.settings, use_cache=not fresh))
                if method == "GET" and route == ["state"]:
                    return self._json(HTTPStatus.OK, app.state())
                if method == "GET" and route == ["events"]:
                    query = parse_qs(url.query)
                    try:
                        after = int((query.get("after") or ["0"])[0])
                    except ValueError as exc:
                        raise ApiError(HTTPStatus.BAD_REQUEST, "after must be a number") from exc
                    events, last = app.bus.since(after)
                    return self._json(HTTPStatus.OK, {"events": events, "last": last})
                if route[:1] == ["ideas"]:
                    from .ideas import IdeaStore

                    store = IdeaStore(app.root)
                    try:
                        if method == "GET" and len(route) == 1:
                            return self._json(HTTPStatus.OK, {"chats": store.list(), "model": app.settings.model})
                        if method == "GET" and len(route) == 2:
                            return self._json(HTTPStatus.OK, store.get(route[1]))
                        if method == "DELETE" and len(route) == 2:
                            store.delete(route[1])
                            return self._json(HTTPStatus.OK, {"deleted": route[1]})
                    except KeyError as exc:
                        raise ApiError(HTTPStatus.NOT_FOUND, "no such chat") from exc
                    if method == "POST" and len(route) == 3 and route[2] == "message":
                        return self._stream(app.ideas_stream(route[1], str(self._body().get("text", ""))))
                    raise ApiError(HTTPStatus.NOT_FOUND, "not found")
                if method == "POST" and route == ["goal", "assist"]:
                    return self._json(HTTPStatus.OK, app.assist_goal(self._body()))
                if method == "POST" and route == ["projects"]:
                    body = self._body()
                    return self._json(HTTPStatus.CREATED, app.create(str(body.get("name", "")), str(body.get("goal", ""))))
                if method == "POST" and len(route) == 3 and route[0] == "projects" and route[2] == "rename":
                    return self._json(HTTPStatus.OK, app.rename(route[1], str(self._body().get("name", "")).strip()))
                if method == "POST" and len(route) == 3 and route[0] == "planning" and route[2] == "cancel":
                    return self._json(HTTPStatus.OK, app.cancel_planning(route[1]))
                if method == "GET" and len(route) == 3 and route[0] == "projects" and route[2] == "download":
                    rel = (parse_qs(url.query).get("path") or [""])[0]
                    target = app.download_file(route[1], rel)
                    with target.open("rb") as f:
                        return self._file(f, target.stat().st_size, target.name)
                if method == "GET" and len(route) == 3 and route[0] == "projects" and route[2] == "archive":
                    f, size = app.archive(route[1])
                    with f:
                        return self._file(f, size, f"{route[1]}.zip", "application/zip")
                if len(route) == 2 and route[0] == "projects":
                    if method == "DELETE":
                        keep = (parse_qs(url.query).get("keep_code") or ["0"])[0] == "1"
                        return self._json(HTTPStatus.OK, app.delete(route[1], keep))
                    if method == "GET":
                        return self._json(HTTPStatus.OK, app.project(route[1]))
                    if method == "PUT":
                        return self._json(HTTPStatus.OK, app.save(route[1], self._body().get("contract") or {}))
                if method == "POST" and len(route) == 3 and route[0] == "projects" and route[2] == "run":
                    return self._json(HTTPStatus.OK, app.start(route[1]))
                if method == "GET" and len(route) == 3 and route[0] == "projects" and route[2] == "files":
                    return self._json(HTTPStatus.OK, app.files(route[1]))
                if method == "POST" and route[:1] == ["projects"] and route[2:] == ["milestones", "draft"] and len(route) == 4:
                    return self._json(HTTPStatus.OK, app.draft_milestone(route[1], self._body()))
                if method == "GET" and len(route) == 3 and route[0] == "projects" and route[2] == "stats":
                    return self._json(HTTPStatus.OK, app.stats(route[1]))
                if method == "GET" and len(route) == 3 and route[0] == "projects" and route[2] == "file":
                    rel = (parse_qs(url.query).get("path") or [""])[0]
                    return self._json(HTTPStatus.OK, app.file(route[1], rel))
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

        def do_DELETE(self):
            self._dispatch("DELETE")

    return Handler


def serve(settings: Settings, projects_root: Path, host: str, port: int, goal_file: Path | None = None,
          config_path: Path | None = None) -> None:
    token = os.environ.get("WEB_TOKEN") or None
    app = App(settings, projects_root, goal_file, config_path=config_path)
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
