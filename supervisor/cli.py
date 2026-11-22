"""Command line: plan a project, review the plan, run it, check status."""

from __future__ import annotations

import argparse
import os
import re
import signal
import sys
from pathlib import Path

from .config import Settings
from .contract import Contract
from .log import RunLog


def _print_contract(contract: Contract) -> None:
    print(f"Goal: {contract.goal}\n")
    if contract.setup:
        print("Setup:")
        for cmd in contract.setup:
            print(f"  $ {cmd}")
        print()
    for i, m in enumerate(contract.milestones, 1):
        print(f"{i}. [{m.status}] {m.title}")
        if m.instructions:
            print(f"     {m.instructions}")
        print(f"     check: {m.check}")
        if m.restarts:
            print(f"     restarts so far: {m.restarts}")
        for note in m.notes:
            print(f"     note: {note}")


def cmd_plan(args, settings: Settings) -> int:
    from .planner import plan

    project = Path(args.project)
    if Contract.path_for(project).exists() and not args.force:
        print(f"{Contract.path_for(project)} already exists (use --force to replace it).")
        return 1
    project.mkdir(parents=True, exist_ok=True)
    print(f"Planning with {settings.model}...", flush=True)
    contract = plan(args.goal, settings.ollama_url, settings.model, timeout=settings.llm_timeout)
    path = contract.save(project)
    _print_contract(contract)
    print(f"\nSaved to {path}\nReview and edit it (especially the check commands), then run:")
    print(f"  supervisor run {project}")
    return 0


def cmd_run(args, settings: Settings) -> int:
    from .runner import Supervisor

    project = Path(args.project)
    contract = Contract.load(project)
    _print_contract(contract)
    if not args.yes:
        answer = input("\nStart? [Y/n] ").strip().lower()
        if answer not in ("", "y", "yes"):
            return 1
    ok = Supervisor(settings, project, contract, RunLog(project), on_escalate=args.on_escalate).run()
    return 0 if ok else 2


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def read_goal(path: Path) -> str:
    """The whole goal file is the goal, minus <!-- comments -->."""
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()


def cmd_auto(args, settings: Settings) -> int:
    """Unattended mode for `docker compose up`. The goal comes from goal.md
    ($GOAL_FILE), everything else from env vars. Plans on first start (unless
    a plan exists), then runs it."""
    from .planner import plan
    from .runner import ESCALATE_MODES, Supervisor

    project = Path(os.environ.get("PROJECT", "").strip() or "project")
    goal_file = Path(os.environ.get("GOAL_FILE", "").strip() or "goal.md")
    goal = read_goal(goal_file)
    on_escalate = os.environ.get("ON_ESCALATE", "").strip().lower() or "stop"
    if on_escalate not in ESCALATE_MODES:
        print(f"ON_ESCALATE must be one of {', '.join(ESCALATE_MODES)}, not {on_escalate!r}")
        return 1
    if on_escalate == "ask":
        print("Note: ON_ESCALATE=ask needs a terminal; attach with `docker attach <container>` to answer.")

    contract_path = Contract.path_for(project)
    if contract_path.exists():
        contract = Contract.load(project)
        if goal and goal != contract.goal:
            print(
                f"Note: goal.md changed, but project '{project}' already has a plan; continuing that plan.\n"
                f"To start the new goal, set a different PROJECT in .env, or delete {contract_path}.\n"
            )
    else:
        if not goal:
            print(
                f"Nothing to do: {goal_file.name} is missing or empty, and project '{project}' has no plan yet.\n"
                "Write what you want built in goal.md (next to docker-compose.yml), "
                "then run `docker compose up --build` again."
            )
            return 1
        project.mkdir(parents=True, exist_ok=True)
        print(f"Planning '{project}' with {settings.model}...", flush=True)
        contract = plan(goal, settings.ollama_url, settings.model, timeout=settings.llm_timeout)
        contract.save(project)
        if _truthy(os.environ.get("REVIEW_PLAN")):
            _print_contract(contract)
            print(
                f"\nPlan saved to projects/{project}/.supervisor/contract.json (REVIEW_PLAN is on).\n"
                "Review or edit it, then run `docker compose up` again to start."
            )
            return 0

    _print_contract(contract)
    print()
    ok = Supervisor(settings, project, contract, RunLog(project), on_escalate=on_escalate).run()
    return 0 if ok else 2


def cmd_web(args, settings: Settings) -> int:
    from .web import serve

    goal_file = os.environ.get("GOAL_FILE", "").strip()
    serve(settings, Path(args.projects), args.host, args.port, Path(goal_file) if goal_file else None)
    return 0


def cmd_status(args, settings: Settings) -> int:
    _print_contract(Contract.load(Path(args.project)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supervisor", description=__doc__)
    parser.add_argument(
        "--config",
        default=os.environ.get("SUPERVISOR_CONFIG", "supervisor.toml"),
        help="settings file (default: $SUPERVISOR_CONFIG or ./supervisor.toml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="draft milestones for a goal")
    p.add_argument("project", help="project folder (created if missing)")
    p.add_argument("goal", help="what to build, in plain language")
    p.add_argument("--force", action="store_true", help="replace an existing plan")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="run (or resume) the milestones")
    r.add_argument("project")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before starting")
    r.add_argument(
        "--on-escalate",
        choices=["ask", "stop", "skip"],
        default="ask",
        help="when the agent can't finish a milestone: ask you (default), stop, or skip it",
    )
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("web", help="web interface (default in Docker)")
    w.add_argument("--host", default=os.environ.get("WEB_HOST", "0.0.0.0"))
    w.add_argument("--port", type=int, default=int(os.environ.get("WEB_PORT", "8765")))
    w.add_argument("--projects", default=".", help="folder holding the projects (default: current folder)")
    w.set_defaults(func=cmd_web)

    a = sub.add_parser("auto",help="unattended: plan from goal.md if needed, then run (env: PROJECT, GOAL_FILE, REVIEW_PLAN, ON_ESCALATE)")
    a.set_defaults(func=cmd_auto)

    s = sub.add_parser("status", help="show milestone progress")
    s.add_argument("project")
    s.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    settings = Settings.load(Path(args.config))
    _install_sigterm_handler()
    return args.func(args, settings)


def _install_sigterm_handler() -> None:
    """`docker compose down`/Ctrl+C on `up` sends SIGTERM. Turn it into
    KeyboardInterrupt so progress is saved and the agent sandbox container is
    removed instead of being left running."""

    def handler(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, handler)
    except (ValueError, AttributeError):  # not main thread / unsupported platform
        pass


if __name__ == "__main__":
    sys.exit(main())
