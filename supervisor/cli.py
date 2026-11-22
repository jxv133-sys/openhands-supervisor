"""Command line: plan a project, review the plan, run it, check status."""

from __future__ import annotations

import argparse
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
    ok = Supervisor(settings, project, contract, RunLog(project)).run()
    return 0 if ok else 2


def cmd_status(args, settings: Settings) -> int:
    _print_contract(Contract.load(Path(args.project)))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="supervisor", description=__doc__)
    parser.add_argument("--config", default="supervisor.toml", help="settings file (default: ./supervisor.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="draft milestones for a goal")
    p.add_argument("project", help="project folder (created if missing)")
    p.add_argument("goal", help="what to build, in plain language")
    p.add_argument("--force", action="store_true", help="replace an existing plan")
    p.set_defaults(func=cmd_plan)

    r = sub.add_parser("run", help="run (or resume) the milestones")
    r.add_argument("project")
    r.add_argument("-y", "--yes", action="store_true", help="don't ask before starting")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("status", help="show milestone progress")
    s.add_argument("project")
    s.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    settings = Settings.load(Path(args.config))
    return args.func(args, settings)


if __name__ == "__main__":
    sys.exit(main())
