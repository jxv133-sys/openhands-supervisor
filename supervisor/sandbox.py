"""One execution lifecycle for everything the supervisor runs in the sandbox.

Setup, milestone checks and final-review checks all go through `Sandbox.run`,
so they execute the same way: same working directory, same shell, same
readiness guarantees. This exists because checks used to pass during
milestones and then "regress" in a review that ran in a fresh sandbox (see
docs/check-execution.md). The causes, each handled here:

1. Undeclared environment. The agent can change the sandbox outside the
   project folder (a symlink, a globally installed package, a tool). Checks
   that depend on that pass in the agent's sandbox and fail in any fresh one
   (a resumed run, a review-only run). `fingerprint()`/`drift()` record such
   changes so they can be declared (in the project's `.openhands/setup.sh`,
   which every fresh sandbox runs) and so a review can tell "the environment
   differs" from "the project regressed".
2. Silent setup failures. Setup errors used to be logged and ignored, so
   checks ran in a half-initialized sandbox. `prepare()` fails loudly.
3. Unverified mounts. Nothing proved the project folder was visible in the
   sandbox. `prepare()` and every `run()` verify a token file the supervisor
   wrote on the host.
4. Infrastructure errors reported as test failures. A sandbox/API exception
   or an unreachable project folder now yields `CheckResult.infra`, which
   callers report as an infrastructure error, never as a regression.
"""

from __future__ import annotations

import secrets
import shlex
import time
from pathlib import Path

from .watchdog import CheckResult

AGENT_SETUP = ".openhands/setup.sh"  # OpenHands' convention: run at every sandbox start
_TOKEN_DIR = ".supervisor"

# Exit codes the wrapper reserves for "the environment isn't usable", always
# paired with a marker line, so a check exiting with the same code by chance
# isn't mistaken for an infrastructure problem.
E_NO_DIR, E_NO_MOUNT = 196, 197
MARK = "[supervisor-sandbox]"

# Where things the project might depend on get installed or linked outside
# the project folder. Listed one level deep: names, types and link targets,
# not contents, so builds and caches don't count as changes.
WATCH = ["/", "/usr/local", "/usr/local/bin", "/usr/local/lib", "/opt", "/srv", "/etc/profile.d",
         "$HOME", "$HOME/.local", "$HOME/.local/bin", "$HOME/bin"]
# Directories whose contents churn on their own (or are per-run scratch).
SKIP = ["/proc", "/sys", "/dev", "/run", "/tmp", "/var", "/mnt", "/media", "/lost+found"]
# Files in $HOME that every shell and tool touches.
NOISE = (".cache", ".bash_history", ".python_history", ".lesshst", ".wget-hsts", ".viminfo",
         ".npm", ".node_repl_history", ".sudo_as_admin_successful", ".rustup/tmp")


class SandboxError(Exception):
    """The sandbox can't run checks reliably. Not a project failure."""


class Sandbox:
    def __init__(self, workspace, project_dir: Path, mount_point: str, *, setup_timeout: int = 1800,
                 watch: list[str] | None = None, skip: list[str] | None = None, log=None):
        self.workspace = workspace
        self.project_dir = Path(project_dir)
        self.mount_point = mount_point
        self.setup_timeout = setup_timeout
        self.watch = watch if watch is not None else WATCH
        self.skip = (skip if skip is not None else SKIP) + [mount_point]
        self.log = log
        self.ready = False
        self.baseline: set[str] = set()
        self.token = f"{_TOKEN_DIR}/sandbox-{secrets.token_hex(6)}"
        self.setup_log: list[dict] = []

    # --- lifecycle ------------------------------------------------------

    def prepare(self, setup: list[str]) -> None:
        """Verify the mount, run setup (the contract's, then the project's
        own .openhands/setup.sh), and record the environment. Raises
        SandboxError, with the evidence, if any step fails."""
        token_file = self.project_dir / self.token
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("written by the supervisor on the host; must be visible in the sandbox\n", encoding="utf-8")
        self.ready = True  # run() needs this; reset below on failure
        try:
            probe = self.run("true", timeout=60)
            if probe.infra:
                raise SandboxError(probe.infra)
            commands = list(setup)
            if (self.project_dir / AGENT_SETUP).is_file():
                commands.append(f"bash {shlex.quote(AGENT_SETUP)}")
            for cmd in commands:
                res = self.run(cmd, timeout=self.setup_timeout)
                self.setup_log.append({"cmd": cmd, "exit_code": res.exit_code, "output": res.output[-2000:]})
                if res.infra:
                    raise SandboxError(res.infra)
                if not res.passed:
                    raise SandboxError(
                        f"setup command failed (exit {res.exit_code}): {cmd}\n{res.output.strip()[-1500:]}\n"
                        "Checks were not run: their results would describe a half-prepared sandbox, not the project."
                    )
            self.baseline = self.fingerprint()
        except BaseException:
            self.ready = False
            raise

    def close(self) -> None:
        try:
            (self.project_dir / self.token).unlink(missing_ok=True)
        except OSError:
            pass

    # --- running ----------------------------------------------------------

    def wrap(self, cmd: str) -> str:
        """The exact shell line every command runs as. A login shell (so
        toolchains set up in profiles are on PATH), started in the project
        folder, which must be the mounted one (the token proves it), with
        pipefail so `cmd | tee` fails when cmd fails."""
        mount, token = shlex.quote(self.mount_point), shlex.quote(self.token)
        script = (
            f'cd {mount} 2>/dev/null || {{ echo "{MARK} project folder {self.mount_point} does not exist in the sandbox" >&2; exit {E_NO_DIR}; }}\n'
            f'test -f {token} || {{ echo "{MARK} {self.mount_point} is not the project folder (the supervisor\'s token file is not visible; is the folder mounted?)" >&2; exit {E_NO_MOUNT}; }}\n'
            f"set -o pipefail\n{cmd}"
        )
        return f"bash -lc {shlex.quote(script)}"

    def run(self, cmd: str, timeout: int, keep: int | None = 8000) -> CheckResult:
        """`keep`: how much of the output's tail to keep (None = all of it)."""
        started = time.monotonic()
        if not self.ready:
            return CheckResult(False, -1, "", 0.0, infra="the sandbox was not prepared (or its preparation failed)")
        try:
            result = self.workspace.execute_command(self.wrap(cmd), cwd=self.mount_point, timeout=timeout)
        except Exception as exc:
            return CheckResult(False, -1, "", time.monotonic() - started,
                               infra=f"the supervisor could not run the command in the sandbox: {type(exc).__name__}: {exc}")
        out, err = result.stdout or "", result.stderr or ""
        output = out + ("\n" + err if err else "")
        if getattr(result, "timeout_occurred", False):
            output += f"\n(command timed out after {timeout}s)"
        seconds = time.monotonic() - started
        tail = output if keep is None else output[-keep:]
        if result.exit_code in (E_NO_DIR, E_NO_MOUNT) and MARK in output:
            reason = next((line.split(MARK, 1)[1].strip() for line in output.splitlines() if MARK in line), "sandbox not usable")
            return CheckResult(False, result.exit_code, tail, seconds, infra=reason)
        return CheckResult(result.exit_code == 0, result.exit_code, tail, seconds)

    # --- environment drift ----------------------------------------------------

    def fingerprint(self) -> set[str]:
        """The sandbox's state outside the project folder, as "path|type|link
        target" lines (see WATCH). Empty if it can't be read."""
        prune = " -o ".join(f"-path {shlex.quote(p)}" for p in self.skip)
        roots = " ".join(f'"{r}"' if r.startswith("$") else shlex.quote(r) for r in self.watch)
        script = (
            f"for d in {roots}; do [ -d \"$d\" ] || continue; "
            f"find \"$d\" -mindepth 1 -maxdepth 1 \\( {prune} \\) -prune -o -printf '%p|%y|%l\\n' 2>/dev/null; done; "
            "python3 - <<'PY' 2>/dev/null\n"
            "import site\n"
            "import os\n"
            "for d in site.getsitepackages() + [site.getusersitepackages()]:\n"
            "    if os.path.isdir(d):\n"
            "        for n in sorted(os.listdir(d)):\n"
            "            if not n.endswith(('.pyc', '__pycache__')):\n"
            "                print(f'{d}/{n}|pkg|')\n"
            "PY\n"
            "true"
        )
        res = self.run(script, timeout=120, keep=None)  # all of it: a cut list would hide changes
        if not res.passed:
            return set()
        home_noise = tuple(f"/{n}" for n in NOISE)
        return {line for line in res.output.splitlines()
                if line.count("|") >= 2 and not line.split("|")[0].endswith(home_noise)}

    def drift(self) -> list[str]:
        """Human-readable changes outside the project folder since prepare()."""
        if not self.ready:
            return []
        return describe(self.baseline, self.fingerprint())


def describe(before: set[str], after: set[str]) -> list[str]:
    def parse(entries):
        out = {}
        for e in entries:
            path, kind, target = (e.split("|", 2) + ["", ""])[:3]
            out[path] = (kind, target)
        return out

    b, a = parse(before), parse(after)
    changes = []
    for path in sorted(set(a) - set(b)):
        kind, target = a[path]
        what = {"l": f"symlink -> {target}", "d": "directory", "f": "file", "pkg": "installed package"}.get(kind, kind)
        changes.append(f"added {path} ({what})")
    for path in sorted(set(b) - set(a)):
        changes.append(f"removed {path}")
    for path in sorted(set(a) & set(b)):
        if a[path] != b[path]:
            changes.append(f"changed {path} ({b[path][0]}{' -> ' + b[path][1] if b[path][1] else ''} became "
                           f"{a[path][0]}{' -> ' + a[path][1] if a[path][1] else ''})")
    return changes


def missing(recorded: list[str], current: set[str]) -> list[str]:
    """Which recorded 'added …' changes this sandbox doesn't have."""
    paths = {e.split("|", 1)[0] for e in current}
    gone = []
    for change in recorded:
        if change.startswith("added "):
            path = change[len("added "):].split(" (", 1)[0]
            if path not in paths:
                gone.append(change)
    return gone
