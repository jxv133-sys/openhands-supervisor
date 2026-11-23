"""Keep tests honest for Rust and Python projects.

"Done" is decided by a check command, and the check usually runs tests the
agent wrote itself. Three ways that goes wrong, and what this module does:

1. The check ran no tests at all. `cargo test test_add` exits 0 when no test
   matches the name, so a typo or a renamed test "passes". `zero_tests()`
   reads the test runners' summaries; the supervisor treats such a run as failed.
2. The agent made the check pass by skipping or deleting tests (`#[ignore]`,
   `pytest.skip`, removing a test that existed before). `scan()` lists every
   test and skip marker; `problems()` compares against the milestone's start.
3. Tests only ever ran filtered, in a sandbox full of hand-installed packages.
   `suite_checks()` gives the final review whole-suite and clean-environment
   commands that run like a fresh machine would.

Pure functions over files and text, no SDK, so all of it is unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .progress import _iter_files

# --- languages ------------------------------------------------------------


def languages(root: Path) -> set[str]:
    root = Path(root)
    langs = set()
    if (root / "Cargo.toml").is_file():
        langs.add("rust")
    if any((root / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")):
        langs.add("python")
    elif "rust" not in langs and any(p.suffix == ".py" for p in _iter_files(root)):
        langs.add("python")
    return langs


# --- 1. did the check run any tests? ------------------------------------------

_CARGO_RESULT = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored")
_PYTEST_COUNTS = re.compile(r"(\d+) (passed|failed|skipped|error|errors|xfailed|xpassed|deselected)")


def zero_tests(output: str) -> str:
    """Why a *passing* check didn't really test anything, or "" if it did (or
    the output isn't from cargo test / pytest, e.g. a plain `cargo build`)."""
    cargo = _CARGO_RESULT.findall(output)
    if cargo:
        passed = sum(int(p) for p, _, _ in cargo)
        ignored = sum(int(i) for _, _, i in cargo)
        if passed == 0:
            if ignored:
                return f"cargo test ran 0 tests ({ignored} ignored)"
            return "cargo test ran 0 tests: no test name matches the filter in the check command"
        return ""
    if "no tests ran" in output or re.search(r"collected 0 items", output):
        return "pytest collected no tests"
    # pytest's last summary line, e.g. "3 passed, 1 skipped in 0.12s" / "2 skipped in 0.01s"
    lines = [ln for ln in output.splitlines() if re.search(r"\bin [\d.]+s\b", ln) and _PYTEST_COUNTS.search(ln)]
    if lines:
        counts = {kind: int(n) for n, kind in _PYTEST_COUNTS.findall(lines[-1])}
        if not counts.get("passed") and not counts.get("xpassed") and not counts.get("failed"):
            if counts.get("skipped"):
                return f"pytest skipped every test ({counts['skipped']} skipped, 0 passed)"
            if counts.get("deselected"):
                return "pytest ran 0 tests: no test matches the -k filter in the check command"
            return "pytest ran 0 tests"
    return ""


# --- 2. tests and skip markers in the source -------------------------------------

_RUST_TEST_ATTR = re.compile(r"#\[(?:tokio::|async_std::)?test\b")
_RUST_FN = re.compile(r"\bfn\s+([A-Za-z_]\w*)")
_RUST_SKIPS = [re.compile(r"#\[ignore\b"), re.compile(r"#\[cfg\(\s*(?:any\(\s*\)|FALSE|never)\s*\)\]")]
_PY_TEST_FN = re.compile(r"^\s*(?:async\s+)?def\s+(test\w*)\s*\(", re.MULTILINE)
_PY_TEST_CLASS = re.compile(r"^\s*class\s+(Test\w*)", re.MULTILINE)
_PY_SKIPS = [
    re.compile(r"@pytest\.mark\.(?:skip|skipif|xfail)\b"),
    re.compile(r"\bpytest\.skip\s*\("),
    re.compile(r"@unittest\.(?:skip|skipIf|skipUnless|expectedFailure)\b"),
    re.compile(r"^\s*pytestmark\s*=.*\bskip", re.MULTILINE),
]


def _is_python_test_file(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


@dataclass
class TestScan:
    tests: set[str] = field(default_factory=set)   # "path::name"
    skips: set[str] = field(default_factory=set)   # "path: <marker line>"


def scan(root: Path) -> TestScan:
    root = Path(root)
    result = TestScan()
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        if path.suffix == ".rs":
            kind = "rust"
        elif _is_python_test_file(rel):
            kind = "python"
        else:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if kind == "rust":
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if _RUST_TEST_ATTR.search(line):
                    for nxt in lines[i:i + 6]:  # attributes may stack before the fn
                        m = _RUST_FN.search(nxt)
                        if m:
                            result.tests.add(f"{rel}::{m.group(1)}")
                            break
                for rx in _RUST_SKIPS:
                    if rx.search(line):
                        result.skips.add(f"{rel}: {line.strip()[:120]}")
        else:
            for name in _PY_TEST_FN.findall(text) + _PY_TEST_CLASS.findall(text):
                result.tests.add(f"{rel}::{name}")
            for line in text.splitlines():
                if any(rx.search(line) for rx in _PY_SKIPS):
                    result.skips.add(f"{rel}: {line.strip()[:120]}")
    return result


def problems(before: TestScan, after: TestScan) -> list[str]:
    """Ways the agent weakened the tests since `before` (the milestone's start)."""
    out = []
    for skip in sorted(after.skips - before.skips):
        out.append(f"added a skip marker: {skip}")
    for test in sorted(before.tests - after.tests):
        out.append(f"removed or renamed an existing test: {test}")
    return out


def problems_feedback(found: list[str]) -> str:
    listed = "\n".join(f"- {p}" for p in found[:20])
    return (
        "\n\n[supervisor] The check passed, but not honestly. You changed the tests instead of the code:\n"
        f"{listed}\n"
        "Undo these test changes (remove the skip markers, restore the tests) and make the code pass the "
        "real tests. Never skip, weaken or delete tests to get a check to pass."
    )


def zero_tests_feedback(reason: str) -> str:
    return (
        f"\n\n[supervisor] The command exited 0 but {reason}, so it proves nothing. "
        "Make sure the tests exist, are not skipped, and that their names match the filter in the check command."
    )


# --- 3. whole-suite checks for the final review ---------------------------------

def suite_checks(root: Path) -> list[dict]:
    """[{requirement, check}] the final review always runs for Rust/Python."""
    langs = languages(root)
    checks = []
    if "rust" in langs:
        checks.append({
            "requirement": "The project builds in release mode and its whole Rust test suite passes",
            "check": "cargo build --release 2>&1 && cargo test 2>&1",
        })
    if "python" in langs:
        root = Path(root)
        install = []
        if any((root / f).is_file() for f in ("pyproject.toml", "setup.py", "setup.cfg")):
            # The project as a package, plus its test/dev extras if it declares any.
            install.append("pip install -q . && (pip install -q '.[test]' 2>/dev/null || pip install -q '.[dev]' 2>/dev/null || true)")
        for req in ("requirements.txt", "requirements-dev.txt", "requirements-test.txt"):
            if (root / req).is_file():
                install.append(f"pip install -q -r {req}")
        steps = " && ".join(install) if install else "true"
        checks.append({
            "requirement": "In a clean virtualenv with only the project's declared dependencies, all Python tests pass",
            "check": (
                'v=$(mktemp -d) && python3 -m venv "$v" && . "$v/bin/activate" && pip install -q --upgrade pip && '
                f'{steps} && pip install -q pytest && python -m pytest -q 2>&1'
            ),
        })
    return checks
