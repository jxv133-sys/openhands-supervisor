from pathlib import Path

import pytest

from supervisor import testguard
from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.log import RunLog
from supervisor.runner import Supervisor

def _sb(ws, project="/nonexistent"):
    """A prepared sandbox around a fake workspace (the fake ignores the wrapper)."""
    from supervisor.sandbox import Sandbox
    sb = Sandbox(ws, project, "/workspace/project")
    sb.ready = True
    return sb


DATA = Path(__file__).parent / "data"  # real cargo test / pytest output


@pytest.mark.parametrize("name, expect", [
    ("rust_match", ""),
    ("rust_nomatch", "no test name matches"),
    ("rust_ignored", "1 ignored"),
    ("py_pass", ""),
    ("py_allskip", "skipped every test"),
    ("py_nomatch", "-k filter"),
])
def test_zero_tests_on_real_output(name, expect):
    reason = testguard.zero_tests((DATA / f"{name}.txt").read_text())
    assert (expect in reason) if expect else reason == ""


def test_zero_tests_ignores_non_test_commands():
    assert testguard.zero_tests("   Compiling todo v0.1.0\n    Finished dev profile in 2.3s") == ""
    assert testguard.zero_tests("") == ""


RUST = '''
fn add(a: i32, b: i32) -> i32 { a + b }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() { assert_eq!(add(2, 3), 5); }
    #[test]
    #[should_panic]
    fn test_overflow() { add(i32::MAX, 1); }
}
'''

PY = '''
import pytest
def test_add():
    assert 1 + 1 == 2
class TestList:
    def test_empty(self):
        assert [] == []
'''


def _project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text(RUST)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(PY)
    (tmp_path / "target").mkdir()  # build output is never scanned
    (tmp_path / "target" / "junk.rs").write_text("#[test]\n#[ignore]\nfn generated() {}")
    return tmp_path


def test_scan_finds_tests(tmp_path):
    s = testguard.scan(_project(tmp_path))
    assert s.tests == {"src/main.rs::test_add", "src/main.rs::test_overflow",
                       "tests/test_app.py::test_add", "tests/test_app.py::TestList", "tests/test_app.py::test_empty"}
    assert s.skips == set()


def test_problems_catch_skips_and_deleted_tests(tmp_path):
    root = _project(tmp_path)
    before = testguard.scan(root)
    (root / "src" / "main.rs").write_text(RUST.replace("    #[test]\n    #[should_panic]", "    #[test]\n    #[ignore]\n    #[should_panic]"))
    (root / "tests" / "test_app.py").write_text(PY.replace("def test_add():", "@pytest.mark.skip(reason='later')\ndef test_add():")
                                                .replace("    def test_empty(self):\n        assert [] == []\n", "    pass\n"))
    found = testguard.problems(before, testguard.scan(root))
    text = "\n".join(found)
    assert "src/main.rs: #[ignore]" in text
    assert "@pytest.mark.skip" in text
    assert "removed or renamed an existing test: tests/test_app.py::test_empty" in text


def test_adding_tests_is_fine(tmp_path):
    root = _project(tmp_path)
    before = testguard.scan(root)
    (root / "tests" / "test_more.py").write_text("def test_new():\n    assert True\n")
    assert testguard.problems(before, testguard.scan(root)) == []


def test_languages_and_suite_checks(tmp_path):
    assert testguard.suite_checks(tmp_path) == []
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "requirements.txt").write_text("requests\n")
    checks = {c["check"] for c in testguard.suite_checks(tmp_path)}
    assert any("cargo build --release" in c and "cargo test" in c for c in checks)
    py = next(c for c in checks if "venv" in c)
    assert "pip install -q -r requirements.txt" in py and "python -m pytest" in py


# --- the runner refuses a dishonest pass ------------------------------------------

class Result:
    def __init__(self, out, code=0):
        self.stdout, self.stderr, self.exit_code = out, "", code


class FakeWorkspace:
    def __init__(self, out):
        self.out = out

    def execute_command(self, cmd, cwd=None, timeout=None):
        return Result(self.out)


def _sup(tmp_path, **settings):
    s = Settings()
    for k, v in settings.items():
        setattr(s, k, v)
    return Supervisor(s, tmp_path, Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "c"}]}),
                      RunLog(tmp_path, echo=False))


def test_check_with_zero_tests_fails(tmp_path):
    res = _sup(tmp_path)._check(_sb(FakeWorkspace((DATA / "rust_nomatch.txt").read_text())), "cargo test test_x")
    assert not res.passed and "proves nothing" in res.output


def test_check_after_skipping_a_test_fails(tmp_path):
    root = _project(tmp_path)
    sup = _sup(root)
    baseline = testguard.scan(root)
    assert sup._check(_sb(FakeWorkspace((DATA / "py_pass.txt").read_text())), "pytest", baseline).passed
    (root / "tests" / "test_app.py").write_text("import pytest\n@pytest.mark.skip\n" + PY.split("import pytest\n", 1)[1])
    res = sup._check(_sb(FakeWorkspace((DATA / "py_pass.txt").read_text())), "pytest", baseline)
    assert not res.passed and "not honestly" in res.output and "@pytest.mark.skip" in res.output


def test_test_guard_can_be_turned_off(tmp_path):
    res = _sup(tmp_path, test_guard=False)._check(_sb(FakeWorkspace((DATA / "rust_nomatch.txt").read_text())), "cargo test x")
    assert res.passed
