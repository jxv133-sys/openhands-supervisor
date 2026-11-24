"""The sandbox lifecycle, run with real bash.

`LocalWorkspace` executes commands with subprocess, like the agent-server
does inside the container. A temp "system root" stands in for the
container's filesystem outside the project, so tests can make the agent
change it (as it did with a /app symlink) and then start a "fresh sandbox"
without that change.
"""

import subprocess

import pytest

from supervisor.sandbox import AGENT_SETUP, Sandbox, SandboxError, describe, missing


class Result:
    def __init__(self, p):
        self.stdout, self.stderr, self.exit_code, self.timeout_occurred = p.stdout, p.stderr, p.returncode, False


class LocalWorkspace:
    def __init__(self):
        self.commands = []

    def execute_command(self, command, cwd=None, timeout=None):
        self.commands.append(command)
        return Result(subprocess.run(command, shell=True, cwd=cwd if cwd and __import__("os").path.isdir(cwd) else None,
                                     capture_output=True, text=True, timeout=timeout))


class BrokenWorkspace:
    def execute_command(self, command, cwd=None, timeout=None):
        raise ConnectionError("agent-server not responding")


@pytest.fixture
def world(tmp_path):
    proj, sysroot = tmp_path / "project", tmp_path / "sysroot"
    proj.mkdir(); sysroot.mkdir()
    (proj / "main.py").write_text('def hello():\n    return "hi"\n')
    return proj, sysroot


def sandbox(proj, sysroot, mount=None, ws=None):
    return Sandbox(ws or LocalWorkspace(), proj, str(mount or proj), watch=[str(sysroot)], skip=[])


def check_via_app(sysroot):
    """The kind of check that 'regressed': copies the project from a path
    outside it into a temp dir and imports it from a Python heredoc."""
    return (f'd=$(mktemp -d) && cp -r {sysroot}/app/. "$d" && cd "$d" && python3 - <<\'EOF\'\n'
            'import main\nassert main.hello() == "hi"\nprint("ok")\nEOF')


def test_reproduces_the_false_regression_and_explains_it(world):
    proj, sysroot = world
    # Milestone sandbox: the agent links /app to the project by hand, and the check passes.
    sb = sandbox(proj, sysroot); sb.prepare([])
    (sysroot / "app").symlink_to(proj)
    assert sb.run(check_via_app(sysroot), 60).passed
    drift = sb.drift()
    assert drift == [f"added {sysroot}/app (symlink -> {proj})"]

    # A fresh sandbox (resumed run / review-only run) doesn't have the link.
    (sysroot / "app").unlink()
    fresh = sandbox(proj, sysroot); fresh.prepare([])
    res = fresh.run(check_via_app(sysroot), 60)
    assert not res.passed and not res.infra  # the command really fails here...
    assert missing(drift, fresh.fingerprint()) == drift  # ...because of a recorded environment change, not the project


def test_project_setup_script_makes_fresh_sandboxes_equivalent(world):
    proj, sysroot = world
    (proj / ".openhands").mkdir()
    (proj / AGENT_SETUP).write_text(f"ln -sfn {proj} {sysroot}/app\n")
    fresh = sandbox(proj, sysroot); fresh.prepare([])  # runs .openhands/setup.sh
    assert fresh.run(check_via_app(sysroot), 60).passed
    assert fresh.drift() == []  # the link is part of the declared environment now


def test_unmounted_project_folder_is_an_infrastructure_error(world, tmp_path):
    proj, sysroot = world
    empty = tmp_path / "empty-mount"; empty.mkdir()  # what a failed/mis-mapped mount looks like
    sb = sandbox(proj, sysroot, mount=empty)
    with pytest.raises(SandboxError, match="not the project folder"):
        sb.prepare([])
    assert not sb.ready
    res = sb.run("true", 10)
    assert res.infra and not res.passed  # nothing runs against an unprepared sandbox


def test_missing_project_folder_is_an_infrastructure_error(world, tmp_path):
    proj, sysroot = world
    with pytest.raises(SandboxError, match="does not exist"):
        sandbox(proj, sysroot, mount=tmp_path / "nope").prepare([])


def test_failed_setup_stops_before_any_check(world):
    proj, sysroot = world
    sb = sandbox(proj, sysroot)
    with pytest.raises(SandboxError, match="setup command failed .*install-toolchain"):
        sb.prepare(["echo installing; false # install-toolchain"])
    assert not sb.ready


def test_sandbox_api_error_is_infra_not_a_failed_check(world):
    proj, sysroot = world
    sb = sandbox(proj, sysroot, ws=BrokenWorkspace())
    sb.ready = True
    res = sb.run("cargo test", 10)
    assert res.infra.startswith("the supervisor could not run the command") and not res.passed


def test_mount_lost_after_preparation_is_detected(world):
    proj, sysroot = world
    sb = sandbox(proj, sysroot); sb.prepare([])
    (proj / sb.token).unlink()  # e.g. the bind mount went away mid-run
    res = sb.run("python3 -c 'print(1)'", 30)
    assert res.infra and "not the project folder" in res.infra


def test_a_check_exiting_197_by_itself_is_still_a_normal_failure(world):
    proj, sysroot = world
    sb = sandbox(proj, sysroot); sb.prepare([])
    res = sb.run("exit 197", 10)
    assert not res.passed and not res.infra


def test_heredocs_pipes_and_cwd_behave(world):
    proj, sysroot = world
    sb = sandbox(proj, sysroot); sb.prepare([])
    assert sb.run("python3 - <<'EOF'\nimport main\nprint(main.hello())\nEOF", 30).output.strip() == "hi"
    assert not sb.run("false | cat", 10).passed  # pipefail
    assert sb.run("pwd", 10).output.strip() == str(proj)


def test_describe_reports_added_removed_changed():
    before = {"/opt/a|d|", "/usr/local/bin/x|l|/old"}
    after = {"/usr/local/bin/x|l|/new", "/app|l|/workspace/project"}
    assert describe(before, after) == ["added /app (symlink -> /workspace/project)", "removed /opt/a",
                                       "changed /usr/local/bin/x (l -> /old became l -> /new)"]
