import pytest

from supervisor import progress
from supervisor.config import Settings


def test_snapshot_detects_add_modify_delete(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}")
    (tmp_path / "keep.txt").write_text("same")
    (tmp_path / "gone.txt").write_text("bye")
    before = progress.snapshot(tmp_path)

    (tmp_path / "src" / "main.rs").write_text("fn main() { println!(); }")
    (tmp_path / "gone.txt").unlink()
    (tmp_path / "new.rs").write_text("x")
    after = progress.snapshot(tmp_path)

    assert progress.changed_files(before, after) == ["gone.txt", "new.rs", "src/main.rs"]


def test_build_output_is_ignored(tmp_path):
    before = progress.snapshot(tmp_path)
    for d in ["target/debug", ".supervisor", "node_modules/x", ".git"]:
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "f").write_text("x")
    assert progress.changed_files(before, progress.snapshot(tmp_path)) == []


def test_file_tree(tmp_path):
    assert progress.file_tree(tmp_path) == "(no files yet)"
    for i in range(5):
        (tmp_path / f"f{i}.txt").write_text("x")
    tree = progress.file_tree(tmp_path, limit=3)
    assert "f0.txt" in tree and "and 2 more" in tree


def test_settings_defaults_without_file(tmp_path):
    s = Settings.load(tmp_path / "missing.toml")
    assert s.model == "qwen-abl:latest"
    assert s.openai_base_url == "http://192.168.50.95:11434/v1"


def test_settings_from_toml(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text(
        '[ollama]\nmodel = "other"\nollama_url = "http://h:1/"\n'
        "[agent]\niterations_per_turn = 10\n[limits]\nstagnant_turns = 5\n"
    )
    s = Settings.load(f)
    assert s.model == "other"
    assert s.openai_base_url == "http://h:1/v1"
    assert s.iterations_per_turn == 10
    assert s.limits.stagnant_turns == 5


def test_settings_rejects_typos(tmp_path):
    f = tmp_path / "s.toml"
    f.write_text("[agent]\niterations_per_trun = 10\n")
    with pytest.raises(ValueError):
        Settings.load(f)


def test_shipped_config_loads():
    from pathlib import Path

    Settings.load(Path(__file__).parent.parent / "supervisor.toml")
