import pytest

from supervisor import ideas
from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.ideas import IdeaStore, ThinkSplitter, Tools, _for_model, chat_turn
from supervisor.web import App


def _app(tmp_path):
    Contract.from_dict({"goal": "A Rust CLI todo app\nwith JSON storage", "milestones": [
        {"title": "Scaffold", "check": "cargo build", "status": "done"}]}).save(tmp_path / "todo-rs")
    (tmp_path / "todo-rs" / "README.md").write_text("# todo\nRun `todo add`.")
    return App(Settings(), tmp_path)


def fake_model(*rounds):
    """Each round is a list of Ollama stream chunks; records the payloads it got."""
    calls = []

    def stream(url, payload, timeout):
        calls.append(payload)
        yield from rounds[len(calls) - 1]

    stream.calls = calls
    return stream


def test_tool_round_then_answer(tmp_path):
    app = _app(tmp_path)
    model = fake_model(
        [{"message": {"thinking": "Let me look."}},
         {"message": {"content": "", "tool_calls": [{"function": {"name": "read_project", "arguments": {"name": "todo-rs"}}}]}}],
        [{"message": {"content": "You could add "}}, {"message": {"content": "due dates."}}, {"done": True}],
    )
    chat = IdeaStore(tmp_path).new()
    events = list(chat_turn(chat, "what next for todo-rs?", Settings(), Tools(app), stream=model))
    kinds = [e["type"] for e in events]
    assert kinds == ["thinking", "tool", "tool_result", "text", "text"]
    assert events[1]["summary"] == "read project todo-rs"
    assert "A Rust CLI todo app" in events[2]["text"] and "[done] Scaffold" in events[2]["text"]
    roles = [m["role"] for m in chat["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert chat["messages"][-1]["content"] == "You could add due dates."
    assert chat["title"] == "what next for todo-rs?"
    second = model.calls[1]["messages"]
    assert second[0]["role"] == "system" and second[-1] == {"role": "tool", "content": events[2]["text"], "tool_name": "read_project"}
    assert "thinking" not in second[2] and "ts" not in second[1]  # only Ollama's fields go back


def test_propose_project_event(tmp_path):
    model = fake_model([{"message": {"tool_calls": [{"function": {"name": "propose_project",
                         "arguments": {"name": "habit-rs", "idea": "A Rust CLI habit tracker."}}}]}}],
                       [{"message": {"content": "Done."}}])
    events = list(chat_turn(IdeaStore(tmp_path).new(), "ok let's do it", Settings(), Tools(_app(tmp_path)), stream=model))
    assert {"type": "propose", "name": "habit-rs", "idea": "A Rust CLI habit tracker."} in events


def test_think_tags_split_across_chunks():
    s = ThinkSplitter()
    out = []
    for piece in ["<thi", "nk>plan it", "</th", "ink>Answer", " here"]:
        out += s.feed(piece)
    joined = {"thinking": "", "text": ""}
    for kind, text in out:
        joined[kind] += text
    assert joined == {"thinking": "plan it", "text": "Answer here"}


def test_stop_keeps_partial_reply(tmp_path):
    model = fake_model([{"message": {"content": "First half"}}, {"message": {"content": " second half"}}])
    chat = IdeaStore(tmp_path).new()
    turn = chat_turn(chat, "hi", Settings(), Tools(_app(tmp_path)), stream=model)
    assert next(turn)["text"] == "First half"
    turn.close()  # the page pressed Stop
    assert chat["messages"][-1] == {**chat["messages"][-1], "role": "assistant", "content": "First half", "stopped": True}


def test_model_error_is_reported(tmp_path):
    def broken(url, payload, timeout):
        raise ConnectionError("ollama unreachable")
        yield  # pragma: no cover

    events = list(chat_turn(IdeaStore(tmp_path).new(), "hi", Settings(), Tools(_app(tmp_path)), stream=broken))
    assert events[-1]["type"] == "error" and "unreachable" in events[-1]["text"]


def test_tools_are_read_only_and_bounded(tmp_path):
    t = Tools(_app(tmp_path))
    assert "todo-rs: A Rust CLI todo app (1/1" in t.run("list_projects", {})
    assert "README.md" in t.run("list_files", {"project": "todo-rs"})
    assert "Run `todo add`" in t.run("read_file", {"project": "todo-rs", "path": "README.md"})
    assert t.run("read_file", {"project": "todo-rs", "path": "../../etc/passwd"}).startswith("error")
    assert t.run("read_project", {"name": "nope"}).startswith("error")
    assert t.run("rm_rf", {}).startswith("error: unknown tool")
    assert "turned off" in Tools(None, allow_web=False).run("fetch_url", {"url": "https://example.com"})


@pytest.mark.parametrize("url", ["http://127.0.0.1:8765/api/state", "http://localhost/", "http://192.168.50.95:11434/api/tags",
                                 "http://10.0.0.1/", "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://x/"])
def test_fetch_refuses_private_and_odd_addresses(url):
    assert ideas.fetch_url(url).startswith("error")


def test_html_to_text():
    page = "<html><head><title>t</title><style>x{}</style></head><body><h1>Serde</h1><p>Fast &amp; generic.</p><script>evil()</script></body></html>"
    text = ideas.html_to_text(page)
    assert "Serde" in text and "Fast & generic." in text and "evil" not in text and "x{}" not in text


def test_history_is_trimmed_from_the_oldest(tmp_path):
    msgs = [{"role": "user", "content": "x" * 1000}, {"role": "tool", "tool_name": "t", "content": "y" * 1000},
            {"role": "user", "content": "recent"}, {"role": "assistant", "content": "answer"}]
    kept = _for_model(msgs, budget_chars=1500)
    assert [m["content"] for m in kept] == ["recent", "answer"]  # no orphaned tool result at the start


def test_store_roundtrip_and_ids(tmp_path):
    store = IdeaStore(tmp_path)
    chat = store.new(); chat["title"] = "t"; store.save(chat)
    assert store.list()[0]["id"] == chat["id"] and store.get(chat["id"])["title"] == "t"
    with pytest.raises(KeyError):
        store.get("../../etc")
    store.delete(chat["id"])
    assert store.list() == []
    assert [p["name"] for p in App(Settings(), tmp_path).state()["projects"]] == []  # .ideas isn't a project
