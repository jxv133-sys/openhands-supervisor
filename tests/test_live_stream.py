import time

from supervisor.chat import LiveStream
from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.log import RunLog
from supervisor.runner import Supervisor


def _collect(interval=10.0):
    out = []
    return out, LiveStream(lambda **r: out.append(r), interval=interval)


def _text(records):
    return "".join(r.get("text", "") for r in records if not r.get("end"))


def test_first_delta_flushes_then_batches_until_end():
    out, live = _collect()
    for tok in ["Hel", "lo", " wor", "ld"]:
        live.delta(tok, None)
    assert out == [{"seq": 1, "text": "Hel", "thinking": ""}]  # the rest waits for the interval
    live.end()
    assert _text(out) == "Hello world"
    assert out[-1] == {"seq": 3, "end": True}
    assert live.snapshot() is None


def test_tail_is_flushed_by_timer_without_more_tokens():
    out, live = _collect(interval=0.05)
    live.delta("a", "think")
    live.delta("b", None)
    time.sleep(0.2)
    assert _text(out) == "ab"
    assert "".join(r["thinking"] for r in out) == "think"


def test_snapshot_plus_later_updates_shows_each_token_once():
    out, live = _collect()
    live.delta("one ", None)
    live.delta("two ", None)  # pending, not published yet
    snap = live.snapshot()
    live.delta("three", None)
    live.end()
    later = [r for r in out if r["seq"] > snap["seq"]]
    assert snap["text"] + _text(later) == "one two three"


def test_end_without_deltas_publishes_nothing():
    out, live = _collect()
    live.end()
    live.delta("", None)
    assert out == []


class Delta:
    def __init__(self, content=None, reasoning_content=None):
        self.content, self.reasoning_content = content, reasoning_content


Delta.__name__ = "StreamingDeltaEvent"


class Content:
    def __init__(self, text):
        self.text = text


class Msg:
    def __init__(self, text):
        self.content = [Content(text)]
        self.reasoning_content = ""


class MessageEvent:
    source = "agent"

    def __init__(self, text):
        self.llm_message = Msg(text)


def test_supervisor_streams_deltas_then_ends_before_final_message(tmp_path):
    contract = Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "true"}]})
    log = RunLog(tmp_path, echo=False)
    seen = []
    log.add_listener(seen.append)
    sup = Supervisor(Settings(), tmp_path, contract, log)
    sup.live._interval = 10.0
    sup._activity("StreamingDeltaEvent", Delta("I will ", "plan"))
    sup._activity("StreamingDeltaEvent", Delta("edit main.rs"))
    sup._activity("MessageEvent", MessageEvent("I will edit main.rs"))
    kinds = [(r["kind"], r.get("end"), r.get("role")) for r in seen]
    assert kinds == [("stream", None, None), ("stream", None, None), ("stream", True, None), ("chat", None, "agent")]
    assert _text([r for r in seen if r["kind"] == "stream"]) == "I will edit main.rs"
    # transient: only the finished message is on disk
    assert [r["role"] for r in log.read_chat()] == ["agent"]


def test_seq_keeps_increasing_across_messages():
    out, live = _collect()
    live.delta("a", None)
    live.end()
    live.delta("b", None)
    live.end()
    seqs = [r["seq"] for r in out]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
