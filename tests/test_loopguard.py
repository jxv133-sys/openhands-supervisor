import random

from supervisor.config import Settings
from supervisor.contract import Contract
from supervisor.log import RunLog
from supervisor.loopguard import RepetitionDetector
from supervisor.runner import Supervisor
from supervisor.watchdog import Action, CheckResult, Limits, MilestoneState, TurnEnd, TurnReport, decide


def _feed(text, chunk=7):
    d = RepetitionDetector()
    for i in range(0, len(text), chunk):  # streamed in small pieces
        d.feed(text[i:i + chunk])
    return d.check()


LOOP = ("Wait, I need to check whether the JSON file exists before reading it. "
        "If it doesn't exist I should create it with an empty list. ")


def test_detects_a_thinking_loop():
    text = "The user wants a todo app. Let me plan the storage first. " + LOOP * 12
    assert "check whether the json file exists" in _feed(text)


def test_detects_loop_that_only_changes_numbers():
    text = "".join(f"Now I look at line {i} of main.rs to find where the parser fails and fix it. " for i in range(40))
    assert _feed(text) is not None


def test_normal_varied_reasoning_is_not_a_loop():
    random.seed(1)
    words = ("the parser storage json file task id list command test cargo build error "
             "write read update remove struct enum match option result serde clap main").split()
    sentences = [" ".join(random.choice(words) for _ in range(random.randint(8, 16))).capitalize() + "." for _ in range(150)]
    text = " ".join(sentences)
    assert len(text) > 5000
    assert _feed(text) is None


def test_short_repeated_phrases_are_fine():
    text = ("Let me check. " * 3 + "I will read src/main.rs and then add the list command with a test named test_list. ") * 3
    text += "Then I run cargo test test_list and fix whatever fails, one file at a time, until it passes. " * 2
    assert _feed(text) is None


def test_looping_turn_restarts_the_milestone():
    d = decide(TurnReport(TurnEnd.LOOPING, error="x"), CheckResult(False, 1, "fail"), "c", MilestoneState(), Limits())
    assert d.action == Action.RESTART and "repeating" in d.reason


class Delta:
    def __init__(self, reasoning):
        self.content, self.reasoning_content = None, reasoning


class FakeConversation:
    def __init__(self):
        self.calls = []

    def interrupt(self):
        self.calls.append("interrupt")


def test_supervisor_interrupts_a_looping_reply(tmp_path):
    contract = Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "true"}]})
    sup = Supervisor(Settings(), tmp_path, contract, RunLog(tmp_path, echo=False))
    sup._conversation = FakeConversation()
    for _ in range(14):
        sup._loop_checked = 0  # don't wait for the 1s throttle in a test
        sup._activity("StreamingDeltaEvent", Delta(LOOP))
    assert sup._looping
    for t in __import__("threading").enumerate():
        if t.name == "loop-guard":
            t.join(2)
    assert sup._conversation.calls == ["interrupt"]


def test_loop_guard_can_be_turned_off(tmp_path):
    s = Settings(); s.loop_guard = False
    sup = Supervisor(s, tmp_path, Contract.from_dict({"goal": "g", "milestones": [{"title": "M", "check": "true"}]}),
                     RunLog(tmp_path, echo=False))
    for _ in range(14):
        sup._loop_checked = 0
        sup._activity("StreamingDeltaEvent", Delta(LOOP))
    assert not sup._looping
