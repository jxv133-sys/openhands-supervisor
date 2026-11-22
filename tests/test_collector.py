"""The collector matches SDK events by class name, so plain fakes with the
same class names and attributes stand in for real SDK events."""

from types import SimpleNamespace

from supervisor.collector import TurnCollector
from supervisor.watchdog import TurnEnd


class ActionEvent(SimpleNamespace):
    pass


class FinishAction(SimpleNamespace):
    pass


class MessageEvent(SimpleNamespace):
    pass


class ConversationErrorEvent(SimpleNamespace):
    pass


class AgentErrorEvent(SimpleNamespace):
    pass


def text_msg(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


def test_finish_action():
    c = TurnCollector()
    c.on_event(ActionEvent(tool_name="terminal", action=SimpleNamespace()))
    c.on_event(ActionEvent(tool_name="finish", action=FinishAction(message="Built it")))
    r = c.report("finished", changed_files=2)
    assert r.end == TurnEnd.FINISHED
    assert r.agent_text == "Built it"
    assert r.actions == 2
    assert r.changed_files == 2
    assert c.tool_counts == {"terminal": 1, "finish": 1}


def test_plain_agent_message():
    c = TurnCollector()
    c.on_event(MessageEvent(source="user", llm_message=text_msg("do the thing")))
    c.on_event(MessageEvent(source="agent", llm_message=text_msg("Should I continue?")))
    r = c.report("finished")
    assert r.end == TurnEnd.MESSAGE
    assert r.agent_text == "Should I continue?"


def test_iteration_cap_from_event():
    c = TurnCollector()
    c.on_event(ConversationErrorEvent(code="MaxIterationsReached", detail="limit 40"))
    assert c.report("error").end == TurnEnd.ITER_CAP


def test_iteration_cap_from_exception_text():
    c = TurnCollector()
    r = c.report("error", exception="ConversationRunError: Agent reached maximum iterations limit (40).")
    assert r.end == TurnEnd.ITER_CAP


def test_stuck_status():
    assert TurnCollector().report("stuck").end == TurnEnd.STUCK


def test_error_status_and_exception():
    assert TurnCollector().report("error").end == TurnEnd.ERROR
    r = TurnCollector().report("idle", exception="ReadTimeout: timed out")
    assert r.end == TurnEnd.ERROR
    assert "ReadTimeout" in r.error


def test_agent_error_event_does_not_end_turn_by_itself():
    c = TurnCollector()
    c.on_event(AgentErrorEvent(error="tool validation failed"))
    c.on_event(ActionEvent(tool_name="finish", action=FinishAction(message="ok")))
    assert c.report("finished").end == TurnEnd.FINISHED


def test_reset_clears_turn():
    c = TurnCollector()
    c.on_event(ActionEvent(tool_name="finish", action=FinishAction(message="ok")))
    c.reset()
    r = c.report("finished")
    assert r.end == TurnEnd.MESSAGE
    assert r.actions == 0


def test_wait_until_quiet_returns():
    c = TurnCollector()
    c.wait_until_quiet(quiet_seconds=0.05, max_wait=1)
