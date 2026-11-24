"""Drives OpenHands (via its SDK) milestone by milestone, applying the watchdog.

The agent runs inside a Docker sandbox (OpenHands agent-server image) with the
project folder mounted at /workspace/project. The supervisor runs on the host,
reads the project files directly to measure progress, and runs each
milestone's check inside the same sandbox, so it sees exactly what the agent sees.
"""

from __future__ import annotations

import shlex
import threading
import time
from collections.abc import Callable
from pathlib import Path

from . import progress, review, testguard
from .chat import LiveStream, chat_record
from .collector import TurnCollector
from .config import Settings
from .contract import DONE, SKIPPED, Contract, Milestone
from .hostpath import host_path
from .log import RunLog
from .loopguard import RepetitionDetector
from .prompts import PROJECT_DIR, handoff_note, milestone_prompt
from .watchdog import Action, CheckResult, Decision, MilestoneState, TurnEnd, decide


class Quit(Exception):
    pass


def _bash(cmd: str) -> str:
    # Login shell, so toolchains installed by setup (rustup, nvm...) are on PATH.
    # pipefail: `cargo build | tee log` must fail when cargo fails.
    return f"bash -lc {shlex.quote('set -o pipefail; ' + cmd)}"


ESCALATE_MODES = ("ask", "stop", "skip")

# A custom escalation handler (used by the web UI) gets
# (milestone, decision, check) and returns one of:
#   ("guidance", "text")   retry the milestone with this hint
#   ("skip", "")           skip the milestone
#   ("stop", "")           stop the run
EscalationHandler = Callable[[Milestone, Decision, CheckResult], tuple[str, str]]


class Supervisor:
    def __init__(
        self,
        settings: Settings,
        project_dir: Path,
        contract: Contract,
        log: RunLog,
        on_escalate: str | EscalationHandler = "ask",
    ):
        if not callable(on_escalate) and on_escalate not in ESCALATE_MODES:
            raise ValueError(f"on_escalate must be one of {ESCALATE_MODES}, not {on_escalate!r}")
        self.settings = settings
        self.project_dir = Path(project_dir).resolve()
        self.contract = contract
        self.log = log
        # ask = prompt in the terminal; stop/skip = unattended; callable = custom (web UI)
        self.on_escalate = on_escalate
        self._stop = threading.Event()
        self._conversation = None
        self._milestone_n = 0  # tags chat records with the milestone they belong to
        # The model's output while it's still being generated (web UI only).
        self.live = LiveStream(lambda **rec: self.log.publish("stream", milestone=self._milestone_n, **rec))
        self._loop = RepetitionDetector()
        self._loop_checked = 0.0
        self._looping = ""  # set when the loop guard cut the current reply off
        # Size of the latest prompt sent to the model: {"tokens", "max"} (web UI)
        self.context: dict | None = None
        self._context_thread: threading.Thread | None = None

    def request_stop(self) -> None:
        """Ask the run to stop (thread-safe). Pauses the agent mid-turn if one is
        running; progress is saved and the run ends at the next safe point."""
        self._stop.set()
        self._interrupt()

    def _interrupt(self) -> None:
        """Cut the agent off now, even mid-reply. interrupt() cancels the model
        call in progress; pause() (the fallback) waits for it to finish."""
        conversation = self._conversation
        if conversation is None:
            return
        for method in ("interrupt", "pause"):
            try:
                getattr(conversation, method)()
                return
            except Exception:
                continue

    def _check_stop(self) -> None:
        if self._stop.is_set():
            raise Quit()

    def _activity(self, kind: str, event) -> None:
        if kind == "StreamingDeltaEvent":  # one token or so of the reply being written
            content, reasoning = getattr(event, "content", None), getattr(event, "reasoning_content", None)
            self.live.delta(content, reasoning)
            self._guard_loop(reasoning, content)
            return
        record = chat_record(event)
        if record:
            self._loop.reset()
            self.live.end()  # the finished message/tool call replaces the live text
            self.log.chat(record, milestone=self._milestone_n)
            if kind in ("ActionEvent", "MessageEvent"):  # each follows a model response
                self._refresh_context()

    def _refresh_context(self) -> None:
        """Publish how full the model's context is, after each model response.
        Off the SDK's event thread: reading stats may need a request to the sandbox."""
        if self._context_thread is not None and self._context_thread.is_alive():
            return

        def read():
            time.sleep(0.5)  # let the server's stats update for this response arrive
            conversation = self._conversation
            if conversation is None:
                return
            tokens = _token_usage(conversation)["context"]
            if tokens and (self.context or {}).get("tokens") != tokens:
                self.context = {"tokens": tokens, "max": self.settings.max_input_tokens}
                self.log.publish("context", milestone=self._milestone_n, **self.context)

        self._context_thread = threading.Thread(target=read, name="context", daemon=True)
        self._context_thread.start()

    def _guard_loop(self, *texts) -> None:
        """Watch the reply being written; cut it off if it loops (see loopguard.py)."""
        if not self.settings.loop_guard or self._looping:
            return
        for text in texts:
            self._loop.feed(text)
        now = time.monotonic()
        if now - self._loop_checked < 1.0:  # checking every token would be wasteful
            return
        self._loop_checked = now
        repeated = self._loop.check()
        if repeated:
            self._looping = repeated
            self.log.say(f"The model is repeating itself mid-reply; cutting it off: \"{repeated}\"")
            # Not on this thread: it's the SDK's event thread, and interrupt() waits on the server.
            threading.Thread(target=self._interrupt, name="loop-guard", daemon=True).start()

    # --- sandbox helpers ------------------------------------------------

    def _exec(self, workspace, cmd: str, timeout: int) -> CheckResult:
        started = time.monotonic()
        try:
            result = workspace.execute_command(_bash(cmd), cwd=PROJECT_DIR, timeout=timeout)
        except Exception as exc:  # sandbox/API failure: report as a failed check
            return CheckResult(False, -1, f"supervisor could not run the command: {exc}", time.monotonic() - started)
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if getattr(result, "timeout_occurred", False):
            output += f"\n(command timed out after {timeout}s)"
        return CheckResult(result.exit_code == 0, result.exit_code, output[-8000:], time.monotonic() - started)

    def _check(self, workspace, cmd: str, baseline: testguard.TestScan | None = None) -> CheckResult:
        """Run a check, and only count a pass that really tested something: it
        must have run at least one test (if it's cargo test / pytest), and, if
        `baseline` is given, the agent must not have skipped or deleted tests
        since then (see testguard.py)."""
        res = self._exec(workspace, cmd, self.settings.check_timeout)
        if not res.passed or not self.settings.test_guard:
            return res
        reason = testguard.zero_tests(res.output)
        if reason:
            return CheckResult(False, 1, res.output + testguard.zero_tests_feedback(reason), res.seconds)
        if baseline is not None:
            found = testguard.problems(baseline, testguard.scan(self.project_dir))
            if found:
                return CheckResult(False, 1, res.output + testguard.problems_feedback(found), res.seconds)
        return res

    def _setup(self, workspace) -> None:
        for cmd in self.contract.setup:
            self.log.say(f"setup: {cmd}")
            res = self._exec(workspace, cmd, self.settings.setup_timeout)
            self.log.event("setup", cmd=cmd, exit_code=res.exit_code, output=res.output[-2000:])
            if not res.passed:
                self.log.say(f"setup command failed (exit {res.exit_code}); continuing anyway.\n{res.output[-1500:]}")

    # --- main loop --------------------------------------------------------

    def run(self) -> bool:
        from pydantic import SecretStr

        from openhands.sdk import LLM
        from openhands.tools.preset.default import get_default_agent
        from openhands.workspace import DockerWorkspace

        s = self.settings
        sampling = {k: v for k, v in (("top_p", s.top_p), ("top_k", s.top_k)) if v is not None}
        llm = LLM(
            **sampling,
            # Not a field of the SDK's LLM; sent to Ollama in the request body.
            litellm_extra_body={"presence_penalty": s.presence_penalty} if s.presence_penalty else {},
            usage_id="agent",
            model=f"openai/{s.model}",
            base_url=s.openai_base_url,
            api_key=SecretStr("ollama"),  # Ollama ignores it, but the client wants one
            max_input_tokens=s.max_input_tokens,
            max_output_tokens=s.max_output_tokens,
            temperature=s.temperature,
            timeout=s.llm_timeout,
            stream=s.stream,  # token deltas for the live view in the web UI's chat
        )
        agent = get_default_agent(llm=llm, cli_mode=True)

        if not self.contract.pending() and not s.final_review:
            self.log.say("All milestones are already done.")
            return True

        mount_source = host_path(str(self.project_dir))
        self.log.say(f"Starting sandbox ({s.server_image}), mounting {mount_source} at {PROJECT_DIR}")
        with DockerWorkspace(
            server_image=s.server_image,
            working_dir=PROJECT_DIR,
            volumes=[f"{mount_source}:{PROJECT_DIR}"],
        ) as workspace:
            try:
                self._check_stop()
                self._setup(workspace)
                round_n = 0
                while True:
                    for milestone in self.contract.pending():
                        if milestone.status != "pending":  # e.g. skipped meanwhile
                            continue
                        self._check_stop()
                        self._run_milestone(workspace, agent, milestone)
                    if not s.final_review:
                        break
                    self._check_stop()
                    round_n += 1
                    # The last round only reports, so the review always ends.
                    if not self._final_review(workspace, round_n, may_add=round_n <= s.review_rounds):
                        break
            except Quit:
                self.log.say("Stopped. Progress is saved; run again to resume.")
                return False
            except KeyboardInterrupt:
                self.log.say("Interrupted. Progress is saved; run again to resume.")
                return False

        done = len(self.contract.completed())
        total = len(self.contract.milestones)
        self.log.say(f"Finished: {done}/{total} milestones done.")
        report = review.load_report(self.project_dir) if s.final_review else None
        if report is not None and report.get("verdict") not in ("pass", "error"):
            return False  # the final review found something still missing
        return done == total

    # --- final review -----------------------------------------------------

    def _final_review(self, workspace, round_n: int, may_add: bool) -> bool:
        """Check the finished project as a whole (see review.py). Returns True
        if it found work for the agent (milestones re-opened or added)."""
        s = self.settings
        report = review.new_report(round_n)
        self._milestone_n = 0
        self.log.say(f"=== Final review (round {round_n}): re-running every check, then testing the goal's requirements")
        self.log.chat({"role": "system", "level": "divider", "text": f"Final review, round {round_n}"})

        # 1. Regressions: a later milestone may have broken an earlier one.
        for i, m in enumerate(self.contract.milestones, 1):
            if m.status != "done":
                continue
            self._check_stop()
            res = self._check(workspace, m.check)
            self.log.chat({"role": "check", "command": m.check, "passed": res.passed, "exit_code": res.exit_code,
                           "text": res.output[-4000:], "decision": "regression check",
                           "reason": f"milestone {i}: {m.title}", "changed": [], "seconds": round(res.seconds, 1)})
            if not res.passed:
                report["regressions"].append({"milestone": i, "title": m.title, "output": res.output[-2000:]})
                if may_add:
                    m.status = "pending"
                    m.restarts = 0
                    m.notes.append(review.regression_note(res.output))

        if report["regressions"]:
            report["verdict"] = "regressions"
            self._finish_review(report, may_add)
            return may_add  # fix those first; requirements get reviewed next round

        # 2. Acceptance. First the whole test suite the way a fresh machine would
        #    run it (Rust/Python, see testguard.py), then one end-to-end command
        #    per requirement of the goal, written by the model.
        requirements = testguard.suite_checks(self.project_dir) if s.test_guard else []
        try:
            tree, sources = review.gather_sources(self.project_dir, int(s.max_input_tokens * 3 * 0.6))
            requirements += review.ask_reviewer(
                review.review_prompt(self.contract, tree, sources), s.ollama_url, s.model, s.llm_timeout
            )
        except Exception as exc:  # model down, bad JSON...: report it, still run the suite checks
            report["error"] = f"{type(exc).__name__}: {exc}"[:500]

        known_checks = {m.check for m in self.contract.milestones}
        for req in requirements:
            self._check_stop()
            entry = {"requirement": req["requirement"], "check": req["check"], "passed": None, "output": ""}
            if req["check"]:
                res = self._check(workspace, req["check"])
                entry["passed"], entry["output"] = res.passed, res.output[-2000:]
                self.log.chat({"role": "check", "command": req["check"], "passed": res.passed, "exit_code": res.exit_code,
                               "text": res.output[-4000:], "decision": "requirement",
                               "reason": req["requirement"], "changed": [], "seconds": round(res.seconds, 1)})
                # Same check already tried as a milestone (e.g. you skipped it): don't add it again.
                if not res.passed and may_add and req["check"] not in known_checks:
                    milestone = review.fix_milestone(req["requirement"], req["check"], res.output)
                    self.contract.milestones.append(milestone)
                    known_checks.add(req["check"])
                    report["added"].append(milestone.title)
            report["requirements"].append(entry)

        failed = [r for r in report["requirements"] if r["passed"] is False]
        report["verdict"] = "fail" if failed else "error" if report["error"] else "pass"
        self._finish_review(report, may_add)
        return bool(report["added"])

    def _finish_review(self, report: dict, may_add: bool) -> None:
        self.contract.save(self.project_dir)
        review.save_report(self.project_dir, report)
        reqs = report["requirements"]
        passed = sum(r["passed"] is True for r in reqs)
        failed = sum(r["passed"] is False for r in reqs)
        unchecked = sum(r["passed"] is None for r in reqs)
        if report["verdict"] == "error":
            text = f"Final review could not run: {report['error']}"
        elif report["verdict"] == "regressions":
            names = ", ".join(f"{r['milestone']}. {r['title']}" for r in report["regressions"])
            text = f"Final review: {len(report['regressions'])} finished milestone(s) are broken again: {names}."
            text += " Re-opened them." if may_add else " (Review rounds used up; left as they are.)"
        else:
            text = f"Final review: {passed}/{len(reqs)} requirements of the goal pass"
            text += f", {unchecked} not checkable by a command." if unchecked else "."
            if report["added"]:
                text += f" Added {len(report['added'])} milestone(s) for the failing ones."
            elif failed:
                text += f" {failed} still fail{'s' if failed == 1 else ''}" + (
                    " (review rounds used up)." if not may_add else " (already tried as milestones).")
        self.log.say(text)
        self.log.chat({"role": "system", "level": "error" if report["verdict"] in ("fail", "error", "regressions") else "info",
                       "text": text})
        self.log.event("review", round=report["round"], verdict=report["verdict"], passed=passed, failed=failed,
                       unchecked=unchecked, regressions=len(report["regressions"]), added=report["added"],
                       error=report["error"])

    def _run_milestone(self, workspace, agent, milestone: Milestone) -> None:
        from openhands.sdk import Conversation

        n = self.contract.index_of(milestone)
        total = len(self.contract.milestones)
        limits = self.settings.limits
        state = MilestoneState(restarts=milestone.restarts)
        handoff = ""
        self.log.say(f"=== Milestone {n}/{total}: {milestone.title}  (check: {milestone.check})")
        self.log.event("milestone_start", n=n, title=milestone.title)
        self._milestone_n = n
        # The tests as they are now: the agent may add tests, not skip or delete these.
        baseline = testguard.scan(self.project_dir)

        while True:  # one iteration per conversation
            self.log.chat(
                {
                    "role": "system",
                    "level": "divider",
                    "text": f"Milestone {n}/{total}: {milestone.title}"
                    + (f" (fresh conversation, restart {state.restarts})" if state.restarts else ""),
                },
                milestone=n,
            )
            collector = TurnCollector(on_activity=self._activity)
            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                callbacks=[collector.on_event],
                max_iteration_per_run=self.settings.iterations_per_turn,
                stuck_detection=True,
            )
            self._conversation = conversation
            message = milestone_prompt(self.contract, milestone, handoff)
            try:
                while True:  # one iteration per turn
                    before = progress.snapshot(self.project_dir)
                    tokens_before = _token_usage(conversation)
                    started = time.monotonic()
                    collector.reset()
                    self._loop.reset()
                    self._looping = ""
                    conversation.send_message(message)
                    exception = ""
                    try:
                        conversation.run()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        exception = f"{type(exc).__name__}: {exc}"
                    collector.wait_until_quiet()
                    self.live.end()
                    self._check_stop()
                    status = _status_of(conversation)
                    changed = progress.changed_files(before, progress.snapshot(self.project_dir))
                    report = collector.report(status, exception, changed_files=len(changed))
                    if self._looping:
                        report.end, report.error = TurnEnd.LOOPING, self._looping
                        self.log.chat({"role": "system", "level": "error", "text":
                                       f"Cut the agent off: it kept repeating itself while thinking.\nRepeated: \"{self._looping}\""},
                                      milestone=n)
                    agent_seconds = time.monotonic() - started
                    tokens = _token_usage(conversation)
                    check_started = time.monotonic()
                    check = self._check(workspace, milestone.check, baseline)
                    check_seconds = time.monotonic() - check_started
                    decision = decide(report, check, milestone.check, state, limits)
                    self.log.chat(
                        {
                            "role": "check",
                            "command": milestone.check,
                            "passed": check.passed,
                            "exit_code": check.exit_code,
                            "text": check.output[-4000:],
                            "decision": decision.action.value,
                            "reason": decision.reason,
                            "changed": changed[:50],
                            "seconds": round(check.seconds, 1),
                        },
                        milestone=n,
                    )

                    self.log.event(
                        "turn",
                        milestone=n,
                        end=report.end.value,
                        actions=report.actions,
                        tools=collector.tool_counts,
                        changed=changed[:50],
                        check_passed=check.passed,
                        check_exit=check.exit_code,
                        check_output=check.output[-3000:],
                        decision=decision.action.value,
                        reason=decision.reason,
                        agent_text=report.agent_text[:2000],
                        error=report.error,
                        duration=round(agent_seconds, 1),
                        check_seconds=round(check_seconds, 1),
                        prompt_tokens=max(0, tokens["prompt"] - tokens_before["prompt"]),
                        completion_tokens=max(0, tokens["completion"] - tokens_before["completion"]),
                        context_tokens=tokens["context"],
                    )
                    self.log.say(
                        f"turn ended: {report.end.value}, {report.actions} actions, "
                        f"{len(changed)} files changed, check {'PASSED' if check.passed else 'failed'} "
                        f"-> {decision.action.value.upper()}: {decision.reason}"
                    )

                    if decision.action == Action.DONE:
                        milestone.status = DONE
                        milestone.restarts = state.restarts
                        self.contract.save(self.project_dir)
                        self.log.say(f"Milestone {n} done.")
                        return
                    # Stop pressed while the check ran (it can take minutes):
                    # don't start another turn or ask the human first.
                    if decision.action == Action.RESTART:
                        milestone.restarts = state.restarts
                        self.contract.save(self.project_dir)
                    self._check_stop()
                    if decision.action == Action.NUDGE:
                        message = decision.message
                        continue
                    if decision.action == Action.RESTART:
                        milestone.restarts = state.restarts
                        self.contract.save(self.project_dir)
                        handoff = handoff_note(
                            decision.reason, progress.file_tree(self.project_dir), milestone.check, check.output
                        )
                        self.log.say(f"Restarting milestone {n} in a fresh conversation (restart {state.restarts}).")
                        break
                    # ESCALATE
                    if self._escalate(milestone, decision, check, state):
                        return  # skipped
                    handoff = handoff_note(
                        "the user stepped in", progress.file_tree(self.project_dir), milestone.check, check.output
                    )
                    break
            finally:
                self._conversation = None
                try:
                    conversation.close()
                except Exception:
                    pass

    def _escalate(self, milestone: Milestone, decision: Decision, check: CheckResult, state: MilestoneState) -> bool:
        """Ask the human (or apply the unattended policy). Returns True if the
        milestone was skipped, False to retry it; raises Quit to stop."""
        mode = self.on_escalate if isinstance(self.on_escalate, str) else "custom"
        self.log.event(
            "escalate",
            title=milestone.title,
            reason=decision.reason,
            detail=decision.message,
            check_output=check.output[-2000:],
            mode=mode,
        )
        if mode == "skip":
            self.log.say(f"Agent can't finish '{milestone.title}' ({decision.reason}). Skipping it (ON_ESCALATE=skip).")
            action, text = "skip", ""
        elif mode == "stop":
            self.log.say(
                f"Agent needs help with '{milestone.title}': {decision.reason}\n"
                f"Last check output:\n{check.output[-1500:]}\n"
                "Stopping. To continue: add a hint to this milestone's \"notes\" in "
                ".supervisor/contract.json and start again, or use the web UI / interactive mode."
            )
            action, text = "stop", ""
        elif mode == "custom":
            action, text = self.on_escalate(milestone, decision, check)
        else:
            action, text = self._ask_terminal(milestone, decision, check)

        if action == "stop":
            raise Quit()
        if action == "skip":
            milestone.status = SKIPPED
            self.contract.save(self.project_dir)
            self.log.event("skip", title=milestone.title)
            return True
        # guidance
        milestone.notes.append(text)
        state.restarts = 0  # fresh budget after human guidance
        state.questions = 0
        state.new_conversation()
        milestone.restarts = 0
        self.contract.save(self.project_dir)
        self.log.event("guidance", title=milestone.title, note=text)
        self.log.say(f"Retrying '{milestone.title}' with your guidance.")
        return False

    def _ask_terminal(self, milestone: Milestone, decision: Decision, check: CheckResult) -> tuple[str, str]:
        print("\n" + "=" * 70 + f"\nThe agent needs help with: {milestone.title}\nReason: {decision.reason}\n")
        if decision.message and decision.message != decision.reason:
            print(f"Agent said:\n{decision.message[-1500:]}\n")
        print(f"Last check output (`{milestone.check}`):\n{check.output[-1500:]}\n")
        print(
            "Type guidance for the agent and press Enter to retry with it,\n"
            "  's' to skip this milestone, or 'q' to quit (progress is saved)."
        )
        while True:
            answer = input("> ").strip()
            if answer.lower() == "q":
                return "stop", ""
            if answer.lower() == "s":
                return "skip", ""
            if answer:
                return "guidance", answer


def _token_usage(conversation) -> dict:
    """Cumulative LLM tokens of this conversation so far, and `context`: how
    full the agent's context is (its latest call's prompt + reply). Zeros if
    the SDK doesn't report them.

    With a Docker sandbox the agent-server sends a stats *snapshot* that drops
    the per-call token list, so `context` comes from the accumulated usage's
    `per_turn_token` (the SDK keeps the latest call's size there "for
    calculating context usage"), read from the agent's own LLM ("agent"), not
    the combined total, whose last call may be the history summarizer's.
    """
    try:
        stats = conversation.conversation_stats
        combined = stats.get_combined_metrics().accumulated_token_usage
    except Exception:
        return {"prompt": 0, "completion": 0, "context": 0}
    context = 0
    try:
        agent_metrics = (getattr(stats, "usage_to_metrics", None) or {}).get("agent")
        if agent_metrics is not None:
            latest = getattr(agent_metrics, "token_usages", None)
            if latest:  # full metrics (local conversations): exact prompt size
                context = int(getattr(latest[-1], "prompt_tokens", 0) or 0)
            elif agent_metrics.accumulated_token_usage is not None:
                context = int(agent_metrics.accumulated_token_usage.per_turn_token or 0)
        if not context and combined is not None:
            context = int(getattr(combined, "per_turn_token", 0) or 0)
    except Exception:
        pass
    return {
        "prompt": int(getattr(combined, "prompt_tokens", 0) or 0),
        "completion": int(getattr(combined, "completion_tokens", 0) or 0),
        "context": context,
    }


def _status_of(conversation) -> str:
    try:
        status = conversation.state.execution_status
    except Exception:
        return ""
    return str(getattr(status, "value", status))
