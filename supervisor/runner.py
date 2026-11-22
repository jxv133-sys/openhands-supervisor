"""Drives OpenHands (via its SDK) milestone by milestone, applying the watchdog.

The agent runs inside a Docker sandbox (OpenHands agent-server image) with the
project folder mounted at /workspace/project. The supervisor runs on the host,
reads the project files directly to measure progress, and runs each
milestone's check inside the same sandbox, so it sees exactly what the agent sees.
"""

from __future__ import annotations

import shlex
import time
from pathlib import Path

from . import progress
from .collector import TurnCollector
from .config import Settings
from .contract import DONE, SKIPPED, Contract, Milestone
from .log import RunLog
from .prompts import PROJECT_DIR, handoff_note, milestone_prompt
from .watchdog import Action, CheckResult, Decision, MilestoneState, decide


class Quit(Exception):
    pass


def _bash(cmd: str) -> str:
    # Login shell, so toolchains installed by setup (rustup, nvm...) are on PATH.
    # pipefail: `cargo build | tee log` must fail when cargo fails.
    return f"bash -lc {shlex.quote('set -o pipefail; ' + cmd)}"


class Supervisor:
    def __init__(self, settings: Settings, project_dir: Path, contract: Contract, log: RunLog):
        self.settings = settings
        self.project_dir = Path(project_dir).resolve()
        self.contract = contract
        self.log = log

    # --- sandbox helpers ------------------------------------------------

    def _exec(self, workspace, cmd: str, timeout: int) -> CheckResult:
        try:
            result = workspace.execute_command(_bash(cmd), cwd=PROJECT_DIR, timeout=timeout)
        except Exception as exc:  # sandbox/API failure: report as a failed check
            return CheckResult(False, -1, f"supervisor could not run the command: {exc}")
        output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
        if getattr(result, "timeout_occurred", False):
            output += f"\n(command timed out after {timeout}s)"
        return CheckResult(result.exit_code == 0, result.exit_code, output[-8000:])

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
        llm = LLM(
            usage_id="agent",
            model=f"openai/{s.model}",
            base_url=s.openai_base_url,
            api_key=SecretStr("ollama"),  # Ollama ignores it, but the client wants one
            max_input_tokens=s.max_input_tokens,
            max_output_tokens=s.max_output_tokens,
            temperature=s.temperature,
            timeout=s.llm_timeout,
        )
        agent = get_default_agent(llm=llm, cli_mode=True)

        pending = self.contract.pending()
        if not pending:
            self.log.say("All milestones are already done.")
            return True

        self.log.say(f"Starting sandbox ({s.server_image}), project mounted at {PROJECT_DIR}")
        with DockerWorkspace(
            server_image=s.server_image,
            working_dir=PROJECT_DIR,
            volumes=[f"{self.project_dir}:{PROJECT_DIR}"],
        ) as workspace:
            self._setup(workspace)
            try:
                for milestone in pending:
                    self._run_milestone(workspace, agent, milestone)
            except Quit:
                self.log.say("Stopped by user. Progress is saved; run again to resume.")
                return False
            except KeyboardInterrupt:
                self.log.say("Interrupted. Progress is saved; run again to resume.")
                return False

        done = len(self.contract.completed())
        total = len(self.contract.milestones)
        self.log.say(f"Finished: {done}/{total} milestones done.")
        return done == total

    def _run_milestone(self, workspace, agent, milestone: Milestone) -> None:
        from openhands.sdk import Conversation

        n = self.contract.index_of(milestone)
        total = len(self.contract.milestones)
        limits = self.settings.limits
        state = MilestoneState(restarts=milestone.restarts)
        handoff = ""
        self.log.say(f"=== Milestone {n}/{total}: {milestone.title}  (check: {milestone.check})")
        self.log.event("milestone_start", n=n, title=milestone.title)

        while True:  # one iteration per conversation
            collector = TurnCollector()
            conversation = Conversation(
                agent=agent,
                workspace=workspace,
                callbacks=[collector.on_event],
                max_iteration_per_run=self.settings.iterations_per_turn,
                stuck_detection=True,
            )
            message = milestone_prompt(self.contract, milestone, handoff)
            try:
                while True:  # one iteration per turn
                    before = progress.snapshot(self.project_dir)
                    collector.reset()
                    conversation.send_message(message)
                    exception = ""
                    try:
                        conversation.run()
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        exception = f"{type(exc).__name__}: {exc}"
                    collector.wait_until_quiet()
                    status = _status_of(conversation)
                    changed = progress.changed_files(before, progress.snapshot(self.project_dir))
                    report = collector.report(status, exception, changed_files=len(changed))
                    check = self._exec(workspace, milestone.check, self.settings.check_timeout)
                    decision = decide(report, check, milestone.check, state, limits)

                    self.log.event(
                        "turn",
                        milestone=n,
                        end=report.end.value,
                        actions=report.actions,
                        tools=collector.tool_counts,
                        changed=changed[:50],
                        check_passed=check.passed,
                        check_exit=check.exit_code,
                        decision=decision.action.value,
                        reason=decision.reason,
                        agent_text=report.agent_text[:2000],
                        error=report.error,
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
                try:
                    conversation.close()
                except Exception:
                    pass

    def _escalate(self, milestone: Milestone, decision: Decision, check: CheckResult, state: MilestoneState) -> bool:
        """Ask the human. Returns True if the milestone was skipped."""
        self.log.event("escalate", title=milestone.title, reason=decision.reason, detail=decision.message)
        print(
            "\n" + "=" * 70
            + f"\nThe agent needs help with: {milestone.title}\nReason: {decision.reason}\n"
        )
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
                raise Quit()
            if answer.lower() == "s":
                milestone.status = SKIPPED
                self.contract.save(self.project_dir)
                self.log.event("skip", title=milestone.title)
                return True
            if answer:
                milestone.notes.append(answer)
                state.restarts = 0  # fresh budget after human guidance
                state.questions = 0
                state.new_conversation()
                milestone.restarts = 0
                self.contract.save(self.project_dir)
                self.log.event("guidance", title=milestone.title, note=answer)
                return False


def _status_of(conversation) -> str:
    try:
        status = conversation.state.execution_status
    except Exception:
        return ""
    return str(getattr(status, "value", status))
