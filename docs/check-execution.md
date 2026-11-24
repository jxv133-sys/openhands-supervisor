# Why milestone checks passed and the final review "regressed"

## What happened

During a run every milestone's acceptance check passed. The final review then
reported 14 regressions. Several said `ModuleNotFoundError: No module named
'main'`. pytest-based checks passed while checks using inline Python heredocs
failed; the same heredoc checks passed when run by hand and in later milestone
turns. The failing checks created a temp directory, copied the project into it
from `/app`, and ran Python there. `/app` did not exist during the failing
review, and later existed as a symlink to the project folder.

## How checks were executed

Every run (`Supervisor.run`) starts **one new sandbox container**, runs the
plan's setup commands once, then runs the milestones and the final review in
that container. Milestone checks and review checks both went through the same
`_exec` call: `bash -lc 'set -o pipefail; <check>'`, working directory = the
project folder. So *within one run* the two paths were already identical.

The divergence was in what surrounded the checks:

1. **Undeclared environment (the root cause here).** The agent works in the
   sandbox with full control. It created `/app → project` (and can install
   system-wide packages, tools, links...) by hand. Those changes live in that
   container only. Checks that depended on them passed in the agent's sandbox.
   Any later run starts a *fresh* container: a resumed run, a container
   restart, or "Run review" on a finished project. There the check fails, and
   the review counted the failure as a regression. `/app` was one instance of
   this class of problem, not the whole of it.
2. **Setup failures were ignored.** A failing setup command was logged and the
   run continued ("continuing anyway"), so checks could run in a
   half-prepared sandbox and their failures were blamed on the project.
3. **The mount was never verified.** Nothing proved the project folder was
   actually visible in the sandbox before checks ran. A missing or mis-mapped
   mount looks exactly like "every check fails".
4. **Infrastructure errors looked like test failures.** An exception reaching
   the sandbox (API error, timeout) became an ordinary failed check (exit -1).
5. **The agent tested in a different environment.** The agent's terminal is a
   long-lived session: whatever it `export`s, `cd`s into or activates stays
   there. Checks run in a fresh shell. So "it passes when I run it" was true
   and irrelevant, which fed the agent's circular investigation.

## The fix (`supervisor/sandbox.py`)

One lifecycle for everything the supervisor runs in the sandbox:

- **Prepare** before any check: write a random token file into the project
  folder on the host and require the sandbox to see it (proves the mount);
  run the plan's setup commands and the project's own `.openhands/setup.sh`
  (OpenHands' convention for "run this in every sandbox"). Any failure raises
  `SandboxError` with the evidence. The run stops as an *infrastructure error*;
  no check result is counted.
- **Run** every command (setup, milestone checks, review checks) through one
  wrapper: login shell, explicit working directory, mount token re-verified,
  `pipefail`. If the folder is missing or isn't the project, the result carries
  `infra` and is reported as such, never as a failure.
- **Drift**: a fingerprint of the sandbox outside the project folder (names,
  types and link targets of what's installed or linked in system locations,
  plus Python site-packages), taken after preparation. When a milestone
  passes in a sandbox that has drifted, the agent is asked once to declare
  those changes in `.openhands/setup.sh` (or make the project independent of
  them), and the changes are recorded on the milestone (`env_changes`).
- **Classify** in the final review: a check that fails because of an `infra`
  result, or that passed during its milestone with recorded changes this
  sandbox lacks, is reported as an **environment** problem (verdict
  `environment`), with the missing changes listed. The milestone is not
  re-opened and no fix milestones are added, because the code didn't break.
  Real failures are still regressions.
- **Tell the agent** how checks really run (fresh login shell in the project
  folder, not its terminal) and where environment changes belong.

Nothing here knows about `/app`, Python imports or any project's checks.

## Tests

- `tests/test_sandbox.py`: the lifecycle with real bash, including a
  reproduction of the failing heredoc check, unmounted/missing folders,
  failed setup, sandbox API errors, a mount lost mid-run.
- `tests/test_check_paths.py`: the real supervisor paths. A milestone check and
  the same check in the review produce the identical command; the fresh-sandbox
  scenario is classified as environment, not regression; a genuine regression
  is still reported and re-opened; a declared `.openhands/setup.sh` makes the
  fresh sandbox equivalent; initialization failures are infrastructure errors
  in the log, the review and the web job.
