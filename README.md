# openhands-supervisor

Babysits an [OpenHands](https://github.com/OpenHands/software-agent-sdk) coding
agent running on a local Ollama model, so you don't have to.

OpenHands does the actual work (reading files, running commands, iterating).
The supervisor does what you were doing by hand:

| The agent... | The supervisor... |
|---|---|
| says it's done | runs the milestone's **check command** itself. If it fails, sends back the real error: "You are NOT done." |
| stops to ask "should I continue?" | answers "yes, decide yourself" (and asks *you* only if it keeps asking) |
| stops mid-task / returns nothing useful | tells it to keep going, with the current check output |
| loops, or repeats the same failure 3 turns in a row | throws away the conversation and restarts the milestone fresh, with a handoff note |
| burns a whole turn without changing any file | restarts the milestone |
| still can't do it after 3 restarts | asks you for guidance (type a hint, skip, or quit) |

When the last milestone is done, a **final review** checks the project as a
whole, again with real commands:

1. Every finished milestone's check runs again. One that a later milestone broke
   goes back to *pending* (with the failure as a note) and the agent fixes it.
2. Your model lists each requirement in the goal and writes one end-to-end test
   command for it (build the program, run it, check the output). The supervisor
   runs them. Each failing requirement becomes a new milestone, with that
   command as its check, and the agent works on it like any other.
3. This repeats up to `review_rounds` times (default 2); the round after that only
   reports. The result shows in the web UI's *Final review* card, in
   `supervisor status`, and in `.supervisor/review.json`. Requirements no command
   can test (e.g. "messages are clear") are listed for you to check yourself.

On a finished project, *Run* becomes *Run final review*. Turn it off with
`final_review = false` in `supervisor.toml`.

Work is split into **milestones**, each run in a fresh conversation with a small
scope. That's where local models do their best work: a 30K-token conversation
full of `apt` output is where they get lost.

"Done" is decided by real commands (`cargo build`, `pytest`, ...), never by the
model's opinion. Progress is measured by hashing the project files on disk, so
it works however the agent writes files.

## Quick start (Docker Compose)

Needs a Linux machine with Docker, and an Ollama server reachable from it
(the default `supervisor.toml` points at `http://192.168.50.95:11434`).

```bash
git clone <this repo> && cd openhands-supervisor
docker compose up --build -d
```

Then open **http://<that-machine>:8765** in a browser.

### Web interface

- **New project**: name it, describe what to build (pre-filled from `goal.md`),
  press *Plan milestones*. Your local model drafts the plan in ~30 seconds.
- **Review the plan**: edit the goal, setup commands, and each milestone's
  title, instructions, check command and notes. Reorder, add, delete, or set a
  milestone back to `pending` to redo it. *Save*.
- **Run**: watch it live on the right: each turn's outcome (actions, files
  changed, check passed/failed, and what the supervisor decided), plus the
  agent's individual steps. The *Chat* tab shows the agent's reply **as the
  model writes it**, token by token (tick *thinking* to watch its reasoning
  too), and a status line says what it's doing right now ("Running $ cargo
  build... 42s"). Set `stream = false` in `supervisor.toml` if your Ollama
  version has trouble streaming with tool calls.
- **Stats** tab: working time, turns, agent actions, check pass rate, restarts,
  how often it needed you, files changed, tokens in/out and peak context size
  (vs `max_input_tokens`); a turn-by-turn timeline (hover a bar for details); a
  per-milestone table; and breakdowns of why the supervisor stepped in, how
  turns ended, which tools the agent used, and the most-changed files. Runs
  logged before this version show no tokens, and their time is estimated from
  log timestamps.
- **When the agent needs help**, a card shows why, what the agent said, and the
  check's output. Type a hint and *Retry with guidance*, or *Skip* / *Stop*.
- **Stop** any time; progress is saved, *Run* resumes. Runs keep going when you
  close the tab; one run at a time (one GPU).
- The browser tab title shows `⚠ <project> needs help` when the agent is waiting
  on you, and (if you allow it) you get a desktop notification while the tab is
  in the background. Finished milestones fold to one line; *Expand all* shows them.
- Shortcuts: `Ctrl+S` saves the plan, `Ctrl+Enter` plans a new project or sends
  guidance. On narrow screens the project list is behind the ☰ button.

The code ends up in `./projects/<name>` on the host.

Optional `.env` settings (copy `.env.example`):

| Variable | Meaning |
|---|---|
| `WEB_PORT` | web UI port (default 8765) |
| `WEB_TOKEN` | if set, the UI asks for this token. Anyone who can open the UI can make the agent run code in its sandbox, so set it on untrusted networks |

Settings (model, context limits, restart limits) are in `supervisor.toml`; edit
it and `docker compose restart`, no rebuild needed.

### Without the web UI

```bash
# Unattended: plans from goal.md into ./projects/$PROJECT, then runs.
docker compose run --rm supervisor auto
# Interactive in the terminal (asks you when the agent is stuck):
docker compose run --rm supervisor run todo-rs
docker compose run --rm supervisor status todo-rs
```

`auto` reads `PROJECT`, `REVIEW_PLAN` (1 = stop after planning) and
`ON_ESCALATE` (`stop` or `skip`) from `.env`. `GOAL`, if set, is used instead
of `goal.md`.

How it works: the supervisor container gets the host's Docker socket, so the
agent sandboxes it starts are normal sibling containers on the host (visible in
`docker ps`). It finds the host path of `./projects` itself, so the sandbox
mounts the right folder. Access to the Docker socket is effectively root access
to the host; only run images you built.

## Install without Docker

Needs Python 3.12+ and Docker (for the agent sandbox).

```bash
cd openhands-supervisor
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest            # unit tests, no Ollama or Docker needed
```

## Use

```bash
# 1. Draft a plan (one call to your local model)
supervisor plan ~/projects/todo-rs "A Rust CLI todo app: add, list, done, remove; stores tasks in a JSON file"

# 2. Review it. Edit ~/projects/todo-rs/.supervisor/contract.json if needed,
#    especially the check commands: they define "done".
supervisor status ~/projects/todo-rs

# 3. Run. Ctrl+C any time; run again to resume from the next unfinished milestone.
supervisor run ~/projects/todo-rs
```

The project folder is mounted into the sandbox at `/workspace/project`, so the
code ends up in your folder on the host.

### The contract

`.supervisor/contract.json`:

```json
{
  "goal": "A Rust CLI todo app ...",
  "setup": ["command -v cargo || (curl -sSf https://sh.rustup.rs | sh -s -- -y)"],
  "milestones": [
    {
      "title": "Scaffold the cargo project",
      "instructions": "Create a binary crate named todo with clap and serde dependencies.",
      "check": "cargo build",
      "status": "pending"
    }
  ]
}
```

- **setup** runs (via `bash -lc`) every time the sandbox starts, before any
  milestone. Keep commands safe to re-run. Installing toolchains here means the
  agent doesn't spend 10 slow turns doing it.
- **check** is one shell command run from the project root; exit code 0 means
  done. Make it strict: `cargo build` only proves it compiles; `cargo test add`
  proves a feature works.
- Set a milestone's `status` back to `"pending"` to redo it, or to `"skipped"`
  to skip it.

### Logs

Everything the supervisor decides goes to `.supervisor/log.jsonl`: each turn's
outcome, files changed, check result, and the decision with its reason.

## Settings

`supervisor.toml` in the current directory (or `--config path`). See the
shipped file for every option with defaults. The important ones:

- `max_input_tokens` must stay well below the model's context window (qwen-abl
  is 49152). OpenHands summarizes history to stay under it. If it's too close,
  Ollama silently cuts the start of the prompt and the agent loses its task.
- `iterations_per_turn`: tool calls per turn before the supervisor checks in.
- `[limits]`: when to restart and when to ask you.

## Notes

- The sandbox image has no Rust/Node toolchains; `setup` installs them each run.
  To skip that, build your own image `FROM ghcr.io/openhands/agent-server:latest-python`
  with the toolchain installed and set `server_image`.
- Files the agent creates may be owned by the container's user. If your host
  user can't edit them afterwards: `sudo chown -R $USER ~/projects/todo-rs`.
