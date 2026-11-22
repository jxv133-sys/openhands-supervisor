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
cp .env.example .env        # put your GOAL (and a PROJECT name) in it
docker compose up --build
```

On first start it plans milestones for your goal, then works through them.
The code ends up in `./projects/<PROJECT>`. Ctrl+C any time; `docker compose up`
resumes from the next unfinished milestone.

`.env` options (see `.env.example`):

| Variable | Meaning |
|---|---|
| `PROJECT` | folder name under `./projects`; use a new one per project |
| `GOAL` | what to build, in plain language (only used when planning a new project) |
| `REVIEW_PLAN` | `1` = stop after planning so you can edit `projects/<PROJECT>/.supervisor/contract.json`, then `up` again |
| `ON_ESCALATE` | when the agent can't finish a milestone even after restarts: `stop` (default) or `skip` |

When it stops for help, it prints why. Add a hint to that milestone's `notes`
in the contract and `up` again, or run it interactively so it asks you in the
terminal:

```bash
docker compose run --rm supervisor run todo-rs
docker compose run --rm supervisor status todo-rs
```

Settings (model, context limits, restart limits) are in `supervisor.toml`; edit
it and restart, no rebuild needed.

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
