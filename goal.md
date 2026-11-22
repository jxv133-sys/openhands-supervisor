<!--
  Write what you want built below. Everything outside these comment markers
  is sent to the planner as the goal. Be concrete: language, features, how
  it's used, where data is stored. Then run: docker compose up --build

  The goal is only read when a project is planned. To build something new
  after that, also change PROJECT in .env (or the old plan just continues).
-->

# Todo CLI in Rust

A command-line todo app written in Rust.

- Commands: `add <title>`, `list`, `done <id>`, `remove <id>`
- Tasks are stored in a JSON file (`todos.json` in the current directory)
- `list` shows each task's id, title, and whether it's done
- Clear error messages for unknown ids or commands
