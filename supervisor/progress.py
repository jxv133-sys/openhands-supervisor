"""Measure real progress from the project files on disk (the project folder is
mounted into the sandbox, so the supervisor sees every change directly).

Counting file changes this way works no matter how the agent wrote them:
the file editor tool, `cat > file`, sed, a code generator, anything.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Build output, dependencies and our own state: changes here are not "progress".
IGNORED_DIRS = {
    ".git", ".supervisor", "target", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cargo", ".idea",
}
MAX_FILE_BYTES = 2_000_000


def _iter_files(root: Path):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.is_symlink():
                continue
            if entry.is_dir():
                if entry.name not in IGNORED_DIRS:
                    stack.append(entry)
            elif entry.is_file():
                yield entry


def snapshot(root: Path) -> dict[str, str]:
    """Map of relative path -> content hash for every tracked project file."""
    root = Path(root)
    result: dict[str, str] = {}
    for path in _iter_files(root):
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                digest = f"size:{path.stat().st_size}:{path.stat().st_mtime_ns}"
            else:
                digest = hashlib.sha1(path.read_bytes()).hexdigest()
        except OSError:
            continue  # e.g. root-owned file created in the sandbox; skip, don't crash
        result[path.relative_to(root).as_posix()] = digest
    return result


def changed_files(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Files added, modified or deleted between two snapshots."""
    keys = set(before) | set(after)
    return sorted(k for k in keys if before.get(k) != after.get(k))


def file_tree(root: Path, limit: int = 80) -> str:
    """Short listing for handoff prompts."""
    files = sorted(snapshot(root))
    shown = files[:limit]
    more = f"\n... and {len(files) - limit} more" if len(files) > limit else ""
    return "\n".join(shown) + more if shown else "(no files yet)"
