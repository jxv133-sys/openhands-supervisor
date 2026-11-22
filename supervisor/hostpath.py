"""Translate a path inside this container to the matching path on the Docker host.

The agent sandbox is started by the *host's* Docker daemon (through the mounted
socket), so a volume like `/projects/todo:/workspace/project` must name the
host's real folder, not our container path. When the supervisor runs in a
container, we look up our own bind mounts with `docker inspect` and map the path.
Outside a container (or if anything fails) paths are returned unchanged.

Set HOST_PROJECTS_DIR to skip detection: it replaces the /projects prefix.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import PurePosixPath

_CONTAINER_ID_RE = re.compile(r"/containers/([0-9a-f]{64})/")


def container_id_from_mountinfo(text: str) -> str | None:
    """Docker bind-mounts /etc/hostname etc. from /var/lib/docker/containers/<id>/,
    which shows up in /proc/self/mountinfo. Works even with network_mode: host,
    where $HOSTNAME is the host's name rather than the container id."""
    match = _CONTAINER_ID_RE.search(text)
    return match.group(1) if match else None


def map_path(path: str, mounts: list[dict]) -> str | None:
    """Map `path` through the longest matching bind mount (Destination -> Source)."""
    target = PurePosixPath(path)
    best: tuple[int, str] | None = None
    for mount in mounts:
        dest, src = mount.get("Destination"), mount.get("Source")
        if not dest or not src:
            continue
        dest_path = PurePosixPath(dest)
        if target == dest_path or dest_path in target.parents:
            depth = len(dest_path.parts)
            if best is None or depth > best[0]:
                rest = target.relative_to(dest_path)
                best = (depth, str(PurePosixPath(src) / rest))
    return best[1] if best else None


def _own_mounts() -> list[dict]:
    with open("/proc/self/mountinfo", encoding="utf-8") as f:
        cid = container_id_from_mountinfo(f.read())
    if not cid:
        return []
    out = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Mounts}}", cid],
        capture_output=True, text=True, timeout=20, check=True,
    ).stdout
    return json.loads(out) or []


def host_path(path: str, projects_mount: str = "/projects") -> str:
    override = os.environ.get("HOST_PROJECTS_DIR")
    if override:
        mapped = map_path(path, [{"Destination": projects_mount, "Source": override}])
        return mapped or path
    if not os.path.exists("/.dockerenv"):
        return path  # not in a container: already a host path
    try:
        return map_path(path, _own_mounts()) or path
    except Exception:
        return path
