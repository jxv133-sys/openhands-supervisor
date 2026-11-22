from supervisor.hostpath import container_id_from_mountinfo, host_path, map_path

CID = "a" * 64
MOUNTINFO = (
    "612 590 0:52 / / rw,relatime master:301 - overlay overlay rw\n"
    f"640 612 259:2 /var/lib/docker/containers/{CID}/hostname /etc/hostname rw - ext4 /dev/nvme0n1p2 rw\n"
)

MOUNTS = [
    {"Type": "bind", "Source": "/var/run/docker.sock", "Destination": "/var/run/docker.sock"},
    {"Type": "bind", "Source": "/home/me/openhands-supervisor/projects", "Destination": "/projects"},
    {"Type": "bind", "Source": "/srv/special", "Destination": "/projects/special"},
]


def test_container_id():
    assert container_id_from_mountinfo(MOUNTINFO) == CID
    assert container_id_from_mountinfo("no docker here") is None


def test_map_path():
    assert map_path("/projects/todo-rs", MOUNTS) == "/home/me/openhands-supervisor/projects/todo-rs"
    assert map_path("/projects", MOUNTS) == "/home/me/openhands-supervisor/projects"
    # longest mount wins
    assert map_path("/projects/special/x", MOUNTS) == "/srv/special/x"
    # prefix must match whole path components
    assert map_path("/projectsX/a", MOUNTS) is None
    assert map_path("/elsewhere", MOUNTS) is None


def test_override_env(monkeypatch):
    monkeypatch.setenv("HOST_PROJECTS_DIR", "/data/proj")
    assert host_path("/projects/todo") == "/data/proj/todo"
    assert host_path("/other/todo") == "/other/todo"
