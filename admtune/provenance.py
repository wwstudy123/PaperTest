from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess


def source_provenance() -> dict:
    root = Path(__file__).resolve().parent.parent
    payload = {"python": platform.python_version(), "dependencies": {}}
    for name in ["numpy", "scipy", "sqlglot", "paramiko", "PyYAML", "requests"]:
        try:
            payload["dependencies"][name] = version(name)
        except PackageNotFoundError:
            payload["dependencies"][name] = "unavailable"
    try:
        def git(*args):
            return subprocess.check_output(
                ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL,
            ).strip()
        payload["git_revision"] = git("rev-parse", "HEAD")
        payload["git_dirty"] = bool(git("status", "--porcelain"))
        files = git("ls-files", "--cached", "--others", "--exclude-standard").splitlines()
        hasher = hashlib.sha256()
        for name in sorted(files):
            path = root / name
            if path.is_file():
                hasher.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
        payload["working_source_sha256"] = hasher.hexdigest()
    except (OSError, subprocess.CalledProcessError):
        payload.update(git_revision="unavailable", git_dirty=None, working_source_sha256=None)
    return payload
