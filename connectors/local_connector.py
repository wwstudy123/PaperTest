"""本地执行适配：供专用容器使用，不需要 SSH 或 sudo 常驻服务。"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import stat
import subprocess
import tempfile
import time

from admtune.security import redact
from connectors.ssh_connector import SSHConnector
from models import CommandResult


class LocalConnector(SSHConnector):
    def connect(self) -> None:
        if os.name != "posix":
            raise RuntimeError("本地实机模式仅支持类 Unix 专用实验环境")

    def close(self) -> None:
        pass

    def run(self, command: str, *, sudo: bool = False, timeout: int = 60, check: bool = False) -> CommandResult:
        if self.dry_run:
            return super().run(command, sudo=sudo, timeout=timeout, check=check)
        actual = self._wrap_sudo(command) if sudo and os.geteuid() != 0 else command
        started = time.monotonic()
        process = subprocess.Popen(
            ["bash", "-c", actual], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            code = process.returncode
        except subprocess.TimeoutExpired:
            # 只终止本次创建的进程组，禁止按 mysqld 等名称全局杀进程。
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            code = 124
            stderr += "\n本地命令超时，已终止本次命令进程组。"
        result = CommandResult(redact(actual), stdout, redact(stderr), code, time.monotonic() - started)
        if self.command_callback:
            self.command_callback(result)
        if check and not result.ok:
            raise RuntimeError(redact(f"本地命令失败（{code}）：{actual}\n{stderr}"))
        return result

    def exists(self, path: str, *, sudo: bool = False) -> bool:
        if self.dry_run:
            return super().exists(path, sudo=sudo)
        if sudo and os.geteuid() != 0:
            return self.run(f"test -e {shlex.quote(path)}", sudo=True).ok
        return Path(path).exists()

    def read_file(self, path: str, *, sudo: bool = False) -> str:
        if self.dry_run:
            return super().read_file(path, sudo=sudo)
        if sudo and os.geteuid() != 0:
            return self.run(f"cat -- {shlex.quote(path)}", sudo=True, check=True).stdout
        return Path(path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str, *, sudo: bool = False) -> None:
        if self.dry_run:
            return super().write_file(path, content, sudo=sudo)
        if sudo and os.geteuid() != 0:
            import base64
            encoded = base64.b64encode(content.encode()).decode()
            self.run(
                f"printf '%s' {shlex.quote(encoded)} | base64 -d > {shlex.quote(path)}",
                sudo=True, check=True,
            )
            return
        target = Path(path)
        if target.is_symlink():
            raise ValueError("本地配置写入拒绝符号链接，请指定实际配置文件")
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.stat() if target.exists() else None
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
                if existing:
                    os.fchmod(handle.fileno(), stat.S_IMODE(existing.st_mode))
                    if os.geteuid() == 0:
                        os.fchown(handle.fileno(), existing.st_uid, existing.st_gid)
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def copy_file(self, source_path: str, target_path: str, *, sudo: bool = False) -> None:
        self.write_file(target_path, self.read_file(source_path, sudo=sudo), sudo=sudo)
