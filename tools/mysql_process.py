"""无 systemd 的专用 MySQL 实例恢复，不操作其他数据库进程。"""
from __future__ import annotations

from pathlib import PurePosixPath
import shlex

from config import MySQLSettings, resolve_env
from connectors.ssh_connector import SSHConnector


def restart_mysql_process(connector: SSHConnector, settings: MySQLSettings) -> None:
    paths = [settings.config_path, settings.mysqld_path, settings.mysqladmin_path,
             settings.socket_path, settings.pid_file]
    if any(not value or not PurePosixPath(value).is_absolute() for value in paths):
        raise ValueError("进程管理模式必须提供配置、二进制、socket 和 PID 文件的绝对路径")
    if not 1 <= settings.startup_timeout_s <= 600:
        raise ValueError("启动等待时间必须在 1 至 600 秒之间")
    quote = shlex.quote
    prefix = f"MYSQL_PWD={quote(resolve_env(settings.mysql_password_env, dry_run=connector.dry_run, default='') or '')}"
    # 关闭前用 PID 和完整启动参数核验归属；不使用 pkill/killall。
    stop = f"""
set -eu
pidfile={quote(settings.pid_file)}
if ! test -r "$pidfile"; then exit 0; fi
pid="$(cat "$pidfile")"
case "$pid" in ''|*[!0-9]*) exit 41;; esac
if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi
test -r "/proc/$pid/cmdline"
tr '\\000' '\\n' < "/proc/$pid/cmdline" | grep -Fqx -- {quote('--defaults-file=' + settings.config_path)}
{prefix} {quote(settings.mysqladmin_path)} --protocol=socket --socket={quote(settings.socket_path)} --user={quote(settings.mysql_user)} --connect-timeout=10 shutdown
for i in $(seq 1 60); do
    if ! test -e "$pidfile"; then exit 0; fi
    if ! kill -0 "$pid" 2>/dev/null; then exit 0; fi
    sleep 1
done
exit 42
"""
    connector.run(stop, sudo=True, check=True, timeout=90)
    connector.run(
        f"{quote(settings.mysqld_path)} --defaults-file={quote(settings.config_path)} --daemonize",
        sudo=True, check=True, timeout=settings.startup_timeout_s,
    )
    connector.run(
        f"for i in $(seq 1 {settings.startup_timeout_s}); do "
        f"{prefix} {quote(settings.mysqladmin_path)} --protocol=socket "
        f"--socket={quote(settings.socket_path)} --user={quote(settings.mysql_user)} "
        "--connect-timeout=2 ping >/dev/null 2>&1 && exit 0; sleep 1; done; exit 43",
        sudo=True, check=True, timeout=settings.startup_timeout_s + 10,
    )
