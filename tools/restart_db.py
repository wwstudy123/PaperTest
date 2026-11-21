from __future__ import annotations

from connectors.ssh_connector import SSHConnector
from config import MySQLSettings


def restart_db(
    connector: SSHConnector, service_name: str, *, mysql_settings: MySQLSettings | None = None,
) -> None:
    if mysql_settings is not None and mysql_settings.service_manager == "process":
        from tools.mysql_process import restart_mysql_process
        restart_mysql_process(connector, mysql_settings)
        return
    if mysql_settings is not None and mysql_settings.service_manager != "systemd":
        raise ValueError("未知 MySQL 服务管理方式")
    connector.run(f"systemctl restart {service_name}", sudo=True, check=True, timeout=120)
    status = connector.run(f"systemctl is-active {service_name}", sudo=True, check=False)
    if status.stdout.strip() != "active":
        raise RuntimeError(
            f"Database service {service_name} is not active after restart: {status.stdout.strip()} {status.stderr}"
        )
