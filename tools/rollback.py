from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from connectors.ssh_connector import SSHConnector
from config import MySQLSettings
from models import BackupRecord
from tools.restart_db import restart_db


def create_config_backup(connector: SSHConnector, *, config_path: str, round_id: int, run_dir: Path) -> BackupRecord:
    content = connector.read_file(config_path, sudo=True) if connector.exists(config_path, sudo=True) else ""
    backup_path = f"/tmp/agenticdb_round_{round_id}_{int(datetime.now().timestamp())}.cnf.bak"
    connector.write_file(backup_path, content)
    local_backup_path = run_dir / f"backup_round_{round_id}.cnf"
    local_backup_path.write_text(content, encoding="utf-8")
    return BackupRecord(
        config_path=config_path,
        backup_path=backup_path,
        created_at=datetime.now().isoformat(timespec="seconds"),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        metadata={"local_backup_path": str(local_backup_path)},
    )


def rollback_db_config(
    connector: SSHConnector,
    backup: BackupRecord,
    *,
    service_name: str,
    restart_required: bool,
    mysql_settings: MySQLSettings | None = None,
) -> None:
    content = connector.read_file(backup.backup_path)
    connector.write_file(backup.config_path, content, sudo=True)
    if restart_required:
        restart_db(connector, service_name, mysql_settings=mysql_settings)
