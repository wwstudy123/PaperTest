"""独立运行前恢复 v2 冷备；保留被替换目录，不覆盖原实例或快照。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from dotenv import dotenv_values

from admtune.contracts import fingerprint
from admtune.dynamic import instance_lock
from admtune.security import redact
from tools.prepare_workset_v2 import cold_copy, file_hash, tree_hashes, validate_identity, DB, TABLES, TABLE_SIZE


def restore(root, base, audit_dir):
    import pymysql
    root, base, audit_dir = root.resolve(), base.resolve(), audit_dir.resolve()
    audit_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    config = base / "my.cnf"
    lock = instance_lock(config)
    state = {"completed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
             "base": str(base), "database": DB, "llm_calls": 0}
    password = dotenv_values(root / ".env")["ADMTUNE_MYSQL_V2_PASSWORD"]
    binary = root.parent / "autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin"
    snapshot = base / "snapshots/initial"
    try:
        prep = json.loads((base / "preparation.json").read_text())
        manifest = json.loads((snapshot / "manifest.json").read_text())
        if not prep.get("completed") or file_hash(snapshot / "manifest.json") != prep["snapshot_manifest_sha256"]:
            raise ValueError("Completed preparation and unchanged snapshot manifest required")
        if file_hash(config) != manifest["config_sha256"]:
            raise ValueError("Restore initial configuration before resetting data")
        for name in ("data", "binlog"):
            if tree_hashes(snapshot / name) != manifest["file_hashes"][name]:
                raise ValueError("Immutable snapshot checksum mismatch")
        size = sum(p.stat().st_size for name in ("data", "binlog")
                   for p in (snapshot / name).rglob("*") if p.is_file())
        import shutil
        if shutil.disk_usage(base).free < size + 2 * 1024**3:
            raise ValueError("Insufficient space; retained data will not be deleted automatically")

        def connect():
            return pymysql.connect(unix_socket=str(base / "mysql.sock"), user="root",
                                   password=password, autocommit=True, read_timeout=120)

        def identity(q):
            q.execute("SELECT VERSION(),@@port,@@datadir,@@innodb_flush_method,"
                      "@@innodb_flush_log_at_trx_commit,@@sync_binlog,@@innodb_doublewrite")
            value = list(q.fetchone())
            validate_identity(value, base / "data")
            return value

        with connect() as c, c.cursor() as q:
            state["identity_before"] = identity(q)
        subprocess.run([str(binary / "mysqladmin"), "--protocol=socket",
                        f"--socket={base}/mysql.sock", "--user=root", "shutdown"],
                       env={**os.environ, "MYSQL_PWD": password}, check=True, capture_output=True, timeout=120)
        for _ in range(200):
            if not (base / "mysql.pid").exists():
                break
            time.sleep(.1)
        if (base / "mysql.pid").exists():
            raise ValueError("PID persists after shutdown; refusing data replacement")
        retained = base / "retained-runs" / audit_dir.name
        retained.mkdir(mode=0o700, parents=True, exist_ok=False)
        state["retained_previous_data"] = str(retained)
        for name in ("data", "binlog"):
            (base / name).rename(retained / name)
            if cold_copy(snapshot / name, base / name, server_stopped=True) != manifest["file_hashes"][name]:
                raise ValueError("Restored bytes differ from snapshot")
        subprocess.run([str(binary / "mysqld"), f"--defaults-file={config}", "--daemonize"],
                       check=True, capture_output=True, timeout=120)
        with connect() as c, c.cursor() as q:
            state["identity_after"] = identity(q)
            rows, samples = [], []
            for i in range(1, TABLES + 1):
                q.execute(f"SELECT COUNT(*),MIN(id),MAX(id) FROM {DB}.sbtest{i}")
                rows.append(list(q.fetchone()))
                q.execute(f"SELECT id,k,c,pad FROM {DB}.sbtest{i} "
                          "WHERE id IN (1,1000000,2000000) ORDER BY id")
                samples.append(q.fetchall())
            if rows != [[TABLE_SIZE, 1, TABLE_SIZE]] * TABLES:
                raise ValueError("Restored row counts mismatch")
            if fingerprint(samples) != manifest["logical_state"]["samples_sha256"]:
                raise ValueError("Restored logical samples mismatch")
            state.update(row_checks=rows, samples_sha256=fingerprint(samples))
            # 每个方法从相同的重建统计开始；不与测试阶段反馈混合。
            q.execute("SET SESSION information_schema_stats_expiry=0")
            for i in range(1, TABLES + 1):
                q.execute(f"ANALYZE TABLE {DB}.sbtest{i}")
                if any(str(row[-1]).upper() != "OK" for row in q.fetchall()):
                    raise ValueError("ANALYZE did not succeed")
        state.update(completed=True, config_sha256=file_hash(config),
                     snapshot_manifest_sha256=file_hash(snapshot / "manifest.json"),
                     restored_data_bytes=size)
    except Exception as exc:
        state["error"] = redact(str(exc).replace(password, "<redacted>"))
        # 未验证的半恢复绝不自动重试；旧目录和冷备均保留，便于人工恢复。
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        path = audit_dir / "restore.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2))
        path.chmod(0o600)
        lock.close()
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/PaperTest"))
    parser.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/admtune-mysql-v2"))
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--dedicated-instance", action="store_true")
    args = parser.parse_args()
    if not args.dedicated_instance or any(not p.is_absolute() for p in (args.root, args.base, args.audit_dir)):
        parser.error("Explicit dedicated instance and absolute paths required")
    result = restore(args.root, args.base, args.audit_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
