"""只读基线统一重置：完整逻辑校验+独立冷启动，不删除或复制旧数据目录。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from dotenv import dotenv_values
from admtune.continuous import atomic_json
from admtune.contracts import fingerprint
from admtune.dynamic import instance_lock
from admtune.security import redact


def logical_state(db):
    checksums, counts, samples = [], [], []
    with db.cursor() as q:
        for i in range(1, 5):
            table = f"admtune_workset_v2_full.sbtest{i}"
            q.execute(f"CHECKSUM TABLE {table} EXTENDED")
            row = q.fetchone()
            if row[1] is None:
                raise ValueError("Full table checksum unavailable")
            checksums.append([row[0], int(row[1])])
            q.execute(f"SELECT COUNT(*),MIN(id),MAX(id) FROM {table}")
            counts.append(list(q.fetchone()))
            q.execute(f"SELECT id,k,c,pad FROM {table} WHERE id IN (1,1000000,2000000) ORDER BY id")
            samples.append(q.fetchall())
    if counts != [[2000000, 1, 2000000]]*4:
        raise ValueError("Unexpected dataset size")
    return {"checksums": checksums, "counts": counts, "samples_sha256": fingerprint(samples)}


def main():
    import pymysql
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="结束后只核验数据和配置，不重启")
    a = parser.parse_args()
    if not a.dedicated_instance:
        parser.error("Dedicated instance required")
    base = Path("/root/autodl-tmp/admtune-mysql-v2")
    binary = Path("/root/autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin")
    cfg = base/"my.cnf"; initial = json.loads((base/"snapshots/initial/manifest.json").read_text())
    assert hashlib.sha256(cfg.read_bytes()).hexdigest() == initial["config_sha256"]
    a.audit.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    assert not a.audit.exists()
    password = dotenv_values(a.root/".env")["ADMTUNE_MYSQL_V2_PASSWORD"]
    def connect():
        return pymysql.connect(unix_socket=str(base/"mysql.sock"), user="root", password=password,
                               autocommit=True, read_timeout=180)
    lock = instance_lock(cfg)
    result = {"completed": False, "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "policy": "readonly_checksum_cold_restart_v1", "data_directory_replaced": False}
    try:
        if subprocess.run(["pgrep", "-x", "sysbench"], capture_output=True).returncode == 0:
            raise ValueError("Another benchmark is active")
        with connect() as db, db.cursor() as q:
            q.execute("SELECT @@port,@@datadir,@@innodb_buffer_pool_size,@@innodb_io_capacity,"
                      "@@innodb_io_capacity_max,@@sort_buffer_size,@@innodb_thread_concurrency")
            identity = list(q.fetchone())
            assert identity[0] == 3308 and Path(identity[1]).resolve() == (base/"data").resolve()
            assert identity[2:] == [1073741824,200,2000,262144,0]
            q.execute("SET GLOBAL innodb_thread_sleep_delay=10000")
            q.execute("SET GLOBAL innodb_adaptive_max_sleep_delay=150000")
            q.execute("SET GLOBAL innodb_concurrency_tickets=5000")
        with connect() as db:
            logical = logical_state(db)
        assert logical["samples_sha256"] == initial["logical_state"]["samples_sha256"]
        registry = a.root/"configs/local/formal-readonly-dataset.json"
        if registry.exists():
            assert json.loads(registry.read_text()) == logical, "Dataset differs from fixed first formal run"
        else:
            atomic_json(registry, logical)
        if a.verify_only:
            result.update(completed=True, verification_only=True, logical_checksums=logical["checksums"],
                          logical_samples_sha256=logical["samples_sha256"], row_counts=logical["counts"],
                          config_sha256=initial["config_sha256"], identity_before=identity, llm_calls=0)
            atomic_json(a.audit, result)
            return 0
        subprocess.run([str(binary/"mysqladmin"), "--protocol=socket", f"--socket={base}/mysql.sock",
                        "--user=root", "shutdown"], env={**os.environ,"MYSQL_PWD":password},
                       check=True, capture_output=True, timeout=120)
        for _ in range(200):
            if not (base/"mysql.pid").exists():
                break
            time.sleep(.1)
        assert not (base/"mysql.pid").exists(), "Never kill unknown server or remove PID files"
        subprocess.run([str(binary/"mysqld"), f"--defaults-file={cfg}",
                        "--innodb-buffer-pool-load-at-startup=OFF",
                        "--innodb-buffer-pool-dump-at-shutdown=OFF", "--daemonize"],
                       check=True, capture_output=True, timeout=120)
        with connect() as db, db.cursor() as q:
            q.execute("SELECT @@port,@@datadir,@@innodb_buffer_pool_load_at_startup,"
                      "@@innodb_buffer_pool_dump_at_shutdown")
            after = list(q.fetchone())
            assert after[0] == 3308 and after[2:] == [0,0]
            assert Path(after[1]).resolve() == (base/"data").resolve()
        result.update(completed=True, logical_checksums=logical["checksums"],
                      logical_samples_sha256=logical["samples_sha256"], row_counts=logical["counts"],
                      config_sha256=initial["config_sha256"], identity_before=identity, identity_after=after,
                      os_page_cache_cleared=False, llm_calls=0)
    except Exception as exc:
        result["error"] = redact(str(exc).replace(password,"<redacted>"))
    finally:
        lock.close()
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(a.audit, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
