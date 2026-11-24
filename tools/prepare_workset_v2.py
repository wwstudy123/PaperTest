"""隔离生成较大 Sysbench 工作集，冷备后实际复制恢复；不覆盖原实例。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import time

from dotenv import dotenv_values, set_key
import yaml

from admtune.contracts import fingerprint
from admtune.provenance import source_provenance
from admtune.security import redact


TABLES, TABLE_SIZE, PORT = 4, 2_000_000, 3308
DB = "admtune_workset_v2_full"


def prepare_command(binary_config):
    # Lua script options in sysbench 1.0.20 must be explicit here. Keeping only
    # C/driver options in the private file prevents silent Lua defaults.
    return ["sysbench", f"--config-file={binary_config}", "/usr/share/sysbench/oltp_read_write.lua",
            f"--tables={TABLES}", f"--table-size={TABLE_SIZE}", "--threads=4", "--rand-seed=42", "prepare"]


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_hashes(root):
    result = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("Snapshot refuses symlinks")
        if p.is_file():
            result[str(p.relative_to(root))] = file_hash(p)
    return result


def cold_copy(source, destination, *, server_stopped):
    if not server_stopped or destination.exists() or source.is_symlink():
        raise ValueError("Cold copies require a stopped server and unused destination")
    expected = tree_hashes(source)
    if not expected:
        raise ValueError("Empty snapshot source")
    shutil.copytree(source, destination)
    destination.chmod(0o700)
    if tree_hashes(destination) != expected:
        raise ValueError("Cold copy checksum mismatch")
    return expected


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    path.chmod(0o600)


def validate_identity(identity, datadir):
    if (str(identity[0]).split("-")[0] != "8.0.36" or int(identity[1]) != PORT
            or Path(identity[2]).resolve() != datadir.resolve()
            or str(identity[3]).upper() != "O_DIRECT"
            or int(identity[4]) != 1 or int(identity[5]) != 1
            or str(identity[6]).upper() not in {"1", "ON", "TRUE"}):
        raise ValueError("Version, identity or durability mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedicated-instance", action="store_true")
    parser.add_argument("--resume-failed-lua-options", action="store_true",
                        help="只修复首次 Lua 规模选项失效；保留原小库并创建新库")
    parser.add_argument("--root", type=Path, default=Path("/root/PaperTest"))
    parser.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/admtune-mysql-v2"))
    parser.add_argument("--binary", type=Path, default=Path(
        "/root/autodl-tmp/admtune-mysql/mysql-8.0.36-linux-glibc2.17-x86_64-minimal/bin"))
    args = parser.parse_args()
    if not args.dedicated_instance:
        parser.error("Explicit dedicated experimental instance authorization required")
    import pymysql
    base, root = args.base.resolve(), args.root.resolve()
    resume = args.resume_failed_lua_options
    if not args.base.is_absolute() or not args.root.is_absolute() or (base.exists() and not resume):
        parser.error("Absolute paths and an unused isolated base directory required")
    previous = None
    if resume:
        previous = json.loads((base / "preparation.json").read_text())
        if (previous.get("completed") or previous.get("error") != "Incomplete generated table"
                or previous.get("step") != "generate_sysbench_4x2000000"
                or previous.get("database") != "admtune_workset_v2"
                or (base / "preparation-attempt-001.json").exists()):
            parser.error("Only the preserved first Lua-options failure can be resumed")
    if shutil.disk_usage(base.parent).free < 20 * 1024**3:
        parser.error("At least 20 GiB free space required before preparation")
    envfile = root / ".env"
    if not envfile.is_file() or envfile.is_symlink():
        parser.error("Existing regular private environment file required")
    config_paths = [root / "configs/local" / f"workset-v2-{s}.yaml" for s in ("target", "benchmark", "knobs")]
    if any(p.exists() for p in config_paths):
        parser.error("Refusing to overwrite existing workset-v2 configurations")
    old = yaml.safe_load((root / "configs/local/autodl-smoke-target.yaml").read_text())
    old_config = Path(old["mysql"]["config_path"])
    old_config_hash = file_hash(old_config)
    if not resume:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", PORT))
    os.umask(0o077)
    if not resume:
        base.mkdir(mode=0o700)
        for name in ("data", "binlog", "logs", "files"):
            (base / name).mkdir(mode=0o700)
    existing_password = dotenv_values(envfile).get("ADMTUNE_MYSQL_V2_PASSWORD")
    if resume and not existing_password:
        parser.error("Existing isolated instance credential required")
    password = existing_password or secrets.token_urlsafe(32)
    state = {"completed": False, "llm_calls": 0, "port": PORT, "database": DB,
             "tables": TABLES, "table_size": TABLE_SIZE, **source_provenance()}
    if resume:
        shutil.copy2(base / "preparation.json", base / "preparation-attempt-001.json")
        for name in ("sysbench-private.conf", "logs/prepare.log"):
            path = base / name
            if path.exists():
                shutil.copy2(path, path.with_name(path.stem + "-attempt-001" + path.suffix))
        state["retained_failed_attempt_sha256"] = file_hash(base / "preparation-attempt-001.json")
        state["retained_partial_database"] = previous["database"]
    def step(name):
        state["step"] = name
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(base / "progress.json", state)
    def run(command, *, timeout=120, env=None):
        r = subprocess.run(command, text=True, capture_output=True, timeout=timeout, env=env)
        if r.returncode:
            raise RuntimeError(redact((r.stderr or r.stdout).replace(password, "<redacted>"))[-3000:])
        return r.stdout
    config = base / "my.cnf"
    config_text = f"""[mysqld]
user=root
basedir={args.binary.parent}
datadir={base}/data
socket={base}/mysql.sock
pid-file={base}/mysql.pid
port={PORT}
bind-address=127.0.0.1
mysqlx=0
log-error={base}/logs/mysql-error.log
secure-file-priv={base}/files
innodb_buffer_pool_size=1073741824
innodb_buffer_pool_instances=1
innodb_buffer_pool_chunk_size=134217728
innodb_redo_log_capacity=268435456
innodb_flush_method=O_DIRECT
innodb_flush_log_at_trx_commit=1
sync_binlog=1
innodb_doublewrite=ON
performance_schema=ON
max_connections=100
sort_buffer_size=262144
innodb_io_capacity=200
innodb_io_capacity_max=2000
log_bin={base}/binlog/mysql-bin
max_binlog_size=134217728
binlog_expire_logs_seconds=600
character-set-server=utf8mb4
collation-server=utf8mb4_0900_ai_ci
"""
    if resume:
        if config.read_text() != config_text:
            parser.error("Existing isolated instance configuration differs; refusing resume")
    else:
        config.write_text(config_text)
    config.chmod(0o600)
    def connect(*, blank=False):
        return pymysql.connect(unix_socket=str(base / "mysql.sock"), user="root",
                               password="" if blank else password, autocommit=True,
                               connect_timeout=5, read_timeout=120)
    def start():
        run([str(args.binary / "mysqld"), f"--defaults-file={config}", "--daemonize"])
        for _ in range(40):
            try:
                c = connect();c.close();return
            except pymysql.Error:
                time.sleep(.5)
        raise TimeoutError("Restored isolated MySQL did not become ready")
    def stop():
        run([str(args.binary / "mysqladmin"), "--protocol=socket", f"--socket={base}/mysql.sock",
             "--user=root", "shutdown"], env={**os.environ, "MYSQL_PWD": password})
        for _ in range(100):
            if not (base / "mysql.pid").exists():
                return
            time.sleep(.1)
        if (base / "mysql.pid").exists():
            raise RuntimeError("MySQL PID file persists; do not copy or restore")
    def logical_state():
        with connect() as c, c.cursor() as q:
            q.execute("SELECT VERSION(),@@port,@@datadir,@@innodb_flush_method,"
                      "@@innodb_flush_log_at_trx_commit,@@sync_binlog,@@innodb_doublewrite")
            identity = list(q.fetchone())
            validate_identity(identity, base / "data")
            rows, samples = [], []
            for i in range(1, TABLES + 1):
                q.execute(f"SELECT COUNT(*),MIN(id),MAX(id) FROM {DB}.sbtest{i}")
                found = list(q.fetchone())
                if found != [TABLE_SIZE, 1, TABLE_SIZE]:
                    raise ValueError("Incomplete generated table")
                rows.append(found)
                q.execute(f"SELECT id,k,c,pad FROM {DB}.sbtest{i} WHERE id IN (1,1000000,2000000) ORDER BY id")
                samples.append(q.fetchall())
            q.execute("SELECT TABLE_NAME,COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_KEY "
                      "FROM information_schema.columns WHERE table_schema=%s ORDER BY TABLE_NAME,ORDINAL_POSITION", (DB,))
            columns = q.fetchall()
            q.execute("SELECT TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX,COLUMN_NAME,NON_UNIQUE "
                      "FROM information_schema.statistics WHERE table_schema=%s ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX", (DB,))
            indexes = q.fetchall()
            q.execute("SELECT TABLE_NAME,DATA_LENGTH,INDEX_LENGTH FROM information_schema.tables "
                      "WHERE table_schema=%s ORDER BY TABLE_NAME", (DB,))
            sizes = q.fetchall()
            return {"identity": identity, "row_checks": rows, "samples_sha256": fingerprint(samples),
                    "schema_sha256": fingerprint({"columns": columns, "indexes": indexes}),
                    "table_statistics_estimates": sizes,
                    "estimated_data_index_bytes": sum(int(r[1]) + int(r[2]) for r in sizes)}
    try:
        if not resume:
            step("initialize_isolated_instance")
            run([str(args.binary / "mysqld"), f"--defaults-file={config}", "--initialize-insecure"])
            run([str(args.binary / "mysqld"), f"--defaults-file={config}", "--daemonize"])
            with connect(blank=True) as c, c.cursor() as q:
                q.execute("ALTER USER 'root'@'localhost' IDENTIFIED BY %s", (password,))
                q.execute(f"CREATE DATABASE {DB} CHARACTER SET utf8mb4")
        else:
            step("resume_into_new_database_preserving_partial")
            with connect() as c, c.cursor() as q:
                q.execute("SELECT VERSION(),@@port,@@datadir,@@innodb_flush_method,"
                          "@@innodb_flush_log_at_trx_commit,@@sync_binlog,@@innodb_doublewrite")
                validate_identity(q.fetchone(), base / "data")
                q.execute("SELECT COUNT(*),MIN(id),MAX(id) FROM admtune_workset_v2.sbtest1")
                if tuple(q.fetchone()) != (10000, 1, 10000):
                    raise ValueError("Partial dataset differs from the captured failure")
                q.execute(f"CREATE DATABASE {DB} CHARACTER SET utf8mb4")
        envfile.chmod(0o600)
        set_key(str(envfile), "ADMTUNE_MYSQL_V2_PASSWORD", password)
        envfile.chmod(0o600)
        private = base / "sysbench-private.conf"
        private.write_text(f"""db-driver=mysql
mysql-host=127.0.0.1
mysql-port={PORT}
mysql-user=root
mysql-password={password}
mysql-db={DB}
tables={TABLES}
table-size={TABLE_SIZE}
threads=4
rand-seed=42
""")
        private.chmod(0o600)
        step("generate_sysbench_4x2000000")
        generated = run(prepare_command(private), timeout=1200)
        (base / "logs/prepare.log").write_text(generated.replace(password, "<redacted>"))
        before = logical_state()
        step("clean_shutdown_and_snapshot")
        stop()
        size = sum(p.stat().st_size for n in ("data", "binlog") for p in (base / n).rglob("*") if p.is_file())
        if shutil.disk_usage(base).free < 2 * size + 2 * 1024**3:
            raise RuntimeError("Insufficient space for immutable snapshot and retained pre-restore copy")
        snapshot = base / "snapshots/initial"
        snapshot.mkdir(parents=True, mode=0o700)
        hashes = {n: cold_copy(base / n, snapshot / n, server_stopped=True) for n in ("data", "binlog")}
        snapshot_info = {"clean_shutdown": True, "config_sha256": file_hash(config), "logical_state": before,
                         "file_hashes": hashes, "source": "sysbench prepare; seed42; 4 threads"}
        write_json(snapshot / "manifest.json", snapshot_info)
        step("restore_rehearsal_preserving_original_directories")
        retained = base / "retained-pre-restore"
        retained.mkdir(mode=0o700)
        for name in ("data", "binlog"):
            (base / name).rename(retained / name)
            if cold_copy(snapshot / name, base / name, server_stopped=True) != hashes[name]:
                raise ValueError("Restore bytes do not match original cold snapshot")
        start()
        after = logical_state()
        if (before["row_checks"] != after["row_checks"] or before["samples_sha256"] != after["samples_sha256"]
                or before["schema_sha256"] != after["schema_sha256"]):
            raise ValueError("Restored logical state mismatch")
        target = {**old, "mysql": {**old["mysql"], "config_path": str(config), "port": PORT,
                  "mysql_password_env": "ADMTUNE_MYSQL_V2_PASSWORD", "database": DB,
                  "service_name": "admtune-mysql-v2", "socket_path": str(base / "mysql.sock"),
                  "pid_file": str(base / "mysql.pid"), "error_log_path": str(base / "logs/mysql-error.log")},
                  "admtune": {**old["admtune"], "hardware_id": socket.gethostname(),
                  "data_id": "workset-v2-4x2000000-" + file_hash(snapshot / "manifest.json")[:16],
                  "schema_id": after["schema_sha256"]}}
        benchmark = yaml.safe_load((root / "configs/local/autodl-smoke-benchmark.yaml").read_text())
        benchmark["workload"].update(database=DB, tables=TABLES, table_size=TABLE_SIZE)
        knobs = yaml.safe_load((root / "configs/local/autodl-smoke-knobs.yaml").read_text())
        knobs["innodb_buffer_pool_size"].update(default=1024**3, min=512 * 1024**2,
                                                max=4 * 1024**3, step=128 * 1024**2)
        for path, payload in zip(config_paths, (target, benchmark, knobs)):
            if path.exists():
                raise ValueError("Output config appeared during preparation; refusing overwrite")
            path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False))
            path.chmod(0o600)
        if file_hash(old_config) != old_config_hash:
            raise ValueError("Original smoke instance configuration changed during preparation")
        state.update(completed=True, step="ready", snapshot_restore_verified=True,
                     snapshot_manifest_sha256=file_hash(snapshot / "manifest.json"),
                     original_smoke_config_sha256=old_config_hash,
                     original_smoke_instance_untouched=True, logical_state=after,
                     cold_snapshot_bytes=size)
    except Exception as exc:
        state.update(error=redact(str(exc).replace(password, "<redacted>")), completed=False)
    write_json(base / "preparation.json", state)
    write_json(base / "progress.json", state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if state["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
