"""Guarded MySQL adapter reusing AgenticDB's connectors, apply and benchmark tools."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

from config import AppConfig, KnobSpec, MySQLSettings
from connectors.mysql_connector import MySQLConnector
from connectors.ssh_connector import SSHConnector
from dbms_policy import PROTECTED_KNOBS, enforce_dbms_only
from tools.apply_db_config import apply_db_config, _format_mysql_sql_value
from tools.restart_db import restart_db
from tools.rollback import create_config_backup
from tools.run_benchmark import run_benchmark
from validators.result_validator import ResultValidator

from .contracts import Objective, fingerprint
from .execution import Measurement, RecoveryFailed
from .space import Space
from .security import redact


ESSENTIAL_INVARIANTS = {
    "innodb_flush_log_at_trx_commit", "sync_binlog", "innodb_doublewrite",
    "transaction_isolation", "performance_schema",
}
RESOURCE_KNOBS = {
    "max_connections", "innodb_buffer_pool_size", "key_buffer_size",
    "sort_buffer_size", "join_buffer_size", "read_buffer_size", "read_rnd_buffer_size",
}


@dataclass
class Snapshot:
    config: dict
    file_content: str
    invariants: dict


class AgenticDBExecutor:
    def __init__(
        self, app: AppConfig, space: Space, objective: Objective, run_dir: Path,
        *, allow_mutations: bool = False, allow_restart: bool = False,
        max_p95_ms: float = 1000,
        connector: SSHConnector | None = None, db: MySQLConnector | None = None,
    ):
        if not isinstance(app.target.active_db_settings, MySQLSettings):
            raise ValueError("Initial real adapter supports MySQL only")
        if app.benchmark.workload.kind != "sysbench":
            raise ValueError("Complete OLAP measurement needs a separate verified adapter")
        if app.benchmark.objective.primary_metric != objective.name or objective.name != "tps":
            raise ValueError("Initial real pilot supports a pure TPS objective only")
        if app.benchmark.objective.formula.lower() not in {"tps", "identity", "throughput"}:
            raise ValueError("Do not mix TPS and TPS/P95 objectives")
        if not math.isfinite(max_p95_ms) or max_p95_ms <= 0:
            raise ValueError("A positive explicit P95 runtime constraint is required")
        self.app, self.space, self.objective, self.run_dir = app, space, objective, run_dir
        self.settings = app.target.active_db_settings
        if not re.fullmatch(r"[A-Za-z0-9_.@-]+", self.settings.service_name):
            raise ValueError("Unsafe service name")
        if not Path(self.settings.config_path).is_absolute():
            raise ValueError("An explicit absolute MySQL config path is required")
        self.allow_mutations, self.allow_restart, self.max_p95_ms = allow_mutations, allow_restart, max_p95_ms
        self.simulated = app.dry_run
        self.connector = connector or SSHConnector(
            app.target.ssh, config_path=self.settings.config_path,
            dbms="mysql", dry_run=app.dry_run,
        )
        self.db = db or MySQLConnector(self.connector, self.settings, dry_run=app.dry_run)
        self.specs = {
            name: KnobSpec(
                name, "string" if k.kind == "enum" else k.kind,
                unit=k.unit, min=k.lower, max=k.upper,
                allowed_values=list(k.choices) or None, restart_required=k.restart_required,
            )
            for name, k in space.knobs.items()
        }
        duration = app.benchmark.workload.duration + app.benchmark.workload.warmup_time
        self.estimated_trial_seconds = duration + (180 if allow_restart else 30)
        self.initial: Snapshot | None = None
        self.pending: Snapshot | None = None
        self.counter = 0

    def close(self):
        self.connector.close()

    def diagnose(self) -> dict:
        workload = self.app.benchmark.workload
        return {
            "kind": workload.kind, "mode": workload.mode, "tables": workload.tables,
            "threads": workload.threads,
            "runtime_metrics": self.db.show_status([
                "Threads_running", "Innodb_buffer_pool_reads", "Innodb_buffer_pool_read_requests",
                "Created_tmp_disk_tables", "Innodb_row_lock_time", "Innodb_data_reads", "Innodb_data_writes",
            ]),
            "usage": "Diagnostic prompt context only; not external workload features or causal proof.",
        }

    def readback(self) -> dict:
        raw = self.db.show_variables(list(self.space.names))
        normalized = {}
        for name in self.space.names:
            value = raw.get(name)
            knob = self.space.knobs[name]
            if value is None:
                raise ValueError(f"MySQL parameter unavailable: {name}")
            if knob.kind == "integer":
                value = int(value) if not isinstance(value, str) or re.fullmatch(r"-?\d+", value) else value
            elif knob.kind == "float":
                value = float(value)
            elif knob.kind == "boolean":
                text = str(value).upper()
                if text not in {"ON", "OFF", "1", "0", "TRUE", "FALSE"}:
                    raise ValueError(f"Invalid boolean readback for {name}")
                value = text in {"ON", "1", "TRUE"}
            normalized[name] = knob.validate(value)
        return self.space.merge(normalized, {})

    def invariants(self) -> dict:
        names = sorted(
            (PROTECTED_KNOBS - {n for n in PROTECTED_KNOBS if n.startswith("skip_")})
            | set(self.space.fixed_resources),
        )
        raw = self.db.show_variables(names)
        if any(raw.get(n) is None for n in ESSENTIAL_INVARIANTS):
            raise ValueError("Cannot verify required MySQL durability/isolation invariants")
        return {n: str(v).upper() for n, v in raw.items() if v is not None}

    def snapshot(self) -> Snapshot:
        if not self.connector.exists(self.settings.config_path, sudo=True):
            raise ValueError("Configured MySQL configuration file does not exist")
        return Snapshot(
            self.readback(), self.connector.read_file(self.settings.config_path, sudo=True),
            self.invariants(),
        )

    def preflight(self) -> dict:
        self.connector.connect()
        if not self.db.is_alive():
            raise ValueError("MySQL health check failed")
        version = self.db.get_version()
        if self.app.target.db_version and version.split("-")[0] != self.app.target.db_version:
            raise ValueError(f"Database version mismatch: expected {self.app.target.db_version}, got {version}")
        resources = self.db.show_variables(sorted(RESOURCE_KNOBS))
        if not self.simulated and any(resources.get(n) is None for n in RESOURCE_KNOBS):
            raise ValueError("Cannot validate fixed per-connection/global memory resource context")
        self.space.fixed_resources = {
            n: int(v) for n, v in resources.items() if v is not None and n not in self.space.names
        }
        self.initial = self.snapshot()
        workload = self.app.benchmark.workload
        if min(workload.duration, workload.threads, workload.tables, workload.table_size) <= 0:
            raise ValueError("Explicit positive duration, threads, table count and size required")
        readiness = {}
        if not self.simulated:
            version_result = self.connector.run("sysbench --version", check=True)
            rows = self.db.query_rows(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = DATABASE();"
            )
            if not rows or int(rows[0][0]) < workload.tables:
                raise ValueError("Database has fewer tables than configured; do not auto-create data")
            schema = self.db.query_rows(
                "SELECT TABLE_NAME,COLUMN_NAME,COLUMN_TYPE,IS_NULLABLE,COLUMN_KEY "
                "FROM information_schema.columns WHERE table_schema=DATABASE() "
                "ORDER BY TABLE_NAME,ORDINAL_POSITION;"
            )
            indexes = self.db.query_rows(
                "SELECT TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX,COLUMN_NAME,NON_UNIQUE "
                "FROM information_schema.statistics WHERE table_schema=DATABASE() "
                "ORDER BY TABLE_NAME,INDEX_NAME,SEQ_IN_INDEX;"
            )
            readiness = {
                "sysbench_version": version_result.stdout.strip(), "table_count": int(rows[0][0]),
                "live_schema_hash": fingerprint({"columns": schema, "indexes": indexes}),
                "server_identity": self.connector.run("hostname; uname -srm; nproc", check=True).stdout.strip(),
                "datadir": self.db.query_rows("SELECT @@datadir;"),
            }
        return {
            "db_version": version, "current_config": self.initial.config,
            "invariants": self.initial.invariants, "space_key": self.space.key,
            "config_sha256": hashlib.sha256(self.initial.file_content.encode()).hexdigest(),
            "simulated": self.simulated, "mutations_enabled": self.allow_mutations,
            "restart_enabled": self.allow_restart,
            "readiness": readiness,
            "note": "Preflight reads DB state; upstream sudo read_file may create temporary read copies.",
        }

    def _verify(self, config: dict, invariants: dict):
        if not self.db.is_alive() or self.readback() != config:
            raise ValueError("Service unhealthy or parameter readback mismatch")
        if self.invariants() != invariants:
            raise ValueError("Durability, semantics or observability invariant changed")

    def _restore_snapshot(self, snapshot: Snapshot):
        try:
            self.connector.write_file(self.settings.config_path, snapshot.file_content, sudo=True)
            needs_restart = any(k.restart_required for k in self.space.knobs.values()) or not self.db.is_alive()
            if needs_restart:
                if not self.allow_restart:
                    raise RecoveryFailed("Recovery needs a service restart, but restart permission was not granted")
                restart_db(self.connector, self.settings.service_name)
            for name, value in snapshot.config.items():
                if not self.space.knobs[name].restart_required:
                    self.db.execute(f"SET GLOBAL {name} = {_format_mysql_sql_value(value)};")
            self._verify(snapshot.config, snapshot.invariants)
            restored = self.connector.read_file(self.settings.config_path, sudo=True)
            if restored != snapshot.file_content:
                raise ValueError("Configuration file restoration mismatch")
        except Exception as exc:
            raise RecoveryFailed(redact(f"Snapshot restoration failed: {exc}")) from exc

    @staticmethod
    def complete_sysbench(output: str, expected_seconds: int) -> bool:
        duration = re.search(r"total time:\s*([\d.]+)s", output, re.IGNORECASE)
        transactions = re.search(r"transactions:\s+(\d+)\s+\(([\d.]+)\s+per sec", output, re.IGNORECASE)
        errors = re.search(r"ignored errors:\s*(\d+)", output, re.IGNORECASE)
        return bool(
            duration and transactions and float(duration.group(1)) >= expected_seconds * 0.95
            and int(transactions.group(1)) > 0 and (errors is None or int(errors.group(1)) == 0)
        )

    def measure(self, config: dict, context: list[float]) -> Measurement:
        if not self.allow_mutations:
            raise PermissionError("Explicit dedicated-instance confirmation is required")
        if self.initial is None:
            raise ValueError("Run preflight before measurement")
        config = self.space.merge(config, {})
        before = self.snapshot()
        delta = {n: v for n, v in config.items() if before.config[n] != v}
        enforce_dbms_only("db_config", delta)
        if any(self.space.knobs[n].restart_required for n in delta) and not self.allow_restart:
            raise PermissionError("Restart-required knobs need explicit restart permission")
        self.pending = before
        self.counter += 1
        self.run_dir.mkdir(parents=True, exist_ok=True)
        create_config_backup(
            self.connector, config_path=self.settings.config_path,
            round_id=self.counter, run_dir=self.run_dir,
        )
        try:
            if delta:
                apply_db_config(
                    self.connector, self.db, config_path=self.settings.config_path,
                    candidate_config=delta, knob_specs=self.specs,
                )
                if any(self.space.knobs[n].restart_required for n in delta):
                    restart_db(self.connector, self.settings.service_name)
            self._verify(config, before.invariants)
            result = run_benchmark(self.connector, self.app.benchmark, self.settings, dry_run=self.app.dry_run)
            (self.run_dir / f"benchmark-{self.counter}.log").write_text(result.raw_output)
            if not ResultValidator().validate(result).passed:
                raise ValueError("Invalid numerical benchmark result")
            if not self.complete_sysbench(result.raw_output, self.app.benchmark.workload.duration):
                raise ValueError("Incomplete Sysbench output, ignored errors, or insufficient measured duration")
            if result.tps is None or result.p95_latency_ms is None:
                raise ValueError("Missing TPS or P95")
            self._verify(config, before.invariants)
            return Measurement(
                result.tps, constraints_ok=result.p95_latency_ms <= self.max_p95_ms,
                metadata={"simulated": self.simulated, "p95_latency_ms": result.p95_latency_ms,
                          "benchmark_log": f"benchmark-{self.counter}.log"},
            )
        except BaseException as exc:
            self._restore_snapshot(before)
            if not isinstance(exc, Exception):
                raise
            return Measurement(None, "execution_failed", False, False, False, {"reason": redact(str(exc))})

    def restore(self, config: dict):
        if not self.allow_mutations:
            raise PermissionError("Mutation permission required for restore")
        if self.pending is None or self.pending.config != config:
            raise RecoveryFailed("Restore target does not match saved pre-trial snapshot")
        self._restore_snapshot(self.pending)

    def restore_initial(self):
        if self.initial is not None and self.allow_mutations:
            self._restore_snapshot(self.initial)
