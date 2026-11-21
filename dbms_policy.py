"""Deterministic DBMS-only boundary, independent of prompts and LLM output."""

PROTECTED_KNOBS = frozenset({
    "innodb_flush_log_at_trx_commit", "sync_binlog", "innodb_doublewrite",
    "skip_innodb_doublewrite", "skip_log_bin", "log_bin", "binlog_format",
    "transaction_isolation", "transaction_read_only", "read_only", "super_read_only",
    "performance_schema", "sql_mode", "foreign_key_checks", "unique_checks",
    "autocommit", "general_log", "slow_query_log",
    "fsync", "full_page_writes", "synchronous_commit", "wal_level",
    "track_counts", "track_activities", "ssl", "autovacuum",
})


def enforce_dbms_only(action: str, candidate: dict) -> None:
    if action not in {"db_config", "repeat_benchmark"}:
        raise ValueError(f"DBMS-only policy rejects action {action!r}")
    if action == "repeat_benchmark" and candidate:
        raise ValueError("repeat_benchmark cannot carry configuration changes")
    for key in candidate:
        name = key.lower().replace("-", "_")
        if name in PROTECTED_KNOBS or "." in name or "/" in name:
            raise ValueError(f"Protected or non-database parameter: {key}")
