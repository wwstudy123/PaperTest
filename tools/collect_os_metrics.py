from __future__ import annotations

from connectors.ssh_connector import SSHConnector


def collect_os_metrics(connector: SSHConnector) -> dict[str, object]:
    """只读 CPU、内存、磁盘与负载信息，不采集可调系统参数。"""
    cpu_result = connector.run("nproc", check=True)
    mem_result = connector.run(
        "free -b | awk '/Mem:/ {print $2\" \"$3\" \"$4}'",
        check=True,
    )
    disk_result = connector.run(
        "df -B1 / | tail -1 | awk '{print $2\" \"$3\" \"$4}'",
        check=True,
    )
    load_result = connector.run("cat /proc/loadavg", check=True)
    uptime_result = connector.run("uptime -p || uptime", check=False)

    mem_total, mem_used, mem_free = [int(part) for part in mem_result.stdout.strip().split()]
    disk_total, disk_used, disk_free = [int(part) for part in disk_result.stdout.strip().split()]
    load_parts = load_result.stdout.strip().split()
    return {
        "cpu_cores": int(cpu_result.stdout.strip()),
        "memory_total_bytes": mem_total,
        "memory_used_bytes": mem_used,
        "memory_free_bytes": mem_free,
        "disk_total_bytes": disk_total,
        "disk_used_bytes": disk_used,
        "disk_free_bytes": disk_free,
        "loadavg_1m": float(load_parts[0]),
        "loadavg_5m": float(load_parts[1]),
        "loadavg_15m": float(load_parts[2]),
        "uptime": uptime_result.stdout.strip(),
    }
