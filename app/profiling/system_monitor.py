"""
Host CPU, RAM, swap, and Ollama process monitor using psutil.
"""

from datetime import datetime, timezone
from typing import List, Optional
import psutil
from app.core.config import logger
from app.schemas.request import OllamaProcessInfo, SystemResources

BYTES_PER_MIB = float(1024 * 1024)


class SystemMonitor:
    """Collects host system telemetry and inspects active Ollama processes."""

    @staticmethod
    def get_system_resources() -> SystemResources:
        """Snapshots CPU, memory, swap, and running Ollama processes."""
        now = datetime.now(timezone.utc)

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        cpu_pct = float(psutil.cpu_percent(interval=0.05))

        load_avg: Optional[List[float]] = None
        if hasattr(psutil, "getloadavg"):
            try:
                raw_load = psutil.getloadavg()
                load_avg = [round(float(x), 2) for x in raw_load]
            except OSError:
                load_avg = None

        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        ollama_procs: List[OllamaProcessInfo] = []
        for proc in psutil.process_iter(attrs=["pid", "name", "status", "memory_info", "cpu_percent"]):
            try:
                pinfo = proc.info
                pname = (pinfo.get("name") or "").lower()
                if "ollama" in pname or "llama_server" in pname:
                    mem_info = pinfo.get("memory_info")
                    rss_b = int(mem_info.rss) if mem_info else 0
                    ollama_procs.append(
                        OllamaProcessInfo(
                            pid=int(pinfo["pid"]),
                            name=str(pinfo.get("name") or "unknown"),
                            rss_bytes=rss_b,
                            rss_mib=round(rss_b / BYTES_PER_MIB, 2),
                            cpu_percent=float(pinfo.get("cpu_percent") or 0.0),
                            status=str(pinfo.get("status") or "unknown"),
                        )
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as exc:
                logger.debug("Skipped transient process during psutil scan: %s", exc)

        return SystemResources(
            cpu_physical_cores=physical_cores,
            cpu_logical_cores=logical_cores,
            cpu_usage_pct=cpu_pct,
            load_avg_1m_5m_15m=load_avg,
            ram_total_bytes=int(vm.total),
            ram_available_bytes=int(vm.available),
            ram_used_bytes=int(vm.used),
            ram_total_mib=round(vm.total / BYTES_PER_MIB, 2),
            ram_available_mib=round(vm.available / BYTES_PER_MIB, 2),
            ram_used_mib=round(vm.used / BYTES_PER_MIB, 2),
            ram_usage_pct=float(vm.percent),
            swap_total_mib=round(sm.total / BYTES_PER_MIB, 2),
            swap_used_mib=round(sm.used / BYTES_PER_MIB, 2),
            swap_usage_pct=float(sm.percent),
            ollama_processes=ollama_procs,
            timestamp_utc=now,
        )


system_monitor = SystemMonitor()