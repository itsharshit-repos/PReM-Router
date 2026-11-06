"""
NVIDIA GPU telemetry monitor backed by NVML (pynvml / nvidia-ml-py).
Strictly reports real hardware metrics and degrades gracefully without fabricating values.
"""

from datetime import datetime, timezone
from typing import Optional
from app.core.config import logger, settings
from app.schemas.request import GPUTelemetry

try:
    import pynvml
    PYNVML_IMPORTED = True
except ImportError:
    pynvml = None  # type: ignore
    PYNVML_IMPORTED = False

BYTES_PER_MIB = float(1024 * 1024)


class GPUMonitor:
    """Manages NVML lifecycle and queries GPU telemetry for a target device index."""

    def __init__(self, device_index: int = settings.gpu_device_index) -> None:
        self.device_index = device_index
        self._initialized = False
        self._init_error: Optional[str] = None

    def initialize(self) -> bool:
        """Initializes NVML library. Safe to call multiple times."""
        if not PYNVML_IMPORTED or pynvml is None:
            self._initialized = False
            self._init_error = "pynvml (nvidia-ml-py) package is not installed."
            logger.warning("NVML unavailable: %s", self._init_error)
            return False

        try:
            pynvml.nvmlInit()
            self._initialized = True
            self._init_error = None
            logger.info("NVML initialized successfully for GPU monitoring.")
            return True
        except pynvml.NVMLError as exc:
            self._initialized = False
            self._init_error = f"NVML initialization failed: {str(exc)}"
            logger.warning("NVML initialization error: %s", self._init_error)
            return False
        except Exception as exc:  # pragma: no cover
            self._initialized = False
            self._init_error = f"Unexpected error initializing NVML: {str(exc)}"
            logger.error("Unexpected NVML error: %s", self._init_error)
            return False

    def shutdown(self) -> None:
        """Cleanly shuts down NVML if initialized."""
        if self._initialized and PYNVML_IMPORTED and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
                logger.info("NVML shut down cleanly.")
            except Exception as exc:
                logger.debug("Error during NVML shutdown: %s", exc)
            finally:
                self._initialized = False

    def get_telemetry(self) -> GPUTelemetry:
        """
        Queries current VRAM, utilization, and thermal metrics from NVML.
        Never fabricates numbers if NVML or the GPU is inaccessible.
        """
        now = datetime.now(timezone.utc)

        if not self._initialized:
            if not self.initialize():
                return GPUTelemetry(
                    available=False,
                    device_index=self.device_index,
                    error=self._init_error or "NVML not initialized",
                    timestamp_utc=now,
                )

        try:
            device_count = pynvml.nvmlDeviceGetCount()
            if self.device_index >= device_count:
                return GPUTelemetry(
                    available=False,
                    device_index=self.device_index,
                    error=f"Requested GPU index {self.device_index} out of range (found {device_count} devices).",
                    timestamp_utc=now,
                )

            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)

            raw_name = pynvml.nvmlDeviceGetName(handle)
            device_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)

            raw_driver = pynvml.nvmlSystemGetDriverVersion()
            driver_version = raw_driver.decode("utf-8") if isinstance(raw_driver, bytes) else str(raw_driver)

            raw_nvml = pynvml.nvmlSystemGetNVMLVersion()
            nvml_version = raw_nvml.decode("utf-8") if isinstance(raw_nvml, bytes) else str(raw_nvml)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_b = int(mem_info.total)
            used_b = int(mem_info.used)
            free_b = int(mem_info.free)

            gpu_util: Optional[float] = None
            mem_util: Optional[float] = None
            try:
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = float(util_rates.gpu)
                mem_util = float(util_rates.memory)
            except pynvml.NVMLError as util_err:
                logger.debug("Could not read GPU utilization rates: %s", util_err)

            temp_c: Optional[int] = None
            try:
                temp_c = int(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except pynvml.NVMLError as temp_err:
                logger.debug("Could not read GPU temperature: %s", temp_err)

            power_w: Optional[float] = None
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = round(float(power_mw) / 1000.0, 2)
            except pynvml.NVMLError:
                pass

            return GPUTelemetry(
                available=True,
                device_index=self.device_index,
                device_name=device_name,
                driver_version=driver_version,
                nvml_version=nvml_version,
                total_vram_bytes=total_b,
                used_vram_bytes=used_b,
                free_vram_bytes=free_b,
                total_vram_mib=round(total_b / BYTES_PER_MIB, 2),
                used_vram_mib=round(used_b / BYTES_PER_MIB, 2),
                free_vram_mib=round(free_b / BYTES_PER_MIB, 2),
                gpu_utilization_pct=gpu_util,
                memory_utilization_pct=mem_util,
                temperature_celsius=temp_c,
                power_usage_watts=power_w,
                error=None,
                timestamp_utc=now,
            )

        except Exception as exc:
            err_msg = f"NVML query failed on device {self.device_index}: {str(exc)}"
            logger.warning(err_msg)
            return GPUTelemetry(
                available=False,
                device_index=self.device_index,
                error=err_msg,
                timestamp_utc=now,
            )


gpu_monitor = GPUMonitor()