"""
Unit tests for GPUMonitor with mocked NVML telemetry (simulating an RTX 3050 6GB Laptop GPU)
and graceful degradation when NVML is absent or raises errors.
"""

from unittest.mock import MagicMock, patch
from app.profiling.gpu_monitor import GPUMonitor


def test_gpu_monitor_mocked_rtx_3050_6gb() -> None:
    """Mocks pynvml to simulate an NVIDIA GeForce RTX 3050 6GB Laptop GPU."""
    mock_nvml = MagicMock()
    mock_nvml.NVMLError = Exception
    mock_nvml.NVML_TEMPERATURE_GPU = 0
    mock_nvml.nvmlDeviceGetCount.return_value = 1

    handle = MagicMock()
    mock_nvml.nvmlDeviceGetHandleByIndex.return_value = handle
    mock_nvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 3050 6GB Laptop GPU"
    mock_nvml.nvmlSystemGetDriverVersion.return_value = b"555.85"
    mock_nvml.nvmlSystemGetNVMLVersion.return_value = b"12.555.85"

    mem_info = MagicMock()
    mem_info.total = 6442450944  # 6144 MiB
    mem_info.used = 2147483648   # 2048 MiB
    mem_info.free = 4294967296   # 4096 MiB
    mock_nvml.nvmlDeviceGetMemoryInfo.return_value = mem_info

    util = MagicMock()
    util.gpu = 34
    util.memory = 21
    mock_nvml.nvmlDeviceGetUtilizationRates.return_value = util
    mock_nvml.nvmlDeviceGetTemperature.return_value = 58
    mock_nvml.nvmlDeviceGetPowerUsage.return_value = 42500  # 42.5 W

    with patch("app.profiling.gpu_monitor.pynvml", mock_nvml), patch(
        "app.profiling.gpu_monitor.PYNVML_IMPORTED", True
    ):
        monitor = GPUMonitor(device_index=0)
        telemetry = monitor.get_telemetry()

        assert telemetry.available is True
        assert telemetry.device_name == "NVIDIA GeForce RTX 3050 6GB Laptop GPU"
        assert telemetry.total_vram_mib == 6144.0
        assert telemetry.used_vram_mib == 2048.0
        assert telemetry.free_vram_mib == 4096.0
        assert telemetry.gpu_utilization_pct == 34.0
        assert telemetry.temperature_celsius == 58
        assert telemetry.power_usage_watts == 42.5
        assert telemetry.error is None


def test_gpu_monitor_graceful_degradation_without_fabricating() -> None:
    """Verifies that when NVML fails to initialize, available=False and all metrics are None."""
    mock_nvml = MagicMock()

    class DummyNVMLError(Exception):
        pass

    mock_nvml.NVMLError = DummyNVMLError
    mock_nvml.nvmlInit.side_effect = DummyNVMLError("NVML Shared Library Not Found")

    with patch("app.profiling.gpu_monitor.pynvml", mock_nvml), patch(
        "app.profiling.gpu_monitor.PYNVML_IMPORTED", True
    ):
        monitor = GPUMonitor(device_index=0)
        telemetry = monitor.get_telemetry()

        assert telemetry.available is False
        assert telemetry.total_vram_bytes is None
        assert telemetry.free_vram_mib is None
        assert telemetry.gpu_utilization_pct is None
        assert "NVML Shared Library Not Found" in (telemetry.error or "")