"""Device selection edge-case tests."""

import sys
import types


def test_cuda_available_zero_devices_returns_cpu(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 0,
    )
    fake.backends = types.SimpleNamespace(mps=None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    sys.modules.pop("src.device", None)
    try:
        import src.device as device

        assert device.get_best_device() == "cpu"
    finally:
        sys.modules.pop("src.device", None)
