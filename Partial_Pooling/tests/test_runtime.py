import sys
from types import ModuleType

from Partial_Pooling import runtime


def test_runtime_selects_exactly_one_gpu_before_numeric_imports(monkeypatch):
    calls = []
    fake_autocvd = ModuleType("autocvd")

    def select_gpu(**kwargs):
        calls.append(kwargs)
        return [7]

    fake_autocvd.autocvd = select_gpu
    monkeypatch.setitem(sys.modules, "autocvd", fake_autocvd)
    monkeypatch.setattr(
        runtime,
        "configure_cpu_limit",
        lambda max_threads: {"active_cpus": max_threads},
    )

    result = runtime.configure_runtime()

    assert calls == [{"num_gpus": 1, "interval": 1}]
    assert result == {"active_cpus": 3}
