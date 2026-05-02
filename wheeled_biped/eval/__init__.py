"""wheeled_biped.eval - benchmark suite for evaluating trained policies."""

from __future__ import annotations

__all__ = ["BenchmarkResult", "run_benchmark"]


def __getattr__(name: str):
    if name in __all__:
        from wheeled_biped.eval.benchmark import BenchmarkResult, run_benchmark

        return {"BenchmarkResult": BenchmarkResult, "run_benchmark": run_benchmark}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
