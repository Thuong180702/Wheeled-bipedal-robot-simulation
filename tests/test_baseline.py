"""
Tests for wheeled_biped.eval.baseline comparison logic.

All tests are pure Python (no JAX, no MuJoCo).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wheeled_biped.eval.baseline import (
    ComparisonResult,
    MetricDelta,
    _REGRESSION_SPECS,
    compare_baselines,
    compare_files,
    load_result,
    _compute_delta,
    _flatten_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    mode: str = "nominal",
    reward_mean: float = 50.0,
    fall_rate: float = 0.10,
    success_rate: float = 0.90,
    reward_p5: float = 20.0,
    **extras,
) -> dict:
    base = {
        "mode": mode,
        "num_episodes": 100,
        "reward_mean": reward_mean,
        "reward_std": 5.0,
        "reward_min": 10.0,
        "reward_p5": reward_p5,
        "reward_p50": 48.0,
        "reward_p95": 75.0,
        "reward_max": 90.0,
        "episode_length_mean": 950.0,
        "episode_length_max": 1000,
        "success_rate": success_rate,
        "fall_rate": fall_rate,
        "timeout_rate": success_rate,
        "mode_metrics": {},
        "checkpoint": "fake/checkpoint",
        "stage": "balance",
        "seed": 0,
    }
    base.update(extras)
    return base


# ---------------------------------------------------------------------------
# Tests: _flatten_metrics
# ---------------------------------------------------------------------------

class TestFlattenMetrics:
    def test_extracts_top_level_floats(self):
        d = _make_result()
        flat = _flatten_metrics(d)
        assert "reward_mean" in flat
        assert "fall_rate" in flat
        assert "success_rate" in flat

    def test_excludes_non_numeric_keys(self):
        d = _make_result()
        flat = _flatten_metrics(d)
        assert "mode" not in flat
        assert "checkpoint" not in flat
        assert "stage" not in flat

    def test_flattens_mode_metrics_scalars(self):
        d = _make_result()
        d["mode_metrics"] = {"overall_height_rmse": 0.05, "height_error_mean": 0.03}
        flat = _flatten_metrics(d)
        assert flat["overall_height_rmse"] == pytest.approx(0.05)
        assert flat["height_error_mean"] == pytest.approx(0.03)

    def test_skips_per_command_list(self):
        d = _make_result()
        d["mode_metrics"] = {"per_command": [{"height_command": 0.5}]}
        flat = _flatten_metrics(d)
        assert "per_command" not in flat


# ---------------------------------------------------------------------------
# Tests: _compute_delta
# ---------------------------------------------------------------------------

class TestComputeDelta:
    def test_lower_better_regression_detected(self):
        # fall_rate increased by 0.10, tolerance is abs 0.05 → regression
        spec = ("lower_better", "abs", 0.05)
        delta = _compute_delta("fall_rate", 0.10, 0.20, spec)
        assert delta.is_regression
        assert not delta.is_improvement

    def test_lower_better_improvement_detected(self):
        spec = ("lower_better", "abs", 0.05)
        delta = _compute_delta("fall_rate", 0.20, 0.05, spec)  # big drop
        assert delta.is_improvement
        assert not delta.is_regression

    def test_lower_better_within_tolerance_ok(self):
        spec = ("lower_better", "abs", 0.05)
        delta = _compute_delta("fall_rate", 0.10, 0.13, spec)  # +0.03 < 0.05 tol
        assert not delta.is_regression
        assert not delta.is_improvement

    def test_higher_better_regression_detected(self):
        # reward_mean dropped 10% with 5% relative tol → regression
        spec = ("higher_better", "rel", 0.05)
        delta = _compute_delta("reward_mean", 100.0, 88.0, spec)
        assert delta.is_regression

    def test_higher_better_improvement_detected(self):
        spec = ("higher_better", "rel", 0.05)
        delta = _compute_delta("reward_mean", 100.0, 115.0, spec)
        assert delta.is_improvement

    def test_higher_better_within_relative_tolerance_ok(self):
        spec = ("higher_better", "rel", 0.05)
        # -3% from 100 → 97, tolerance is 5% of 100 = 5 → within tol
        delta = _compute_delta("reward_mean", 100.0, 97.0, spec)
        assert not delta.is_regression


# ---------------------------------------------------------------------------
# Tests: compare_baselines
# ---------------------------------------------------------------------------

class TestCompareBaselines:
    def test_no_regression_when_identical(self):
        d = _make_result(reward_mean=50.0, fall_rate=0.10)
        result = compare_baselines(d, d)
        assert result.passed
        assert result.regressions == []

    def test_fall_rate_regression_detected(self):
        baseline = _make_result(fall_rate=0.10)
        current = _make_result(fall_rate=0.20)   # +0.10 > tol 0.05
        result = compare_baselines(current, baseline)
        reg_names = [d.metric for d in result.regressions]
        assert "fall_rate" in reg_names
        assert not result.passed

    def test_success_rate_regression_detected(self):
        baseline = _make_result(success_rate=0.90)
        current = _make_result(success_rate=0.70)  # -0.20 > tol 0.05
        result = compare_baselines(current, baseline)
        reg_names = [d.metric for d in result.regressions]
        assert "success_rate" in reg_names

    def test_reward_mean_regression_detected(self):
        baseline = _make_result(reward_mean=100.0)
        current = _make_result(reward_mean=85.0)   # -15% > tol 5%
        result = compare_baselines(current, baseline)
        reg_names = [d.metric for d in result.regressions]
        assert "reward_mean" in reg_names

    def test_improvement_detected(self):
        baseline = _make_result(fall_rate=0.20)
        current = _make_result(fall_rate=0.05)   # -0.15, big improvement
        result = compare_baselines(current, baseline)
        imp_names = [d.metric for d in result.improvements]
        assert "fall_rate" in imp_names

    def test_tolerance_override_tightens_check(self):
        baseline = _make_result(fall_rate=0.10)
        current = _make_result(fall_rate=0.13)   # +0.03, within default 0.05 tol
        # Tighten to 0.01
        result = compare_baselines(current, baseline, tolerances={"fall_rate": 0.01})
        reg_names = [d.metric for d in result.regressions]
        assert "fall_rate" in reg_names

    def test_mode_metrics_height_rmse_regression(self):
        baseline = _make_result()
        baseline["mode_metrics"] = {"overall_height_rmse": 0.05}
        current = _make_result()
        current["mode_metrics"] = {"overall_height_rmse": 0.09}  # +80%, tol 10%
        result = compare_baselines(current, baseline)
        reg_names = [d.metric for d in result.regressions]
        assert "overall_height_rmse" in reg_names

    def test_passed_property_true_when_no_regressions(self):
        d = _make_result()
        result = compare_baselines(d, d)
        assert result.passed is True

    def test_passed_property_false_when_regression(self):
        baseline = _make_result(fall_rate=0.10)
        current = _make_result(fall_rate=0.30)
        result = compare_baselines(current, baseline)
        assert result.passed is False

    def test_result_mode_from_current(self):
        baseline = _make_result(mode="nominal")
        current = _make_result(mode="push_recovery")
        result = compare_baselines(current, baseline)
        assert result.mode == "push_recovery"


# ---------------------------------------------------------------------------
# Tests: compare_files (disk I/O round-trip)
# ---------------------------------------------------------------------------

class TestCompareFiles:
    def test_roundtrip_identical(self, tmp_path):
        d = _make_result()
        p_base = tmp_path / "baseline.json"
        p_curr = tmp_path / "current.json"
        p_base.write_text(json.dumps(d))
        p_curr.write_text(json.dumps(d))
        result = compare_files(p_curr, p_base)
        assert result.passed

    def test_roundtrip_regression(self, tmp_path):
        baseline = _make_result(fall_rate=0.10)
        current = _make_result(fall_rate=0.30)
        p_base = tmp_path / "baseline.json"
        p_curr = tmp_path / "current.json"
        p_base.write_text(json.dumps(baseline))
        p_curr.write_text(json.dumps(current))
        result = compare_files(p_curr, p_base)
        assert not result.passed

    def test_load_result_roundtrip(self, tmp_path):
        d = _make_result()
        p = tmp_path / "result.json"
        p.write_text(json.dumps(d))
        loaded = load_result(p)
        assert loaded["reward_mean"] == pytest.approx(d["reward_mean"])


# ---------------------------------------------------------------------------
# Tests: ComparisonResult helpers
# ---------------------------------------------------------------------------

class TestComparisonResult:
    def _make_comparison(self, regressions=0, improvements=0, ok=2):
        deltas = []
        for i in range(regressions):
            deltas.append(MetricDelta(
                metric=f"reg_{i}", baseline_value=1.0, current_value=0.5,
                delta=-0.5, direction="higher_better", tolerance=0.01,
                is_regression=True, is_improvement=False,
            ))
        for i in range(improvements):
            deltas.append(MetricDelta(
                metric=f"imp_{i}", baseline_value=0.5, current_value=1.0,
                delta=0.5, direction="higher_better", tolerance=0.01,
                is_regression=False, is_improvement=True,
            ))
        for i in range(ok):
            deltas.append(MetricDelta(
                metric=f"ok_{i}", baseline_value=1.0, current_value=1.0,
                delta=0.0, direction="higher_better", tolerance=0.01,
                is_regression=False, is_improvement=False,
            ))
        return ComparisonResult(
            mode="nominal", baseline_file="b.json", current_file="c.json",
            deltas=deltas,
        )

    def test_regressions_list(self):
        r = self._make_comparison(regressions=2, improvements=1)
        assert len(r.regressions) == 2

    def test_improvements_list(self):
        r = self._make_comparison(regressions=0, improvements=3)
        assert len(r.improvements) == 3

    def test_ok_list(self):
        r = self._make_comparison(regressions=1, improvements=1, ok=2)
        assert len(r.ok) == 2

    def test_passed_no_regressions(self):
        r = self._make_comparison(regressions=0)
        assert r.passed

    def test_passed_with_regressions(self):
        r = self._make_comparison(regressions=1)
        assert not r.passed

    def test_to_dict_serialisable(self):
        r = self._make_comparison(regressions=1, improvements=1)
        d = r.to_dict()
        # Should be JSON-serialisable (no numpy, no jax arrays)
        import json as _json
        _json.dumps(d)  # should not raise
        assert d["passed"] is False
        assert d["num_regressions"] == 1
        assert d["num_improvements"] == 1
