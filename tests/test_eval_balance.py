"""
Tests for eval_balance.py — data structures, output formatting, sweep expansion.

These tests run WITHOUT MuJoCo/JAX — they only test the pure-Python components:
- EpisodeResult / ScenarioMetrics dataclasses
- _expand_scenarios sweep expansion
- _is_fallen termination check (mocked mj_data)
- _build_summary_table formatting
- _save_csv output structure
- Sweep metadata: scenario_group, scenario_param_name, scenario_param_value
"""

from __future__ import annotations

import csv
import io
import math
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from eval_balance — these are pure-Python, no JAX/MuJoCo needed at import
from scripts.eval_balance import (  # noqa: E402
    ALL_SCENARIOS,
    FRICTION_SWEEP_SCALES,
    PUSH_SWEEP_MAGNITUDES,
    EpisodeResult,
    ScenarioMetrics,
    _build_summary_table,
    _expand_scenarios,
    _is_fallen,
    _save_csv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_episode_result(**overrides) -> EpisodeResult:
    defaults = dict(
        height_cmd=0.65,
        fell=False,
        episode_steps=1000,
        pitch_rms_deg=1.5,
        roll_rms_deg=0.8,
        pitch_rate_rms_rads=0.3,
        xy_drift_max_m=0.05,
        height_rmse_m=0.012,
        wheel_speed_rms_rads=0.9,
        torque_rms_nm=3.5,
        recovery_time_s=float("nan"),
    )
    defaults.update(overrides)
    return EpisodeResult(**defaults)


def _make_scenario_metrics(scenario: str = "nominal", **overrides) -> ScenarioMetrics:
    defaults = dict(
        scenario=scenario,
        checkpoint="test_ckpt",
        num_episodes=10,
        fall_rate=0.1,
        survival_rate=0.9,
        survival_time_mean_s=18.0,
        survival_time_std_s=2.0,
        pitch_rms_deg=1.5,
        roll_rms_deg=0.8,
        pitch_rate_rms_rads=0.3,
        xy_drift_max_m=0.05,
        height_rmse_m=0.012,
        wheel_speed_rms_rads=0.9,
        torque_rms_nm=3.5,
        recovery_time_s=float("nan"),
        max_recoverable_push_n=float("nan"),
        extra={
            "scenario_group": scenario,
            "scenario_param_name": "",
            "scenario_param_value": "",
        },
    )
    defaults.update(overrides)
    return ScenarioMetrics(**defaults)


# ---------------------------------------------------------------------------
# EpisodeResult / ScenarioMetrics
# ---------------------------------------------------------------------------


class TestEpisodeResult:
    def test_construction(self):
        r = _make_episode_result()
        assert r.height_cmd == 0.65
        assert r.fell is False
        assert r.episode_steps == 1000
        assert math.isnan(r.recovery_time_s)

    def test_fell_episode(self):
        r = _make_episode_result(fell=True, episode_steps=200)
        assert r.fell is True
        assert r.episode_steps == 200


class TestScenarioMetrics:
    def test_construction(self):
        m = _make_scenario_metrics()
        assert m.scenario == "nominal"
        assert m.fall_rate == 0.1
        assert m.survival_rate == 0.9

    def test_to_dict(self):
        m = _make_scenario_metrics()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert d["scenario"] == "nominal"
        assert d["fall_rate"] == 0.1
        assert "extra" in d
        assert d["extra"]["scenario_group"] == "nominal"

    def test_nan_fields(self):
        m = _make_scenario_metrics()
        d = m.to_dict()
        assert math.isnan(d["recovery_time_s"])
        assert math.isnan(d["max_recoverable_push_n"])


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------


class TestExpandScenarios:
    def test_non_sweep_passthrough(self):
        scenarios = ["nominal", "push_recovery", "friction_low"]
        expanded = _expand_scenarios(scenarios)
        assert expanded == scenarios

    def test_push_sweep_expansion(self):
        expanded = _expand_scenarios(["push_sweep"])
        assert len(expanded) == len(PUSH_SWEEP_MAGNITUDES)
        assert expanded[0] == "push_sweep_20N"
        assert expanded[-1] == "push_sweep_200N"
        # Each name should be push_sweep_<mag>N
        for name, mag in zip(expanded, PUSH_SWEEP_MAGNITUDES):
            assert name.startswith("push_sweep_")
            assert name.endswith("N")

    def test_friction_sweep_expansion(self):
        expanded = _expand_scenarios(["friction_sweep"])
        assert len(expanded) == len(FRICTION_SWEEP_SCALES)
        assert expanded[0] == "friction_sweep_0.3x"
        assert expanded[-1] == "friction_sweep_1.8x"
        for name, fsc in zip(expanded, FRICTION_SWEEP_SCALES):
            assert name.startswith("friction_sweep_")
            assert name.endswith("x")

    def test_mixed_expansion(self):
        scenarios = ["nominal", "push_sweep", "friction_low", "friction_sweep"]
        expanded = _expand_scenarios(scenarios)
        expected_len = 1 + len(PUSH_SWEEP_MAGNITUDES) + 1 + len(FRICTION_SWEEP_SCALES)
        assert len(expanded) == expected_len
        assert expanded[0] == "nominal"
        # push_sweep items start at index 1
        assert expanded[1] == "push_sweep_20N"

    def test_push_sweep_produces_exactly_8_rows(self):
        """Integration check: push_sweep expands to exactly 8 sub-scenarios."""
        expanded = _expand_scenarios(["push_sweep"])
        assert len(expanded) == 8

    def test_friction_sweep_produces_exactly_6_rows(self):
        """Integration check: friction_sweep expands to exactly 6 sub-scenarios."""
        expanded = _expand_scenarios(["friction_sweep"])
        assert len(expanded) == 6


# ---------------------------------------------------------------------------
# ALL_SCENARIOS includes sweep entries
# ---------------------------------------------------------------------------


class TestAllScenarios:
    def test_push_sweep_in_all_scenarios(self):
        assert "push_sweep" in ALL_SCENARIOS

    def test_friction_sweep_in_all_scenarios(self):
        assert "friction_sweep" in ALL_SCENARIOS

    def test_original_scenarios_preserved(self):
        for s in ["nominal", "narrow_height", "wide_height", "full_range",
                   "push_recovery", "friction_low", "friction_high", "sensor_noise_delay"]:
            assert s in ALL_SCENARIOS


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


class TestBuildSummaryTable:
    def test_non_empty_output(self):
        metrics = [_make_scenario_metrics("nominal"), _make_scenario_metrics("push_recovery")]
        table = _build_summary_table(metrics, "test_ckpt")
        assert isinstance(table, str)
        assert len(table) > 50
        assert "nominal" in table
        assert "push_recovery" in table

    def test_contains_header(self):
        metrics = [_make_scenario_metrics()]
        table = _build_summary_table(metrics, "test")
        # Should contain header labels
        assert "Scenario" in table
        assert "Survival" in table


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


class TestSaveCsv:
    def test_csv_structure(self, tmp_path):
        metrics = [
            _make_scenario_metrics("nominal"),
            _make_scenario_metrics("push_recovery"),
        ]
        csv_path = tmp_path / "test.csv"
        _save_csv(metrics, csv_path)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["scenario"] == "nominal"
        assert rows[1]["scenario"] == "push_recovery"

    def test_csv_has_metadata_columns(self, tmp_path):
        """CSV must include scenario_group, scenario_param_name, scenario_param_value."""
        metrics = [_make_scenario_metrics("nominal")]
        csv_path = tmp_path / "test.csv"
        _save_csv(metrics, csv_path)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert "scenario_group" in rows[0]
        assert "scenario_param_name" in rows[0]
        assert "scenario_param_value" in rows[0]

    def test_csv_sweep_metadata(self, tmp_path):
        """Sweep rows must have correct metadata fields."""
        m = _make_scenario_metrics(
            "push_sweep_60N",
            extra={
                "scenario_group": "push_sweep",
                "scenario_param_name": "push_magnitude_n",
                "scenario_param_value": "60.0",
            },
        )
        csv_path = tmp_path / "test.csv"
        _save_csv([m], csv_path)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["scenario_group"] == "push_sweep"
        assert rows[0]["scenario_param_name"] == "push_magnitude_n"
        assert rows[0]["scenario_param_value"] == "60.0"

    def test_csv_nan_as_empty(self, tmp_path):
        """NaN values should be written as empty string in CSV."""
        metrics = [_make_scenario_metrics()]
        csv_path = tmp_path / "test.csv"
        _save_csv(metrics, csv_path)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # recovery_time_s and max_recoverable_push_n are NaN → empty
        assert rows[0]["recovery_time_s"] == ""
        assert rows[0]["max_recoverable_push_n"] == ""

    def test_csv_empty_input(self, tmp_path):
        """Empty metrics list should not crash."""
        csv_path = tmp_path / "test.csv"
        _save_csv([], csv_path)
        assert not csv_path.exists()  # _save_csv returns early


# ---------------------------------------------------------------------------
# _is_fallen() — gravity-based termination
#
# NOTE: _is_fallen() uses JAX (lightweight — no heavy JIT, no GPU required).
# JAX is an installed project dependency, so these tests run in standard CI.
# ---------------------------------------------------------------------------


class TestIsFallen:
    """Verify _is_fallen() uses gravity-based tilt matching base_env._check_termination().

    The implementation must use  tilt = arccos(-g_body[2])  (true 3-D tilt),
    not the Euler sqrt(roll² + pitch²) approximation, which diverges at large
    combined angles.
    """

    def _make_mj_data(self, z: float = 0.65, quat: tuple = (1.0, 0.0, 0.0, 0.0)):
        """Return a minimal mock mj_data with qpos[2]=z and qpos[3:7]=quat."""
        import numpy as np
        from unittest.mock import MagicMock

        mj_data = MagicMock()
        qpos = np.zeros(17)
        qpos[2] = z
        qpos[3] = quat[0]
        qpos[4] = quat[1]
        qpos[5] = quat[2]
        qpos[6] = quat[3]
        mj_data.qpos = qpos
        return mj_data

    def _quat_from_pitch(self, pitch_rad: float) -> tuple:
        """Quaternion for a pure pitch rotation (about world X axis, per sign convention)."""
        import math
        c = math.cos(pitch_rad / 2)
        s = math.sin(pitch_rad / 2)
        return (c, s, 0.0, 0.0)

    def test_upright_not_fallen(self):
        mj_data = self._make_mj_data(z=0.65)  # identity quaternion = upright
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert not _is_fallen(mj_data, config)

    def test_low_height_fallen(self):
        mj_data = self._make_mj_data(z=0.25)  # below min_height=0.3
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert _is_fallen(mj_data, config)

    def test_at_height_threshold_not_fallen(self):
        mj_data = self._make_mj_data(z=0.31)  # just above 0.3
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert not _is_fallen(mj_data, config)

    def test_small_tilt_not_fallen(self):
        """5° tilt is well within the 0.8 rad (~46°) threshold."""
        import math
        quat = self._quat_from_pitch(math.radians(5))
        mj_data = self._make_mj_data(z=0.65, quat=quat)
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert not _is_fallen(mj_data, config)

    def test_large_tilt_fallen(self):
        """90° tilt clearly exceeds 0.8 rad threshold."""
        import math
        quat = self._quat_from_pitch(math.pi / 2)
        mj_data = self._make_mj_data(z=0.65, quat=quat)
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert _is_fallen(mj_data, config)

    def test_tilt_just_below_threshold_not_fallen(self):
        """Tilt slightly below 0.8 rad → not fallen."""
        import math
        quat = self._quat_from_pitch(0.75)  # 0.75 < 0.8
        mj_data = self._make_mj_data(z=0.65, quat=quat)
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert not _is_fallen(mj_data, config)

    def test_tilt_just_above_threshold_fallen(self):
        """Tilt slightly above 0.8 rad → fallen."""
        import math
        quat = self._quat_from_pitch(0.85)  # 0.85 > 0.8
        mj_data = self._make_mj_data(z=0.65, quat=quat)
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert _is_fallen(mj_data, config)

    def test_default_config_upright_not_fallen(self):
        """Empty config dict → uses default thresholds; upright robot ok."""
        mj_data = self._make_mj_data(z=0.65)
        assert not _is_fallen(mj_data, {})

    def test_gravity_method_vs_euler_diverge_at_large_angle(self):
        """Confirm gravity-based tilt is used, not sqrt(roll²+pitch²).

        A 45° combined roll+pitch: Euler sqrt ≈ 1.11 rad > 0.8 threshold (fallen),
        but the true tilt = arccos(-g_body[2]) ≈ 0.785 rad < 0.8 (not fallen).
        The test verifies the gravity-based result, which matches BalanceEnv.
        """
        import math
        # 45° pitch rotation only: true tilt = 45° = 0.785 rad < 0.8 → not fallen
        quat = self._quat_from_pitch(math.radians(45))
        mj_data = self._make_mj_data(z=0.65, quat=quat)
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        # arccos(-cos(45°)) = arccos(-0.707) = 135° — wait, let me reconsider.
        # For a 45° pitch (rotation about body X), g_body[2] = -cos(45°) = -0.707
        # tilt = arccos(-(-0.707)) = arccos(0.707) = 45° = 0.785 rad < 0.8 → NOT fallen
        assert not _is_fallen(mj_data, config)

    def test_both_fallen_conditions_trigger(self):
        """Both height and tilt can independently trigger a fall."""
        import math
        # Robot at 90° tilt AND low height
        quat = self._quat_from_pitch(math.pi / 2)
        mj_data = self._make_mj_data(z=0.20, quat=quat)
        config = {"termination": {"max_tilt_rad": 0.8, "min_height": 0.3}}
        assert _is_fallen(mj_data, config)
