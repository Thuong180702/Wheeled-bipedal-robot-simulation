"""Tests for Step E root-cause diagnostic helpers.

Tests pure helper functions from scripts.diagnose_step_e_root_causes
before the full diagnostic script is implemented.
"""

import math
import os
from pathlib import Path

import pytest


# ---- Test that module is importable (RED gate) ----

def test_module_importable():
    """The diagnostic module must be importable from scripts/."""
    import importlib
    mod = importlib.import_module("scripts.diagnose_step_e_root_causes")
    assert mod is not None


# ---- Axis sign helpers ----

def test_current_sagittal_axis_zero_yaw():
    from scripts.diagnose_step_e_root_causes import current_sagittal_axis
    axis = current_sagittal_axis(0.0)
    assert axis[0] == pytest.approx(0.0, abs=1e-10)
    assert axis[1] == pytest.approx(1.0, abs=1e-10)


def test_flipped_sagittal_axis_is_exact_negative():
    from scripts.diagnose_step_e_root_causes import current_sagittal_axis, flipped_sagittal_axis
    for yaw in [0.0, 0.3, math.pi / 4, -1.2]:
        cur = current_sagittal_axis(yaw)
        flip = flipped_sagittal_axis(yaw)
        assert flip[0] == pytest.approx(-cur[0], abs=1e-12)
        assert flip[1] == pytest.approx(-cur[1], abs=1e-12)


def test_current_sagittal_axis_nonzero_yaw():
    from scripts.diagnose_step_e_root_causes import current_sagittal_axis
    yaw = math.radians(30)
    axis = current_sagittal_axis(yaw)
    assert axis[0] == pytest.approx(math.sin(yaw), abs=1e-10)
    assert axis[1] == pytest.approx(math.cos(yaw), abs=1e-10)


# ---- Velocity frame sample ----

def test_velocity_frame_sample_fields():
    from scripts.diagnose_step_e_root_causes import velocity_frame_sample
    sample = velocity_frame_sample(
        raw_com_vy=0.5,
        projected_sagittal_velocity=0.3,
        actual_passed_to_controller=0.5,
    )
    assert sample["raw_com_vy"] == 0.5
    assert sample["projected_sagittal_velocity"] == 0.3
    assert sample["actual_value_passed_to_controller_as_sagittal_velocity_m_s"] == 0.5
    assert sample["difference"] == pytest.approx(0.5 - 0.3)


def test_velocity_frame_sample_zero_difference_when_projected_matches_raw():
    from scripts.diagnose_step_e_root_causes import velocity_frame_sample
    sample = velocity_frame_sample(
        raw_com_vy=0.7,
        projected_sagittal_velocity=0.7,
        actual_passed_to_controller=0.7,
    )
    assert sample["difference"] == pytest.approx(0.0)


# ---- Stop-gate for 5000-step runs ----

def test_should_run_5000_gate_both_survive():
    from scripts.diagnose_step_e_root_causes import should_run_5000_gate
    assert should_run_5000_gate(True, True) is True


def test_should_run_5000_gate_one_terminated():
    from scripts.diagnose_step_e_root_causes import should_run_5000_gate
    assert should_run_5000_gate(True, False) is False
    assert should_run_5000_gate(False, True) is False
    assert should_run_5000_gate(False, False) is False


# ---- Posture metric: stable-roll hip-roll error percentage ----

def test_percent_abs_error_gt_threshold_while_roll_stable_basic():
    from scripts.diagnose_step_e_root_causes import percent_abs_error_gt_threshold_while_roll_stable
    # 4 samples: 3 have roll stable, 1 has roll unstable
    hip_roll_errors = [0.15, 0.05, 0.12, 0.03]
    roll_y_values = [0.01, 0.10, 0.02, 0.04]
    # Stable roll samples: 0, 2, 3 (abs(roll_y) < 0.05)
    # Of those, sample 0 and 2 have abs(error) > 0.10, so 2/3 = 66.67%
    result = percent_abs_error_gt_threshold_while_roll_stable(
        hip_roll_errors, roll_y_values, error_threshold=0.10, roll_stable_threshold=0.05
    )
    assert result == pytest.approx(100.0 * 2.0 / 3.0, abs=0.01)


def test_percent_abs_error_gt_threshold_while_roll_stable_mixed():
    from scripts.diagnose_step_e_root_causes import percent_abs_error_gt_threshold_while_roll_stable
    hip_roll_errors = [0.15, 0.05, 0.03, 0.08]
    roll_y_values = [0.01, 0.02, 0.10, 0.03]
    # Stable roll samples: 0, 1, 3 (roll_y < 0.05)
    # Of those, abs(error) > 0.10: sample 0 only → 1/3 = 33.33%
    result = percent_abs_error_gt_threshold_while_roll_stable(
        hip_roll_errors, roll_y_values, error_threshold=0.10, roll_stable_threshold=0.05
    )
    assert result == pytest.approx(100.0 / 3.0, abs=0.01)


def test_percent_abs_error_gt_threshold_while_roll_stable_no_stable():
    from scripts.diagnose_step_e_root_causes import percent_abs_error_gt_threshold_while_roll_stable
    hip_roll_errors = [0.15, 0.05]
    roll_y_values = [0.10, 0.20]
    result = percent_abs_error_gt_threshold_while_roll_stable(
        hip_roll_errors, roll_y_values, error_threshold=0.10, roll_stable_threshold=0.05
    )
    assert result == 0.0


def test_percent_abs_error_gt_threshold_while_roll_stable_uses_absolute_error():
    from scripts.diagnose_step_e_root_causes import percent_abs_error_gt_threshold_while_roll_stable
    hip_roll_errors = [-0.15, 0.05]
    roll_y_values = [0.01, 0.01]
    # Both have stable roll. Sample 0: abs(-0.15) > 0.10 → True. Sample 1: abs(0.05) > 0.10 → False.
    result = percent_abs_error_gt_threshold_while_roll_stable(
        hip_roll_errors, roll_y_values, error_threshold=0.10, roll_stable_threshold=0.05
    )
    assert result == pytest.approx(50.0)


# ---- Artifact validation ----

def test_validate_required_artifacts_all_present(tmp_path):
    from scripts.diagnose_step_e_root_causes import validate_required_artifacts
    files = ["a.csv", "b.json"]
    for f in files:
        (tmp_path / f).write_text("ok")
    missing = validate_required_artifacts(tmp_path, files)
    assert missing == []


def test_validate_required_artifacts_some_missing(tmp_path):
    from scripts.diagnose_step_e_root_causes import validate_required_artifacts
    (tmp_path / "a.csv").write_text("ok")
    missing = validate_required_artifacts(tmp_path, ["a.csv", "b.json", "c.md"])
    assert "b.json" in missing
    assert "c.md" in missing
    assert "a.csv" not in missing


def test_validate_required_artifacts_empty_dir(tmp_path):
    from scripts.diagnose_step_e_root_causes import validate_required_artifacts
    missing = validate_required_artifacts(tmp_path, ["x.csv"])
    assert missing == ["x.csv"]
