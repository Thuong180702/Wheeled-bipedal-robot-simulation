"""Regression tests for removal of failed E0b position-containment runtime."""

from pathlib import Path

from wheeled_biped.controllers.sagittal_wheel_balance_controller import (
    SagittalWheelBalanceController,
)


def test_sagittal_wheel_balance_controller_has_no_position_containment_knobs():
    controller = SagittalWheelBalanceController()

    forbidden_attrs = [
        "enable_position_containment",
        "kp_position",
        "kd_position_velocity",
        "position_deadband_m",
        "position_soft_limit_m",
        "position_hard_limit_m",
        "max_position_bias",
        "pitch_gate_threshold_rad",
        "roll_gate_threshold_rad",
    ]

    for attr in forbidden_attrs:
        assert not hasattr(controller, attr)


def test_no_failed_e0b_runtime_logic_remains_in_controller_source():
    controller_source = Path(
        "wheeled_biped/controllers/sagittal_wheel_balance_controller.py"
    ).read_text(encoding="utf-8")

    forbidden_tokens = [
        "enable_position_containment",
        "position_correction_proportional",
        "position_correction_velocity",
        "position_correction_raw",
        "containment_violation",
        "in_soft_zone",
        "in_hard_zone",
    ]

    for token in forbidden_tokens:
        assert token not in controller_source
