# tests/test_balance_core_structural_invariants.py
import pandas as pd
import pytest
from wheeled_biped.validation.structural_invariant_checker import (
    StructuralInvariantChecker,
    ArchitectureRegressionError,
)


def test_wrong_controller_mode_fails():
    """Non-balance-core controller mode should fail."""
    df = pd.DataFrame({
        "controller_mode": ["legacy-wbc", "legacy-wbc"],
        "ownership_violation_count": [0, 0],
        "hidden_torque_norm": [0.0, 0.0],
    })

    checker = StructuralInvariantChecker()
    with pytest.raises(ArchitectureRegressionError, match="controller_mode"):
        checker.check_all(df)


def test_all_invariants_pass():
    """Valid balance-core telemetry should pass all checks."""
    df = pd.DataFrame({
        "controller_mode": ["balance-core"] * 3,
        "step": [0, 1, 2],
        "ownership_violation_count": [0, 0, 0],
        "active_torque_owner_per_joint": [
            str(['shape_posture'] * 10),
            str(['shape_posture'] * 10),
            str(['shape_posture'] * 10),
        ],
        "hidden_torque_norm": [0.0, 0.0, 0.0],
        "tau_shape_posture_per_joint": [str([0.0] * 10)] * 3,
        "tau_support_feedforward_per_joint": [str([0.0] * 10)] * 3,
        "tau_sagittal_wheel_balance_per_joint": [str([0.0] * 10)] * 3,
        "tau_lateral_roll_balance_per_joint": [str([0.0] * 10)] * 3,
        "tau_total_raw_per_joint": [str([0.0] * 10)] * 3,
        "tau_total_clipped_per_joint": [str([0.0] * 10)] * 3,
        "tau_final_per_joint": [str([0.0] * 10)] * 3,
        "actuator_ctrl_per_joint": [str([0.0] * 10)] * 3,
        "torque_saturation_mask_per_joint": [str([False] * 10)] * 3,
        "torque_rate_saturation_mask_per_joint": [str([False] * 10)] * 3,
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 3,
        "contact_duration_s": [0.0, 0.002, 0.004],
    })

    checker = StructuralInvariantChecker()
    results = checker.check_all(df)

    assert results["controller_mode"] == "PASS"
    assert results["ownership_violations"] == "PASS"
    assert results["torque_owners"] == "PASS"
    assert results["hidden_torque"] == "PASS"


def test_non_finite_torque_detected():
    """Non-finite torque values should fail structural check."""
    df = pd.DataFrame({
        "controller_mode": ["balance-core"] * 3,
        "step": [0, 1, 2],
        "ownership_violation_count": [0, 0, 0],
        "active_torque_owner_per_joint": [str(['shape_posture'] * 10)] * 3,
        "hidden_torque_norm": [0.0, 0.0, 0.0],
        "tau_shape_posture_per_joint": [
            str([0.0] * 10),
            "[0.0, 1.0, 1e309, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]",  # 1e309 overflows to inf
            str([0.0] * 10),
        ],
        "tau_support_feedforward_per_joint": [str([0.0] * 10)] * 3,
        "tau_sagittal_wheel_balance_per_joint": [str([0.0] * 10)] * 3,
        "tau_lateral_roll_balance_per_joint": [str([0.0] * 10)] * 3,
        "tau_total_raw_per_joint": [str([0.0] * 10)] * 3,
        "tau_total_clipped_per_joint": [str([0.0] * 10)] * 3,
        "tau_final_per_joint": [str([0.0] * 10)] * 3,
        "actuator_ctrl_per_joint": [str([0.0] * 10)] * 3,
        "torque_saturation_mask_per_joint": [str([False] * 10)] * 3,
        "torque_rate_saturation_mask_per_joint": [str([False] * 10)] * 3,
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 3,
        "contact_duration_s": [0.0, 0.002, 0.004],
    })

    checker = StructuralInvariantChecker()
    with pytest.raises(ArchitectureRegressionError, match="non-finite"):
        checker.check_all(df)


def test_hidden_torque_exceeds_tolerance():
    """Hidden torque above tolerance should fail."""
    df = pd.DataFrame({
        "controller_mode": ["balance-core"] * 3,
        "step": [0, 1, 2],
        "ownership_violation_count": [0, 0, 0],
        "active_torque_owner_per_joint": [str(['shape_posture'] * 10)] * 3,
        "hidden_torque_norm": [0.0, 1e-3, 0.0],  # Exceeds 1e-6 tolerance
        "tau_shape_posture_per_joint": [str([0.0] * 10)] * 3,
        "tau_support_feedforward_per_joint": [str([0.0] * 10)] * 3,
        "tau_sagittal_wheel_balance_per_joint": [str([0.0] * 10)] * 3,
        "tau_lateral_roll_balance_per_joint": [str([0.0] * 10)] * 3,
        "tau_total_raw_per_joint": [str([0.0] * 10)] * 3,
        "tau_total_clipped_per_joint": [str([0.0] * 10)] * 3,
        "tau_final_per_joint": [str([0.0] * 10)] * 3,
        "actuator_ctrl_per_joint": [str([0.0] * 10)] * 3,
        "torque_saturation_mask_per_joint": [str([False] * 10)] * 3,
        "torque_rate_saturation_mask_per_joint": [str([False] * 10)] * 3,
        "contact_supervisor_state": ["DOUBLE_CONTACT"] * 3,
        "contact_duration_s": [0.0, 0.002, 0.004],
    })

    checker = StructuralInvariantChecker()
    with pytest.raises(ArchitectureRegressionError, match="Hidden torque"):
        checker.check_all(df)
