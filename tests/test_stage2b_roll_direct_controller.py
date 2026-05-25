"""Unit tests for Stage2B Direct Roll Controller.

Tests verify that the direct roll controller correctly maps roll errors to hip_roll torques
without routing through contact force distribution.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.stage2b_roll_direct_controller import Stage2BRollDirectController


@pytest.fixture
def controller():
    """Create controller with standard config."""
    return Stage2BRollDirectController(
        k_roll=100.0,
        k_roll_rate=20.0,
        k_roll_integral=0.0,
        tau_hip_roll_max=15.0,
        max_roll_moment=30.0,
    )


def test_output_only_on_hip_roll_joints(controller):
    """Output torques only on hip_roll joints [0,5]."""
    tau_roll, diag = controller.compute_roll_torques(
        roll_y=0.1,  # 5.7 deg
        roll_rate_y=0.0,
    )

    # Verify only hip_roll joints are nonzero
    assert tau_roll[0] != 0.0, "l_hip_roll should be nonzero"
    assert tau_roll[5] != 0.0, "r_hip_roll should be nonzero"

    # Verify all other joints are zero
    for idx in [1, 2, 3, 4, 6, 7, 8, 9]:
        assert tau_roll[idx] == 0.0, f"Joint {idx} should be zero"


def test_hip_pitch_knee_remain_zero(controller):
    """Hip_pitch/knee joints remain zero."""
    tau_roll, diag = controller.compute_roll_torques(
        roll_y=0.2,
        roll_rate_y=0.1,
    )

    support_joints = [2, 3, 7, 8]
    for idx in support_joints:
        assert tau_roll[idx] == 0.0, f"Support joint {idx} should be zero"


def test_positive_roll_error_produces_restoring_torque(controller):
    """Positive roll error (rolling right) produces negative moment (roll left correction)."""
    controller.set_equilibrium_reference(0.0)

    tau_roll, diag = controller.compute_roll_torques(
        roll_y=0.1,  # positive roll error
        roll_rate_y=0.0,
    )

    # Positive roll error should produce negative moment
    assert diag["m_roll_cmd"] < 0.0, f"Positive roll error should produce negative moment, got {diag['m_roll_cmd']}"

    # Verify torques oppose the error
    # Left hip_roll: tau = -M_roll / 2, so negative M_roll gives positive tau
    # Right hip_roll: tau = +M_roll / 2, so negative M_roll gives negative tau
    assert tau_roll[0] > 0.0, "Left hip_roll should be positive for positive roll error"
    assert tau_roll[5] < 0.0, "Right hip_roll should be negative for positive roll error"


def test_negative_roll_error_produces_restoring_torque(controller):
    """Negative roll error (rolling left) produces positive moment (roll right correction)."""
    controller.set_equilibrium_reference(0.0)

    tau_roll, diag = controller.compute_roll_torques(
        roll_y=-0.1,  # negative roll error
        roll_rate_y=0.0,
    )

    # Negative roll error should produce positive moment
    assert diag["m_roll_cmd"] > 0.0, f"Negative roll error should produce positive moment, got {diag['m_roll_cmd']}"

    # Verify torques oppose the error
    # Left hip_roll: tau = -M_roll / 2, so positive M_roll gives negative tau
    # Right hip_roll: tau = +M_roll / 2, so positive M_roll gives positive tau
    assert tau_roll[0] < 0.0, "Left hip_roll should be negative for negative roll error"
    assert tau_roll[5] > 0.0, "Right hip_roll should be positive for negative roll error"


def test_moment_saturation_works(controller):
    """Moment saturation clips total roll moment."""
    controller.set_equilibrium_reference(0.0)

    # Request very large roll correction
    tau_roll, diag = controller.compute_roll_torques(
        roll_y=1.0,  # 57 deg - very large error
        roll_rate_y=0.0,
    )

    # Verify moment is clipped
    assert abs(diag["m_roll_clipped"]) <= controller.max_roll_moment, (
        f"Moment should be clipped to {controller.max_roll_moment}, got {abs(diag['m_roll_clipped'])}"
    )
    assert diag["moment_saturated"], "Moment saturation flag should be set"


def test_individual_torque_saturation_works(controller):
    """Individual hip_roll torques are clipped."""
    controller.set_equilibrium_reference(0.0)

    # Request very large roll correction
    tau_roll, diag = controller.compute_roll_torques(
        roll_y=1.0,  # 57 deg
        roll_rate_y=0.0,
    )

    # Verify individual torques are clipped
    assert abs(tau_roll[0]) <= controller.tau_hip_roll_max, (
        f"Left hip_roll should be clipped to {controller.tau_hip_roll_max}, got {abs(tau_roll[0])}"
    )
    assert abs(tau_roll[5]) <= controller.tau_hip_roll_max, (
        f"Right hip_roll should be clipped to {controller.tau_hip_roll_max}, got {abs(tau_roll[5])}"
    )


def test_zero_roll_error_produces_zero_torque(controller):
    """Zero roll error produces zero torque."""
    controller.set_equilibrium_reference(0.0)

    tau_roll, diag = controller.compute_roll_torques(
        roll_y=0.0,
        roll_rate_y=0.0,
    )

    assert diag["m_roll_cmd"] == 0.0, "Zero error should produce zero moment"
    assert tau_roll[0] == 0.0, "Left hip_roll should be zero"
    assert tau_roll[5] == 0.0, "Right hip_roll should be zero"


def test_roll_rate_damping(controller):
    """Roll rate produces damping torque."""
    controller.set_equilibrium_reference(0.0)

    # Positive roll rate (rolling right)
    tau_roll_pos, diag_pos = controller.compute_roll_torques(
        roll_y=0.0,
        roll_rate_y=0.1,
    )

    # Negative roll rate (rolling left)
    tau_roll_neg, diag_neg = controller.compute_roll_torques(
        roll_y=0.0,
        roll_rate_y=-0.1,
    )

    # Verify damping opposes velocity
    assert diag_pos["m_roll_cmd"] < 0.0, "Positive roll rate should produce negative damping moment"
    assert diag_neg["m_roll_cmd"] > 0.0, "Negative roll rate should produce positive damping moment"


def test_equilibrium_reference_offset(controller):
    """Equilibrium reference shifts the zero point."""
    controller.set_equilibrium_reference(0.1)

    # At equilibrium roll
    tau_roll_eq, diag_eq = controller.compute_roll_torques(
        roll_y=0.1,
        roll_rate_y=0.0,
    )

    # Above equilibrium
    tau_roll_above, diag_above = controller.compute_roll_torques(
        roll_y=0.2,
        roll_rate_y=0.0,
    )

    assert diag_eq["roll_error"] == 0.0, "At equilibrium should have zero error"
    assert diag_eq["m_roll_cmd"] == 0.0, "At equilibrium should produce zero moment"

    assert diag_above["roll_error"] > 0.0, "Above equilibrium should have positive error"
    assert diag_above["m_roll_cmd"] < 0.0, "Above equilibrium should produce negative moment"
