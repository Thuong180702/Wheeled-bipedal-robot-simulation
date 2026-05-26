"""Unit tests for Stage2C Sagittal State-Feedback Controller.

Tests verify that the sagittal state-feedback controller correctly maps state errors
and wheel velocity to wheel torques with the verified sign convention and proper damping.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.stage2c_sagittal_state_feedback_controller import Stage2CSagittalStateFeedbackController


@pytest.fixture
def controller():
    """Create controller with standard config."""
    return Stage2CSagittalStateFeedbackController(
        k_pitch=20.0,
        k_pitch_rate=6.0,
        k_com_y=2.0,
        k_com_vy=4.0,
        k_cp_y=8.0,
        k_wheel_vel=0.3,
        max_tau_wheel=8.0,
    )


@pytest.fixture
def equilibrium_state():
    """Standard equilibrium state."""
    return {
        'pitch_x': 0.0,
        'com_y': -0.013535,
        'cp_y': -0.013535,
    }


def test_output_only_on_wheel_joints(controller, equilibrium_state):
    """Output torques only on wheel joints [4,9]."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.1,
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    # Verify only wheel joints are nonzero
    assert tau_wheel[4] != 0.0, "l_wheel should be nonzero"
    assert tau_wheel[9] != 0.0, "r_wheel should be nonzero"

    # Verify all other joints are zero
    for idx in [0, 1, 2, 3, 5, 6, 7, 8]:
        assert tau_wheel[idx] == 0.0, f"Joint {idx} should be zero"


def test_positive_pitch_produces_positive_torque(controller, equilibrium_state):
    """Positive pitch error (falling forward) produces positive torque (move backward).

    Sign convention verified by debug_wheel_sagittal_sign_simple.py:
    - Positive pitch_x means falling forward (-Y direction)
    - Positive wheel torque moves robot backward (+Y direction)
    - To oppose forward fall, need backward motion
    - Therefore: positive pitch → positive torque
    """
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.1,  # positive pitch error (falling forward)
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    assert diag["pitch_error"] > 0, "Pitch error should be positive"
    assert diag["tau_wheel_raw"] > 0, "Positive pitch should produce positive torque"
    assert tau_wheel[4] > 0, "Left wheel torque should be positive"
    assert tau_wheel[9] > 0, "Right wheel torque should be positive"


def test_negative_pitch_produces_negative_torque(controller, equilibrium_state):
    """Negative pitch error (falling backward) produces negative torque (move forward)."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=-0.1,  # negative pitch error (falling backward)
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    assert diag["pitch_error"] < 0, "Pitch error should be negative"
    assert diag["tau_wheel_raw"] < 0, "Negative pitch should produce negative torque"
    assert tau_wheel[4] < 0, "Left wheel torque should be negative"
    assert tau_wheel[9] < 0, "Right wheel torque should be negative"


def test_positive_wheel_velocity_produces_braking_torque(controller, equilibrium_state):
    """Positive wheel velocity produces negative damping torque (brake)."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.0,
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=10.0,  # positive wheel velocity
        wheel_vel_right=10.0,
    )

    assert diag["wheel_vel_mean"] > 0, "Wheel velocity mean should be positive"
    assert diag["term_wheel_vel"] > 0, "Wheel velocity term should be positive (before negation)"
    # The control law subtracts term_wheel_vel, so positive wheel_vel produces negative torque
    assert diag["tau_wheel_raw"] < 0, "Positive wheel velocity should produce negative (braking) torque"
    assert tau_wheel[4] < 0, "Left wheel torque should be negative (braking)"
    assert tau_wheel[9] < 0, "Right wheel torque should be negative (braking)"


def test_zero_errors_with_positive_wheel_velocity_produces_braking_only(controller, equilibrium_state):
    """If all errors are zero but wheel velocity is positive, controller commands braking torque."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.0,
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=5.0,
        wheel_vel_right=5.0,
    )

    # Verify all errors are zero
    assert diag["pitch_error"] == 0.0, "Pitch error should be zero"
    assert diag["pitch_rate_x"] == 0.0, "Pitch rate should be zero"
    assert diag["com_y_error"] == 0.0, "CoM Y error should be zero"
    assert diag["com_vy"] == 0.0, "CoM Y velocity should be zero"
    assert diag["cp_y_error"] == 0.0, "CP Y error should be zero"

    # Verify only wheel velocity damping is active
    assert diag["term_pitch"] == 0.0, "Pitch term should be zero"
    assert diag["term_pitch_rate"] == 0.0, "Pitch rate term should be zero"
    assert diag["term_com_y"] == 0.0, "CoM Y term should be zero"
    assert diag["term_com_vy"] == 0.0, "CoM Y velocity term should be zero"
    assert diag["term_cp_y"] == 0.0, "CP Y term should be zero"
    assert diag["term_wheel_vel"] > 0, "Wheel velocity term should be nonzero"

    # Verify braking torque is commanded
    assert diag["tau_wheel_raw"] < 0, "Should command negative (braking) torque"


def test_all_zero_produces_zero_torque(controller, equilibrium_state):
    """If all errors and wheel velocity are zero, output is zero."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.0,
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    # Verify all errors and states are zero
    assert diag["pitch_error"] == 0.0, "Pitch error should be zero"
    assert diag["pitch_rate_x"] == 0.0, "Pitch rate should be zero"
    assert diag["com_y_error"] == 0.0, "CoM Y error should be zero"
    assert diag["com_vy"] == 0.0, "CoM Y velocity should be zero"
    assert diag["cp_y_error"] == 0.0, "CP Y error should be zero"
    assert diag["wheel_vel_mean"] == 0.0, "Wheel velocity mean should be zero"

    # Verify all terms are zero
    assert diag["term_pitch"] == 0.0, "Pitch term should be zero"
    assert diag["term_pitch_rate"] == 0.0, "Pitch rate term should be zero"
    assert diag["term_com_y"] == 0.0, "CoM Y term should be zero"
    assert diag["term_com_vy"] == 0.0, "CoM Y velocity term should be zero"
    assert diag["term_cp_y"] == 0.0, "CP Y term should be zero"
    assert diag["term_wheel_vel"] == 0.0, "Wheel velocity term should be zero"

    # Verify output is zero
    assert diag["tau_wheel_raw"] == 0.0, "Torque command should be zero"
    assert tau_wheel[4] == 0.0, "Left wheel torque should be zero"
    assert tau_wheel[9] == 0.0, "Right wheel torque should be zero"


def test_torque_saturation(controller, equilibrium_state):
    """Torque saturation clips excessive commands."""
    controller.set_equilibrium_reference(**equilibrium_state)

    # Request very large pitch correction
    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=1.0,  # 57 deg - very large error
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    # Verify torque is clipped
    assert abs(diag["tau_wheel_clipped"]) <= controller.max_tau_wheel, (
        f"Torque should be clipped to {controller.max_tau_wheel}, got {abs(diag['tau_wheel_clipped'])}"
    )
    assert diag["saturated"], "Saturation flag should be set"


def test_telemetry_includes_all_terms(controller, equilibrium_state):
    """Telemetry includes all term contributions."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.1,
        pitch_rate_x=0.05,
        com_y=-0.01,
        com_vy=0.02,
        cp_y=-0.01,
        wheel_vel_left=5.0,
        wheel_vel_right=5.0,
    )

    # Verify all diagnostic fields are present
    required_fields = [
        "pitch_error",
        "pitch_rate_x",
        "com_y_error",
        "com_vy",
        "cp_y_error",
        "wheel_vel_left",
        "wheel_vel_right",
        "wheel_vel_mean",
        "term_pitch",
        "term_pitch_rate",
        "term_com_y",
        "term_com_vy",
        "term_cp_y",
        "term_wheel_vel",
        "tau_wheel_raw",
        "tau_wheel_clipped",
        "saturated",
    ]

    for field in required_fields:
        assert field in diag, f"Diagnostic field '{field}' should be present"


def test_wheel_velocity_damping_opposes_motion(controller, equilibrium_state):
    """Wheel velocity damping always opposes wheel motion."""
    controller.set_equilibrium_reference(**equilibrium_state)

    # Test positive wheel velocity
    tau_pos, diag_pos = controller.compute_wheel_torques(
        pitch_x=0.0,
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=10.0,
        wheel_vel_right=10.0,
    )

    # Test negative wheel velocity
    tau_neg, diag_neg = controller.compute_wheel_torques(
        pitch_x=0.0,
        pitch_rate_x=0.0,
        com_y=-0.013535,
        com_vy=0.0,
        cp_y=-0.013535,
        wheel_vel_left=-10.0,
        wheel_vel_right=-10.0,
    )

    # Verify damping opposes motion
    assert diag_pos["wheel_vel_mean"] > 0, "Positive wheel velocity"
    assert diag_pos["tau_wheel_raw"] < 0, "Should produce negative (braking) torque"

    assert diag_neg["wheel_vel_mean"] < 0, "Negative wheel velocity"
    assert diag_neg["tau_wheel_raw"] > 0, "Should produce positive (braking) torque"


def test_equilibrium_reference_offset(controller):
    """Equilibrium reference shifts the zero point."""
    controller.set_equilibrium_reference(
        pitch_x=0.1,
        com_y=-0.01,
        cp_y=-0.01,
    )

    # At equilibrium pitch
    tau_wheel_eq, diag_eq = controller.compute_wheel_torques(
        pitch_x=0.1,
        pitch_rate_x=0.0,
        com_y=-0.01,
        com_vy=0.0,
        cp_y=-0.01,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    # Above equilibrium
    tau_wheel_above, diag_above = controller.compute_wheel_torques(
        pitch_x=0.2,
        pitch_rate_x=0.0,
        com_y=-0.01,
        com_vy=0.0,
        cp_y=-0.01,
        wheel_vel_left=0.0,
        wheel_vel_right=0.0,
    )

    assert diag_eq["pitch_error"] == 0.0, "At equilibrium should have zero error"
    assert diag_eq["tau_wheel_raw"] == 0.0, "At equilibrium should produce zero torque"

    assert diag_above["pitch_error"] > 0.0, "Above equilibrium should have positive error"
    assert diag_above["tau_wheel_raw"] > 0.0, "Above equilibrium should produce positive torque"


def test_combined_state_feedback(controller, equilibrium_state):
    """Combined state feedback produces correct total torque."""
    controller.set_equilibrium_reference(**equilibrium_state)

    tau_wheel, diag = controller.compute_wheel_torques(
        pitch_x=0.05,
        pitch_rate_x=0.1,
        com_y=-0.01,
        com_vy=0.05,
        cp_y=-0.01,
        wheel_vel_left=2.0,
        wheel_vel_right=2.0,
    )

    # Verify individual terms are computed
    assert diag["term_pitch"] != 0.0, "Pitch term should be nonzero"
    assert diag["term_pitch_rate"] != 0.0, "Pitch rate term should be nonzero"
    assert diag["term_com_y"] != 0.0, "CoM Y term should be nonzero"
    assert diag["term_com_vy"] != 0.0, "CoM Y velocity term should be nonzero"
    assert diag["term_cp_y"] != 0.0, "CP Y term should be nonzero"
    assert diag["term_wheel_vel"] != 0.0, "Wheel velocity term should be nonzero"

    # Verify total torque is sum of terms (with wheel_vel negated)
    expected_tau = (
        diag["term_pitch"]
        + diag["term_pitch_rate"]
        + diag["term_com_y"]
        + diag["term_com_vy"]
        + diag["term_cp_y"]
        - diag["term_wheel_vel"]  # Negative sign in control law
    )

    assert abs(diag["tau_wheel_raw"] - expected_tau) < 1e-6, (
        f"Total torque should equal sum of terms, got {diag['tau_wheel_raw']}, expected {expected_tau}"
    )
