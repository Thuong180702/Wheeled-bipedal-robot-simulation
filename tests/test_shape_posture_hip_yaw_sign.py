"""Tests for hip-yaw sign correctness in shape posture controller.

These tests verify that the hip-yaw PD controller applies torque in the correct
direction to oppose position errors. The joint axes are NOT inverted - the
negation was an error that has been fixed.
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.balance_core_types import ACTION_DIM


class TestHipYawSignCorrectness:
    """Test hip-yaw torque sign correctness after sign fix.

    The joint axes are NOT inverted. Standard PD control applies:
    - Positive error (pos < ref) -> positive torque (to increase position)
    - Negative error (pos > ref) -> negative torque (to decrease position)
    """

    @pytest.fixture
    def controller(self):
        """Create shape posture controller with standard gains."""
        return ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            kp_hip_pitch=30.0,
            kd_hip_pitch=4.0,
            kp_knee=40.0,
            kd_knee=5.0,
        )

    def test_left_hip_yaw_positive_error_applies_correct_torque(self, controller):
        """Left hip-yaw positive error (pos < ref) should apply positive torque.

        - error = ref - pos = 0 - (-0.1) = +0.1 (positive)
        - positive error means position is below reference
        - standard PD: positive torque increases position
        - expect tau > 0
        """
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(-0.1)  # pos < ref, error = +0.1

        tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

        assert tau[1] > 0.0, f"Left hip-yaw: positive error should apply positive torque, got {tau[1]}"

    def test_left_hip_yaw_negative_error_applies_correct_torque(self, controller):
        """Left hip-yaw negative error (pos > ref) should apply negative torque.

        - error = ref - pos = 0 - 0.1 = -0.1 (negative)
        - negative error means position is above reference
        - standard PD: negative torque decreases position
        - expect tau < 0
        """
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(0.1)  # pos > ref, error = -0.1

        tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

        assert tau[1] < 0.0, f"Left hip-yaw: negative error should apply negative torque, got {tau[1]}"

    def test_right_hip_yaw_positive_error_applies_correct_torque(self, controller):
        """Right hip-yaw positive error should apply positive torque."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[6].set(-0.1)  # error = +0.1

        tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

        assert tau[6] > 0.0, f"Right hip-yaw: positive error should apply positive torque, got {tau[6]}"

    def test_right_hip_yaw_negative_error_applies_correct_torque(self, controller):
        """Right hip-yaw negative error should apply negative torque."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[6].set(0.1)  # error = -0.1

        tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

        assert tau[6] < 0.0, f"Right hip-yaw: negative error should apply negative torque, got {tau[6]}"

    def test_hip_yaw_damping_opposes_velocity(self, controller):
        """Hip-yaw damping torque should oppose joint velocity.

        Standard PD: tau_damping = -kd * velocity
        - Positive velocity -> negative damping torque (slows positive motion)
        - Negative velocity -> positive damping torque (slows negative motion)
        """
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Left hip-yaw with positive velocity
        joint_vel = joint_vel.at[1].set(1.0)

        tau_pos_vel, _ = controller.compute(q_ref, joint_pos, joint_vel)

        assert tau_pos_vel[1] < 0.0, "Positive velocity should result in negative damping torque"

        # Right hip-yaw with negative velocity
        joint_vel = jnp.zeros(ACTION_DIM).at[6].set(-1.0)

        tau_neg_vel, _ = controller.compute(q_ref, joint_pos, joint_vel)

        assert tau_neg_vel[6] > 0.0, "Negative velocity should result in positive damping torque"

    def test_hip_yaw_torque_proportional_to_error(self, controller):
        """Hip-yaw torque magnitude should be proportional to error magnitude."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Small error
        joint_pos_small = joint_pos.at[1].set(-0.05)
        tau_small, _ = controller.compute(q_ref, joint_pos_small, joint_vel)

        # Large error
        joint_pos_large = joint_pos.at[1].set(-0.20)
        tau_large, _ = controller.compute(q_ref, joint_pos_large, joint_vel)

        # Larger error should produce larger torque (same sign)
        assert abs(tau_large[1]) > abs(tau_small[1]), \
            f"Larger error should produce larger torque: small={tau_small[1]}, large={tau_large[1]}"
        assert jnp.sign(tau_large[1]) == jnp.sign(tau_small[1]), \
            "Both errors have same sign, torques should have same sign"

    def test_hip_roll_hip_pitch_knee_unchanged(self, controller):
        """Hip-roll, hip-pitch, and knee joints should be unaffected by hip-yaw sign fix."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Set errors for hip-pitch and knee
        joint_pos = joint_pos.at[2].set(-0.1)  # left hip-pitch
        joint_pos = joint_pos.at[3].set(-0.1)  # left knee

        tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

        # Hip-pitch and knee should use standard PD (no negation)
        # Positive error (pos < ref) should produce positive torque
        assert tau[2] > 0.0, "Left hip-pitch should use standard PD (no negation)"
        assert tau[3] > 0.0, "Left knee should use standard PD (no negation)"

        # Hip-roll should be zero (not controlled by shape posture)
        assert tau[0] == 0.0, "Left hip-roll should remain zero"
        assert tau[5] == 0.0, "Right hip-roll should remain zero"

    def test_no_wbc_interference(self, controller):
        """Shape posture controller should not depend on WBC."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM).at[1].set(-0.1)
        joint_vel = jnp.zeros(ACTION_DIM)

        tau, diagnostics = controller.compute(q_ref, joint_pos, joint_vel)

        # Verify torque is computed (not zero)
        assert tau[1] != 0.0, "Controller should produce nonzero torque"

        # Verify no WBC keys in diagnostics (shape posture is standalone)
        assert "wbc" not in str(diagnostics).lower(), "Shape posture should not reference WBC"

    def test_left_right_hip_yaw_independent(self, controller):
        """Left and right hip-yaw should be controlled independently."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Left hip-yaw with positive error, right with negative error
        joint_pos = joint_pos.at[1].set(-0.1)  # left: error = +0.1
        joint_pos = joint_pos.at[6].set(0.1)   # right: error = -0.1

        tau, _ = controller.compute(q_ref, joint_pos, joint_vel)

        # Standard PD:
        # Left with positive error should get POSITIVE torque
        # Right with negative error should get NEGATIVE torque
        assert tau[1] > 0.0, "Left hip-yaw should apply positive torque for positive error"
        assert tau[6] < 0.0, "Right hip-yaw should apply negative torque for negative error"
        # Equal magnitude errors should produce equal magnitude torques (opposite signs)
        assert abs(abs(tau[1]) - abs(tau[6])) < 0.01, "Equal errors should produce equal magnitude torques"
