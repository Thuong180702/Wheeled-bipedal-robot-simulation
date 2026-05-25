"""Tests for StaticFeedforwardController (Stage 2B gravity compensation).

Validates the Phase C configuration:
- Sign: +empirical (positive)
- Scale: 0.5
- Joint group: knee [3, 8]
- Ramp: instant
- Effective feedforward: -7.75, -7.90 Nm
"""

import numpy as np
import pytest

from wheeled_biped.controllers.static_feedforward_controller import (
    StaticFeedforwardController,
    KNEE_INDICES,
    HIP_PITCH_INDICES,
    HIP_PITCH_KNEE_INDICES,
    JOINT_GROUP_MAP,
)


@pytest.fixture
def empirical_feedforward():
    """Empirical feedforward from Phase B/C validation (from gain sweep telemetry)."""
    return np.array([
        -0.1,  # l_hip_roll
        0.0,   # l_hip_yaw
        4.1,   # l_hip_pitch
        -15.5, # l_knee
        0.0,   # l_wheel
        0.1,   # r_hip_roll
        -0.0,  # r_hip_yaw
        3.2,   # r_hip_pitch
        -15.8, # r_knee
        0.0,   # r_wheel
    ])


class TestStaticFeedforwardControllerDefaults:
    """Test default validated configuration from Phase C."""

    def test_default_config_applies_torque_only_to_knees(self, empirical_feedforward):
        """Default validated config applies torque only to knees [3,8]."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            # Defaults: scale=0.5, joint_group='knee', ramp_mode='instant', sign='positive'
        )

        tau_ff = controller.compute_feedforward()

        # Check only knee joints have non-zero torque
        assert tau_ff[3] != 0.0, "Left knee should have feedforward torque"
        assert tau_ff[8] != 0.0, "Right knee should have feedforward torque"

        # Check all other joints are zero
        non_knee_indices = [0, 1, 2, 4, 5, 6, 7, 9]
        for i in non_knee_indices:
            assert tau_ff[i] == 0.0, f"Joint {i} should have zero feedforward torque"

    def test_effective_knee_torques_match_validated_values(self, empirical_feedforward):
        """Effective knee torques match approximately -7.75 and -7.90 Nm."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            scale=0.5,
            joint_group='knee',
            ramp_mode='instant',
            sign='positive',
        )

        tau_ff = controller.compute_feedforward()

        # Expected: scale=0.5 * empirical_ff[3] = 0.5 * (-15.5) = -7.75
        # Expected: scale=0.5 * empirical_ff[8] = 0.5 * (-15.8) = -7.90
        assert np.isclose(tau_ff[3], -7.75, atol=0.01), f"Left knee torque should be -7.75 Nm, got {tau_ff[3]}"
        assert np.isclose(tau_ff[8], -7.90, atol=0.01), f"Right knee torque should be -7.90 Nm, got {tau_ff[8]}"

    def test_wheel_torques_remain_zero(self, empirical_feedforward):
        """Wheel torques remain zero with default knee-only configuration."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
        )

        tau_ff = controller.compute_feedforward()

        assert tau_ff[4] == 0.0, "Left wheel should have zero feedforward torque"
        assert tau_ff[9] == 0.0, "Right wheel should have zero feedforward torque"

    def test_ramp_instant_gives_full_torque_immediately(self, empirical_feedforward):
        """Ramp=instant gives full torque immediately at step 0."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            ramp_mode='instant',
        )

        tau_ff_step0 = controller.compute_feedforward(step=0)
        tau_ff_step1 = controller.compute_feedforward(step=1)

        # Both should be full torque
        assert np.allclose(tau_ff_step0, tau_ff_step1), "Instant ramp should give same torque at step 0 and 1"
        assert np.isclose(tau_ff_step0[3], -7.75, atol=0.01), "Step 0 should have full torque"


class TestStaticFeedforwardControllerScale:
    """Test scale parameter behavior."""

    def test_scale_changes_torque_magnitude_correctly(self, empirical_feedforward):
        """Scale changes torque magnitude correctly."""
        controller_scale_025 = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            scale=0.25,
            joint_group='knee',
        )
        controller_scale_050 = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            scale=0.50,
            joint_group='knee',
        )
        controller_scale_100 = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            scale=1.00,
            joint_group='knee',
        )

        tau_025 = controller_scale_025.compute_feedforward()
        tau_050 = controller_scale_050.compute_feedforward()
        tau_100 = controller_scale_100.compute_feedforward()

        # Check scaling relationship
        assert np.isclose(tau_025[3], -15.5 * 0.25, atol=0.01)
        assert np.isclose(tau_050[3], -15.5 * 0.50, atol=0.01)
        assert np.isclose(tau_100[3], -15.5 * 1.00, atol=0.01)

        # Check that scale=0.5 is exactly double scale=0.25
        assert np.allclose(tau_050, tau_025 * 2.0, atol=0.01)


class TestStaticFeedforwardControllerSign:
    """Test sign parameter behavior."""

    def test_sign_inversion_changes_sign_correctly(self, empirical_feedforward):
        """Sign inversion changes sign correctly."""
        controller_positive = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            sign='positive',
            joint_group='knee',
        )
        controller_negative = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            sign='negative',
            joint_group='knee',
        )

        tau_positive = controller_positive.compute_feedforward()
        tau_negative = controller_negative.compute_feedforward()

        # Negative should be exact opposite of positive
        assert np.allclose(tau_negative, -tau_positive, atol=1e-6)

    def test_positive_sign_is_validated_default(self, empirical_feedforward):
        """Positive sign is the validated default, not negative."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            # Default sign='positive'
        )

        assert controller.sign == 'positive', "Default sign should be 'positive' (validated)"


class TestStaticFeedforwardControllerJointGroups:
    """Test joint group parameter behavior."""

    def test_knee_joint_group(self, empirical_feedforward):
        """Knee joint group applies torque only to knees [3, 8]."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            joint_group='knee',
        )

        tau_ff = controller.compute_feedforward()

        # Only knees should have torque
        assert tau_ff[3] != 0.0
        assert tau_ff[8] != 0.0
        assert np.sum(np.abs(tau_ff[[0, 1, 2, 4, 5, 6, 7, 9]])) == 0.0

    def test_hip_pitch_joint_group(self, empirical_feedforward):
        """Hip pitch joint group applies torque only to hip pitch [2, 7]."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            joint_group='hip_pitch',
        )

        tau_ff = controller.compute_feedforward()

        # Only hip pitch should have torque
        assert tau_ff[2] != 0.0
        assert tau_ff[7] != 0.0
        assert np.sum(np.abs(tau_ff[[0, 1, 3, 4, 5, 6, 8, 9]])) == 0.0

    def test_hip_pitch_knee_joint_group(self, empirical_feedforward):
        """Hip pitch+knee joint group applies torque to [2, 3, 7, 8]."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            joint_group='hip_pitch_knee',
        )

        tau_ff = controller.compute_feedforward()

        # Hip pitch and knee should have torque
        assert tau_ff[2] != 0.0
        assert tau_ff[3] != 0.0
        assert tau_ff[7] != 0.0
        assert tau_ff[8] != 0.0
        assert np.sum(np.abs(tau_ff[[0, 1, 4, 5, 6, 9]])) == 0.0


class TestStaticFeedforwardControllerRamp:
    """Test ramp mode behavior."""

    def test_instant_ramp_full_torque_at_step_zero(self, empirical_feedforward):
        """Instant ramp gives full torque at step 0."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            ramp_mode='instant',
        )

        tau_ff = controller.compute_feedforward(step=0)
        expected_full = empirical_feedforward[3] * 0.5  # scale=0.5 default

        assert np.isclose(tau_ff[3], expected_full, atol=0.01)

    def test_short_ramp_gradual_increase(self, empirical_feedforward):
        """Short ramp (5 steps) gradually increases torque."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            ramp_mode='short',
        )

        tau_step0 = controller.compute_feedforward(step=0)
        tau_step2 = controller.compute_feedforward(step=2)
        tau_step5 = controller.compute_feedforward(step=5)

        # Step 0 should be zero (0/5 = 0.0)
        assert np.isclose(tau_step0[3], 0.0, atol=0.01)

        # Step 2 should be 40% (2/5 = 0.4)
        expected_step2 = empirical_feedforward[3] * 0.5 * 0.4
        assert np.isclose(tau_step2[3], expected_step2, atol=0.01)

        # Step 5 should be full (5/5 = 1.0)
        expected_full = empirical_feedforward[3] * 0.5
        assert np.isclose(tau_step5[3], expected_full, atol=0.01)

    def test_medium_ramp_gradual_increase(self, empirical_feedforward):
        """Medium ramp (10 steps) gradually increases torque."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            ramp_mode='medium',
        )

        tau_step0 = controller.compute_feedforward(step=0)
        tau_step5 = controller.compute_feedforward(step=5)
        tau_step10 = controller.compute_feedforward(step=10)

        # Step 0 should be zero
        assert np.isclose(tau_step0[3], 0.0, atol=0.01)

        # Step 5 should be 50% (5/10 = 0.5)
        expected_step5 = empirical_feedforward[3] * 0.5 * 0.5
        assert np.isclose(tau_step5[3], expected_step5, atol=0.01)

        # Step 10 should be full
        expected_full = empirical_feedforward[3] * 0.5
        assert np.isclose(tau_step10[3], expected_full, atol=0.01)


class TestStaticFeedforwardControllerTelemetry:
    """Test telemetry fields are populated correctly."""

    def test_telemetry_fields_populated(self, empirical_feedforward):
        """Telemetry fields are populated correctly."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            scale=0.5,
            joint_group='knee',
            ramp_mode='instant',
            sign='positive',
        )

        # Compute feedforward to advance step counter
        controller.compute_feedforward()

        telemetry = controller.get_telemetry()

        # Check all required fields exist
        assert 'tau_feedforward_per_joint' in telemetry
        assert 'tau_feedforward_norm' in telemetry
        assert 'feedforward_ramp' in telemetry
        assert 'feedforward_joint_group' in telemetry
        assert 'feedforward_scale' in telemetry
        assert 'feedforward_sign' in telemetry
        assert 'feedforward_step' in telemetry
        assert 'feedforward_ramp_factor' in telemetry

        # Check values
        assert telemetry['feedforward_ramp'] == 'instant'
        assert telemetry['feedforward_joint_group'] == 'knee'
        assert telemetry['feedforward_scale'] == 0.5
        assert telemetry['feedforward_sign'] == 'positive'
        assert telemetry['feedforward_step'] == 0
        assert telemetry['feedforward_ramp_factor'] == 1.0

        # Check tau_feedforward_per_joint is a list of 10 elements
        assert len(telemetry['tau_feedforward_per_joint']) == 10

        # Check norm is positive
        assert telemetry['tau_feedforward_norm'] > 0.0


class TestStaticFeedforwardControllerValidation:
    """Test input validation."""

    def test_invalid_empirical_feedforward_shape(self):
        """Invalid empirical feedforward shape raises ValueError."""
        with pytest.raises(ValueError, match="must be shape \\(10,\\)"):
            StaticFeedforwardController(
                empirical_feedforward=np.zeros(5),  # Wrong shape
            )

    def test_invalid_joint_group(self, empirical_feedforward):
        """Invalid joint group raises ValueError."""
        with pytest.raises(ValueError, match="joint_group must be one of"):
            StaticFeedforwardController(
                empirical_feedforward=empirical_feedforward,
                joint_group='invalid',
            )

    def test_invalid_ramp_mode(self, empirical_feedforward):
        """Invalid ramp mode raises ValueError."""
        with pytest.raises(ValueError, match="ramp_mode must be one of"):
            StaticFeedforwardController(
                empirical_feedforward=empirical_feedforward,
                ramp_mode='invalid',
            )

    def test_invalid_sign(self, empirical_feedforward):
        """Invalid sign raises ValueError."""
        with pytest.raises(ValueError, match="sign must be one of"):
            StaticFeedforwardController(
                empirical_feedforward=empirical_feedforward,
                sign='invalid',
            )


class TestStaticFeedforwardControllerReset:
    """Test reset functionality."""

    def test_reset_clears_step_counter(self, empirical_feedforward):
        """Reset clears internal step counter."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
        )

        # Advance step counter
        controller.compute_feedforward()
        controller.compute_feedforward()
        controller.compute_feedforward()

        assert controller.current_step == 3

        # Reset
        controller.reset()

        assert controller.current_step == 0


class TestPhaseCWrongRecommendationNotDefault:
    """Test that the old wrong recommendation from Phase C is not the default."""

    def test_default_is_not_negative_empirical_scale_025(self, empirical_feedforward):
        """Default configuration is NOT the wrong -empirical scale=0.25 recommendation."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
            # All defaults
        )

        # Verify defaults are the validated configuration, not the wrong one
        assert controller.sign == 'positive', "Default should be 'positive', not 'negative'"
        assert controller.scale == 0.5, "Default should be 0.5, not 0.25"
        assert controller.joint_group == 'knee', "Default should be 'knee'"
        assert controller.ramp_mode == 'instant', "Default should be 'instant'"

    def test_validated_config_matches_phase_c_best(self, empirical_feedforward):
        """Validated default configuration matches Phase C best configuration."""
        controller = StaticFeedforwardController(
            empirical_feedforward=empirical_feedforward,
        )

        # Phase C best: +empirical, scale=0.5, knee, instant
        assert controller.sign == 'positive'
        assert controller.scale == 0.5
        assert controller.joint_group == 'knee'
        assert controller.ramp_mode == 'instant'
