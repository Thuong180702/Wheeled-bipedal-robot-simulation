"""Tests for pitch rate consistency estimator.

Verifies that the estimator correctly detects sign mismatches between measured
pitch_rate and finite-difference derivative, and substitutes FD rate when
inconsistent.
"""

import pytest
from wheeled_biped.controllers.pitch_rate_consistency_estimator import (
    PitchRateConsistencyEstimator,
)


class TestPitchRateConsistencyEstimator:
    """Test suite for PitchRateConsistencyEstimator."""

    def test_initialization(self):
        """Test estimator initialization with valid parameters."""
        estimator = PitchRateConsistencyEstimator(
            dt=0.01,
            min_rate_for_sign_check=0.01,
            filter_alpha=0.3,
        )
        assert estimator.dt == 0.01
        assert estimator.min_rate_for_sign_check == 0.01
        assert estimator.filter_alpha == 0.3
        assert estimator.prev_pitch_x is None
        assert estimator.prev_corrected_rate == 0.0

    def test_invalid_filter_alpha(self):
        """Test that invalid filter_alpha raises ValueError."""
        with pytest.raises(ValueError, match="filter_alpha must be in"):
            PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.0)
        with pytest.raises(ValueError, match="filter_alpha must be in"):
            PitchRateConsistencyEstimator(dt=0.01, filter_alpha=1.5)

    def test_first_step_uses_measured_rate(self):
        """Test that first step uses measured rate (no previous pitch)."""
        estimator = PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.3)

        result = estimator.estimate(pitch_x=0.05, pitch_rate_measured=0.5)

        # First step: no FD rate available, use measured
        assert result.pitch_rate_measured == 0.5
        assert result.pitch_rate_fd == 0.5  # FD falls back to measured
        assert result.sign_mismatch is False
        assert result.source_used == "measured"
        # Corrected rate is filtered: 0.3 * 0.0 (prev) + 0.7 * 0.5 (selected) = 0.35
        assert abs(result.pitch_rate_corrected - 0.35) < 1e-6

    def test_consistent_positive_rates(self):
        """Test that consistent positive rates use measured rate."""
        estimator = PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.3)

        # Step 1: establish previous pitch
        estimator.estimate(pitch_x=0.05, pitch_rate_measured=0.5)

        # Step 2: pitch increases, measured rate positive
        result = estimator.estimate(pitch_x=0.055, pitch_rate_measured=0.5)

        # FD rate = (0.055 - 0.05) / 0.01 = 0.5 rad/s
        assert abs(result.pitch_rate_fd - 0.5) < 1e-6
        assert result.pitch_rate_measured == 0.5
        assert result.sign_mismatch is False
        assert result.source_used == "measured"

    def test_consistent_negative_rates(self):
        """Test that consistent negative rates use measured rate."""
        estimator = PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.3)

        # Step 1: establish previous pitch
        estimator.estimate(pitch_x=0.05, pitch_rate_measured=-0.5)

        # Step 2: pitch decreases, measured rate negative
        result = estimator.estimate(pitch_x=0.045, pitch_rate_measured=-0.5)

        # FD rate = (0.045 - 0.05) / 0.01 = -0.5 rad/s
        assert abs(result.pitch_rate_fd - (-0.5)) < 1e-6
        assert result.pitch_rate_measured == -0.5
        assert result.sign_mismatch is False
        assert result.source_used == "measured"

    def test_sign_mismatch_detected(self):
        """Test that sign mismatch is detected and FD rate is used."""
        estimator = PitchRateConsistencyEstimator(
            dt=0.01,
            min_rate_for_sign_check=0.01,
            filter_alpha=0.3,
        )

        # Step 1: establish previous pitch
        estimator.estimate(pitch_x=0.05, pitch_rate_measured=0.5)

        # Step 2: pitch increases but measured rate is negative (artifact)
        result = estimator.estimate(pitch_x=0.055, pitch_rate_measured=-0.5)

        # FD rate = (0.055 - 0.05) / 0.01 = 0.5 rad/s (positive)
        # Measured rate = -0.5 rad/s (negative)
        # Sign mismatch detected
        assert abs(result.pitch_rate_fd - 0.5) < 1e-6
        assert result.pitch_rate_measured == -0.5
        assert result.sign_mismatch is True
        assert result.source_used == "finite_difference"
        # Corrected rate should be filtered FD rate
        # First correction: 0.3 * 0.5 (prev) + 0.7 * 0.5 (FD) = 0.5
        assert abs(result.pitch_rate_corrected - 0.5) < 0.1

    def test_near_zero_rates_no_sign_check(self):
        """Test that near-zero rates do not trigger sign mismatch."""
        estimator = PitchRateConsistencyEstimator(
            dt=0.01,
            min_rate_for_sign_check=0.01,
            filter_alpha=0.3,
        )

        # Step 1: establish previous pitch
        estimator.estimate(pitch_x=0.05, pitch_rate_measured=0.001)

        # Step 2: small pitch change, measured rate has opposite sign but below threshold
        result = estimator.estimate(pitch_x=0.0501, pitch_rate_measured=-0.001)

        # Both rates are below min_rate_for_sign_check (0.01 rad/s)
        # No sign mismatch should be flagged
        assert result.sign_mismatch is False
        assert result.source_used == "measured"

    def test_filtering_smooths_corrected_rate(self):
        """Test that low-pass filter smooths corrected rate."""
        estimator = PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.5)

        # Step 1: establish previous pitch
        result1 = estimator.estimate(pitch_x=0.0, pitch_rate_measured=0.0)
        assert abs(result1.pitch_rate_corrected - 0.0) < 1e-6

        # Step 2: sudden jump in measured rate
        result2 = estimator.estimate(pitch_x=0.01, pitch_rate_measured=1.0)
        # Corrected = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        assert abs(result2.pitch_rate_corrected - 0.5) < 1e-6

        # Step 3: rate stays at 1.0
        result3 = estimator.estimate(pitch_x=0.02, pitch_rate_measured=1.0)
        # Corrected = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        assert abs(result3.pitch_rate_corrected - 0.75) < 1e-6

    def test_reset_clears_state(self):
        """Test that reset() clears estimator state."""
        estimator = PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.3)

        # Run a few steps
        estimator.estimate(pitch_x=0.05, pitch_rate_measured=0.5)
        estimator.estimate(pitch_x=0.055, pitch_rate_measured=0.5)

        # Reset
        estimator.reset()

        assert estimator.prev_pitch_x is None
        assert estimator.prev_corrected_rate == 0.0

    def test_consistency_error_computed(self):
        """Test that consistency error is computed correctly."""
        estimator = PitchRateConsistencyEstimator(dt=0.01, filter_alpha=0.3)

        # Step 1: establish previous pitch
        estimator.estimate(pitch_x=0.05, pitch_rate_measured=0.5)

        # Step 2: measured rate differs from FD rate
        result = estimator.estimate(pitch_x=0.055, pitch_rate_measured=0.6)

        # FD rate = (0.055 - 0.05) / 0.01 = 0.5 rad/s
        # Measured rate = 0.6 rad/s
        # Consistency error = 0.6 - 0.5 = 0.1 rad/s
        assert abs(result.consistency_error - 0.1) < 1e-6

    def test_step_e_transient_scenario(self):
        """Test the Step E transient scenario: pitch increasing, rate flips negative."""
        estimator = PitchRateConsistencyEstimator(
            dt=0.005,  # 200 Hz control
            min_rate_for_sign_check=0.01,
            filter_alpha=0.3,
        )

        # Step 1235: pitch = 5.556 deg = 0.097 rad, rate = +0.0572 rad/s
        result1 = estimator.estimate(
            pitch_x=0.097,
            pitch_rate_measured=0.0572,
        )
        assert result1.sign_mismatch is False
        assert result1.source_used == "measured"

        # Step 1236: pitch = 5.576 deg = 0.0973 rad, rate = -0.0503 rad/s (artifact)
        result2 = estimator.estimate(
            pitch_x=0.0973,
            pitch_rate_measured=-0.0503,
        )

        # FD rate = (0.0973 - 0.097) / 0.005 = 0.6 rad/s (positive)
        # Measured rate = -0.0503 rad/s (negative)
        # Sign mismatch detected
        assert result2.sign_mismatch is True
        assert result2.source_used == "finite_difference"
        assert result2.pitch_rate_fd > 0.0
        assert result2.pitch_rate_measured < 0.0
        # Corrected rate should be positive (from FD)
        assert result2.pitch_rate_corrected > 0.0


def test_estimator_integration_with_controller():
    """Integration test: estimator prevents damping sign flip."""
    estimator = PitchRateConsistencyEstimator(dt=0.005, filter_alpha=0.3)

    # Simulate Step E transient scenario
    # Step 1235: normal operation
    result1 = estimator.estimate(pitch_x=0.097, pitch_rate_measured=0.0572)
    kd_pitch = 10.0
    tau_pitch_rate_1 = kd_pitch * result1.pitch_rate_corrected
    assert tau_pitch_rate_1 > 0.0  # Positive damping torque

    # Step 1236: artifact causes measured rate to flip
    result2 = estimator.estimate(pitch_x=0.0973, pitch_rate_measured=-0.0503)
    tau_pitch_rate_2 = kd_pitch * result2.pitch_rate_corrected

    # Without correction: tau_pitch_rate = 10.0 * (-0.0503) = -0.503 Nm (wrong sign)
    # With correction: tau_pitch_rate should remain positive
    assert tau_pitch_rate_2 > 0.0  # Corrected damping torque maintains correct sign
    assert result2.sign_mismatch is True
    assert result2.source_used == "finite_difference"
