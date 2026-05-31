"""Unit tests for smart position hold capture gate.

Tests verify:
1. Sign convention: pitch direction maps to required capture direction
2. Conflict detection: tau_position opposing capture triggers gate
3. Gating behavior: gate factor transitions smoothly
4. Recovery detection: pitch reversal and capture recovery restore gate
5. Capture point calculation: inverted pendulum model
"""

import pytest
import numpy as np
from wheeled_biped.controllers.position_hold_capture_gate import (
    PositionHoldCaptureGate,
    CaptureGateDiagnostics,
)


class TestCaptureDirectionDetection:
    """Test required capture direction computation."""

    def test_forward_pitch_requires_forward_capture(self):
        """Positive pitch (forward lean) requires forward capture direction."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            enable_capture_point=False,  # Use pitch-based fallback
            warmup_steps=0,  # Disable warmup for testing
        )

        # Forward pitch above threshold
        pitch_x_rad = 0.10  # ~5.7 deg
        pitch_rate_x_rad_s = 0.0
        com_y_m = 0.0
        com_vy_m_s = 0.0
        support_center_y_m = 0.0
        com_z_m = 0.4

        required_dir, cp_rel, com_err = gate.compute_required_capture_direction(
            pitch_x_rad, pitch_rate_x_rad_s, com_y_m, com_vy_m_s, support_center_y_m, com_z_m
        )

        assert required_dir == 1.0, "Forward pitch should require forward capture"

    def test_backward_pitch_requires_backward_capture(self):
        """Negative pitch (backward lean) requires backward capture direction."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            enable_capture_point=False,
            warmup_steps=0,
        )

        pitch_x_rad = -0.10
        pitch_rate_x_rad_s = 0.0
        com_y_m = 0.0
        com_vy_m_s = 0.0
        support_center_y_m = 0.0
        com_z_m = 0.4

        required_dir, _, _ = gate.compute_required_capture_direction(
            pitch_x_rad, pitch_rate_x_rad_s, com_y_m, com_vy_m_s, support_center_y_m, com_z_m
        )

        assert required_dir == -1.0, "Backward pitch should require backward capture"

    def test_small_pitch_no_capture_needed(self):
        """Small pitch below threshold requires no capture."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            enable_capture_point=False,
            warmup_steps=0,
        )

        pitch_x_rad = 0.02  # Below threshold
        pitch_rate_x_rad_s = 0.0
        com_y_m = 0.0
        com_vy_m_s = 0.0
        support_center_y_m = 0.0
        com_z_m = 0.4

        required_dir, _, _ = gate.compute_required_capture_direction(
            pitch_x_rad, pitch_rate_x_rad_s, com_y_m, com_vy_m_s, support_center_y_m, com_z_m
        )

        assert required_dir == 0.0, "Small pitch should require no capture"

    def test_capture_point_ahead_requires_forward_capture(self):
        """Capture point ahead of support requires forward capture."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            enable_capture_point=True,
            warmup_steps=0,
        )

        # CoM ahead with forward velocity
        pitch_x_rad = 0.01  # Small pitch
        pitch_rate_x_rad_s = 0.0
        com_y_m = 0.15  # 15cm ahead
        com_vy_m_s = 0.2  # Forward velocity
        support_center_y_m = 0.0
        com_z_m = 0.4

        required_dir, cp_rel, _ = gate.compute_required_capture_direction(
            pitch_x_rad, pitch_rate_x_rad_s, com_y_m, com_vy_m_s, support_center_y_m, com_z_m
        )

        # Capture point = com + com_vel / omega
        # omega = sqrt(9.81 / 0.4) ~ 4.95
        # cp = 0.15 + 0.2 / 4.95 ~ 0.19 m (above 10cm threshold)
        assert cp_rel > 0.10, "Capture point should be ahead of support"
        assert required_dir == 1.0, "Capture point ahead should require forward capture"


class TestConflictDetection:
    """Test position-capture conflict detection."""

    def test_forward_capture_backward_tau_position_conflict(self):
        """Forward capture + backward tau_position = conflict."""
        gate = PositionHoldCaptureGate(warmup_steps=0)

        tau_position_raw = -5.0  # Backward torque
        required_capture_direction = 1.0  # Forward capture needed

        conflict = gate.detect_conflict(tau_position_raw, required_capture_direction)

        assert conflict is True, "Backward tau_position should conflict with forward capture"

    def test_forward_capture_forward_tau_position_no_conflict(self):
        """Forward capture + forward tau_position = no conflict."""
        gate = PositionHoldCaptureGate(warmup_steps=0)

        tau_position_raw = 5.0  # Forward torque
        required_capture_direction = 1.0  # Forward capture needed

        conflict = gate.detect_conflict(tau_position_raw, required_capture_direction)

        assert conflict is False, "Forward tau_position should not conflict with forward capture"

    def test_backward_capture_forward_tau_position_conflict(self):
        """Backward capture + forward tau_position = conflict."""
        gate = PositionHoldCaptureGate(warmup_steps=0)

        tau_position_raw = 5.0  # Forward torque
        required_capture_direction = -1.0  # Backward capture needed

        conflict = gate.detect_conflict(tau_position_raw, required_capture_direction)

        assert conflict is True, "Forward tau_position should conflict with backward capture"

    def test_no_capture_needed_no_conflict(self):
        """No capture needed = no conflict regardless of tau_position."""
        gate = PositionHoldCaptureGate(warmup_steps=0)

        tau_position_raw = -5.0
        required_capture_direction = 0.0  # No capture needed

        conflict = gate.detect_conflict(tau_position_raw, required_capture_direction)

        assert conflict is False, "No conflict when no capture is needed"


class TestGatingBehavior:
    """Test gate factor transitions and gating logic."""

    def test_conflict_reduces_gate_factor(self):
        """Conflict should reduce gate factor toward conflict value."""
        gate = PositionHoldCaptureGate(
            gate_factor_conflict=0.0,
            gate_factor_normal=1.0,
            smooth_ramp_steps=10,
            warmup_steps=0,
        )

        # Initial state: normal
        assert gate._gate_factor == 1.0

        # Trigger conflict
        for _ in range(20):  # More than smooth_ramp_steps
            gate.update_gate_factor(
                conflict_detected=True,
                pitch_reversal_detected=False,
                capture_recovery_detected=False,
            )

        assert gate._gate_factor == 0.0, "Gate factor should reach conflict value"

    def test_recovery_restores_gate_factor(self):
        """Recovery should restore gate factor toward normal value."""
        gate = PositionHoldCaptureGate(
            gate_factor_conflict=0.0,
            gate_factor_normal=1.0,
            smooth_ramp_steps=10,
            warmup_steps=0,
        )

        # Drive to conflict state
        for _ in range(20):
            gate.update_gate_factor(True, False, False)

        assert gate._gate_factor == 0.0

        # Trigger recovery
        for _ in range(20):
            gate.update_gate_factor(
                conflict_detected=False,
                pitch_reversal_detected=True,
                capture_recovery_detected=False,
            )

        assert gate._gate_factor == 1.0, "Gate factor should restore to normal"

    def test_smooth_ramp_transitions(self):
        """Gate factor should transition smoothly, not jump."""
        gate = PositionHoldCaptureGate(
            gate_factor_conflict=0.0,
            gate_factor_normal=1.0,
            smooth_ramp_steps=10,
            warmup_steps=0,
        )

        gate_factors = []
        for _ in range(15):
            factor = gate.update_gate_factor(True, False, False)
            gate_factors.append(factor)

        # Check monotonic decrease
        for i in range(len(gate_factors) - 1):
            assert gate_factors[i] >= gate_factors[i + 1], "Gate factor should decrease monotonically"

        # Check no jumps larger than ramp rate
        ramp_rate = 1.0 / 10
        for i in range(len(gate_factors) - 1):
            delta = abs(gate_factors[i] - gate_factors[i + 1])
            assert delta <= ramp_rate + 1e-6, f"Gate factor jump {delta} exceeds ramp rate {ramp_rate}"

    def test_apply_gate_reduces_tau_position_during_conflict(self):
        """apply_gate should reduce tau_position when conflict detected."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            gate_factor_conflict=0.0,
            smooth_ramp_steps=0,  # Instant transition for test
            enable_capture_point=False,
            warmup_steps=0,
        )

        tau_position_raw = -10.0  # Backward torque
        pitch_x_rad = 0.10  # Forward pitch (requires forward capture)
        pitch_rate_x_rad_s = -0.05  # Pitch rate not yet reversed
        com_y_m = 0.2  # Ahead of support
        com_vy_m_s = 0.05  # Forward velocity
        support_center_y_m = 0.0
        com_z_m = 0.4

        tau_gated, diag = gate.apply_gate(
            tau_position_raw,
            pitch_x_rad,
            pitch_rate_x_rad_s,
            com_y_m,
            com_vy_m_s,
            support_center_y_m,
            com_z_m,
        )

        assert diag.position_opposes_capture is True, "Conflict should be detected"
        assert diag.gate_active is True, "Gate should be active"
        assert abs(tau_gated) < abs(tau_position_raw), "Gated torque should be reduced"
        assert tau_gated == 0.0, "With gate_factor=0.0, gated torque should be zero"


class TestRecoveryDetection:
    """Test pitch reversal and capture recovery detection."""

    def test_pitch_reversal_detected_when_pitch_small_and_rate_low(self):
        """Pitch reversal detected when pitch and pitch_rate are small."""
        gate = PositionHoldCaptureGate(pitch_threshold_rad=0.05, warmup_steps=0)

        pitch_x_rad = 0.01  # Small pitch
        pitch_rate_x_rad_s = 0.05  # Low pitch rate
        cp_rel = 0.0

        pitch_rev, cap_rec = gate.detect_recovery(pitch_x_rad, pitch_rate_x_rad_s, cp_rel)

        assert pitch_rev is True, "Pitch reversal should be detected"

    def test_pitch_reversal_not_detected_when_pitch_large(self):
        """Pitch reversal not detected when pitch is large."""
        gate = PositionHoldCaptureGate(pitch_threshold_rad=0.05, warmup_steps=0)

        pitch_x_rad = 0.10  # Large pitch
        pitch_rate_x_rad_s = 0.05
        cp_rel = 0.0

        pitch_rev, _ = gate.detect_recovery(pitch_x_rad, pitch_rate_x_rad_s, cp_rel)

        assert pitch_rev is False, "Pitch reversal should not be detected with large pitch"

    def test_capture_recovery_detected_when_cp_near_support(self):
        """Capture recovery detected when capture point is near support."""
        gate = PositionHoldCaptureGate(warmup_steps=0)

        pitch_x_rad = 0.10
        pitch_rate_x_rad_s = 0.0
        cp_rel = 0.05  # 5cm from support (within 10cm threshold)

        _, cap_rec = gate.detect_recovery(pitch_x_rad, pitch_rate_x_rad_s, cp_rel)

        assert cap_rec is True, "Capture recovery should be detected"

    def test_capture_recovery_not_detected_when_cp_far(self):
        """Capture recovery not detected when capture point is far from support."""
        gate = PositionHoldCaptureGate(warmup_steps=0)

        pitch_x_rad = 0.10
        pitch_rate_x_rad_s = 0.0
        cp_rel = 0.15  # 15cm from support (beyond 10cm threshold)

        _, cap_rec = gate.detect_recovery(pitch_x_rad, pitch_rate_x_rad_s, cp_rel)

        assert cap_rec is False, "Capture recovery should not be detected when cp is far"


class TestIntegration:
    """Integration tests for full gate behavior."""

    def test_baseline_transient_scenario(self):
        """Test gate behavior in baseline transient scenario (step 1360)."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            gate_factor_conflict=0.0,
            smooth_ramp_steps=10,
            enable_capture_point=True,
            warmup_steps=0,
        )

        # Baseline transient peak conditions (from sign verification)
        tau_position_raw = -11.900  # Backward torque
        pitch_x_rad = 0.0985  # 5.64 deg forward
        pitch_rate_x_rad_s = -0.1011  # Reversing
        com_y_m = 0.595  # Ahead of equilibrium
        com_vy_m_s = 0.033  # Small forward velocity
        support_center_y_m = 0.0  # Assume equilibrium at origin
        com_z_m = 0.4

        tau_gated, diag = gate.apply_gate(
            tau_position_raw,
            pitch_x_rad,
            pitch_rate_x_rad_s,
            com_y_m,
            com_vy_m_s,
            support_center_y_m,
            com_z_m,
        )

        # Verify conflict detected
        assert diag.required_capture_direction == 1.0, "Forward capture required"
        assert diag.tau_position_direction == -1.0, "Backward tau_position"
        assert diag.position_opposes_capture is True, "Conflict should be detected"

        # Verify gating active
        assert diag.gate_active is True, "Gate should be active"
        assert abs(tau_gated) < abs(tau_position_raw), "Torque should be gated"

    def test_steady_state_no_gating(self):
        """Test gate remains inactive during steady state."""
        gate = PositionHoldCaptureGate(
            pitch_threshold_rad=0.05,
            gate_factor_conflict=0.0,
            smooth_ramp_steps=10,
            enable_capture_point=True,
            warmup_steps=0,
        )

        # Steady state: small pitch, small position error
        tau_position_raw = -1.0  # Small backward torque
        pitch_x_rad = 0.02  # Small pitch
        pitch_rate_x_rad_s = 0.01  # Small pitch rate
        com_y_m = 0.05  # Small position error
        com_vy_m_s = 0.0  # No velocity
        support_center_y_m = 0.0
        com_z_m = 0.4

        tau_gated, diag = gate.apply_gate(
            tau_position_raw,
            pitch_x_rad,
            pitch_rate_x_rad_s,
            com_y_m,
            com_vy_m_s,
            support_center_y_m,
            com_z_m,
        )

        # Verify no gating in steady state
        assert diag.required_capture_direction == 0.0, "No capture required"
        assert diag.position_opposes_capture is False, "No conflict"
        assert diag.gate_active is False, "Gate should be inactive"
        assert tau_gated == tau_position_raw, "Torque should not be gated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
