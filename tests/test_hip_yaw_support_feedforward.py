"""Test hip-yaw support-error feedforward (HY-FF) implementation.

Validates:
1. HY-FF disabled by default
2. Height gate continuous function
3. Compensation computation
4. Telemetry fields
5. No side effects on baseline
"""

import pytest
import jax.numpy as jnp

from wheeled_biped.controllers.shape_posture_controller import (
    ShapePostureController,
    compute_hip_yaw_support_feedforward_height_gate,
    BALANCE_CORE_HIP_YAW_AUTHORITY,
)
from wheeled_biped.controllers.balance_core_types import ACTION_DIM


class TestHipYawSupportFeedforward:
    """Test HY-FF hip-yaw support-error feedforward compensation."""

    def test_hy_ff_disabled_by_default(self):
        """HY-FF should be disabled by default."""
        controller = ShapePostureController()
        assert controller.enable_hip_yaw_support_feedforward is False
        assert controller.k_support_hip_yaw == 0.0

    def test_hy_ff_does_not_affect_baseline_when_disabled(self):
        """HY-FF disabled should produce identical output to baseline."""
        # Baseline controller
        baseline = ShapePostureController(
            kp_hip_yaw=15.0,
            kd_hip_yaw=3.0,
        )

        # HY-FF controller with disabled flag
        hy_ff_disabled = ShapePostureController(
            kp_hip_yaw=15.0,
            kd_hip_yaw=3.0,
            enable_hip_yaw_support_feedforward=False,
            k_support_hip_yaw=2.0,  # Should be ignored
            tau_max_support_comp=1.0,
            support_comp_sign=1.0,
        )

        # Test inputs
        q_ref = jnp.array([0.0, 0.0, 0.9, 1.7, 0.0, 0.0, 0.0, 0.9, 1.7, 0.0])
        joint_pos = jnp.array([0.0, 0.05, 0.85, 1.65, 0.0, 0.0, -0.05, 0.95, 1.75, 0.0])
        joint_vel = jnp.zeros(ACTION_DIM)

        tau_baseline, diag_baseline = baseline.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=0.1,  # Should be ignored
            target_com_height=0.30,
        )

        tau_disabled, diag_disabled = hy_ff_disabled.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=0.1,  # Should be ignored when disabled
            target_com_height=0.30,
        )

        # Torques should be identical
        assert jnp.allclose(tau_baseline, tau_disabled, atol=1e-6)

        # HY-FF diagnostics should show disabled
        assert diag_disabled["hip_yaw_comp_active"] is False
        assert diag_disabled["hip_yaw_comp_height_gate"] == 0.0
        assert diag_disabled["hip_yaw_comp_tau_left"] == 0.0
        assert diag_disabled["hip_yaw_comp_tau_right"] == 0.0

    def test_height_gate_continuous(self):
        """Height gate should be continuous with correct boundary values."""
        # Test boundary values
        gate_low = compute_hip_yaw_support_feedforward_height_gate(0.300)
        gate_mid = compute_hip_yaw_support_feedforward_height_gate(0.3465)  # midpoint
        gate_high = compute_hip_yaw_support_feedforward_height_gate(0.393)

        assert abs(float(gate_low) - 1.0) < 1e-6, f"z=0.300 should give gate≈1.0, got {gate_low}"
        assert 0.4 < float(gate_mid) < 0.6, f"z=0.3465 should give gate≈0.5, got {gate_mid}"
        assert abs(float(gate_high)) < 1e-6, f"z=0.393 should give gate≈0.0, got {gate_high}"

        # Test beyond boundaries
        gate_below = compute_hip_yaw_support_feedforward_height_gate(0.250)
        gate_above = compute_hip_yaw_support_feedforward_height_gate(0.450)

        assert abs(float(gate_below) - 1.0) < 1e-6, "Below z_low should saturate at 1.0"
        assert abs(float(gate_above)) < 1e-6, "Above z_high should saturate at 0.0"

        # Test continuity (derivative should be smooth)
        z_test = 0.35
        eps = 0.001
        gate_minus = compute_hip_yaw_support_feedforward_height_gate(z_test - eps)
        gate_plus = compute_hip_yaw_support_feedforward_height_gate(z_test + eps)
        gate_center = compute_hip_yaw_support_feedforward_height_gate(z_test)

        # Should vary smoothly
        assert abs(float(gate_plus - gate_minus)) < 0.1, "Gate should vary smoothly"

    def test_hy_ff_compensation_computation(self):
        """HY-FF compensation should compute correctly."""
        controller = ShapePostureController(
            kp_hip_yaw=15.0,
            kd_hip_yaw=3.0,
            enable_hip_yaw_support_feedforward=True,
            k_support_hip_yaw=4.0,
            tau_max_support_comp=2.0,
            support_comp_sign=1.0,
        )

        q_ref = jnp.array([0.0, 0.0, 0.9, 1.7, 0.0, 0.0, 0.0, 0.9, 1.7, 0.0])
        joint_pos = jnp.array([0.0, 0.05, 0.85, 1.65, 0.0, 0.0, -0.05, 0.95, 1.75, 0.0])
        joint_vel = jnp.zeros(ACTION_DIM)

        # Test at low height with support error
        support_error = 0.2  # 20 cm forward drift
        target_height = 0.30  # Low height, gate should be ~1.0

        tau, diag = controller.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=support_error,
            target_com_height=target_height,
        )

        # Verify HY-FF is active
        assert diag["hip_yaw_comp_active"] is True
        assert diag["hip_yaw_comp_height_gate"] > 0.9, "Gate should be ~1.0 at z=0.30"

        # Verify compensation magnitude
        expected_comp = 1.0 * 4.0 * 0.2  # sign * k * support_error at full gate
        assert abs(diag["hip_yaw_comp_tau_left"] - expected_comp) < 0.1
        assert abs(diag["hip_yaw_comp_tau_right"] + expected_comp) < 0.1  # Opposite sign

        # Verify left/right antisymmetry
        assert diag["hip_yaw_comp_tau_left"] == -diag["hip_yaw_comp_tau_right"]

    def test_hy_ff_compensation_clamping(self):
        """HY-FF compensation should clamp correctly."""
        controller = ShapePostureController(
            enable_hip_yaw_support_feedforward=True,
            k_support_hip_yaw=10.0,  # Large gain
            tau_max_support_comp=1.0,  # Small clamp
            support_comp_sign=1.0,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Large support error should saturate compensation
        large_support_error = 1.0  # 1 meter drift
        target_height = 0.30  # Full gate

        tau, diag = controller.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=large_support_error,
            target_com_height=target_height,
        )

        # Raw compensation should be large
        raw_comp = diag["hip_yaw_comp_tau_left"]
        assert abs(raw_comp) > 1.0, "Raw compensation should exceed clamp"

        # Final torque contribution should be clamped
        # (This is verified via tau[1] and tau[6] being within bounds)
        # Since tau includes PD term, we check clipping flags instead
        assert diag["hip_yaw_comp_tau_left_clipped"] is True
        assert diag["hip_yaw_comp_tau_right_clipped"] is True

    def test_hy_ff_uses_target_height_not_variant(self):
        """HY-FF should use target height value, not variant name."""
        controller = ShapePostureController(
            enable_hip_yaw_support_feedforward=True,
            k_support_hip_yaw=2.0,
            tau_max_support_comp=1.0,
            support_comp_sign=1.0,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Same target height should give same gate, regardless of "variant"
        _, diag_low = controller.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=0.1,
            target_com_height=0.30,  # Value, not variant name
        )

        _, diag_high = controller.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=0.1,
            target_com_height=0.45,  # Different height
        )

        # Low height should have high gate
        assert diag_low["hip_yaw_comp_height_gate"] > 0.9

        # High height should have zero gate
        assert diag_high["hip_yaw_comp_height_gate"] < 0.1

    def test_hy_ff_telemetry_fields_exist(self):
        """All HY-FF telemetry fields should be present."""
        controller = ShapePostureController(
            enable_hip_yaw_support_feedforward=True,
            k_support_hip_yaw=2.0,
            tau_max_support_comp=1.0,
            support_comp_sign=1.0,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        _, diag = controller.compute(
            q_ref=q_ref,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            support_position_error=0.1,
            target_com_height=0.35,
        )

        # All required telemetry fields must exist
        required_fields = [
            "hip_yaw_comp_active",
            "hip_yaw_comp_height_gate",
            "hip_yaw_comp_support_error_m",
            "hip_yaw_comp_tau_left",
            "hip_yaw_comp_tau_right",
            "hip_yaw_comp_tau_left_clipped",
            "hip_yaw_comp_tau_right_clipped",
            "hip_yaw_comp_sign",
            "hip_yaw_comp_k_support",
            "hip_yaw_comp_tau_max",
        ]

        for field in required_fields:
            assert field in diag, f"Missing telemetry field: {field}"

    def test_hy_ff_sign_parameter(self):
        """HY-FF sign parameter should affect compensation direction."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)
        support_error = 0.1
        target_height = 0.30

        # Sign +1.0
        controller_plus = ShapePostureController(
            enable_hip_yaw_support_feedforward=True,
            k_support_hip_yaw=2.0,
            support_comp_sign=1.0,
        )

        # Sign -1.0
        controller_minus = ShapePostureController(
            enable_hip_yaw_support_feedforward=True,
            k_support_hip_yaw=2.0,
            support_comp_sign=-1.0,
        )

        _, diag_plus = controller_plus.compute(
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel,
            support_position_error=support_error, target_com_height=target_height,
        )

        _, diag_minus = controller_minus.compute(
            q_ref=q_ref, joint_pos=joint_pos, joint_vel=joint_vel,
            support_position_error=support_error, target_com_height=target_height,
        )

        # Signs should be opposite
        assert diag_plus["hip_yaw_comp_tau_left"] == -diag_minus["hip_yaw_comp_tau_left"]
        assert diag_plus["hip_yaw_comp_tau_right"] == -diag_minus["hip_yaw_comp_tau_right"]

    def test_balance_core_authority_unchanged(self):
        """BALANCE_CORE_HIP_YAW_AUTHORITY should remain unchanged."""
        # This ensures we haven't globally modified hip-yaw gains
        assert BALANCE_CORE_HIP_YAW_AUTHORITY.kp_hip_yaw == 15.0
        assert BALANCE_CORE_HIP_YAW_AUTHORITY.kd_hip_yaw == 3.0
