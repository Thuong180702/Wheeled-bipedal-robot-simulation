"""Tests for hip-yaw divergence damping (HY2-DIV) in shape posture controller.

These tests verify that:
1. HY2-DIV is disabled by default
2. HY2-DIV uses z_ref for height gate (not variant name)
3. Height gate is continuous and correct
4. HY2-DIV produces correct antisymmetric torque
5. HY2-DIV opposes divergence correctly
6. Torque clamp works
7. No hip-roll output changes
8. No WBC path changes
9. Telemetry fields exist
"""

import jax.numpy as jnp
import pytest

from wheeled_biped.controllers.shape_posture_controller import (
    ShapePostureController,
    HipYawDivergenceProfile,
    HY2_DIV_BASELINE,
    HY2_DIV_A0,
    HY2_DIV_A1,
    HY2_DIV_A2,
    HY2_DIV_A3,
    HY2_DIV_B1,
    HY2_DIV_B2,
    HY2_DIV_B3,
    HY2_DIV_C1,
    ALL_HY2_DIV_CANDIDATES,
)
from wheeled_biped.controllers.balance_core_types import ACTION_DIM


class TestHipYawDivergenceDampingBasics:
    """Test HY2-DIV is disabled by default and configurable."""

    def test_hy2_div_disabled_by_default(self):
        """HY2-DIV should be disabled when no divergence damping parameters set."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
        )
        assert controller.enable_hip_yaw_divergence_damping == False
        assert controller.k_divergence == 0.0
        assert controller.k_divergence_rate == 0.0

    def test_hy2_div_can_be_enabled(self):
        """HY2-DIV can be enabled with divergence damping parameters."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            tau_max_divergence=0.5,
        )
        assert controller.enable_hip_yaw_divergence_damping == True
        assert controller.k_divergence == 5.0
        assert controller.k_divergence_rate == 1.0
        assert controller.tau_max_divergence == 0.5

    def test_hy2_div_gate_z_params_passed(self):
        """HY2-DIV z_low/z_high gate parameters should be stored correctly."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            tau_max_divergence=1.0,
            divergence_gate_z_low=0.300,
            divergence_gate_z_high=0.500,
        )
        assert controller.divergence_gate_z_low == 0.300
        assert controller.divergence_gate_z_high == 0.500

    def test_enabled_vs_gate_active_separate(self):
        """HY2-DIV enabled vs gate_active should be separate concepts."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            divergence_gate_z_low=0.300,
            divergence_gate_z_high=0.393,
        )
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)
        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        # At nominal height (z=0.404), A0 gate should be inactive
        _, diag_nominal = controller.compute(q_ref, joint_pos, joint_vel, target_com_height=0.404)
        assert diag_nominal["hip_yaw_div_enabled"] == True
        assert diag_nominal["hip_yaw_div_gate_active"] == False
        assert diag_nominal["hip_yaw_div_height_gate"] < 0.01

        # At low height (z=0.300), A0 gate should be active
        _, diag_low = controller.compute(q_ref, joint_pos, joint_vel, target_com_height=0.300)
        assert diag_low["hip_yaw_div_enabled"] == True
        assert diag_low["hip_yaw_div_gate_active"] == True
        assert diag_low["hip_yaw_div_height_gate"] > 0.99

    def test_b1_gate_active_at_nominal_height(self):
        """B1 with z_high=0.500 should be gate_active at nominal height (z=0.404)."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            divergence_gate_z_low=0.300,
            divergence_gate_z_high=0.500,
        )
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)
        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        _, diag = controller.compute(q_ref, joint_pos, joint_vel, target_com_height=0.404)
        assert diag["hip_yaw_div_enabled"] == True
        assert diag["hip_yaw_div_gate_active"] == True
        assert diag["hip_yaw_div_height_gate"] > 0.4

    def test_effective_k_follows_gate(self):
        """HY2-DIV effective_k/effective_kd should follow gate value."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
        )
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)
        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        # At low height (gate=1.0), effective_k = k * gate = 5.0 * 1.0 = 5.0
        _, diag_low = controller.compute(q_ref, joint_pos, joint_vel, target_com_height=0.300)
        assert abs(diag_low["hip_yaw_div_effective_k"] - 5.0) < 0.01
        assert abs(diag_low["hip_yaw_div_effective_kd"] - 1.0) < 0.01

        # At nominal height (gate=0.0), effective_k = 0
        _, diag_nominal = controller.compute(q_ref, joint_pos, joint_vel, target_com_height=0.404)
        assert abs(diag_nominal["hip_yaw_div_effective_k"]) < 0.01
        assert abs(diag_nominal["hip_yaw_div_effective_kd"]) < 0.01
    """Test HipYawDivergenceProfile dataclass and height gate."""

    def test_hy2_div_baseline_profile(self):
        """HY2_DIV_BASELINE should have correct conservative gains."""
        assert HY2_DIV_BASELINE.name == "hy2_div_baseline"
        assert HY2_DIV_BASELINE.k_divergence == 5.0
        assert HY2_DIV_BASELINE.k_divergence_rate == 1.0
        assert HY2_DIV_BASELINE.tau_max_divergence == 0.5
        assert HY2_DIV_BASELINE.z_low == 0.300
        assert HY2_DIV_BASELINE.z_high == 0.393

    def test_height_gate_at_z_low(self):
        """At z = z_low (0.300), gate should be 1.0 (fully active)."""
        gate = HY2_DIV_BASELINE.gate(0.300)
        assert abs(gate - 1.0) < 0.01, f"At z=0.300, gate should be 1.0, got {gate}"

    def test_height_gate_at_z_high(self):
        """At z = z_high (0.393), gate should be ~0.0 (inactive)."""
        gate = HY2_DIV_BASELINE.gate(0.393)
        assert abs(gate) < 0.01, f"At z=0.393, gate should be ~0, got {gate}"

    def test_height_gate_above_z_high(self):
        """Above z_high, gate should be 0.0."""
        gate = HY2_DIV_BASELINE.gate(0.45)
        assert gate == 0.0, f"Above z_high, gate should be 0.0, got {gate}"

    def test_height_gate_continuous(self):
        """Height gate should be continuous (no jumps)."""
        z_values = [0.30, 0.32, 0.35, 0.37, 0.39, 0.40, 0.45]
        gates = [float(HY2_DIV_BASELINE.gate(z)) for z in z_values]

        # Gate should monotonically decrease
        for i in range(len(gates) - 1):
            assert gates[i] >= gates[i+1] - 1e-6, \
                f"Gate should monotonically decrease: z={z_values[i]}->{z_values[i+1]}: {gates[i]}->{gates[i+1]}"

        # Gate should be bounded [0, 1]
        for g in gates:
            assert 0.0 <= g <= 1.0, f"Gate should be in [0, 1], got {g}"


class TestHipYawDivergenceTorqueSign:
    """Test HY2-DIV produces correct antisymmetric torque to oppose divergence."""

    @pytest.fixture
    def controller(self):
        """Create controller with HY2-DIV enabled."""
        return ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            tau_max_divergence=0.5,
        )

    def test_positive_divergence_gets_correcting_torque(self, controller):
        """When left ahead of right (positive divergence), HY2-DIV should apply corrective torques.

        Positive divergence: l_error > 0, r_error < 0
        Corrective torque: tau_div_left < 0 (slow left), tau_div_right > 0 (speed right)
        """
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Set up positive divergence: left ahead (l_error = +0.1), right behind (r_error = -0.1)
        joint_pos = joint_pos.at[1].set(-0.1)  # l_pos > ref, l_error = +0.1 (behind ref but ahead of r)
        joint_pos = joint_pos.at[6].set(0.1)   # r_pos < ref, r_error = -0.1 (ahead of ref but behind l)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30  # Active height gate
        )

        # HY2-DIV should apply correcting torques
        div_tau_left = diagnostics["hip_yaw_div_left"]
        div_tau_right = diagnostics["hip_yaw_div_right"]

        # For positive divergence: tau_div_left should be negative, tau_div_right positive
        assert div_tau_left < 0, f"Positive divergence: left should get negative torque, got {div_tau_left}"
        assert div_tau_right > 0, f"Positive divergence: right should get positive torque, got {div_tau_right}"

    def test_negative_divergence_gets_correcting_torque(self, controller):
        """When left behind right (negative divergence), HY2-DIV should apply corrective torques.

        Negative divergence: l_error < 0, r_error > 0
        Corrective torque: tau_div_left > 0 (speed left), tau_div_right < 0 (slow right)
        """
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Set up negative divergence: left behind (l_error = -0.1), right ahead (r_error = +0.1)
        joint_pos = joint_pos.at[1].set(0.1)   # l_pos > ref, l_error = -0.1 (ahead of ref but behind r)
        joint_pos = joint_pos.at[6].set(-0.1)  # r_pos < ref, r_error = +0.1 (behind ref but ahead of l)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30  # Active height gate
        )

        div_tau_left = diagnostics["hip_yaw_div_left"]
        div_tau_right = diagnostics["hip_yaw_div_right"]

        # For negative divergence: tau_div_left should be positive, tau_div_right negative
        assert div_tau_left > 0, f"Negative divergence: left should get positive torque, got {div_tau_left}"
        assert div_tau_right < 0, f"Negative divergence: right should get negative torque, got {div_tau_right}"

    def test_zero_divergence_produces_zero_div_torque(self, controller):
        """When divergence is zero, HY2-DIV should produce zero torque."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Both joints at reference (no divergence)
        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30
        )

        div_tau_left = diagnostics["hip_yaw_div_left"]
        div_tau_right = diagnostics["hip_yaw_div_right"]

        assert abs(div_tau_left) < 1e-6, f"Zero divergence: left torque should be ~0, got {div_tau_left}"
        assert abs(div_tau_right) < 1e-6, f"Zero divergence: right torque should be ~0, got {div_tau_right}"


class TestHipYawDivergenceHeightGate:
    """Test HY2-DIV height gate behavior."""

    @pytest.fixture
    def controller(self):
        return ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            tau_max_divergence=0.5,
        )

    def test_height_gate_active_at_low_height(self, controller):
        """HY2-DIV should be active at low heights (z=0.300)."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30
        )

        gate = diagnostics["hip_yaw_div_height_gate"]
        assert gate > 0.9, f"At z=0.30, gate should be ~1.0, got {gate}"

    def test_height_gate_inactive_at_nominal_height(self, controller):
        """HY2-DIV should be inactive at nominal heights (z=0.45)."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.45
        )

        gate = diagnostics["hip_yaw_div_height_gate"]
        assert gate < 0.01, f"At z=0.45, gate should be ~0, got {gate}"

    def test_height_gate_intermediate_at_mid_height(self, controller):
        """At mid height (z=0.35), gate should be partial."""
        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.35
        )

        gate = diagnostics["hip_yaw_div_height_gate"]
        assert 0.1 < gate < 0.9, f"At z=0.35, gate should be partial, got {gate}"


class TestHipYawDivergenceTorqueClamp:
    """Test HY2-DIV torque clamping."""

    def test_torque_clamp_respected(self):
        """HY2-DIV torque should not exceed tau_max_divergence."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=50.0,  # Large gain
            k_divergence_rate=10.0,
            tau_max_divergence=0.5,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Large divergence should saturate
        joint_pos = joint_pos.at[1].set(-1.0)  # Large error
        joint_pos = joint_pos.at[6].set(1.0)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30
        )

        div_tau_left = diagnostics["hip_yaw_div_left"]
        div_tau_right = diagnostics["hip_yaw_div_right"]

        # After clipping, torque should be within bounds
        assert abs(div_tau_left) <= 0.5 + 1e-6, f"Left torque should be clamped to 0.5, got {div_tau_left}"
        assert abs(div_tau_right) <= 0.5 + 1e-6, f"Right torque should be clamped to 0.5, got {div_tau_right}"

        # Clipped flags should be set since we're saturating
        assert diagnostics["hip_yaw_div_left_clipped"], "Left should be clipped for large divergence"
        assert diagnostics["hip_yaw_div_right_clipped"], "Right should be clipped for large divergence"

    def test_clipped_flag_set_when_saturating(self):
        """Clipped flag should be set when torque saturates."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=50.0,  # Large gain
            k_divergence_rate=10.0,
            tau_max_divergence=0.5,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(-1.0)
        joint_pos = joint_pos.at[6].set(1.0)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30
        )

        # Should be clipped for large divergence
        left_clipped = diagnostics["hip_yaw_div_left_clipped"]
        right_clipped = diagnostics["hip_yaw_div_right_clipped"]

        # At least one should be clipped
        assert left_clipped or right_clipped, "Large divergence should trigger clipping"


class TestHipYawDivergenceNoSideEffects:
    """Test HY2-DIV has no unintended side effects."""

    def test_hip_roll_unchanged_by_hy2_div(self):
        """HY2-DIV should not affect hip-roll torques."""
        controller_disabled = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
        )
        controller_enabled = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Set some errors
        joint_pos = joint_pos.at[0].set(-0.1)  # hip roll

        tau_disabled, _ = controller_disabled.compute(q_ref, joint_pos, joint_vel)
        tau_enabled, _ = controller_enabled.compute(q_ref, joint_pos, joint_vel)

        # Hip roll should be zero (not controlled by shape posture)
        assert tau_disabled[0] == 0.0
        assert tau_enabled[0] == 0.0

    def test_hip_pitch_unchanged_by_hy2_div(self):
        """HY2-DIV should not affect hip-pitch torques."""
        controller_disabled = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            kp_hip_pitch=30.0,
            kd_hip_pitch=4.0,
        )
        controller_enabled = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            kp_hip_pitch=30.0,
            kd_hip_pitch=4.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[2].set(-0.1)  # hip pitch

        tau_disabled, _ = controller_disabled.compute(q_ref, joint_pos, joint_vel)
        tau_enabled, _ = controller_enabled.compute(q_ref, joint_pos, joint_vel)

        # Hip pitch torques should be same
        assert abs(tau_disabled[2] - tau_enabled[2]) < 1e-6, \
            "Hip pitch torque should be unchanged by HY2-DIV"

    def test_hy2_div_telemetry_fields_exist(self):
        """HY2-DIV should populate all telemetry fields."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=5.0,
            k_divergence_rate=1.0,
            tau_max_divergence=0.5,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        joint_pos = joint_pos.at[1].set(-0.1)
        joint_pos = joint_pos.at[6].set(0.1)

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30
        )

        # Check all expected telemetry fields
        expected_fields = [
            # enabled vs gate_active are separate
            "hip_yaw_div_enabled",
            "hip_yaw_div_gate_active",
            # gate value
            "hip_yaw_div_height_gate",
            # effective gains (gate-scaled)
            "hip_yaw_div_effective_k",
            "hip_yaw_div_effective_kd",
            "hip_yaw_div_effective_tau_max",
            # raw torques
            "hip_yaw_div_left",
            "hip_yaw_div_right",
            # clipping flags
            "hip_yaw_div_left_clipped",
            "hip_yaw_div_right_clipped",
            # static config
            "hip_yaw_div_k_divergence",
            "hip_yaw_div_k_divergence_rate",
            "hip_yaw_div_tau_max",
            "hip_yaw_div_z_low",
            "hip_yaw_div_z_high",
        ]

        for field in expected_fields:
            assert field in diagnostics, f"Telemetry field '{field}' should exist"


class TestHipYawDivergenceRateDamping:
    """Test HY2-DIV rate damping."""

    def test_divergence_rate_produces_correcting_torque(self):
        """Divergence rate should produce damping torque."""
        controller = ShapePostureController(
            kp_hip_yaw=5.0,
            kd_hip_yaw=1.0,
            enable_hip_yaw_divergence_damping=True,
            k_divergence=0.0,  # No proportional
            k_divergence_rate=2.0,  # Only derivative
            tau_max_divergence=1.0,
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)

        # Set up divergence velocity: left moving forward, right moving backward
        joint_vel = joint_vel.at[1].set(1.0)  # left vel > 0
        joint_vel = joint_vel.at[6].set(-1.0)  # right vel < 0

        tau, diagnostics = controller.compute(
            q_ref, joint_pos, joint_vel,
            target_com_height=0.30
        )

        div_tau_left = diagnostics["hip_yaw_div_left"]
        div_tau_right = diagnostics["hip_yaw_div_right"]

        # With positive divergence rate: left is moving ahead faster
        # Damping should: left gets negative (slow down), right gets positive (speed up)
        assert div_tau_left < 0, f"Positive divergence rate: left should get negative torque, got {div_tau_left}"
        assert div_tau_right > 0, f"Positive divergence rate: right should get positive torque, got {div_tau_right}"


class TestHipYawDivergenceCandidateProfiles:
    """Test all Phase 3 authority candidate profiles."""

    def test_a0_profile_correct_gains(self):
        """A0 should match current baseline (k=5, kd=1, tau_max=0.5, z_high=0.393)."""
        assert HY2_DIV_A0.name == "hy2_div_A0"
        assert HY2_DIV_A0.k_divergence == 5.0
        assert HY2_DIV_A0.k_divergence_rate == 1.0
        assert HY2_DIV_A0.tau_max_divergence == 0.5
        assert HY2_DIV_A0.z_low == 0.300
        assert HY2_DIV_A0.z_high == 0.393

    def test_a1_profile_2x_tau_max(self):
        """A1 should have 2x tau_max (k=5, kd=1, tau_max=1.0, z_high=0.393)."""
        assert HY2_DIV_A1.name == "hy2_div_A1"
        assert HY2_DIV_A1.k_divergence == 5.0
        assert HY2_DIV_A1.k_divergence_rate == 1.0
        assert HY2_DIV_A1.tau_max_divergence == 1.0
        assert HY2_DIV_A1.z_low == 0.300
        assert HY2_DIV_A1.z_high == 0.393

    def test_a2_profile_4x_tau_max(self):
        """A2 should have 4x tau_max (k=5, kd=1, tau_max=2.0, z_high=0.393)."""
        assert HY2_DIV_A2.name == "hy2_div_A2"
        assert HY2_DIV_A2.k_divergence == 5.0
        assert HY2_DIV_A2.k_divergence_rate == 1.0
        assert HY2_DIV_A2.tau_max_divergence == 2.0
        assert HY2_DIV_A2.z_low == 0.300
        assert HY2_DIV_A2.z_high == 0.393

    def test_a3_profile_moderate_gain_increase(self):
        """A3 should have moderate gain increase (k=7.5, kd=1.5, tau_max=1.0, z_high=0.393)."""
        assert HY2_DIV_A3.name == "hy2_div_A3"
        assert HY2_DIV_A3.k_divergence == 7.5
        assert HY2_DIV_A3.k_divergence_rate == 1.5
        assert HY2_DIV_A3.tau_max_divergence == 1.0
        assert HY2_DIV_A3.z_low == 0.300
        assert HY2_DIV_A3.z_high == 0.393

    def test_b1_profile_extended_gate(self):
        """B1 should have extended height gate (k=5, kd=1, tau_max=1.0, z_high=0.500)."""
        assert HY2_DIV_B1.name == "hy2_div_B1"
        assert HY2_DIV_B1.k_divergence == 5.0
        assert HY2_DIV_B1.k_divergence_rate == 1.0
        assert HY2_DIV_B1.tau_max_divergence == 1.0
        assert HY2_DIV_B1.z_low == 0.300
        assert HY2_DIV_B1.z_high == 0.500

    def test_b2_profile_extended_gate_4x_tau(self):
        """B2 should have extended gate + 4x tau_max (k=5, kd=1, tau_max=2.0, z_high=0.500)."""
        assert HY2_DIV_B2.name == "hy2_div_B2"
        assert HY2_DIV_B2.k_divergence == 5.0
        assert HY2_DIV_B2.k_divergence_rate == 1.0
        assert HY2_DIV_B2.tau_max_divergence == 2.0
        assert HY2_DIV_B2.z_low == 0.300
        assert HY2_DIV_B2.z_high == 0.500

    def test_b3_profile_extended_gate_moderate_gain(self):
        """B3 should have extended gate + moderate gain (k=7.5, kd=1.5, tau_max=1.0, z_high=0.500)."""
        assert HY2_DIV_B3.name == "hy2_div_B3"
        assert HY2_DIV_B3.k_divergence == 7.5
        assert HY2_DIV_B3.k_divergence_rate == 1.5
        assert HY2_DIV_B3.tau_max_divergence == 1.0
        assert HY2_DIV_B3.z_low == 0.300
        assert HY2_DIV_B3.z_high == 0.500

    def test_c1_profile_strong_damping(self):
        """C1 should have strong damping (k=10, kd=2, tau_max=1.5, z_high=0.500)."""
        assert HY2_DIV_C1.name == "hy2_div_C1"
        assert HY2_DIV_C1.k_divergence == 10.0
        assert HY2_DIV_C1.k_divergence_rate == 2.0
        assert HY2_DIV_C1.tau_max_divergence == 1.5
        assert HY2_DIV_C1.z_low == 0.300
        assert HY2_DIV_C1.z_high == 0.500

    def test_all_candidates_list_complete(self):
        """ALL_HY2_DIV_CANDIDATES should contain all 8 profiles."""
        assert len(ALL_HY2_DIV_CANDIDATES) == 8
        names = {p.name for p in ALL_HY2_DIV_CANDIDATES}
        expected = {
            "hy2_div_A0", "hy2_div_A1", "hy2_div_A2", "hy2_div_A3",
            "hy2_div_B1", "hy2_div_B2", "hy2_div_B3", "hy2_div_C1"
        }
        assert names == expected

    def test_b1_gate_active_at_nominal(self):
        """B1 with z_high=0.500 should be active at nominal height (z=0.404)."""
        gate = HY2_DIV_B1.gate(0.404)
        assert gate > 0.4, f"B1 at z=0.404 should have gate > 0.4, got {gate}"

    def test_a0_gate_inactive_at_nominal(self):
        """A0 with z_high=0.393 should be inactive at nominal height (z=0.404)."""
        gate = HY2_DIV_A0.gate(0.404)
        assert gate < 0.01, f"A0 at z=0.404 should have gate < 0.01, got {gate}"

    def test_a1_allows_larger_torque(self):
        """A1 with tau_max=1.0 should allow larger torque than A0 (0.5)."""
        controller_a0 = ShapePostureController(
            enable_hip_yaw_divergence_damping=True,
            k_divergence=10.0,  # Large enough to saturate
            k_divergence_rate=2.0,
            tau_max_divergence=0.5,  # A0
        )
        controller_a1 = ShapePostureController(
            enable_hip_yaw_divergence_damping=True,
            k_divergence=10.0,  # Same gain
            k_divergence_rate=2.0,
            tau_max_divergence=1.0,  # A1
        )

        q_ref = jnp.zeros(ACTION_DIM)
        joint_pos = jnp.zeros(ACTION_DIM)
        joint_vel = jnp.zeros(ACTION_DIM)
        joint_pos = joint_pos.at[1].set(-1.0)
        joint_pos = joint_pos.at[6].set(1.0)

        _, diag_a0 = controller_a0.compute(q_ref, joint_pos, joint_vel, target_com_height=0.30)
        _, diag_a1 = controller_a1.compute(q_ref, joint_pos, joint_vel, target_com_height=0.30)

        # A1 should have larger magnitude (up to 1.0 vs 0.5)
        assert abs(diag_a1["hip_yaw_div_left"]) > abs(diag_a0["hip_yaw_div_left"]), \
            "A1 should allow larger torque than A0"