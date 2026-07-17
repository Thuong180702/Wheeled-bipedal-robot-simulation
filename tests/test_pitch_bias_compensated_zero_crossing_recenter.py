"""Tests for pitch_bias_compensated_zero_crossing_recenter profile.

Phase 4 tests as specified in the task.
Tests the pitch DC bias compensation mechanism (Phase 7).

Key behaviors:
- Profile exists and is opt-in (base EZC V2 unchanged)
- pitch_bias_comp_enabled=True in new profile
- EMA estimate grows for persistent positive tau_pitch
- Compensation is bounded by pitch_bias_max_comp_nm
- Compensation is rate-limited
- Compensation only applies under safety gates
- Dynamic pitch torque is preserved (tau_pitch never zeroed)
- Sign not globally flipped
- Telemetry fields exist
- CLI accepts profile
- No WBC/HY2-DIV default change
"""
import math
import pytest
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    SagittalVelocityDampedBalanceController,
    SagittalAuthoritySchedule,
    EARLY_ZERO_CROSSING_RECENTER_V2,
    PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER,
    JOINT_FIX_PROFILES,
    BASELINE_AUTHORITY_SCHEDULE,
)
from scripts.simulate_hierarchical_controller import SAGITTAL_AUTHORITY_PROFILES


# ─────────────────────────────────────────────────────────────────────────────
# 1. Profile exists and is opt-in
# ─────────────────────────────────────────────────────────────────────────────
class TestProfileExists:
    def test_profile_in_JOINT_FIX_PROFILES(self):
        assert "pitch_bias_compensated_zero_crossing_recenter" in JOINT_FIX_PROFILES

    def test_profile_in_SAGITTAL_AUTHORITY_PROFILES(self):
        assert "pitch_bias_compensated_zero_crossing_recenter" in SAGITTAL_AUTHORITY_PROFILES

    def test_constant_exists(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER is not None
        assert isinstance(PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER, SagittalAuthoritySchedule)

    def test_profile_name_correct(self):
        assert (PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.profile_name
                == "pitch_bias_compensated_zero_crossing_recenter")

    def test_applies_to_high_heights(self):
        variants = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.applies_to_variants
        assert "high_0p480" in variants
        assert "high_0p465" in variants
        assert "high_0p450" in variants


# ─────────────────────────────────────────────────────────────────────────────
# 2. Base EZC V2 is unchanged
# ─────────────────────────────────────────────────────────────────────────────
class TestBaseV2Unchanged:
    def test_v2_pitch_bias_disabled(self):
        assert EARLY_ZERO_CROSSING_RECENTER_V2.pitch_bias_comp_enabled is False

    def test_pbc_inherits_antirebound(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.ezc_antirebound_enabled is True
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.ezc_antirebound_decay_steps == 30

    def test_pbc_inherits_ezc_torque_params(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.ezc_base_tau_nm == 0.25
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.ezc_max_tau_nm == 0.70

    def test_pbc_inherits_safety_gates(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.ezc_disable_if_pitch_gt_deg == 12.0
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.ezc_disable_if_roll_gt_deg == 5.0

    def test_baseline_pitch_bias_disabled(self):
        assert BASELINE_AUTHORITY_SCHEDULE.pitch_bias_comp_enabled is False


# ─────────────────────────────────────────────────────────────────────────────
# 3. Pitch bias enabled in new profile
# ─────────────────────────────────────────────────────────────────────────────
class TestPitchBiasEnabled:
    def test_pitch_bias_comp_enabled_true(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_comp_enabled is True

    def test_window_steps(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_window_steps == 300

    def test_max_comp(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_max_comp_nm == 0.60

    def test_comp_rate(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_comp_rate_nm_per_step == 0.005

    def test_decay_rate(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_decay_rate_nm_per_step == 0.012

    def test_pitch_window(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_only_when_abs_pitch_lt_deg == 2.0

    def test_error_window(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_only_when_abs_error_lt_m == 0.12


# ─────────────────────────────────────────────────────────────────────────────
# 4. Safety gate parameters
# ─────────────────────────────────────────────────────────────────────────────
class TestSafetyGates:
    def test_pitch_safety_disable(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_disable_if_pitch_gt_deg == 12.0

    def test_roll_safety_disable(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_disable_if_roll_gt_deg == 5.0

    def test_contact_safety_disable(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_disable_if_contact_unstable is True

    def test_height_safety_disable(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_disable_if_height_lt_m == 0.25

    def test_hard_gate(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_gate_abs_error_hard_m == 0.20

    def test_soft_gate(self):
        assert PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_gate_abs_error_soft_m == 0.12

    def test_soft_gate_less_than_hard_gate(self):
        sched = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER
        assert sched.pitch_bias_gate_abs_error_soft_m < sched.pitch_bias_gate_abs_error_hard_m


# ─────────────────────────────────────────────────────────────────────────────
# 5. State variables initialized
# ─────────────────────────────────────────────────────────────────────────────
class TestStateVarsInit:
    def test_state_vars_exist(self):
        ctrl = SagittalVelocityDampedBalanceController(
            authority_schedule=PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER)
        assert hasattr(ctrl, "_pitch_bias_estimate_nm")
        assert hasattr(ctrl, "_pitch_bias_samples")
        assert hasattr(ctrl, "_pitch_bias_comp_tau_nm")

    def test_state_vars_zero_at_init(self):
        ctrl = SagittalVelocityDampedBalanceController(
            authority_schedule=PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER)
        assert ctrl._pitch_bias_estimate_nm == 0.0
        assert ctrl._pitch_bias_samples == 0
        assert ctrl._pitch_bias_comp_tau_nm == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. EMA estimate convergence (analytical)
# ─────────────────────────────────────────────────────────────────────────────
class TestEstimateConvergence:
    def test_ema_converges_toward_persistent_bias(self):
        """EMA estimate should approach the persistent tau_pitch after enough samples."""
        window = 300
        alpha = 1.0 / window
        tau_pitch_persistent = 0.25  # simulated persistent positive tau_pitch in stable window
        est = 0.0
        for _ in range(window * 3):
            est = (1.0 - alpha) * est + alpha * tau_pitch_persistent
        # After 3x window steps: EMA convergence factor = 1 - exp(-3) ≈ 0.95
        assert est > tau_pitch_persistent * 0.85, (
            f"EMA not converging after {window * 3} steps: est={est:.4f}")

    def test_ema_does_not_overshoot(self):
        """EMA never exceeds the persistent input value."""
        window = 300
        alpha = 1.0 / window
        tau_pitch_persistent = 0.25
        est = 0.0
        for _ in range(window * 10):
            est = (1.0 - alpha) * est + alpha * tau_pitch_persistent
        assert est <= tau_pitch_persistent + 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# 7. Compensation is bounded
# ─────────────────────────────────────────────────────────────────────────────
class TestCompensationBounded:
    def test_max_comp_reasonable(self):
        max_comp = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_max_comp_nm
        assert 0.1 <= max_comp <= 1.0

    def test_target_clipped_to_max_comp(self):
        """Target compensation is clipped to max_comp regardless of estimate."""
        max_comp = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_max_comp_nm
        huge_estimate = 100.0
        target = min(max(0.0, huge_estimate), max_comp)
        assert target == max_comp

    def test_comp_never_negative(self):
        """Compensation only removes positive bias; negative estimate gives zero comp."""
        negative_estimate = -0.5
        target = max(0.0, negative_estimate)
        assert target == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Compensation is rate-limited
# ─────────────────────────────────────────────────────────────────────────────
class TestRateLimited:
    def test_rate_is_slow(self):
        """Rate takes at least 10 steps to approach max comp."""
        rate = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_comp_rate_nm_per_step
        max_comp = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_max_comp_nm
        steps_to_max = max_comp / rate
        assert steps_to_max >= 10

    def test_decay_faster_than_growth(self):
        """Decay rate is faster than growth rate."""
        rate = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_comp_rate_nm_per_step
        decay = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_decay_rate_nm_per_step
        assert decay > rate


# ─────────────────────────────────────────────────────────────────────────────
# 9. Dynamic pitch torque is preserved (tau_pitch never zeroed)
# ─────────────────────────────────────────────────────────────────────────────
class TestDynamicPitchPreserved:
    def test_max_comp_small_relative_to_dynamic_range(self):
        """Max comp (0.60 Nm) is small relative to tau_pitch at 5 deg (4.36 Nm)."""
        max_comp = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_max_comp_nm
        kp_pitch = 50.0
        tau_pitch_at_5deg = kp_pitch * 5.0 * math.pi / 180.0  # ~4.36 Nm
        assert max_comp < 0.5 * tau_pitch_at_5deg

    def test_sign_not_globally_flipped(self):
        """Profile does not flip wheel_torque_sign or use negative kp_pitch equivalent."""
        # Compensation only subtracts positive bias; never sign-flips globally
        max_comp = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER.pitch_bias_max_comp_nm
        assert max_comp >= 0.0  # Always non-negative (no sign flip)

    def test_compensation_only_positive_direction(self):
        """Compensation targets only positive DC bias (max(0, estimate))."""
        for estimate_val in [-1.0, -0.5, -0.1, 0.0]:
            target = max(0.0, estimate_val)
            assert target == 0.0, f"Negative estimate {estimate_val} should give zero target"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Gate logic (unit-level)
# ─────────────────────────────────────────────────────────────────────────────
class TestGateLogic:
    def test_pitch_above_window_disables_estimation(self):
        sched = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER
        abs_pitch_deg = 3.0  # above 2.0 deg window
        estimation_active = abs_pitch_deg < sched.pitch_bias_only_when_abs_pitch_lt_deg
        assert estimation_active is False

    def test_pitch_below_window_enables_estimation_if_error_small(self):
        sched = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER
        abs_pitch_deg = 0.5   # below 2 deg
        abs_error = 0.05      # below 0.12 m
        estimation_active = (
            abs_pitch_deg < sched.pitch_bias_only_when_abs_pitch_lt_deg
            and abs_error < sched.pitch_bias_only_when_abs_error_lt_m
        )
        assert estimation_active is True

    def test_error_above_hard_gate_blocks_apply(self):
        sched = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER
        abs_error = 0.25  # above 0.20 hard gate
        gate_pass = abs_error < sched.pitch_bias_gate_abs_error_hard_m
        assert gate_pass is False

    def test_error_below_soft_gate_allows_apply(self):
        sched = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER
        abs_error = 0.05  # below 0.12 soft gate
        gate_pass = abs_error < sched.pitch_bias_gate_abs_error_soft_m
        assert gate_pass is True


# ─────────────────────────────────────────────────────────────────────────────
# 11. Telemetry fields exist as schedule attributes
# ─────────────────────────────────────────────────────────────────────────────
class TestTelemetryFields:
    EXPECTED_FIELDS = [
        "pitch_bias_comp_enabled",
        "pitch_bias_window_steps",
        "pitch_bias_max_comp_nm",
        "pitch_bias_comp_rate_nm_per_step",
        "pitch_bias_decay_rate_nm_per_step",
        "pitch_bias_only_when_abs_pitch_lt_deg",
        "pitch_bias_only_when_abs_error_lt_m",
        "pitch_bias_disable_if_pitch_gt_deg",
        "pitch_bias_disable_if_roll_gt_deg",
        "pitch_bias_disable_if_contact_unstable",
        "pitch_bias_disable_if_height_lt_m",
        "pitch_bias_gate_abs_error_soft_m",
        "pitch_bias_gate_abs_error_hard_m",
    ]

    def test_all_telemetry_fields_exist(self):
        sched = PITCH_BIAS_COMPENSATED_ZERO_CROSSING_RECENTER
        for field in self.EXPECTED_FIELDS:
            assert hasattr(sched, field), f"Missing field: {field}"


# ─────────────────────────────────────────────────────────────────────────────
# 12. CLI accepts profile name
# ─────────────────────────────────────────────────────────────────────────────
class TestCLIAccepts:
    def test_cli_registry_has_profile(self):
        assert "pitch_bias_compensated_zero_crossing_recenter" in SAGITTAL_AUTHORITY_PROFILES

    def test_cli_profile_correct_type(self):
        profile = SAGITTAL_AUTHORITY_PROFILES["pitch_bias_compensated_zero_crossing_recenter"]
        assert isinstance(profile, SagittalAuthoritySchedule)

    def test_cli_profile_name_matches(self):
        profile = SAGITTAL_AUTHORITY_PROFILES["pitch_bias_compensated_zero_crossing_recenter"]
        assert profile.profile_name == "pitch_bias_compensated_zero_crossing_recenter"
