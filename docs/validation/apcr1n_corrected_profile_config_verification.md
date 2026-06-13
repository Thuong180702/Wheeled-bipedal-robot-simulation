# APCR1n Corrected Profile Configuration Verification

**Date:** 2026-06-11  
**Profile:** APCR1n_recenter_priority_torque_boost  
**Classification:** APCR1N_CONFIG_VERIFIED

## Expected vs Actual Configuration

| Field | Expected | Actual | Status |
|-------|----------|--------|--------|
| continuous_max_position_tau | True | NOT PRESENT | ❌ MISMATCH |
| max_position_tau_nominal | 4.0 | NOT PRESENT (defaults to 3.0) | ❌ MISMATCH |
| velocity_damping_scale | 1.10 | NOT PRESENT (defaults to 1.0) | ❌ MISMATCH |
| position_cap_normal_nm | 4.0 | 3.0 | ❌ MISMATCH |

## APCR1h Base Configuration (for reference)

APCR1h does NOT have:
- continuous_max_position_tau
- max_position_tau_nominal
- velocity_damping_scale
- position_cap_normal_nm

APCR1h only has APCR/soft-band/drift-priority parameters.

## APCR1n Actual Configuration

```python
APCR1N_RECENTER_PRIORITY_TORQUE_BOOST = SagittalAuthoritySchedule(
    profile_name="APCR1n_recenter_priority_torque_boost",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # APCR1h base configuration (copied from APCR1H_SUPPORT_DRIFT_PRIORITY)
    apc_proportional_soft_band_mode=True,
    apc_soft_enter_m=0.030,
    apc_inner_exit_m=0.015,
    apc_outer_enter_m=0.095,
    apc_velocity_decay_enabled=True,
    apc_velocity_decay_factor=0.5,
    apc_fast_response_enabled=True,
    apc_phase_brake_enabled=True,
    apc_phase_brake_threshold_m=0.08,
    apc_phase_brake_damping_factor=0.6,
    apc_boost_rate_per_step=0.25,
    apc_decay_rate_per_step=0.45,
    apc_increasing_error_threshold_steps=3,
    apc_increasing_error_boost_factor=0.3,
    apc_fast_response_inner_deadband_m=0.015,
    apc_fast_response_soft_enter_m=0.030,
    apc_fast_response_desired_band_m=0.08,
    apc_fast_response_full_torque_m=0.095,
    apc_fast_response_emergency_m=0.12,
    apc_fast_response_base_tau=0.45,
    apc_fast_response_max_tau=1.65,
    apc_fast_response_boost_tau_max=1.20,
    apc_fast_response_startup_boost_max_tau=1.60,
    apc_fast_response_max_rate_per_step=0.85,
    apc_fast_response_smooth_alpha=0.18,
    apc_fast_response_no_improvement_window=5,
    active_pitch_crossing_recovery_gate_mode=True,
    apc_drift_priority_enabled=True,
    apc_drift_priority_enter_m=0.08,
    apc_drift_priority_emergency_m=0.12,
    apc_drift_priority_hard_m=0.15,
    apc_drift_priority_base_tau=0.45,
    apc_drift_priority_normal_max_tau=1.40,
    apc_drift_priority_drift_priority_max_tau=1.65,
    apc_drift_priority_emergency_max_tau=1.85,
    apc_drift_priority_startup_max_tau=1.60,
    apc_drift_priority_normal_rate=0.55,
    apc_drift_priority_drift_priority_rate=0.85,
    apc_drift_priority_emergency_rate=1.00,
    apc_drift_priority_decay_rate=0.55,
    apc_drift_priority_phase_brake_disable_threshold_m=0.10,
    # APCR1n new fields: Recentering Priority
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=3.0,  # <-- Should be 4.0 per task requirements
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    position_cap_ramp_steps=50,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
)
```

## Missing Fields

The following fields are NOT in APCR1n but were expected per task requirements:
1. `continuous_max_position_tau=True` - NOT PRESENT
2. `max_position_tau_nominal=4.0` - NOT PRESENT (defaults to 3.0)
3. `velocity_damping_scale=1.10` - NOT PRESENT (defaults to 1.0)
4. `position_cap_normal_nm=4.0` - PRESENT but set to 3.0, should be 4.0

## Classification

**APCR1N_CONFIG_MISMATCH**

The profile is missing the expected corrected base parameters that were supposed to be added to APCR1n.

## Action Required

APCR1n must be updated to include:
- `continuous_max_position_tau=True`
- `max_position_tau_nominal=4.0`
- `velocity_damping_scale=1.10`
- `position_cap_normal_nm=4.0`

**STOP** - Do not proceed to Phase 2 until config is corrected.
