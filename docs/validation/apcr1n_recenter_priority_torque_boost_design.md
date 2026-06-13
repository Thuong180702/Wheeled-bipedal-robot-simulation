# APCR1n Recenter Priority Torque Boost Design

## Date
2026-06-11

## Based On
APCR1h_support_drift_priority_fast_recenter

## Not Based On
APCR1m_conditional_pitch_blend_recenter

APCR1m is only useful for audit insight, not as a base. APCR1m has 2.4x worse drift than APCR1h due to:
- Wheel velocity damping 3.5x larger (5.0 Nm vs 1.4 Nm)
- Position cap saturated 77.3%
- Final torque fights drift 62.8%

## APCR1n Goal

Improve support drift control beyond APCR1h while preserving startup stability.

## Root Cause Analysis from APCR1m Audit

### Problem 1: Wheel Velocity Damping Dominance
- APCR1m wheel damping = 5.0 Nm
- APCR1h wheel damping = 1.42 Nm
- APCR1m has 3.5x larger wheel damping
- This causes excessive wheel braking during recenter
- Result: wheel damping fights drift recovery 62.8% of RECENTER steps

### Problem 2: Position Cap Saturation
- tau_position capped at ±3 Nm
- Raw tau_position would reach about ±15 Nm
- Saturation rate = 77.3%
- During RECENTER saturation = 87.3%
- Position torque sign is correct but limited by cap

### Solution: APCR1n

APCR1n = APCR1h base + targeted fixes:
1. Wheel damping override during RECENTER when it fights drift
2. Position cap boost during safe RECENTER

## APCR1n Profile Design

### A. Wheel Damping Override During RECENTER

**Purpose**: Reduce wheel velocity damping only when it fights support recenter.

**Parameters**:
```python
vd_wheel_damping_recenter_override_enabled = True
vd_wheel_damping_recenter_scale = 0.30  # Reduce to 30% of baseline
vd_wheel_damping_recenter_min_abs_nm = 0.50  # Minimum damping preserved
vd_wheel_damping_preserve_if_opposes_drift = True
```

**Logic**:
```
During RECENTER:
    if wheel damping component fights support drift recovery:
        scale wheel damping by 0.30
        preserve minimum damping only if needed for stability
    else:
        preserve wheel damping
```

**Safety**:
- Only affects RECENTER state
- Only affects when wheel damping fights drift
- Preserves minimum damping for stability

### B. Position Cap Boost During Safe RECENTER

**Purpose**: Position torque is correct but capped too low in APCR1m.

**Parameters**:
```python
position_cap_recenter_boost_enabled = True
position_cap_normal_nm = 3.0  # Current APCR1h cap
position_cap_recenter_nm = 5.0  # Boosted cap during RECENTER
position_cap_emergency_nm = 6.0  # Emergency cap
position_cap_ramp_steps = 50  # Gradual ramp to boosted cap
```

**Safety Gates**:
Enable recenter cap boost only if:
- Contact valid
- CoM Z safe
- Roll safe
- Pitch not beyond hard safety
- No hidden torque
- No ownership violation

### C. Startup Guard

**Purpose**: Preserve APCR1h startup stability.

**Parameters**:
```python
recenter_priority_startup_guard_steps = 100
```

**Logic**:
- During startup guard: use APCR1h behavior
- After startup guard: allow recenter priority modifications when safe

### D. Final Torque Direction Monitor (Telemetry Only)

**Purpose**: Verify that modifications improve final torque direction correctness.

**Telemetry Fields**:
```python
apcr1n_recenter_priority_active
apcr1n_startup_guard_active
apcr1n_wheel_damping_override_active
apcr1n_wheel_damping_scale
apcr1n_wheel_damping_before
apcr1n_wheel_damping_after
apcr1n_wheel_damping_fights_drift
apcr1n_position_cap_boost_active
apcr1n_position_cap_current
apcr1n_tau_position_raw
apcr1n_tau_position_after_cap
apcr1n_position_saturated
apcr1n_safety_gate_pass
apcr1n_final_torque_direction_correct
apcr1n_final_torque_fights_drift
apcr1n_physical_drift_column_used
```

## Implementation Details

### Profile Definition
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
    position_cap_normal_nm=3.0,
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    position_cap_ramp_steps=50,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
)
```

### Telemetry Fields to Add

```python
@dataclass
class APCR1nTelemetry:
    apcr1n_recenter_priority_active: bool = False
    apcr1n_startup_guard_active: bool = True
    apcr1n_wheel_damping_override_active: bool = False
    apcr1n_wheel_damping_scale: float = 1.0
    apcr1n_wheel_damping_before: float = 0.0
    apcr1n_wheel_damping_after: float = 0.0
    apcr1n_wheel_damping_fights_drift: bool = False
    apcr1n_position_cap_boost_active: bool = False
    apcr1n_position_cap_current: float = 3.0
    apcr1n_tau_position_raw: float = 0.0
    apcr1n_tau_position_after_cap: float = 0.0
    apcr1n_position_saturated: bool = False
    apcr1n_safety_gate_pass: bool = False
    apcr1n_final_torque_direction_correct: bool = True
    apcr1n_final_torque_fights_drift: bool = False
    apcr1n_physical_drift_column_used: str = "active_pitch_crossing_signed_error_m"
```

## Expected Behavior

### During Startup (0-100 steps)
- APCR1n behaves exactly like APCR1h
- No wheel damping override
- No position cap boost
- Startup guard active = True

### After Startup, Outside RECENTER
- APCR1n behaves exactly like APCR1h
- Startup guard active = False
- Wheel damping override inactive
- Position cap boost inactive

### After Startup, In RECENTER, Wheel Damping Fights Drift
- Wheel damping reduced by 0.30x scale
- Minimum 0.5 Nm damping preserved
- Position cap boosted to 5.0 Nm if safety gates pass

### After Startup, In RECENTER, Wheel Damping Opposes Drift
- Wheel damping preserved (1.0x scale)
- Position cap boosted to 5.0 Nm if safety gates pass

## Expected Metrics Improvement

| Metric | APCR1h | APCR1n Target | APCR1m |
|--------|--------|----------------|--------|
| max \|e\| (m) | 0.178 | < 0.178 | 0.434 |
| P2P (m) | 0.249 | ≤ 0.249 | 0.833 |
| outside ±0.15 | 9.7% | < 9.7% | 54.0% |
| wheel damping (Nm) | 1.42 | 1.5-2.0 | 5.00 |
| position saturation | low | reduced | 77.3% |
| final torque fights drift | ? | reduced | 62.8% |

## Files to Modify

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Add APCR1n dataclass fields
   - Add APCR1N_RECENTER_PRIORITY_TORQUE_BOOST profile
   - Add to profile registry
   - Implement wheel damping override logic
   - Implement position cap boost logic
   - Add APCR1n telemetry fields

2. `scripts/simulate_hierarchical_controller.py`
   - Add APCR1n profile to CLI selection
   - Add APCR1n telemetry collection

3. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Add APCR1n profile tests

## Files to Create

1. `docs/validation/apcr1n_recenter_priority_torque_boost_design.md` (this file)

## Success Criteria

APCR1n succeeds if:
1. max |e| < APCR1h (0.178 m)
2. outside ±0.15 < APCR1h (9.7%)
3. P2P ≤ APCR1h (0.249 m)
4. Startup stability preserved
5. Wheel damping reduced toward 1.5-2.0 Nm during RECENTER
6. Position saturation reduced
7. Final torque direction correctness improved
