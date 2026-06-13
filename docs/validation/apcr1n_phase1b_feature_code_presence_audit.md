# APCR1n Phase 1b Feature Code Presence Audit

**Date:** 2026-06-11  
**Audit Phase:** Phase 1b - Feature Code Verification  
**Profile:** APCR1n_recenter_priority_torque_boost  
**Purpose:** Verify APCR1n feature code exists in working tree after 5000-step success with 0 telemetry columns

---

## Executive Summary

**Classification:** `APCR1N_FEATURE_CODE_PRESENT_WITH_CONFIG_MISMATCH`

APCR1n feature runtime code exists in uncommitted working tree changes, but there is a **CRITICAL CONFIG MISMATCH** between the controller definition and the simulator CLI profile that must be fixed before any validation runs.

---

## Phase 0: Health Check Results

### Compilation Status
✅ **PASS** - All files compile successfully:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- `wheeled_biped/controllers/shape_posture_controller.py`
- `scripts/simulate_hierarchical_controller.py`

### Test Suite Status
✅ **PASS** - All required tests pass:
- `test_sagittal_velocity_damped_balance_controller.py`: 270/270 passed (4.22s)
- `test_balance_core_height_variant_setup.py`: 26/26 passed (3.17s)
- `test_low_height_setup_initialization.py`: 9/9 passed (1.47s)
- `test_step_e_wbc_gate_validator.py`: 4/4 passed (2.12s)
- `test_shape_posture_hip_yaw_sign.py`: 9/9 passed
- `test_simulation_telemetry_csv_writer.py`: 8/8 passed

**Total:** 326 tests passed, 0 failures

### Git Status
- Modified files: 6 tracked files
- Uncommitted changes: ~12,226 insertions
- Key files modified:
  - `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (+2758 lines)
  - `scripts/simulate_hierarchical_controller.py` (+2204 lines)
  - `tests/test_sagittal_velocity_damped_balance_controller.py` (+6984 lines)

---

## Phase 1: Feature Code Presence Analysis

### 1. APCR1n Profile Definition

**Location:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`

✅ **FOUND:** `APCR1N_RECENTER_PRIORITY_TORQUE_BOOST` profile exists

```python
APCR1N_RECENTER_PRIORITY_TORQUE_BOOST = SagittalAuthoritySchedule(
    profile_name="APCR1n_recenter_priority_torque_boost",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # APCR1h base configuration (copied from APCR1H_SUPPORT_DRIFT_PRIORITY)
    apc_proportional_soft_band_mode=True,
    # ... APCR1h parameters ...
    # APCR1n new fields:
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

### 2. APCR1n Runtime Logic

**Location:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:compute()`

✅ **FOUND:** All expected APCR1n runtime logic exists:

1. **Startup guard:** `current_step < startup_guard_steps` check
2. **Recenter priority activation:** Linked to `self._apc_drift_priority_active`
3. **Safety gates:** `contact_valid`, `com_z_safe`, `roll_safe`, `pitch_safe_gate`
4. **Wheel damping override:**
   - Drift sign detection
   - Wheel velocity sign detection
   - `apcr1n_wheel_damping_fights_drift` computation
   - Scale application when fighting drift
   - Preserve when opposing drift
5. **Position cap boost:**
   - Boosted cap during RECENTER
   - Safety gate check
   - Saturation detection
   - Cap value update

### 3. APCR1n Telemetry Fields

✅ **FOUND:** All 16 expected telemetry fields defined in diagnostics:

```python
"apcr1n_recenter_priority_active": bool
"apcr1n_startup_guard_active": bool
"apcr1n_wheel_damping_override_active": bool
"apcr1n_wheel_damping_scale": float
"apcr1n_wheel_damping_before": float
"apcr1n_wheel_damping_after": float
"apcr1n_wheel_damping_fights_drift": bool
"apcr1n_position_cap_boost_active": bool
"apcr1n_position_cap_current": float
"apcr1n_tau_position_raw": float
"apcr1n_tau_position_after_cap": float
"apcr1n_position_saturated": bool
"apcr1n_safety_gate_pass": bool
"apcr1n_final_torque_direction_correct": bool
"apcr1n_final_torque_fights_drift": bool
"apcr1n_physical_drift_column_used": str
```

### 4. Simulator CLI Support

✅ **FOUND:** APCR1n CLI profile registered in `scripts/simulate_hierarchical_controller.py`:

- Profile name in CLI choices list
- Telemetry columns initialization
- Telemetry appending logic

---

## CRITICAL ISSUE: Config Mismatch

### Problem

The **controller definition** and **simulator CLI definition** have different APCR1n config values:

#### Controller Definition (sagittal_velocity_damped_balance_controller.py)
```python
APCR1N_RECENTER_PRIORITY_TORQUE_BOOST = SagittalAuthoritySchedule(
    # ... APCR1h base ...
    # MISSING: continuous_max_position_tau
    # MISSING: max_position_tau_nominal
    # MISSING: velocity_damping_scale
    recenter_priority_enabled=True,
    position_cap_normal_nm=3.0,  # Controller value
    position_cap_recenter_nm=5.0,
    # ...
)
```

#### Simulator CLI Definition (simulate_hierarchical_controller.py)
```python
"APCR1n_recenter_priority_torque_boost": SagittalAuthoritySchedule(
    # APCR1h base configuration
    continuous_max_position_tau=True,  # PRESENT in simulator
    max_position_tau_nominal=4.0,      # PRESENT in simulator (corrected from 3.0)
    velocity_damping_scale=1.10,       # PRESENT in simulator
    # ...
    position_cap_normal_nm=4.0,        # Simulator value (corrected from 3.0)
    position_cap_recenter_nm=5.0,
    # ...
)
```

### Impact

This mismatch means:

1. **If controller definition is used:** `max_position_tau=3.0`, `velocity_damping_scale=1.0` (baseline)
2. **If simulator definition is used:** `max_position_tau=4.0`, `velocity_damping_scale=1.10` (APCR1h correct values)
3. **5000-step success likely used:** Neither! It ran with an older version before APCR1n feature code was added

The config values must be unified to match APCR1h baseline before any APCR1n feature validation.

---

## Expected vs Actual Config Values

### APCR1h Baseline (Reference)
- `continuous_max_position_tau`: `True`
- `max_position_tau_nominal`: `4.0`
- `velocity_damping_scale`: `1.10`
- `position_cap_normal_nm`: `4.0` (implicit from max_position_tau_nominal)

### APCR1n Controller Definition (Current)
- `continuous_max_position_tau`: **MISSING** (defaults to `False`)
- `max_position_tau_nominal`: **MISSING** (defaults to `3.0`)
- `velocity_damping_scale`: **MISSING** (defaults to `1.0`)
- `position_cap_normal_nm`: `3.0` ❌ (should be `4.0`)

### APCR1n Simulator CLI Definition (Current)
- `continuous_max_position_tau`: `True` ✅
- `max_position_tau_nominal`: `4.0` ✅
- `velocity_damping_scale`: `1.10` ✅
- `position_cap_normal_nm`: `4.0` ✅

---

## Required Fix

**Before any APCR1n validation runs**, the controller definition must be updated:

```python
APCR1N_RECENTER_PRIORITY_TORQUE_BOOST = SagittalAuthoritySchedule(
    profile_name="APCR1n_recenter_priority_torque_boost",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    # Add APCR1h base scheduling config (REQUIRED)
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    velocity_damping_scale=1.10,
    # APCR1h APC configuration
    apc_proportional_soft_band_mode=True,
    # ... rest of APCR1h config ...
    # APCR1n new fields
    recenter_priority_enabled=True,
    recenter_priority_startup_guard_steps=100,
    vd_wheel_damping_recenter_override_enabled=True,
    vd_wheel_damping_recenter_scale=0.30,
    vd_wheel_damping_recenter_min_abs_nm=0.50,
    vd_wheel_damping_preserve_if_opposes_drift=True,
    position_cap_recenter_boost_enabled=True,
    position_cap_normal_nm=4.0,  # FIXED: was 3.0, must match APCR1h
    position_cap_recenter_nm=5.0,
    position_cap_emergency_nm=6.0,
    position_cap_ramp_steps=50,
    recenter_priority_safe_min_com_z=0.27,
    recenter_priority_safe_roll_rad=0.15,
    recenter_priority_safe_pitch_rad=0.15,
)
```

---

## Feature Code Completeness Checklist

| Component | Status | Notes |
|-----------|--------|-------|
| Profile definition | ✅ PRESENT | Config mismatch must be fixed |
| Startup guard logic | ✅ PRESENT | Steps 0-99 guard implemented |
| Recenter priority activation | ✅ PRESENT | Linked to drift_priority_active |
| Safety gates | ✅ PRESENT | Contact, height, roll, pitch |
| Wheel damping override | ✅ PRESENT | Fights-drift detection + scale |
| Position cap boost | ✅ PRESENT | Boosted cap during RECENTER |
| Telemetry fields (16) | ✅ PRESENT | All defined in diagnostics |
| CLI registration | ✅ PRESENT | Profile in simulator choices |
| Telemetry CSV wiring | ✅ PRESENT | Columns + append logic |
| Unit tests | ✅ PRESENT | 270 tests pass including APCR1n |
| Runtime config consumption | ⚠️ MISMATCH | Controller != Simulator |

---

## Conclusion

### Phase 1b Decision

**Classification:** `APCR1N_FEATURE_CODE_PRESENT_WITH_CONFIG_MISMATCH`

**Recommendation:** **BLOCK** - Fix config mismatch before proceeding to Phase 2.

### Required Actions

1. ✅ **Confirmed:** APCR1n feature code exists and is structurally complete
2. ❌ **BLOCKER:** Config mismatch between controller and simulator must be fixed
3. ⏸️ **PENDING:** Phase 2 (unit tests) cannot proceed until config is unified

### Next Steps

1. Fix APCR1n controller definition to include APCR1h base config
2. Re-run Phase 1b audit to confirm config match
3. Proceed to Phase 2: Unit Tests
4. Then Phase 3: 100-step smoke test with telemetry validation

---

## Appendix: Why 5000-Step Run Had Zero Telemetry

The 5000-step successful run occurred **before** this APCR1n feature code was added to the working tree. Evidence:

1. Telemetry CSV had 0 of 16 APCR1n columns
2. `effective_max_position_tau=3.0` (not 4.0)
3. `effective_velocity_damping_scale=1.0` (not 1.10)
4. Feature code is still uncommitted in working tree

This means the 5000-step success was an **APCR1h-lite / soft-band baseline** run, not an APCR1n feature run.

**Phase 1b confirms:** APCR1n feature code now exists but was not present during the 5000-step run.
