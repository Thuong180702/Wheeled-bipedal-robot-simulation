# APCR1n_recenter_priority_torque_boost Final Validation Report

**Date:** 2026-06-11
**Profile:** APCR1n_recenter_priority_torque_boost
**Based on:** APCR1h_support_drift_priority_fast_recenter
**Test:** 2000-step continuous simulation at low_0p300 (0.30 m height)

---

## Summary

APCR1n successfully completed 2000 steps without falling, with **improved drift performance** compared to APCR1h baseline.

| Metric | APCR1h (1000-step) | APCR1n CORRECTED (2000-step) | Change |
|--------|-------------------|-------------------------------|--------|
| max \|e\| | 0.1775 m | 0.1714 m | **-3.4% better** |
| P2P | 0.2491 m | 0.1854 m | **-25.6% better** |
| outside ±0.15 | 9.7% | 2.6% | **-7.0 pp better** |
| mean \|e\| | 0.0745 m | 0.0608 m | -18.4% better |
| Final error | 0.1668 m | 0.0035 m | **-97.9% better** |

---

## Profile Corrections Applied

The initial APCR1n profile had incorrect parameter values that caused 2.4x worse drift. These were corrected:

```python
# In SAGITTAL_AUTHORITY_PROFILES["APCR1n_recenter_priority_torque_boost"]:

# ADDED: Missing parameters from APCR1h base
continuous_max_position_tau=True,  # Was missing - critical for position authority
max_position_tau_nominal=4.0,     # Was 3.0 - matches APCR1h
velocity_damping_scale=1.10,      # Was missing - matches APCR1h

# FIXED: Corrected position cap
position_cap_normal_nm=4.0,       # Was 3.0 - matches APCR1h
```

---

## APCR1n New Features

### 1. Startup Guard (100 steps)
- Preserves APCR1h behavior for first 100 steps
- Prevents premature feature activation during initialization

### 2. Wheel Damping Override
- When RECENTER activates and wheel damping opposes drift recovery:
  - Scale wheel damping by 0.30x
  - Minimum 0.50 Nm preserved if opposing drift
- Reduces "fighting" between wheel damping and drift recovery

### 3. Position Cap Boost
- During safe RECENTER, boost position cap from 4.0 Nm to 5.0 Nm
- Emergency boost up to 6.0 Nm
- Safety gates ensure safe conditions before activation

### 4. Safety Gates
- Minimum COM Z: 0.27 m
- Maximum roll: 0.15 rad
- Maximum pitch: 0.15 rad

---

## Feature Activity During Validation

| Feature | Activation Count | Total Steps | Notes |
|---------|-----------------|-------------|-------|
| startup_guard_active | 100 | 2000 | First 100 steps only |
| recenter_priority_active | 0 | 2000 | Never activated |
| position_cap_boost_active | 0 | 2000 | Never activated |
| wheel_damping_override_active | 0 | 2000 | Never activated |

**Analysis:** RECENTER state was never reached during this 2000-step run. The improved drift performance (vs APCR1h) comes from matching APCR1h's position authority, not from the APCR1n-specific features.

---

## Window Analysis (500-step)

| Window | max \|e\| | mean \|e\| | outside ±0.15 |
|--------|-----------|------------|---------------|
| 0-500 | 0.1714 m | 0.0702 m | 10.6% |
| 500-1000 | 0.1090 m | 0.0518 m | 0.0% |
| 1000-1500 | 0.1186 m | 0.0654 m | 0.0% |
| 1500-2000 | 0.1188 m | 0.0559 m | 0.0% |

The first 500 steps show higher drift due to startup transient. After stabilization, drift remains well within ±0.15 m band.

---

## Final State

- **CoM height range:** 0.282 - 0.295 m
- **Robot pitch range:** -1.0 - 7.8 deg
- **Robot roll range:** 0.0 - 0.8 deg
- **Final drift error:** 0.0035 m (excellent recovery)
- **Fall status:** None (survived all 2000 steps)

---

## Root Cause: Initial Profile Mismatch

The initial APCR1n profile design confused APCR1m's parameters with APCR1h's parameters:

| Parameter | APCR1m | APCR1h (correct) | APCR1n (initial) |
|-----------|--------|------------------|------------------|
| position_cap | 3.0 Nm | 4.0 Nm | 3.0 Nm ❌ |
| wheel_damping | 5.0 Nm | 1.42 Nm | Not set ❌ |

The design doc referenced:
> "APCR1m wheel damping = 5.0 Nm vs APCR1h = 1.42 Nm (3.5x larger)"
> "APCR1m position cap ±3 Nm"

But the fix was intended to be based on APCR1h, not APCR1m. The position cap of 3.0 Nm reduced authority by 25% compared to APCR1h.

---

## Validation Results

| Phase | Result | Notes |
|-------|--------|-------|
| 1000-step (initial) | **FAIL** | max \|e\| = 0.246 m (2.4x worse than APCR1h) |
| 1000-step (corrected) | **PASS** | max \|e\| = 0.171 m (3.4% better than APCR1h) |
| 2000-step (corrected) | **PASS** | max \|e\| = 0.171 m, no fall |

---

## Files Modified

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added 14 new dataclass fields to SagittalAuthoritySchedule
   - Added APCR1n recenter priority logic in compute method
   - Added 17 APCR1n telemetry fields to diagnostics

2. `scripts/simulate_hierarchical_controller.py`
   - Added APCR1n profile to SAGITTAL_AUTHORITY_PROFILES dict
   - Added APCR1n to profile choices
   - Added APCR1n telemetry column initialization
   - Added APCR1n telemetry field population
   - Fixed profile parameters to match APCR1h base

3. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Added 12 APCR1n tests (all 270 tests pass)

---

## Recommendations

1. **Use APCR1n with corrected profile** (continuous_max_position_tau=True, max_position_tau_nominal=4.0)

2. **Monitor RECENTER activation** in future runs - the recenter priority features are designed to activate during high-drift events

3. **Consider tightening safety gates** if position cap boost should activate more frequently:
   - `recenter_priority_safe_min_com_z=0.27` may be too high
   - `recenter_priority_safe_roll_rad=0.15` may be too tight

4. **Future testing:** Run APCR1n under conditions that trigger RECENTER (higher drift scenarios) to validate the wheel damping override and position cap boost features

---

## Conclusion

APCR1n_recenter_priority_torque_boost is **validated** with the corrected profile. The initial parameter mismatch has been fixed, and the profile now outperforms APCR1h baseline:
- 3.4% better max drift
- 25.6% better P2P
- 7.0 pp fewer band violations
- 2000-step survival without fall

The APCR1n-specific features (wheel damping override, position cap boost) did not activate during this benign run. They remain available for more challenging scenarios where RECENTER would be triggered.