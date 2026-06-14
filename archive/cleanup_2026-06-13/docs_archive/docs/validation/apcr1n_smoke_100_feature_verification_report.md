# APCR1n Phase 3: 100-Step Smoke Test Feature Verification Report

**Date:** 2026-06-11  
**Test:** APCR1n 100-step smoke test with telemetry validation  
**Profile:** `APCR1n_recenter_priority_torque_boost`  
**Setup:** `low_0p300`  
**Classification:** `APCR1N_SMOKE_100_TELEMETRY_PASS`

---

## Executive Summary

✅ **PASS**: All APCR1n telemetry columns present in CSV  
✅ **PASS**: Startup guard active for all 100 steps (correct behavior)  
✅ **PASS**: No feature activation during startup guard (correct behavior)  
✅ **PASS**: Runtime config consumed correctly  
⚠️  **NOTE**: `effective_max_position_tau=6.0` from height scheduling (expected at low_0p300)

---

## Test Configuration

```yaml
Controller Mode: balance-core
Sagittal Controller: velocity-damped
VD Profile: APCR1n_recenter_priority_torque_boost
Height Variant: low_0p300 (target CoM Z = 0.300m)
Steps: 100
Telemetry Decimation: 1
Startup Guard Steps: 100
```

---

## Telemetry Column Validation

### Required APCR1n Columns (16 total)

All 16 APCR1n telemetry columns **present and populated**:

1. ✅ `apcr1n_recenter_priority_active`
2. ✅ `apcr1n_startup_guard_active`
3. ✅ `apcr1n_wheel_damping_override_active`
4. ✅ `apcr1n_wheel_damping_scale`
5. ✅ `apcr1n_wheel_damping_before`
6. ✅ `apcr1n_wheel_damping_after`
7. ✅ `apcr1n_wheel_damping_fights_drift`
8. ✅ `apcr1n_position_cap_boost_active`
9. ✅ `apcr1n_position_cap_current`
10. ✅ `apcr1n_tau_position_raw`
11. ✅ `apcr1n_tau_position_after_cap`
12. ✅ `apcr1n_position_saturated`
13. ✅ `apcr1n_safety_gate_pass`
14. ✅ `apcr1n_final_torque_direction_correct`
15. ✅ `apcr1n_final_torque_fights_drift`
16. ✅ `apcr1n_physical_drift_column_used`

---

## Runtime Config Consumption

### APCR1n Config Values

```yaml
continuous_max_position_tau: True
max_position_tau_nominal: 4.0
velocity_damping_scale: 1.10
position_cap_normal_nm: 4.0
position_cap_recenter_nm: 5.0
position_cap_emergency_nm: 6.0
vd_wheel_damping_recenter_scale: 0.30
```

### Runtime Consumption Verification

| Config Field | Expected | Observed | Status |
|---|---|---|---|
| `velocity_damping_scale` | 1.10 | 1.10 | ✅ PASS |
| `max_position_tau_nominal` | 4.0 | 4.0 (in scheduler) | ✅ PASS |
| `effective_max_position_tau` | varies by height | 6.0 | ✅ PASS (height-scheduled) |
| `apcr1n_position_cap_current` | 4.0 or 6.0 | 6.0 | ✅ PASS (initial cap) |

### Height Scheduling Explanation

At `low_0p300` (z_ref=0.300m), the height scheduler computes:
- `z_low = 0.30`, `z_high = 0.50`
- `k_nominal = 4.0`, `k_low_max = 6.0`
- At z_ref = z_low, scheduler uses **k_low_max = 6.0**

This is **expected and correct** behavior. The 6.0 Nm cap comes from **legacy height scheduling**, not APCR1n emergency mode.

---

## Startup Guard Behavior

### Startup Guard Status

```
Total steps: 100
Steps with startup_guard_active=True: 100 (100%)
Steps with startup_guard_active=False: 0 (0%)
```

✅ **Correct**: Startup guard active for all 100 steps (< 100-step threshold)

### Feature Activation During Startup Guard

| Feature | Activations | Expected | Status |
|---|---|---|---|
| Recenter Priority | 0 | 0 | ✅ PASS |
| Wheel Damping Override | 0 | 0 | ✅ PASS |
| Position Cap Boost | 0 | 0 | ✅ PASS |

✅ **Correct**: No APCR1n torque-changing features activated during startup guard

---

## Simulation Health

```
Status: Completed full simulation without falling
Terminated: False
CoM height range: 0.293 - 0.295 m
Robot pitch range: -0.0 - 7.7 deg
Robot roll range: 0.0 - 0.2 deg
Max torques:
  Hip roll: 9.63 Nm
  Wheels: 1.66 Nm
  Total: 8.88 Nm
```

✅ Robot remained stable for all 100 steps  
✅ No crashes, NaNs, or failures  
✅ No WBC ownership violations  
✅ No hidden torque applied

---

## Telemetry File

**Path:** `outputs/hierarchical_controller_sim/telemetry_1781185346.csv`

**Stats:**
- Total columns: 594
- Populated columns: 580
- Empty columns: 14
- Data rows: 100
- APCR1n columns: 16 (all populated)

---

## Classification

**APCR1N_SMOKE_100_TELEMETRY_PASS**

### Pass Criteria Met

1. ✅ All 16 APCR1n telemetry columns present
2. ✅ Columns populated with valid data
3. ✅ Startup guard worked correctly (100/100 steps)
4. ✅ No feature activation during startup guard
5. ✅ Runtime config consumed:
   - `velocity_damping_scale=1.10` ✅
   - `max_position_tau_nominal=4.0` ✅ (used in height scheduler)
   - Height-scheduled cap=6.0 Nm ✅ (expected at low_0p300)
6. ✅ Simulation completed without failure
7. ✅ No WBC violations

---

## Next Steps

Proceed to **Phase 4: Activation Trigger Test**

Purpose: Prove APCR1n features CAN activate under eligible conditions:
- Disable startup guard (step > 100)
- Create high drift (abs(e) > 0.08)
- Ensure safety gates pass
- Verify feature activation in actual controller runtime

---

## Appendix: Sample Telemetry (Step 10)

```
apcr1n_recenter_priority_active: False
apcr1n_startup_guard_active: True
apcr1n_wheel_damping_override_active: False
apcr1n_position_cap_boost_active: False
apcr1n_position_cap_current: 6.0
effective_max_position_tau: 6.0
effective_velocity_damping_scale: 1.1
```

**Analysis:** Startup guard active, all boost features inactive (correct).
