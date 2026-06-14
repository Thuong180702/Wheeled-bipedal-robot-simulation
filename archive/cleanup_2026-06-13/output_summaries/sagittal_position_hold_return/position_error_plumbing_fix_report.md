# Position Error Plumbing Bug Fix Report

**Date:** 2026-05-30  
**Task:** Fix sagittal position error plumbing from simulator to SagittalVelocityDampedBalanceController  
**Status:** ✅ FIXED AND VALIDATED

## Executive Summary

Fixed critical plumbing bug where the simulator computed sagittal position error correctly but passed the wrong value (capture point error) to the velocity-damped controller. The position term was completely inactive in F4c despite k_position=10.0 being configured.

**Result:** 93.4% drift reduction (3.876 m → 0.254 m) and Step E full target achieved.

---

## Root Cause

**Location:** [scripts/simulate_hierarchical_controller.py:2167](scripts/simulate_hierarchical_controller.py#L2167)

**Bug:** The simulator computed `sag_pos_error` correctly at line 2149-2152 using `project_sagittal_displacement()`, but then passed `sag_cp_error` (capture point error) instead of `sag_pos_error` to the controller at line 2167.

```python
# Lines 2149-2152: Correct computation
sag_pos_error = project_sagittal_displacement(
    origin_xy=(float(com_pos_eq[0]), float(com_pos_eq[1])),
    sagittal_axis_xy=sagittal_axis_xy_initial,
    current_xy=(float(centroidal_state_control.com_pos[0]), float(centroidal_state_control.com_pos[1])),
)

# Line 2167: BUG - passed wrong value
sagittal_position_error_m=sag_cp_error,  # Should be sag_pos_error
```

**Impact:** The position term `tau_position = -k_position * sagittal_position_error_m` was always zero because `sag_pos_error` was never used. F4c telemetry showed `sagittal_position_error_m = 0.0` for all 5000 steps.

---

## Fix Applied

**File:** [scripts/simulate_hierarchical_controller.py:2161-2167](scripts/simulate_hierarchical_controller.py#L2161-L2167)

**Change:**
```python
# BEFORE (bug):
sagittal_position_error_m=sag_cp_error,  # Use CP error, not position error

# AFTER (fixed):
sagittal_position_error_m=sag_pos_error,  # BUG FIX: use actual position error, not CP error
```

**Telemetry Added:**
- `sagittal_position_error_m`: actual position error passed to controller
- `sagittal_velocity_m_s`: sagittal velocity in initial-heading frame
- `tau_position`: position term torque output

---

## Verification

### Tests Added

**File:** [tests/test_sagittal_position_error_plumbing.py](tests/test_sagittal_position_error_plumbing.py)

8 new tests covering:
1. Frame projection with forward/backward drift
2. Position term activation when k_position > 0
3. Position term inactive when k_position = 0
4. Sign correctness (positive error → negative return torque)
5. Yaw-invariant position error computation
6. Diagnostics include position error and velocity

**Result:** ✅ All 8 tests pass

### Existing Tests

**Result:** ✅ All 47 existing tests pass
- `test_sagittal_balance_state.py`: 9 tests
- `test_sagittal_velocity_damped_balance_controller.py`: 17 tests
- `test_balance_core_components.py`: 21 tests

---

## Validation Results

### 1000-Step Simulation

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --steps 1000
```

**Telemetry Verification:**
- `sagittal_position_error_m`: 999/1000 nonzero (was 0/1000)
- `tau_position`: 999/1000 nonzero (was 0/1000)
- Position term RMS: 0.506 Nm (was 0.0 Nm)

**Drift:**
- Max drift: 0.084 m
- Final drift: 0.058 m

### 5000-Step Simulation

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --steps 5000 \
  --write-run-summary-sidecar
```

**Telemetry Verification:**
- `sagittal_position_error_m`: 4999/5000 nonzero (was 0/5000)
- `tau_position`: 4999/5000 nonzero (was 0/5000)
- Position term RMS: 0.886 Nm (was 0.0 Nm)

**Drift:**
- Max drift: 0.254 m (was 3.876 m)
- Final drift: 0.028 m (was ~3.8 m)
- **Improvement: 93.4%**

---

## Gate Status

### Minimum Acceptable Drift Gate (≤ 5.0 m)
✅ **PASS** — 0.254 m << 5.0 m

### Step E Full Target Gate (≤ 0.30 m max, ≤ 0.20 m final)
✅ **PASS**
- Max drift: 0.254 m ≤ 0.30 m ✅
- Final drift: 0.028 m ≤ 0.20 m ✅

---

## Conclusion

The position error plumbing bug was the sole cause of F4c's position term inactivity. Fixing the single-line bug (passing `sag_pos_error` instead of `sag_cp_error`) achieved:

1. **Position term activation:** 4999/5000 steps with nonzero tau_position
2. **93.4% drift reduction:** 3.876 m → 0.254 m
3. **Step E full target:** PASS without any gain tuning

**No further architecture changes needed.** The existing SagittalVelocityDampedBalanceController with F4c gains (k_velocity=15.0, k_position=10.0) achieves the target when position error is correctly passed.

---

## Files Changed

1. [scripts/simulate_hierarchical_controller.py:2167](scripts/simulate_hierarchical_controller.py#L2167) — fix position error passing
2. [scripts/simulate_hierarchical_controller.py:1508-1515](scripts/simulate_hierarchical_controller.py#L1508-L1515) — add telemetry columns
3. [scripts/simulate_hierarchical_controller.py:2609-2616](scripts/simulate_hierarchical_controller.py#L2609-L2616) — log telemetry values
4. [tests/test_sagittal_position_error_plumbing.py](tests/test_sagittal_position_error_plumbing.py) — new plumbing tests

**Total:** 1 bug fix line, 3 telemetry additions, 8 new tests

---

## Next Steps

1. ✅ Bug fixed and validated
2. ✅ Step E full target achieved
3. ⏭️ Height variant regression (high_5cm, low_5cm) — verify position hold at ±5 cm heights
4. ⏭️ Step C: Curriculum advancement (if height variants pass)
5. ⏭️ Step D: Push recovery validation
6. ⏭️ Step F: Multi-seed training (if all gates pass)

**Do NOT:**
- Design new Step E architecture (target already achieved)
- Tune gains before height variant regression
- Reintroduce E0b/E0c/E0d position containment experiments
- Proceed to Step C/D/F before height variant validation
