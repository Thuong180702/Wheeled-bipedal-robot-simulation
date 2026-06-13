# Step E Position Regulator Fix Report

**Date:** 2026-05-31  
**Fix:** Add explicit support-position velocity damping  
**Implementation:** SagittalVelocityDampedBalanceController with k_support_velocity parameter

---

## Executive Summary

**Fix Applied:** Added explicit support-center velocity damping term to prevent transient position excursions during nominal standing.

**Root Cause Addressed:** Primary cause C (position_gain_or_authority_insufficient) and secondary cause D (missing_position_velocity_regulation).

**Implementation Status:** Complete. Code changes, tests, and telemetry updates applied.

---

## Fix Selection Rationale

### Why Support Velocity Damping?

**Root cause analysis identified:**
1. **Primary:** Position authority saturated (tau_position clipped from -11.9 Nm to -3.0 Nm)
2. **Secondary:** Support velocity reached +0.397 m/s with no direct damping

**Fix B (support velocity damping) was selected because:**
- Directly addresses the secondary cause (missing velocity regulation)
- Prevents velocity buildup that leads to position saturation
- Preserves position authority limit (no destabilization risk)
- Physics-based standard control technique
- Simple, testable, and conservative

**Rejected alternatives:**
- **Fix A (repair capture gate):** Gate logic is correct; not relevant to this failure
- **Fix C (integral term):** No persistent bias; final error is small
- **Fix D (position reference governor):** More complex; adds state and tuning burden
- **Fix E (increase position authority):** Risky; could destabilize pitch balance

---

## Implementation Details

### Control Law Addition

**New term:**
```python
support_position_velocity_m_s = (current_support_position_error_m - prev_support_position_error_m) / dt
tau_support_velocity = -k_support_velocity * support_position_velocity_m_s
```

**Updated control law:**
```python
tau_common = wheel_torque_sign * (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity +  # NEW TERM
    tau_position + tau_cp + tau_com_vy
)
```

**Sign convention:**
- Positive support_position_velocity (forward drift) → negative tau_support_velocity (opposes drift)
- Negative support_position_velocity (backward drift) → positive tau_support_velocity (opposes drift)

### Parameter Addition

**New parameter:** `k_support_velocity` (N·s/m)
- Default: 0.0 (disabled for backward compatibility)
- Proposed initial value: 10.0 N·s/m
- CLI argument: `--vd-k-support-velocity`

**Expected behavior at proposed gain:**
- At peak velocity (0.397 m/s): tau_support_velocity = -3.97 Nm
- Comparable to max_position_tau (3.0 Nm) but applied during velocity buildup
- Should prevent velocity from reaching 0.397 m/s in first place

### Code Changes

**Files modified:**
1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added `k_support_velocity` parameter to `__init__`
   - Added `dt` parameter for velocity computation
   - Added `prev_support_position_error_m` state variable
   - Added support velocity computation in `compute()`
   - Added `tau_support_velocity` term to control law
   - Updated diagnostics to include new signals

2. `scripts/simulate_hierarchical_controller.py`
   - Added `--vd-k-support-velocity` CLI argument
   - Passed `k_support_velocity` to controller instantiation
   - Passed `dt=control_dt` to controller
   - Added telemetry logging for `support_position_velocity_m_s`, `tau_support_velocity`, `k_support_velocity`

3. `tests/test_support_velocity_damping.py` (new file)
   - 10 unit tests covering velocity computation, sign convention, torque composition
   - All tests pass

### Telemetry Updates

**New telemetry columns:**
- `support_position_velocity_m_s`: Rate of change of support-center position error
- `tau_support_velocity`: Support velocity damping torque
- `k_support_velocity`: Gain value for logging/verification

---

## Test Results

### Unit Tests

**New tests (test_support_velocity_damping.py):**
- ✅ `test_support_velocity_computation_forward_drift`
- ✅ `test_support_velocity_computation_backward_drift`
- ✅ `test_tau_support_velocity_opposes_forward_drift`
- ✅ `test_tau_support_velocity_opposes_backward_drift`
- ✅ `test_tau_support_velocity_zero_when_disabled`
- ✅ `test_tau_support_velocity_included_in_total_torque`
- ✅ `test_support_velocity_damping_with_position_control`
- ✅ `test_diagnostics_include_k_support_velocity`
- ✅ `test_no_wbc_no_legacy_sources`
- ✅ `test_kp_cp_remains_disabled`

**Result:** 10/10 passed

**Existing tests (test_sagittal_velocity_damped_balance_controller.py):**
- ✅ All 25 existing tests still pass
- No regressions introduced

---

## Validation Protocol

### Command Structure

**Baseline (no support velocity damping):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps <N> \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 0.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```

**With support velocity damping:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps <N> \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```

### Validation Sequence

**V1:** 500 steps nominal (quick smoke test)
**V2:** 1000 steps nominal
**V3:** 2000 steps nominal
**V4:** 5000 steps nominal (full validation)

**If V4 passes hard minimum:**
- V5: 500 steps high_5cm
- V6: 500 steps low_5cm

### Acceptance Criteria

**Preferred gate:**
- max |support_position_error_m| ≤ 0.10 m
- final support_position_error_m ≤ 0.05 m
- Height variants pass

**Fallback gate:**
- max |support_position_error_m| ≤ 0.15 m
- final support_position_error_m ≤ 0.10 m
- Height variants pass

**Hard minimum gate:**
- max |support_position_error_m| ≤ 0.30 m
- final support_position_error_m ≤ 0.10 m
- No posture regression
- No WBC active
- No ownership violations

### Metrics to Report

**Position control:**
- support_position_error_m: min, max, final
- support_position_velocity_m_s: min, max, mean, std
- tau_position_raw: min, max
- tau_position_clipped: min, max
- tau_support_velocity: min, max, mean, std
- tau_position saturation rate

**Posture/stance:**
- com_z_m: min, max, range
- pitch/roll/yaw: min, max, range
- hip_roll common-mode drift
- joint_error_norm: max

**Controller integrity:**
- tau_wbc_norm: should be 0.0
- ownership_violation_count: should be 0
- hidden_torque_norm: should be 0.0
- kp_cp: should be 0.0
- active sagittal controller: should be "velocity-damped"

---

## Expected Improvements

### Before Fix (Baseline)

**From telemetry_1780208317.csv (k_support_velocity = 0.0):**
- Max support_position_error_m: **0.595 m** at step 1360
- Final support_position_error_m: 0.039 m
- Peak support_position_velocity: **+0.397 m/s**
- tau_position saturation: 558/2000 steps (27.9%)
- **All position gates FAILED**

### After Fix (Expected)

**With k_support_velocity = 10.0:**
- Max support_position_error_m: **< 0.30 m** (hard minimum target)
- Final support_position_error_m: < 0.10 m
- Peak support_position_velocity: **< 0.15 m/s** (damped)
- tau_position saturation: < 10% (reduced)
- **Hard minimum gate PASS** (minimum requirement)

**Optimistic target (if gain is well-tuned):**
- Max support_position_error_m: **< 0.10 m** (preferred gate)
- Final support_position_error_m: < 0.05 m
- Peak support_position_velocity: < 0.05 m/s
- **Preferred gate PASS**

---

## Gain Tuning Guidance

### Initial Conservative Gain

**k_support_velocity = 10.0 N·s/m**
- At 0.397 m/s: produces -3.97 Nm damping
- Comparable to max_position_tau (3.0 Nm)
- Should significantly reduce velocity buildup

### If Hard Minimum Still Fails

**Try k_support_velocity = 15.0 or 20.0 N·s/m**
- Stronger damping for faster velocity reduction
- Monitor for oscillation or instability

### If Preferred Gate Fails But Hard Minimum Passes

**Try k_support_velocity = 12.0 or 15.0 N·s/m**
- Moderate increase for tighter position hold
- Balance between damping and smoothness

### If Oscillation Occurs

**Reduce k_support_velocity to 5.0 or 7.5 N·s/m**
- Weaker damping to avoid over-correction
- May need to increase k_position instead

---

## Verification Checklist

**Before declaring Step E complete:**

- [ ] V1 (500 steps) completes without crash
- [ ] V2 (1000 steps) shows reduced max position error vs baseline
- [ ] V3 (2000 steps) shows reduced velocity buildup vs baseline
- [ ] V4 (5000 steps) passes at least hard minimum gate
- [ ] Telemetry confirms support_position_velocity_m_s is logged
- [ ] Telemetry confirms tau_support_velocity is logged and nonzero
- [ ] Telemetry confirms tau_wbc_norm = 0.0 (WBC disabled)
- [ ] Telemetry confirms ownership_violation_count = 0
- [ ] Telemetry confirms kp_cp = 0.0
- [ ] Posture metrics remain acceptable (no regression)
- [ ] Height variants (high_5cm, low_5cm) pass if V4 passes

**If preferred gate passes:**
- [ ] Generate final validation summary
- [ ] Update Step E status to PASS
- [ ] Proceed to Step C (height recovery)

**If only hard minimum passes:**
- [ ] Document as partial success
- [ ] Note gain tuning needed for preferred gate
- [ ] Decide whether to tune further or proceed to Step C

**If hard minimum fails:**
- [ ] Analyze telemetry for failure mode
- [ ] Try alternative gains (15.0, 20.0)
- [ ] Consider hybrid fix (velocity damping + increased position authority)
- [ ] Do NOT proceed to Step C

---

## Integration Notes

### Backward Compatibility

**Default behavior unchanged:**
- `k_support_velocity` defaults to 0.0 (disabled)
- Existing simulations without `--vd-k-support-velocity` flag behave identically
- No breaking changes to existing code

### CLI Usage

**Enable support velocity damping:**
```bash
--vd-k-support-velocity 10.0
```

**Disable support velocity damping (default):**
```bash
--vd-k-support-velocity 0.0
```
or omit the flag entirely.

### Telemetry Compatibility

**New columns added:**
- `support_position_velocity_m_s`
- `tau_support_velocity`
- `k_support_velocity`

**Existing columns unchanged:**
- All existing telemetry columns remain
- CSV format compatible with existing analysis scripts

---

## Next Steps

1. **Run validation sequence V1-V4**
2. **Analyze telemetry for each run**
3. **Compare before/after position error metrics**
4. **Generate final validation summary**
5. **If hard minimum passes:** Proceed to Step C
6. **If hard minimum fails:** Tune gain or investigate further

---

## Files Generated

- `step_e_position_regulator_fix_report.md` (this file)
- `step_e_position_regulator_fix_report.json` (next)
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` (modified)
- `scripts/simulate_hierarchical_controller.py` (modified)
- `tests/test_support_velocity_damping.py` (new)

---

## Status

**Implementation:** ✅ Complete  
**Unit Tests:** ✅ 10/10 passed  
**Regression Tests:** ✅ 25/25 passed  
**Validation:** 🔄 In progress (V1 running)  
**Step E Status:** ⏳ Pending validation results
