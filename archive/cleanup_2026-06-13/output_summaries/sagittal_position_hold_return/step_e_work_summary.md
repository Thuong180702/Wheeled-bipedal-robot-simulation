# Step E Position Regulator Work Summary

**Date:** 2026-05-31  
**Session:** Root cause diagnosis and fix implementation

---

## Work Completed

### 1. Root Cause Diagnosis ✅

**Task 1: Capture Gate Analysis**
- Analyzed telemetry around peak error (step 1360)
- Found capture gate never activated (0/2000 steps)
- Identified reason: capture point stayed within ±0.064m (below 0.10m threshold)
- **Conclusion:** Gate logic is correct but irrelevant to this failure mode

**Task 2: Root Cause Classification**
- **Primary cause:** C. position_gain_or_authority_insufficient
  - tau_position saturated at -3.0 Nm (wanted -11.9 Nm, 74% reduction)
  - Saturated for 558/2000 steps (27.9%)
  - Effective gain dropped from 20.0 to 5.04 N/m during saturation
- **Secondary cause:** D. missing_position_velocity_regulation
  - Support velocity reached +0.397 m/s with no direct damping
  - No explicit support-center velocity damping term exists
  - Position error grew because velocity was not directly opposed

**Task 3: Control Law Review**
- Documented current control law and gains
- Explained why final error is small (0.039m) but transient error is large (0.595m)
- Controller is good steady-state regulator but poor transient limiter

**Task 4: Fix Selection**
- Selected **Fix B: Add explicit support-position velocity damping**
- Rationale: Directly addresses secondary cause, preserves position authority, physics-based, testable
- Rejected alternatives: capture gate repair (not relevant), integral term (no bias), position reference governor (too complex), increase authority (risky)

### 2. Implementation ✅

**Code Changes:**
1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Added `k_support_velocity` parameter (default 0.0)
   - Added `dt` parameter for velocity computation
   - Added `prev_support_position_error_m` state variable
   - Implemented support velocity computation: `(current - prev) / dt`
   - Added `tau_support_velocity = -k_support_velocity * support_position_velocity`
   - Included new term in control law
   - Updated diagnostics

2. `scripts/simulate_hierarchical_controller.py`
   - Added `--vd-k-support-velocity` CLI argument
   - Added parameter to `build_balance_core_controllers()` function
   - Passed parameter to controller instantiation
   - Added telemetry logging for new signals

3. `tests/test_support_velocity_damping.py` (new)
   - 10 comprehensive unit tests
   - All tests pass ✅

**Telemetry Updates:**
- `support_position_velocity_m_s`: Rate of change of support position error
- `tau_support_velocity`: Support velocity damping torque
- `k_support_velocity`: Gain value for verification

### 3. Testing ✅

**Unit Tests:**
- New tests: 10/10 passed ✅
- Existing tests: 25/25 passed ✅
- No regressions introduced

**Test Coverage:**
- Support velocity computation (forward/backward drift)
- Torque sign convention (opposes drift)
- Torque composition with position control
- Disabled state (k_support_velocity = 0.0)
- Diagnostics completeness
- No WBC/legacy sources
- kp_cp remains disabled

### 4. Documentation ✅

**Reports Generated:**
1. `step_e_position_regulator_root_cause_report.md`
   - Capture gate diagnosis
   - Root cause classification with evidence
   - Control law review
   - Fix selection rationale

2. `step_e_position_regulator_root_cause_report.json`
   - Structured data for analysis

3. `step_e_position_regulator_fix_report.md`
   - Implementation details
   - Test results
   - Validation protocol
   - Gain tuning guidance

4. `step_e_position_regulator_fix_report.json` (pending)

---

## Current Status

**Implementation:** ✅ Complete  
**Unit Tests:** ✅ All passed  
**Validation:** 🔄 In progress (V1 running)

---

## Next Steps

### Immediate (In Progress)

1. ✅ V1 (500 steps) - Running now
2. ⏳ Analyze V1 telemetry
3. ⏳ V2 (1000 steps)
4. ⏳ V3 (2000 steps)
5. ⏳ V4 (5000 steps)

### After V4 Completes

**If hard minimum passes (max ≤ 0.30m):**
- Run V5/V6 height variants
- Generate final validation summary
- Update Step E status
- Proceed to Step C

**If hard minimum fails:**
- Analyze failure mode
- Try alternative gains (15.0, 20.0)
- Consider hybrid fix
- Do NOT proceed to Step C

---

## Key Findings

### What We Learned

1. **Capture gate is not the problem**
   - Gate logic is correct
   - Capture point stayed within threshold
   - This is a support-drift problem, not a capture-conflict problem

2. **Position authority saturation is the primary bottleneck**
   - max_position_tau = 3.0 Nm is insufficient for large errors
   - At 0.595m error, controller wanted 11.9 Nm but could only apply 3.0 Nm
   - Effective gain dropped 75% during saturation

3. **Missing velocity damping allowed error to grow**
   - Support velocity reached 0.397 m/s unchecked
   - No direct opposition to support-center velocity
   - High velocity + saturated position torque = continued drift

4. **Controller works well for small errors**
   - Final error 0.039m is acceptable
   - Proves controller CAN regulate position when error is small
   - Problem is transient excursion prevention, not steady-state regulation

### What We Fixed

**Added explicit support-center velocity damping:**
- `tau_support_velocity = -k_support_velocity * support_position_velocity`
- Directly opposes support drift velocity
- Prevents velocity buildup that leads to position saturation
- Conservative initial gain: 10.0 N·s/m

**Expected improvement:**
- Reduce peak velocity from 0.397 m/s to < 0.15 m/s
- Reduce max position error from 0.595m to < 0.30m (hard minimum target)
- Reduce position authority saturation from 27.9% to < 10%

---

## Technical Details

### Control Law Before Fix

```python
tau_common = wheel_torque_sign * (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_position + tau_cp + tau_com_vy
)
```

### Control Law After Fix

```python
support_position_velocity = (current_position_error - prev_position_error) / dt
tau_support_velocity = -k_support_velocity * support_position_velocity

tau_common = wheel_torque_sign * (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity +  # NEW TERM
    tau_position + tau_cp + tau_com_vy
)
```

### Gains

**Unchanged:**
- kp_pitch = 50.0
- kd_pitch = 10.0
- k_velocity = 15.0
- k_position = 20.0
- max_position_tau = 3.0

**New:**
- k_support_velocity = 10.0 (proposed initial value)

---

## Files Modified

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
2. `scripts/simulate_hierarchical_controller.py`
3. `tests/test_support_velocity_damping.py` (new)

## Files Generated

1. `outputs/sagittal_position_hold_return/step_e_position_regulator_root_cause_report.md`
2. `outputs/sagittal_position_hold_return/step_e_position_regulator_root_cause_report.json`
3. `outputs/sagittal_position_hold_return/step_e_position_regulator_fix_report.md`
4. `outputs/sagittal_position_hold_return/step_e_work_summary.md` (this file)

---

## Validation Commands

**Baseline (no support velocity damping):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 0.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```

**With support velocity damping (proposed fix):**
```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 2000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-capture-gate \
  --vd-capture-gate-use-cp
```

---

## Status Summary

✅ **Root cause identified:** Position authority insufficient + missing velocity damping  
✅ **Fix implemented:** Support-center velocity damping term  
✅ **Tests passing:** 10/10 new + 25/25 existing  
✅ **Documentation complete:** Root cause report + fix report  
🔄 **Validation in progress:** V1 (500 steps) running  
⏳ **Step E status:** Pending validation results
