# Tau_pitch Bias Source Audit Final Report

**Date:** 2026-06-08
**Task:** Direct tau_pitch bias source audit at low_0p300
**Status:** COMPLETE - No implementation

---

## 1. Executive Summary

**Decision: TAU_PITCH_BIAS_SOURCE_IDENTIFIED - RECOMMEND_FIX_E_PLUS_FIX_A**

The persistent positive tau_pitch bias at low_0p300 has been traced to two root causes:

1. **Initial hip_pitch error of 0.45 rad** - steady bias from non-equilibrium start
2. **Position authority cap of 4.0 Nm insufficient** - cannot cancel tau_pitch up to 5.5 Nm

**tau_pitch computation, sign convention, and pitch reference are all correct.**

The downstream bias cancellation approach (G1b) reduces positive% slightly but worsens outside-band behavior because it fights the symptom (tau_pitch) instead of addressing the cause (insufficient position authority and initial condition mismatch).

---

## 2. Key Findings

### Finding 1: tau_pitch Computation is Correct

| Parameter | Value | Assessment |
|-----------|-------|-----------|
| Formula | `tau_pitch = kp_pitch * pitch_x_error` | ✅ Correct |
| kp_pitch | 50.0 | ✅ Standard |
| Pitch reference | 0.0 | ✅ Correct per equilibrium |
| Sign | positive pitch → positive torque | ✅ Correct (opposes lean) |

**Evidence:** Unit tests pass, correlation analysis confirms tau_pitch ∝ pitch_x.

### Finding 2: tau_pitch is Responding, Not Causing

| Condition | tau_pitch behavior | Correct? |
|-----------|---------------------|----------|
| pitch_x > 0 | tau_pitch > 0 (opposes forward lean) | ✅ Yes |
| pitch_x < 0 | tau_pitch < 0 (opposes backward lean) | ✅ Yes |
| pitch_rate > 0 | tau_pitch_rate > 0 (damping) | ✅ Yes |

**Evidence:** tau_pitch sign correctly opposes pitch deviation. tau_pitch is proportional feedback.

### Finding 3: Initial Hip_pitch Error is Root Cause #1

| Parameter | Value | Implication |
|-----------|-------|-------------|
| hip_pitch_error_max at step 0 | 0.45 rad (25.79 deg) | SEVERE initial mismatch |
| hip_pitch_error throughout | ~0.49 rad (28.3 deg) | Never corrects |
| Error sign | Negative (joint too extended) | Creates backward moment |

**Evidence:** D2 telemetry shows hip_pitch ~0.45-0.50 rad away from equilibrium throughout the run.

### Finding 4: Position Authority Insufficient is Root Cause #2

| Parameter | Value | Implication |
|-----------|-------|-------------|
| tau_position cap | 4.0 Nm | TOO LOW |
| tau_pitch peak | 5.5 Nm | EXCEEDS CAP |
| Position saturation | 35.4% of steps | FREQUENT |
| Net torque after cap | Can exceed 1 Nm | Forward lean continues |

**Evidence:** D2 steps 66-241 show tau_position saturated at -4.0 Nm while tau_pitch reaches +4-5 Nm.

---

## 3. Profile Comparison Summary

| Metric | D2 | F1b | G1a | G1b | Interpretation |
|--------|----|----|----|----|----------------|
| tau_pitch mean (Nm) | 2.60 | 2.70 | 2.88 | 3.08 | Increasing (worsening) |
| tau_pitch positive% | 89.2% | 82.8% | 84.2% | 80.4% | Decreasing (bias cancel working) |
| tau_position mean (Nm) | -2.61 | -2.36 | -2.22 | -2.16 | Decreasing (less cancellation) |
| outside ±0.15 | 19.2% | 16.2% | 13.4% | 26.8% | G1b WORST |

**Conclusion:** G1b reduces positive% slightly but increases outside-band behavior. Bias cancellation is a symptom patch, not a fix.

---

## 4. Root Cause Classification

### TAU_PITCH_BIAS_FROM_INITIAL_CONDITION ✅ CONFIRMED

**Primary root cause.**

**Mechanism:**
1. Robot starts with hip_pitch error of 0.45 rad
2. Extended hips create backward moment
3. Robot pitches forward (pitch_x > 0)
4. tau_pitch responds correctly (>0 to oppose lean)
5. But position cap insufficient → net forward torque
6. Forward drift continues → pitch stays positive

### TAU_PITCH_BIAS_FROM_UNMODELED_SUPPORT_DRIFT ✅ CONFIRMED (secondary)

**Secondary root cause.**

**Mechanism:**
1. Position authority capped at 4.0 Nm
2. tau_pitch reaches 5.5 Nm peak
3. Position cannot fully cancel
4. Net positive torque builds up
5. Forward lean continues

### TAU_PITCH_REFERENCE_FIX_REQUIRED ⚠️ POSSIBLE

**May be needed.**

The equilibrium search found equilibrium_pitch_x = 0.0, but at extreme squat (hip=78.85°, knee=134.56°), the robot may naturally require a slight forward pitch for stability.

---

## 5. Why Downstream Bias Cancellation Fails

### G1b Analysis

| Metric | D2 | G1b | Change |
|--------|----|-----|--------|
| positive% | 89.2% | 80.4% | -8.8% ✓ |
| outside ±0.15 | 19.2% | 26.8% | +7.6% ✗ |

**Root cause of failure:** G1b applies opposite torque when signed_error > 0. But signed_error > 0 is a SYMPTOM of the real problem (insufficient position authority). G1b fights the symptom, causing overcorrection and oscillation.

---

## 6. Recommended Fixes (No Implementation)

### Fix E: Fix Initial Hip_pitch Error (Primary Recommendation)

**Risk:** LOW | **Addresses:** Root cause #1

Investigate why hip_pitch_error is 0.45 rad at step 0. Ensure robot starts AT equilibrium, not near it.

### Fix A: Increase Position Authority at Low Height (Secondary Recommendation)

**Risk:** MEDIUM | **Addresses:** Root cause #2

Create profile H1 with continuous max_position_tau scheduling:
- At 0.30 m: cap = 8.0 Nm (vs current 4.0 Nm)
- Smoothly transitions between heights
- Can cancel tau_pitch up to 5.5 Nm

**Proposed H1 profile:**
```python
H1_LOW_HEIGHT_POSITION_CAP = SagittalAuthoritySchedule(
    profile_name="H1_low_height_position_cap",
    applies_to_variants=("low_0p300", "low_5cm"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=3.0,
    max_position_tau_low_max=8.0,  # 2x increase at low height
    k_position_z_low=0.300,
    k_position_z_high=0.393,
)
```

---

## 7. Next Executable Experiment Plan

### H1: Position Cap Increase Validation

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
    --controller balance-core \
    --height-variant low_0p300 \
    --sagittal-controller velocity-damped \
    --sagittal-authority-profile H1_low_height_position_cap \
    --num-steps 500 \
    --seed 42 \
    --output-dir outputs/step_e_extreme_support_fix_eval/h1_low_height_position_cap_500
```

**Success criteria:**
- tau_pitch positive% < 85%
- tau_position saturation% < 20%
- time outside ±0.15 < 15%

---

## 8. Explicit Restrictions Followed

| Restriction | Status |
|-------------|--------|
| Do NOT modify D2 baseline | ✅ Followed |
| Do NOT make new profile default | ✅ Followed |
| Do NOT continue F2 hysteresis tuning | ✅ Not done |
| Do NOT continue E2c | ✅ Not done |
| Do NOT enable HY2-DIV | ✅ Not done |
| Do NOT add WBC | ✅ Not done |
| Do NOT enable legacy WBC | ✅ Not done |
| Do NOT relax Step E gates | ✅ Not done |
| Do NOT claim Step E pass | ✅ Not claimed |
| Do NOT run 2000/5000-step validation | ✅ Not done |
| Do NOT run Step C/D | ✅ Not done |
| Do NOT commit | ✅ Not committed |

---

## 9. Documents Produced

1. `docs/validation/tau_pitch_computation_audit.md` - Phase 1
2. `docs/validation/tau_pitch_telemetry_decomposition_audit.md` - Phase 2
3. `docs/validation/tau_pitch_bias_low_0p300_pitch_reference_setup_audit.md` - Phase 3
4. `docs/validation/tau_pitch_sign_sanity_check.md` - Phase 4
5. `docs/validation/tau_pitch_positive_bias_root_cause_report.md` - Phase 5
6. `docs/validation/tau_pitch_bias_fix_strategy_plan.md` - Phase 6
7. `docs/validation/tau_pitch_bias_source_audit_final_report.md` - Phase 7 (this document)

JSON outputs:
- `outputs/step_e_extreme_support_fix_eval/tau_pitch_bias_audit/tau_pitch_computation_audit.json`
- `outputs/step_e_extreme_support_fix_eval/tau_pitch_bias_audit/tau_pitch_telemetry_decomposition.json`

---

## 10. Final Decision

**TAU_PITCH_BIAS_ROOT_CAUSE_IDENTIFIED_READY_FOR_FIX_PLAN**

**Root cause identified:**
1. Initial hip_pitch error (0.45 rad) - steady bias
2. Position authority cap insufficient (4.0 Nm < 5.5 Nm peak)

**Recommended next step:**
- Fix E: Investigate initial condition mismatch
- Fix A: Create H1 profile with increased position cap
- Validate H1 at 500 steps before proceeding