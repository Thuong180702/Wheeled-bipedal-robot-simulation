# Position Hold Refinement Report

**Date:** 2026-05-30  
**Configuration:** F4c (k_velocity=15.0, k_position=10.0) — bug-fixed  
**Status:** REFINEMENT TESTED — GATES NOT MET — ARCHITECTURAL ISSUE IDENTIFIED

---

## Executive Summary

The bug-fixed F4c achieves 93.4% drift reduction (3.876 m → 0.254 m) and passes the Step E full target (≤0.30 m). However, the new practical target of ±0.10 m (preferred) or ±0.15 m (fallback) is not met by any gain refinement tested.

**Root cause:** The max drift is a **transient overshoot** (Type A) at step 1666 (33% through the run), driven by a pitch spike to 11.8 deg. The pitch stabilization term (~10 Nm) overwhelms both velocity damping (-0.1 to -0.8 Nm) and position return (-1.4 to -2.5 Nm). This is an architectural coupling issue, not a tuning issue.

**Steady state is already excellent:** The last 20% of the run (steps 4000-5000) stays within ±0.033 m — well inside the ±0.10 m preferred target.

---

## Task 1: Drift Classification

**Type: A — Transient Overshoot**

| Metric | Value |
|--------|-------|
| Max drift | 0.254 m at step 1666 (33% through run) |
| Final drift | 0.028 m |
| Steady state range (last 20%) | 0.021–0.033 m |
| Zero crossings | 6 |
| Positive steps | 4867/5000 (97.3%) |

**Phase breakdown:**

| Phase | Steps | Mean pos error | Max pos error |
|-------|-------|---------------|---------------|
| Initial drift | 0–500 | 0.040 m | 0.081 m |
| Buildup | 500–1000 | 0.047 m | 0.084 m |
| Transient peak | 1000–1666 | 0.137 m | 0.254 m |
| Recovery | 1666–2500 | 0.141 m | 0.254 m |
| Return | 2500–3500 | 0.037 m | 0.096 m |
| Steady state | 3500–5000 | 0.026 m | 0.041 m |

**What drives the transient:**

During steps 1000–1666, pitch spikes to 11.8 deg. The pitch term (`sagittal_term_pitch`) reaches 5–10 Nm mean, while position return (`tau_position`) is only -1.4 to -2.5 Nm. The pitch term wins, driving the robot forward.

| Torque term | Transient mean (0–2000) | Steady mean (3000–5000) |
|-------------|------------------------|------------------------|
| `sagittal_term_pitch` | 3.96 Nm | 1.05 Nm |
| `sagittal_term_pitch_rate` | 0.08 Nm | 0.01 Nm |
| `sagittal_term_com_vy` | -0.14 Nm | -0.02 Nm |
| `tau_position` | -0.97 Nm | -0.26 Nm |
| `sagittal_balance_torque_final` | 0.06 Nm | 0.00 Nm |

**Steady state equilibrium:** Pitch settles at 1.2 deg, producing `sagittal_term_pitch` = 1.06 Nm, balanced by `tau_position` = -0.26 Nm. Net torque ≈ 0.

---

## Task 2: Sign and Frame Verification

**Status: CORRECT — no sign or frame issues found**

| Check | Result |
|-------|--------|
| Positive pos error → negative tau_position | PASS (mean -0.705 Nm when pos_error > 0.01) |
| Negative pos error → positive tau_position | PASS (mean +0.115 Nm when pos_error < -0.01) |
| Positive sagittal_vel → negative damping | PASS (mean -0.825 Nm when vel > 0.01) |
| Negative sagittal_vel → positive damping | PASS (mean +0.893 Nm when vel < -0.01) |
| Initial-heading-frame projection | CORRECT |
| Yaw drift contamination | LOW (correlation = -0.29, max yaw drift = 6.1 deg) |

No sign or frame issues. The bug fix is correct.

---

## Task 3: Controlled Gain Refinements

### Option A: k_position increase

| Config | Max drift | Final drift | Max pitch | Gate |
|--------|-----------|-------------|-----------|------|
| k_pos=10.0 (F4c) | 0.254 m | 0.028 m | 11.84 deg | FAIL |
| k_pos=12.5 (1000-step) | 0.091 m | 0.075 m | 4.60 deg | — |
| k_pos=15.0 (5000-step) | 0.236 m | 0.188 m | 12.24 deg | FAIL |

**Finding:** k_position=15.0 destabilizes the robot. Pitch range expands to -10.1 to +12.2 deg (vs -0.8 to +11.8 deg baseline). 2031 steps with pitch >5 deg, 906 steps with pitch >8 deg. Final drift increases to 0.188 m. Increasing k_position creates a conflict with pitch stabilization during transients.

### Option C: k_velocity increase

| Config | Max drift | Final drift | Max pitch | Improvement | Gate |
|--------|-----------|-------------|-----------|-------------|------|
| k_vel=15.0 (F4c) | 0.254 m | 0.028 m | 11.84 deg | baseline | FAIL |
| k_vel=17.5 | 0.249 m | 0.026 m | 11.68 deg | 2.2% | FAIL |
| k_vel=20.0 | 0.245 m | 0.026 m | 11.60 deg | 3.8% | FAIL |

**Finding:** Velocity damping shows diminishing returns. The pitch spike at step 1666 is 11.8 deg regardless of k_velocity. The pitch term (~10 Nm) overwhelms velocity damping (-0.1 to -0.8 Nm). Increasing k_velocity beyond 20.0 would not close the gap.

---

## Task 4: Validation Protocol

Best candidate: k_velocity=20.0, k_position=10.0 (3.8% improvement, most stable)

| Metric | Value |
|--------|-------|
| Max sagittal pos error | 0.245 m |
| Min sagittal pos error | -0.000 m |
| Final sagittal pos error | 0.026 m |
| Max abs sagittal pos error | 0.245 m |
| Max pitch | 11.60 deg |
| Pitch range | 0.0 to 11.6 deg |
| Roll range | -2.4 to 0.2 deg |
| CoM z range | 0.363–0.409 m |
| Wheel vel range | not extracted |
| Torque saturation | not elevated |

Height variant regression (high_5cm, low_5cm) not run — gates not met, no point running variants on a failing configuration.

---

## Task 5: Acceptance Gates

| Gate | Threshold | Best result | Status |
|------|-----------|-------------|--------|
| Preferred practical target | ±0.10 m max | 0.245 m | **FAIL** |
| Acceptable fallback | ±0.15 m max | 0.245 m | **FAIL** |
| Final drift | ≤0.05 m | 0.026 m | PASS |
| Stability (no pitch/roll regression) | no regression | k_pos=15.0 destabilizes | FAIL for k_pos=15.0 |

**Current result vs targets:**

| Metric | Old F4c (inactive) | Bug-fixed F4c | Best refined | Target |
|--------|-------------------|---------------|--------------|--------|
| Max drift | 3.876 m | 0.254 m | 0.245 m | ≤0.10 m |
| Final drift | ~3.8 m | 0.028 m | 0.026 m | ≤0.05 m |
| Improvement | — | 93.4% | 93.7% | — |

The final drift target is already met. The max drift target is not met by any gain refinement.

---

## Root Cause Analysis

The transient overshoot is caused by a **pitch spike at step 1666** that is driven by the WBC/posture system, not the sagittal controller. The sagittal controller's pitch stabilization term (`kp_pitch=50.0`) amplifies the pitch spike into a large forward torque.

**Why gain tuning cannot solve this:**

1. The pitch term at peak is ~10 Nm. Position return is ~2.5 Nm. The ratio is 4:1.
2. Increasing k_position to counteract this creates a conflict: position return drives wheels backward, pitch stabilization drives wheels forward. During the transient, this conflict amplifies pitch oscillations (k_pos=15.0 destabilizes).
3. Increasing k_velocity reduces the pitch spike by ~2-4% but cannot overcome a 4:1 torque ratio.

**What would actually help:**

- Investigate why pitch spikes to 11.8 deg at step 1666 (WBC/posture issue, not sagittal controller issue)
- Pitch rate limiting or pitch term saturation during transients
- Decoupling pitch stabilization from position return in the sagittal controller
- These are architectural changes, not gain tuning

---

## Comparison Summary

| Configuration | Max drift | Final drift | Status |
|---------------|-----------|-------------|--------|
| Old F4c (position term inactive) | 3.876 m | ~3.8 m | FAIL |
| Bug-fixed F4c (k_vel=15.0, k_pos=10.0) | 0.254 m | 0.028 m | PASS Step E, FAIL ±0.15 m |
| k_vel=17.5, k_pos=10.0 | 0.249 m | 0.026 m | FAIL ±0.15 m |
| k_vel=20.0, k_pos=10.0 | 0.245 m | 0.026 m | FAIL ±0.15 m |
| k_vel=15.0, k_pos=15.0 | 0.236 m | 0.188 m | FAIL (destabilized) |

---

## Verification

| Check | Status |
|-------|--------|
| No WBC changes | CONFIRMED |
| No E0b/E0c/E0d reintroduced | CONFIRMED |
| Torque ownership unchanged | CONFIRMED |
| Sagittal controllers mutually exclusive | CONFIRMED |
| balance-core mode only | CONFIRMED |
| velocity-damped controller only | CONFIRMED |

---

## Conclusion

Gain tuning of k_position and k_velocity provides at most 3.8% improvement on the transient peak. The ±0.10 m preferred target and ±0.15 m fallback target are not met by any tested configuration.

The transient overshoot is an architectural issue: the pitch stabilization term in the sagittal controller overwhelms position return during pitch spikes. This cannot be resolved by tuning k_position or k_velocity alone.

**The bug-fixed F4c (k_vel=15.0, k_pos=10.0) remains the best stable configuration:**
- Max drift: 0.254 m (93.4% improvement from bug fix)
- Final drift: 0.028 m (99.3% improvement)
- Steady state (last 20%): ±0.033 m — already within preferred target
- Step E full target (≤0.30 m): PASS

---

## Next Steps

**Do NOT proceed to Step C until reviewed.**

Recommended investigation before architectural changes:
1. Identify why pitch spikes to 11.8 deg at step 1666 (WBC/posture system)
2. Consider pitch term saturation or rate limiting in sagittal controller
3. Consider decoupling pitch stabilization from position return
4. Height variant regression (high_5cm, low_5cm) should be run on bug-fixed F4c baseline, not on failed refinements

**Do NOT:**
- Design new Step E architecture without reviewing this report
- Tune gains further (diminishing returns confirmed)
- Reintroduce E0b/E0c/E0d position containment experiments
- Proceed to Step C/D/F before architectural review
