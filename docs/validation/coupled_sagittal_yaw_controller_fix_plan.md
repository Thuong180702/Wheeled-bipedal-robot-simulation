# Coupled Sagittal-Yaw Controller Fix Plan

**Date:** 2026-06-04  
**Status:** PLANNING  
**Context:** Both HY-FF and continuous k_position fixes failed independently

---

## Executive Summary

Two independent fixes have been attempted and both failed to resolve the low_0p300 boundary failure:

1. **HY-FF (Hip-Yaw Support-Error Feedforward):** Best candidate (sign=-1.0, k=2.0) reduced hip-yaw from 0.2137 to 0.1941 rad (9.2% improvement), but still **177% over threshold** (0.070 rad)

2. **Continuous k_position scheduling:** Best candidate (k=100) reduced support drift from 0.243 to 0.203 m (16% improvement), but still **35% over threshold** (0.150 m)

**Root Cause Classification:** Hip-yaw and sagittal support errors are **coupled through robot dynamics at extreme flexion**. Neither symptom can be fixed independently because:
- Support drift creates disturbance inputs to hip-yaw control
- Hip-yaw error may induce secondary sagittal disturbances
- Extreme flexion (z=0.300m) reduces actuator effectiveness for both axes

**Investigation Strategy:** Systematically explore mechanism space before implementing joint fixes:
1. Audit advanced hip-yaw local rejection mechanisms (divergence damping, common-mode compensation, support-velocity lead, integral)
2. Audit sagittal authority transmission failures (saturation, damping, pitch conflict, wheel effectiveness)
3. Design evidence-based joint fix combining best mechanisms from both audits

---

## Problem Statement

### Acceptance Gates (all must pass)

**Step E (standing balance at fixed height):**
- `support_position_error_max_abs <= 0.15 m`
- `hip_yaw_abs_max <= 0.07 rad`
- `percent(hip_yaw_abs > 0.10 rad) = 0%`
- `pitch_x_max_abs <= 0.10 rad`
- `roll_y_max_abs <= 0.05 rad`
- `final_height_error <= 0.02 m`
- `contact_valid >= 99.9%`
- `non_wheel_contacts = 0`
- `WBC_applied = false`
- `hidden_torque = 0`
- `ownership_violations = 0`

**Step C (height recovery from initial perturbation):**
- All Step E criteria
- `height_recovered = true`
- `recovery_time_s < timeout`

**Continuity:**
- All scheduled parameters continuous over z ∈ [0.300, 0.480]
- No discontinuities, no variant-name-only control law

**Regression:**
- nominal, low_tiny, high_tiny, low_small, high_small all pass

### Current Baseline Results (low_0p300)

| Metric | Baseline | Threshold | Status |
|--------|----------|-----------|--------|
| support_position_error | 0.243 m | ≤ 0.150 m | **FAIL (62% over)** |
| hip_yaw_abs_max | 0.2137 rad | ≤ 0.070 rad | **FAIL (205% over)** |
| pitch_x_max_abs | 0.095 rad | ≤ 0.100 rad | PASS (marginal) |
| roll_y_max_abs | 0.015 rad | ≤ 0.050 rad | PASS |

### Results After Independent Fixes

**HY-FF best (C: sign=-1.0, k=2.0):**
- support: 0.2380 m (2% worse than baseline)
- hip_yaw: 0.1941 rad (9.2% better, but **177% over threshold**)
- pitch: 0.094 rad (PASS)
- **Verdict:** FAIL

**Continuous k_position best (E3: k=100):**
- support: 0.2031 m (16% better, but **35% over threshold**)
- hip_yaw: 0.2054 rad (4% better, but **193% over threshold**)
- pitch: 0.1001 rad (**marginal FAIL**, 0.1% over threshold)
- **Verdict:** FAIL

**Key Insight:** Improvements are modest and **neither fix addresses the root coupling**.

---

## Mechanism Space Classification

### A. Hip-Yaw Local Rejection Mechanisms

**A1. Baseline PD control** (current)
- Proportional: `tau = -kp * error`
- Damping: `tau = -kd * velocity`
- Status: Insufficient at low heights

**A2. Support-error feedforward (HY-FF)** (evaluated, failed)
- Formula: `tau_comp = sign * k_support * support_error * height_gate`
- Sign: -1.0 correct
- Best gain: k=2.0
- Result: 9.2% improvement, insufficient

**A3. Divergence damping/authority** (not yet evaluated)
- Error mode: `e_div = l_hip_yaw_error - r_hip_yaw_error`
- Velocity mode: `e_div_dot = l_hip_yaw_vel - r_hip_yaw_vel`
- Compensation: antisymmetric torque opposing divergence
- Rationale: If hip-yaw problem is primarily left/right asymmetry rather than common-mode yaw

**A4. Common-mode yaw compensation** (not yet evaluated)
- Error mode: `e_common = l_error + r_error`
- Body yaw: root/body yaw angle
- Compensation: symmetric torque opposing body yaw drift
- Rationale: If hip-yaw problem is primarily body yaw rotation rather than left/right divergence

**A5. Support-velocity lead compensation** (not yet evaluated)
- Formula: `tau_comp = k_support * support_error + k_support_vel * support_error_rate`
- Rationale: Support-error feedforward may lag disturbance; velocity term provides lead compensation
- Sign: TBD (sweep required)

**A6. Hip-yaw integral with anti-windup** (not yet evaluated)
- Formula: `tau_comp = -kp * error - kd * velocity - ki * integral(error)`
- Anti-windup: clamp integral, reset at run start
- Rationale: If error is slowly accumulating (not oscillatory)
- Warning: Can destabilize if coupled dynamics present

**A7. Increased PD gains** (not yet evaluated systematically)
- Rationale: Current kp=5.0, kd=1.0 may be too low
- Candidate: kp=15.0, kd=3.0 (balance_core_candidate_b)
- Warning: May conflict with support control or cause high-frequency oscillations

### B. Sagittal Support Authority Mechanisms

**B1. Baseline position return** (current)
- k_position = 40.0 (nominal)
- max_position_tau = 3.0 Nm
- Status: Insufficient at low heights

**B2. Continuous k_position scheduling** (evaluated, failed)
- E3: k_position 40 → 100 at z=0.300
- Result: 16% improvement, insufficient
- Failure modes: E3 caused pitch exceedance (0.1001 > 0.100)

**B3. Position torque cap increase** (not yet evaluated)
- max_position_tau: 3.0 → 6.0 Nm at low heights
- Rationale: k_position increase may be saturated by torque cap

**B4. Velocity damping increase** (not yet evaluated)
- k_velocity: 15.0 → 25.0 or 30.0 at low heights
- Rationale: Support velocity damping may be too weak

**B5. Support velocity damping** (not yet evaluated)
- Direct damping on support_error_rate
- tau_comp = -k_support_vel * support_error_rate
- Rationale: Position return may be too slow; velocity damping provides immediate resistance

**B6. Support integral with anti-windup** (not yet evaluated)
- Formula: `tau_comp = k_position * error + ki_support * integral(error)`
- Rationale: If position return alone has persistent steady-state drift
- Warning: Integral can destabilize if coupling present

**B7. Wheel torque saturation/rate limit audit** (not yet evaluated)
- Check if wheel torque commands saturate or rate-limit
- Check if effective wheel authority is reduced at extreme flexion

**B8. Pitch-position conflict audit** (not yet evaluated)
- Check if pitch stabilization dominates and conflicts with position return
- Check if tau_pitch and tau_position oppose each other

### C. Joint Coupled Mechanisms

**C1. Coupled sagittal-yaw stabilizer** (not yet evaluated)
- Explicit compensation for support-yaw interaction
- Example: `tau_hip_yaw += f(support_error, support_velocity, body_yaw)`
- Rationale: If coupling is fundamental to dynamics, treat jointly

**C2. Multi-axis LQR/MPC** (not yet evaluated, high complexity)
- Joint optimization over sagittal + yaw state
- Requires accurate linearization at extreme flexion
- May require controller redesign (Step D fallback)

**C3. Combined height-gated schedule** (not yet evaluated)
- Schedule multiple parameters jointly based on evidence
- Example: increase k_position + max_tau + k_velocity + hip_yaw_kp together
- Ensure all schedules use same height gate for consistency

---

## Investigation Phases

### Phase 2: Deep Hip-Yaw Disturbance-Rejection Audit

**Goal:** Classify which advanced hip-yaw mechanism(s) are most promising.

**Script:** `scripts/audit_hip_yaw_advanced_rejection.py`

**Output:** `outputs/advanced_hip_yaw_rejection_audit/`

**Analysis:**
1. Load baseline and best HY-FF candidate telemetry
2. Compute:
   - Left/right hip-yaw errors
   - Divergence: `e_div = abs(l_error - r_error)`
   - Common-mode: `e_common = abs(l_error + r_error)`
   - Body yaw angle and rate
   - Support error and rate
   - Wheel torque and velocity
   - Hip-yaw torque components (proportional, damping, HY-FF)
3. Lag correlation analysis:
   - support_error → hip_yaw divergence (time lag)
   - support_velocity → hip_yaw divergence
   - body_yaw → hip_yaw common_mode
   - pitch → hip_yaw
4. Classification:
   - `divergence_dominant` vs `common_mode_dominant`
   - `support_velocity_lead_needed` (if error lags velocity)
   - `integral_needed` (if persistent steady-state error)
   - `pd_gains_too_low` (if response is sluggish)
   - `not_locally_rejectable` (if coupling dominates)

**Deliverables:**
- `advanced_hip_yaw_rejection_summary.json`
- `advanced_hip_yaw_rejection_report.md`
- `hip_yaw_error_phase_portrait.csv`
- `hip_yaw_divergence_vs_support.csv`
- `hip_yaw_body_yaw_coupling.csv`
- `advanced_hip_yaw_mechanism_classification.json` (critical for Phase 3)

### Phase 3: Advanced Hip-Yaw Candidate Experiments

**Goal:** Test most promising mechanism from Phase 2 classification.

**Candidates:**
- HY2-DIV: Divergence damping
- HY2-COMMON: Common-mode/body-yaw compensation
- HY2-SV: Support-velocity lead compensation
- HY2-I: Hip-yaw integral with anti-windup
- HY2-COMBO: Best two mechanisms combined (not all stacked)

**Constraints:**
- All disabled by default
- All continuous height-gated (z_low=0.300, z_high=0.393)
- No variant-name control
- No global changes
- Sign sweep required for each new mechanism

**Evaluation:**
- Each candidate: low_0p300, high_0p480, nominal @ 1000 steps
- If any passes hip-yaw gate: extend to 5000 steps + five-variant regression
- If all fail: classify as `ADVANCED_HIP_YAW_LOCAL_FIX_FAILED_REQUIRES_JOINT_FIX`

**Acceptance for advanced hip-yaw task:**
- hip_yaw_abs_max <= 0.07 rad
- percent(hip_yaw > 0.10) = 0%
- support_position_error must not worsen >10% vs baseline
- pitch, roll, height, contact, WBC, ownership: same as Step E

**Note:** This phase does NOT require support <= 0.15 yet (only testing hip-yaw improvement).

### Phase 4: Sagittal Authority Transmission Audit

**Goal:** Classify why continuous k_position alone failed.

**Script:** `scripts/audit_low_height_sagittal_authority_transmission.py`

**Output:** `outputs/low_height_sagittal_authority_transmission_audit/`

**Analysis:**
1. Load baseline, E1, E2, E3, and best hip-yaw candidate (if any)
2. Compute:
   - Effective k_position (scheduled value)
   - Support error and rate
   - tau_position_raw (before clipping)
   - tau_position_clipped (after max_position_tau cap)
   - Saturation flag: `abs(tau_position_raw) > max_position_tau`
   - Wheel torque raw/final (left/right)
   - Wheel torque saturation, rate-limit
   - Wheel velocity
   - k_velocity, tau_sagittal_velocity
   - Does damping oppose drift?
   - tau_pitch magnitude and rate
   - Does pitch conflict with position?
   - Contact validity, WBC, hidden, ownership
3. Event order:
   - First position saturation
   - First wheel saturation
   - First pitch exceedance
4. Classification:
   - `position_torque_cap_saturation`
   - `wheel_torque_saturation`
   - `wheel_torque_rate_limit`
   - `insufficient_velocity_damping`
   - `support_velocity_underdamped`
   - `pitch_position_conflict`
   - `extreme_flexion_wheel_effectiveness_loss`
   - `contact_coupling_limits_authority`
   - `coupled_sagittal_yaw_dynamics`

**Deliverables:**
- `authority_transmission_summary.json`
- `authority_transmission_report.md`
- `authority_saturation_comparison.csv`
- `event_order_comparison.csv`
- `authority_failure_classification.json` (critical for Phase 5)

### Phase 5: Joint Low-Height Sagittal-Yaw Fix Design

**Goal:** Design evidence-based joint candidate family.

**Document:** `docs/validation/joint_low_height_sagittal_yaw_fix_design.md`

**Design Principles:**
1. All schedules continuous (smoothstep)
2. All use z_ref, not variant name
3. All disabled by default (opt-in via profile selection)
4. No global nominal changes
5. Stop at first passing candidate (don't over-engineer)

**Candidate Template:**

```python
J0_baseline:
  # No changes, reference

J1_support_cap:
  # If Phase 4 shows position_torque_cap_saturation
  k_position: 40 → 80 at z=0.300
  max_position_tau: 3.0 → 6.0 at z=0.300
  k_velocity: 15.0 (unchanged)
  hip_yaw: baseline

J2_support_cap_damping:
  # If Phase 4 shows insufficient_velocity_damping
  k_position: 40 → 80
  max_position_tau: 3.0 → 6.0
  k_velocity: 15.0 → 25.0 at z=0.300
  hip_yaw: baseline

J3_support_cap_strong_damping:
  # If Phase 4 shows support_velocity_underdamped
  k_position: 40 → 80
  max_position_tau: 3.0 → 6.0
  k_velocity: 15.0 → 30.0 at z=0.300
  hip_yaw: baseline

J4_support_integral:
  # If Phase 4 shows persistent steady-state drift
  k_position: 40 → 60 or 80
  max_position_tau: 3.0 → 6.0
  support_integral: enable with anti-windup
  hip_yaw: baseline

J5_coupled:
  # Best support schedule from J1-J4
  # PLUS best hip-yaw mechanism from Phase 3 (if it helped)
  # OR coupled support-yaw compensation if required
```

**Decision Logic:**
- If Phase 3 found a hip-yaw mechanism that improved hip-yaw without worsening support, include it in J5
- If Phase 3 found no improvement, J5 uses coupled compensation (e.g., support-error → hip-yaw feedforward combined with support authority increase)
- If Phase 4 shows multiple failure modes, stack only compatible mechanisms (e.g., cap + damping OK, cap + integral may conflict)

### Phase 6: Implement and Evaluate Joint Fix Candidates

**Goal:** Find smallest candidate that passes all gates.

**Script:** `scripts/evaluate_joint_low_height_sagittal_yaw_fix.py`

**Output:** `outputs/joint_low_height_sagittal_yaw_fix/`

**Evaluation Protocol:**

For each candidate J0, J1, J2, J3, J4, J5:

1. **Phase 6.1:** low_0p300 Step E 1000
   - If FAIL: log and continue to next candidate
   - If PASS: continue to Phase 6.2

2. **Phase 6.2:** low_0p300 Step E 5000
   - Verify stability over longer horizon
   - If FAIL: log and continue to next candidate
   - If PASS: continue to Phase 6.3

3. **Phase 6.3:** high_0p480 Step E 5000
   - Verify no regression at high height
   - If FAIL: log and continue to next candidate
   - If PASS: continue to Phase 6.4

4. **Phase 6.4:** Step C low_0p300 5000
   - Verify height recovery works at low height
   - If FAIL: log and continue to next candidate
   - If PASS: continue to Phase 6.5

5. **Phase 6.5:** Step C high_0p480 5000
   - Verify height recovery at high height
   - If FAIL: log and continue to next candidate
   - If PASS: continue to Phase 6.6

6. **Phase 6.6:** Practical height grid Step E
   - Heights: 0.300, 0.330, 0.360, 0.393 (or low_small), nominal, high_small, 0.450, 0.480
   - 1000 steps each
   - All must pass
   - If any FAIL: log and continue to next candidate
   - If all PASS: continue to Phase 6.7

7. **Phase 6.7:** Step C grid
   - Heights: 0.300, 0.360, nominal, 0.480
   - 5000 steps each
   - All must pass
   - If any FAIL: log and continue to next candidate
   - If all PASS: continue to Phase 6.8

8. **Phase 6.8:** Five-variant regression
   - Variants: nominal, low_tiny, high_tiny, low_small, high_small
   - Step E 5000 each
   - All must pass
   - If any FAIL: log and continue to next candidate
   - If all PASS: **CANDIDATE SELECTED**, stop evaluation

**Stop-at-first-pass:** As soon as one candidate passes all phases, select it and stop. Do not evaluate higher candidates.

**If all fail:** Final decision = `JOINT_FIX_REQUIRED` or `CONTROLLER_REDESIGN_REQUIRED`

### Phase 7: Acceptance Criteria Validation

**Goal:** Verify selected candidate passes all gates.

**Already validated by Phase 6 protocol**, but summarize results:

**Step E gates:**
- support_position_error <= 0.15 m ✓
- hip_yaw_abs_max <= 0.07 rad ✓
- pitch, roll, height, contact, WBC, ownership ✓

**Step C gates:**
- All Step E gates ✓
- height_recovered = true ✓
- recovery_time valid ✓

**Continuity:**
- Schedule continuous verification (script: `check_schedule_continuity.py`)

**Regression:**
- Five-variant all pass ✓

**If all pass:** Final decision = `BOUNDARY_RANGE_PASS`, ready for Step D

### Phase 8: Tests

**Goal:** Ensure implementation correctness and no regressions.

**New unit tests:**
- All new schedules disabled by default
- Schedule uses z_ref, not variant name
- Schedule is continuous (smoothstep)
- Clamps work correctly
- Anti-windup works (if integral implemented)
- Previous-step signals reset at simulation start
- No WBC enabled
- No hip-roll modification
- No global hip-yaw gain change unless profile explicitly selected
- Telemetry fields exist

**Regression tests:**
- `pytest tests/test_hip_yaw_support_feedforward.py` (9 tests, should still pass)
- `pytest tests/test_sagittal_velocity_damped_balance_controller.py` (40 tests, should still pass)
- `pytest tests/test_step_c_height_recovery.py` (should pass except expected diff failure)
- `pytest tests/test_step_e_*.py` (all should pass)
- `pytest tests/test_balance_core_*.py` (all should pass)

### Phase 9: Final Report

**Goal:** Document complete investigation and final decision.

**Document:** `docs/validation/joint_low_height_sagittal_yaw_fix_final_report.md`

**JSON Summary:** `outputs/joint_low_height_sagittal_yaw_fix/joint_low_height_sagittal_yaw_fix_summary.json`

**Required Contents:**
1. Executive summary
2. Files changed
3. Tests run and results
4. Phase 2 advanced hip-yaw audit result
5. Phase 3 advanced hip-yaw candidates (if any passed)
6. Phase 4 sagittal authority transmission classification
7. Phase 5 joint fix design
8. Phase 6 candidate comparison table
9. Selected candidate (if any)
10. low_0p300 Step E result (before/after)
11. high_0p480 Step E result
12. low_0p300 Step C result
13. high_0p480 Step C result
14. Practical height grid result
15. Five-variant regression result
16. Support drift before/after
17. Hip-yaw before/after
18. Pitch before/after
19. Height/contact status
20. WBC status, hidden torque max, ownership violation max
21. Schedule continuity evidence
22. Final decision code

**Final Decision Codes:**
- `BOUNDARY_RANGE_PASS` (ready for Step D)
- `ADVANCED_HIP_YAW_LOCAL_FIX_FAILED_REQUIRES_JOINT_FIX` (Phase 3 all failed)
- `JOINT_FIX_REQUIRED` (Phase 6 all failed, more work needed)
- `JOINT_FIX_CAUSED_REGRESSION` (Phase 6 candidate passed low but broke high/nominal)
- `CONTROLLER_REDESIGN_REQUIRED` (fundamental architecture issue, need Step D alternative)
- `NEW_ROOT_CAUSE_FOUND` (investigation revealed unexpected failure mode)

---

## Design Constraints (STRICT)

### Allowed

✅ Audit deeper coupling mechanisms  
✅ Implement continuous height/reference-based schedules  
✅ Implement low-height-only candidate profiles  
✅ Implement hip-yaw divergence controller  
✅ Implement hip-yaw integral with anti-windup (if justified)  
✅ Implement support-velocity or support-acceleration lead compensation (if justified)  
✅ Implement coupled sagittal-yaw stabilizer (if justified)  
✅ Add telemetry  
✅ Add tests  
✅ Update docs  

### Prohibited

❌ Do NOT add WBC  
❌ Do NOT enable legacy WBC paths  
❌ Do NOT use root-z-only perturbation  
❌ Do NOT relax thresholds  
❌ Do NOT shrink target heights  
❌ Do NOT modify hip-roll logic  
❌ Do NOT globally change hip-yaw gains  
❌ Do NOT use variant-name-only patches  
❌ Do NOT use discontinuous step/bucket schedules  
❌ Do NOT keep increasing one gain blindly  
❌ Do NOT deploy HY-FF by default  
❌ Do NOT claim BOUNDARY_RANGE_PASS unless all gates pass  
❌ Do NOT proceed to Step D until BOUNDARY_RANGE_PASS  

---

## Success Criteria

**Minimum for BOUNDARY_RANGE_PASS:**

1. All Step E runs pass (low_0p300, high_0p480, practical grid)
2. All Step C runs pass (low_0p300, high_0p480, selected grid)
3. All five-variant regression runs pass
4. Schedule continuity verified
5. Tests pass (no regressions)
6. WBC false, hidden torque 0, ownership 0
7. Implementation disabled by default (opt-in)
8. No prohibited constraints violated

**If any fail:** Document reason and assign appropriate final decision code.

---

## Risk Assessment

**High Risk:**
- Coupling may be fundamental to wheeled-biped dynamics at extreme flexion
- Joint fix may require simultaneous increase of multiple parameters → brittleness
- Increased authority may cause instability at intermediate heights
- Pitch exceedance risk (already seen in E3)

**Medium Risk:**
- Integral terms may cause overshoot or instability
- Support-velocity lead may amplify noise
- Hip-yaw divergence damping may conflict with body yaw control

**Low Risk:**
- Schedule continuity issues (mitigated by smoothstep implementation)
- Telemetry collection (already proven in HY-FF and k_position work)

**Mitigation Strategies:**
- Stop at first passing candidate (don't over-tune)
- Validate at every height in practical grid
- Verify five-variant regression (catch intermediate-height failures)
- Phase 2/4 audits provide evidence before implementing (avoid blind tuning)
- All schedules disabled by default (safe fallback to baseline)

---

## Timeline Estimate

**Phase 2 (audit):** ~1 hour (script + analysis)  
**Phase 3 (experiments):** ~2-4 hours (depends on how many candidates)  
**Phase 4 (audit):** ~1 hour  
**Phase 5 (design):** ~1 hour (document + review)  
**Phase 6 (evaluation):** ~4-8 hours (depends on candidate count and pass/fail)  
**Phase 7 (validation):** ~0 hours (already done in Phase 6)  
**Phase 8 (tests):** ~1 hour  
**Phase 9 (report):** ~1 hour  

**Total:** ~11-18 hours

**Critical Path:** Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6

If Phase 3 finds no improvement AND Phase 4 shows fundamental coupling, Phase 5 design will be straightforward (coupled fix required). If Phase 3 finds partial improvement, Phase 5 will need to combine mechanisms carefully.

---

## Open Questions

1. **Is hip-yaw error primarily divergence or common-mode?** (Phase 2 will answer)
2. **Does support-error feedforward lag the actual disturbance?** (Phase 2 lag correlation)
3. **Is position torque cap the primary saturation bottleneck?** (Phase 4 will answer)
4. **Does pitch control conflict with support control at low heights?** (Phase 4 will answer)
5. **Can any advanced hip-yaw mechanism pass independently?** (Phase 3 will answer)
6. **Is coupling fundamental or can it be mitigated with better local control?** (Phases 2-4 will clarify)

---

## Next Step

Proceed to **Phase 2: Deep Hip-Yaw Disturbance-Rejection Audit**.

Create `scripts/audit_hip_yaw_advanced_rejection.py` and run analysis on baseline and best HY-FF candidate.
