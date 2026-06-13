# E2 Next Candidate Design Plan

## Status
**PHASE 6: Candidate Design Only** — Do NOT implement until approved.

## Evidence Summary

### E2 Hip-Yaw Regression Root Cause

**Classification**: `E2_HIP_YAW_REGRESSION_FROM_RESTRICTIVE_INTEGRAL_GATE`

### Key Evidence

1. **Event Order**:
   - E2 hip_yaw > 0.10 rad at step 447 (AFTER support improved at step 89)
   - D2 hip_yaw > 0.10 rad at step 328 (EARLIER than D2 support at step 91)
   - E2 hip_yaw regression is DELAYED but WORSE

2. **Torque Coupling**:
   - E2 tau_position_raw max: 1.28 Nm (9.1× higher than D2's 0.14 Nm)
   - E2 tau_position_raw vs hip_yaw correlation: 0.73 (strong positive)
   - D2 tau_position_raw vs hip_yaw correlation: 0.63 (moderate positive)
   - E2 divergence vs tau_position correlation: 0.73 (strong positive)

3. **Hip-Yaw Torque Amplification**:
   - E2 l_hip_yaw_tau_final_max: 1.81 Nm (+35% vs D2)
   - E2 r_hip_yaw_tau_final_max: 2.07 Nm (+30% vs D2)
   - E2 divergence_error_max: 0.122 rad (+30% vs D2)

4. **Integral Gate Comparison**:
   - E2 integral_active_count: 31 (vs E1_after 39, D2 0)
   - E2 pitch threshold: 0.03 rad (restrictive)
   - E1_after pitch threshold: 0.12 rad (permissive)
   - E2 position cap: 5.0 Nm (higher)
   - E1_after position cap: 4.0 Nm (lower)

### Root Cause Analysis

**Primary Cause**: Restrictive pitch threshold (0.03 rad) in E2

The restrictive 0.03 rad threshold means:
1. Integral activates only in "safe_steady_state" (near-zero pitch error)
2. When integral activates, it applies corrections after the system has ALREADY drifted
3. These late corrections couple into hip-yaw through the WBC/posture layer
4. The higher cap (5.0 Nm) allows more aggressive corrections that exacerbate coupling

**Secondary Factor**: Higher position cap (5.0 Nm)

The 5.0 Nm cap means:
1. Larger position authority available for correction
2. Combined with late activation, creates larger transients
3. These transients excite hip-yaw divergence mode

### Candidate Designs

---

## Candidate E2b: Gate Alignment with Cap Increase

### Design
- Same as E2 EXCEPT:
  - `max_position_tau_low_max = 5.0 Nm` (keep E2 cap)
  - `integral_pitch_error_threshold_rad = 0.12` (use E1_after gate)
- Purpose: Isolate cap effect while using the permissive E1 gate

### Rationale
- E1_after showed NO hip_yaw regression with 0.12 threshold
- If E2b has no hip_yaw regression, the 0.03 threshold is the culprit
- If E2b still regresses, the 5.0 Nm cap is the culprit

### Expected Benefit
- Support improvement similar to E2 (from 5.0 Nm cap)
- No hip_yaw regression (from permissive 0.12 gate)

### Risk
- If 5.0 Nm cap alone causes regression, E2b will still fail
- Mitigation: design E2c as fallback

### Telemetry Required
- `hip_yaw_abs_max` < 0.10 rad (gate pass)
- `support_position_error_m` crossings < 62 (E2 level)
- `tau_position_raw` vs `hip_yaw_abs_max` correlation < 0.5
- `integral_active_count` > 35 (more permissive gate = more activations)

### 500-Step Pass Criteria
- `hip_yaw_abs_max < 0.10 rad` ✓
- `support_position_error_m` crossings ≤ 70 (not worse than E2)
- `integral_gate_reason` shows "safe_steady_state" at least 35 times

### Stop Condition
- If `hip_yaw_abs_max > 0.10 rad` at 500 steps → stop, design E2c

### Rollback Rule
- If E2b fails hip_yaw gate → revert to D2 baseline

---

## Candidate E2c: Intermediate Cap with Permissive Gate

### Design
- `max_position_tau_low_max = 4.5 Nm` (between D2 4.0 and E2 5.0)
- `integral_pitch_error_threshold_rad = 0.12` (permissive E1 gate)
- Purpose: Reduce hip-yaw regression while keeping some support improvement

### Rationale
- If E2b still regresses, the cap itself is problematic
- 4.5 Nm provides 12.5% more authority than D2
- Permissive gate allows earlier integral activation

### Expected Benefit
- Support improvement (moderate, from 4.5 Nm cap)
- No hip_yaw regression (from both cap reduction and permissive gate)

### Risk
- Support improvement may be less than E2
- Mitigation: trade-off is acceptable if hip_yaw stays under 0.10 rad

### Telemetry Required
- `hip_yaw_abs_max` < 0.10 rad
- `support_position_error_m` crossings < 80 (better than D2's 96)
- `tau_position_raw` max < 1.0 Nm

### 500-Step Pass Criteria
- `hip_yaw_abs_max < 0.10 rad` ✓
- `support_position_error_m` crossings ≤ 80

### Stop Condition
- If `hip_yaw_abs_max > 0.10 rad` → stop, design E2d

### Rollback Rule
- If E2c fails → revert to D2 baseline

---

## Candidate E2d: Ramped Position Authority

### Design
- `max_position_tau_low_max = 5.0 Nm` (keep E2 cap)
- `integral_pitch_error_threshold_rad = 0.12` (permissive gate)
- **ADD**: Smooth ramp-in of position authority over first 100 steps OR until support error < 0.10 m
- Purpose: Avoid abrupt position correction transient

### Rationale
- If E2b/E2c still show some regression, the abrupt cap increase causes transients
- Ramped authority allows system to settle before applying full correction

### Expected Benefit
- Support improvement (from 5.0 Nm cap)
- No hip_yaw regression (from ramp + permissive gate)

### Risk
- Implementation complexity
- Ramp timing may need tuning

### Telemetry Required
- `hip_yaw_abs_max` < 0.10 rad
- `support_position_error_m` crossings < 65
- `tau_position_raw` ramp-in visible in telemetry

### 500-Step Pass Criteria
- `hip_yaw_abs_max < 0.10 rad` ✓
- `support_position_error_m` crossings ≤ 65

### Stop Condition
- If `hip_yaw_abs_max > 0.10 rad` after ramp-in → stop, reject E2 family

### Rollback Rule
- If E2d fails → reject E2 family entirely, do not pursue position cap increases

---

## Decision Tree

```
Start: E2 fails hip_yaw gate

         │
         ▼
    Try E2b (5.0Nm cap + 0.12 gate)
         │
         ├─► hip_yaw < 0.10 ✓ → PROCEED to 2000-step
         │
         └─► hip_yaw > 0.10 ✗
                    │
                    ▼
               Try E2c (4.5Nm cap + 0.12 gate)
                    │
                    ├─► hip_yaw < 0.10 ✓ → PROCEED to 2000-step
                    │
                    └─► hip_yaw > 0.10 ✗
                               │
                               ▼
                          Try E2d (5.0Nm + ramp + 0.12 gate)
                               │
                               ├─► hip_yaw < 0.10 ✓ → PROCEED to 2000-step
                               │
                               └─► hip_yaw > 0.10 ✗
                                          │
                                          ▼
                                     REJECT E2 family
                                     Position cap increases cause hip_yaw regression
                                     Consider D2 baseline only
```

---

## Files Generated

- `outputs/step_e_extreme_support_fix_eval/e2_next_candidate_design_plan.json`
- `docs/validation/e2_next_candidate_design_plan.md` (this file)

---

## Restrictions (Repeated)

Do NOT:
- Modify D2 baseline
- Change default controller behavior
- Enable HY2-DIV
- Add WBC
- Enable legacy WBC
- Relax Step E gates
- Run 2000-step validation
- Run 5000-step validation
- Run Step C
- Run Step D
- Commit

Allowed:
- Implement E2b/E2c/E2d only after explicit approval
- Run 500-step smoke test per candidate
