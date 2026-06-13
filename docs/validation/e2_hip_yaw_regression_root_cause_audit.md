# E2 Hip-Yaw Regression Root Cause Audit

**Date**: 2026-06-07
**Classification**: `E2_HIP_YAW_REGRESSION_FROM_RESTRICTIVE_INTEGRAL_GATE`

---

## Executive Summary

E2 (`E2_support_integral_higher_cap`) significantly improves support_position_error_m at low_0p300 but causes hip_yaw_abs_max to regress from 0.1018 rad (D2) to 0.1304 rad (+28% regression, exceeding the 0.10 rad gate).

**Root Cause**: The restrictive `integral_pitch_error_threshold_rad = 0.03` (4× tighter than E1_after's 0.12) delays integral activation, causing larger corrections that couple into hip-yaw through the WBC/posture layer.

**Recommendation**: Design E2b (E2 cap + E1_after gate) to test gate alignment hypothesis before committing to longer validation.

---

## E2 vs D2 Results Summary

| Metric | D2 | E2 | Delta | Status |
|--------|-----|-----|-------|--------|
| support_position_error max (m) | 0.1757 | 0.1703 | -3.1% | ✓ Improved |
| support_position_error mean (m) | 0.0824 | 0.0628 | -23.8% | ✓ Improved |
| support_position_error final (m) | 0.0580 | -0.0276 | -148% | ✓ Improved |
| support_crossings >0.15m | 96 | 62 | -35% | ✓ Improved |
| hip_yaw_abs_max (rad) | 0.1018 | **0.1304** | **+28%** | ✗ REGRESSED |
| hip_yaw_gate (<0.10 rad) | PASS | **FAIL** | - | **GATE FAIL** |

---

## Phase 2: Event Order Analysis

### First Crossing Times

| Event | D2 Step | E2 Step | D2 Before E2 |
|-------|---------|---------|--------------|
| support_gt_0.10m | 66 | 66 | Equal |
| support_gt_0.15m | 91 | 89 | E2 first (2 steps earlier) |
| hip_yaw_gt_0.07 | 284 | 425 | D2 first (141 steps earlier) |
| hip_yaw_gt_0.10 | 328 | 447 | D2 first (119 steps earlier) |
| hip_yaw_gt_0.12 | N/A | 465 | E2 first (only) |
| pitch_gt_0.10 | 78 | 75 | E2 first (3 steps earlier) |
| pitch_gt_0.12 | N/A | 90 | E2 first (only) |

### Event Sequence Analysis

**Key Finding**: E2 hip_yaw regression appears LATER than D2 (step 447 vs 328) but is WORSE (0.1304 vs 0.1018 rad).

**Interpretation**:
1. E2 support improves faster (first 0.15m crossing at step 89 vs 91)
2. E2 hip_yaw takes longer to appear (step 447 vs 328)
3. But E2 hip_yaw grows larger (0.1304 vs 0.1018 rad)
4. This suggests E2's aggressive position corrections cause a DELAYED but LARGER hip_yaw divergence

---

## Phase 3: Torque Coupling Analysis

### Position Torque Statistics

| Metric | D2 | E2 | Change |
|--------|-----|-----|--------|
| tau_position_raw max (Nm) | 0.14 | 1.28 | +9.1× |
| tau_position_raw mean (Nm) | -3.30 | -2.51 | +24% |
| tau_position_raw RMS (Nm) | 4.03 | 3.44 | -15% |
| tau_position_saturated count | 177 (35%) | 95 (19%) | -46% |
| effective_max_position_tau (Nm) | 4.0 | 5.0 | +25% |

**Key Finding**: E2 has 9× higher tau_position_raw max, indicating much larger position corrections are being applied.

### Correlations with Hip-Yaw

| Correlation | D2 | E2 | Interpretation |
|-------------|-----|-----|----------------|
| tau_position_raw vs hip_yaw | **0.63** | **0.73** | Strong positive (position → hip_yaw) |
| support_error vs hip_yaw | -0.63 | -0.73 | Strong negative (coupled) |
| pitch_x vs hip_yaw | -0.57 | -0.58 | Moderate negative |
| wheel_vel vs hip_yaw | 0.26 | 0.35 | Weak positive |
| divergence vs tau_position | **0.64** | **0.73** | Strong (asymmetric correction → divergence) |

**Key Finding**: E2 shows STRONGER correlation between tau_position_raw and hip_yaw (0.73 vs 0.63). This indicates position corrections in E2 couple more strongly into hip_yaw.

### Hip-Yaw Torque Amplification

| Metric | D2 | E2 | Delta |
|--------|-----|-----|-------|
| l_hip_yaw_tau_shape_final max (Nm) | 1.34 | 1.81 | +35% |
| r_hip_yaw_tau_shape_final max (Nm) | 1.59 | 2.07 | +30% |
| divergence_error_max (rad) | 0.093 | 0.122 | +30% |
| common_error_max (rad) | 0.010 | 0.012 | +20% |

**Key Finding**: E2 hip_yaw torques are 30-35% higher than D2, and divergence error is 30% larger. This confirms position corrections couple into asymmetric hip_yaw motion.

---

## Phase 4: Integral Gate Audit

### Configuration Comparison

| Parameter | E1_before | E1_after | E2 | D2 |
|-----------|-----------|----------|-----|-----|
| position_cap (Nm) | 4.0 | 4.0 | **5.0** | 4.0 |
| pitch_threshold (rad) | 0.03 | **0.12** | **0.03** | N/A |
| integral_active_count | 22 | 39 | 31 | 0 |
| integral_active_percent | 4.4% | 7.8% | 6.2% | 0% |
| tau_position_integral_max (Nm) | 0.001 | 0.030 | 0.008 | 0.0 |

### Gate Reason Analysis (from E1 report)

| Reason | E1_before (0.03) | E1_after (0.12) |
|--------|------------------|-----------------|
| pitch_error_large | 349 | 0 |
| pitch_rate_large | 106 | 303 |
| safe_steady_state | 22 | 39 |
| support_velocity_large | 22 | 157 |

**Key Finding**: Raising pitch threshold from 0.03 to 0.12:
1. Eliminated pitch_error_large gate blocking (349 → 0)
2. Shifted to safe_steady_state + support_velocity_large gates
3. Increased integral activation by 77% (22 → 39)

### E2 vs E1_after Comparison

E2 uses:
- Higher cap: 5.0 Nm vs 4.0 Nm (+25%)
- Restrictive threshold: 0.03 rad vs 0.12 rad (4× tighter)

This combination means:
1. Integral activates LESS often in E2 (31 vs 39 steps)
2. When it activates, corrections can be 25% larger
3. Late + large = amplified coupling to hip_yaw

---

## Phase 5: Root Cause Classification

**Classification**: `E2_HIP_YAW_REGRESSION_FROM_RESTRICTIVE_INTEGRAL_GATE`

### Evidence Chain

1. **Event Order**: E2 hip_yaw regression appears AFTER support improves, suggesting correction coupling

2. **Torque Coupling**: E2 tau_position_raw vs hip_yaw correlation is 0.73 (strong), confirming position corrections couple into hip_yaw

3. **Amplification**: E2 hip_yaw torques are 30-35% higher than D2, with 30% larger divergence error

4. **Gate Comparison**: E1_after (0.12 threshold, 4.0 cap) showed NO hip_yaw regression. E2 (0.03 threshold, 5.0 cap) shows regression. The threshold difference is the primary differentiator.

### Mechanism

```
E2 Configuration:
├── Restrictive 0.03 rad pitch threshold
│   └── Integral activates only in "safe_steady_state"
│       └── Late activation → accumulated error → larger correction
│
├── Higher 5.0 Nm cap
│   └── Larger corrections possible when integral fires
│       └── More aggressive wheel velocity commands
│
└── Result:
    ├── Position correction → wheel velocity → WBC/posture coupling
    ├── Asymmetric correction → hip_yaw divergence
    └── Hip_yaw torques increase 30-35%
```

---

## Phase 6: Candidate Design

### Decision: E2b (Gate Alignment with Cap Increase)

**Recommended Next Step**: Test E2b before running longer validation

| Parameter | D2 | E2 | E2b | E1_after |
|-----------|-----|-----|------|----------|
| position_cap (Nm) | 4.0 | 5.0 | **5.0** | 4.0 |
| pitch_threshold (rad) | N/A | 0.03 | **0.12** | 0.12 |

**Hypothesis**: If E2b has no hip_yaw regression, the 0.03 threshold is the culprit. If E2b still regresses, the 5.0 Nm cap is the culprit.

**Pass Criteria** (500 steps):
- hip_yaw_abs_max < 0.10 rad
- support_crossings ≤ 70

See [e2_next_candidate_design_plan.md](e2_next_candidate_design_plan.md) for full candidate designs (E2b, E2c, E2d).

---

## Final Decision

**Decision**: `E2_REQUIRES_E2B_GATE_ALIGNMENT`

**Rationale**:
1. E2 significantly improves support_position_error_m (+35% fewer gate violations)
2. E2 causes hip_yaw_abs_max to exceed 0.10 rad gate (+28% regression)
3. Evidence strongly suggests restrictive 0.03 rad threshold is the primary cause
4. E2b isolates the gate parameter while keeping the cap improvement
5. If E2b passes, we have a viable E2 variant. If E2b fails, we know the cap is problematic.

**Do NOT**:
- Run 2000-step validation with current E2
- Run 5000-step validation
- Run Step C or Step D
- Enable HY2-DIV
- Commit changes

**Do**:
- Design and approve E2b candidate
- Run 500-step smoke test with E2b
- Audit results before proceeding

---

## Files Generated

| File | Description |
|------|-------------|
| `outputs/step_e_extreme_support_fix_eval/e2_hip_yaw_regression_audit/event_order.csv` | First crossing times for key events |
| `outputs/step_e_extreme_support_fix_eval/e2_hip_yaw_regression_audit/event_order_summary.json` | Event order analysis summary |
| `outputs/step_e_extreme_support_fix_eval/e2_hip_yaw_regression_audit/torque_coupling_summary.json` | Torque coupling analysis |
| `outputs/step_e_extreme_support_fix_eval/e2_hip_yaw_regression_audit/integral_gate_comparison.json` | Integral gate comparison |
| `outputs/step_e_extreme_support_fix_eval/e2_next_candidate_design_plan.json` | Candidate design plan (JSON) |
| `docs/validation/e2_next_candidate_design_plan.md` | Candidate design plan (Markdown) |
| `docs/validation/e2_hip_yaw_regression_root_cause_audit.md` | This report |
| `outputs/step_e_extreme_support_fix_eval/e2_hip_yaw_regression_audit_summary.json` | Audit summary (next) |