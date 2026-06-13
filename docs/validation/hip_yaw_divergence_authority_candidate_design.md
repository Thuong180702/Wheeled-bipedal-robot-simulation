# Hip-Yaw Divergence Authority Candidate Design

**Date:** 2026-06-05
**Based on:** `hip_yaw_divergence_fix_eval_report.md` (HY2_DIV_PARTIAL)
**Decision:** `DIVERGENCE_ROOT_CAUSE_IDENTIFIED_READY_FOR_FIX`

## Root Cause Summary

Current HY2-DIV baseline has insufficient torque authority:
- k_divergence = 5.0
- k_divergence_rate = 1.0
- tau_max_divergence = 0.5 Nm
- z_high = 0.393m (gate inactive at nominal/high)

**Result:** 88.7% clipping rate at low_0p300 indicates gains too weak.

## Candidate Design

### Candidate Group A: Low-Height Authority Only
**Purpose:** Check whether low_0p300 improves when HY2-DIV has enough torque.

| ID | k_div | kd_div | tau_max | z_low | z_high | Notes |
|----|-------|--------|---------|-------|--------|-------|
| A0 | 5.0 | 1.0 | 0.5 Nm | 0.300 | 0.393 | Current baseline |
| A1 | 5.0 | 1.0 | 1.0 Nm | 0.300 | 0.393 | 2x tau_max |
| A2 | 5.0 | 1.0 | 2.0 Nm | 0.300 | 0.393 | 4x tau_max |
| A3 | 7.5 | 1.5 | 1.0 Nm | 0.300 | 0.393 | Moderate gain increase |

### Candidate Group B: Height-Gate Coverage
**Purpose:** Check whether nominal/high improve when HY2-DIV is active across the full standing height range.

| ID | k_div | kd_div | tau_max | z_low | z_high | Notes |
|----|-------|--------|---------|-------|--------|-------|
| B1 | 5.0 | 1.0 | 1.0 Nm | 0.300 | 0.500 | Extended gate |
| B2 | 5.0 | 1.0 | 2.0 Nm | 0.300 | 0.500 | Extended gate + 4x tau_max |
| B3 | 7.5 | 1.5 | 1.0 Nm | 0.300 | 0.500 | Extended gate + moderate gain |

### Candidate Group C: Strong Damping (Only if A/B give partial improvement)
**Purpose:** Use stronger but still bounded damping.

| ID | k_div | kd_div | tau_max | z_low | z_high | Notes |
|----|-------|--------|---------|-------|--------|-------|
| C1 | 10.0 | 2.0 | 1.5 Nm | 0.300 | 0.500 | Strong damping |

**C1 will NOT be run unless A/B results justify it.**

## Evaluation Matrix

| Candidate | nominal | low_0p300 | high_0p480 |
|-----------|---------|-----------|------------|
| A0 | - | - | - |
| A1 | Smoke | Smoke | Smoke |
| A2 | Smoke | Smoke | Smoke |
| A3 | Smoke | Smoke | Smoke |
| B1 | Smoke | Smoke | Smoke |
| B2 | Smoke | Smoke | Smoke |
| B3 | Smoke | Smoke | Smoke |
| C1 | N/A | N/A | N/A |

## Pass/Fail Criteria

### 100-Step Smoke Pass
- Survived 100 steps
- No contact loss
- Final height error <= 0.02 m
- WBC applied = false
- Hidden torque = 0
- Ownership violations = 0
- Divergence max does not worsen by more than 20% vs post-sign-fix baseline
- Roll does not collapse
- Sign correctness remains acceptable

### 500-Step Validation Pass
- Survived 500 steps
- Divergence RMS improves vs current HY2-DIV baseline (A0)
- Divergence RMS improves vs post-sign-fix baseline, OR at minimum does not worsen
- hip_yaw_abs_max improves or does not worsen
- support_position_error does not worsen by more than 20%
- Height/contact remain valid
- Roll does not collapse
- WBC applied = false
- Hidden torque = 0
- Ownership violations = 0
- Clipping percent is reported

### 5000-Step Step E Pass
**Primary gates:**
- nominal divergence RMS < 0.10 rad
- low_0p300 divergence RMS < 0.30 rad
- high_0p480 divergence RMS < 0.25 rad
- Sign correct left > 90%
- Sign correct right > 95%
- Survived 5000 steps
- WBC applied = false
- Hidden torque = 0
- Ownership violations = 0

**Secondary gates:**
- support_position_error does not worsen by more than 20%
- Roll does not collapse
- Contact validity remains high
- Height error remains acceptable

## Expected Results

### Post-Sign-Fix Baseline
| Height | div RMS | Notes |
|--------|---------|-------|
| nominal | 0.2446 rad | HY2-DIV inactive |
| low_0p300 | 0.3690 rad | HY2-DIV inactive |
| high_0p480 | 0.3399 rad | HY2-DIV inactive |

### Current HY2-DIV Baseline (A0)
| Height | div RMS | HY2-DIV Active | Clipping |
|--------|---------|----------------|----------|
| nominal | 0.2451 rad | 0% | 0% |
| low_0p300 | 0.4934 rad | 100% | 88.7% |
| high_0p480 | 0.3399 rad | 0% | 0% |

### Target (Step E gates)
| Height | div RMS Target |
|--------|----------------|
| nominal | < 0.10 rad |
| low_0p300 | < 0.30 rad |
| high_0p480 | < 0.25 rad |

## Do Not Touch

- Per-joint PD gains (correct for error correction)
- Hip-roll controller
- Sagittal controller
- WBC paths
- Other divergence mechanisms

## Self-Review Checklist

- [ ] All candidates isolated (one change each)
- [ ] Candidates ordered by risk (A < B < C)
- [ ] C1 only run if A/B partial
- [ ] Pass/fail criteria clear
- [ ] No ambiguous thresholds
- [ ] Evaluation matrix complete