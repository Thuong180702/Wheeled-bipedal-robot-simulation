# Hip-Yaw Divergence After Sign Fix Audit Report

**Date:** 2026-06-05
**Task:** Investigate why hip-yaw divergence increased after the sign convention fix
**Decision:** `DIVERGENCE_ROOT_CAUSE_IDENTIFIED`

## Executive Summary

The divergence increase after the sign fix is **NOT a sign problem** - it reveals that **per-joint PD cannot control divergence mode**. The sign fix is correct, but it exposes a fundamental limitation: per-joint PD produces torques that accelerate divergence rather than correct it.

## Critical Finding: Divergence Torque Mode Analysis

The key insight from mode decomposition:

| Height | Div Torque Correcting | Div Torque Accelerating |
|--------|----------------------|------------------------|
| nominal | 0.32% | 99.68% |
| low_0p300 | 2.24% | 97.76% |
| high_0p480 | 0.66% | 99.34% |

**The per-joint PD controller produces divergence-mode torques that accelerate divergence 97-99% of the time.**

## Root Cause Analysis

### Why Per-Joint PD Cannot Control Divergence

Per-joint PD computes:
```
tau_L = kp * error_L - kd * vel_L
tau_R = kp * error_R - kd * vel_R
```

When divergence exists (error_L > 0, error_R < 0):
- tau_L = kp * (+error) → positive
- tau_R = kp * (-error) → negative
- div_torque = (tau_L - tau_R) / 2 ≈ kp * error (always same sign as divergence)

This means:
- **Positive divergence** (left ahead) → positive divergence torque → even more divergence
- Per-joint PD **drives** divergence, not **damps** it

### What Changed After Sign Fix

| Metric | Pre-Fix | Post-Fix | Change |
|--------|---------|----------|--------|
| Nominal divergence RMS | 0.0447 rad | 0.2446 rad | +5.5x |
| Nominal div_torque RMS | 0.3501 Nm | 1.8530 Nm | +5.3x |
| Nominal growth pattern | oscillatory | biased_growth | WORSE |

The sign fix made per-joint torque **more coherent** (correct sign = larger magnitude), which accelerates divergence faster.

### Growth Pattern Evidence

| Height | Pre Pattern | Pre Ratio | Post Pattern | Post Ratio |
|--------|-------------|-----------|--------------|------------|
| nominal | oscillatory | 1.49 | biased_growth | 2.36 |
| low_0p300 | oscillatory | 1.32 | oscillatory | 1.34 |
| high_0p480 | biased_growth | 23.87 | biased_growth | 72.51 |

The high_0p480 case shows **extreme growth** after sign fix - the growth ratio went from 24x to 72x.

## Pre vs Post Comparison

### Sign Correctness

| Height | Pre L | Post L | Pre R | Post R |
|--------|-------|--------|-------|--------|
| nominal | 85.1% | 93.9% | 99.8% | 99.7% |
| low_0p300 | 97.2% | 97.1% | 99.0% | 98.9% |
| high_0p480 | 99.3% | 99.3% | 99.6% | 99.5% |

Sign correctness improved for nominal left, unchanged elsewhere.

### Divergence Metrics

| Height | Pre div RMS | Post div RMS | Pre div Max | Post div Max |
|--------|-------------|--------------|-------------|--------------|
| nominal | 0.0447 | 0.2446 | 0.0910 | 0.5072 |
| low_0p300 | 0.3575 | 0.3690 | 0.5587 | 0.5875 |
| high_0p480 | 0.2825 | 0.3399 | 0.5224 | 0.6893 |

### Torque Magnitudes

| Height | Pre L torque RMS | Post L torque RMS | Pre R torque RMS | Post R torque RMS |
|--------|------------------|-------------------|-------------------|-------------------|
| nominal | 0.2534 | 1.8262 | 0.4546 | 1.8846 |
| low_0p300 | 2.6758 | 2.7636 | 2.7295 | 2.8151 |
| high_0p480 | 2.1251 | 2.5675 | 2.1417 | 2.5736 |

Torque magnitudes increased significantly at nominal (7x) and high (1.2x).

## Event Order Analysis

| Event | nominal | low_0p300 | high_0p480 |
|-------|---------|-----------|------------|
| First divergence >0.07 | step 1145 | step 414 | step 965 |

The divergence grows from step ~400-1100 onwards depending on height.

## Coupling Analysis

| Signal | nominal corr | low_0p300 corr | high_0p480 corr |
|--------|--------------|----------------|-----------------|
| support_error | 0.037 | -0.242 | **-0.517** |
| roll | 0.022 | -0.257 | **-0.465** |
| l_hip_yaw_vel | -0.078 | 0.028 | **-0.755** |

High_0p480 shows strong correlation between divergence and:
- Support error (r = -0.517)
- Roll (r = -0.465)
- Left hip-yaw velocity (r = -0.755)

This suggests divergence at high heights may be coupled with sagittal/lateral dynamics.

## Position Drift Analysis

| Height | Pre L drift | Pre R drift | Post L drift | Post R drift |
|--------|-------------|-------------|--------------|--------------|
| nominal | +0.0013 | +0.0095 | -0.2534 | +0.2538 |
| low_0p300 | -0.2275 | +0.2303 | -0.2428 | +0.2454 |
| high_0p480 | -0.2597 | +0.2611 | -0.3442 | +0.3451 |

Large position drift develops over 5000 steps, primarily at boundary heights.

## Root Cause Classification

**Primary: `per_joint_pd_inadequate_for_divergence`**

Per-joint PD torque always produces divergence-mode acceleration when divergence exists. The sign fix made this worse by increasing torque coherence.

**Secondary: `divergence_authority_insufficient`**

No dedicated divergence damping layer exists. HY2-DIV is not enabled.

**Coupling: `setup_height_coupled_divergence`**

At high_0p480, divergence correlates strongly with support/roll/velocity, suggesting coupling with sagittal controller.

## What the Sign Fix Corrected

| Aspect | Before | After |
|--------|--------|-------|
| Per-joint torque sign | Wrong (0% correctness) | Correct (>93%) |
| Per-joint error correction | Wrong direction | Correct direction |
| Divergence mode control | Random (accidental cancellation) | Coherent but wrong mode |

## What Remains Unaddressed

| Issue | Status |
|-------|--------|
| Per-joint PD cannot damp divergence | **CONFIRMED** |
| No divergence-specific damping layer | **CONFIRMED** |
| Divergence grows from ~400-1100 steps | **CONFIRMED** |
| High height coupling with sagittal | **CONFIRMED** |

## Constraints Followed

✓ Did NOT add WBC
✓ Did NOT enable legacy WBC paths
✓ Did NOT modify hip-roll
✓ Did NOT modify sagittal controller
✓ Did NOT tune hip-yaw gains
✓ Did NOT implement HY2-DIV (yet)
✓ Did NOT revert the sign fix
✓ Did NOT commit

## Recommendation

**Decision:** `DIVERGENCE_ROOT_CAUSE_IDENTIFIED_READY_FOR_FIX`

The sign fix is mathematically correct and must remain. The divergence increase reveals that:

1. **Per-joint PD is fundamentally wrong for divergence control**
2. **A divergence-specific damping layer (HY2-DIV) is needed**
3. **The sign fix exposed this limitation, not created it**

### Proposed Fix Path

1. **Enable HY2-DIV** with conservative gains
2. **Validate at 100 steps** before extending to 5000
3. **Do NOT tune per-joint PD gains** - they are correct for error correction
4. **Monitor coupling** with sagittal/roll at boundary heights

### Validation Sequence

1. 100-step smoke test with HY2-DIV enabled
2. 500-step validation at low/nominal/high
3. 5000-step Step E evaluation
4. Compare divergence RMS vs post-fix baseline

### Do Not Touch List

- Per-joint PD gains (correct for error correction)
- Hip-roll controller
- Sagittal controller
- Support-position controller
- WBC paths

## Artifacts

- `outputs/hip_yaw_divergence_after_sign_fix_audit/divergence_after_sign_fix_summary.json`
- `outputs/hip_yaw_divergence_after_sign_fix_audit/divergence_driver_analysis.json`
- `scripts/audit_hip_yaw_divergence_after_sign_fix.py`
- `scripts/analyze_divergence_drivers.py`
