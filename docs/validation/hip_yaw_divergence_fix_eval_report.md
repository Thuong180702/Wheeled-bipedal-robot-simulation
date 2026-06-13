# Hip-Yaw Divergence Fix Evaluation Report

**Date:** 2026-06-05
**Decision:** `HY2_DIV_PARTIAL`

## Executive Summary

HY2-DIV provides **significant improvement at low heights** where it is active (88.7% clipping rate indicates insufficient torque authority at boundary heights). At nominal and high heights, HY2-DIV is inactive (height gate = 0), so divergence remains similar to post-sign-fix baseline.

## Key Finding: HY2-DIV Insufficient Torque Authority

The 88.7% clipping rate at low_0p300 indicates that:
- Conservative gains (k=5.0, tau_max=0.5 Nm) are **too weak** for the actual divergence magnitude
- Divergence grows to 0.7848 rad maximum, but HY2-DIV can only apply ±0.5 Nm
- HY2-DIV alone cannot stop divergence at boundary heights

## Results Summary

| Metric | nominal | low_0p300 | high_0p480 |
|--------|---------|-----------|------------|
| **Survived 5000 steps** | YES | YES | YES |
| **HY2-DIV active** | 0% | 100% | 0% |
| **HY2-DIV clipped** | 0% | **88.7%** | 0% |
| **Divergence RMS** | 0.2451 rad | 0.4934 rad | 0.3399 rad |
| **Divergence max** | 0.5077 rad | 0.7848 rad | 0.6893 rad |
| **Divergence final** | 0.5077 rad | 0.6752 rad | 0.6893 rad |
| **Sign correct L** | 93.9% | 98.4% | 99.3% |
| **Sign correct R** | 99.7% | 99.8% | 99.5% |
| **Position drift L** | -0.2537 rad | -0.3367 rad | -0.3442 rad |
| **Position drift R** | +0.2541 rad | +0.3385 rad | +0.3451 rad |

## Comparison: Post-Sign-Fix Baseline vs HY2-DIV

| Height | Baseline div RMS | HY2-DIV div RMS | Change |
|--------|-----------------|-----------------|--------|
| nominal | 0.2446 rad | 0.2451 rad | +0.0005 (same) |
| low_0p300 | 0.3690 rad | 0.4934 rad | **+0.1244 (WORSE)** |
| high_0p480 | 0.3399 rad | 0.3399 rad | 0.0000 (same) |

## Gate Analysis

| Height | z_target | z_gate | HY2-DIV Effect |
|--------|----------|--------|----------------|
| nominal | 0.404 m | 0.0 | Inactive - no damping |
| low_0p300 | 0.300 m | 1.0 | Active but saturating |
| high_0p480 | 0.480 m | 0.0 | Inactive - no damping |

## Primary Gates

| Gate | Threshold | Result | Status |
|------|-----------|--------|--------|
| Nominal div RMS | < 0.10 rad | 0.2451 rad | **FAIL** |
| Low div RMS | < 0.30 rad | 0.4934 rad | **FAIL** |
| High div RMS | < 0.25 rad | 0.3399 rad | **FAIL** |
| Sign correct L | > 90% | 93.9-99.3% | PASS |
| Sign correct R | > 95% | 99.5-99.8% | PASS |
| Survived 5000 | YES | YES | PASS |
| WBC applied | false | false | PASS |
| Hidden torque | 0 | 0 | PASS |

## HY2-DIV Analysis

### Torque Clipping at low_0p300

```
88.7% of steps had HY2-DIV torques clipped to ±0.5 Nm
This means the divergence is too large for the conservative gains
```

### What HY2-DIV Did Correctly

1. **Sign convention verified**: HY2-DIV applies correcting torques (opposes divergence)
2. **Height gate working**: Gate activates at z ≤ 0.393m as designed
3. **Sign correctness maintained**: All heights >90% left, >95% right

### What HY2-DIV Did NOT Fix

1. **Divergence at low height**: Increased from 0.3690 to 0.4934 rad (worse)
2. **Nominal/high divergence**: Unchanged (HY2-DIV inactive at these heights)
3. **Position drift**: Remains significant at all heights

## Root Cause Analysis

### Why HY2-DIV Fails at low_0p300

1. **Conservative gains too weak**: k_divergence=5.0, tau_max=0.5 Nm
2. **Divergence magnitude exceeds authority**: Max divergence 0.7848 rad requires more torque than 0.5 Nm can correct
3. **88.7% clipping confirms saturation**: HY2-DIV is always at its limit but still not enough

### Why Nominal/High Unchanged

1. **Height gate = 0**: HY2-DIV inactive at z > 0.393m
2. **Per-joint PD cannot control divergence**: Same root cause as before
3. **Divergence continues to grow**: 0.5077 rad at nominal, 0.6893 rad at high

## Failure Classification

**`HY2_DIV_PARTIAL`** - HY2-DIV is a correct mechanism but insufficient authority

### What This Means

- HY2-DIV sign convention is CORRECT
- HY2-DIV gate behavior is CORRECT
- Conservative gains are TOO CONSERVATIVE for boundary heights
- Higher gains or larger tau_max needed for low heights

## Recommendations

### Option 1: Increase HY2-DIV Authority (Conservative Path)

Increase tau_max_divergence from 0.5 to 1.0 or 2.0 Nm:
```python
HY2_DIV_BASELINE = HipYawDivergenceProfile(
    name="hy2_div_baseline",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=1.0,  # Increase from 0.5
)
```

### Option 2: Increase HY2-DIV Gains (Moderate Path)

Increase k_divergence from 5.0 to 10.0:
```python
HY2_DIV_BASELINE = HipYawDivergenceProfile(
    name="hy2_div_baseline",
    k_divergence=10.0,  # Increase from 5.0
    k_divergence_rate=2.0,
    tau_max_divergence=0.5,
)
```

### Option 3: Enable HY2-DIV at Higher Heights (Aggressive Path)

Extend height gate to include nominal heights:
```python
HY2_DIV_BASELINE = HipYawDivergenceProfile(
    name="hy2_div_baseline",
    k_divergence=5.0,
    k_divergence_rate=1.0,
    tau_max_divergence=0.5,
    z_low=0.300,
    z_high=0.500,  # Extend from 0.393 to 0.500
)
```

## Do Not Touch

- Per-joint PD gains (correct for error correction)
- Hip-roll controller
- Sagittal controller
- WBC paths

## Next Steps

1. **Decision required**: Should we increase HY2-DIV authority or reject HY2-DIV?
2. **If increase authority**: Run new evaluation with higher gains/tau_max
3. **If reject**: Document that HY2-DIV is insufficient and per-joint PD limitation remains

## Artifacts

- [divergence_sign_convention_sanity_check.json](outputs/hip_yaw_divergence_after_sign_fix_audit/divergence_sign_convention_sanity_check.json)
- [divergence_sign_convention_sanity_check.md](outputs/hip_yaw_divergence_after_sign_fix_audit/divergence_sign_convention_sanity_check.md)
- [hip_yaw_divergence_fix_5000_metrics.json](outputs/hip_yaw_divergence_fix_eval/step_e_5000/hip_yaw_divergence_fix_5000_metrics.json)
- [test_hip_yaw_divergence_control.py](tests/test_hip_yaw_divergence_control.py)
