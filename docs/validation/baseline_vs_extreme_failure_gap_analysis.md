# Phase 4: Baseline vs Extreme Failure Gap Analysis

**Date:** 2026-06-06
**Phase:** BASELINE_VS_EXTREME_FAILURE_GAP_ANALYSIS

---

## 1. Old Five-Variant Baseline Telemetry

### Step E Metrics (from `five_variant_step_e_step_c_baseline_verification_report.md`)

| Variant | Height (m) | Support max (m) | HipYaw max (rad) | Div max (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) |
|---------|-------------|-----------------|------------------|--------------|-----------------|----------------|-------------------|
| low_small | 0.394 | 0.106 | 0.057 | ~0.05 | 0.071 | 0.014 | 3.99 |
| low_tiny | 0.399 | 0.110 | 0.042 | ~0.04 | 0.073 | 0.012 | 4.04 |
| nominal | 0.404 | 0.106 | 0.056 | ~0.05 | 0.071 | 0.013 | 3.87 |
| high_tiny | 0.409 | 0.124 | 0.038 | ~0.04 | 0.092 | 0.011 | 4.12 |
| high_small | 0.414 | 0.135 | 0.030 | ~0.03 | 0.096 | 0.009 | 4.77 |

### Baseline Characteristics

- **Hip-Yaw bounded:** 0.030-0.057 rad (well under 0.10 rad target)
- **Divergence bounded:** ~0.03-0.05 rad (negligible)
- **Support bounded:** 0.106-0.135 m (under 0.15 m gate)
- **Wheel velocity bounded:** 3.87-4.77 rad/s (under 5.0 rad/s gate)
- **Contact valid:** 99.98%
- **HY2-DIV:** Disabled (gate=0 at all heights)
- **Profile:** candidate_D2_wheel_velocity_damping_light

---

## 2. Extreme HY2-DIV/Posture Telemetry

### A0 5000-Step Results (from `posture_standing_validation_a0_5000_report.md`)

| Height | Survived | Support max (m) | HipYaw max (rad) | Div RMS (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) |
|--------|----------|-----------------|------------------|--------------|-----------------|----------------|-------------------|
| nominal (0.404m) | ✓ | 0.159 | 0.254 | 0.245 | 0.089 | 0.014 | - |
| low_0p300 (0.300m) | ✓ | 0.110 | 0.393 | 0.493 | 0.154 | 0.012 | - |
| high_0p480 (0.480m) | ✓ | 0.378 | 0.345 | 0.340 | 0.092 | 0.002 | - |

### Extreme Characteristics

- **Hip-Yaw exceeds:** 0.254-0.393 rad (vs baseline 0.030-0.057 rad)
- **Divergence exceeds:** 0.245-0.493 rad (vs baseline ~0.03-0.05 rad)
- **Support drift higher:** 0.110-0.378 m (vs baseline 0.106-0.135 m)
- **HY2-DIV:** Enabled (A0 profile), but gate=0 at nominal/high
- **A0 tau_max=0.5:** 88.74% clipping at low_0p300

---

## 3. Direct Comparison: Baseline vs Extreme

### Hip-Yaw / Divergence Comparison

| Metric | Baseline (0.394-0.414m) | Extreme nominal (0.404m) | Extreme low (0.300m) | Extreme high (0.480m) |
|--------|-------------------------|------------------------|----------------------|----------------------|
| HipYaw max | 0.030-0.057 rad | **0.254 rad** | **0.393 rad** | **0.345 rad** |
| Divergence | ~0.03-0.05 rad | **0.245 rad** | **0.493 rad** | **0.340 rad** |
| HY2-DIV | Disabled | Disabled (gate=0) | Active (gate=1) | Disabled (gate=0) |
| Clip% | N/A | 0% | **88.74%** | 0% |

**Key difference:** Extreme heights show 5-10× higher hip-yaw/divergence vs baseline.

### Support Drift Comparison

| Metric | Baseline (0.394-0.414m) | Extreme nominal | Extreme low | Extreme high |
|--------|-------------------------|----------------|-------------|--------------|
| Support max | 0.106-0.135 m | 0.159 m | 0.110 m | **0.378 m** |
| Support final | ~0.06-0.08 m | 0.065 m | 0.103 m | 0.039 m |

**Key difference:** high_0p480 shows 3× higher support drift vs baseline.

---

## 4. What Changes at Extremes

### At low_0p300 (0.300m)

| Change | Evidence |
|--------|---------|
| **Height geometry** | hip_pitch=1.376, knee=2.348 (vs nominal hip_pitch~0.90, knee~1.40) |
| **Joint torque demand** | Higher due to more crouched pose |
| **Support width** | 0.347m (vs 0.297m at high, 0.297m nominal) |
| **HY2-DIV gate** | 100% active (below z_low=0.300) |
| **HY2-DIV clipping** | 88.74% (tau_max=0.5 insufficient) |
| **Divergence RMS** | 0.493 rad (vs baseline ~0.05 rad) |

### At high_0p480 (0.480m)

| Change | Evidence |
|--------|---------|
| **Height geometry** | hip_pitch=0.626, knee=1.223 (more upright) |
| **Support width** | 0.297m (narrower, higher tip-over risk) |
| **HY2-DIV gate** | 0% (above z_high=0.393) |
| **Support drift** | 0.378 m (3× baseline) |
| **Divergence correlation** | r=-0.517 (support_error), r=-0.465 (roll), r=-0.755 (velocity) |
| **Divergence RMS** | 0.340 rad (vs baseline ~0.04 rad) |

### At nominal (0.404m) Under Posture Testing

| Metric | Baseline | Posture Test (A0) | Difference |
|--------|---------|-------------------|------------|
| HipYaw max | 0.056 rad | 0.254 rad | **+4.5×** |
| Divergence RMS | ~0.05 rad | 0.245 rad | **+4.9×** |
| Support max | 0.106 m | 0.159 m | **+1.5×** |

**Note:** Nominal shows worse metrics under posture testing despite same height. Possible causes:
1. Different controller configuration (HY2-DIV enabled but gate=0)
2. Different initial conditions
3. Different step count/simulation parameters

---

## 5. Failure Mechanism Classification

### low_0p300 Failure Mechanism

**Primary:** `LOW_HEIGHT_GEOMETRY_COUPLED_INSUFFICIENT_HY2_AUTHORITY`

| Evidence | Classification |
|----------|---------------|
| 88.74% HY2-DIV clipping | HY2-DIV CLIPPING |
| hip_pitch=1.376, knee=2.348 | LOW_HEIGHT_GEOMETRY |
| Divergence 0.493 rad | DIVERGENCE_MODE_GROWS |
| Survived 5000 steps | NO_COLLAPSE |

**Analysis:**
1. Low height requires more crouched pose (hip_pitch=1.376 rad vs 0.90 rad nominal)
2. HY2-DIV is fully active at low height (gate=1.0)
3. But tau_max=0.5 is insufficient → 88.74% clipping
4. Per-joint PD accelerates divergence 97-99% of the time
5. Result: divergence continues to grow despite HY2-DIV

**Classification:** `HY2_DIV_CLIPPING_PRIMARY`

---

### high_0p480 Failure Mechanism

**Primary:** `HIGH_HEIGHT_NO_HY2_ACTIVE_PLUS_SUPPORT_DRIFT_COUPLING`

| Evidence | Classification |
|----------|---------------|
| HY2-DIV gate=0 | HY2_NOT_ACTIVE |
| Support drift 0.378m | SUPPORT_DRIFT |
| r=-0.517 (support_error) | SUPPORT_COUPLING |
| r=-0.465 (roll) | ROLL_COUPLING |
| r=-0.755 (velocity) | VELOCITY_COUPLING |
| Divergence 0.340 rad | DIVERGENCE_MODE_GROWS |

**Analysis:**
1. High height is above z_high=0.393 → HY2-DIV gate=0
2. No divergence damping available at high height
3. Support drift is 3× higher (0.378m vs 0.10m baseline)
4. Strong correlation between divergence and support/roll dynamics
5. Result: divergence grows with support drift coupling

**Classification:** `SUPPORT_DRIFT_COUPLING_PRIMARY`

---

### nominal (posture test) Failure Mechanism

**Primary:** `UNKNOWN_REGRESSION_OR_TEST_MISMATCH`

| Evidence | Classification |
|----------|---------------|
| HipYaw 0.254 rad vs baseline 0.056 rad | **5× REGRESSION** |
| Same height (0.404m) | SAME_HEIGHT |
| HY2-DIV enabled but gate=0 | HY2_INACTIVE |
| No clipping | NOT_AUTHORITY |

**Analysis:**
1. Nominal height should behave like baseline (0.404m)
2. But hip-yaw/divergence is 5× worse than baseline
3. HY2-DIV is enabled but gate=0 (should not affect behavior)
4. Possible causes:
   a. Different simulation parameters (timestep, integration)
   b. Different initial conditions
   c. Controller code changes affecting nominal behavior
   d. Measurement at different simulation horizon

**Classification:** `REQUIRES_INVESTIGATION`

---

## 6. Summary: Why Extremes Fail vs Baseline

| Height | Primary Failure | Secondary Failure | Evidence |
|--------|----------------|-------------------|----------|
| **low_0p300** | HY2-DIV CLIPPING | LOW_HEIGHT_GEOMETRY | 88.74% clip, hip_pitch=1.376 |
| **high_0p480** | SUPPORT_DRIFT_COUPLING | HY2_NOT_ACTIVE | r=-0.517, support=0.378m |
| **nominal (posture)** | UNKNOWN | TEST_MISMATCH | 5× worse vs baseline same height |

### Baseline Success Factors

1. **Height in validated range:** 0.394-0.414m
2. **Moderate joint angles:** hip_pitch ~0.90 rad, knee ~1.40 rad
3. **Adequate support width:** ~0.297m
4. **Low support drift:** 0.106-0.135m
5. **Per-joint PD sufficient:** divergence ~0.03-0.05 rad bounded

### Extreme Failure Factors

1. **Low height:** Crouched geometry → higher torque demand → HY2 clips
2. **High height:** Narrower support + no HY2 → support drift → divergence
3. **Posture test nominal:** Unknown regression or test mismatch

---

## 7. Implications for Height Extension

### Low-Side Extension Strategy

1. **Problem:** HY2-DIV tau_max=0.5 clips at low heights
2. **Solution options:**
   - Increase tau_max (A1/A2 tried, didn't improve)
   - Improve damping gain (not tried systematically)
   - Separate low-height divergence controller
   - Investigate geometry coupling

### High-Side Extension Strategy

1. **Problem:** No HY2 active + support drift coupling
2. **Solution options:**
   - Extend HY2 gate upward (B1/B2 tried, worsened nominal)
   - Address support drift separately
   - Investigate coupling mechanism

### Immediate Investigation Needed

1. **Nominal regression:** Why does posture test show 5× worse vs baseline?
2. **Support drift:** What causes 0.378m drift at high_0p480?
3. **Coupling mechanism:** How does support drift drive divergence?

---

## 8. Files Created

- `outputs/height_range_extension_strategy_audit/baseline_vs_extreme_failure_gap_analysis.json`
- (this document)