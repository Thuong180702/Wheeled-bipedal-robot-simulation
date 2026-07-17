# K2 Stability Improvement Objective — K2_STABILITY_SCORE

**Phase:** 1 — Define New Quality Objective and Score
**Date:** 2026-06-30
**Target:** `K2_JAX_DEDICATED_REALTIME_STABILITY_IMPROVED_PASS`

## Purpose

Define a multi-objective quality score for evaluating K2 JAX controller
improvements. This score replaces the old K2 comparison as the promotion target.
The objective is NOT matching old K2 — it is improving real robot behavior
beyond the current dedicated JAX baseline.

## Design Principles

1. **Safety is non-tradeable.** Any fall, NaN/Inf, or safety violation = automatic fail.
2. **No single metric dominates.** Pitch RMS improvement alone is insufficient if
   other metrics regress.
3. **Regression limits are hard.** Improving pitch by worsening hip-yaw, drift,
   support, or torque quality beyond regression limits → fail.
4. **Physically meaningful.** Every metric maps to a real robot behavior concern.
5. **Continuous and smooth.** Scores use continuous functions, not discrete thresholds.

## K2_STABILITY_SCORE Composition

### Score = Σ(w_i × S_i) with hard safety gate

Where w_i are weights and S_i ∈ [0, 1] are per-dimension scores.

### Hard Safety Gate (pass/fail, not scored)

| Condition | Failure Criteria |
|-----------|-----------------|
| Fall | Any scenario fall → AUTOMATIC FAIL |
| NaN/Inf | Any NaN or Inf in state, torque, or telemetry → AUTOMATIC FAIL |
| Hip-yaw joint max | > 0.35 rad in any scenario → AUTOMATIC FAIL |
| Catastrophic instability | Pitch or roll exceeding ±45° for > 10 consecutive steps → AUTOMATIC FAIL |
| Performance | < 50 Hz mean → AUTOMATIC FAIL |

### Soft Score Components and Weights

#### 1. Posture Stability Score (weight = 0.30)

Measures pitch/roll oscillation quality. Lower is better.

```
S_posture = 0.50 × f_pitch(pitch_rms_deg) + 0.25 × f_roll(roll_rms_deg) + 0.15 × f_angvel(ang_vel_rms) + 0.10 × f_peak(max(pitch_peak, roll_peak))
```

Scoring functions (all use smoothstep for continuous transitions):
```
f_pitch(x) = 1.0 - smoothstep(x, 1.5, 6.0)     # Perfect at <1.5°, zero at >6.0° RMS
f_roll(x)  = 1.0 - smoothstep(x, 0.5, 3.0)     # Perfect at <0.5°, zero at >3.0° RMS
f_angvel(x)= 1.0 - smoothstep(x, 5.0, 30.0)    # Perfect at <5°/s, zero at >30°/s RMS
f_peak(x)  = 1.0 - smoothstep(x, 3.0, 15.0)    # Perfect at <3°, zero at >15° peak
```

#### 2. Support / Drift Score (weight = 0.20)

Measures support stability and position drift. Lower is better.

```
S_support = 0.35 × f_support(support_rms_m) + 0.25 × f_displacement(final_displacement_m) + 0.20 × f_sagittal(|sagittal_drift_m|) + 0.20 × f_lateral(|lateral_drift_m|)
```

```
f_support(x)    = 1.0 - smoothstep(x, 0.005, 0.03)   # Perfect at <5mm, zero at >30mm RMS
f_displacement(x)= 1.0 - smoothstep(x, 0.02, 0.20)   # Perfect at <2cm, zero at >20cm
f_sagittal(x)   = 1.0 - smoothstep(x, 0.02, 0.15)   # Perfect at <2cm, zero at >15cm
f_lateral(x)    = 1.0 - smoothstep(x, 0.01, 0.08)    # Perfect at <1cm, zero at >8cm
```

#### 3. Leg Health / Hip-Yaw Score (weight = 0.15)

Measures hip-yaw divergence and leg symmetry. Lower is better.

```
S_leg = 0.40 × f_hy_max(hip_yaw_joint_max_rad) + 0.25 × f_hy_rms(hip_yaw_div_rms_rad) + 0.20 × f_symmetry(leg_posture_error_rms) + 0.15 × f_hip_pitch_sym(hip_pitch_symmetry_error)
```

```
f_hy_max(x)     = 1.0 - smoothstep(x, 0.05, 0.25)    # Perfect at <0.05, zero at >0.25 rad
f_hy_rms(x)     = 1.0 - smoothstep(x, 0.02, 0.15)    # Perfect at <0.02, zero at >0.15 rad
f_symmetry(x)   = 1.0 - smoothstep(x, 0.05, 0.50)    # Perfect at <0.05, zero at >0.50 rad
f_hip_pitch_sym(x)=1.0-smoothstep(x, 1.0, 10.0)     # Perfect at <1°, zero at >10° RMS
```

#### 4. Dynamic Height Score (weight = 0.15)

Measures height tracking quality. Lower RMSE and smoother transitions are better.

```
S_dyn_height = 0.40 × f_rmse(height_rmse_m) + 0.30 × f_overshoot(overshoot_m) + 0.20 × f_smoothness(transition_smoothness) + 0.10 × f_qref(q_ref_tracking_error_rms)
```

```
f_rmse(x)       = 1.0 - smoothstep(x, 0.005, 0.04)    # Perfect at <5mm, zero at >40mm RMSE
f_overshoot(x)  = 1.0 - smoothstep(x, 0.01, 0.06)    # Perfect at <1cm, zero at >6cm
f_smoothness(x) = 1.0 - smoothstep(x, 0.1, 2.0)      # Perfect at <0.1 jerk, zero at >2.0
f_qref(x)       = 1.0 - smoothstep(x, 0.02, 0.20)    # Perfect at <0.02, zero at >0.20 rad
```

#### 5. Torque Quality Score (weight = 0.10)

Measures torque efficiency and smoothness. Lower is better.

```
S_torque = 0.35 × f_tau_rms(torque_rms_pooled) + 0.25 × f_tau_peak(torque_peak_total) + 0.25 × f_tau_rate(torque_rate_rms) + 0.15 × f_sat_count(torque_saturation_count)
```

```
f_tau_rms(x)    = 1.0 - smoothstep(x, 1.0, 8.0)      # Perfect at <1 Nm, zero at >8 Nm RMS
f_tau_peak(x)   = 1.0 - smoothstep(x, 5.0, 30.0)     # Perfect at <5 Nm, zero at >30 Nm peak
f_tau_rate(x)   = 1.0 - smoothstep(x, 50.0, 400.0)   # Perfect at <50 Nm/s, zero at >400 Nm/s RMS
f_sat_count(x)  = 1.0 - smoothstep(x, 5, 100)         # Perfect at <5 steps, zero at >100 steps
```

#### 6. Robustness Score (weight = 0.10)

Measures long-run stability and disturbance recovery.

```
S_robust = 0.35 × f_contact_loss(contact_loss_frac) + 0.30 × f_drift_rate(drift_rate_m_per_kstep) + 0.20 × f_post_push(post_pitch_rms_500_deg) + 0.15 × f_stability(stability_score)
```

```
f_contact_loss(x) = 1.0 - smoothstep(x, 0.001, 0.05)  # Perfect at <0.1%, zero at >5%
f_drift_rate(x)   = 1.0 - smoothstep(x, 0.01, 0.20)  # Perfect at <1cm/kstep, zero at >20cm/kstep
f_post_push(x)    = 1.0 - smoothstep(x, 2.0, 10.0)   # Perfect at <2°, zero at >10° RMS
f_stability(x)    = x                                   # Pass through (already 0-1)
```

## Aggregate K2_STABILITY_SCORE

```
K2_STABILITY_SCORE = 0.30 × S_posture + 0.20 × S_support + 0.15 × S_leg + 0.15 × S_dyn_height + 0.10 × S_torque + 0.10 × S_robust
```

Range: [0, 1]. Higher is better.

### Classification Thresholds

| Score Range | Classification |
|-------------|---------------|
| ≥ 0.80 | STABILITY_IMPROVED_PASS |
| 0.60 – 0.79 | STABILITY_PARTIAL |
| 0.40 – 0.59 | STABILITY_REGRESSED |
| < 0.40 or any safety fail | SAFETY_FAIL |

## Regression Limits

A candidate that improves aggregate score but regresses any individual metric
beyond these absolute limits fails:

| Metric | Hard Regression Limit |
|--------|----------------------|
| pitch_rms_deg | Must not increase by > 1.0° absolute or > 20% relative |
| hip_yaw_joint_max_rad | Must not increase by > 0.05 rad absolute |
| support_rms_m | Must not increase by > 0.015 m absolute |
| height_rmse_m | Must not increase by > 0.015 m absolute (dynamic scenarios) |
| final_displacement_m | Must not increase by > 0.10 m absolute |
| torque_peak_total_nm | Must not increase by > 5.0 Nm absolute |
| contact_loss_frac | Must not increase by > 0.02 absolute |
| achieved_hz | Must not drop below 50 Hz |

## Per-Scenario Weighting

The aggregate score is computed as a weighted average across scenarios:

- Fixed-height scenarios (Step C + Step E): weight = 17/39 each → equal weight per scenario
- Push scenarios (Step D): weight = 12/39 each
- Dynamic height scenarios: weight = 5/39 each
- Long-run scenarios: weight = 5/39 each

This gives equal total weight to each scenario. For the aggregate, scenario scores
are computed first, then averaged.

## Score Implementation

The score will be computed by `scripts/evaluate_k2_stability_improvement.py`
(Phase 2) from the quality metrics generated by `scripts/analyze_k2_behavior_quality.py`.

### smoothstep Definition

```python
def smoothstep(x, low, high):
    """Smoothstep from 0 at x<=low to 1 at x>=high."""
    t = (x - low) / max(high - low, 1e-9)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)
```

## Acceptance Criteria

- [ ] Hard safety gates correctly identified and non-tradeable
- [ ] All 6 soft dimensions have physically motivated scoring functions
- [ ] Weights sum to 1.0 and reflect control priorities
- [ ] Regression limits are absolute and scenario-independent
- [ ] No single metric can hide poor performance in other dimensions
- [ ] Scoring functions are continuous (no discrete thresholds)
- [ ] Old K2 metrics are NOT the target — current baseline is

## Next Phase

Phase 2 will implement `scripts/evaluate_k2_stability_improvement.py` to
compute K2_STABILITY_SCORE from quality analysis data.
