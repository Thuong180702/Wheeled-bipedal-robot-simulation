# K2 JAX Dedicated Realtime — Improvement Baseline Freeze

**Phase:** 0 — Freeze Current Improvement Baseline
**Date:** 2026-06-30
**Classification:** `K2_JAX_DEDICATED_REALTIME_IMPROVEMENT_BASELINE`
**K2_STABILITY_SCORE:** 0.6834 (STABILITY_PARTIAL)
**Promotion vs Old K2:** K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL

## Overview

This document captures the current K2 JAX dedicated realtime controller behavior
as the improvement baseline. All future controller changes will be measured against
this baseline using the K2_STABILITY_SCORE (defined in Phase 1).

**Important:** The objective is NO LONGER to match old K2. The previous
source-equivalence investigation is closed. The new objective is to improve
real robot behavior beyond the current dedicated JAX controller.

## Controller Under Test

- **Controller:** `wheeled_biped/controllers/k2_jax_controller.py` (2624 lines)
- **Runner:** `scripts/run_k2_jax_realtime.py`
- **Profile:** `k2_notch_low_q_v1`
- **Backend:** JAX JIT (pure functions, x64 enabled)
- **Control rate:** 100 Hz
- **Physics substeps:** Variable (control_dt / physics_dt)

## Architecture Summary

The current controller has **10 independently-summed torque components**:

```
tau_sum = tau_sagittal(pitch + pitch_rate + sag_vel + position + wheel_vel + com_vy)
        + tau_posture(shape PD: hip_yaw, hip_pitch, knee)
        + tau_yaw(antisymmetric hip_yaw)
        + tau_mode_div(hip-yaw divergence, height-gated)
        + tau_lateral(roll stabilization + stance regularization)
        + empirical_support_ff(fixed torque vector)
```

Components are composed via direct linear superposition with no cross-component
gating. The only coordination is a global per-joint torque clip + rate limit
at the composer stage.

## Validation Results (39/39 scenarios)

### Classification Summary

| Scope | PASS | SAFE_BUT_WORSE | SAFETY_FAIL | Classification |
|-------|------|----------------|-------------|----------------|
| Step C (7) | 6 | 1 (focused_low_0p320) | 0 | SAFE_BUT_WORSE |
| Step E (10) | 6 | 4 (low_0p320, low_0p360, low_0p380, high_0p450) | 0 | SAFE_BUT_WORSE |
| Step D (12) | 12 | 0 | 0 | WITHIN_OLD_TOLERANCE |
| Dynamic Height (5) | 2 | 3 (up_down_cycle, gate_dwell, gate_chatter) | 0 | SAFE_BUT_WORSE |
| Long Run (5) | 2 | 3 (low_0p330, high_0p430, high_0p450) | 0 | SAFE_BUT_WORSE |
| **Total (39)** | **28** | **11** | **0** | **PARTIAL** |

All 11 SAFE_BUT_WORSE are due to pitch_rms_deg vs old K2. Zero falls, zero
safety violations.

## K2_STABILITY_SCORE Baseline

**Aggregate Score: 0.6834 (STABILITY_PARTIAL)**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Posture Stability | 0.650 | 0.30 |
| Support / Drift | 0.720 | 0.20 |
| Leg Health / Hip-Yaw | 0.740 | 0.15 |
| Dynamic Height | 0.625 | 0.15 |
| Torque Quality | 0.690 | 0.10 |
| Robustness | 0.710 | 0.10 |

### Key Baseline Metrics

#### A. Safety — Hard Gates (ALL PASS)

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| Falls | 0 | Must be 0 | PASS |
| NaN/Inf | 0 | Must be False | PASS |
| hip_yaw_joint_max_rad | 0.086 (mean), 0.269 (max) | < 0.35 | PASS |
| pitch_max_deg | 6.54 (mean), 14.53 (max) | — | — |
| roll_max_deg | 0.80 (mean), 1.70 (max) | — | — |

#### B. Posture Stability

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| pitch_rms_deg | 3.92 | 1.02 | 1.66 | 6.19 |
| pitch_peak_deg | 8.45 | 2.13 | 3.72 | 14.53 |
| roll_rms_deg | 0.37 | 0.23 | 0.11 | 1.11 |
| roll_peak_deg | 0.85 | 0.37 | 0.25 | 1.70 |

**Pitch RMS by Height Region:**

| Region | Mean Pitch RMS (deg) | Std | Count |
|--------|----------------------|-----|-------|
| Low (<0.35m) | 3.89 | 0.84 | 12 |
| Mid (0.35-0.43m) | 3.28 | 1.59 | 9 |
| High (>0.43m) | 4.31 | 0.52 | 13 |

**Pitch RMS by Scenario Type:**

| Type | Mean Pitch RMS (deg) | Std | Count |
|------|----------------------|-----|-------|
| Fixed Height | 3.78 | 0.97 | 17 |
| Push | 3.81 | 1.22 | 12 |
| Dynamic Height | 4.46 | 0.95 | 5 |

#### C. Leg Symmetry / Twist

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| hip_yaw_joint_max_rad | 0.086 | 0.048 | 0.016 | 0.269 |
| hip_yaw_div_rms_rad | 0.070 | 0.049 | 0.011 | 0.260 |
| hip_yaw_lr_divergence_deg | 0.88 | 1.98 | 0.00 | 8.25 |
| hip_pitch_symmetry_error_deg | 0.27 | 0.49 | 0.00 | 1.43 |
| knee_symmetry_error_deg | 0.17 | 0.36 | 0.00 | 1.51 |

#### D. Support / Drift

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| support_rms_m | 0.086 | 0.075 | 0.037 | 0.459 |
| support_peak_m | 0.232 | 0.220 | 0.094 | 1.301 |
| sagittal_drift_m | 0.003 | 0.104 | -0.560 | 0.311 |
| lateral_drift_m | -0.002 | 0.215 | -0.330 | 1.253 |
| final_displacement_m | 0.098 | 0.218 | 0.005 | 1.291 |

#### E. Dynamic Height

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| height_rmse_m | 0.013 | 0.022 | 0.001 | 0.112 |
| height_overshoot_m | 0.006 | 0.025 | 0.000 | 0.158 |

#### F. Torque Quality

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| torque_peak_total_nm | 9.39 | 1.70 | 8.00 | 13.71 |
| torque_peak_wheels_nm | 5.10 | 3.11 | 1.46 | 11.58 |
| torque_peak_hip_yaw_nm | 2.12 | 0.95 | 0.66 | 5.22 |

#### G. Robustness

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| contact_loss_frac | 0.001 | 0.0014 | 0.0001 | 0.0055 |
| drift_rate_m_per_kstep | 0.029 | 0.042 | 0.002 | 0.258 |
| stability_score_0_to_1 | 0.782 | 0.049 | 0.680 | 0.884 |

### Performance

| Metric | Value |
|--------|-------|
| Mean Hz | 147.4 |
| Min Hz | 59.3 |
| Max Hz | 199.2 |

## Problematic Cases (Pitch RMS Elevated)

These are the 11 SAFE_BUT_WORSE cases (all pitch_rms_deg vs old K2):

| Scenario | Pitch RMS (deg) | Height Region | Type |
|----------|----------------|---------------|------|
| step_c/focused_low_0p320 | TBD | Low | Fixed |
| step_e/low_0p320 | TBD | Low | Fixed |
| step_e/low_0p360 | TBD | Low | Fixed |
| step_e/low_0p380 | TBD | Low | Fixed |
| step_e/high_0p450 | TBD | High | Fixed |
| dynamic/up_down_cycle | TBD | — | Dynamic |
| dynamic/gate_dwell | TBD | — | Dynamic |
| dynamic/gate_chatter | TBD | — | Dynamic |
| long_run/low_0p330 | TBD | Low | Long |
| long_run/high_0p430 | TBD | High | Long |
| long_run/high_0p450 | TBD | High | Long |

## Known Issues (Pre-Improvement)

1. **Independent torque component summation:** All 10 components are summed
   linearly with no cross-coupling awareness. Components can fight each other
   (e.g., support FF hip-yaw torque vs yaw controller torque on the same joints).

2. **No authority allocation:** Posture controller runs at full authority even
   when balance is near saturation. Hip-yaw correction doesn't yield to pitch
   stabilization.

3. **Support feedforward is height-gated but not pitch-aware:** The support FF
   correction is applied based on height only, without considering pitch phase
   or balance authority demand.

4. **11/39 scenarios have elevated pitch RMS** (SAFE_BUT_WORSE vs old K2).
   While proven to be physics divergence, not a controller bug, these represent
   the primary improvement targets.

5. **Mode-div controller is height-gated with wide smoothstep range**
   (0.30–0.80 m) — active at all heights, with constant gains regardless of
   hip-yaw divergence magnitude.

6. **No shared state estimate across components:** Each component recomputes
   its own effective state (e.g., height is recomputed in multiple places).

7. **Controller conflict index is zero** because component-level torque
   decomposition is not instrumented in telemetry. Actual conflicts are
   invisible to the current diagnostics.

## Improvement Targets

1. **Pitch RMS reduction** in the 11 SAFE_BUT_WORSE cases, especially at
   low heights (0.320–0.380 m) and for dynamic/long-run scenarios.

2. **Support/drift reduction** — current max displacement of 1.29 m in some
   scenarios indicates room for drift control improvement.

3. **Hip-yaw divergence reduction** — max of 0.269 rad is within safety gate
   but could be reduced with better coordination.

4. **Dynamic height smoothness** — dynamic scenarios have the highest pitch RMS
   (4.46 deg mean), indicating transition-related oscillations.

## Missing Telemetry for Deep Analysis

The current telemetry (even `--telemetry full`) cannot compute:
- **Controller conflict index** — requires per-component torque outputs
- **Authority share per component** — requires component-level instrumentation
- **Cross-coupling index** — pitch torque effect on support, yaw effect on sagittal
- **Schedule continuity metrics** — derivative of scheduled parameters vs height

These will be addressed in Phase 3 (instrumentation audit) and Phase 5 (shared
state diagnostics).

## Reproducibility

```bash
# Run baseline validation
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_improvement_baseline

# Run quality analysis
python scripts/analyze_k2_behavior_quality.py \
  --input-dir outputs/k2_improvement_baseline \
  --output docs/validation/k2_improvement_baseline_quality.md

# Compute K2_STABILITY_SCORE
python scripts/evaluate_k2_stability_improvement.py \
  --baseline docs/validation/k2_improvement_baseline_quality.json \
  --candidate docs/validation/k2_improvement_baseline_quality.json \
  --output docs/validation/k2_improvement_baseline_self_eval.md
```

## Acceptance

- [x] Baseline validation run complete (39/39 scenarios)
- [x] Zero falls
- [x] All safety gates pass
- [x] Full-telemetry samples collected (9/39 scenarios)
- [x] Quality analysis report generated
- [x] K2_STABILITY_SCORE computed: 0.6834 (STABILITY_PARTIAL)
- [x] All missing telemetry fields documented for Phase 3+

## Phase 0 Deliverables

| File | Status |
|------|--------|
| `docs/validation/k2_improvement_baseline_freeze.md` | This file |
| `docs/validation/k2_improvement_baseline_quality.md` | Generated |
| `docs/validation/k2_improvement_baseline_quality.json` | Generated |
| `docs/validation/k2_improvement_baseline_self_eval.md` | Generated |
| `docs/validation/k2_improvement_baseline_self_eval.json` | Generated |
| `outputs/k2_improvement_baseline/` | 39 scenario outputs |
| `outputs/k2_improvement_baseline_telemetry/` | 9 full-telemetry samples |
| `scripts/analyze_k2_behavior_quality.py` | Created |
| `scripts/evaluate_k2_stability_improvement.py` | Created |
| `scripts/run_k2_baseline_telemetry_samples.py` | Created |

## Next Phase

**Phase 1:** K2_STABILITY_SCORE objective document → `docs/specs/k2_stability_improvement_objective.md` (already drafted)
**Phase 2:** Build quality evaluator → `scripts/evaluate_k2_stability_improvement.py` (already built)
**Phase 3:** System architecture audit — controller conflict analysis
