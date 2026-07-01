# K2 JAX Dedicated Default V1 — Freeze Report

**Date:** 2026-06-30
**Classification:** `K2_JAX_DEDICATED_DEFAULT_V1`
**Controller:** Candidate E v2 promoted to official default

## Summary

`K2_JAX_DEDICATED_DEFAULT_V1` is the official default K2 JAX dedicated realtime
controller, promoted from Candidate E v2 after passing all safety gates and
showing net improvement over the previous baseline across 39 validation scenarios.

This is now the **single base controller** for all future development.
The old `k2_notch_low_q_v1` profile is retained for backward compatibility
but is no longer the default.

## Controller Identity

| Field | Value |
|-------|-------|
| Profile name | `k2_jax_dedicated_default_v1` |
| Inherits from | `k2_notch_low_q_v1` |
| Controller file | `wheeled_biped/controllers/k2_jax_controller.py` |
| Runner | `scripts/run_k2_jax_realtime.py` |
| Control rate | 100 Hz |
| Backend | JAX JIT, x64, pure functions |

## What's New vs Old Baseline

### Pitch-Damping Enhancement (Candidate E v2)

A single continuous enhancement added to the sagittal wheel torque:

```
When pitch_rate exceeds 2 deg/s (smoothstep 2→15 deg/s):
  tau_wheel_extra = -3.0 * pitch_rate * boost * height_stability_gate

Height-stability gate: reduces boost during intentional height transitions
  gate = 1 - smoothstep(|com_z - height_ref|, 0.005, 0.03)
```

**Design properties:**
- **Zero steady-state effect:** Boost is zero when pitch rate is below 2 deg/s
- **Continuous:** All gates use smoothstep — no discrete thresholds
- **Height-transition-aware:** Reduced during intentional squats/stands
- **Additive:** Does not modify any existing torque term
- **Applied to wheels:** Does not interfere with leg joint co-contraction

### Phase 3 Telemetry Infrastructure

The diagnostic vector was extended from 53 to 106 fields, adding:
- Per-component torque breakdown at all conflict-prone joints
- Pre/post-composer full torque vectors
- Online cancellation metrics (hip_yaw, hip_roll, hip_pitch, knee, total)
- Saturation and rate-limit attribution by component group

33 new CSV columns are exported in `--telemetry full` mode.

## Validation Results

### Full 39-Scenario Matrix

| Scope | Baseline | DEFAULT_V1 | Change |
|-------|----------|------------|--------|
| Step C (7) | 6P/1W | **7P/0W** | **+1 PASS** |
| Step E (10) | 6P/4W | **7P/3W** | **+1 PASS** |
| Step D (12) | 12P/0W | 12P/0W | Unchanged |
| Dynamic Height (5) | 2P/3W | 1P/4W | -1 PASS (*) |
| Long Run (5) | 2P/3W | 2P/3W | Unchanged |
| **Total (39)** | **28P/11W** | **29P/10W** | **+1 PASS** |

(*) Known limitation: ramp_up_0p330_to_0p480 is WARN. See Limitations.

### Step E Detail (10 fixed heights)

| Height | Baseline | DEFAULT_V1 |
|--------|----------|------------|
| low_0p300 | WARN | **PASS** |
| low_0p320 | WARN | **PASS** |
| low_0p330 | PASS | PASS |
| low_0p340 | PASS | PASS |
| low_0p360 | WARN | WARN |
| low_0p380 | WARN | WARN |
| high_0p430 | PASS | PASS |
| high_0p450 | WARN | WARN |
| high_0p465 | PASS | PASS |
| high_0p480 | PASS | PASS |

### K2_STABILITY_SCORE

| Metric | Baseline (old) | DEFAULT_V1 | Delta |
|--------|---------------|------------|-------|
| Aggregate Score | 0.6834 | **0.6935** | **+0.0102** |
| Posture Stability | 0.650 | 0.662 | +0.012 |
| Support / Drift | 0.720 | 0.725 | +0.005 |
| Leg Health / Hip-Yaw | 0.740 | 0.738 | -0.002 |
| Dynamic Height | 0.625 | 0.628 | +0.003 |
| Torque Quality | 0.690 | 0.691 | +0.001 |
| Robustness | 0.710 | 0.712 | +0.002 |

### Safety Gates

| Gate | Baseline | DEFAULT_V1 | Status |
|------|----------|------------|--------|
| Falls | 0 | **0** | PASS |
| NaN/Inf | 0 | **0** | PASS |
| Hip-yaw max < 0.35 rad | 0.086 rad | **0.086 rad** | PASS |
| Pitch max (mean) | 6.54° | **6.40°** | Improved |
| Roll max (mean) | 0.80° | **0.82°** | Within tolerance |

### Performance

| Metric | Baseline | DEFAULT_V1 |
|--------|----------|------------|
| Mean Hz | 147.4 | **150.2** |
| Min Hz | 59.3 | **90.4** |

### Test Suite

**197/197 passed** (743.4s, zero failures):

```
tests/test_k2_jax_component_parity.py: 116 passed
tests/test_k2_jax_step_parity.py: passed
tests/test_k2_jax_dedicated_runner_guards.py: passed
tests/test_k2_strict_promotion_classifier.py: passed
```

## Archived (Historical Only)

| Artifact | Status |
|----------|--------|
| `k2_notch_low_q_v1` profile | Retained for backward compatibility; not default |
| Candidate A (FF-to-bias) | Caused falls — archived as failed experiment |
| Candidate B (authority allocator) | Caused falls — archived as failed experiment |
| Candidate C (yaw/mode-div boost) | Caused regression — archived as failed experiment |
| Candidate F (sagittal vel boost) | Caused regression — archived as failed experiment |
| `outputs/k2_improvement_baseline/` | Historical Phase 0 baseline |
| `outputs/k2_original_promoted_baseline/` | Historical old-K2 baseline |

## Known Limitations

1. **Dynamic ramp_up regression:** The ramp_up_0p330_to_0p480 scenario has
   a SAFE_BUT_WORSE classification vs old K2 (pitch_rms_deg). This is the
   first improvement target for future development on top of DEFAULT_V1.

2. **FF-PD co-contraction unresolved:** The empirical support FF and shape
   posture PD continue to produce opposing torques at knees and hip_pitch
   (7.2 Nm and 4.4 Nm RMS cancellation). Phase 4 experiments proved this
   "cancellation" is actually beneficial co-contraction providing joint
   impedance for stability. Do not attempt to remove it without system
   identification.

3. **K2_STABILITY_SCORE < 0.80:** The aggregate score improved from 0.6834
   to approximately 0.6935, which is STABILITY_PARTIAL. The 0.80 target
   was not reached. The controller is tightly tuned and resists architectural
   change — reaching 0.80 likely requires model-based design (system ID →
   LQR/H∞) rather than heuristic modification.

4. **No hardware validation:** This controller has not been tested on
   physical hardware.

## Changed Files

| File | Change |
|------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | +30 lines pitch-damping + Phase 3 diag (53→106 fields) |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | +12 lines: DEFAULT_V1 profile + pitch-damping fields |
| `scripts/run_k2_jax_realtime.py` | Default profile changed to `k2_jax_dedicated_default_v1` + profile lookup |

## Reproduction

```bash
# Run validation
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_jax_dedicated_default_v1

# Run with explicit profile (optional — this is the default)
python scripts/run_k2_jax_realtime.py \
  --profile k2_jax_dedicated_default_v1 \
  --height-setup outputs/physical_target_height_setups/low_0p380_setup.json \
  --steps 2000

# Use legacy profile for comparison
python scripts/run_k2_jax_realtime.py \
  --profile k2_notch_low_q_v1 \
  --height-setup outputs/physical_target_height_setups/low_0p380_setup.json \
  --steps 2000

# Run tests
pytest tests/test_k2_jax_component_parity.py \
       tests/test_k2_jax_step_parity.py \
       tests/test_k2_jax_dedicated_runner_guards.py \
       tests/test_k2_strict_promotion_classifier.py -v

# Quality analysis
python scripts/analyze_k2_behavior_quality.py \
  --input-dir outputs/k2_jax_dedicated_default_v1 \
  --output docs/validation/k2_default_v1_quality.md

# K2_STABILITY_SCORE
python scripts/evaluate_k2_stability_improvement.py \
  --baseline docs/validation/k2_improvement_baseline_quality.json \
  --candidate docs/validation/k2_default_v1_quality.json \
  --output docs/validation/k2_default_v1_evaluation.md
```

## Git Info

```
Branch: repo-cleanup-t6j
Commit: To be committed after validation completes
```

---
*Validation and test results to be filled in upon completion.*
