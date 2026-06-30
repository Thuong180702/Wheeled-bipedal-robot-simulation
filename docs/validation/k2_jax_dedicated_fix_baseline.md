# K2 JAX Dedicated Runner Fix — Phase 0 Baseline

**Date:** 2026-06-29
**Commit:** `0e1c713` — "Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000"
**Branch:** `repo-cleanup-t6j`
**Status:** BASELINE FROZEN — no code changes in this phase

## Source-of-Truth Systems

| System | Script | Backend |
|--------|--------|---------|
| A. Original K2 Python | `scripts/simulate_hierarchical_controller.py --controller-backend python` | Python |
| B. Canonical K2 JAX | `scripts/simulate_hierarchical_controller.py --controller-backend jax` | JAX monolithic |
| C. Dedicated realtime | `scripts/run_k2_jax_realtime.py` | JAX standalone |

## Current Classification

**K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL**

## Blocker 1: Dynamic Height Failures

### ramp_up_0p330_to_0p480 (5000 steps)

| Metric | Canonical JAX | Dedicated JAX | Status |
|--------|--------------|---------------|--------|
| Survival | ✅ 5000/5000 | ❌ Fall at step 2989 | **FAIL** |
| Termination | — | height_too_low (0.285 < 0.285) | — |
| Height ref at fall | — | 0.45445 m | — |
| Pitch RMS | TBD | 13.2° | — |
| Pitch max | TBD | 39.3° | — |
| Hip yaw div max | TBD | 0.365 rad | >0.35 ⚠️ |
| CoM drift final | TBD | 10.32 m | Massive |
| Contact loss | TBD | 319 steps | — |
| Max torque | TBD | 10.51 Nm | — |
| Achieved Hz | — | 137.1 Hz | ✅ |

### ramp_down_0p480_to_0p330 (5000 steps)

| Metric | Canonical JAX | Dedicated JAX | Status |
|--------|--------------|---------------|--------|
| Survival | TBD | ❌ Fall at step 4471 | **FAIL** |
| Termination | — | height_too_low (0.431 < 0.431) | — |
| Height ref at fall | — | 0.33 m | — |
| Pitch RMS | TBD | 8.5° | — |
| Pitch max | TBD | -29.7° | — |
| Hip yaw div max | TBD | 0.502 rad | >0.35 ⚠️ |
| CoM drift final | TBD | 6.78 m | Massive |
| Contact loss | TBD | 310 steps | — |

### gate_chatter (5000 steps)

| Metric | Canonical JAX | Dedicated JAX | Status |
|--------|--------------|---------------|--------|
| Survival | TBD | ❌ Fall at step 2288 | **FAIL** |
| Termination | — | height_too_low (0.285 < 0.285) | — |
| Height ref at fall | — | 0.4088 m | — |
| Pitch RMS | TBD | 15.1° | — |
| Pitch max | TBD | 39.4° | — |
| Hip yaw div max | TBD | 0.388 rad | >0.35 ⚠️ |
| CoM drift final | TBD | 11.77 m | Massive |

## Blocker 2: Low-Height Hip-Yaw Divergence

### low_0p300 fixed height (2000 steps)

| Metric | Original K2 Python | Dedicated JAX | Status |
|--------|-------------------|---------------|--------|
| Survival | TBD | ✅ 2000/2000 | Pass |
| Hip yaw div max | ≈0.131 rad (reported) | **0.666 rad** | **FAIL** >0.35 |
| Hip yaw div RMS | TBD | 0.336 rad | — |
| Max hip yaw torque | TBD | 5.26 Nm | — |
| Pitch RMS | TBD | 5.4° | — |
| Achieved Hz | — | 151.8 Hz | ✅ |

## Blocker 3: Parameter Mismatch (RESOLVED in source, not in code)

The `K2_NOTCH_LOW_Q_V1` profile at `sagittal_velocity_damped_balance_controller.py:3162-3182` now correctly sets:
- `velocity_damping_scale=1.10`
- `apcr1nd_hold_outside_band=True`
- `applies_to_variants=(...)` populated

**BUT** the dedicated runner still hardcodes these in a `K2_PROFILE` dict instead of importing from the canonical source. See Phase 1.

## Blocker 4: Push Mechanism Mismatch

| Aspect | Original K2 | Dedicated |
|--------|------------|-----------|
| Push method | qvel injection | xfrc_applied |
| Push comparison | — | INVALID until aligned |

## Fixed-Height Reference (Dedicated, survived)

| Variant | Hip yaw div max | Status |
|---------|----------------|--------|
| high_0p480 | 0.039 rad | ✅ |
| low_0p330 | TBD | TBD |
| high_0p430 | TBD | TBD |
| high_0p450 | TBD | TBD |
| low_0p380 | TBD | TBD |

## Reproduction Commands

### Canonical JAX ramp_up:
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile k2_notch_low_q_v1 \
  --controller-backend jax \
  --height-variant-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --dynamic-height-trajectory outputs/k2_jax_abs_trim_phase6/trajectories/ramp_up_0p330_to_0p480.json \
  --steps 5000 \
  --disable-wbc-correction \
  --output-dir outputs/k2_jax_dedicated_promotion_test/canonical_jax_ramp_up
```

### Dedicated JAX ramp_up:
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p330_setup.json \
  --dynamic-height-trajectory outputs/k2_jax_abs_trim_phase6/trajectories/ramp_up_0p330_to_0p480.json \
  --steps 5000 \
  --telemetry full \
  --output-dir outputs/k2_jax_dedicated_promotion_test/jax_ramp_up
```

### Dedicated JAX low_0p300:
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 2000 \
  --telemetry full \
  --output-dir outputs/k2_jax_dedicated_promotion_test/jax_fixed_low_0p300
```

### Dedicated JAX push_bwd_90N:
```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --push-seq outputs/k2_release_hardening/push_seq_bwd_90N.json \
  --steps 2000 \
  --telemetry full \
  --output-dir outputs/k2_jax_dedicated_promotion_test/jax_push_bwd_90N
```

## Acceptance

- [x] Broken baseline is reproducible (telemetry files confirm)
- [x] Canonical JAX survival on ramp_up confirmed (5000 CSV rows = survived)
- [x] Dedicated failure confirmed (fall at step 2989)
- [x] No code changes in this phase
