# K2 JAX Dedicated Runner — Phase 1 Parameter Source-of-Truth Fix

**Date:** 2026-06-29
**Commit:** `0e1c713` (baseline) → post-fix
**Status:** COMPLETE — 0 control-affecting parameter mismatches

## Root Cause

The dedicated runner (`scripts/run_k2_jax_realtime.py`) maintained a hardcoded `K2_PROFILE` dict duplicating values from the canonical `K2_NOTCH_LOW_Q_V1` profile in `sagittal_velocity_damped_balance_controller.py`. This violated the single-source-of-truth principle and risked parameter drift.

## Changes Made

### 1. Import canonical profile source (`run_k2_jax_realtime.py`)

**Before:**
```python
# Hardcoded K2_PROFILE dict with duplicate values
K2_PROFILE = {
    "velocity_damping_scale": 1.1,
    "apcr1nd_hold_outside_band": True,
    ...
}
```

**After:**
```python
from wheeled_biped.controllers.sagittal_velocity_damped_balance_controller import (
    K2_NOTCH_LOW_Q_V1 as _K2_AUTH_SCHED,
)
```

### 2. Read params from canonical profile

**Before:** `pack_params_stage2(velocity_damping_scale=K2_PROFILE["velocity_damping_scale"], ...)`

**After:** `pack_params_stage2(velocity_damping_scale=vel_damp_scale, ...)` where `vel_damp_scale` is computed via `_K2_AUTH_SCHED.is_active_for_variant()` — identical logic to the canonical path at `simulate_hierarchical_controller.py:5470-5475`.

### 3. Added `--dump-k2-params` flag

Writes all control-affecting JAX params and equilibrium constants to a JSON file for comparison between dedicated and canonical paths.

```bash
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --dump-k2-params outputs/param_dump.json --steps 10 --quiet --telemetry off
```

## Verified Parameter Parity

| Parameter | Canonical Source | Dedicated Source | Match |
|-----------|-----------------|------------------|-------|
| `velocity_damping_scale` | `_auth_sched.velocity_damping_scale` via `is_active_for_variant()` | `_K2_AUTH_SCHED.velocity_damping_scale` via `is_active_for_variant()` | ✅ |
| `apcr1nd_hold_outside_band` | `_auth_sched.apcr1nd_hold_outside_band` | `_K2_AUTH_SCHED.apcr1nd_hold_outside_band` | ✅ |
| `apcr1nd_startup_guard_steps` | `_auth_sched.recenter_priority_startup_guard_steps` | Same | ✅ |
| `apcr1nd_safe_min_com_z` | `_auth_sched.recenter_priority_safe_min_com_z` | Same | ✅ |
| `apcr1nd_safe_roll_rad` | `_auth_sched.recenter_priority_safe_roll_rad` | Same | ✅ |
| `apcr1nd_safe_pitch_rad` | `_auth_sched.recenter_priority_safe_pitch_rad` | Same | ✅ |
| `apcr1nd_direct_enter_m` | `_auth_sched.apcr1nd_direct_enter_m` | Same | ✅ |
| `apcr1nd_release_inner_m` | `_auth_sched.apcr1nd_release_inner_m` | Same | ✅ |
| `apcr1nd_converging_release_steps` | `_auth_sched.apcr1nd_converging_release_steps` | Same | ✅ |
| `k_velocity` | 15.0 (from sagittal controller) | 15.0 (constant) | ✅ |
| `mode_div_soft_gain` | 0.80 | 0.80 | ✅ |
| `mode_div_ref_source` | `"disabled"` (default) | `"disabled"` (default) | ✅ |

## Tests

`tests/test_k2_jax_dedicated_param_parity.py` — 20 tests, all passing:
- Flat param array identity for 5 variants (None, high_0p480, low_0p300, low_0p330, high_0p430)
- Unpacked control param matching for 3 variants
- velocity_damping_scale gating (1.0 baseline, 1.1 for supported variants)
- apcr1nd_hold_outside_band source-of-truth = True
- No hardcoded K2_PROFILE dict in dedicated runner
- --dump-k2-params produces valid JSON

## Acceptance

- [x] 0 control-affecting parameter mismatches
- [x] velocity_damping_scale resolved (reads from canonical source)
- [x] apcr1nd_hold_outside_band resolved (reads from canonical source)
- [x] K2 profile source-of-truth documented: `K2_NOTCH_LOW_Q_V1` at `sagittal_velocity_damped_balance_controller.py:3162`
- [x] No hidden hardcoded mismatch remains
- [x] Tests added for future mismatch detection
