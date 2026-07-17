# K2 JAX Python APCR1ND Source Trace — Phase 1

**Date:** 2026-06-28
**Branch:** repo-cleanup-t6j

## Key Finding

APCR1ND IS active for K2_NOTCH_LOW_Q_V1. The `recenter_priority_enabled`, `recenter_priority_direct_enabled`, `apcr1nd_tuned_enabled`, and `vd_wheel_damping_recenter_override_enabled` fields are all True, inherited through the profile chain from `SUPPORT_POSITION_OUTER_LOOP_PITCH_REF` → `HEIGHT_SCHEDULED_PITCH_EQUILIBRIUM_TRIM` → (deeper base).

## The Actual Divergence

At step 207 of ramp_up:
```
tau_wheel_velocity_left:  py=-1.576 (k=0.5 × wv=3.15) → no override
                          jx=-0.788 (k=0.5 × wv=1.576 → 50% scaled) → override applied
```

**JAX over-applies the APCR1ND override** because it checks only `wheel_scale < 1.0` (position error band) without the full gating chain. Python requires `apcr1n_recenter_priority_active = True` which gates on: startup guard, safety, drift direction, and band hysteresis.

## A. Activation Gates

### Python source: `sagittal_velocity_damped_balance_controller.py`

| # | Gate Variable | File:Line | Condition | K2 Value | JAX Status |
|---|--------------|-----------|-----------|----------|------------|
| A1 | `recenter_priority_enabled` | :734, field | Must be True | **True** | ✅ `_K2_APCR_ENABLED` reads from K2 profile |
| A2 | `recenter_priority_startup_guard_steps` | :732, field | Skip first N steps | 40 (default) | ❌ **MISSING** — no startup guard in JAX |
| A3 | `recenter_priority_direct_enabled` | :751, field | Use direct trigger vs APC | **True** | ❌ **MISSING** — JAX doesn't check this |
| A4 | `apcr1nd_direct_recenter_priority_active` | :6342-6489 | Direct trigger active state | Computed each step | ❌ **MISSING** — entire gating logic missing |
| A5 | `apcr1nd_tuned_enabled` | :764, field | Use tuned band thresholds | **True** | ✅ `_K2_APCR_TUNED` |
| A6 | Safety: `contact_valid` | :6378 | Contact required | Assumed True | ❌ **MISSING** |
| A7 | Safety: `com_z_safe` | :6375 | `com_z >= safe_min_com_z` (0.25 default) | ~always pass | ❌ **MISSING** |
| A8 | Safety: `roll_safe` | :6376 | `abs(roll) <= safe_roll_rad` (0.30 default) | ~always pass | ❌ **MISSING** |
| A9 | Safety: `pitch_safe` | :6377 | `abs(pitch) <= safe_pitch_rad` (0.30 default) | ~always pass | ❌ **MISSING** |

## B. Drift Detection

| # | Scalar | File:Line | Formula | K2 Value | JAX Status |
|---|--------|-----------|---------|----------|------------|
| B1 | `signed_error` | :6360 | `sagittal_position_error_m` | float | ✅ Passed as `sag_pos_err` |
| B2 | `abs_error` | :6361 | `abs(signed_error)` | float | ✅ Computed in JAX |
| B3 | `e_dot` | :6362 | `signed_error - _apcr1nd_prev_error` | float | ❌ **MISSING** — no `_apcr1nd_prev_error` state |
| B4 | `moving_away` | :6364 | `signed_error * e_dot > 0.0` | bool | ❌ **MISSING** |
| B5 | `converging` | :6365 | `not moving_away and abs(e_dot) > 1e-6` | bool | ❌ **MISSING** |
| B6 | `drift_sign` | :6549-6550 | `1.0 if signed_error > 0 else -1.0` | ±1 | ✅ In JAX override function |
| B7 | `wheel_vel_sign` | :6559-6560 | `sign(wheel_vel_mean)` | ±1 | ✅ In JAX override function |
| B8 | `damping_fights_drift` | :6565-6567 | `abs(drift_sign - wheel_vel_sign) < 0.5` | bool | ✅ In JAX override function |

## C. Band-Based Wheel Damping Override

| # | Scalar | File:Line | Python Value (K2) | JAX Status |
|---|--------|-----------|-------------------|------------|
| C1 | `soft_enter_m` | :6578 | 0.05 | ✅ `_K2_APCR_SOFT_ENTER_M` |
| C2 | `desired_band_m` | :6577 | 0.08 | ✅ `_K2_APCR_DESIRED_BAND_M` |
| C3 | `hard_band_m` | :6575 | 0.10 | ✅ `_K2_APCR_HARD_BAND_M` |
| C4 | `emergency_band_m` | :6576 | 0.12 | ✅ `_K2_APCR_EMERGENCY_BAND_M` |
| C5 | Scale: normal | :6590 | 1.0 | ✅ |
| C6 | Scale: soft | :6588 | 0.50 (K2 tuned) | ✅ |
| C7 | Scale: desired | :6586 | 0.30 | ✅ |
| C8 | Scale: hard | :6584 | 0.15 | ✅ |
| C9 | Scale: emergency | :6582 | 0.10 | ✅ |
| C10 | `preserve_damping_if_helps` | :6593 | True | ✅ |
| C11 | `min_damping` | :6621 | 0.50 Nm | ✅ |
| C12 | `apply_override` gate | :6606-6608 | `tuned & scale<1.0` | ✅ (but MISSES prior gate A4) |

## D. Tuned Variant Hysteresis (MISSING from JAX)

| # | Scalar | File:Line | Purpose | JAX Status |
|---|--------|-----------|---------|------------|
| D1 | `_apcr1nd_tuned_recenter_held` | :6417, :6441-6461 | Latch: stays active until released | ❌ **MISSING** |
| D2 | `_apcr1nd_tuned_converging_steps` | :6390-6393 | Counts consecutive converging steps | ❌ **MISSING** |
| D3 | `release_inner_m` | :6385 | Release hysteresis threshold | ❌ **MISSING** |
| D4 | `hold_outside_band` | :6386 | Keep active when outside desired band | ❌ **MISSING** |
| D5 | `converging_release_steps` | :6387 | Steps before converging release | ❌ **MISSING** |
| D6 | Emergency entry (`abs_error >= desired_band_m`) | :6422 | Immediate activation regardless of drift | ❌ **MISSING** |
| D7 | Soft entry (`soft_enter_m <= abs_error < direct_enter_m` AND moving_away) | :6420 | Early entry with drift | ❌ **MISSING** |
| D8 | Direct entry (`abs_error >= direct_enter_m` AND moving_away) | :6421 | Standard entry with drift | ❌ **MISSING** |
| D9 | Release by inner band (`abs_error <= release_inner_m`) | :6429 | Release when error small | ❌ **MISSING** |
| D10 | Release by converging | :6430-6433 | Release after sustained convergence | ❌ **MISSING** |

## E. Interaction with Sagittal Torque Assembly

**Python** (lines 6605-6631):
1. Gate A4 (`apcr1n_recenter_priority_active`) must pass
2. Then check band → wheel_scale
3. Then `apply_override` = (tuned & scale<1.0) OR (!tuned & fights_drift)
4. If apply: scale tau_wheel_vel_left/right by wheel_scale
5. Then min-clamp: if |tau| < min_damping, set to ±min_damping
6. Tau_wheel_vel is modified IN PLACE before the final tau assembly

**JAX** (lines 1527-1540):
1. Calls `k2_jax_apcr1nd_wheel_damping_override()` AFTER sagittal assembly
2. Function applies band scaling and min-clamp unconditionally when `wheel_scale < 1.0`
3. **MISSES** the `apcr1n_recenter_priority_active` gate (A4)
4. **MISSES** all tuned variant hysteresis (D1-D10)

## F. State Fields Required for Parity

| Python State Variable | Type | Purpose | JAX Status |
|----------------------|------|---------|------------|
| `_apcr1nd_step_counter` | int | Startup guard counter | ❌ NOT IN JAX STATE |
| `_apcr1nd_prev_error` | float | Previous signed error for e_dot | ❌ NOT IN JAX STATE |
| `_apcr1nd_tuned_converging_steps` | int | Consecutive converging steps | ❌ NOT IN JAX STATE |
| `_apcr1nd_tuned_recenter_held` | bool | Hold/latch state | ❌ NOT IN JAX STATE |

## G. Params Required for Parity

| Param | K2 Value | Purpose |
|-------|----------|---------|
| `recenter_priority_startup_guard_steps` | 40 | Skip first N steps |
| `recenter_priority_safe_min_com_z` | 0.25 m | Height safety gate |
| `recenter_priority_safe_roll_rad` | 0.30 rad | Roll safety gate |
| `recenter_priority_safe_pitch_rad` | 0.30 rad | Pitch safety gate |
| `apcr1nd_direct_enter_m` | 0.06 m | Direct entry threshold |
| `apcr1nd_release_inner_m` | 0.03 m | Release hysteresis |
| `apcr1nd_hold_outside_band` | True | Hold outside desired band |
| `apcr1nd_converging_release_steps` | 15 | Steps before converging release |

## H. Summary

The JAX APCR1ND override function (`k2_jax_apcr1nd_wheel_damping_override`) has correct band-based scaling and min-clamp logic (categories B, C above), verified to match Python line-for-line (lines 6565-6631).

**The gap is the activation gate (category A + D):** Python requires `apcr1n_recenter_priority_active = True` (gated by startup guard, safety, drift direction, and tuned variant hysteresis) before the override can apply. JAX applies the override whenever `wheel_scale < 1.0` regardless of these gates.

This causes **over-application**: JAX reduces wheel damping when position error exceeds 0.05m, even when Python's gating holds the override at bay (e.g., during startup guard or when drift is converging).
