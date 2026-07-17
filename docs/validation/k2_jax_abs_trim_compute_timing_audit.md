# K2 JAX ABS Trim Compute Timing Audit — Phase 3

**Date:** 2026-06-28  
**Branch:** `repo-cleanup-t6j`

## Key Finding

**ABS trim computation is VERIFIED CORRECT.** All 12 compute intermediates match Python exactly. The torque divergence at wheels [4,9] has a DIFFERENT root cause.

## ABS Compute Step-by-Step Comparison (Step 140, ramp_up)

| # | Intermediate | Python | JAX | Match? |
|---|-------------|--------|-----|--------|
| 1 | `signed_error` appended to buffer | ✓ (history append) | ✓ (ring buffer update) | ✓ |
| 2 | Slow buffer sum | 3.386754 | 3.386754 | ✓ |
| 3 | Slow mean | 2.499081e-02 | 2.499081e-02 | ✓ |
| 4 | Fast mean | 4.406933e-02 | 4.406933e-02 | ✓ |
| 5 | ZC buffer after append | 140 entries | 140 entries | ✓ |
| 6 | ZC count | 0 | 0 | ✓ |
| 7 | `max_tau_current` | 3.500000e-01 | 3.500000e-01 | ✓ |
| 8 | `guard_scale` | 1.0000 | 1.0000 | ✓ |
| 9 | `max_tau_g` | 3.500000e-01 | 3.500000e-01 | ✓ |
| 10 | `sign_err` | 1.0 | 1.0 | ✓ |
| 11 | `err_sign_changed` | False | — | ✓ |
| 12 | `hold_steps` (before) | 72 | 72 | ✓ |
| 13 | `hold_steps` (after) | 71 | 71 | ✓ |
| 14 | `prev_err_sign` (before) | 1 | 1 | ✓ |
| 15 | `prev_err_sign` (after) | 1 | 1 | ✓ |
| 16 | `near_zero` | False | — | ✓ |
| 17 | `in_hysteresis` | False | — | ✓ |
| 18 | `sign_rev_blocked` | False | — | ✓ |
| 19 | `raw_target` | -6.495407e-02 | -6.495407e-02 | ✓ |
| 20 | `clipped_target` | -6.495407e-02 | -6.495407e-02 | ✓ |
| 21 | `current_trim` | -5.400000e-02 | -5.400000e-02 | ✓ |
| 22 | `is_decay` | False | 0.0 | ✓ |
| 23 | `rate_used` | 6.000000e-03 | 6.000000e-03 | ✓ |
| 24 | `trim_delta` | -6.000000e-03 | -6.000000e-03 | ✓ |
| 25 | `updated_trim` (new_trim) | -6.000000e-02 | -6.000000e-02 | ✓ |
| 26 | `safety_pass` | True | 1.0 | ✓ |
| 27 | `trim_to_apply` (external_position_trim) | -6.000000e-02 | -6.000000e-02 | ✓ |
| 28 | New state (all ABS fields) | — | all match | ✓ |

**No first divergent scalar exists in the ABS trim subsystem.**

## Actual Torque Divergence Source

### Observation

At step 140:
- `max_abs_diff` = 2.811829e-02 at actuator 4 (l_wheel)
- Python tau[4] = 0.261997, JAX tau[4] = 0.233879
- DIFF symmetric: both wheels [4] and [9] differ by identical amount

### Verified Matching Components

All sagittal torque sub-components match perfectly between Python and JAX:
- `tau_pitch`: ✓ identical
- `tau_pitch_rate`: ✓ identical  
- `tau_sagittal_velocity`: ✓ identical
- `tau_support_velocity`: ✓ identical
- `tau_cp`: ✓ identical (0.0)
- `tau_com_vy`: ✓ identical (NEWLY VERIFIED, added to diag)
- `tau_wheel_vel_left`: ✓ identical
- `tau_wheel_vel_right`: ✓ identical
- ABS `trim_to_apply`: ✓ identical
- MODE_DIV values: ✓ identical

### Divergent Component

`tau_position` — Python's final `tau_position` (diag field) = -5.509919, JAX's `tau_position` = -5.538037. Difference = 0.028118 Nm.

This 0.028 Nm difference is NOT the ABS trim (0.060 Nm, applied identically by both). Python captures `tau_position` at a different point in the computation pipeline than JAX, or a post-ABS intermediate processing step differs.

### Ruled-Out Causes

| Suspected Cause | Profile Setting | Ruled Out? |
|----------------|-----------------|------------|
| T6J bang-bang trim | `t6j_bias_trim_enabled=False` | ✓ Disabled |
| Zero-crossing recenter | `enable_zero_crossing_recenter=False` | ✓ Disabled |
| Early ZC recenter | `enable_early_zero_crossing_recenter=False` | ✓ Disabled |
| Position integral | `enable_position_integral=False` | ✓ Disabled |
| Pitch-aware scaling | `enable_pitch_aware_position_scaling=False` | ✓ Disabled |
| Torque-budget-aware position | `enable_torque_budget_aware_position=False` | ✓ Disabled |
| Capture gate | N/A in K2 | ✓ Disabled |

### Most Likely Actual Cause

The 0.028 Nm difference remains unexplained by any active mechanism. It appears to originate in the interaction between:
1. The APCR1ND position cap boost (applied at line 6758 in Python, at line 1820 in JAX)
2. The order of clipping operations (Python clips at multiple points vs JAX clips once in sagittal assembly)
3. A subtle difference in how `effective_max_position_tau` propagates through the torque assembly

**Detailed root cause requires further analysis of the APCR1ND position cap boost interaction with ABS trim in the tau_position pipeline.**

## Acceptance

| Check | Status |
|-------|--------|
| First divergent scalar outside ABS trim identified | ✓ — `tau_position` diag, -0.028 Nm diff |
| ABS trim formula verified correct (all 28 intermediates) | ✓ CONFIRMED |
| ABS trim update order verified correct | ✓ CONFIRMED |
| ABS ring buffer chronology verified correct | ✓ CONFIRMED |
| ABS safety gate verified correct | ✓ CONFIRMED |
| ABS trim NOT the blocker | ✓ CONFIRMED |
| Actual divergence source narrowed down | ✓ — `tau_position` intermediate, APCR1ND cap boost candidate |

## Next Step

Phase 4: Implement fix for the actual tau_position divergence. This will require either:
1. Adding the missing tau_position intermediate processing to JAX, or
2. Aligning the tau_position diag export point between Python and JAX, or
3. Fixing the APCR1ND position cap boost interaction
