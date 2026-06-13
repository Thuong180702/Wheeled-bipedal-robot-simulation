# APCR1k_support_hysteresis_early_entry Design

## Purpose

APCR1k lowers the hysteresis outer entry threshold from 0.08 m to 0.05 m to catch drift earlier before it accumulates momentum.

## Problem Statement

APCR1j analysis revealed:
- **Primary cause**: Torque transmission loss (~18% reduction from APCR command to final wheel torque)
- **Secondary cause**: Late entry at 0.08 m allows drift momentum to accumulate

APCR1k addresses the secondary cause by entering RECENTER at 0.05 m instead of 0.08 m.

## Design

APCR1k is based on APCR1j with the following changes:

### Hysteresis Thresholds

| Parameter | APCR1j | APCR1k | Change |
|-----------|--------|--------|--------|
| `apc_outer_enter_m` | 0.08 | **0.05** | Lower |
| `apc_inner_exit_m` | 0.03 | 0.03 | Keep |
| `apc_opposite_release_m` | 0.03 | 0.03 | Keep |

### Torque Authority

| Parameter | APCR1j | APCR1k | Change |
|-----------|--------|--------|--------|
| `apc_max_cross_tau` | 2.0 | 2.0 | Keep |
| `apc_hysteresis_recenter_max_tau` | 2.0 | 2.0 | Keep |
| `apc_hysteresis_emergency_max_tau` | 2.2 | 2.2 | Keep |
| `apc_hysteresis_recenter_rate_per_step` | 1.1 | 1.1 | Keep |
| `apc_hysteresis_emergency_rate_per_step` | 1.3 | 1.3 | Keep |

### Expected Behavior

With APCR1k:
- RECENTER_FROM_POSITIVE starts when e > +0.05 m
- RECENTER_FROM_NEGATIVE starts when e < -0.05 m
- RECENTER holds until e <= +0.03 (inner exit) or e <= -0.03 (opposite release)
- RECENTER holds until e >= -0.03 (inner exit) or e >= +0.03 (opposite release)

### Entry Timing Improvement

| Metric | APCR1j | APCR1k | Improvement |
|--------|--------|--------|-------------|
| First RECENTER step (e > 0.05) | 46 | **46** | Same |
| First RECENTER step (e > 0.08) | 58 | **46** | 12 steps earlier |
| Entry e at RECENTER start | 0.0817 m | **0.0521 m** | 0.03 m earlier |

### Expected Results

Based on APCR1j data:
- RECENTER starts at step 46 (e = 0.0521 m) instead of step 58 (e = 0.0817 m)
- This prevents drift from accumulating to 0.08 m before corrective action begins
- Expected max_e reduction: ~10-15% (from 0.1826 m to ~0.15-0.16 m)

### What APCR1k Does NOT Change

1. **Torque transmission loss**: APCR1k does not fix the ~18% reduction in final wheel torque
2. **Gate behavior**: Safety gates remain unchanged
3. **Exit logic**: Inner exit and opposite release thresholds remain at 0.03 m
4. **Torque magnitude**: APCR still produces 2.0 Nm, final wheel torque still limited to ~1.64 Nm

### Rationale

Lowering entry threshold to 0.05 m:
1. Catches drift earlier before momentum accumulates
2. Matches the user's requirement: "e > +0.05 should start RECENTER_FROM_POSITIVE"
3. Keeps inner exit at 0.03 m for hysteresis
4. Uses same torque authority as APCR1j (no unnecessary changes)

### Profile Name

`APCR1k_support_hysteresis_early_entry`

This name clearly indicates:
- APCR1k: Next iteration after APCR1j
- support_hysteresis: Uses support drift hysteresis control
- early_entry: Lower entry threshold than previous profiles

## Implementation Notes

1. Copy APCR1j profile configuration
2. Change `apc_outer_enter_m` from 0.08 to 0.05
3. Keep all other parameters identical to APCR1j
4. Add telemetry for the new profile
5. Do NOT change APCR1j (preserve for comparison)