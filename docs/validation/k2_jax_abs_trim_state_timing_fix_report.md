# K2 JAX ABS Trim State/Timing Fix Report — Phase 4

**Date:** 2026-06-28  
**Branch:** `repo-cleanup-t6j`

## Executive Summary

**Root cause was NOT in ABS trim.** ABS trim state packing, computation, and application are all correct (verified across 28 intermediates). The actual divergence was in **tau_position clipping order** — Python applies two sequential clips (height-scheduled cap then APCR1ND boost cap) while JAX applied a single combined clip. The combined clip was looser when the APCR1ND boost cap exceeded the height-scheduled cap.

## First Divergent Scalar

| Field | Python Value | JAX Value (before fix) | Diff |
|-------|-------------|----------------------|------|
| `tau_position` (diag, step 140) | -5.509919 Nm | -5.538037 Nm | 0.028118 Nm |
| `tau[4]` (l_wheel, step 140) | 0.261997 Nm | 0.233879 Nm | 0.028118 Nm |
| `effective_max_position_tau` | 5.509919 Nm | 7.0 Nm (combined) | — |

## Exact Root Cause

### Python clipping order (correct, source of truth):

```
tau_position = -kpos * sag_pos_err                    # tau_position_p = -5.478
tau_position = tau_position + external_position_trim  # + (-0.060) = -5.538
tau_position = clip(tau_position, -max_pos_tau, +max_pos_tau)  # clip to -5.510  ← TIGHTER
# ... later (APCR1ND boost) ...
tau_position = clip(tau_position, -boosted_cap, +boosted_cap)   # clip to -7.0   ← LOOSER
```

Final Python tau_position = -5.510 (saturated at max_pos_tau).

### JAX clipping order (before fix, wrong):

```
tau_position = -kpos * sag_pos_err                    # tau_position_p = -5.478
tau_position = tau_position + external_position_trim  # + (-0.060) = -5.538
tau_position = clip(tau_position, -max(max_pos_tau, boosted_cap), +max(...))  # clip to -7.0
```

Final JAX tau_position = -5.538 (NOT clipped, because 7.0 > 5.538).

**The single combined clip using `max(max_pos_tau, boosted_cap)` gave a looser bound than Python's two-stage clip where the first stage (height-scheduled cap) is tighter.**

### Why the difference in clipping matters for wheel torque

The wheel torque formula is:
```
tau_wheel = tau_pitch + tau_pitch_rate + tau_sag_vel + tau_support_vel
          + tau_position + tau_cp + tau_com_vy + tau_wheel_vel
```

Since `tau_position` is part of the wheel torque sum, the 0.028 Nm difference in `tau_position` propagates directly to wheel torques [4,9] as a 0.028 Nm difference.

## Exact Fix

### Files changed

**`wheeled_biped/controllers/k2_jax_controller.py`** — lines 1833-1850 (sagittal torque assembly call)

### Change: Two-stage position torque clipping

**Before (wrong):**
```python
effective_max_pos_tau = jnp.maximum(max_pos_tau, _boosted_cap)  # Combined cap
tau_sag, sag_diag = k2_jax_sagittal_torque_assembly(
    ...
    effective_max_position_tau=effective_max_pos_tau,  # Single loose clip
    ...
)
```

**After (correct):**
```python
effective_max_pos_tau = jnp.maximum(max_pos_tau, _boosted_cap)  # Still computed for reference

tau_sag, sag_diag = k2_jax_sagittal_torque_assembly(
    ...
    effective_max_position_tau=max_pos_tau,  # First clip: height-scheduled cap
    ...
)

# Second clip: APCR1ND boost cap (matching Python svdbc.py:6758)
_pos_clip_boosted = jnp.clip(sag_diag["tau_position"], -_boosted_cap, _boosted_cap)
sag_diag["tau_position"] = _pos_clip_boosted
# Recompute wheel torques with re-clipped tau_position
_tau_common_boosted = 1.0 * (
    sag_diag["tau_pitch"] + sag_diag["tau_pitch_rate"]
    + sag_diag["tau_sagittal_velocity"] + sag_diag["tau_support_velocity"]
    + _pos_clip_boosted + sag_diag["tau_cp"] + sag_diag["tau_com_vy"]
)
tau_sag = tau_sag.at[4].set(_tau_common_boosted + sag_diag["tau_wheel_vel_left"])
tau_sag = tau_sag.at[9].set(_tau_common_boosted + sag_diag["tau_wheel_vel_right"])
```

### Why this is a port of Python semantics, not a hack

1. Python's `compute()` at line 5770 clips tau_position to `effective_max_position_tau` (height-scheduled, unboosted) after ABS trim addition.
2. Python's `compute()` at line 6758 re-clips tau_position to `boosted_cap` (APCR1ND position cap boost).
3. The JAX fix replicates this exact two-stage clipping order.
4. The first clip uses `max_pos_tau` (same height-scheduled value as Python's `effective_max_position_tau`).
5. The second clip uses `_boosted_cap` (same APCR1ND boost cap as Python's `boosted_cap`).
6. No empirical correction factors, no gain tuning, no formula changes beyond the clip order.

### Verification

**Before fix:** ramp_up step 140 `max_abs_diff = 2.81e-02 Nm` (FAIL)  
**After fix:** ramp_up step 140 `max_abs_diff = 9.54e-08 Nm` (PASS, floor level)

The 9.54e-08 residual is the WBC initialization artifact (identical to fixed-height scenarios), not a control divergence.

## What was NOT the cause

| Suspected Cause | Investigation Result |
|----------------|---------------------|
| ABS trim state packing | ✓ CORRECT — verified by reading `_jax_state_synced[21]` |
| ABS trim computation | ✓ CORRECT — all 28 intermediates match |
| ABS trim safety gate | ✓ CORRECT — gates match |
| ZC recenter | ✓ DISABLED in K2 profile |
| Position integral | ✓ DISABLED in K2 profile |
| T6J bang-bang trim | ✓ DISABLED in K2 profile |
| Diagnostic bug (reading `_jax_state` instead of `_jax_state_synced`) | Fixed in Phase 0 |
