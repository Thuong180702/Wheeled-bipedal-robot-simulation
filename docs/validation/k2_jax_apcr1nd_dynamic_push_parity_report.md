# K2 JAX APCR1ND Dynamic/Push Parity Report — Phase 6

**Date:** 2026-06-28
**Branch:** repo-cleanup-t6j

## Both-Synced Parity Results (Post-Fix)

| # | Scenario | Steps | Max Abs Diff | Divergent Step | Divergent Actuator | Wheel[4,9] | HY[1,6] | Verdict |
|---|----------|-------|-------------|----------------|--------------------|------------|---------|---------|
| 1 | fixed_high_0p480 | 50 | 9.54e-08 | 2 | 8 (r_knee) | <1e-7 | <1e-7 | **PASS** |
| 2 | fixed_low_0p330 | 50 | 9.54e-08 | 2 | 8 (r_knee) | <1e-7 | <1e-7 | **PASS** |
| 3 | ramp_up | 500 | 9.54e-08 | 86 | 8 (r_knee) | <1e-7 | <1e-7 | **PASS** ✅ |
| 4 | ramp_down | 500 | 9.54e-08 | 2 | 8 (r_knee) | <1e-7 | <1e-7 | **PASS** |
| 5 | gate_chatter | 500 | 9.54e-08 | 86 | 8 (r_knee) | <1e-7 | <1e-7 | **PASS** ✅ |
| 6 | push_fwd_90N | 300 | 3.00e+00 | 205 | 4 (l_wheel) | 3.0 | <1e-7 | **FAIL** ⚠️ |
| 7 | push_bwd_90N | 300 | 3.30e+00 | 170 | 9 (r_wheel) | 3.0 | <1e-7 | **FAIL** ⚠️ |

### Pre-Fix vs Post-Fix Comparison

| Scenario | Pre-Fix | Post-Fix | Change |
|----------|---------|----------|--------|
| fixed_high_0p480 | 9.54e-08 | 9.54e-08 | Unchanged (already passed) |
| fixed_low_0p330 | 9.54e-08 | 9.54e-08 | Unchanged (already passed) |
| ramp_up | **7.88e-01** | **9.54e-08** | ✅ FIXED |
| ramp_down | 9.54e-08 | 9.54e-08 | Unchanged (already passed) |
| gate_chatter | **7.92e-01** | **9.54e-08** | ✅ FIXED |
| push_fwd_90N | 3.00e+00 | 3.00e+00 | Unchanged (separate issue) |
| push_bwd_90N | 3.30e+00 | 3.30e+00 | Unchanged (separate issue) |

## Root Cause of Dynamic Height Parity Fix

The JAX APCR1ND gate function had the **priority order inverted**: JAX checked `activate` before `release`, but Python checks `release` before `activate` (line 6437 of the sagittal controller).

In Python's if/elif chain:
```python
if release_by_inner_band or release_by_converging:
    active = False  # RELEASE TAKES PRIORITY
elif emergency_entry or hold_outside_band_condition:
    active = True
elif direct_entry or soft_entry:
    active = True
elif hold_condition:
    active = True  # Only reached if NOT releasing
```

The `hold_condition` is `prev_active and abs_error > release_inner_m`. When the position error is slowly converging (decreasing), BOTH `release_by_converging` (enough consecutive converging steps) AND `hold_condition` (still above inner release band) can be True simultaneously. Python correctly prioritizes release. The JAX `jnp.where(activate, 1.0, jnp.where(release, 0.0, ...))` incorrectly prioritized activate.

**Fix:** Reversed the `jnp.where` nesting to check `release` first:
```python
new_recenter_held = jnp.where(
    release, 0.0,           # RELEASE takes priority
    jnp.where(activate, 1.0, recenter_held),
)
```

## Push Parity (Separate Issue)

Push parity remains FAILING at 3.0 Nm max difference. This was failing BEFORE the APCR1ND changes and is NOT affected by them. The 3.0 Nm value corresponds to `max_tau_wheel=5.0` clipped by the composer, suggesting a torque composer clipping difference during large push-induced position errors. This is a separate root cause outside the scope of the APCR1ND wheel damping override fix.

## Classification

**K2_JAX_APCR1ND_DYNAMIC_PARITY_PASS_PUSH_BLOCKED**

- Dynamic height parity: **PASS** (ramp_up, ramp_down, gate_chatter all <1e-5) ✅
- Fixed-height parity: **PASS** (unchanged at 9.54e-08) ✅
- Push parity: **FAIL** (pre-existing, separate root cause) ⚠️
- Hip-yaw [1,6]: **EXACT** (<1e-7 across all scenarios) ✅
- Tests: **131/131 PASS** ✅
