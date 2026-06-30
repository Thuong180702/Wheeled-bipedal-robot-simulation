# K2 JAX Dynamic Height / Push Parity Baseline

**Date:** 2026-06-28
**Branch:** repo-cleanup-t6j

## Phase 0: Baseline Reproduction

### Pre-fix parity results (both-synced mode)

| Scenario | Steps | Max Diff | Divergent Index | Verdict |
|----------|-------|----------|-----------------|---------|
| fixed_high_0p480 | 50 | 9.54e-08 | 8 (r_knee) | PASS |
| fixed_low_0p330 | 50 | 6.54e-02 | 1 (l_hip_yaw) | FAIL |
| ramp_up (dynamic) | 300 | 4.95e-01 | 4 (l_wheel) | FAIL |
| push_fwd_90N | 200 | 4.30e+00 | 4 (l_wheel) | FAIL |

### Height-related diagnostic values (ramp_up, step 0)

| Scalar | Python Value | JAX Value | Match? |
|--------|-------------|-----------|--------|
| schedule_height_source | `filtered_current_fallback` | — | N/A (JAX uses `height_ref`) |
| schedule_height_ref_m | 0.400411 | 0.330000 | **NO** |
| JAX height_ref | — | 0.330 | — |
| notch gate | 0.0 | 0.0 | YES |
| k_position | 40.0 | 40.0 | YES |
| max_position_tau | 4.0 | 4.0 | YES |
| calib_kp | — | 1.30 | — |

### Confirmed findings

1. **fixed_high_0p480**: PASSES (<1e-7) — height scheduling matches (both use `target_reference` = 0.480m)
2. **fixed_low_0p330**: FAILS at hip_yaw [1] — **separate issue** (not height-scheduling-related)
3. **ramp_up**: FAILS at wheel [4] — **confirmed** height scalar mismatch at step 0
4. **push_fwd_90N**: FAILS at wheel [4] — push never fires within 150 steps (interval=200); divergence is accumulated drift

## Root Causes Identified

### Root Cause 1: Height reference sentinel (simulation script)

**File:** `scripts/simulate_hierarchical_controller.py`, line 6553

When `--dynamic-height-trajectory` is used WITHOUT `--height-variant-setup`:
- Python K2 receives `commanded_height_ref_m=None` → falls back to `filtered_com_z`
- JAX receives `commanded_height_ref_m=float(height_cmd)` → uses dynamic target height
- These differ (e.g., 0.40 vs 0.33 at step 0) → different height-scheduled gains

**Fix:** Pass -1.0 sentinel to JAX when Python would receive None:
```python
commanded_height_ref_m=float(height_variant_setup["target_com_z_m"]) if (height_variant_setup and "target_com_z_m" in height_variant_setup) else -1.0,
```
JAX `schedule_h` falls back to `0.9*filtered_com_z + 0.1*com_z` when `height_ref <= 0`.

### Root Cause 2: Notch gate uses height_ref instead of schedule_h (JAX controller)

**File:** `wheeled_biped/controllers/k2_jax_controller.py`, line 1286

JAX notch gate used raw `height_ref` instead of computed `schedule_h`. When height_ref is the -1.0 sentinel, notch gate would be 0 regardless of actual height.

**Fix:** Moved `schedule_h` computation before notch gate; notch gate uses `schedule_h`:
```python
schedule_h = jnp.where(height_ref > 0.0, height_ref, 0.9 * filtered_com_z + 0.1 * com_z)
notch_gate = smoothstep_gate_jax(schedule_h, 0.42, 0.48)
```

### Root Cause 3: velocity_damping_scale mismatch

**File:** `scripts/simulate_hierarchical_controller.py`, line 5314

K2_NOTCH_LOW_Q_V1 has `applies_to_variants` containing specific height variants. When a recognized variant is active (`schedule_active=True`), Python uses `velocity_damping_scale=1.10`. When no variant is active (`schedule_active=False`), Python uses `1.0`.

JAX always used the constructor value (1.10) without checking `schedule_active`.

**Fix:** Compute effective `velocity_damping_scale` at params init time:
```python
_eff_velocity_damping_scale = 1.0
if height_variant_setup is not None:
    _vname = height_variant_setup.get("variant_name")
    if _vname and sagittal_controller.authority_schedule.is_active_for_variant(_vname):
        _eff_velocity_damping_scale = float(authority_schedule.velocity_damping_scale)
```

### Root Cause 4: APCR1ND wheel damping override (PARTIALLY FIXED)

**File:** `wheeled_biped/controllers/k2_jax_controller.py` (new function `k2_jax_apcr1nd_wheel_damping_override`)

K2 applies band-based wheel damping scaling with minimum clamp:
- Emergency band (>0.12m): scale=0.1
- Hard band (>0.10m): scale=0.15
- Desired band (>0.08m): scale=0.3
- Soft band (>0.05m): scale=0.5
- Normal (<0.05m): scale=1.0
- Minimum damping: 0.5 Nm
- Preserve damping if it opposes drift

JAX did not implement any of this, causing up to 35x wheel velocity damping differences when position error exceeds 0.05m.

**Fix:** Added `k2_jax_apcr1nd_wheel_damping_override()` function implementing band-based scaling and min-clamp. This shifts the first divergent step from 129 to 158 but does NOT achieve full parity due to complex gating conditions in Python's APCR1ND logic that are impractical to fully replicate.

## Post-Fix Parity

| Scenario | Pre-fix Diff | Post-fix Diff | Change |
|----------|-------------|---------------|--------|
| fixed_high_0p480 | 9.54e-08 | 9.54e-08 | NO CHANGE (already passing) |
| ramp_up | 4.95e-01 | 5.74e-01 | Similar magnitude, shifted from step 129 → 158 |
| push_fwd_90N | 4.30e+00 | Not re-tested | — |

## Remaining Gaps

1. **APCR1ND full gating logic**: Python's APCR1ND has complex conditions (`support_centering_trim_enabled`, `apcr1n_recenter_active`, `apcr1n_wheel_damping_fights_drift` with drift sign detection via sliding window) that are not fully implemented in JAX. The band-based scaling and min-clamp are implemented, but the activation gate doesn't match Python in all cases.

2. **fixed_low_0p330 hip_yaw divergence**: Separate root cause — not height-scheduling-related. Likely involves mode-div or support-FF differences at low heights.

## Files Changed

1. `wheeled_biped/controllers/k2_jax_controller.py`:
   - `schedule_h` computation moved before notch gate (line ~1282)
   - Notch gate uses `schedule_h` instead of `height_ref` (line ~1295)
   - Added `k2_jax_apcr1nd_wheel_damping_override()` function (lines 596-690)
   - APCR1ND override called after sagittal assembly (line ~1527)

2. `scripts/simulate_hierarchical_controller.py`:
   - JAX height_ref sentinel: -1.0 when no variant setup (line 6553)
   - Effective velocity_damping_scale computed from schedule_active (lines 5303-5310)
