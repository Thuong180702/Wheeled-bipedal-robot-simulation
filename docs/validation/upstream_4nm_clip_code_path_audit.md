# Upstream 4.0 Nm Clip Code Path Audit

**Date:** 2026-06-12  
**Classification:** UPSTREAM_CLIP_FROM_MAX_POSITION_TAU_NOMINAL  
**Status:** Root cause identified

---

## Executive Summary

**FOUND: The 4.0 Nm upstream clip is `max_position_tau_nominal`.**

The clip occurs at **line 2009** of `sagittal_velocity_damped_balance_controller.py`, where `tau_position` is clipped to `±effective_max_position_tau` BEFORE the APCR1n tuned cap boost layer (line 2353).

Both T5 and T6B have `max_position_tau_nominal=4.0`, which becomes `effective_max_position_tau=4.0` for high_0p480 height. This upstream clip prevents the tuned emergency cap boost (7.0 vs 8.0 Nm) from ever receiving signals > 4.0 Nm.

---

## Code Path Map

### Pipeline Overview

```
tau_position_raw (computed from k_position × error)
    ↓
tau_position_before_clip (after capture gate, pitch-aware scaling)
    ↓
[LINE 2009] UPSTREAM CLIP: tau_position = clip(tau_position_before_clip, ±effective_max_position_tau)
    ↓  [4.0 Nm max for both T5 and T6B at high_0p480]
    ↓
[LINE 2353] APCR1n TUNED CAP BOOST: tau_position = clip(tau_position, ±boosted_cap)
    ↓  [T5: 7.0 Nm, T6B: 8.0 Nm - but input already clipped to 4.0]
    ↓
final wheel torque composition
```

---

## Critical Code Locations

### 1. Upstream Clip (Line 2009)

**File:** `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`  
**Line:** 2009

```python
else:
    # Legacy fixed-cap clipping
    tau_position = float(jnp.clip(tau_position_before_clip, -effective_max_position_tau, effective_max_position_tau))
    tau_position_total_bound_clipped = False
    tau_position_saturated = abs(tau_position_before_clip) >= effective_max_position_tau * 0.99
    tau_position_saturation_reason = "fixed_cap" if tau_position_saturated else "none"
    position_authority_reason = tau_position_saturation_reason
```

**What this does:**
- Clips `tau_position_before_clip` to `±effective_max_position_tau`
- This is the "legacy fixed-cap clipping" path
- Runs when `enable_torque_budget_aware_position=False`

### 2. Effective Max Position Tau Computation (Lines 1684-1696)

**File:** Same  
**Lines:** 1684-1696

```python
# max_position_tau scheduling (Phase 6 joint fix)
if self.authority_schedule.continuous_max_position_tau:
    effective_max_position_tau = scheduled_k_position(
        z_ref=schedule_height_ref,
        k_nominal=self.authority_schedule.max_position_tau_nominal,
        k_low_max=self.authority_schedule.max_position_tau_low_max,
        z_low=self.authority_schedule.k_position_z_low,
        z_high=self.authority_schedule.k_position_z_high,
    )
else:
    effective_max_position_tau = self.authority_schedule.max_position_tau_for_variant(
        height_variant_name,
        self.max_position_tau,
    )
```

**What this does:**
- Computes `effective_max_position_tau` based on height scheduling
- For high_0p480 (0.48m), height is above `z_high` (0.40m typically)
- So `smoothstep_value → 0`, which means `effective_max_position_tau → max_position_tau_nominal`

### 3. T5 Profile Configuration (Lines 1202-1230)

**File:** Same  
**Lines:** 1202-1230 (abbreviated)

```python
APCR1ND_T5_BAND_LIMITED_BALANCED = SagittalAuthoritySchedule(
    profile_name="APCR1nD_T5_band_limited_balanced",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,  # ← UPSTREAM CLIP VALUE
    max_position_tau_low_max=8.0,
    # ... tuned caps ...
    apcr1nd_position_cap_desired_nm=5.5,
    apcr1nd_position_cap_hard_nm=6.5,
    apcr1nd_position_cap_emergency_nm=7.0,  # ← TUNED CAP (never reached)
    # ...
)
```

### 4. T6B Profile Configuration (Lines 1287-1315)

**File:** Same  
**Lines:** 1287-1315 (abbreviated)

```python
T6B_HIGH_STRONGER_EMERGENCY = SagittalAuthoritySchedule(
    profile_name="T6B_high_stronger_emergency",
    applies_to_variants=("low_0p300", "low_0p330", "low_0p360", "extreme_height"),
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,  # ← UPSTREAM CLIP VALUE (SAME AS T5!)
    max_position_tau_low_max=8.0,
    # ... tuned caps ...
    apcr1nd_position_cap_desired_nm=5.8,
    apcr1nd_position_cap_hard_nm=7.0,
    apcr1nd_position_cap_emergency_nm=8.0,  # ← TUNED CAP (never reached)
    # ...
)
```

### 5. APCR1n Tuned Cap Boost (Lines 2323-2364)

**File:** Same  
**Lines:** 2323-2364 (abbreviated)

```python
# Position cap boost
if (self.authority_schedule.position_cap_recenter_boost_enabled and
    apcr1n_safety_gate_pass):
    # Determine boosted cap
    if self.authority_schedule.apcr1nd_tuned_enabled:
        # Tuned variant: band-aware position cap scaling
        abs_error = abs(signed_error)
        # ... band determination ...
        if abs_error >= emergency_band_m:
            boosted_cap = self.authority_schedule.apcr1nd_position_cap_emergency_nm
        # ...
    else:
        # Original APCR1n: use configured boosted cap
        boosted_cap = self.authority_schedule.position_cap_recenter_nm

    tau_position_before_boost = tau_position  # ← Already clipped to 4.0 Nm!

    # Apply boosted cap to tau_position
    tau_position = float(jnp.clip(tau_position, -boosted_cap, boosted_cap))
```

**What this does:**
- Applies the tuned cap (T5: 7.0, T6B: 8.0) to `tau_position`
- But `tau_position` already clipped to 4.0 Nm at line 2009
- So this second clip has no effect when the tuned cap > 4.0 Nm

---

## Height Scheduling Behavior

**Function:** `scheduled_k_position()` (lines 36-65)

**Behavior at high_0p480 (0.48m):**

Assume:
- `z_low = 0.30`
- `z_high = 0.40`
- `z_ref = 0.48` (high_0p480)
- `k_nominal = 4.0` (max_position_tau_nominal)
- `k_low_max = 8.0` (max_position_tau_low_max)

Computation:
```
u = (z_high - z_ref) / (z_high - z_low)
  = (0.40 - 0.48) / (0.40 - 0.30)
  = -0.08 / 0.10
  = -0.8

u_clamped = max(0.0, min(1.0, u)) = max(0.0, min(1.0, -0.8)) = 0.0

smoothstep(0.0) = 0.0

k_position = k_nominal + (k_low_max - k_nominal) * smoothstep
           = 4.0 + (8.0 - 4.0) * 0.0
           = 4.0
```

**Result:** At high_0p480, `effective_max_position_tau = 4.0 Nm` for both T5 and T6B.

---

## Why T6B Had No Effect

**T5 Authority Path:**
1. `tau_position_before_clip` reaches 7.485 Nm (from telemetry)
2. **Line 2009**: Clip to `±4.0` → `tau_position = 4.0 Nm`
3. **Line 2353**: Clip to `±7.0` → `tau_position = 4.0 Nm` (no change, already below)

**T6B Authority Path:**
1. `tau_position_before_clip` reaches 7.485 Nm (same raw torque as T5)
2. **Line 2009**: Clip to `±4.0` → `tau_position = 4.0 Nm` ← **SAME CLIP**
3. **Line 2353**: Clip to `±8.0` → `tau_position = 4.0 Nm` (no change, already below)

**Conclusion:** T6B's emergency cap boost (7.0 → 8.0) operates on a signal that's already been clipped to 4.0 Nm. Raising the tuned cap from 7.0 to 8.0 has zero effect because the input never exceeds 4.0 Nm.

---

## Is This Intentional?

**Evidence it's intentional:**
- `max_position_tau_nominal` is explicitly configured to 4.0 for both profiles
- The scheduling function is designed to reduce authority at HIGH heights
- Comment says "Phase 6 joint fix" suggesting deliberate design

**Evidence it may be accidental:**
- The tuned cap layer (T5: 7.0, T6B: 8.0) suggests the designer expected higher torques
- Why configure a 7.0/8.0 Nm emergency cap if the upstream cap is 4.0?
- `max_position_tau_low_max=8.0` suggests 8.0 Nm is considered safe at LOW heights

**Interpretation:**
The height scheduling is intentional (reduce position authority at high heights for safety), but the interaction with the tuned cap layer may be unintended. The designer may have expected:
- Low heights (0.30-0.40m): `max_position_tau` schedules up to 8.0 Nm, tuned cap further refines
- High heights (0.48m): `max_position_tau` still allows baseline authority, tuned cap provides emergency boost

Instead, what happens:
- Low heights: `max_position_tau` schedules up to 8.0 Nm ✓
- High heights: `max_position_tau` clamps to 4.0 Nm, blocking tuned cap ✗

---

## Recommendation

**Classification:** `UPSTREAM_CLIP_FROM_MAX_POSITION_TAU_NOMINAL`

**Next Steps:**
1. **Phase 2**: Add diagnostic telemetry to confirm this in runtime data
2. **Phase 3**: Run short paired T5/T6B diagnostic with telemetry
3. **Phase 4**: Determine if 4.0 Nm nominal is safety-critical or can be raised for high_0p480
4. **Phase 5**: Design architecture fixes:
   - Option A: Raise `max_position_tau_nominal` to 6.0 or 8.0 for extreme_height
   - Option B: Make tuned cap bypass the upstream clip during emergency
   - Option C: Height-schedule the upstream cap differently for high vs low heights

**Do NOT implement fixes until Phases 2-4 confirm this hypothesis with data.**

---

## Files Referenced

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
  - Line 2009: Upstream clip
  - Lines 1684-1696: Effective max position tau computation
  - Lines 1202-1230: T5 profile
  - Lines 1287-1315: T6B profile
  - Lines 2323-2364: APCR1n tuned cap boost

---

**Status:** Phase 1 complete  
**Date:** 2026-06-12
