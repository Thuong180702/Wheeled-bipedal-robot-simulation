# T6 High-Height Transient Suppression Design

**Date:** 2026-06-12  
**Purpose:** Design opt-in T6 variants to reduce high_0p480 drift during transient window (steps 500-3500)  
**Based on:** T5 high_0p480 band failure audit

---

## Design Constraints

**MUST PRESERVE:**
- T5 unchanged
- All APCR1nD baselines unchanged
- No WBC path changes
- No HY2-DIV default changes
- Opt-in only (explicitly named profiles)
- Hard safety gates (contact, height, roll, pitch thresholds)
- Tuned telemetry fields

**ALLOWED:**
- Create new T6 variants based on T5
- Adjust band thresholds, position caps, damping scales
- Add T6-specific telemetry fields
- Height-aware or transient-aware logic

---

## T5 Baseline Configuration (Reference)

```yaml
tuned_variant_name: "T5"
tuned_band_control_enable: true
tuned_band_thresholds:
  soft_band_m: 0.05
  desired_band_m: 0.08
  hard_band_m: 0.10
  emergency_band_m: 0.12
  release_inner_m: 0.03
tuned_position_caps:
  normal_cap: 4.0
  soft_cap: 4.5
  desired_cap: 5.5
  hard_cap: 6.5
  emergency_cap: 7.0
tuned_damping_scales:
  normal_scale: 1.0
  soft_scale: 0.50
  desired_scale: 0.30
  hard_scale: 0.15
  emergency_scale: 0.10
tuned_hold_outside_band: true
tuned_strict_release: true
```

---

## T5 Failure Analysis Summary

**Root causes at high_0p480:**
1. **EMERGENCY_TOO_LATE** — Emergency threshold (0.12 m) crossed at step 94, but emergency band not entered until step 1040 (946 steps late)
2. **AUTHORITY_TOO_WEAK** — Emergency cap 7.0 Nm insufficient for extreme height gravitational coupling
3. **DAMPING_TOO_STRONG** — Emergency damping 0.10 may still fight corrective torque at extreme height

**Problem window:** Steps 500-3500 (windows 2-7)  
**Target improvement:** Reduce "outside ±0.08 m" from 68.5% to ≤30%

---

## T6A: High_Early_Hard_Band

**Strategy:** Enter hard/emergency bands sooner to engage higher authority before drift escalates.

**Changes from T5:**
```yaml
tuned_variant_name: "T6A_high_early_hard_band"
tuned_band_thresholds:
  soft_band_m: 0.05          # unchanged
  desired_band_m: 0.07       # tighter (was 0.08)
  hard_band_m: 0.085         # tighter (was 0.10)
  emergency_band_m: 0.105    # tighter (was 0.12)
  release_inner_m: 0.03      # unchanged
tuned_position_caps:         # all unchanged
  normal_cap: 4.0
  soft_cap: 4.5
  desired_cap: 5.5
  hard_cap: 6.5
  emergency_cap: 7.0
tuned_damping_scales:        # all unchanged
  normal_scale: 1.0
  soft_scale: 0.50
  desired_scale: 0.30
  hard_scale: 0.15
  emergency_scale: 0.10
```

**Rationale:**
- Emergency at 0.105 m means emergency band enters ~850 steps earlier
- Desired/hard bands also tighter, providing graduated response sooner
- Authority levels unchanged — only timing changes
- Conservative: tests whether EMERGENCY_TOO_LATE alone explains failure

**Risk:** May enter high bands too often, causing unnecessary authority escalation

---

## T6B: High_Stronger_Emergency

**Strategy:** Keep thresholds same but increase authority in high bands.

**Changes from T5:**
```yaml
tuned_variant_name: "T6B_high_stronger_emergency"
tuned_band_thresholds:      # all unchanged
  soft_band_m: 0.05
  desired_band_m: 0.08
  hard_band_m: 0.10
  emergency_band_m: 0.12
  release_inner_m: 0.03
tuned_position_caps:
  normal_cap: 4.0            # unchanged
  soft_cap: 4.5              # unchanged
  desired_cap: 5.8           # stronger (was 5.5)
  hard_cap: 7.0              # stronger (was 6.5)
  emergency_cap: 8.0         # stronger (was 7.0)
tuned_damping_scales:
  normal_scale: 1.0          # unchanged
  soft_scale: 0.50           # unchanged
  desired_scale: 0.30        # unchanged
  hard_scale: 0.10           # more aggressive (was 0.15)
  emergency_scale: 0.05      # more aggressive (was 0.10)
```

**Rationale:**
- Addresses AUTHORITY_TOO_WEAK by increasing emergency cap to 8.0 Nm
- Also strengthens hard/desired bands for graduated response
- Reduces emergency damping to 0.05 to allow more aggressive wheel acceleration
- Timing unchanged — only strength changes

**Risk:** 8.0 Nm may be too aggressive, could cause oscillation or instability

---

## T6C: High_Early_Plus_Stronger

**Strategy:** Combine T6A (earlier entry) + T6B (stronger authority).

**Changes from T5:**
```yaml
tuned_variant_name: "T6C_high_early_plus_stronger"
tuned_band_thresholds:
  soft_band_m: 0.05          # unchanged
  desired_band_m: 0.07       # tighter (T6A)
  hard_band_m: 0.085         # tighter (T6A)
  emergency_band_m: 0.105    # tighter (T6A)
  release_inner_m: 0.03      # unchanged
tuned_position_caps:
  normal_cap: 4.0            # unchanged
  soft_cap: 4.5              # unchanged
  desired_cap: 5.8           # stronger (T6B)
  hard_cap: 7.0              # stronger (T6B)
  emergency_cap: 8.0         # stronger (T6B)
tuned_damping_scales:
  normal_scale: 1.0          # unchanged
  soft_scale: 0.50           # unchanged
  desired_scale: 0.25        # slightly more aggressive (was 0.30)
  hard_scale: 0.10           # more aggressive (T6B)
  emergency_scale: 0.05      # more aggressive (T6B)
```

**Rationale:**
- Maximum transient suppression: earlier entry + stronger authority
- Addresses both EMERGENCY_TOO_LATE and AUTHORITY_TOO_WEAK simultaneously
- Most aggressive non-conditional variant

**Risk:** May be too aggressive overall, could cause overshoot or oscillation

---

## T6D: High_Transient_Boost

**Strategy:** Apply T6C logic only during high-height transient window, fallback to T5 after recovery.

**Activation conditions:**
```python
# T6D boost active when ALL of:
# 1. Height >= 0.45 m (high height)
# 2. Steps 500-3500 OR (not yet recovered for 500 consecutive steps)
# 3. Hard safety gates still passed

# Recovery defined as:
# - abs_error < 0.08 m for 500 consecutive steps
# - OR step > 3500 and abs_error < 0.10 m

# After recovery: fallback to T5 thresholds/caps/scales
```

**Boost configuration (during activation):**
```yaml
# Same as T6C during boost
tuned_band_thresholds:
  desired_band_m: 0.07
  hard_band_m: 0.085
  emergency_band_m: 0.105
tuned_position_caps:
  desired_cap: 5.8
  hard_cap: 7.0
  emergency_cap: 8.0
tuned_damping_scales:
  desired_scale: 0.25
  hard_scale: 0.10
  emergency_scale: 0.05

# T5 baseline after recovery
```

**Additional telemetry:**
- `t6_transient_boost_active` (bool)
- `t6_recovered_flag` (bool)
- `t6_recovery_counter` (int, counts consecutive steps inside ±0.08 m)

**Rationale:**
- T6C authority only when needed (high height + transient window)
- Avoids permanent over-aggressiveness
- Preserves T5 behavior at low height and after recovery

**Risk:** Hysteresis logic may cause chatter if drift oscillates around recovery threshold

---

## T6E: High_Pitch_Aware_Boost

**Strategy:** Boost authority when pitch coupling is strong (pitch magnitude or pitch rate elevated).

**Activation conditions:**
```python
# T6E pitch-aware boost active when ALL of:
# 1. Height >= 0.45 m (high height)
# 2. abs_error > 0.08 m (outside desired band)
# 3. (abs_pitch > 0.07 rad (4 deg) OR abs_pitch_rate > 0.10 rad/s)
# 4. Hard safety gates still passed

# Boost applied ONLY when pitch coupling detected
```

**Boost configuration:**
```yaml
# Add to current band's position cap:
position_cap_boost: +1.0 Nm

# Reduce current band's damping scale:
damping_scale_reduction: -0.05 (min 0.05)

# Example: if in desired band (5.5 Nm, 0.30 scale):
# Boosted: 6.5 Nm, 0.25 scale
```

**Additional telemetry:**
- `t6_pitch_aware_boost_active` (bool)
- `t6_pitch_magnitude_trigger` (bool)
- `t6_pitch_rate_trigger` (bool)

**Rationale:**
- Addresses PITCH_COUPLING_DOMINATES directly
- Only boosts when pitch is elevated, not always
- Preserves T5 behavior when pitch is nominal

**Risk:** Pitch-based logic may not react fast enough if pitch spikes suddenly

---

## Variant Comparison Table

| Variant | Thresholds | Caps | Damping | Conditional | Target Root Cause |
|---------|------------|------|---------|-------------|-------------------|
| T5      | Baseline   | Baseline | Baseline | No | (reference) |
| T6A     | Earlier    | Same | Same | No | EMERGENCY_TOO_LATE |
| T6B     | Same       | Stronger | More aggressive | No | AUTHORITY_TOO_WEAK, DAMPING_TOO_STRONG |
| T6C     | Earlier    | Stronger | More aggressive | No | All root causes |
| T6D     | Earlier (transient only) | Stronger (transient only) | More aggressive (transient only) | Yes (height + window) | All root causes, transient-specific |
| T6E     | Same       | +1.0 boost | -0.05 reduction | Yes (height + pitch) | PITCH_COUPLING_DOMINATES |

---

## Selection Criteria for Phase 6 Analysis

**Primary metric:** Outside ±0.08 m % during 2000-step run

**Secondary metrics:**
- Outside ±0.10 m %
- Outside ±0.15 m %
- Max |e|
- Mean |e|
- Survival rate
- Contact/height/roll stability
- Wheel velocity max/RMS
- Pitch RMS

**Ranking logic:**
1. Must survive 2000 steps
2. Lowest "outside ±0.08 m %"
3. If tied, lowest "outside ±0.10 m %"
4. If tied, lowest max |e|
5. Stability must be preserved (no WBC violations, wheel vel < 7 rad/s, pitch < 10 deg)

---

## Implementation Notes

**Code structure:**
```python
# In sagittal_velocity_damped_balance_controller.py

TUNED_VARIANT_CONFIGS = {
    "T5": {...},  # existing
    "T6A_high_early_hard_band": {...},
    "T6B_high_stronger_emergency": {...},
    "T6C_high_early_plus_stronger": {...},
    "T6D_high_transient_boost": {...},
    "T6E_high_pitch_aware_boost": {...},
}

# For T6D: add transient boost state tracking
# For T6E: add pitch-aware boost logic
```

**T6-specific telemetry fields:**
- `t6_high_mode_active` (bool) — true if height >= 0.45 m
- `t6_transient_boost_active` (bool) — true if T6D boost active
- `t6_pitch_aware_boost_active` (bool) — true if T6E boost active
- `t6_recovered_flag` (bool) — true if T6D recovery condition met
- `t6_recovery_counter` (int) — consecutive steps inside ±0.08 m

---

## Next Steps (Phase 3)

1. Implement all 5 T6 variants in sagittal controller
2. Verify T5 unchanged via diff
3. Add T6 tests (existence, thresholds, caps, scales, conditional logic)
4. Run Phase 5 screening (2000 steps high_0p480 for each T6 variant)

---

**Status:** Design complete, ready for implementation  
**Date:** 2026-06-12  
**Phase:** 2 (T6 Variant Design) COMPLETE
