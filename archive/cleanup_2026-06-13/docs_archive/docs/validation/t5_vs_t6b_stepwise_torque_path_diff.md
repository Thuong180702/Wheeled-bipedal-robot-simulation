# T5 vs T6B Step-by-Step Torque Path Comparison

**Date:** 2026-06-12  
**Status:** Phase 2 complete  
**Classification:** T6B_CONFIG_DIFFERS_BUT_FINAL_TORQUE_IDENTICAL

---

## Executive Summary

**T6B configuration differs from T5, but produces IDENTICAL torques at every stage.**

**Key Findings:**
- Authority parameters differ as expected (position cap 52.4%, damping scale 17.7%)
- Raw position torque **IDENTICAL** between T5 and T6B
- Capped position torque **IDENTICAL** (after cap application)
- Final wheel torques **IDENTICAL**
- Dynamics **IDENTICAL** (wheel velocity, CoM height)

**Critical Interpretation:** T6B's higher caps (8.0 Nm vs 7.0 Nm) were correctly applied, but the raw position torque never reached T5's 7.0 Nm cap. Raising the cap to 8.0 Nm had zero effect because the torque signal entering the cap was identical.

---

## Detailed Comparison Results

### Drift Metrics: IDENTICAL ✓

All drift metrics between T5 and T6B are identical:
- `active_pitch_crossing_signed_error_m`: IDENTICAL
- `tuned_band_state_id`: IDENTICAL
- `tuned_abs_error`: IDENTICAL
- `tuned_error_rate`: IDENTICAL

**Meaning:** Same drift → same band state → same APCR activation pattern.

### Authority Parameters: DIFFER ✓

Configuration parameters differ as expected:

| Parameter | Differ % | Max Diff | Mean Diff | First Diff Step |
|-----------|----------|----------|-----------|-----------------|
| `tuned_position_cap_current` | 52.4% | 1.000 Nm | 0.259 Nm | 99 |
| `tuned_wheel_damping_scale` | 17.7% | 0.050 | 0.009 | 143 |

- Max cap difference = 1.0 Nm = exactly (T6B emergency 8.0) - (T5 emergency 7.0)
- Max damping difference = 0.05 = exactly (T6B emergency 0.05) - (T5 emergency 0.10)

**Meaning:** T6B configuration was correctly applied. Controller switched between different caps and damping scales.

### Intermediate Torques: IDENTICAL ✗

ALL intermediate torques are identical:

| Torque Signal | Status |
|---------------|--------|
| `tau_pitch` | IDENTICAL |
| `tau_position_raw` | IDENTICAL ← **Before cap** |
| `apcr1n_tau_position_after_cap` | IDENTICAL ← **After cap** |
| `active_pitch_crossing_tau` | IDENTICAL |
| `sagittal_balance_torque_raw` | IDENTICAL |

**Critical Finding:** `tau_position_raw` (before cap) is IDENTICAL, and `apcr1n_tau_position_after_cap` (after cap) is ALSO identical. This means:

1. The raw torque signal entering the cap was the same for T5 and T6B
2. Even though T6B applied a higher cap (8.0 vs 7.0), the output after capping was identical
3. **The raw torque never exceeded 7.0 Nm**

### Final Torques: IDENTICAL ✗

ALL final torques are identical:

| Final Torque | Status |
|--------------|--------|
| `final_wheel_tau_with_apc` | IDENTICAL |
| `tau_smooth_l_wheel` | IDENTICAL |
| `tau_smooth_r_wheel` | IDENTICAL |
| `tau_total_clipped_l_wheel` | IDENTICAL |
| `tau_total_clipped_r_wheel` | IDENTICAL |

**Meaning:** Identical intermediate torques → identical final torques → identical wheel commands.

### State/Dynamics: IDENTICAL ✓

ALL state variables are identical:

| State Variable | Status |
|----------------|--------|
| `qvel_l_wheel` | IDENTICAL |
| `qvel_r_wheel` | IDENTICAL |
| `com_z` | IDENTICAL |

**Meaning:** Identical wheel torques → identical wheel velocities → identical dynamics.

---

## Root Cause Analysis

**Why did T6B produce identical torques despite different caps?**

### Hypothesis: Raw Torque Never Saturated T5's 7.0 Nm Cap

**Evidence:**
1. `tau_position_raw` IDENTICAL between T5 and T6B
2. `apcr1n_tau_position_after_cap` IDENTICAL between T5 and T6B
3. If raw torque exceeded 7.0 Nm:
   - T5 would clip to 7.0 Nm
   - T6B would clip to 8.0 Nm
   - After-cap torques would DIFFER
4. But after-cap torques are IDENTICAL
5. Therefore: raw torque stayed ≤ 7.0 Nm for both

**Implication:** Raising the emergency cap from 7.0 → 8.0 Nm cannot help if the raw torque signal never reaches 7.0 Nm. T6B's authority boost addressed a **non-existent saturation problem**.

### Alternative Hypothesis: Damping Scale Change Cancelled by Feedback

**Evidence:**
1. Damping scale differs (0.10 vs 0.05 in emergency band)
2. But `tau_position_raw` IDENTICAL
3. Damping affects wheel velocity damping term
4. Lower damping → less opposition to wheel motion
5. But if wheel velocity is determined by closed-loop feedback, damping change might be absorbed

**Status:** Need Phase 4 to audit damping contribution.

---

## Classification

**T6B_CONFIG_DIFFERS_BUT_FINAL_TORQUE_IDENTICAL**

T6B configuration parameters differ from T5 as designed, but every torque signal (raw, capped, final) remains identical. This indicates T6B's parameter changes did not reach the torque output because:

1. The raw torque signal was identical (same drift → same controller input)
2. The cap was not saturating (raising cap had no effect)
3. The damping change either had no effect or was cancelled by feedback

---

## Next Steps

**Phase 3:** Position cap saturation audit
- Count steps where T5 `tau_position_raw` exceeds 7.0 Nm
- Count steps where T5 `tau_position_raw` exceeds 8.0 Nm
- Determine whether cap was ever relevant

**Phase 4:** Damping scale effect audit
- Audit whether damping contribution differs between T5 and T6B
- Check if damping fights or helps drift recovery
- Determine whether damping change could matter

---

## Artifacts

**Analysis Script:**
- `compare_t5_t6b_torque_path.py`

**Data:**
- `outputs/.../t5_vs_t6b_stepwise_torque_path_diff.json`
- `outputs/.../t5_vs_t6b_stepwise_torque_path_diff.csv`

**Status:** Phase 2 complete  
**Classification:** T6B_CONFIG_DIFFERS_BUT_FINAL_TORQUE_IDENTICAL  
**Date:** 2026-06-12
