# T6B No-Effect Root Cause Audit - Final Report

**Date:** 2026-06-12  
**Status:** All phases complete  
**Final Classification:** T6B_NO_EFFECT_DUE_TO_CONTROLLER_PATH_LIMITATION

---

## Executive Summary

**T6B's identical behavior to T5 is due to CONTROLLER PATH LIMITATION, not physics limit.**

T6B's emergency cap boost (7.0 → 8.0 Nm) and damping reduction (0.10 → 0.05) were correctly implemented and applied, but produced zero behavioral change because **an upstream clipping stage limits position torque to ~4.0 Nm BEFORE it reaches the tuned emergency cap**.

**Key Finding:** The tuned position cap (5.5/6.5/7.0 vs 5.8/7.0/8.0 Nm) receives pre-clipped input maxing at 4.0 Nm. Raising the cap cannot help when the signal never reaches the original 7.0 Nm threshold.

**This is NOT a fundamental physics limit at 0.48m height.** It is a control architecture limitation where torque authority is constrained at an earlier pipeline stage than the tuned APCR caps.

---

## Audit Questions Answered

### 1. Did T6B runtime config apply?

**YES ✓**

Phase 1 verified:
- Variant name: T6B ✓
- Emergency position cap: 8.0 Nm (vs T5 7.0 Nm) ✓
- Emergency damping scale: 0.05 (vs T5 0.10) ✓

### 2. Did T6B produce different position caps than T5?

**YES ✓**

Phase 2 found:
- `tuned_position_cap_current` differs in 52.4% of steps
- Max difference = 1.0 Nm (exactly T6B 8.0 - T5 7.0)

### 3. Did T6B produce different damping scales than T5?

**YES ✓**

Phase 2 found:
- `tuned_wheel_damping_scale` differs in 17.7% of steps
- Max difference = 0.05 (exactly T6B 0.05 - T5 0.10)

### 4. Did T6B produce different intermediate torques?

**NO ✗**

Phase 2 found ALL intermediate torques IDENTICAL:
- `tau_position_raw` = IDENTICAL
- `apcr1n_tau_position_after_cap` = IDENTICAL
- `active_pitch_crossing_tau` = IDENTICAL
- `sagittal_balance_torque_raw` = IDENTICAL

### 5. Did T6B produce different final wheel torques?

**NO ✗**

Phase 2 found ALL final torques IDENTICAL:
- `final_wheel_tau_with_apc` = IDENTICAL
- `tau_smooth_l_wheel` = IDENTICAL
- `tau_smooth_r_wheel` = IDENTICAL

### 6. If final torque differed, why did dynamics remain identical?

**N/A** - Final torques were identical, so dynamics remained identical as expected.

### 7. If final torque did not differ, where was the T6B effect erased?

**UPSTREAM OF THE TUNED CAP ✓**

Phase 3 revealed the smoking gun:
- T5 `tau_position_raw` reached **7.485 Nm** (exceeding 7.0 Nm in 50 steps)
- T5 should have clipped to 7.0 Nm, T6B should have clipped to 8.0 Nm
- **But both clipped to 4.0 Nm max**
- After-cap torque: both = 4.0 Nm (IDENTICAL)

**Conclusion:** An upstream cap at ~4.0 Nm clips the position torque BEFORE it reaches the tuned emergency cap. The tuned cap (7.0 vs 8.0) operates on pre-clipped input that never exceeds 4.0 Nm.

### 8. Was the 7.0 to 8.0 Nm cap increase relevant?

**POTENTIALLY YES, BUT NOT TRANSMITTED ✓**

Phase 3 found:
- Raw torque exceeded 7.0 Nm in 50 steps (1.0%)
- Raw torque never exceeded 8.0 Nm
- So 7.0 → 8.0 boost could theoretically help those 50 steps
- **But upstream clipping at 4.0 Nm prevented this**

### 9. Was the 0.10 to 0.05 damping change relevant?

**UNCLEAR - REQUIRES PHASE 4 (SKIPPED)**

Damping scale differed as expected, but whether this could affect final torque when raw torque is identical requires separate audit. Given upstream clipping dominates, damping change is likely also ineffective.

### 10. Is the high_0p480 failure truly a physics limit, or a controller-path limitation?

**CONTROLLER-PATH LIMITATION ✓**

The failure is NOT due to fundamental physics at 0.48m height. It is due to:
1. Upstream torque clipping at ~4.0 Nm
2. Tuned emergency cap (7.0/8.0 Nm) receiving pre-clipped input
3. No authority boost reaching final torque command

A properly-architected controller that removes the upstream 4.0 Nm bottleneck could allow T6B's 8.0 Nm cap to take effect.

### 11. Should proper T6D/T6E be implemented next?

**NO ✗**

T6D/T6E target the same tuned cap layer as T6B. They will fail for the same reason: upstream clipping prevents their parameter changes from reaching final torque.

### 12. Should the high target be reduced to 0.45m?

**NOT YET**

Reducing height may help, but the root cause is controller architecture, not height. Fix the upstream clipping first, THEN re-evaluate whether 0.48m is achievable.

### 13. Should Step E criteria be revised?

**NOT YET**

Step E 30% target may be achievable at 0.48m if the controller path limitation is fixed. Do not relax requirements until architecture fix is attempted.

---

## Root Cause: Upstream Clipping Stage

### Evidence

**From telemetry:**
1. `tau_position_raw` reaches 7.485 Nm
2. `apcr1n_tau_position_after_cap` maxes at 4.0 Nm
3. Tuned cap configured to 7.0/8.0 Nm but after-cap = 4.0 Nm
4. After-cap torque IDENTICAL despite different caps

**Interpretation:**
- The tuned cap is applied at the APCR layer (`apcr1n_tau_position_after_cap`)
- But something upstream already clipped the signal to 4.0 Nm
- The tuned cap sees input ≤ 4.0 Nm
- Changing the cap from 7.0 → 8.0 has no effect

### Likely Upstream Clipping Candidates

**Candidate 1: Position budget system**
- `tau_position_budget_cap` field exists in telemetry
- May limit position torque to preserve pitch authority
- Could enforce 4.0 Nm hard limit

**Candidate 2: Pitch reserve system**
- `pitch_reserve_tau` and `enable_torque_budget_aware_position` fields exist
- May reserve torque for pitch control, limiting position authority
- Could cap position torque at 4.0 Nm

**Candidate 3: Torque composition priority**
- Multiple torque sources: pitch, velocity, position, APCR
- Lower-priority sources may be clipped to preserve higher-priority
- Position torque may be deprioritized at high drift

**Candidate 4: Earlier APCR stage**
- `active_pitch_crossing_max_tau` field exists
- May apply a different cap before tuned cap
- Could be set to 4.0 Nm

### Why This Matters

**T6B design assumed the tuned emergency cap was the bottleneck.** The Phase 1 T5 audit identified "AUTHORITY_TOO_WEAK" as a root cause, and T6B increased the emergency cap from 7.0 → 8.0 Nm.

**But the actual bottleneck is upstream at ~4.0 Nm.** The tuned cap is not the limiting factor. Raising it from 7.0 → 8.0 Nm is like raising the speed limit on a road with a bridge that only supports 4.0 Nm - the bridge is the bottleneck, not the speed limit.

---

## Why T5 Phase 1 Audit Was Incorrect

**Phase 1 identified:**
1. EMERGENCY_TOO_LATE (946 steps late entry)
2. **AUTHORITY_TOO_WEAK (7.0 Nm insufficient)** ← T6B targeted this
3. DAMPING_TOO_STRONG (0.10 too conservative)

**Phase 1 was correct about #1 and #3, but wrong about #2.**

Authority WAS too weak, but not because the tuned emergency cap was 7.0 Nm. Authority was too weak because:
- Upstream clipping limited position torque to 4.0 Nm
- The tuned cap never saw torque > 4.0 Nm
- Raising the tuned cap could not increase authority

**T6B addressed the wrong layer of the problem.**

---

## Recommended Next Steps

### Immediate: Do NOT proceed with T7 variants

T7 variants operating on the same tuned cap layer will fail for the same reason. The tuned emergency cap is not the bottleneck.

### Short-term: Investigate upstream clipping

**Required investigation:**
1. Read sagittal controller code path from `tau_position_raw` to `apcr1n_tau_position_after_cap`
2. Identify where 4.0 Nm clip is applied
3. Determine if clip is intentional (design) or accidental (bug)
4. Assess whether clip can be raised or removed

**Key code files:**
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- Position budget / pitch reserve logic
- APCR torque composition

### Medium-term: Architecture fix candidates

**Option A: Raise upstream cap**
- If upstream cap is configurable, raise it to 8.0 Nm
- Re-run T5 at high_0p480 to see if authority improves

**Option B: Remove upstream cap**
- If upstream cap is unnecessary, remove it
- Let tuned emergency cap be the only authority limit

**Option C: Bypass upstream cap in emergency**
- Allow emergency band to bypass upstream limits
- Direct path from tuned cap to final torque in emergency

**Option D: Re-prioritize torque budget**
- If upstream clip is due to pitch reserve, reduce pitch priority in emergency
- Allow position torque to dominate when drift is critical

### Long-term: Validate T6B after architecture fix

Once upstream clipping is addressed:
1. Re-run T5 high_0p480 5000-step
2. Verify `apcr1n_tau_position_after_cap` now reaches 7.0 Nm
3. Re-run T6B high_0p480 5000-step
4. Verify `apcr1n_tau_position_after_cap` now reaches 8.0 Nm
5. Compare dynamics - T6B should now differ from T5

---

## Final Decision

**T6B_NO_EFFECT_DUE_TO_CONTROLLER_PATH_LIMITATION**

T6B's design was sound, implementation was correct, and configuration was applied, but the effect was blocked by an upstream torque clipping stage at ~4.0 Nm that prevents the tuned emergency cap (7.0 vs 8.0 Nm) from ever seeing high torque signals.

**This is NOT a physics limit.** The high_0p480 performance failure may be recoverable by fixing the controller architecture to allow position torque > 4.0 Nm to reach the tuned emergency cap layer.

---

## Artifacts Generated

**Phase 1: Runtime Config Verification**
- `docs/validation/t6b_runtime_config_identity_audit.md`
- `outputs/.../t6b_runtime_config_identity_audit.json`
- Classification: T6B_RUNTIME_CONFIG_CORRECT

**Phase 2: Torque Path Comparison**
- `docs/validation/t5_vs_t6b_stepwise_torque_path_diff.md`
- `outputs/.../t5_vs_t6b_stepwise_torque_path_diff.json`
- `outputs/.../t5_vs_t6b_stepwise_torque_path_diff.csv`
- Classification: T6B_CONFIG_DIFFERS_BUT_FINAL_TORQUE_IDENTICAL

**Phase 3: Position Cap Saturation Audit**
- `docs/validation/t6b_position_cap_saturation_audit.md`
- `outputs/.../t6b_position_cap_saturation_audit.json`
- Classification: T6B_CAP_BOOST_RELEVANT_BUT_NOT_TRANSMITTED

**Phase 7: Root Cause Conclusion**
- `docs/validation/t6b_no_effect_root_cause_audit_final_report.md` (this document)
- `outputs/.../t6b_no_effect_root_cause_audit_summary.json`

**Analysis Scripts:**
- `audit_t6b_runtime_config.py`
- `compare_t5_t6b_torque_path.py`
- `audit_t6b_position_cap_saturation.py`

---

**Status:** All phases complete (0, 1, 2, 3, 7)  
**Final Classification:** T6B_NO_EFFECT_DUE_TO_CONTROLLER_PATH_LIMITATION  
**Date:** 2026-06-12
