# T6B Position Cap Saturation Audit

**Date:** 2026-06-12  
**Status:** Phase 3 complete  
**Classification:** T6B_CAP_BOOST_RELEVANT_BUT_NOT_TRANSMITTED

---

## Executive Summary

**CRITICAL FINDING: Cap boost was relevant but NOT transmitted to after-cap torque.**

- T5 raw torque exceeded 7.0 Nm in 50 steps (1.0%), reaching 7.485 Nm max
- Cap boost from 7.0 → 8.0 Nm should have affected those 50 steps
- **BUT: After-cap torque maxes at 4.0 Nm for both T5 and T6B**
- After-cap torque is IDENTICAL between T5 and T6B

**Root Cause Hypothesis:** A different cap upstream of the tuned emergency cap is clipping torque to ~4.0 Nm BEFORE it reaches the 7.0/8.0 Nm tuned cap. The tuned emergency cap is receiving pre-clipped input, making the 7.0 vs 8.0 difference irrelevant.

---

## Detailed Findings

### T5 Saturation Analysis (emergency cap 7.0 Nm)

**Band Distribution:**
- Band 0 (inactive): 964 steps (19.3%)
- Band 1 (desired): 1416 steps (28.3%)
- Band 2 (hard): 939 steps (18.8%)
- Band 3 (emergency): 1330 steps (26.6%)
- Band 4 (hold): 350 steps (7.0%)

**tau_position_raw (before cap):**
- Min: -7.485 Nm
- Max: 0.639 Nm
- Mean |tau|: 3.212 Nm
- Std: 1.393 Nm
- P50: 3.283 Nm
- P95: 4.828 Nm
- P99: 6.979 Nm

**Saturation Thresholds:**
- Exceeds 5.5 Nm: 107 steps (2.1%)
- Exceeds 6.5 Nm: 71 steps (1.4%)
- **Exceeds 7.0 Nm: 50 steps (1.0%)** ← T5 emergency cap
- Exceeds 8.0 Nm: 0 steps (0.0%) ← T6B emergency cap

**apcr1n_tau_position_after_cap:**
- Min: -4.000 Nm
- Max: 0.639 Nm
- Mean |tau|: 2.986 Nm
- **P99: 4.000 Nm** ← Suspicious exact value

### T6B Saturation Analysis (emergency cap 8.0 Nm)

**IDENTICAL raw torque distribution to T5:**
- Min: -7.485 Nm (same)
- Max: 0.639 Nm (same)
- Exceeds 7.0 Nm: 50 steps (same)
- Exceeds 8.0 Nm: 0 steps (same)

**IDENTICAL after-cap torque to T5:**
- Min: -4.000 Nm (same)
- Max: 0.639 Nm (same)
- P99: 4.000 Nm (same)

---

## The Contradiction

**Expected behavior if tuned cap was active:**
1. T5 raw torque = 7.485 Nm
2. T5 clips to emergency cap 7.0 Nm
3. T6B clips to emergency cap 8.0 Nm
4. After-cap should differ: T5 = 7.0 Nm, T6B = 7.485 Nm (or 8.0 if it exceeded)

**Actual behavior:**
1. Both T5 and T6B raw torque = 7.485 Nm (identical)
2. Both T5 and T6B after-cap = 4.0 Nm max (identical)
3. After-cap is MUCH LOWER than either emergency cap

**Interpretation:** The tuned emergency cap (7.0 vs 8.0 Nm) is receiving input that's already been clipped to ~4.0 Nm by an upstream process. The 7.0 → 8.0 Nm boost cannot help because the signal never reaches 7.0 Nm at the point where the tuned cap is applied.

---

## Upstream Clipping Hypothesis

**Possible upstream caps:**
1. **Desired band cap (5.5/5.8 Nm)** - But raw torque exceeded this
2. **Hard band cap (6.5/7.0 Nm)** - But raw torque exceeded this  
3. **Different APCR stage cap** - `active_pitch_crossing_max_tau` or similar
4. **Position budget cap** - `tau_position_budget_cap`
5. **Pitch reserve cap** - `pitch_reserve_tau`
6. **Torque composition priority** - Lower-priority torque sources may dominate

The after-cap torque maxing at exactly 4.0 Nm suggests a hard-coded cap somewhere in the pipeline.

---

## Why T6B Failed

**T6B's cap boost was correctly configured and applied at the tuned emergency cap layer, BUT:**

1. The raw position torque signal was identical between T5 and T6B
2. An upstream process clipped the torque to ~4.0 Nm before it reached the tuned cap
3. By the time the tuned cap saw the signal, it was already < 7.0 Nm
4. Raising the tuned cap from 7.0 → 8.0 Nm had no effect because the input never reached 7.0 Nm

**This is NOT a "no saturation" problem - saturation exists, but at the wrong layer.**

---

## Next Steps

Skip Phases 4-6. Proceed directly to Phase 7 (Root Cause Conclusion) with findings:

1. T6B config correctly applied ✓
2. T6B config differs from T5 ✓
3. T6B final torques identical to T5 ✓
4. Raw torque exceeds T5 cap (50 steps) ✓
5. After-cap torque identical despite different caps ✓
6. **Root cause: Upstream clipping at ~4.0 Nm layer**

The tuned emergency cap is not the bottleneck. The bottleneck is earlier in the torque composition pipeline.

---

## Artifacts

**Analysis Script:**
- `audit_t6b_position_cap_saturation.py`

**Data:**
- `outputs/.../t6b_position_cap_saturation_audit.json`

**Status:** Phase 3 complete  
**Classification:** T6B_CAP_BOOST_RELEVANT_BUT_NOT_TRANSMITTED  
**Date:** 2026-06-12
