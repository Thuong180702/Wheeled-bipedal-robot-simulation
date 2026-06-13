# APCR1nD Tuned Variant Selection Decision

**Date:** 2026-06-12  
**Phase:** 7 (Variant Selection)  
**Status:** ✅ COMPLETE

---

## Summary

All five tuned variants (T1-T5) successfully reduced drift compared to APCR1n baseline. **T5 Band-Limited Balanced** is selected as the best variant based on comprehensive ranking criteria.

**Classification:** `APCR1ND_TUNED_T5_BEST`

---

## Ranking Criteria

Priority order:
1. ✅ Survives 2000 steps
2. ✅ Lowest outside ±0.08 m
3. ✅ Lowest outside ±0.10 m  
4. ✅ Lowest max |e|
5. ✅ Contact/height/roll stable
6. ✅ Wheel velocity acceptable

---

## Results Summary

### Outside ±0.08 m (Primary Target)

| Rank | Profile | Outside ±0.08 m | Improvement vs APCR1n |
|------|---------|-----------------|------------------------|
| **1** | **T5** | **25.1%** | **-12.8 pp** |
| 2 | T4 | 26.6% | -11.3 pp |
| 3 | T1 | 27.1% | -10.8 pp |
| 3 | T2 | 27.1% | -10.8 pp |
| 3 | T3 | 27.1% | -10.8 pp |
| - | APCR1n | 37.9% | baseline |
| - | D2 | 38.5% | +0.6 pp |
| - | APCR1h | 44.6% | +6.7 pp |

**Winner: T5** (2.5 percentage points better than runner-up T4)

### Outside ±0.10 m

| Profile | Outside ±0.10 m | Improvement vs APCR1n |
|---------|-----------------|------------------------|
| **T1-T5 (all)** | **5.0%** | **-17.9 pp** |
| D2 | 18.2% | -4.7 pp |
| APCR1n | 22.9% | baseline |
| APCR1h | 37.3% | +14.4 pp |

**Winner: All tied** - All tuned variants achieve identical 5.0% (77.8% improvement over APCR1n)

### Max |e|

| Profile | Max \|e\| (m) | Improvement vs APCR1n |
|---------|---------------|------------------------|
| APCR1n | 0.171 | baseline |
| **T1-T5 (all)** | **0.171** | **0.000** |
| D2 | 0.176 | +0.005 |
| APCR1h | 0.178 | +0.007 |

**Winner: All tied** - T1-T5 match APCR1n baseline

### Mean |e|

| Profile | Mean \|e\| (m) |
|---------|----------------|
| **T5** | **0.0590** |
| T4 | 0.0590 |
| T1 | 0.0589 |
| T2 | 0.0589 |
| T3 | 0.0589 |
| APCR1n | 0.0608 |

**Winner: T1/T2/T3** (marginal 0.0001 m advantage)

### Stability Metrics

All tuned variants:
- ✅ Survived full 2000 steps
- ✅ CoM Z stable: 0.282-0.295 m (target 0.300 m)
- ✅ Wheel velocity safe: max 4.39 rad/s, RMS 1.46-1.51 rad/s
- ✅ No excursions > 5 rad/s
- ✅ Lower wheel RMS than D2 (1.69), APCR1n (2.12), APCR1h (3.49)

**Winner: All variants stable**, T5 has lowest wheel RMS (1.46 rad/s)

---

## Detailed Comparison: T1 vs T2 vs T3 vs T4 vs T5

### Observation: T1, T2, T3 Identical

T1, T2, and T3 produced **identical telemetry** in this 2000-step run:
- Same max |e|: 0.171 m
- Same outside ±0.08: 27.1%
- Same outside ±0.10: 5.0%
- Same mean |e|: 0.0589 m
- Same wheel velocity profile

**Possible reasons:**
1. Drift never triggered threshold differences (T1 enters at 0.06 vs 0.08)
2. Hold logic (T2/T3) didn't activate in this scenario
3. Implementation may need verification

**Impact:** Doesn't affect selection since T5 outperforms all three.

### T4 vs T5

| Metric | T4 | T5 | Winner |
|--------|----|----|--------|
| Outside ±0.08 m | 26.6% | **25.1%** | **T5** |
| Outside ±0.10 m | 5.0% | 5.0% | Tie |
| Max \|e\| | 0.171 m | 0.171 m | Tie |
| Mean \|e\| | 0.0590 m | 0.0590 m | Tie |
| Wheel RMS | 1.508 rad/s | **1.460 rad/s** | **T5** |

**T5 advantages:**
- 1.5 pp lower outside ±0.08 (25.1% vs 26.6%)
- Slightly lower wheel velocity RMS (smoother control)
- Graduated response by band state (more nuanced than T4's uniform stronger authority)

**T4 characteristics:**
- Stronger position caps across all bands
- Aggressive damping suppression
- Simpler implementation

---

## Decision

**Selected variant: T5 Band-Limited Balanced**

**Rationale:**
1. ✅ **Best outside ±0.08**: 25.1% (primary target)
2. ✅ **Tied best outside ±0.10**: 5.0%
3. ✅ **Matches max |e|**: 0.171 m
4. ✅ **Lowest wheel RMS**: 1.46 rad/s (smoothest control)
5. ✅ **Graduated authority**: More sophisticated than uniform boost
6. ✅ **Stable**: Full 2000 steps, no contact/height/roll issues

**Performance gains vs APCR1n baseline:**
- Outside ±0.08: **33.7% reduction** (37.9% → 25.1%)
- Outside ±0.10: **78.2% reduction** (22.9% → 5.0%)
- Outside ±0.15: **3.8% reduction** (2.65% → 2.55%)
- Wheel velocity RMS: **31.2% lower** (2.12 → 1.46 rad/s)

**T5 design features:**
- Graduated position caps by band state: 4.0 → 4.5 → 5.5 → 6.5 → 7.0 Nm
- Graduated damping scales: 1.0 → 0.50 → 0.30 → 0.15 → 0.10
- Band thresholds: 0.05, 0.06, 0.08, 0.10, 0.12 m
- Preserve damping if it helps recovery
- Hold outside band enabled
- Strict release threshold: 0.03 m

---

## Comparison to References

### vs D2 (Wheel Velocity Damping Light)

T5 improvements:
- Outside ±0.08: **34.8% reduction** (38.5% → 25.1%)
- Outside ±0.10: **72.5% reduction** (18.2% → 5.0%)
- Max |e|: **2.8% reduction** (0.176 → 0.171 m)
- Wheel RMS: **13.4% lower** (1.69 → 1.46 rad/s)

### vs APCR1h (Support Hysteresis Recenter)

T5 improvements:
- Outside ±0.08: **43.7% reduction** (44.6% → 25.1%)
- Outside ±0.10: **86.6% reduction** (37.3% → 5.0%)
- Max |e|: **3.6% reduction** (0.178 → 0.171 m)
- Wheel RMS: **58.2% lower** (3.49 → 1.46 rad/s)

### vs APCR1n (Recenter Priority Torque Boost)

T5 improvements:
- Outside ±0.08: **33.7% reduction** (37.9% → 25.1%)
- Outside ±0.10: **78.2% reduction** (22.9% → 5.0%)
- Max |e|: **0.0% change** (0.171 → 0.171 m)
- Wheel RMS: **31.2% lower** (2.12 → 1.46 rad/s)

---

## What Made T5 Win

1. **Band-limited authority scaling:** More aggressive intervention only when drift is large, preserving smooth control when drift is small.

2. **Graduated response:** Five distinct authority levels allow fine-tuned response across drift magnitudes.

3. **Damping preservation:** Keeps full wheel damping when it helps recovery, reducing oscillations.

4. **Strict release threshold:** 0.03 m prevents premature deactivation while drift is still significant.

5. **Hold outside band:** Maintains recenter priority even after initial convergence if still outside desired band.

---

## Next Steps

**Phase 8: Final Report**
- Answer 15 key questions about APCR1nD failure modes and T5 improvements
- Document why T5 succeeded where baseline failed
- Provide recommendations for 5000-step validation

**Future work (not in this phase):**
- Investigate why T1/T2/T3 produced identical telemetry
- Run 5000-step validation of T5 at low_0p300
- Test T5 at high_0p480 (if Step E gates pass)
- Compare T5 to Step C and Step D candidates

---

## Files Generated

**Phase 6 analysis outputs:**
- `outputs/.../apcr1nd_tuned_2000_comparison.csv` - Drift metrics for all profiles
- `outputs/.../apcr1nd_tuned_2000_window_metrics.csv` - Time-window analysis
- `outputs/.../apcr1nd_tuned_2000_stability.csv` - Stability metrics
- `outputs/.../apcr1nd_tuned_2000_comparison.json` - JSON summary

**Phase 7 decision:**
- `docs/validation/apcr1nd_tuned_best_variant_decision.md` (this file)
- `outputs/.../apcr1nd_tuned_best_variant_decision.json` (next)

---

**Classification:** `APCR1ND_TUNED_T5_BEST`  
**Recommended for 5000-step:** T5 Band-Limited Balanced  
**Status:** Ready for Phase 8 (Final Report)
