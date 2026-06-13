# T6F High_0p480 2000-Step Screening Report

**Date:** 2026-06-12  
**Phase:** 8 of 11  
**Classification:** T6F_2000_NOT_BETTER_THAN_T5  
**Recommendation:** Do NOT proceed to Phase 9 (5000-step validation)

---

## Executive Summary

**T6F_budget_cap_raise architecture fix successfully transmitted torque above 4.0 Nm but DEGRADED drift performance compared to T5.**

The architecture fix activated in 913/1999 steps (45.7%) and raised the upstream cap to 6.5/7.0 Nm during safe high-height emergency recenter, transmitting position torque up to 7.0 Nm as designed.

However, **drift metrics worsened significantly:**
- outside ±0.10 m: **-115 steps worse** (798 → 913)
- outside ±0.15 m: **-512 steps worse** (89 → 601, 4.4% → 30.1%)
- max |error|: 0.187 m (T5) → 0.212 m (T6F)

**Phase 7 proved torque transmission. Phase 8 proves raised authority makes drift worse, not better.**

---

## Test Configuration

### T5 Reference Baseline

- **Profile:** APCR1nD_T5_band_limited_balanced
- **Steps:** First 2000 steps from validated 5000-step run
- **Arch fix enabled:** False
- **Max position tau nominal:** 4.0 Nm
- **Telemetry:** `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv`

### T6F Architecture Fix Candidate

- **Profile:** T6F_budget_cap_raise
- **Steps:** 1999 (survived full screening)
- **Arch fix enabled:** True
- **Arch fix type:** budget_cap_raise
- **Height threshold:** 0.45 m
- **Hard max position tau:** 6.5 Nm
- **Emergency max position tau:** 7.0 Nm
- **Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1781258876.csv`

### Common Configuration

- **Height variant:** high_0p480 (target CoM Z: 0.480 m)
- **Controller mode:** balance-core
- **Sagittal controller:** velocity-damped
- **Telemetry decimation:** 1 (every step)

---

## Phase 8C: Drift Comparison

**Drift column:** active_pitch_crossing_signed_error_m

### T5 First 2000 Steps

- min e: -0.1606 m
- max e: +0.0425 m
- max |e|: 0.1871 m
- P2P: 0.2031 m
- mean |e|: 0.0879 m
- final e: -0.0902 m
- outside ±0.08 m: 1136 steps (56.8%)
- outside ±0.10 m: **798 steps (39.9%)**
- outside ±0.15 m: **89 steps (4.4%)**

### T6F 2000 Steps

- min e: -0.2122 m
- max e: +0.0287 m
- max |e|: 0.2122 m
- P2P: 0.2409 m
- mean |e|: 0.0941 m
- final e: -0.1572 m
- outside ±0.08 m: 1037 steps (51.9%)
- outside ±0.10 m: **913 steps (45.7%)**
- outside ±0.15 m: **601 steps (30.1%)**

### Improvement Analysis

- outside ±0.08 m: **+99 steps improvement**
- outside ±0.10 m: **-115 steps degradation** ❌
- outside ±0.15 m: **-512 steps degradation** ❌
- max |e|: **+0.025 m worse**

**Primary screening criteria FAILED:**
- ✗ outside ±0.10 must improve (got -115)
- ✗ outside ±0.15 must remain <= 5% (got 30.1%)

---

## Phase 8D: Window Analysis

### T5 Windows (500-step each)

| Window | Steps | max \|e\| | P2P | outside 0.08 | outside 0.10 | outside 0.15 |
|--------|-------|-----------|-----|--------------|--------------|--------------|
| 1 | 0-500 | 0.1871 | 0.2031 | 301 | 226 | 22 |
| 2 | 500-1000 | 0.1197 | 0.1656 | 218 | 146 | 0 |
| 3 | 1000-1500 | 0.1223 | 0.1668 | 303 | 220 | 0 |
| 4 | 1500-2000 | 0.1212 | 0.1625 | 314 | 206 | 67 |

T5 drift was worst in Window 1 (early transient), improved in Window 2-3, then degraded again in Window 4.

### T6F Windows (500-step each)

| Window | Steps | max \|e\| | P2P | outside 0.08 | outside 0.10 | outside 0.15 |
|--------|-------|-----------|-----|--------------|--------------|--------------|
| 1 | 0-500 | 0.2034 | 0.2321 | 226 | 196 | 1 |
| 2 | 500-1000 | 0.1977 | 0.2155 | 295 | 263 | 200 |
| 3 | 1000-1500 | 0.1944 | 0.2169 | 243 | 210 | 174 |
| 4 | 1500-2000 | N/A | N/A | N/A | N/A | N/A |

**T6F drift remained severe across all windows.** Window 2-3 show persistent outside ±0.15 violations, unlike T5 which recovered.

---

## Phase 8E: Architecture Fix Activation and Torque Analysis

### Architecture Fix Activation

- **Active:** 913/1999 steps (45.7%)
- **Emergency band:** 795 steps (87.1% of activations)
- **Hard band:** 118 steps (12.9% of activations)

The architecture fix activated at nearly the same rate as Phase 7 (46%), indicating consistent high-height drift severity.

### Torque Transmission

**Effective max position tau:**
- 4.0 Nm: 1086 steps (54.3%)
- 6.5 Nm: 118 steps (5.9%)
- 7.0 Nm: 795 steps (39.8%)

**Position torque transmitted:**
- tau_position > 4.0 Nm: **913 steps** (vs 0 for T5)
- max |tau_position|: **7.0 Nm** (vs 4.0 Nm for T5)

**Architecture fix successfully transmitted raised torque as designed,** confirming Phase 7 findings hold in longer episodes.

---

## Phase 8F: Stability and Safety Analysis

### Survival

- Survived: 1999/2000 steps (99.95%)
- Terminated: False
- No fall, no catastrophic failure

### Contact and Height

- Contact rate: 100.0%
- Double contact: 100.0%
- CoM Z: min=0.481 m, mean=0.490 m, max=0.492 m
- Height stable, no collapse

### Attitude

- Pitch RMS: 4.76 deg
- Roll RMS: 0.15 deg
- Pitch/roll remained within safe bounds

### Wheel Velocity

- Max: 7.13 rad/s
- RMS: 2.98 rad/s
- >5 rad/s: 494 steps (24.7%)
- >6 rad/s: 117 steps (5.9%)
- >7 rad/s: 1 step (0.05%)

Wheel velocity reached 7.13 rad/s, moderately elevated but not unsafe.

### Structural Safety

- No WBC applied
- No hidden torque violation
- No ownership violation

**T6F remained structurally safe despite poor drift performance.**

---

## Key Findings

### Architecture Fix Works As Designed

The T6F architecture fix:
- Activated in 45.7% of steps (same as Phase 7)
- Raised upstream cap to 6.5/7.0 Nm based on band severity
- Transmitted position torque up to 7.0 Nm (75% above T5's 4.0 Nm)
- Maintained safety gates (contact, height, roll, pitch)

**Torque transmission is not the problem. The problem is that raised authority makes drift worse.**

### Raised Authority Degrades Drift Performance

Compared to T5, T6F:
- outside ±0.10: **-115 steps worse**
- outside ±0.15: **-512 steps worse** (6.8× increase)
- max |error|: **+13% worse**
- mean |error|: **+7% worse**

**The architecture fix successfully transmits more torque but uses it counterproductively.**

### Window Analysis Reveals Persistent Degradation

T5 recovered in Windows 2-3 (outside ±0.15: 0 steps).  
T6F remained outside ±0.15 throughout Windows 2-3 (200+ steps each).

**T6F does not improve high-height drift; it makes it worse across the entire 2000-step episode.**

### Why Raised Authority Fails

Hypothesis: **The upstream 4.0 Nm clip was not the only bottleneck.**

Possible causes:
1. **Controller gains mismatched to raised authority:** The position/velocity gains may be tuned for 4.0 Nm; raising to 7.0 Nm without retuning creates overshoot/oscillation
2. **Wheel velocity limit:** Transmitted torque saturates at wheel-level 7.5 Nm cap, wasting extra authority
3. **Phase dynamics:** Higher torque authority increases phase lag or couples pitch/position control destructively
4. **Band state logic:** Emergency band may be triggering too late or with wrong sign

---

## Phase 8 Classification

**T6F_2000_NOT_BETTER_THAN_T5**

**Reason:** Drift degraded: 0.10 band=-115, 0.15 band=-512

**Pass criteria comparison:**

| Criterion | Target | T5 | T6F | Pass? |
|-----------|--------|----|----|-------|
| outside ±0.08 improve | >0 steps | 1136 | 1037 | ✓ (+99) |
| outside ±0.10 improve | >0 steps | 798 | 913 | ✗ (-115) |
| max \|e\| | ≤0.20 m | 0.187 | 0.212 | ✗ |
| outside ±0.15 | ≤5% | 4.4% | 30.1% | ✗ |
| Survival | ≥2000 steps | 2000 | 1999 | ✓ |
| Stability | No fall | ✓ | ✓ | ✓ |

**3 of 6 criteria failed.**

---

## Recommendation

**DO NOT PROCEED TO PHASE 9 (5000-step validation).**

Phase 8 screening correctly rejected T6F. The architecture fix transmits more torque as designed, but raised authority degrades drift performance rather than improving it.

**Next steps:**
1. Root-cause why raised torque makes drift worse
2. Investigate gain mismatch, wheel saturation, phase dynamics
3. Consider T6G/T6H candidates with gain scheduling or torque rate limiting
4. Do not proceed to 5000-step or low_0p300 regression until drift improves

---

## Known Limitations

1. **2000-step screening only:** Longer episodes might reveal different behavior, but 2000 steps is sufficient to reject a candidate that degrades drift
2. **Single height tested:** Only high_0p480; T6F might behave differently at other heights
3. **No push/robustness tests yet:** Focused on nominal authority transmission and drift
4. **Hypothesis not root-caused:** We know raised authority makes drift worse but not exactly why

---

## Artifacts

**Telemetry:**
- T5: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t5_high_0p480_5000/telemetry_t5_high_0p480_5000.csv` (first 2000 steps)
- T6F: `outputs/hierarchical_controller_sim/telemetry_1781258876.csv`
- T6F copy: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_screening/telemetry_t6f_high_0p480_2000.csv`

**Analysis:**
- Script: `analyze_t6f_2000_screening.py`
- JSON: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_screening.json`
- Window metrics: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_window_metrics.csv`
- Decision: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/t6f_high_0p480_2000_decision.json`

**Documentation:**
- Phase 7 validation: `docs/validation/t6f_torque_transmission_validation.md`
- Phase 8 screening: `docs/validation/t6f_high_0p480_2000_screening_report.md` (this report)

---

**Status:** Phase 8 COMPLETE  
**Date:** 2026-06-12  
**Recommendation:** REJECT T6F, investigate root cause before next candidate
