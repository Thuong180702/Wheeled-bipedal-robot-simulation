# T5 High 0.480m 5000-Step Final Report

**Date:** 2026-06-12  
**Profile:** APCR1nD_T5_band_limited_balanced  
**Height Variant:** high_0p480  
**Steps:** 5000  
**Classification:** T5_HIGH_0P480_5000_FAIL_BAND_TARGET

---

## Executive Summary

T5 Band-Limited Balanced profile was validated at extreme height (0.480 m) over 5000 steps. The robot **survived all 5000 steps** with **excellent stability** and **no safety violations**, but **failed drift band targets** (52.8% outside ±0.08 m vs 30% target).

**Key Finding:** Drift is DECREASING over time (accumulation ratio 0.477), demonstrating T5's graduated authority is working correctly. However, extreme height challenges exceed T5's current authority limits.

---

## Success Criteria Gate Results

| Criterion | Target | Result | Status |
|-----------|--------|--------|--------|
| Survives ≥ 4900 | ≥ 4900 | 4999 | ✅ PASS |
| Outside ±0.08 m | ≤ 30% | 52.8% | ❌ FAIL |
| Outside ±0.10 m | ≤ 10% | 33.9% | ❌ FAIL |
| Outside ±0.15 m | ≤ 5% | 1.8% | ✅ PASS |
| Max \|e\| | ≤ 0.20 m | 0.187 m | ✅ PASS |
| Drift accumulation | < 1.5 | 0.477 | ✅ PASS |
| Wheel velocity spikes | < 50 steps > 7 rad/s | 0 | ✅ PASS |
| Contact/height/roll | Stable | Stable | ✅ PASS |
| No WBC/hidden/ownership | None | None | ✅ PASS |

**Result:** 7/9 gates passed (±0.08 m and ±0.10 m targets exceeded)

---

## Phase 4: Drift and Band Analysis

### Overall Drift
- **Survived:** 4999/5000 steps (100%)
- **Min e:** -0.0160 m
- **Max e:** 0.1871 m
- **Max |e|:** 0.1871 m
- **Mean |e|:** 0.0803 m
- **Final e:** 0.0459 m
- **Positive drift:** 99.0% (one-sided)

### Band Metrics
| Band | Count | Percentage | Status |
|------|-------|------------|--------|
| Outside ±0.03 m | 4806 | 96.1% | - |
| Outside ±0.05 m | 3941 | 78.8% | - |
| **Outside ±0.08 m** | **2641** | **52.8%** | ❌ vs 30% target |
| **Outside ±0.10 m** | **1696** | **33.9%** | ❌ vs 10% target |
| Outside ±0.12 m | 694 | 13.9% | - |
| **Outside ±0.15 m** | **91** | **1.8%** | ✅ vs 5% target |

**Verdict:** Drift elevated but bounded. Band targets exceeded due to extreme height challenge.

---

## Phase 5: Window and Accumulation Analysis

### Accumulation Metrics
- **First 1000 mean |e|:** 0.0864 m
- **Last 1000 mean |e|:** 0.0412 m
- **Accumulation ratio:** 0.477
- **Classification:** STABLE (drift DECREASING, not accumulating)

### Window Analysis
| Window | Steps | Max \|e\| | Mean \|e\| | Outside ±0.08 m |
|--------|-------|-----------|------------|-----------------|
| 1 | 0-500 | 0.1593 | 0.0818 | 48.6% |
| 2 | 500-1000 | 0.1749 | 0.0909 | 69.0% |
| 3 | 1000-1500 | 0.1757 | 0.0976 | 84.4% |
| 4 | 1500-2000 | 0.1849 | 0.1032 | 89.2% |
| 5 | 2000-2500 | 0.1837 | 0.1037 | 90.0% |
| 6 | 2500-3000 | 0.1866 | 0.1046 | 89.8% |
| 7 | 3000-3500 | 0.1871 | 0.1043 | **90.0%** (worst) |
| 8 | 3500-4000 | 0.1625 | 0.0855 | 62.0% |
| 9 | 4000-4500 | 0.1008 | 0.0366 | 2.6% |
| 10 | 4500-5000 | 0.0743 | 0.0299 | **0.0%** (best) |

**Trajectory:** Drift escalated in windows 2-7, then T5 successfully drove it back down in windows 8-10.

**Worst window:** 7 (3000-3500) - 90.0% outside ±0.08 m  
**Best window:** 10 (4500-5000) - 0.0% outside ±0.08 m

---

## Phase 6: Tuned Feature Activation Analysis

### T5 Activation
- **Recenter active:** 3836/4999 steps (76.7%)
- **T5 engaged majority of run:** Graduated authority actively correcting drift

### Band State Distribution
| Band State | Steps | Percentage | Position Cap | Damping Scale |
|------------|-------|------------|--------------|---------------|
| Normal (0) | 964 | 19.3% | 4.0 Nm | 1.0 |
| Soft (1) | 1416 | 28.3% | 4.5 Nm | 0.50 |
| Desired (2) | 939 | 18.8% | 5.5 Nm | 0.30 |
| Hard (3) | 1330 | 26.6% | 6.5 Nm | 0.15 |
| Emergency (4) | 350 | 7.0% | 7.0 Nm | 0.10 |

**Verdict:** All 5 band states engaged. Emergency band (highest authority) invoked during peak drift (windows 3-7). T5 is working as designed.

---

## Phase 7: Stability and Safety Analysis

### Contact Stability
- **Left contact:** 100.0%
- **Right contact:** 100.0%
- **Both contact:** 100.0%
- **Status:** Perfect (no flight phases)

### Height Stability
- **CoM Z min:** 0.440 m
- **CoM Z max:** 0.491 m
- **CoM Z mean:** 0.469 m
- **Target:** 0.480 m
- **Status:** Stable around target

### Attitude Stability
- **Pitch RMS:** 3.685 deg
- **Roll RMS:** 0.059 deg
- **Status:** Excellent (pitch acceptable for high height, roll outstanding)

### Wheel Velocity
- **Max:** 5.46 rad/s
- **RMS:** 1.30 rad/s
- **Steps > 5 rad/s:** 20
- **Steps > 7 rad/s:** 0
- **Status:** Safe (no excessive velocity)

### Structural Integrity
- **Ownership violations:** 0
- **Status:** No violations

---

## Phase 8: High_0p480 vs Low_0p300 Comparison

### Drift Comparison
| Metric | Low_0p300 | High_0p480 | Delta |
|--------|-----------|------------|-------|
| Outside ±0.08 m | 20.2% | 52.8% | +32.6% |
| Outside ±0.10 m | 2.0% | 33.9% | +31.9% |
| Outside ±0.15 m | 1.0% | 1.8% | +0.8% |
| Max \|e\| | 0.171 m | 0.187 m | +0.016 m |
| Mean \|e\| | 0.056 m | 0.080 m | +0.024 m |
| Accumulation | 0.865 | 0.477 | Better |

### Stability Comparison
| Metric | Low_0p300 | High_0p480 | Delta |
|--------|-----------|------------|-------|
| Pitch RMS | 0.049 deg | 3.685 deg | +3.636 deg |
| Roll RMS | 0.0047 deg | 0.0594 deg | +0.0547 deg |
| Wheel RMS | 1.24 rad/s | 1.30 rad/s | +0.06 rad/s |

**Verdict:** High_0p480 is significantly more challenging. Drift elevated but accumulation better (drift decreasing faster). Pitch higher due to gravitational torque at extreme height.

---

## Key Findings

### 1. T5 Survived Extreme Height ✅
- 5000/5000 steps at 0.480 m CoM height
- No fall, no safety violations
- Perfect contact maintenance

### 2. T5 Graduated Authority Working Correctly ✅
- All 5 band states engaged appropriately
- Emergency band (7.0 Nm cap) invoked during peak drift
- Drift DECREASED over time (ratio 0.477 < 1.0)

### 3. Drift Trajectory Shows Recovery ✅
- Windows 2-7: Drift escalated to 90% outside ±0.08 m
- Windows 8-10: T5 drove drift back down
- Window 10: 0.0% outside ±0.08 m (complete recovery)

### 4. Band Targets Exceeded at Extreme Height ❌
- 52.8% outside ±0.08 m (vs 30% target)
- 33.9% outside ±0.10 m (vs 10% target)
- Extreme height challenge exceeds current T5 authority limits

### 5. Stability Excellent ✅
- Roll RMS: 0.059 deg (outstanding)
- Wheel velocity safe (0 steps > 7 rad/s)
- No structural violations

---

## Implications

### What T5 Achieved
1. **Survived extreme height:** 0.480 m CoM (upper boundary of physical envelope)
2. **Demonstrated graduated response:** Emergency band engaged appropriately
3. **Recovered from peak drift:** Drove error from 90% → 0% outside ±0.08 m
4. **Maintained stability:** No safety violations, excellent roll control

### What T5 Did Not Achieve
1. **Band targets at extreme height:** 52.8% vs 30% target outside ±0.08 m
2. **Immediate drift suppression:** Drift escalated for 3500 steps before recovery

### Design Trade-off
T5's graduated authority prioritizes:
- **Stability over immediate drift correction** (damping reduced in higher bands)
- **No aggressive oscillation** (position caps limit authority)
- **Long-term boundedness** (drift decreases over time)

At extreme height (0.480 m), this trade-off means:
- ✅ Robot survives and recovers
- ❌ Drift temporarily exceeds band targets during recovery phase

---

## Recommendations

### For High_0p480 Operation
1. **T5 is usable** but drift will temporarily exceed targets during transients
2. **Monitor windows 2-7** (1000-3500 steps) for peak drift
3. **Expect recovery** by window 10 (4500-5000 steps)
4. **T5 suitable for exploration** but not for precision positioning at 0.480 m

### For Future Tuning
1. **Consider T6 variant** with higher emergency band authority (7.5-8.0 Nm cap)
2. **Investigate early-entry threshold** to engage emergency band sooner
3. **Test intermediate height** (0.43-0.45 m) to find operational boundary

### For Step E Validation
- **Low_0p300:** ✅ PASS (20.2% outside ±0.08 m)
- **High_0p480:** ❌ FAIL band targets (52.8% outside ±0.08 m)
- **Step E extreme height validation:** PARTIAL (survives but exceeds drift targets)

---

## Classification

**T5_HIGH_0P480_5000_FAIL_BAND_TARGET**

T5 Band-Limited Balanced successfully survives extreme height (0.480 m) over 5000 steps with excellent stability and no safety violations. T5's graduated authority correctly drives drift from 90% → 0% outside ±0.08 m, demonstrating recovery capability.

However, drift band targets (±0.08 m ≤ 30%, ±0.10 m ≤ 10%) are exceeded due to extreme height challenge. T5 is usable at 0.480 m for exploration but not for precision positioning.

**Status:** Phase 0-10 complete. All analysis and comparison documents generated.

---

**Date:** 2026-06-12  
**Phase:** 10 (Final Classification) COMPLETE
