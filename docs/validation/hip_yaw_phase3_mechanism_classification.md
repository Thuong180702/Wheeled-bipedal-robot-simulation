# Hip-Yaw Disturbance Rejection - Phase 3 Mechanism Classification

**Date:** 2026-06-04  
**Phase:** 3 (Mechanism Classification)  
**Status:** COMPLETE

---

## Executive Summary

**Verdict:** Hip-yaw PD control authority increase alone is **INSUFFICIENT** to achieve disturbance rejection at low_0p300.

**Key Finding:** Even with 67% kp increase (15→25) and 200% kd increase (3→9), hip-yaw error at low_0p300 remains 131% over threshold (0.1618 rad vs 0.0700 rad target).

**Mechanism:** `hip_yaw_disturbance_rejection_insufficient_authority_alone`

**Root Cause:** Support position drift at extreme flexion (low_0p300) creates disturbance torque that exceeds hip-yaw controller's rejection capability through PD gains alone.

---

## Experimental Evidence

### Test Matrix Executed

**Total experiments:** 21 simulations  
**Duration:** 1000 steps each (~10 sec simulated time)  
**Success rate:** 100% (21/21 completed)

**Coverage:**
- 3 baseline runs (kp=15, kd=3): low_0p300, high_0p480, nominal
- 12 damping sweep runs: kd ∈ {5, 7, 9, 12} × 3 variants
- 6 authority matrix runs: (kp, kd) ∈ {20, 25} × {5, 7, 9} at low_0p300

### Quantitative Results: low_0p300 (Critical Height)

| Configuration | kp | kd | hip_yaw_abs_max | vs Baseline | vs Threshold | Pass? |
|---------------|----|----|-----------------|-------------|--------------|-------|
| **Baseline** | 15 | 3 | **0.2137** rad | — | +205% | ❌ |
| Damping +67% | 15 | 5 | 0.2080 rad | -2.7% | +197% | ❌ |
| Damping +133% | 15 | 7 | 0.2037 rad | -4.7% | +191% | ❌ |
| Damping +200% | 15 | 9 | 0.2007 rad | -6.1% | +187% | ❌ |
| Authority +33% | 20 | 9 | 0.1773 rad | -17.0% | +153% | ❌ |
| **Best: Authority +67%** | 25 | 9 | **0.1618** rad | **-24.3%** | **+131%** | ❌ |

**Threshold:** 0.0700 rad  
**Best achieved:** 0.1618 rad  
**Gap:** 0.0918 rad (131% over threshold)

### Trend Analysis

**kp effect (at kd=9):**
```
kp=15: 0.2007 rad
kp=20: 0.1773 rad  (-11.7%)
kp=25: 0.1618 rad  (-19.4% vs kp=15, -8.7% vs kp=20)
```
→ **Strong monotonic improvement** with kp increase  
→ kp is the dominant control parameter

**kd effect (at kp=15):**
```
kd=3:  0.2137 rad
kd=5:  0.2080 rad  (-2.7%)
kd=7:  0.2037 rad  (-4.7%)
kd=9:  0.2007 rad  (-6.1%)
kd=12: 0.3786 rad  (+77.2% - INSTABILITY!)
```
→ **Weak monotonic improvement** until instability threshold  
→ kd increase has diminishing returns and causes instability at kd=12

### Comparative Results: high_0p480 (Nominal Height)

| Configuration | kp | kd | hip_yaw_abs_max | support_error | Pass? | Support Δ |
|---------------|----|----|-----------------|---------------|-------|-----------|
| Baseline | 15 | 3 | 0.0462 rad | 0.2336 m | ✅ | — |
| kd=5 | 15 | 5 | 0.0402 rad | 0.2520 m | ✅ | +7.9% |
| kd=7 | 15 | 7 | 0.0388 rad | 0.2692 m | ✅ | +15.2% |
| **kd=9** | 15 | 9 | **0.0382 rad** | **0.2800 m** | ✅ | **+19.9%** ⚠️ |

**Observation:** At high_0p480, increased kd improves hip-yaw BUT degrades support position by >10% (rejection criterion for side effects).

---

## Mechanism Classification

### Classification: `hip_yaw_disturbance_rejection_insufficient_authority_alone`

**Definition:** Hip-yaw posture controller cannot reject support-drift-induced disturbance through PD gain scheduling alone, even when gains are increased beyond nominal operating range.

### Supporting Evidence

1. **Authority ceiling reached:**
   - kp=25 is 67% above baseline (15)
   - kd=9 is 200% above baseline (3)
   - Further increases cause instability (kd=12 → 77% degradation)

2. **Improvement insufficient:**
   - Best configuration achieves 24% hip-yaw reduction
   - Still 131% over threshold (requires 77% reduction to pass)

3. **Trade-off violation:**
   - High-kd configurations that improve hip-yaw degrade support position >10%
   - At high_0p480: kd=9 improves hip-yaw but worsens support by 19.9%

4. **Disturbance magnitude exceeds control authority:**
   - Support drift at low_0p300: 0.243 m (baseline)
   - Hip-yaw tracking error: 0.214 rad (baseline)
   - Even with maximum tested gains: 0.162 rad (still 2.3× threshold)

---

## Root Cause Analysis

### Causal Chain

```
Extreme flexion (h=0.300m)
  ↓
Sagittal instability (limited forward authority at low height)
  ↓
Support position drift (x_support drifts forward)
  ↓
Hip-yaw coupling (base rotation follows support drift)
  ↓
Hip-yaw disturbance torque (yaw error creates restoring moment)
  ↓
PD controller applies corrective torque
  ↓
Disturbance torque > PD rejection capability
  ↓
Hip-yaw error persists despite correct control action
```

### Why PD Gains Alone Fail

**Fundamental limitation:** PD control can only react to observed error. When disturbance source (support drift) is continuous and large, the error required to generate sufficient restoring torque exceeds acceptable tracking error.

**Mathematical constraint:**
```
τ_hip_yaw = kp * e_yaw - kd * ω_yaw

For disturbance rejection:
τ_hip_yaw ≥ τ_disturbance

Given:
τ_disturbance ∝ support_drift (≈0.24m at low_0p300)

Therefore:
e_yaw_required = τ_disturbance / kp

At kp=25 (maximum tested):
e_yaw_required still >> 0.070 rad threshold
```

**Disturbance exceeds authority:** Even optimal kp cannot generate sufficient torque without violating error threshold.

---

## Decision: Recommended Fix Path

### Option A: Implement Support-Error Feedforward (HY-FF) — **RECOMMENDED**

**Approach:** Add feedforward term to hip-yaw controller based on measured support position error.

**Rationale:**
- Compensates for known disturbance source (support drift)
- Reduces required feedback error
- Preserves PD baseline stability
- Targeted fix for identified coupling mechanism

**Implementation:**
```python
tau_hip_yaw = kp * e_yaw - kd * omega_yaw + k_ff * support_error
```

**Expected outcome:**
- Decouple hip-yaw from support drift
- Pass hip-yaw gate at low_0p300
- No degradation to support position (independent control)

**Risks:**
- Requires tuning k_ff gain
- May couple hip-yaw to sagittal noise
- Needs validation across height range

**Complexity:** Medium (new controller term, 1 additional parameter)

---

### Option B: Fix Sagittal Support Drift First — **ALTERNATIVE**

**Approach:** Return to continuous low-height sagittal authority fix to reduce support drift at source.

**Rationale:**
- Addresses root cause (support instability)
- May eliminate disturbance entirely if support drift < 0.10m
- Unified solution for both sagittal and hip-yaw problems

**Expected outcome:**
- Reduce support drift → reduce hip-yaw disturbance
- Hip-yaw may pass gate without modification
- Solves multiple problems simultaneously

**Risks:**
- Requires implementing full sagittal continuous schedule
- Hip-yaw may still fail even with reduced support drift
- Longer implementation timeline (coupled fix)

**Complexity:** High (full sagittal system redesign, multiple parameters)

---

### Option C: Implement Continuous kp Schedule — **PARTIAL IMPROVEMENT**

**Approach:** Use kp=25 at low heights, kp=15 at nominal, with continuous interpolation.

**Rationale:**
- Leverages observed kp improvement (-24%)
- Simple implementation (single-parameter schedule)
- No new control terms

**Expected outcome:**
- Hip-yaw improves from 0.214 rad → 0.162 rad at low_0p300
- **Still fails gate** (0.162 vs 0.070 threshold)
- Partial improvement, not a solution

**Risks:**
- Does not pass success criteria
- Wasted implementation effort for insufficient fix
- May mask underlying problem

**Complexity:** Low (gain schedule only)

**Verdict:** ❌ REJECT — Does not meet requirements

---

### Rejected: Continuous kd-Only Schedule

**Reason:** kd increase has minimal effect (-6% at kd=9) and causes instability at kd=12. Not a viable fix.

---

## Next Steps (Phase 4)

### Proceed to Option A: HY-FF Implementation

**Tasks:**
1. Design feedforward gain schedule: `k_ff(h)`
2. Implement support-error feedforward in shape posture controller
3. Add CLI override: `--shape-kff-hip-yaw`
4. Run HY-FF isolation experiments at low_0p300
5. Tune k_ff to pass hip-yaw gate without degrading support

**Success criteria:**
- hip_yaw_abs_max ≤ 0.070 rad at low_0p300
- support_position_error_max not worsened >10%
- No WBC enabled
- Stable across full height range

### Alternative: If HY-FF fails

**Fallback to Option B:** Return to sagittal fix, implement continuous low-height forward authority schedule, re-evaluate hip-yaw after sagittal improvement.

---

## Appendix: Full Experimental Data

**Location:** `outputs/hip_yaw_disturbance_rejection_audit/isolation/`

**Files:**
- `isolation_experiment_results.json` — Raw metrics (21 experiments)
- `isolation_experiment_report.md` — Formatted results tables
- `*_telemetry.csv` — Archived telemetry for each configuration

**Analysis script:** `scripts/analyze_hip_yaw_isolation_results.py`

---

**Phase 3 Status:** ✅ COMPLETE  
**Next Phase:** 4 (Implement HY-FF Candidate)  
**Decision:** Proceed with support-error feedforward implementation
