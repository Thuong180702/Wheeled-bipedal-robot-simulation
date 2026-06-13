# T5 High_0p480 Band Failure Audit

**Date:** 2026-06-12  
**Classification:** T5_HIGH_FAIL_MIXED_CAUSES

---

## Executive Summary

T5 band-limited balanced successfully survived 5000 steps at extreme height (0.480 m) with excellent stability, but **failed drift band targets** during windows 2-7 (steps 500-3500). Root cause analysis identifies **three contributing failures**:

1. **EMERGENCY_TOO_LATE** — Emergency band entered ~900 steps after |e| > 0.12 m threshold
2. **AUTHORITY_TOO_WEAK** — Emergency cap 7.0 Nm insufficient for extreme height transients
3. **DAMPING_TOO_STRONG** — Emergency damping scale 0.55 still preserves too much damping

---

## Threshold Crossing Analysis

| Threshold | First Crossing Step | Error (m) | Band State | Position Cap (Nm) | Damping Scale | Recenter Active |
|-----------|---------------------|-----------|------------|-------------------|---------------|-----------------|
| ±0.05 m   | 71                  | 0.0523    | normal     | 4.0               | 1.00          | No              |
| ±0.08 m   | 81                  | 0.0825    | normal     | 4.0               | 1.00          | No              |
| ±0.10 m   | 87                  | 0.1014    | normal     | 4.0               | 1.00          | No              |
| **±0.12 m** | **94**            | **0.1222** | **normal** | **4.0**         | **1.00**      | **No**          |
| ±0.15 m   | 106                 | 0.1509    | emergency  | 7.0               | 1.00          | Yes             |

**Critical Finding:** When |e| crossed the 0.12 m emergency threshold at step 94, T5 was still in **normal band** (4.0 Nm cap). Emergency band did not activate until step 1040 — **946 steps late**.

---

## Windows 2-7 Deep Dive (Steps 500-3500)

### Overall Statistics
- **Steps outside ±0.08 m:** 2055/3000 (68.5%)
- **Max |e|:** 0.1223 m
- **Mean |e|:** 0.0918 m

### Band State Distribution
| Band State | Steps | Percentage |
|------------|-------|------------|
| Hard       | 1095  | 36.5%      |
| Soft       | 785   | 26.2%      |
| Desired    | 737   | 24.6%      |
| Emergency  | 224   | 7.5%       |
| Normal     | 159   | 5.3%       |

**Key Finding:** Emergency band only active for **7.5%** of the problem window, despite sustained high drift.

### Emergency Band Performance
- **First emergency entry:** Step 1040 (946 steps after threshold crossed)
- **Emergency position cap:** 7.0 Nm
- **Emergency damping scale:** 0.55 (mean during emergency)

---

## Root Cause Analysis

### Cause 1: EMERGENCY_TOO_LATE

**Evidence:**
- Emergency threshold (±0.12 m) crossed at step 94 in normal band
- Emergency band first activated at step 1040
- **Delay: 946 steps** (~9.5 seconds)

**Impact:**
- Robot operated with 4.0 Nm cap while drift exceeded emergency threshold
- Insufficient authority to prevent drift escalation during critical early window

**T6 Implication:** Need earlier emergency entry or tighter thresholds at high height.

---

### Cause 2: AUTHORITY_TOO_WEAK

**Evidence:**
- Emergency position cap: 7.0 Nm
- Mean position cap during windows 2-7: 5.62 Nm
- Max position cap during windows 2-7: 7.00 Nm

**Context:**
- High_0p480 creates larger gravitational torque arm
- Pitch RMS during problem window: 4.063 deg (elevated)
- 7.0 Nm cap may be insufficient for extreme height transient suppression

**T6 Implication:** Consider 7.5-8.0 Nm emergency cap for high height.

---

### Cause 3: DAMPING_TOO_STRONG

**Evidence:**
- Emergency damping scale: 0.55 (mean during emergency)
- Minimum damping scale during windows 2-7: 0.10
- T5 preserves >50% of nominal damping even in emergency band

**Context:**
- T5 design prioritizes stability over immediate drift correction
- At extreme height, preserved damping may fight corrective wheel acceleration
- Emergency damping scale 0.55 is higher than T5 specification (0.10)

**Note:** Telemetry reports 0.55 mean during emergency, but T5 spec says emergency = 0.10. Need to verify whether emergency scale is properly applied at high height.

**T6 Implication:** Reduce emergency damping scale to 0.05 or less for high height.

---

## Window 7 vs Window 10 Comparison

### Window 7 (3000-3500) — WORST
- **Outside ±0.08 m:** 450/500 (90.0%)
- **Mean |e|:** 0.0976 m
- **Band state mode:** hard
- **Position cap mean:** 5.86 Nm
- **Damping scale mean:** 0.617
- **Pitch RMS:** 4.378 deg

### Window 10 (4500-5000) — BEST
- **Outside ±0.08 m:** 0/499 (0.0%)
- **Mean |e|:** 0.0378 m
- **Band state mode:** normal
- **Position cap mean:** 4.15 Nm
- **Damping scale mean:** 0.945
- **Pitch RMS:** 1.967 deg

### What Changed
- **Mean |e| reduction:** -0.0598 m (61% improvement)
- **Pitch RMS reduction:** -2.411 deg (55% improvement)

**Key Insight:** Recovery in window 10 coincided with **pitch reduction**. As pitch decreased, gravitational coupling weakened, allowing T5 to finally drive drift back down. This suggests pitch coupling is a significant contributor at extreme height.

---

## Implications for T6 Design

### Problem Statement
T5's graduated authority is **too slow and too weak** for extreme height (0.480 m) transients. The robot survives and eventually recovers, but drift exceeds targets for 3000 steps (steps 500-3500) before recovery begins.

### Root Causes Prioritized
1. **EMERGENCY_TOO_LATE** (highest priority)
2. **AUTHORITY_TOO_WEAK** (high priority)
3. **DAMPING_TOO_STRONG** (medium priority, pending spec verification)

### T6 Design Directions

**Option A: Earlier Emergency Entry**
- Tighten emergency threshold from 0.12 m to 0.10 m or 0.105 m
- Enter hard/desired bands earlier as well
- Rationale: Emergency authority must engage before drift reaches 0.12 m

**Option B: Stronger Emergency Authority**
- Increase emergency position cap from 7.0 Nm to 7.5-8.0 Nm
- Reduce emergency damping scale from 0.10 to 0.05 (if not already applied)
- Rationale: Emergency authority must overcome extreme height gravitational coupling

**Option C: Combined (Early + Stronger)**
- Combine A + B for maximum transient suppression
- Risk: May be too aggressive, could cause oscillation

**Option D: Transient-Only Boost**
- Apply Option C only during high-height transient window (steps 500-3500)
- Fallback to T5 after recovery
- Rationale: Avoid permanent over-aggressiveness

**Option E: Pitch-Aware Boost**
- Boost authority when pitch > threshold (e.g., 4 deg)
- Rationale: Pitch coupling is the driver at extreme height

---

## Next Steps (Phase 2)

1. Design 5 T6 variants (T6A-E) based on root cause analysis
2. Implement as opt-in profiles, preserve T5 unchanged
3. Run 2000-step high_0p480 screening for all T6 variants
4. Select best candidate based on drift band metrics
5. Run 5000-step validation for best candidate only

---

**Classification:** T5_HIGH_FAIL_MIXED_CAUSES  
**Date:** 2026-06-12  
**Phase:** 1 (Band Failure Audit) COMPLETE
