# K2 Correct PARTIAL — Full Matrix Report

**Date:** 2026-06-30
**Phase:** 1 — COMPLETE FULL MATRIX WITH CORRECTED METRICS
**Output:** `outputs/k2_correct_partial_pitch_validation/`

---

## 1. Final Scorecard

| Scope | Scenarios | PASS | SAFE_BUT_WORSE | SAFETY_FAIL |
|-------|-----------|------|----------------|-------------|
| Step C | 7 | 6 | 1 | 0 |
| Step E | 10 | 6 | 4 | 0 |
| Step D | 12 | 12 | 0 | 0 |
| Dynamic | 5 | 2 | 3 | 0 |
| Long-Run | 5 | 2 | 3 | 0 |
| **Total** | **39** | **28** | **11** | **0** |

**Final Classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`

---

## 2. Detailed Results

### Step C — Fixed-Height Dynamic (7 scenarios)

| Scenario | Result | Key |
|----------|--------|-----|
| C1_slow_ladder_up_down | PASS | WITHIN_OLD_TOLERANCE |
| C2_random_500dwell | PASS | WITHIN_OLD_TOLERANCE |
| C3_random_200dwell | PASS | WITHIN_OLD_TOLERANCE |
| C4_abrupt_stress | PASS | WITHIN_OLD_TOLERANCE |
| C5_long_random | PASS | WITHIN_OLD_TOLERANCE |
| focused_low_0p320 | **SAFE_BUT_WORSE** | pitch_rms_deg |
| focused_high_0p480 | PASS | WITHIN_OLD_TOLERANCE |

### Step E — Fixed-Height Sweep (10 scenarios)

| Scenario | Result | Pitch RMS |
|----------|--------|-----------|
| low_0p300 | PASS | WITHIN_OLD_TOLERANCE |
| low_0p320 | **SAFE_BUT_WORSE** | 3.69° vs 2.83° (+0.86°) |
| low_0p330 | PASS | WITHIN_OLD_TOLERANCE |
| low_0p340 | PASS | WITHIN_OLD_TOLERANCE |
| low_0p360 | **SAFE_BUT_WORSE** | 3.12° vs 1.90° (+1.22°) |
| low_0p380 | **SAFE_BUT_WORSE** | 5.24° vs 3.33° (+1.91°) |
| high_0p430 | PASS | WITHIN_OLD_TOLERANCE |
| high_0p450 | **SAFE_BUT_WORSE** | 4.68° vs 2.75° (+1.93°) |
| high_0p465 | PASS | WITHIN_OLD_TOLERANCE |
| high_0p480 | PASS | WITHIN_OLD_TOLERANCE |

### Step D — Push Matrix (12 scenarios)

ALL 12 PASS (WITHIN_OLD_TOLERANCE). Zero SAFE_BUT_WORSE. Zero SAFETY_FAIL.
Post-push 500-step windows, corrected hip-yaw baseline, support RMS computed.

### Dynamic Height (5 scenarios)

| Scenario | Result | Notes |
|----------|--------|-------|
| ramp_up_0p330_to_0p480 | PASS | Survives, correct q_ref mode |
| ramp_down_0p480_to_0p330 | PASS | Survives, correct q_ref mode |
| up_down_cycle | **SAFE_BUT_WORSE** | pitch_rms_deg |
| gate_dwell | **SAFE_BUT_WORSE** | pitch_rms_deg |
| gate_chatter | **SAFE_BUT_WORSE** | pitch_rms_deg |

**All 5 scenarios survive** — 0 falls, 0 SAFETY_FAIL.

### Long-Run (5 scenarios)

| Scenario | Result |
|----------|--------|
| low_0p330 | **SAFE_BUT_WORSE** |
| mid_0p400 | PASS |
| high_0p430 | **SAFE_BUT_WORSE** |
| high_0p450 | **SAFE_BUT_WORSE** |
| high_0p480 | PASS |

---

## 3. Critical Milestones Confirmed

| Check | Status |
|-------|--------|
| Step D 12/12 PASS | ✅ |
| Dynamic all 5 survive | ✅ (0 falls) |
| Zero SAFETY_FAIL in any scope | ✅ |
| Hip-yaw EXACT_OR_BETTER | ✅ |
| Performance ≥120 Hz | ✅ |
| Step C C5_long_random PASS | ✅ (was SAFE_BUT_WORSE in old run) |

---

## 4. Remaining SAFE_BUT_WORSE Cases (11 total)

All SAFE_BUT_WORSE are caused by `pitch_rms_deg` exceeding the tolerance of `min(1.0°, 0.3 × original)`.

| # | Scope | Scenario | Pitch Delta |
|---|-------|----------|-------------|
| 1 | Step C | focused_low_0p320 | +0.86° |
| 2 | Step E | low_0p320 | +0.86° |
| 3 | Step E | low_0p360 | +1.22° |
| 4 | Step E | low_0p380 | +1.91° |
| 5 | Step E | high_0p450 | +1.93° |
| 6 | Dynamic | up_down_cycle | +0.60° |
| 7 | Dynamic | gate_dwell | +3.14° |
| 8 | Dynamic | gate_chatter | +1.76° |
| 9 | Long-Run | low_0p330 | +1.10° |
| 10 | Long-Run | high_0p430 | varies |
| 11 | Long-Run | high_0p450 | +1.10° |

---

## 5. Reproduction

```bash
# Re-run full matrix
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_correct_partial_pitch_validation

# Classify
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --classify-only \
  --output-dir outputs/k2_correct_partial_pitch_validation
```
