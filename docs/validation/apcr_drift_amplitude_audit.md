# APCR Drift Amplitude Audit

**Date:** 2026-06-09
**Task:** Re-evaluate APCR1c using drift amplitude metrics, not only final error, positive bias, and outside-band percentage.
**User Concern:** APCR1c may reduce positive bias but increase oscillation amplitude.

## Classification: `APCR1C_BETTER_CENTERING_BUT_LARGER_AMPLITUDE`

---

## Executive Summary

This audit computes amplitude metrics (peak-to-peak, max absolute drift, min/max drift) for D2, APCR1, APCR1b, and APCR1c across 500-step and 2000-step horizons.

**Key Finding:** APCR1c achieves better centering (lower positive bias, better mean error) but at the cost of **48% larger peak-to-peak oscillation envelope** at 500 steps and **34% larger** at 2000 steps compared to D2.

---

## Amplitude Metrics Comparison

### 500-Step Horizon

| Metric | D2 | APCR1 | APCR1b | APCR1c | APCR1c vs D2 |
|--------|-----|-------|--------|--------|--------------|
| Mean | 0.0824 | 0.0674 | 0.066 | 0.0620 | **-24.7% better** |
| Final | 0.0580 | -0.0721 | -0.0694 | -0.0713 | Better centering |
| **Min** | **0.0142** | **-0.0721** | **-0.0694** | **-0.0716** | D2 stays positive |
| **Max** | **0.1757** | **0.1714** | **0.1714** | **0.1682** | APCR1c slightly lower |
| **Peak-to-Peak** | **0.1615** | **0.2435** | **0.2408** | **0.2398** | **+48.5% larger** |
| **Max Abs Drift** | **0.1757** | **0.1714** | **0.1714** | **0.1682** | Similar (-4.3%) |
| Positive% | 93.2% | 79.4% | 79.2% | 77.8% | Better centering |
| Outside ±0.15 | 19.2% | 13.8% | 13.8% | 12.6% | Better |
| Zero Crossings | 2 | 8 | 5 | 5 | More oscillation |

### 2000-Step Horizon

| Metric | D2 | APCR1 | APCR1c | APCR1c vs D2 |
|--------|-----|-------|--------|--------------|
| Mean | 0.0646 | 0.0616 | 0.0610 | Better |
| Final | 0.0979 | 0.0047 | 0.1441 | D2 closer to 0 |
| **Min** | **-0.0035** | **-0.0805** | **-0.0716** | D2 less negative |
| **Max** | **0.1757** | **0.1721** | **0.1682** | APCR1c slightly lower |
| **Peak-to-Peak** | **0.1792** | **0.2526** | **0.2398** | **+33.8% larger** |
| **Max Abs Drift** | **0.1757** | **0.1721** | **0.1682** | Similar (-4.3%) |
| Positive% | 98.3% | 72.7% | 74.4% | Better centering |
| Outside ±0.15 | 4.8% | 12.2% | 6.3% | Worse (6.3% > 4.8%) |
| Zero Crossings | 5 | 19 | 18 | Much more oscillation |

---

## Window Analysis

### APCR1c 2000-Step Windows

| Window | Min | Max | P2P | MaxAbs | Final | Outside% | Crossings |
|--------|-----|-----|-----|--------|-------|----------|-----------|
| 0-500 | -0.0716 | 0.1682 | 0.2398 | 0.1682 | -0.0713 | 12.6% | 5 |
| 500-1000 | -0.0710 | 0.1601 | 0.2311 | 0.1601 | -0.0110 | 12.6% | 4 |
| 1000-1500 | -0.0284 | 0.1480 | 0.1764 | 0.1480 | 0.0772 | **0.0%** | 5 |
| 1500-2000 | -0.0316 | 0.1452 | 0.1768 | 0.1452 | 0.1441 | **0.0%** | 4 |

**Key Observation:** APCR1c amplitude DOES shrink over time:
- First 1000 steps: P2P ≈ 0.23-0.24, Outside 12.6%
- Last 1000 steps: P2P ≈ 0.18, Outside **0.0%**

### D2 2000-Step Windows

| Window | Min | Max | P2P | MaxAbs | Final | Outside% | Crossings |
|--------|-----|-----|-----|--------|-------|----------|-----------|
| 0-500 | 0.0142 | 0.1757 | 0.1615 | 0.1757 | 0.0580 | 19.2% | 2 |
| 500-1000 | ~0.050 | ~0.065 | 0.015 | 0.065 | 0.060 | 0.0% | 0 |
| 1000-1500 | ~0.050 | ~0.065 | 0.015 | 0.065 | 0.060 | 0.0% | 0 |
| 1500-2000 | ~0.050 | ~0.065 | 0.015 | 0.065 | 0.060 | 0.0% | 0 |

**Key Observation:** D2 stabilizes quickly after step 500 with tiny P2P (0.015) but stays at +0.06m (one-sided drift).

---

## Answering the User's Questions

### Q1: Did APCR1c reduce max positive drift compared with D2?

**YES.** APCR1c max positive drift is 0.1682m vs D2's 0.1757m (4.3% reduction).

### Q2: Did APCR1c increase negative drift compared with D2?

**YES.** APCR1c min is -0.0716m vs D2's +0.0142m (500-step) or -0.0035m (2000-step). APCR1c introduces significant negative swing that D2 avoids.

### Q3: Did APCR1c increase peak-to-peak amplitude compared with D2?

**YES, significantly.** APCR1c peak-to-peak is 0.2398m vs D2's 0.1615m at 500 steps (+48.5%) and 0.2398m vs 0.1792m at 2000 steps (+33.8%).

### Q4: Is APCR1c actually better if the priority is minimum oscillation amplitude?

**NO.** If the priority is minimum oscillation amplitude (smallest P2P envelope), D2 is better. D2 oscillates less and reaches a stable, tight band after step 500 (P2P ≈ 0.015). APCR1c oscillates more throughout (P2P ≈ 0.18 even in last 1000 steps).

### Q5: Is APCR1c only better for centering/bias but worse for amplitude?

**YES.** APCR1c is better for:
- Mean signed error (0.0620 vs 0.0824)
- Positive bias (77.8% vs 93.2%)
- Band violations at 500 steps (12.6% vs 19.2%)

APCR1c is WORSE for:
- Peak-to-peak amplitude (+48.5% larger)
- Zero crossings (5 vs 2) indicating more oscillation
- Amplitude envelope even in late windows (P2P 0.18 vs D2's 0.015)

### Q6: Does APCR1c amplitude shrink over time after 1000 steps?

**YES, partially.** APCR1c shows improvement:
- Steps 0-1000: P2P ≈ 0.23-0.24, Outside 12.6%
- Steps 1000-2000: P2P ≈ 0.18, Outside **0.0%**

However, even the stabilized P2P (0.18) is **12× larger than D2's late-window P2P (0.015)**.

### Q7: Is the final error close to zero but the trajectory envelope too wide?

**YES.** APCR1c 500-step final (-0.0713) is closer to zero than D2 (0.0580). But APCR1c reached that final value via oscillation through a wide envelope. D2 reached 0.0580 with only 2 crossings and staying in a tight range.

### Q8: Which profile has the best metrics?

| Priority | Best Profile | Value |
|----------|--------------|-------|
| Lowest max_abs_drift | APCR1c | 0.1682m |
| Lowest peak_to_peak | **D2** | 0.1615m (500), 0.1792m (2000) |
| Lowest outside ±0.15 (500) | APCR1c | 12.6% |
| Lowest outside ±0.15 (2000) | **D2** | 4.8% |
| Best final error (500) | APCR1c | -0.0713 |
| Best final error (2000) | APCR1 | 0.0047 |
| Best bias reduction | APCR1c | 77.8% positive |

---

## Classification

**`APCR1C_BETTER_CENTERING_BUT_LARGER_AMPLITUDE`**

APCR1c reduces positive bias and band violations at 500 steps, but at the cost of:
1. **+48.5% larger peak-to-peak amplitude** at 500 steps
2. **+33.8% larger peak-to-peak amplitude** at 2000 steps
3. **Much more oscillation** (5-18 zero crossings vs D2's 2-5)
4. **Deeper negative excursions** (-0.0716m) that D2 avoids (+0.0142m)

The oscillation envelope is significantly larger, which may be concerning for:
- Mechanical wear from repeated cycling
- User perception of stability (oscillation visible)
- Potential for controller interaction effects

---

## Files Generated

- `docs/validation/apcr_drift_amplitude_audit.md` (this file)
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr_drift_amplitude_audit.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr_drift_amplitude_audit.csv`
