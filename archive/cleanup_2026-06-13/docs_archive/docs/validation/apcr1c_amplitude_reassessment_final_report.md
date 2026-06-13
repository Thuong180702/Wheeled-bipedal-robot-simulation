# APCR1c Amplitude Reassessment Final Report

**Date:** 2026-06-09
**Task:** Re-evaluate APCR1c using drift amplitude metrics
**User Concern:** APCR1c may reduce positive bias but increase oscillation amplitude
**Classification:** `APCR1C_BETTER_CENTERING_BUT_LARGER_AMPLITUDE`

---

## Executive Summary

This report reassesses APCR1c using comprehensive amplitude metrics (peak-to-peak, max absolute drift, min/max drift) beyond the original metrics of final error, positive bias, and outside-band percentage.

**Primary Finding:** APCR1c achieves better centering (lower mean error, reduced positive bias) but at the cost of **48.5% larger peak-to-peak oscillation envelope** at 500 steps compared to D2. The oscillation envelope remains significantly larger than D2 even after APCR1c stabilizes.

---

## Key Question Answers

### Q1: Did APCR1c reduce max positive drift compared with D2?

**YES.** APCR1c max positive drift is 0.1682m vs D2's 0.1757m (4.3% reduction).

| Horizon | D2 Max | APCR1c Max | Reduction |
|---------|--------|------------|----------|
| 500-step | 0.1757m | 0.1682m | -4.3% |
| 2000-step | 0.1757m | 0.1682m | -4.3% |

### Q2: Did APCR1c increase negative drift compared with D2?

**YES.** APCR1c introduces significant negative drift that D2 avoids.

| Horizon | D2 Min | APCR1c Min | Assessment |
|---------|--------|------------|-----------|
| 500-step | +0.0142m | -0.0716m | APCR1c negative |
| 2000-step | -0.0035m | -0.0716m | APCR1c 20× deeper |

D2 stays near zero or slightly positive. APCR1c oscillates to -0.07m, which is concerning.

### Q3: Did APCR1c increase peak-to-peak amplitude compared with D2?

**YES, significantly.** This is the core finding of this audit.

| Horizon | D2 P2P | APCR1c P2P | Increase |
|---------|--------|------------|----------|
| 500-step | 0.1615m | 0.2398m | **+48.5%** |
| 2000-step | 0.1792m | 0.2398m | **+33.8%** |

APCR1c's oscillation envelope is 1.3-1.5× larger than D2's.

### Q4: Is APCR1c actually better if the priority is minimum oscillation amplitude?

**NO.** If the priority is minimum oscillation amplitude (smallest P2P envelope), D2 is clearly better.

| Metric | D2 | APCR1c | Winner |
|--------|-----|--------|--------|
| Peak-to-peak (500) | 0.1615m | 0.2398m | **D2** |
| Peak-to-peak (2000) | 0.1792m | 0.2398m | **D2** |
| Late window P2P | ~0.015m | ~0.180m | **D2** (12× tighter) |
| Zero crossings | 2-5 | 5-18 | **D2** (less oscillation) |

### Q5: Is APCR1c only better for centering/bias but worse for amplitude?

**YES.** APCR1c is better for centering/bias metrics but worse for amplitude metrics.

**APCR1c BETTER metrics:**
- Mean signed error: 0.0620 vs D2's 0.0824 (-24.7%)
- Positive bias: 77.8% vs D2's 93.2% (-15.4pp)
- Outside ±0.15 at 500 steps: 12.6% vs 19.2% (-34%)

**APCR1c WORSE metrics:**
- Peak-to-peak amplitude: +48.5% larger
- Negative drift introduced: -0.0716m vs +0.0142m
- Zero crossings: 5 vs 2 (more oscillation)
- Late-window amplitude: 0.18m vs D2's 0.015m (12× larger)

### Q6: Does APCR1c amplitude shrink over time after 1000 steps?

**YES, but insufficiently.**

| Window | APCR1c P2P | APCR1c Outside | D2 Late P2P |
|--------|------------|----------------|--------------|
| 0-500 | 0.2398m | 12.6% | - |
| 500-1000 | 0.2311m | 12.6% | - |
| 1000-1500 | 0.1764m | **0.0%** | 0.015m |
| 1500-2000 | 0.1768m | **0.0%** | 0.015m |

APCR1c does stabilize in the last 1000 steps (0% outside band), but P2P remains at 0.18m, which is still **12× larger than D2's stabilized P2P of 0.015m**.

### Q7: Is the final error close to zero but the trajectory envelope too wide?

**YES.** APCR1c reaches a final error closer to zero at 500 steps (-0.0713 vs 0.0580), but the trajectory to reach that point goes through a wide oscillation envelope.

At 2000 steps, APCR1c's final error (0.1441) is actually worse than D2's (0.0979).

### Q8: Which profile has the best metrics overall?

| Metric | Best Profile | Value | Notes |
|--------|-------------|-------|-------|
| Lowest max_abs_drift | APCR1c | 0.1682m | Slightly better than D2 |
| Lowest peak_to_peak | **D2** | 0.16-0.18m | Much better |
| Lowest outside ±0.15 (500) | APCR1c | 12.6% | Better |
| Lowest outside ±0.15 (2000) | **D2** | 4.8% | Better |
| Best final error (500) | APCR1c | -0.0713 | Closer to 0 |
| Best final error (2000) | APCR1 | 0.0047 | Closest to 0 |
| Best bias reduction | APCR1c | 77.8% positive | Best centering |

---

## Classification

**`APCR1C_BETTER_CENTERING_BUT_LARGER_AMPLITUDE`**

APCR1c reduces positive bias and band violations at 500 steps, but at the cost of:
1. **+48.5% larger peak-to-peak amplitude** at 500 steps
2. **+33.8% larger peak-to-peak amplitude** at 2000 steps
3. **Significant negative drift introduced** (-0.0716m)
4. **Much more oscillation** (5-18 zero crossings vs D2's 2-5)
5. **Late-window amplitude 12× larger** than D2's stabilized envelope

---

## Metric Priority Recommendation

Before running 5000-step validation, consider which metric matters most:

| Priority | Recommended Metric | Current Best | Notes |
|----------|-------------------|--------------|-------|
| If centering matters | positive%, mean error | APCR1c | Better bias reduction |
| If band violations matter (500) | outside ±0.15 | APCR1c | 12.6% vs 19.2% |
| If band violations matter (2000) | outside ±0.15 | **D2** | 4.8% vs 6.3% |
| If amplitude matters | peak-to-peak | **D2** | Much smaller envelope |
| If oscillation matters | zero crossings | **D2** | Fewer crossings |
| If final error matters (2000) | final error | **APCR1** | 0.0047 |

**If the primary goal is minimum oscillation amplitude with tight envelope, D2 is the better choice despite worse centering.**

**If the primary goal is balanced drift (centering), APCR1c is better but with larger amplitude cost.**

---

## Recommended Next Step

**Do NOT run 5000-step with APCR1c yet.**

Design APCR1d with **proportional torque shaping** to reduce oscillation amplitude while maintaining centering benefits:

| Parameter | APCR1c | APCR1d Target |
|-----------|--------|---------------|
| Torque mode | constant | proportional |
| Peak-to-peak target | 0.24m | < 0.20m |
| Max abs drift target | 0.17m | < 0.17m |

APCR1d validation plan:
1. 500-step smoke test at low_0p300
2. Compute amplitude metrics (P2P, MaxAbs, min/max)
3. Compare vs APCR1c - target P2P < 0.20m
4. If improved, proceed to 2000-step validation
5. If 2000-step shows P2P < 0.18m, proceed to 5000-step

---

## Final Decision

**Classification:** `APCR1C_BETTER_CENTERING_BUT_LARGER_AMPLITUDE`

**The "APCR1c is best" conclusion DOES CHANGE when amplitude is prioritized.**

APCR1c is better for:
- Centering (lower mean error, reduced positive bias)
- Short-horizon band violations (12.6% vs 19.2% at 500 steps)

APCR1c is WORSE for:
- Peak-to-peak oscillation (+48.5% larger at 500 steps)
- Late-window stability (P2P 0.18m vs D2's 0.015m)
- Long-horizon band violations (6.3% vs D2's 4.8% at 2000 steps)

**Recommendation:** Design APCR1d with proportional torque shaping before committing to 5000-step validation.

---

## Files Generated

| File | Description |
|------|-------------|
| `docs/validation/apcr_drift_amplitude_audit.md` | Detailed amplitude analysis |
| `docs/validation/apcr_amplitude_constrained_candidate_design.md` | APCR1d design options |
| `docs/validation/apcr1c_amplitude_reassessment_final_report.md` | This summary report |
| `outputs/.../apcr_drift_amplitude_audit.json` | JSON metrics data |
| `outputs/.../apcr_drift_amplitude_audit.csv` | CSV comparison table |
| `outputs/.../apcr_amplitude_constrained_candidate_design.json` | APCR1d design spec |
| `outputs/.../apcr1c_amplitude_reassessment_summary.json` | This summary in JSON |
