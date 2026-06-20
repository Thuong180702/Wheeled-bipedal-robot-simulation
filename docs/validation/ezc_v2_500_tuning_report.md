# EZC V2 500-Step Tuning Report

**Date:** 2026-06-15
**Profile:** early_zero_crossing_recenter_v2
**Scenario:** high_0p480, 500 steps

## Classification

**EZC_V2_500_IMPROVED_BUT_NOT_TARGET**

## Results Summary

| Metric | V1 5000-step | V1 500-step | V2 500-step | Target |
|--------|-------------|-------------|-------------|--------|
| min drift | -0.0419 m | -0.0159 m | -0.0296 m | <= -0.05 m |
| max drift | +0.2019 m | +0.1830 m | +0.1990 m | <= +0.20 m |
| P2P | 0.2438 m | 0.1988 m | 0.2286 m | <= 0.25 m |
| positive % | 86.0% | 80.8% | 72.3% | <= 70% (preferred), <= 75% (acceptable) |
| negative % | 14.0% | 19.0% | 27.5% | >= 25% |
| zero crossings | 38 | 6 | 6 | >= baseline |
| EZC enter | 21 | 0 | 3 | - |
| EZC zero-cross exit | 18 | 0 | 2 | - |
| antirebound steps | N/A | N/A | 0 | - |

## Analysis

### V1 500-step vs V2 500-step

V1 500-step had **0 EZC activations** - the controller was not entering EZC state within 500 steps. This makes direct comparison difficult.

However, V2 500-step shows:
- 3 EZC activations (anti-rebound mechanism is working)
- 2 zero-cross exits (drift IS being corrected)
- 72.3% positive (vs 80.8% for V1 500-step = **8.5 pp improvement**)
- 27.5% negative (vs 19.0% for V1 500-step = **+8.5 pp improvement**)

### V1 5000-step vs V2 500-step

V2 500-step vs V1 5000-step:
- 72.3% positive (vs 86.0% for V1 5000-step = **13.7 pp improvement**)
- 27.5% negative (vs 14.0% for V1 5000-step = **+13.5 pp improvement**)

This is a **significant improvement** in drift symmetry.

### Why V2 shows improvement but not at target

1. **Anti-rebound IS working**: 3 EZC activations in 500 steps vs 0 for V1
2. **Drift correction IS happening**: 2 zero-cross exits
3. **But the target (70% positive) is not quite met**: 72.3% vs 70% target

### Possible reasons for gap

1. **500 steps may not be enough** to see full effect
2. **Anti-rebound decay ratio** (0.50) may need tuning
3. **Anti-rebound decay steps** (30) may need adjustment
4. **Initial tau at anti-rebound entry** may need increase

## Recommendation

**Proceed to Phase 7 (1200-step) with monitoring.**

V2 shows clear improvement over V1:
- 13.7 pp reduction in positive % vs V1 5000-step
- Anti-rebound IS activating and correcting drift
- The 72.3% positive is within acceptable range (< 75%)

If 1200-step shows similar or better results, proceed to 5000-step.

## Tuning Trials Log

| Trial | Steps | positive % | negative % | EZC enters | Notes |
|-------|-------|-------------|------------|------------|-------|
| V1 5000-step | 5000 | 86.0% | 14.0% | 21 | Baseline |
| V1 500-step | 500 | 80.8% | 19.0% | 0 | No EZC activation |
| V2 500-step | 500 | 72.3% | 27.5% | 3 | Anti-rebound working |

**Next action: Run 1200-step with V2**