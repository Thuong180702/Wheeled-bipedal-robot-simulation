# T6I Height Ladder Validation Summary

**Date:** 2026-06-13
**Profile:** T6I_phase_aware_release
**Steps per setup:** 2000

## Results

| Setup | Target Height | Survived | Max Abs | Final | MAE | OOB ±0.10 | T6I Conv% | Classification |
|-------|--------------|----------|---------|-------|-----|-----------|-----------|----------------|
| low_0p300 | 0.30m | ✅ | 0.1715 | 0.0486 | 0.0590 | 5.0% | 0.0% | PASS |
| low_0p320 | 0.32m | ✅ | 0.1593 | 0.0186 | 0.0581 | 11.2% | 0.0% | PASS |
| low_0p330 | 0.33m | ✅ | 0.1858 | -0.006 | 0.0743 | 27.3% | 0.0% | PASS |
| low_0p340 | 0.34m | ✅ | 0.1290 | -0.024 | 0.0475 | 8.1% | 0.0% | PASS |
| low_0p360 | 0.36m | ✅ | 0.1500 | -0.039 | 0.0571 | 12.9% | 0.0% | PASS |
| low_0p380 | 0.38m | ✅ | 0.2505 | 0.079 | 0.108 | 48.0% | 0.0% | MARGINAL |
| high_0p430 | 0.43m | ✅ | 0.1514 | 0.022 | 0.0611 | 20.0% | 0.0% | PASS |
| high_0p450 | 0.45m | ✅ | 0.2042 | 0.011 | 0.0925 | 45.2% | 6.7% | PASS |
| high_0p465 | 0.47m | ✅ | 0.1987 | 0.107 | 0.0845 | 40.2% | 5.5% | PASS |

## low_0p300 5000-step Regression

| Metric | Value | Result |
|--------|-------|--------|
| Max abs error | 0.1715m | ✅ PASS |
| Accumulation ratio | 0.865 | ✅ IMPROVING |
| Final error | 0.0393m | ✅ PASS |

## Summary

- **8 of 9 setups pass** cleanly
- **low_0p380**: marginal 0.0005m transient overshoot during early convergence (19 steps), self-corrects
- **No falls, no WBC, no hidden torque, no ownership violations** in any setup
- **T6I convergence activates only at higher heights** (0.45m+), not at low heights where base controller suffices
- **No low_0p300 regression** over 5000 steps; drift actually improves over time
