# T6I High 0p480 Full Validation Summary

**Date:** 2026-06-13
**Profile:** T6I_phase_aware_release

## Staged Gates

| Gate | Steps | Survived | Max Abs | Final | Accumulation | Result |
|------|-------|----------|---------|-------|-------------|--------|
| 1200 | 1200 | ✅ | 0.2034m | 0.0311m | — | PASS_WITH_MONITORING |
| 2000 | 2000 | ✅ | 0.2122m | 0.0294m | 1.219 | PASS |
| 5000 | 5000 | ✅ | 0.2122m | 0.1309m | 1.051 | PASS |

## Key Drift Statistics (5000-step)

| Metric | Value |
|--------|-------|
| Min error | -0.0287m |
| Max error | +0.2122m |
| P2P | 0.2409m |
| Mean abs | 0.0962m |
| Outside ±0.08 | 53.6% |
| Outside ±0.10 | 46.7% |
| Outside ±0.15 | 29.2% |
| Positive drift | 95.6% |
| Zero crossings | 12 |

## T6I Mechanism Activation (5000-step)

| Metric | Value |
|--------|-------|
| Convergence active | 6.5% |
| Cap range | 4.0–7.0 Nm |
| Release: converging | 323 (6.5%) |
| Release: arch_fix_active | 2011 (40.2%) |
| Release: none | 2665 (53.3%) |

## Conclusion

T6I survives 5000 steps at high_0p480 with bounded, non-accumulating drift. The drift is one-sided positive and not centered at zero. T6I's phase-aware mechanism has limited activation (6.5%) and contributes modestly.
