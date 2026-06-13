# T6I High 0p480 — 2000-step Validation Report

**Date:** 2026-06-13
**Profile:** T6I_phase_aware_release
**Setup:** high_0p480_setup.json (target CoM Z: 0.480m)
**Classification:** T6I_HIGH_0P480_2000_PASS_PROCEED_5000

## Pass Criteria

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Survives 2000 steps | 1999 rows (2000 simulated) | ≥2000 | ✅ PASS |
| Max abs error < 0.25m | 0.2122m | <0.25 | ✅ PASS |
| Final error < 0.18m | 0.0294m | <0.18 | ✅ PASS |
| No drift accumulation | ratio=1.219 | <1.5 | ✅ PASS |
| Contact stable | L=1.000, R=1.000 | >0.99 | ✅ PASS |
| No WBC | — | — | ✅ PASS |
| No hidden torque | — | — | ✅ PASS |
| No ownership violation | — | — | ✅ PASS |

## Overall Drift Metrics

| Metric | Value |
|--------|-------|
| Min error | -0.0287m |
| Max error | 0.2122m |
| Max abs error | 0.2122m |
| Final error | 0.0294m |
| P2P drift | 0.2409m |
| Mean abs error | 0.0941m |
| Outside ±0.08 | 1037 (51.9%) |
| Outside ±0.10 | 913 (45.7%) |
| Outside ±0.15 | 601 (30.1%) |
| Positive % | 91.7% |
| Zero crossings | 8 |

## 500-Step Windows

| Window | Max Abs | Final | MAE | OOB ±0.08 | OOB ±0.10 | OOB ±0.15 | Conv% | Cap Mean |
|--------|---------|-------|-----|-----------|-----------|-----------|-------|----------|
| 0-500 | 0.2034 | 0.1440 | 0.0821 | 45.2% | 39.2% | 24.2% | 4.8% | 5.11 |
| 500-1000 | 0.1977 | 0.0305 | 0.1059 | 59.0% | 52.6% | 36.4% | 6.2% | 5.53 |
| 1000-1500 | 0.1944 | 0.1623 | 0.0882 | 48.6% | 42.0% | 25.2% | 6.4% | 5.21 |
| 1500-2000 | 0.2122 | 0.0294 | 0.1001 | 54.7% | 48.9% | 34.5% | 5.4% | 5.43 |

**Accumulation ratio (last500/first500):** 1.219 — PASS (<1.5), preferred <1.2 nearly met.

## Stability Metrics

| Metric | Value |
|--------|-------|
| CoM Z min/mean/max | 0.481/0.490/0.492m |
| Pitch max/RMS | 8.58°/4.76° |
| Roll max/RMS | 0.28°/0.14° |
| Wheel vel max/RMS | 7.13/3.94 rad/s |
| Wheel vel >5/>6/>7 | 494/161/10 steps |

## T6I Phase-Aware Release

| Metric | Value |
|--------|-------|
| Convergence active % | 5.7% |
| Cap range | 4.0-7.0 Nm |
| Rate-limited steps | 40 |
| Release: none | 1086 (54.3%) |
| Release: arch_fix_active | 799 (40.0%) |
| Release: converging | 114 (5.7%) |

## Observations

1. **Drift is bounded and stable.** Max abs error (0.2122m) is below threshold. No accumulation trend.
2. **Accumulation ratio 1.219** is within pass criteria (<1.5) and close to preferred (<1.2).
3. **Oscillatory but bounded.** The error oscillates between positive peaks and near-zero troughs, with 8 zero crossings over 2000 steps.
4. **10 steps have wheel velocity >7 rad/s** — a minor concern but no instability.
5. **Convergence detection activates only 5.7%** — the T6I mechanism activates rarely at this height variant.

## Decision

All pass criteria met. Accumulation ratio is acceptable. Proceed to 5000-step validation.
