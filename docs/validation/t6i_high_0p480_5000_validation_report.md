# T6I High 0p480 — 5000-step Validation Report

**Date:** 2026-06-13
**Profile:** T6I_phase_aware_release
**Setup:** high_0p480_setup.json (target CoM Z: 0.480m)
**Classification:** T6I_HIGH_0P480_5000_PASS_PROCEED_HEIGHT_LADDER

## Pass Criteria

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Survives 5000 steps | 4999 rows (5000 simulated) | ≥5000 | ✅ PASS |
| Max abs error < 0.25m | 0.2122m | <0.25 | ✅ PASS |
| Final error < 0.18m | 0.1309m | <0.18 | ✅ PASS |
| Accumulation ratio < 1.5 | 1.051 | <1.5 | ✅ PASS |
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
| Final error | 0.1309m |
| P2P drift | 0.2409m |
| Mean abs error | 0.0962m |
| Outside ±0.08 | 2681 (53.6%) |
| Outside ±0.10 | 2333 (46.7%) |
| Outside ±0.15 | 1458 (29.2%) |
| Positive % | 95.6% |
| Zero crossings | 12 |

## Stability Metrics

| Metric | Value |
|--------|-------|
| CoM Z min/mean/max | 0.459/0.483/0.492m |
| Pitch max/RMS | 8.58°/4.80° |
| Roll max/RMS | 0.28°/0.11° |
| Wheel vel max | 7.13 rad/s |
| Wheel vel >5/>6/>7 | 841/209/10 steps |

## T6I Phase-Aware Release

| Metric | Value |
|--------|-------|
| Convergence active % | 6.5% |
| Cap range | 4.0–7.0 Nm |
| Rate-limited steps | 97 |
| Release: none | 2665 (53.3%) |
| Release: arch_fix_active | 2011 (40.2%) |
| Release: converging | 323 (6.5%) |

## 500-Step Windows

| Window | Max Abs | MAE | OOB ±0.08 | OOB ±0.10 | OOB ±0.15 | Conv% | Cap Mean |
|--------|---------|-----|-----------|-----------|-----------|-------|----------|
| 0-500 | 0.2034 | 0.0821 | 45.2% | 39.2% | — | 4.8% | 5.11 |
| 500-1000 | 0.1977 | 0.1059 | 59.0% | 52.6% | — | 6.2% | 5.53 |
| 1000-1500 | 0.1944 | 0.0882 | 48.6% | 42.0% | — | 6.4% | 5.21 |
| 1500-2000 | 0.2122 | 0.0999 | 54.6% | 48.8% | — | 5.4% | 5.42 |
| 2000-2500 | 0.2058 | 0.0836 | 45.4% | 39.2% | — | 6.0% | 5.13 |
| 2500-3000 | 0.1940 | 0.1065 | 60.6% | 53.8% | — | 6.4% | 5.56 |
| 3000-3500 | 0.1894 | 0.0882 | 47.4% | 39.8% | — | 7.0% | 5.13 |
| 3500-4000 | 0.1863 | 0.1104 | 64.2% | 56.2% | — | 7.0% | 5.62 |
| 4000-4500 | 0.1812 | 0.0911 | 47.8% | 39.2% | — | 6.4% | 5.11 |
| 4500-5000 | 0.1764 | 0.1065 | 63.5% | 55.9% | — | 9.0% | 5.57 |

**Accumulation ratio (last1000/first1000):** 1.051 — well within bounds.
**Worst window by max_abs:** 1500-2000 (0.2122)
**Worst window by OOB ±0.10:** 3500-4000 (56.2%)

## Key Findings

1. **Drift is bounded, not accumulating.** Accumulation ratio 1.051 is near-unity. No growth trend.
2. **Drift is one-sided positive.** 95.6% of error values are positive. The controller stabilizes at a positive offset (~0.08–0.15m) rather than centering around zero.
3. **Precision is limited.** 53.6% outside ±0.08m, 46.7% outside ±0.10m. The controller keeps drift bounded within 0.25m but does not tightly center it.
4. **T6I convergence activates more over time.** Convergence rises from 4.8% to 9.0% in the last window.
5. **Max abs error slightly decreasing in later windows.** From 0.2034 in 0-500 to 0.1764 in 4500-5000.
6. **CoM Z dips to 0.459m** during the run — below target 0.480m. This is a moderate height excursion but no fall.
7. **No premature release or secondary divergence detected.**

## Decision

All hard pass criteria met. Accumulation ratio is excellent (1.051). Drift is bounded and not growing. The primary concern is that drift is one-sided and not precise — T6I keeps the robot stable but does not tightly center the drift. This is acceptable for stability validation.

**Proceed to Phase 5 (Height Ladder).**
