# T6I High 0p480 — 1200-step Validation Report

**Date:** 2026-06-13
**Profile:** T6I_phase_aware_release
**Setup:** high_0p480_setup.json (target CoM Z: 0.480m)
**Classification:** T6I_HIGH_0P480_1200_PASS_WITH_MONITORING

## Pass Criteria

| Criterion | Value | Threshold | Result |
|-----------|-------|-----------|--------|
| Survives 1200 steps | 1199 rows (1200 simulated) | ≥1200 | ✅ PASS |
| Max abs error < 0.25m | 0.2034m | <0.25 | ✅ PASS |
| Final error < 0.18m | 0.0311m | <0.18 | ✅ PASS |
| Contact stable | L=1.000, R=1.000 | >0.99 | ✅ PASS |
| No WBC | False | — | ✅ PASS |
| No hidden torque | 0.0 | 0 | ✅ PASS |
| No ownership violation | 0 | 0 | ✅ PASS |
| No premature release | None detected | — | ✅ PASS |
| No secondary divergence | None detected | — | ✅ PASS |

## Drift Metrics

| Metric | Value |
|--------|-------|
| Min error | -0.0160m |
| Max error | 0.2034m |
| Max abs error | 0.2034m |
| Final error | 0.0311m |
| P2P drift | 0.2194m |
| Mean abs error | 0.0944m |
| Outside ±0.08 | 627 (52.3%) |
| Outside ±0.10 | 552 (46.0%) |
| Outside ±0.15 | 363 (30.3%) |
| Positive % | 93.4% |
| Negative % | 6.5% |
| Zero crossings | 4 |

## Stability Metrics

| Metric | Value |
|--------|-------|
| CoM Z min/mean/max | 0.481/0.490/0.492m |
| Pitch max | 0.1466 rad (8.4°) |
| Pitch RMS | 0.0836 rad (4.8°) |
| Roll max | 0.0024 rad (0.14°) |
| Roll RMS | 0.0019 rad (0.11°) |
| Wheel vel max | 6.67 rad/s |
| Wheel vel RMS | 3.84 rad/s |
| Wheel vel >5 | 284 steps (23.7%) |
| Wheel vel >6 | 69 steps (5.8%) |
| Wheel vel >7 | 0 steps |

## T6I Phase-Aware Release

| Metric | Value |
|--------|-------|
| Convergence active % | 5.7% |
| Cap min/mean/max | 4.0/5.32/7.0 Nm |
| Cap decay active % | 5.7% |
| Rate-limit active | 27 steps (2.3%) |
| Release: none | 647 (54.0%) |
| Release: arch_fix_active | 484 (40.4%) |
| Release: converging | 68 (5.7%) |
| Arch fix active % | 46.0% |

## Monitoring Concerns

1. **One-sided drift:** 93.4% positive drift. The controller stabilizes to a positive offset rather than centering near zero.
2. **High outside-band %:** 52.3% outside ±0.08m, 46.0% outside ±0.10m. Drift is bounded but not precise.
3. **Low convergence activation:** Only 5.7% of steps had convergence detection active. The T6I phase-aware mechanism activates rarely at this height.
4. **Wheel velocity:** 23.7% of steps have wheel velocity >5 rad/s. No steps exceed 7 rad/s.

## Decision

All hard pass criteria met. Drift is bounded within 0.25m and final error is well within 0.18m. However, drift is one-sided and precision is limited. The result is stable enough to proceed to 2000-step validation with monitoring.

**Proceed to Phase 3 (2000-step).**
