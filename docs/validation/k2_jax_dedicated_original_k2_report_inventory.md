# K2 JAX Dedicated Realtime — Original K2 Promotion Report Inventory

**Date:** 2026-06-29
**Phase:** 0 — Locate Original K2 Promotion/Evaluation Reports
**Search scope:** `docs/validation/`, `outputs/`, `scripts/`, `tests/`

## Summary

93+ documents found across 11 tiers. The critical original K2 Python (non-JAX) promotion path consists of 4 reports that established the behavioral baseline the dedicated JAX runner must match.

---

## Tier 1: Original K2 Python (Non-JAX) Promotion Reports — THE REFERENCE BASELINE

These are the reports the dedicated JAX runner MUST be compared against.

| # | Report | Date | Controller | Final Classification | Scenarios |
|---|--------|------|------------|---------------------|-----------|
| 1 | [k2_notch_low_q_v1_create_and_validate_report.md](k2_notch_low_q_v1_create_and_validate_report.md) | 2026-06-25 | K2_NOTCH_LOW_Q_V1 (Python) | `K2_STRONG_PASS_READY_FOR_PROMOTION` | 7 scenarios: 3 heights × equilibrium/PRBS + 90N push |
| 2 | [k2_step_d_push_matrix_validation_report.md](k2_step_d_push_matrix_validation_report.md) | 2026-06-25 | K2_NOTCH_LOW_Q_V1 (Python) | `K2_STEP_D_STRONG_PASS_PROMOTE_READY` | 12 conditions: 3 heights × 2 dir × 2 magnitudes |
| 3 | [k2_step_c_e_validation_and_best_current_promotion_report.md](k2_step_c_e_validation_and_best_current_promotion_report.md) | 2026-06-25 | K2_NOTCH_LOW_Q_V1 (Python) | `K2_STEP_C_E_STRONG_PASS_PROMOTE_NOW` | Step C (7 cases) + Step E (10 heights) |
| 4 | [k2_post_promotion_long_run_and_dynamic_height_regression_report.md](k2_post_promotion_long_run_and_dynamic_height_regression_report.md) | 2026-06-25 | K2_NOTCH_LOW_Q_V1 (Python) | `K2_POST_PROMOTION_INVALID` / `K2_POST_PROMOTION_MIXED_KEEP_CURRENT_BEST_MONITOR` | 5 long-run (6000-step) + 5 dynamic height |

## Tier 2: K2 JAX Promotion Reports — Background Context

These document the JAX port parity work. The dedicated runner inherits from this work.

| # | Report | Date | Classification | Relevance |
|---|--------|------|----------------|-----------|
| 5 | [k2_jax_strict_clone_final_decision_report.md](k2_jax_strict_clone_final_decision_report.md) | 2026-06-27 | `K2_JAX_STRICT_CLONE_PROMOTION_PASS` | Torque parity verified |
| 6 | [k2_jax_default_promotion_final_report.md](k2_jax_default_promotion_final_report.md) | 2026-06-29 | `K2_JAX_DEFAULT_PROMOTION_PASS` | JAX is default backend, 164/164 tests |
| 7 | [k2_jax_full_semantic_port_release_hardening_final_report.md](k2_jax_full_semantic_port_release_hardening_final_report.md) | 2026-06-29 | `K2_JAX_FULL_SEMANTIC_PORT_RELEASE_HARDENING_PASS` | All mechanisms closed |

## Tier 3: Dedicated Realtime Runner Reports — What Exists Now

| # | Report | Date | Classification | Relevance |
|---|--------|------|----------------|-----------|
| 8 | [k2_jax_dedicated_realtime_runner_final_report.md](k2_jax_dedicated_realtime_runner_final_report.md) | 2026-06-29 | `K2_JAX_DEDICATED_REALTIME_RUNNER_PASS` | **Speed-only promotion (187.5 Hz), NOT behavioral equivalence** |
| 9 | [k2_jax_dedicated_promotion_scope.md](k2_jax_dedicated_promotion_scope.md) | 2026-06-29 | PENDING | Defines scope; gate #3 requires torque parity, gate #4 requires dynamic-height parity |
| 10 | [k2_jax_dedicated_runner_visual_final_report.md](k2_jax_dedicated_runner_visual_final_report.md) | 2026-06-29 | `K2_JAX_DEDICATED_VISUAL_PASS` | Visual mode validated |
| 11 | [k2_jax_dedicated_runner_benchmark.md](k2_jax_dedicated_runner_benchmark.md) | 2026-06-29 | 187.5 Hz | Performance benchmark only |
| 12 | [k2_jax_dedicated_runner_regression_guard.md](k2_jax_dedicated_runner_regression_guard.md) | 2026-06-29 | Functional scenarios | Basic smoke: fixed_high, push_bwd only |

## Tier 4: Supporting Scripts and Tests

| # | File | Purpose |
|---|------|---------|
| 13 | `scripts/validate_k2_step_c_e_fixed_height.py` | Step C/E validation (original K2 Python baseline runner) |
| 14 | `scripts/validate_k2_step_d_push_matrix.py` | Step D push matrix (original K2 Python baseline runner) |
| 15 | `scripts/validate_k2_post_promotion_long_run.py` | Long-run 6000-step (original K2 Python baseline runner) |
| 16 | `scripts/validate_k2_dynamic_height_gate_crossing.py` | Dynamic height gate-crossing (original K2 Python baseline runner) |
| 17 | `scripts/simulate_hierarchical_controller.py` | Monolithic simulator (Python + JAX backends) |
| 18 | `scripts/run_k2_jax_realtime.py` | **Dedicated JAX realtime runner (CANDIDATE)** |
| 19 | `tests/test_k2_jax_component_parity.py` | 85 JAX component parity tests |
| 20 | `tests/test_k2_jax_step_parity.py` | 55 JAX step parity tests |
| 21 | `tests/test_k2_jax_backend_cli.py` | 20 JAX backend CLI tests |

## Tier 5: Original K2 Baseline Output Data

| # | Directory | Contents |
|---|-----------|----------|
| 22 | `outputs/k2_step_c_e_promotion_validation/` | Step C (7) + Step E (10) K2 Python runs |
| 23 | `outputs/k2_step_d_push_matrix_validation/` | 12 K2 Python push runs |
| 24 | `outputs/k2_post_promotion_long_run/` | 5 K2 Python 6000-step equilibrium runs |
| 25 | `outputs/k2_dynamic_height_gate_crossing/` | 5 K2 Python dynamic height runs |
| 26 | `outputs/k2_notch_low_q_v1_validation/` | 7 K2 Python initial validation runs |

---

## Key Finding: Existing Dedicated Runner Promotion Was Speed-Only

The [dedicated runner final report](k2_jax_dedicated_realtime_runner_final_report.md) classifies as `K2_JAX_DEDICATED_REALTIME_RUNNER_PASS` based on:
- Speed: 187.5 Hz (7.9× improvement)
- Functional smoke: fixed_high, push_bwd only
- Test regression: 11/11 stage 1 tests
- No controller semantics changed

**It does NOT:**
- Compare behavior against original K2 Python baseline
- Validate Step C/D/E scenarios
- Validate dynamic height beyond ramp_up benchmark
- Compare posture, drift, yaw, or leg twist metrics
- Run any of the original K2 promotion validation suites

## Metrics Extracted from Original K2 Reports

### Step C/E Metrics
- pitch_rms_deg, support_rms_m, hip_yaw_max (rad), LF_power (0.15-0.55 Hz), WIP_power (2.0-3.0 Hz), fell (bool)

### Step D (Push) Metrics
- Post-push pitch RMS (500-step window), post-push support RMS, LF power, WIP power, hip_yaw_max, falls

### Safety Gates
- Falls=0, Hip-yaw ≤ 0.35 rad, No hidden torque (>0.5 Nm), No WBC, real_simulation source

### Acceptance Thresholds from Original K2 Reports
- Falls: 0 (absolute)
- Hip-yaw: ≤ 0.35 rad
- Hidden torque: ≤ 0.5 Nm
- WBC: None allowed
- NaN/Inf: 0
- Classification: STRONG_BETTER, BETTER, EQUIVALENT, WORSE_BUT_SAFE, REGRESSION
- Regression = K2 falls where K1 does not, OR hip-yaw > 0.35, OR WIP K2 > 10× K1
