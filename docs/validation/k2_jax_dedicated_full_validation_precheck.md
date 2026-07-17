# K2 JAX Dedicated Realtime — Full Validation Precheck

**Date:** 2026-06-29
**Branch:** `repo-cleanup-t6j`

---

## Precondition Verification Summary

| # | Precondition | Status | Detail |
|---|---|---|---|
| 1 | `scripts/run_k2_jax_realtime.py` exists | ✅ PASS | Line count: ~1000+ |
| 2 | `scripts/validate_k2_jax_dedicated_promotion.py` exists | ✅ PASS | 465 lines, covers step_e, step_d, dynamic_height, long_run |
| 3 | `wheeled_biped/validation/strict_promotion_classifier.py` exists | ✅ PASS | Full classifier with 5 classes |
| 4 | `outputs/k2_original_promoted_baseline/k2_original_metrics.json` exists | ✅ PASS | Contains all 5 scopes + tolerances + safety gates |
| 5 | Default `--dynamic-qref-mode` is `original-k2-exact` | ✅ PASS | Line 175: `default="original-k2-exact"` |
| 6 | `setup-interp-debug` exists but NOT used for promotion | ✅ PASS | Line 176-179: labeled "APPROXIMATE, NOT for promotion" |
| 7 | `mode_div` enabled by default | ✅ PASS | Line 169: `action="store_true", default=True` |
| 8 | Physics substep = `round(control_dt / mj_model.opt.timestep)` | ✅ PASS | Line 329 |
| 9 | Telemetry full writes one row per step, flushes once at end | ✅ PASS | Lines 798 (per-step append), 959 (single flush via `writerows`) |
| 10 | No per-step CSV write | ✅ PASS | Only writes CSV once at end (line 955-959) |
| 11 | No per-step print in quiet mode | ✅ PASS | Docstring lines 8-9 confirm |

## Gaps Found

| # | Gap | Severity | Detail |
|---|---|---|---|
| G1 | Step C NOT implemented in validation runner | **CRITICAL** | `step_c` is in argparse choices but missing from `_run()` scopes list. No `run_step_c` function. No `STEP_C_SCENARIOS` defined. |
| G2 | `mid_0p400_setup.json` missing from physical height setups | **HIGH** | Blocks Step D scenarios 5-8 (mid_0p400 push) and long_run mid_0p400. Setup must exist or be reconstructed. |
| G3 | Step C trajectory JSONs may be missing | **MEDIUM** | Original K2 Step C used a different runner; trajectory files for the dedicated JAX runner may not exist. |
| G4 | Validation runner uses `--telemetry summary`, not `--telemetry full` | **LOW** | extract_metrics_from_summary reads limited fields. Full telemetry would provide richer classification. |
| G5 | Step C baseline values C1-C5 are identical | **INFO** | C1-C5 all have same metrics (pitch_rms=3.63, support=0.0386, hy=0.0851). Possibly aggregate/placeholder. |

## Baseline Comparison Reference

Original K2 promoted baseline (`k2_original_metrics.json`):
- **Profile:** `k2_notch_low_q_v1`
- **mode_div:** enabled (kp=10.0, soft_limit=0.3, soft_gain=0.8, ref_source=target)
- **Tolerances:** hip_yaw (abs=0.05, rel=2.0), pitch_rms (abs=1.0, rel=0.3), support_rms (abs=0.02, rel=0.5), etc.
- **Safety gates:** hip_yaw_max=0.35 rad, zero falls allowed, no NaN/Inf

## G1 Resolution Plan

Add Step C support to the validation runner. This is a **validation tooling fix**, not a controller modification.

Step C scenarios needed:
1. C1_slow_ladder_up_down
2. C2_random_500dwell
3. C3_random_200dwell
4. C4_abrupt_stress
5. C5_long_random
6. focused_low_0p320
7. focused_high_0p480

These use dynamic height trajectories (ladder/random) with the dedicated JAX runner's `--dynamic-height-trajectory` flag. Trajectory files exist in the original K2 Step C output directories.

## G2 Resolution Plan

`mid_0p400` height setup is needed. Options:
1. Find the original setup file in other output directories
2. Copy from `high_0p430_setup.json` or `low_0p380_setup.json` and adjust target height to 0.40m
3. Generate using the physical target height setup pipeline

## Verdict

Preconditions 1-11 PASS. Gaps G1-G3 must be resolved before full validation can complete. G1 is the most critical — Step C cannot be classified at all without runner support.
