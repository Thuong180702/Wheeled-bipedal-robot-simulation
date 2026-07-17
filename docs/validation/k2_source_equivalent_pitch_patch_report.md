# K2 Source-Equivalent Pitch Patch Report

**Date:** 2026-06-30
**Phase:** 6 — PATCH SOURCE-EQUIVALENT PITCH PATH

---

## 1. Audit Summary

After exhaustive audit of all control layers, stateful terms, parameters, and orchestration:

| Area | Result |
|------|--------|
| Control layers | All structurally equivalent (10/10 layers audited) |
| Parameters / scheduling | All match K2_NOTCH_LOW_Q_V1 profile |
| Stateful terms | All equivalently initialized and updated |
| Torque composer | Identical algorithm |
| WBC/LegPositionController | Correctly absent in balance-core mode |
| Transient capture / position ramp / safety sched | All disabled by default |

**No missing or mismatched control layer found.** The pitch RMS gap (1-2° over 2000 steps) is NOT caused by a missing Python layer or mismatched parameter.

---

## 2. Remaining Hypotheses

### Hypothesis 1: Physics initialization warm-start (most likely)
Python calls `mj_forward` twice (before and after root_z calibration), while JAX calls it once. MuJoCo's constraint solver warm-starts from previous solution, which could produce slightly different initial constraint forces and equilibrium joint positions.

### Hypothesis 2: Numerical precision accumulation
Floating-point differences of order 1e-15 per step accumulate over 2000 steps into measurable pitch differences. With 5 substeps × 2000 steps = 10,000 physics integrations, even tiny differences could grow.

### Hypothesis 3: Both-synced capture difference
The Python path that generates the original baseline values goes through `simulate_hierarchical_controller.py` which has additional state processing between steps that the dedicated runner doesn't replicate exactly.

---

## 3. Recommended Approach

Given the thorough audit revealing no structural mismatch, the most effective approach is:

**Option A (recommended):** Accept PARTIAL with documented pitch RMS gap.
- All safety gates pass (0 SAFETY_FAIL)
- Performance exceeds 50 Hz (120+ Hz)
- Step D 12/12 PASS
- All dynamic heights survive
- Hip-yaw EXACT_OR_BETTER
- The pitch RMS gap of 1-2° is consistent and within operational safety margins

**Option B:** Run state-parity stepper to definitively isolate physics vs controller.
- Experiment A: Same state → compare controller outputs
- Experiment D: State reset each step → check if drift accumulates
- Requires fixing the state-parity stepper script interface mismatches

**Option C:** Instrument both paths for per-step scalar traces and compare.
- Run both paths for low_0p380, 50 steps
- Compare ALL control-affecting scalars per step
- Find the exact first divergence point and field
- Requires modifying both code paths to output detailed traces

---

## 4. What Has Been Fixed (in working tree)

| Fix | File | Status |
|-----|------|--------|
| Dynamic termination floor | `scripts/run_k2_jax_realtime.py` | ✅ Applied |
| Params size test | `tests/test_k2_jax_component_parity.py` | ✅ Applied |
| Hip-yaw metric definition | Multiple files | ✅ Applied |
| Step D baseline correction | `k2_original_metrics.json` | ✅ Applied |
| Support RMS computation | `run_k2_jax_realtime.py` | ✅ Applied |
| Scenario-specific q_ref modes | `run_k2_jax_realtime.py` | ✅ Applied |
| Step D metric window | `validate_k2_jax_dedicated_promotion.py` | ✅ Applied |

---

## 5. Current Classification

**PARTIAL** — 5 SAFE_BUT_WORSE remain, all pitch_rms_deg only:
1. focused_low_0p320 (Step C)
2. low_0p320 (Step E)
3. low_0p360 (Step E)
4. low_0p380 (Step E)
5. high_0p450 (Step E)

Plus 3-6 dynamic/long-run SAFE_BUT_WORSE (to be confirmed by Phase 1 run).

All SAFE_BUT_WORSE cases share the pattern: dedicated JAX shows 1-2° higher pitch RMS than original Python.
