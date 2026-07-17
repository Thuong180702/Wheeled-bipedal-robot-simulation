# K2 Both-Synced Full-Sim Pitch State Freeze

**Date:** 2026-06-30
**Phase:** 0 — FREEZE CURRENT SOURCE-EQUIVALENT PARTIAL STATE
**Classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`

This document freezes the current PARTIAL state at the start of the full-sim both-synced pitch RMS investigation. It records exactly what is known, what is hypothesized, and what remains unproven.

---

## 1. Repository State

| Field | Value |
|-------|-------|
| **Commit** | `0e1c7135e22b4cb852f71a795426cd3d3f19753a` |
| **Short hash** | `0e1c713` |
| **Commit message** | `Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000` |
| **Branch** | `repo-cleanup-t6j` |
| **Working tree** | Modified (uncommitted fixes for metric corrections, dynamic termination floor, support RMS) |

---

## 2. Final Scorecard

| Scope | Scenarios | PASS | SAFE_BUT_WORSE | SAFETY_FAIL |
|-------|-----------|------|----------------|-------------|
| Step C | 7 | 6 | 1 | 0 |
| Step E | 10 | 6 | 4 | 0 |
| Step D | 12 | 12 | 0 | 0 |
| Dynamic | 5 | 2 | 3 | 0 |
| Long-Run | 5 | 2 | 3 | 0 |
| **Total** | **39** | **28** | **11** | **0** |

**Performance:** ≥120 Hz (all scenarios well above 50 Hz minimum)

---

## 3. All 11 SAFE_BUT_WORSE Cases — Exact Values

Pitch RMS tolerance rule: `min(1.0°, 0.3 × original)` — from `k2_original_metrics.json`.

### Step C (1 case)

| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Tolerance (°) | Output path |
|---|----------|----------|----------|-----------|----------------|-------------|
| 1 | focused_low_0p320 | 2.83 | 3.69 | +0.86 | 0.849 | `step_c/low_0p320_focused/` |

### Step E (4 cases)

| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Tolerance (°) | Output path |
|---|----------|----------|----------|-----------|----------------|-------------|
| 1 | low_0p320 | 2.83 | 3.69 | +0.86 | 0.849 | `step_e/low_0p320/` |
| 2 | low_0p360 | 1.90 | 3.12 | +1.22 | 0.570 | `step_e/low_0p360/` |
| 3 | low_0p380 | 3.33 | 5.24 | +1.91 | 0.999 | `step_e/low_0p380/` |
| 4 | high_0p450 | 2.75 | 4.68 | +1.93 | 0.825 | `step_e/high_0p450/` |

### Dynamic Height (3 cases)

| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Output path |
|---|----------|----------|----------|-----------|-------------|
| 1 | up_down_cycle_0p330_0p480_0p330 | 3.32 | 3.92 | +0.60 | `dynamic_height/up_down_cycle/` |
| 2 | gate_dwell_0p420_0p450_0p480 | 3.05 | 6.19 | +3.14 | `dynamic_height/gate_dwell/` |
| 3 | gate_chatter_0p400_0p470 | 2.98 | 4.74 | +1.76 | `dynamic_height/gate_chatter/` |

### Long-Run (3 cases)

| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Output path |
|---|----------|----------|----------|-----------|-------------|
| 1 | low_0p330 | 3.97 | 5.07 | +1.10 | `long_run/low_0p330/` |
| 2 | high_0p450 | 3.45 | 4.55 | +1.10 | `long_run/high_0p450/` |
| 3 | high_0p430 | ~5.6 | 3.77 | −1.83 | `long_run/high_0p430/` |

---

## 4. Prior Conclusions (from previous investigations)

### Confirmed Equivalent

| Area | Verification Method | Status |
|------|-------------------|--------|
| **Height schedules** | Dense-grid formula audit | ✅ Exact |
| **Support FF** | Formula + parameter audit | ✅ Exact |
| **Warm-start** | No effect on 2000-step RMS | ✅ Eliminated |
| **Yaw-aware compensation** | Inactive in balance-core mode | ✅ Eliminated |
| **Torque composer** | Algorithm audit | ✅ Exact |
| **Stateful terms** | Init + update audit (7 terms) | ✅ Structurally equivalent |
| **Control layers** | 10/10 layers audited | ✅ Structurally equivalent |
| **Pitch RMS metric** | Formula/window/frame parity confirmed | ✅ Same |
| **Hip-yaw metric** | Corrected from divergence error to joint angle | ✅ EXACT_OR_BETTER |

### Minor Differences Found (negligible impact)

| Area | Detail | Impact |
|------|--------|--------|
| `filtered_com_z` init | Python 0.4 vs JAX 0.0 | Affects first ~20 steps only; decays with α=0.9 |
| Outer loop prev_error | Python `None` vs JAX `0.0` | Only if support error non-zero at step 0 |

---

## 5. Unresolved Gap

The prior conclusion of "numerical accumulation" is a **hypothesis, not final proof**.

### Evidence For

- Torques are bit-identical at steps 0-1 across all 10 actuators
- Divergence begins at step 2
- All 10 control layers are structurally equivalent
- All 7 stateful terms have equivalent init and update
- Pitch RMS metric formula, window, and frame are confirmed identical
- No missing control layer found after exhaustive audit

### Evidence Against

- The simplified state-parity stepper experiments D/E/F were **deferred or incomplete**
- No full-sim both-synced first-divergence proof exists
- The specific first divergent scalar has **never been identified**
- The causal chain from first divergence to pitch RMS has **never been traced**
- Why some heights amplify the divergence (low_0p360: +1.22°) while adjacent heights don't (low_0p330: +0.33°) is **unexplained**

### What "Numerical Accumulation" Would Require to Prove

1. Identify the **exact first divergent scalar** and step
2. Show that **both paths compute identically** for that scalar given identical inputs
3. Show that the **input to that scalar diverged** from a prior scalar (trace backward)
4. Continue tracing until reaching a **physics-level difference** (qpos/qvel/ctrl) at a substep
5. Show that this physics difference is **below MuJoCo solver tolerance** (no algorithmic source)
6. Conclude: the difference is truly numerical (1e-15 level), amplified by chaotic dynamics

**None of steps 1-6 have been completed.** The prior conclusion skipped directly to step 6.

---

## 6. Passing Controls (for reference)

| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Tolerance (°) | Status |
|---|----------|----------|----------|-----------|----------------|--------|
| 1 | low_0p300 | 2.68 | 2.91 | +0.23 | 0.804 | WITHIN_OLD_TOLERANCE |
| 2 | low_0p330 | 3.63 | 3.96 | +0.33 | 1.000 | WITHIN_OLD_TOLERANCE |
| 3 | low_0p340 | 2.97 | 1.86 | −1.11 | 0.891 | EXACT_OR_BETTER |
| 4 | high_0p430 | 4.98 | 3.13 | −1.85 | 1.000 | EXACT_OR_BETTER |
| 5 | high_0p465 | 3.55 | 3.62 | +0.07 | 1.000 | WITHIN_OLD_TOLERANCE |
| 6 | high_0p480 | 3.96 | 4.28 | +0.32 | 1.000 | WITHIN_OLD_TOLERANCE |

---

## 7. Key Files and Paths

| Item | Path |
|------|------|
| Baseline metrics | `outputs/k2_original_promoted_baseline/k2_original_metrics.json` |
| Scenario specs | `outputs/k2_original_promoted_baseline/scenario_specs.json` |
| Full metrics comparison | `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json` |
| Validation output | `outputs/k2_jax_dedicated_promotion_validation/` |
| Setup files | `outputs/physical_target_height_setups/` |
| Dedicated runner | `scripts/run_k2_jax_realtime.py` |
| Source runner | `scripts/simulate_hierarchical_controller.py` |
| Promotion validator | `scripts/validate_k2_jax_dedicated_promotion.py` |
| Existing trace tool | `scripts/trace_k2_source_vs_dedicated.py` |
| JAX controller | `wheeled_biped/controllers/k2_jax_controller.py` |
| Sagittal controller | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` |
| Signal filters | `wheeled_biped/controllers/signal_filters.py` |
| Strict classifier | `wheeled_biped/validation/strict_promotion_classifier.py` |
| Previous both-synced matrix | `docs/validation/k2_jax_full_both_synced_parity_matrix.md` |
| Control layer audit | `docs/validation/k2_python_vs_jax_pitch_control_layer_coverage.md` |
| Stateful terms audit | `docs/validation/k2_pitch_stateful_terms_audit.md` |
| Source-equivalent patch report | `docs/validation/k2_source_equivalent_pitch_patch_report.md` |

---

## 8. Non-Negotiable Rules (repeated for reference)

- Do NOT relax tolerance.
- Do NOT accept PARTIAL as final.
- Do NOT call numerical accumulation the root cause without first proving the exact first divergent scalar and why it cannot be source-equivalently removed.
- Do NOT tune gains blindly.
- Do NOT introduce scenario-specific hacks.
- Do NOT replace continuous height schedules with discrete height buckets.
- Do NOT change metric definitions.
- Do NOT regress Step D, dynamic survival, hip-yaw, support RMS, or performance.
- Every function, schedule, interpolation, gate, state update, and torque composition step must be proven source-equivalent with runtime traces, not only code inspection.

---

## 9. Acceptance

- [x] Current PARTIAL state is documented and reproducible.
- [x] Correct classification: `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`
- [x] Correct scorecard: 28 PASS, 11 SAFE_BUT_WORSE, 0 SAFETY_FAIL
- [x] All 11 SAFE_BUT_WORSE cases enumerated with exact pitch RMS values, deltas, tolerances, and output paths
- [x] Prior conclusions documented: height schedules exact, support FF exact, warm-start no effect, yaw-aware compensation inactive, torque composer exact, stateful terms structurally equivalent
- [x] Unresolved gap clearly stated: simplified state-parity stepper D/E/F incomplete, no full-sim both-synced first-divergence proof
- [x] This phase documents that previous "numerical accumulation" conclusion is a hypothesis, not final proof.
- [x] No code changes in this phase.

---

## 10. Reproduction Commands

### Full matrix (all scopes):
```bash
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_jax_dedicated_promotion_validation
```

### Classify only:
```bash
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --classify-only \
  --output-dir outputs/k2_jax_dedicated_promotion_validation
```

### Existing trace tool (for reference):
```bash
python scripts/trace_k2_source_vs_dedicated.py \
  --scenario step_e --height low_0p380 --steps 200
```
