# Phase 1 — WBC Reconnaissance Report

**Date:** 2026-07-30  
**Git branch:** main  
**Paper commit reference:** f92640c (cited in paper L719)  
**Current controller:** ACC / V3_ANCHOR (four-source torque composer)

---

## 1. Paper Headline Numbers (from `paper/main.tex` on disk)

These are the "ground truth" numbers all WBC comparisons must reference:

| Metric | Value | Source (line) |
|--------|-------|---------------|
| ACC idle CoM RMS (standing, conservative) | **0.73 ± 0.07 mm** (N=10, 95%CI [0.68,0.78]) | L50, L706 |
| ACC idle CoM RMS (clean, EMA₀=0 transient) | **0.42 ± 0.03 mm** (N=5) | L626, L639 |
| ACC Fmin (omnidirectional) | **83 N** | L50, L70, L706 |
| ACC Fmed (omnidirectional) | **117 N** | L50, L70, L706 |
| ACC Fmax (clean, 0ms delay, forward) | **96 ± 1 N** | L622, L639, L702 |
| ACC Fmax (any non-zero delay) | **~47-50 N** | L622, L640-644 |
| Timing budget (end-to-end, non-RTOS worst-case) | **~9.5 ms** | L704 |
| Margin below 10 ms performance threshold | **0.5 ms** | L694, L704 |
| Classical baselines (6 variants) | all fail (100%) | L697 |
| PPO baseline fall rate | 85% (5.4M steps, single seed) | L699 |
| "Advanced WBC/MPC methods are deferred" | L699 | See §5 below |

---

## 2. Classification Table

### 2.1 W1 — WBC-as-Controller (Standalone QP-WBC as PRIMARY balance controller)

These files implement or support a standalone WBC that generates joint torques directly as the primary controller:

| Path | Classification | Evidence |
|------|:---:|-----------|
| `wheeled_biped/wbc/structured_qp_problem.py` | **W1 core** | Builds QP with 5 tasks (COM height, torso orientation, posture damping, wheel regularization, contact force). Used by both WBC-only and Assist arms. |
| `wheeled_biped/wbc/phase3b_cached_stack.py` | **W1 core** | Cached QP builder. `feasibility_only` hardcode bug (F2). |
| `wheeled_biped/wbc/phase3c_rolling_qp.py` | **W1 core** | Rolling-constraint-aware offline QP. |
| `wheeled_biped/wbc/phase3d2_fast_solver.py` | **W1 core** | Fast solver with OSQP backend. WBC solve time: mean 0.45ms (see timing). |
| `wheeled_biped/wbc/phase3d3_incremental_qp.py` | **W1 (unused)** | Incremental QP designed for real-time but NEVER integrated to production path. Has `time.perf_counter` instrumentation. |
| `wheeled_biped/controllers/integrated_wbc.py` | **W1** | "Integrated whole-body controller with proper force-to-torque mapping." Combines centroidal wrench, force distribution, Jacobian. 64-DOF model. Has integral term (k_roll_integral=100). |
| `wheeled_biped/controllers/centroidal_balance_controller.py` | **W1** | "Centroidal WBC with CoM regulation and capture point tracking." Roll + CoM lateral/sagittal + CP + height tasks. 60% authority budget. |
| `wheeled_biped/controllers/wbc_balance_controller.py` | **W1-simple** | "WBC balance controller with force/torque-based stabilization." Simple PD torque (roll/pitch), NO QP. Different architecture from the main QP-WBC. |
| `wheeled_biped/controllers/hierarchical_vmc_lqr.py` | **W1** | "Hierarchical VMC+LQR controller (Phase B.7 Task 3)." 4-layer: posture IK → CoM VMC → wheel LQR → roll/yaw PD. Standalone. NOT imported by any production module. |
| `wheeled_biped/controllers/dual_rate_balance_controller.py` | **W1** | "Dual-rate time-scale separation controller (Phase B.9 Task 8)." Fast 50Hz wheel LQR + slow 5Hz height IK. Goal: "maximize standalone survival time before residual RL." |

**Supporting WBC infrastructure (shared by W1 and W2):**

| Path | Role |
|------|------|
| `wheeled_biped/wbc/offline_counterfactual_evaluation.py` | Offline evaluation pipeline |
| `wheeled_biped/wbc/offline_qp_wbc.py` | Offline QP solve wrapper |
| `wheeled_biped/wbc/offline_three_arm_counterfactual.py` | Three-arm clone evaluation (V3, WBC-only, V3+WBC) |
| `wheeled_biped/wbc/offline_task_stack.py` | Task stack configuration |
| `wheeled_biped/wbc/offline_rolling_constants.py` | Rolling constraint constants |
| `wheeled_biped/wbc/persistent_osqp_backend.py` | Persistent OSQP workspace |
| `wheeled_biped/wbc/qp_solver_backends.py` | QP solver backends (OSQP, SLSQP) |
| `wheeled_biped/wbc/phase3d3e_jax_dynamics_cache.py` | JAX dynamics cache for incremental QP |
| `wheeled_biped/controllers/centroidal_state_estimator.py` | Centroidal state estimation |
| `wheeled_biped/controllers/centroidal_wrench_computer.py` | Centroidal wrench computation |
| `wheeled_biped/controllers/contact_jacobian.py` | Contact Jacobian (had F1 bug: zero-gradient for leg joints) |
| `wheeled_biped/controllers/force_distribution.py` | Force distribution |
| `wheeled_biped/controllers/simple_force_distributor.py` | Simple force distributor |
| `wheeled_biped/controllers/unified_force_distributor.py` | Unified force distributor |
| `wheeled_biped/controllers/qp_allocator.py` | QP allocator |
| `wheeled_biped/dynamics/jax_contact_dynamics.py` | Contact dynamics (location of F1 bug) |

### 2.2 W2 — WBC-as-Assist (torque-blend on top of ACC/V3)

The core WBC QP is the same as W1. W2 is defined by the **blending architecture**:

```
τ_assist = τ_v3 + α·(τ_wbc − τ_v3)
```

With a 7-term multiplicative gate:

```
αⱼ = α_max · g_stability · g_height · g_push · g_divergence · Aⱼ · K_roleⱼ
```

K_role = [0.12, 0.05, 0.60, 0.60, 0.35, 0.12, 0.05, 0.60, 0.60, 0.35]

| Path | Classification | Evidence |
|------|:---:|-----------|
| `scripts/evaluate_v3_vs_assist.py` | **W2** | Single-scenario V3 vs Assist comparison (864 lines). |
| `scripts/promote_v3_vs_assist.py` | **W2** | Full batch 48-scenario promotion eval (1213 lines). Adaptive α_max=0.35. |
| `scripts/run_v3_assist_comparison.py` | **W2** | Multi-scenario V3 vs Assist comparison (933 lines). |
| `scripts/phase3d_full_batch_execution.py` | **W2** | 225-scenario three-arm evaluation (V3, WBC-only, V3+WBC). Has per-step timing. |

**Posture-Guided variant (W2-alt):**
A third blending mode where WBC outputs posture targets (q_ref) rather than torque:

| Path | Classification |
|------|:---:|
| `docs/validation/posture_guided_vs_torque_blend_report.md` | **W2-alt** |

Achieved: pitch −17%, drift −18%, yaw −20%, height −1.2%, but roll +7% (vs torque-blend's +35%). Better than pure torque-blend but still not a net improvement.

### 2.3 ACC / V3_ANCHOR — Current Controller (NOT WBC)

These are the ACTIVE, currently-used controller files:

| Path | Role |
|------|------|
| `wheeled_biped/controllers/balance_core_torque_composer.py` | Four-source torque composer (ACC core) |
| `wheeled_biped/controllers/balance_core_types.py` | Shared types |
| `wheeled_biped/controllers/k2_jax_controller.py` | JAX controller with 11 independent control laws |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Current production balance controller |
| `wheeled_biped/controllers/sagittal_balance_state.py` | Balance state computation |
| `wheeled_biped/controllers/sagittal_wheel_balance_controller.py` | Wheel balance component |
| `wheeled_biped/controllers/lateral_roll_balance_controller.py` | Lateral roll component |
| `wheeled_biped/controllers/shape_posture_controller.py` | Shape/posture PD component |
| `wheeled_biped/controllers/support_feedforward_controller.py` | Support feedforward |
| `wheeled_biped/controllers/composer.py` | Hip-yaw ownership validation |
| `wheeled_biped/controllers/torque_ownership_validator.py` | Torque source validator |
| `wheeled_biped/controllers/calibrated_outer_loop_functions.py` | Outer-loop calibration functions |

### 2.4 Deleted Files (from git history)

| Path | Commit | Notes |
|------|--------|-------|
| `wheeled_biped/wbc/offline_rolling_constants.py` | 2f9abde | "30-byte garbage file causing NameError on import" |
| `outputs/phase_b8_diagnostics/telemetry_hierarchical_vmc_lqr_h*.csv` | d16570c | 6 CSV files from Phase B.8 hierarchical VMC evaluation |

---

## 3. WBC Metrics Data

### 3.1 WBC-only (W1) — Standalone Performance

| Metric | WBC-only | V3/ACC | Ratio | Source |
|--------|:--------:|:------:|:-----:|--------|
| Total falls (225 scenarios) | 5,985 | 347 | **17.2× worse** | `V3_vs_V3_Assist_comparison_report.md` L100-103 |
| Pitch RMS | 41.8° | 15.9° | 2.63× worse | Ibid. L152 |
| Yaw drift | 8.0° | 4.24° | 1.89× worse | Ibid. L153 |
| Single-push falls | 3,353 | 118 | 28.4× worse | Ibid. L154 |
| Height RMS | 0.253 m | 0.261 m | 0.97× (better) | Ibid. L16 |
| Roll RMS | 96.8° | 84.2° | 1.15× worse | Ibid. L18 |
| Planar drift | 2.57 m | 2.42 m | 1.06× worse | Ibid. L19 |
| Falls (Phase 3D, 4 step_e) | 14,713 | 0 | — | `full_batch_verdict.json` L31 |
| Survival time (h=0.60, torque prototype) | 0.52 s | — | 100% fall | `best_torque_wbc_summary.json` L304 |
| Posture error | — | — | 8.05× worse | `full_batch_verdict.json` L43 |
| Yaw error | — | — | 15.43× worse | `full_batch_verdict.json` L52 |
| Solver success rate | 85.85% | — | 14.15% failure | `full_batch_solver_timing.json` L11 |

**Failure mechanism:** Velocity-damping posture task (not position tracking), single linearization at wrong height (0.67m vs 0.45-0.60m operating range), no gravity compensation feedforward, no wheel-ground rolling model, QP solve failures create torque discontinuities.

### 3.2 WBC-assist (W2) — Performance vs V3

| Metric | V3 | V3+WBC Assist | Δ | Source |
|--------|:--:|:------------:|:--:|--------|
| **225-scenario batch:** all metrics | baseline | **identical** (all ratios = 1.000000) | 0 | `V3_vs_V3_Assist_comparison_report.md` L107-114 |
| Falls (48-scenario quick) | 0 | 0 | 0 | `summary.json` L17-18 |
| Classification | — | 48/48 ASSIST_EQUIVALENT | — | `summary.json` L14-16 |

**Keyframe test (with gate overrides to force WBC contribution, h=0.53m):**

| Metric | V3 | Assist | Δ | Source |
|--------|:--:|:------:|:--:|--------|
| Pitch RMS | 2.44° | 2.04° | **−17%** ✅ | `V3_vs_WBC_root_cause_analysis.md` L178-185 |
| Drift | 0.070 m | 0.058 m | **−18%** ✅ | Ibid. |
| Yaw drift | 0.012° | 0.010° | **−20%** ✅ | Ibid. |
| Height RMSE | 0.123 m | 0.121 m | **−1.2%** ✅ | Ibid. |
| **Roll RMS** | 0.20° | **0.27°** | **+35%** ❌ | Ibid. |

**Verdict: Gate blocks WBC 99%+ of the time.** When forced open, WBC helps sagittal dimensions but degrades lateral (roll). Net effect = zero measurable improvement.

### 3.3 Known Critical Bugs in WBC

From `v3_wbc_assist_root_cause_audit.md`:

| ID | Severity | Description |
|----|:--------:|-------------|
| F1 | CRITICAL | Contact Jacobian zero-gradient: all leg joint columns = 0 (autodiff through cached FK) |
| F2 | CRITICAL | Fast solver hardcodes `feasibility_only` — drops entire task stack |
| F3 | CRITICAL | Offline eval passes base-z (0.536m) as height_ref instead of CoM-z (0.404m) → 13.2 cm error |
| F4 | CRITICAL | Gate adaptive assist height reference conflict with V3's CoM command |

**Note:** These bugs mean WBC performance was measured with a BROKEN implementation. However, the root cause analysis in `V3_vs_WBC_root_cause_analysis.md` argues the architectural conflict (decentralized V3 vs centralized QP blending) is fundamental, not bug-dependent.

---

## 4. Timing Instrumentation

### 4.1 Available Timing Data

From `outputs/phase3d_full_batch_execution/full_batch_solver_timing.json` (OFFLINE batch, 4 scenarios):

| Metric | Value |
|--------|-------|
| WBC solve time (mean) | **0.45 ms** |
| WBC solve time (p95) | 2.94 ms |
| WBC solve time (p99) | 3.43 ms |
| WBC solve time (max) | 8.63 ms |
| QP build time (mean) | **105.85 ms** |
| QP build time (p95) | 604.50 ms |
| Full step time (mean) | **257.96 ms** |
| Full step time (p95) | 1290.46 ms |

### 4.2 Timing Data Limitations

- **`production_realtime_wbc_injection: false`** (`full_batch_verdict.json` L71): WBC was NEVER injected into the real-time control loop
- **`wbc_torque_offline_clone_only: true`** (L76): WBC torque was only evaluated offline on simulation clones
- The QP build time (106 ms mean) is from Python/NumPy offline path, not JAX JIT
- The `phase3d3_incremental_qp.py` has comprehensive `time.perf_counter` instrumentation but was never integrated into real-time
- The per-step timing added in commit `f025274` exists in `phase3d_full_batch_execution.py` only (offline batch path)

### 4.3 What We DON'T Have

- **No real-time WBC compute time measurement** (µs per step in the actual control loop)
- **No ACC+WBC end-to-end timing in the production path**
- **No JAX-JIT WBC timing** (the incremental QP path was designed for this but never integrated)

### 4.4 What We CAN Measure

The incremental QP path in `phase3d3_incremental_qp.py` has full timing harness:
- `snapshot_time_s`, `build_time_s`, `csc_patch_time_s`, `osqp_update_time_s`, `osqp_solve_time_s`

But it's only been run in offline batch mode. The solve-only time (~0.45 ms mean) plus matrix patching (~? ms) would be the real-time overhead, but we don't have these numbers integrated into the production loop.

---

## 5. "deferred" Sentence in Paper

**Line 699:** `Advanced WBC/MPC methods are deferred.`

This single sentence needs to be split into:
- **WBC:** NO LONGER deferred — investigated during development (see §3 above), found empirically inferior
- **Convex MPC full-plant:** STILL deferred — never implemented; ill-conditioning κ(A)≈10^10 (Limitation 7, L697)

---

## 6. Points of Ambiguity — DECISIONS NEEDED

### 6.1 Posture-Guided variant: W2 or separate thing?
The posture-guided approach (WBC→q_ref, V3→τ) is architecturally different from W2's torque-blend. Should it be:
- (a) Treated as a sub-variant of W2 ("W2-posture-guided"),
- (b) Treated as a separate W3 category, or
- (c) Mentioned only qualitatively as "we also tried..." without separate metrics?

**Recommendation:** (a) — mention as a W2 variant that eliminated torque conflict but still didn't improve net performance.

### 6.2 How to handle the known WBC bugs (F1-F4)?
The bugs mean WBC was evaluated with a partially broken implementation. Options:
- (a) Report numbers as-is with a footnote acknowledging the bugs (honest but potentially misleading to reviewers)
- (b) Report the root cause architectural argument as primary, and mention bugs as secondary
- (c) Explicitly note "WBC was evaluated with known implementation bugs; however, the fundamental torque-blend architectural conflict is independent of these bugs"

**Recommendation:** (c) — acknowledge both the bugs and the architectural argument. The `V3_vs_WBC_root_cause_analysis.md` makes a compelling architectural case that the decentralized-vs-centralized conflict is fundamental.

### 6.3 Should we re-run WBC after fixing bugs?
Options:
- (a) Yes — fix F1-F4 and re-evaluate to get "fair" WBC numbers
- (b) No — the existing data plus architectural argument is sufficient for the paper
- (c) Only re-run if reviewer asks

**Recommendation:** (b) — the data we have is sufficient for a development-time investigation report. The architectural conflict argument is strong. If a reviewer asks, we have the code and can re-run.

---

## 7. MUST-RUN List (for Phase 2)

Currently empty — no `[NEEDS-RERUN]` entries identified yet. The following would need re-run if we want real-time timing:

1. **Real-time WBC timing measurement:** Write a harness that measures ACC-only vs ACC+WBC compute time per step in the actual control loop. Need to integrate incremental QP into the production path or measure the offline WBC's per-step cost. Command format TBD after Phase 2 classification.

---

*End of Phase 1 Report. STOPPING for user confirmation before Phase 2.*
