# K2 Python → JAX Correctness/Parity Audit — Phase 0: Baseline and Rules

**Date:** 2026-06-27
**Audit type:** Correctness/parity (NOT coverage)
**Coverage baseline:** `K2_JAX_PORT_COVERAGE_COMPLETE_READY_FOR_PARITY_FIX`
**Profile:** `k2_notch_low_q_v1`
**Controller mode:** `balance-core`
**JAX mode for audit:** `--controller-backend both` (teacher-forcing)

---

## 1. Source of Truth

**Python K2 is the source of truth.** All JAX mechanisms must produce identical control torque to Python K2 within strict tolerances.

### Primary Python Source Files

| File | Key Content | Lines |
|------|------------|-------|
| `scripts/simulate_hierarchical_controller.py` | Simulation loop, controller wiring, teacher-forcing comparison | 8468 |
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | `SagittalVelocityDampedBalanceController.compute()` — sagittal wheel torque | 8317 |
| `wheeled_biped/controllers/balance_core_torque_composer.py` | `BalanceCoreTorqueComposer.compose()` — sum/clip/rate-limit | ~100 |
| `wheeled_biped/controllers/shape_posture_controller.py` | `ShapePostureController` — PD posture [0,1,2,3,5,6,7,8] | ~200 |
| `wheeled_biped/controllers/lateral_roll_balance_controller.py` | `LateralRollBalanceController` — roll [0,5] | ~150 |
| `wheeled_biped/controllers/yaw_controller.py` | `YawController` — yaw [1,6] | ~50 |
| `wheeled_biped/controllers/mode_hip_yaw_divergence_controller.py` | `ModeBasedHipYawDivergenceController` — mode-div [1,6] | ~100 |
| `wheeled_biped/controllers/support_feedforward_controller.py` | `SupportFeedforwardController` — empirical FF [2,3,7,8] | ~80 |
| `wheeled_biped/controllers/signal_filters.py` | `BiquadNotchFilter`, `smoothstep_gate` | ~150 |
| `wheeled_biped/controllers/calibrated_outer_loop_functions_v2.py` | PCHIP-interpolated calibrated gains | ~200 |
| `wheeled_biped/controllers/physics_equilibrium_feedforward.py` | PCHIP-interpolated physics FF | ~150 |

### Primary JAX Source File

| File | Key Content | Lines |
|------|------------|-------|
| `wheeled_biped/controllers/k2_jax_controller.py` | Complete JAX port — all sub-controllers, composer, ABS, outer loop, notch | 1331+ |

---

## 2. Mechanism Inventory (from coverage audit)

### Total: 72 mechanisms identified
### Active control-affecting: 50 mechanisms

| Category | Count | JAX Status |
|----------|-------|-----------|
| Input/State | 11 | 4 FULL, 2 PARTIAL, 5 EXTERNAL |
| Sagittal Balance | 17 | 9 FULL, 8 INACTIVE_ZERO |
| ABS Trim | 11 | 11 FULL |
| Outer Loop | 7 | 6 FULL, 1 PARTIAL (safety gates) |
| Leg/Body Controllers | 5 | 4 FULL, 1 OPT-IN |
| Composer | 7 | 5 FULL, 1 MISSING (diag), 1 PYTHON_ONLY |

---

## 3. Correctness Status Definitions

| # | Status | Definition |
|---|--------|-----------|
| 1 | EXACT_MATCH | Formula, params, inputs, state, order, output match within strict tolerance |
| 2 | PARAMETER_MISMATCH | Mechanism exists but gain/threshold/constant differs |
| 3 | FORMULA_MISMATCH | Mechanism exists but math differs |
| 4 | SIGN_OR_INDEX_MISMATCH | Actuator index, sign convention, or L/R mapping differs |
| 5 | INPUT_BOUNDARY_MISMATCH | JAX receives different value than Python uses |
| 6 | STATE_UPDATE_MISMATCH | State init, update timing, formula, dtype, or pre/post-physics timing differs |
| 7 | INSERTION_ORDER_MISMATCH | Same source but different pipeline position |
| 8 | GATE_OR_SAFETY_MISMATCH | Enable/disable condition, height gate, safety gate, or clamp differs |
| 9 | PRECISION_ONLY_MISMATCH | All match but diff attributable to dtype/interpolation precision |
| 10 | DIAGNOSTIC_ONLY_DIFFERENCE | Diff exists but doesn't affect control torque |
| 11 | UNKNOWN_NEEDS_TRACE | Insufficient evidence |

---

## 4. Tolerance Thresholds

| Comparison Type | Threshold |
|----------------|-----------|
| Parameters (gains, constants) | Exact equality (0.0) |
| Input fields (packed values) | 1e-12 |
| State fields (persistent state) | 1e-12 |
| Intermediate scalar terms | 1e-10 |
| Torque terms (per-component) | 1e-8 |
| Final tau (per-actuator) | 1e-5 |

---

## 5. Teacher-Forcing Scenarios

| # | Scenario | Height | Push | Description |
|---|----------|--------|------|-------------|
| 1 | fixed_high_0p480 | 0.48 m | None | Fixed high height, notch fully active |
| 2 | ramp_down | 0.48→0.33 m | None | Dynamic descending height transition |
| 3 | push_fwd_90N | 0.40 m | +90 N fwd | Forward push recovery |
| 4 | push_bwd_90N | 0.40 m | -90 N bwd | Backward push recovery |
| 5 | ramp_up | 0.33→0.48 m | None | Dynamic ascending height transition |
| 6 | gate_chatter | 0.40→0.47 m | None | Notch gate boundary oscillation |

---

## 6. Known Issues from Coverage Audit

### G2: Pitch reference offset application (HIGH RISK)
- **Python:** `pitch_x_error = body_pitch_x - (pitch_eq + total_offset_deg_to_rad)` — computed externally
- **JAX:** Receives pre-adjusted `pitch_x` via `pack_input_k2(pitch_x_rad=pitch_x_error)`
- **JAX internally:** Computes `total_pitch_ref_offset_deg` but does NOT apply it (line 1171-1173)
- **Risk:** If external offset != internal computation, pitch_x diverges → sagittal torque diverges

### G3: Outer loop safety gates (MEDIUM RISK)
- **Python:** Zeros outer-loop target when pitch/roll/contact/error thresholds fail
- **JAX:** Computes outer-loop target unconditionally (no safety gate)

### J7: tau_support_ff excluded (INTENTIONAL)
- JAX computes `tau_support_ff` but excludes it from `tau_sum`
- Documented as necessary — Python balance-core has no equivalent
- Only affects hip_yaw [1,6], not wheels [4,9]

### Known mismatch: ~0.01 Nm wheel torque at step 1
- Actuator [4] (left wheel)
- Appears at push_fwd step 1
- Root cause: NOT YET ISOLATED

---

## 7. Audit Constraints

1. **Do NOT fix** — identify only
2. **Do NOT tune** — no parameter changes
3. **Do NOT optimize** — no performance changes
4. **Do NOT change controller behavior** — diagnostic instrumentation only
5. **Do NOT loosen thresholds** — strict parity thresholds
6. **Do NOT make JAX default** — keep Python as default backend
7. **Do NOT claim strict clone pass** — only report facts
8. **Do NOT hide behind "functional validation passes"** — numeric parity required
9. **Find the exact wrong ported part** — file/function/line

---

## 8. Deliverable Map

| Phase | Deliverable | Path |
|-------|------------|------|
| 0 | Baseline and Rules | `docs/validation/k2_jax_correctness_audit_baseline.md` (this file) |
| 1 | Correctness Status Matrix | `docs/validation/k2_jax_correctness_matrix.md` + `.csv` |
| 2 | Parameter Parity Audit | `docs/validation/k2_jax_parameter_parity_audit.md` |
| 3 | Teacher-Forcing Ledger | `outputs/k2_jax_correctness_audit/teacher_forcing_<scenario>_steps0_20.csv` + report |
| 4 | Wheel Torque Root Cause | `docs/validation/k2_jax_wheel_step1_root_cause_audit.md` |
| 5 | Input Boundary Audit | `docs/validation/k2_jax_input_boundary_parity_audit.md` |
| 6 | State Update Parity Audit | `docs/validation/k2_jax_state_update_parity_audit.md` |
| 7 | Formula and Order Audit | `docs/validation/k2_jax_formula_and_order_parity_audit.md` |
| 8 | Difference Classification | `docs/validation/k2_jax_difference_classification_report.md` |
| 9 | Final Audit Report | `docs/validation/k2_jax_full_correctness_audit_report.md` |

---

## 9. Existing Diagnostic Infrastructure

The simulation script already has:
- `--controller-backend both` mode with teacher-forcing comparison
- `[BOTH@step]` output showing max_tau_diff, per-actuator torques, and selected state fields
- `k2_jax_input_flat_to_dict()` for input inspection
- `k2_jax_diag_flat_to_dict()` for diagnostics inspection
- `k2_jax_controller_step()` returns `(tau_final, new_state, diag_flat)`

What's missing for the audit:
- Per-term sagittal torque breakdown (tau_pitch, tau_pitch_rate, tau_position, etc.)
- Python-side per-term values for comparison
- Notch filter internal state comparison
- ABS internal state comparison (slow_mean, fast_mean, zc_count, trim_tau, guard_trigger)
- Outer loop state comparison (pitch_ref, support_error, support_error_rate)
- Parameter runtime value comparison
- Input field value comparison
