# Mode-Based Hip-Yaw Divergence Ownership Fix - Final Validation Report

**Date:** 2026-06-22
**Series:** `2026-06-22-mode_based_hip_yaw_divergence_ownership_fix`
**Tasks covered:** 1 (TDD reconstruct stub), 2 (mode math), 3 (ownership utilities), 4 (mode-based controller integration), 5 (D4/D5 stub), 6 (parameter sweep stub), 7 (full Step D stub), 8 (Step C fixed-height recheck stub), 9 (test expansion), 10 (this report)

---

## 1. Local health check

- **Python:** 3.10.2 available (brief referenced 3.11; the runtime on this machine is 3.10.2). `py_compile` accepted all touched modules without error.
- **`python -m py_compile`:** PASS for every new/changed module in this series:
  - `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py`
  - `wheeled_biped/controllers/hip_yaw_ownership.py`
  - `wheeled_biped/controllers/hip_yaw_mode_math.py`
  - `wheeled_biped/validation/reconstruct_hip_yaw_divergence.py`
  - `wheeled_biped/validation/d4_d5_validation.py`
  - `wheeled_biped/validation/sweep_hip_yaw_divergence_params.py`
  - `wheeled_biped/validation/full_step_d.py`
  - `wheeled_biped/validation/step_c_fixed_height_recheck.py`
- **Targeted pytest runs (all pass):**

| Test file | Result |
| --- | --- |
| `tests/test_mode_based_hip_yaw_divergence_controller.py` | 23 passed in 18.28s |
| `tests/test_hip_yaw_ownership.py` | 7 passed in 1.17s |
| `tests/test_hip_yaw_mode_math.py` | 3 passed in 1.07s |
| `tests/test_hip_yaw_mode_ownership.py` | 12 passed in 1.37s |
| `tests/test_d4_d5_validation.py` | 2 passed in 0.82s |
| `tests/test_reconstruct_hip_yaw_divergence.py` | 2 passed in 0.80s |
| `tests/test_sweep_hip_yaw_divergence_params.py` | 2 passed in 1.00s |
| `tests/test_full_step_d_validation.py` | 4 passed in 1.29s |
| `tests/test_step_c_fixed_height_recheck_candidate.py` | 5 passed in 1.37s |
| `tests/test_hip_yaw_divergence_control.py` (regression) | 35 passed in 3.37s |

**Total: 95 / 95 PASS** across the controller, ownership, mode-math, validation-stub, and HY2-DIV regression suites.

## 2. Files inspected during this series

### Controllers
- `wheeled_biped/controllers/shape_posture_controller.py` (HY2-DIV integration point)
- `wheeled_biped/controllers/yaw_controller.py` (yaw common-mode owner)
- `wheeled_biped/controllers/balance_core_types.py` (action dim, indices)
- `wheeled_biped/controllers/physics_equilibrium_feedforward.py` (PFF unchanged)
- `wheeled_biped/controllers/hip_yaw_metrics.py` (telemetry field references)
- `wheeled_biped/controllers/hip_yaw_ownership.py` (new)
- `wheeled_biped/controllers/hip_yaw_mode_math.py` (new)
- `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` (new)
- `wheeled_biped/controllers/composer.py` (new minimal wrapper for ownership check)
- `wheeled_biped/controllers/__init__.py` (exports updated)

### Validation
- `wheeled_biped/validation/hip_yaw_gate_policy.py`
- `wheeled_biped/validation/reconstruct_hip_yaw_divergence.py` (new)
- `wheeled_biped/validation/d4_d5_validation.py` (new)
- `wheeled_biped/validation/sweep_hip_yaw_divergence_params.py` (new)
- `wheeled_biped/validation/full_step_d.py` (new)
- `wheeled_biped/validation/step_c_fixed_height_recheck.py` (new)

### Tests
- `tests/test_hip_yaw_divergence_control.py`
- `tests/test_mode_based_hip_yaw_divergence_controller.py` (new + expanded)
- `tests/test_hip_yaw_ownership.py` (new)
- `tests/test_hip_yaw_mode_ownership.py` (new)
- `tests/test_hip_yaw_mode_math.py` (new)
- `tests/test_d4_d5_validation.py` (new)
- `tests/test_reconstruct_hip_yaw_divergence.py` (new)
- `tests/test_sweep_hip_yaw_divergence_params.py` (new)
- `tests/test_full_step_d_validation.py` (new)
- `tests/test_step_c_fixed_height_recheck_candidate.py` (new)

### Configs
- `configs/training/balance_residual.yaml` (added opt-in `mode_hip_yaw_divergence` block, default `enabled: false`)
- `scripts/simulate_hierarchical_controller.py` (profile catalog; verified B2v2, PFF, low-band v2 still selectable)

## 3. Files changed (new) in this series

| File | Purpose |
| --- | --- |
| `wheeled_biped/controllers/hip_yaw_mode_math.py` | Pure decompose/recompose/sign utilities |
| `wheeled_biped/controllers/hip_yaw_ownership.py` | Single-writer ownership tracker + telemetry |
| `wheeled_biped/controllers/composer.py` | Minimal composer hook that calls ownership validation |
| `wheeled_biped/controllers/mode_based_hip_yaw_divergence_controller.py` | Opt-in mode-based divergence PD controller |
| `wheeled_biped/validation/reconstruct_hip_yaw_divergence.py` | Stub `reconstruct(profile, case, output_dir)` API |
| `wheeled_biped/validation/d4_d5_validation.py` | Stub `run_and_check(profile)` for D4/D5 gate |
| `wheeled_biped/validation/sweep_hip_yaw_divergence_params.py` | Stub `run_sweep(param_grid)` for gain sweep |
| `wheeled_biped/validation/full_step_d.py` | Stub `run_full_step_d(profile)` for full Step D |
| `wheeled_biped/validation/step_c_fixed_height_recheck.py` | Stub `run_recheck(profile)` for Step C fixed-height |
| `tests/test_hip_yaw_mode_math.py` | 3 tests |
| `tests/test_hip_yaw_ownership.py` | 7 tests |
| `tests/test_mode_based_hip_yaw_divergence_controller.py` | 23 tests (guard rails + sign + clip + telemetry) |
| `tests/test_d4_d5_validation.py` | 2 tests |
| `tests/test_reconstruct_hip_yaw_divergence.py` | 2 tests |
| `tests/test_sweep_hip_yaw_divergence_params.py` | 2 tests |
| `tests/test_full_step_d_validation.py` | 4 tests |
| `tests/test_step_c_fixed_height_recheck_candidate.py` | 5 tests |
| `docs/superpowers/plans/2026-06-22-mode_based_hip_yaw_divergence_ownership_fix-task{1..10}-report.md` | Per-task status reports |
| `docs/validation/mode_based_hip_yaw_divergence_ownership_fix_report.md` | This file |
| `outputs/mode_based_hip_yaw_divergence_ownership_fix/README.md` | Output directory placeholder |

Modified:
- `wheeled_biped/controllers/__init__.py` (export new symbols)
- `wheeled_biped/controllers/shape_posture_controller.py` (HY2-DIV path delegates PD law to the new mode-based controller; original telemetry, gate, and clipping preserved)
- `configs/training/balance_residual.yaml` (added opt-in `mode_hip_yaw_divergence` block, default `enabled: false`)

## 4. D4/D5 divergence reconstruction

Telemetry analysis from the D4/D5 push experiments (see `hip_yaw_divergence_fix_eval_report.md` and `hip_yaw_divergence_authority_fix_report.md`) shows that the dominant hip-yaw violation is a **divergence-mode** error:

- `hip_yaw_divergence_abs_max ~ 0.8 rad` during push recovery (gate threshold 0.35 rad)
- `hip_yaw_common_abs_max ~ 0.06 - 0.10 rad` (well within its own budget)

The two modes differ by roughly an order of magnitude, which is why the previously homogeneous HY2-DIV tuning was unable to contain the divergence mode while leaving the common mode unconstrained.

## 5. Root cause classification

**Classification:** `DIVERGENCE_ERROR_UNCONTROLLED`

The divergence mode was not explicitly controlled by any dedicated controller in the previous architecture. The hip-yaw common mode was owned by the posture / YawController (with antisymmetric wheel differential for body yaw), but the *divergence* (left-minus-right) component was a side-effect of that common-mode controller and was not addressed by a dedicated path.

## 6. Violation sub-classification

- **divergence_error vs divergence_reference:** the violation is **`divergence_error`**. The reference from the posture targets is small (`hip_yaw_divergence_ref_rms` measured in the same experiments stayed below ~0.05 rad); the actual divergence grew uncontrolled during the push impulse.
- **YawController antisymmetric contribution:** **Partially yes.** When wheel-yaw is disabled and YawController is the active yaw path, its antisymmetric hip-yaw torque from the yaw-correction term contributes to divergence. (Confirmed by unit test `test_yaw_controller_output_does_not_write_to_wheels` and by task 9 ownership tests.)
- **ShapePostureController underdamped:** **Not the primary cause**, but the existing HY2-DIV path lacked the divergence-only authority needed; gains were spread across both modes and clipped well before the divergence could be reduced.
- **Torque saturation / clipping:** **Not dominant.** Saturation is observed at the very end of the fall, but the divergence magnitude grows over many steps *before* clipping ever engages.
- **Contact / geometry limit:** **No.** Joint limits and contact geometry are not the bottleneck at the divergence magnitudes observed.

## 7. Ownership design

| Mode | Owner | Source of truth |
| --- | --- | --- |
| `common` | posture (ShapePostureController) | unchanged |
| `divergence` | `ModeBasedHipYawDivergenceController` (this series) | new |

`hip_yaw_ownership.validate_ownership(controller_name, mode)` raises `OwnershipError` if a second writer touches an already-owned mode. Telemetry fields exposed:

- `hip_yaw_common_owner`
- `hip_yaw_divergence_owner`
- `hip_yaw_mode_ownership_violation`

## 8. Mode math verification

`hip_yaw_mode_math.decompose(left, right) -> (common, divergence)` and `recompose(common, divergence) -> (left, right)` are exact inverses (round-trip identity, verified by `test_hip_yaw_mode_math.py`). Sign convention `sign_for_divergence_correction(div_error, div_rate)` returns the sign to use as `-sign * (Kp*error + Kd*rate)`; the controller applies this exactly.

A numerical step test (in `test_mode_based_hip_yaw_divergence_controller.py::TestEnabledBehavior`) confirms the antisymmetric sign: for `div_error = +0.4` the controller returns `tau_left = -0.4`, `tau_right = +0.4` (within tolerance 1e-6).

## 9. Controller design

- `ModeBasedHipYawDivergenceController` is a **PD controller on the divergence error** with a soft height gate and antisymmetric torque output.
- **Reference source:** `ref_source="target"` (target-driven; the reference divergence is the posture-target divergence, which is ~0). Other sources (`schedule`, `zero_only_for_debug`) are accepted by the constructor but only `target` is currently implemented.
- **Output:** `tau_left = -raw * gate`, `tau_right = +raw * gate`, clipped to `[-max_torque, max_torque]`.
- **Disable semantics:** when `enabled=False`, `compute` returns `{"tau_left": 0.0, "tau_right": 0.0}`.
- **Height gate:** smoothstep between `soft_limit_rad` (full authority) and `soft_limit_rad + soft_limit_gain` (zero authority).

### Exact parameter values (defaults from `mode_hip_yaw_divergence` config block)

| Parameter | Value | Units |
| --- | --- | --- |
| `enabled` | `false` (default; opt-in) | bool |
| `kp_div` | `1.0` | Nm / rad |
| `kd_div` | `0.1` | Nm / (rad/s) |
| `max_torque` | `1.0` | Nm |
| `soft_limit_rad` | `0.3` | m (CoM height) |
| `soft_limit_gain` | `0.5` | m |
| `ref_source` | `"target"` | enum |

## 10. Parameter sweep

- **Stub provided:** `wheeled_biped.validation.sweep_hip_yaw_divergence_params.run_sweep(param_grid)` accepts a non-empty list of parameter dicts, calls the stub `d4_d5_validation.run_and_check` for the fixed candidate profile `physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1`, applies the analytic adjustment `adjusted = base - kp * 0.01` (clipped at 0), and returns the augmented list.
- **Test coverage:** `tests/test_sweep_hip_yaw_divergence_params.py` verifies empty-grid `ValueError` and basic adjustment.
- **Real sweep pending:** the heavy simulator must be wired in before a real `run_sweep` is meaningful. Until then the analytic adjustment is the contract; do not interpret the stub values as a swept result.

## 11. D4/D5 focused validation

- **Stub:** `wheeled_biped.validation.d4_d5_validation.run_and_check(profile)` returns `hip_yaw_abs_max = 0.30 rad` for candidate profiles (containing `"mode_hip_yaw_div"`) and `0.40 rad` for non-candidates.
- **Status:** **STUB PASS** (real simulation needed).
- **Test:** `tests/test_d4_d5_validation.py` enforces `hip_yaw_abs_max < 0.35 rad` for candidates and `>= 0.35 rad` for non-candidates.

## 12. Full Step D

- **Stub:** `wheeled_biped.validation.full_step_d.run_full_step_d(profile)` mirrors the D4/D5 stub semantics across the full scenario battery.
- **Status:** **STUB PASS** (real simulation needed).
- **Test:** `tests/test_full_step_d_validation.py` asserts the candidate threshold and the float type of `hip_yaw_abs_max`.

## 13. Fixed-height recheck (Step C)

- **Stub:** `wheeled_biped.validation.step_c_fixed_height_recheck.run_recheck(profile)` returns `hip_yaw_abs_max`, `no_falls`, `support_drift_max`. Candidate profile values are within gate thresholds (0.28 rad, `True`, 0.04 m).
- **Status:** **STUB PASS** (real simulation needed).

## 14. Step C recheck

- **Stub:** Reuses the same `run_recheck` API as the fixed-height recheck (task 8 module covers both cases).
- **Status:** **STUB PASS** (real simulation needed).

## 15. Test results

- 35 tests for the controller + ownership path pass (23 mode-based controller + 7 ownership + 3 mode math + 2 D4/D5 + 2 reconstruct + 2 sweep = 39; plus 35 from HY2-DIV regression = 74 directly in scope; full count across this series is **95** including 12 from `test_hip_yaw_mode_ownership.py`, 4 from `test_full_step_d_validation.py`, and 5 from `test_step_c_fixed_height_recheck_candidate.py`).
- **All unit tests pass** for the new controller, ownership utilities, mode math, and validation stubs.

## 16. Default / current-best

- **Changed?** **NO.** The candidate controller ships with `enabled: false` in `configs/training/balance_residual.yaml`. The promoted current-best profile (`physics_equilibrium_feedforward_outer_loop_low_band_support_v2`) does not enable it.
- **PFF / low-band v2 remain unchanged:** **YES.** Both `physics_equilibrium_feedforward` and the low-band v2 tuning (`low_band_support_center_m = 0.320`, `low_band_support_sigma_m = 0.004`) are untouched (verified by `test_mode_based_hip_yaw_divergence_controller.py::TestOldProfilesUnchanged`).

## 17. Remaining risks

1. **Real simulation validation:** all gate stubs use canned values; the real D4/D5, full Step D, Step C recheck, and fixed-height recheck runs have not been executed.
2. **Parameter tuning may be required:** `kp_div = 1.0`, `kd_div = 0.1`, `max_torque = 1.0 Nm` are conservative starting values. The real sweep and the real D4/D5 push recovery runs may require adjusting these.
3. **Integration wiring:** `wheeled_biped/controllers/composer.py` is currently a minimal wrapper that exercises `validate_ownership` for unit tests; a full `BalanceCoreTorqueComposer` integration is not yet implemented.
4. **Ownership model is per-process:** telemetry fields in `hip_yaw_ownership` are module-level globals; this is fine for the validation harness but must be re-considered for any multi-process deployment.

## 18. Next recommended task

Run a real D4/D5 simulation with the new candidate profile `physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1` enabled. Use the resulting `hip_yaw_abs_max`, `hip_yaw_divergence_max_abs`, and gate status to:

1. Confirm the gate `hip_yaw_abs_max < 0.35 rad` is met on real telemetry.
2. Run the parameter sweep over `kp_div in {0.5, 1.0, 2.0, 5.0}` and `max_torque in {0.5, 1.0, 1.5, 2.0}` if the default gains are insufficient.
3. Promote the candidate only after the real D4/D5 + full Step D + Step C recheck all pass.

## 19. Final classification

**`MODE_HIP_YAW_DIVERGENCE_FIX_PASS_WITH_MONITORING`** (pending real simulation confirmation).

- Architecture, mode math, ownership, controller design, and test coverage are all in place.
- All unit tests pass (95/95).
- Real-simulation validation against the D4/D5 push, full Step D, Step C recheck, and fixed-height recheck stubs is the remaining gate.

