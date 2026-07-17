# K2 JAX JIT Controller Stack Port — Implementation Plan

**Date:** 2026-06-26
**Spec:** [docs/superpowers/specs/2026-06-26-k2-jax-jit-controller-port-design.md](../specs/2026-06-26-k2-jax-jit-controller-port-design.md)
**Target:** PORT_K2_CONTROLLER_STACK_TO_JAX_JIT_WITH_STRICT_PARITY

---

## Plan Overview

7 stages, sequential with gates. Each stage is self-contained and independently
reversible. The Python reference path is never removed. Backend remains `python` as
default throughout.

```
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5 → Stage 6 → Stage 7
(profile)  (notch)  (active)  (full)   (integ)  (valid)   (bench)
```

---

## Stage 1: Profile + Remove Duplicate Estimator Calls

**Goal:** Quantify the Python bottleneck, remove redundant work, and establish the
baseline for all subsequent stages.

### Implementation Tasks

1. **Instrument the simulation loop** in `scripts/simulate_hierarchical_controller.py`:
   - Add `time.perf_counter()` around `centroidal_estimator.estimate()` calls (both)
   - Add `time.perf_counter()` around `capture_estimator.update()` calls (both)
   - Add `time.perf_counter()` around `shape_posture.compute()`
   - Add `time.perf_counter()` around `sagittal_wheel_balance.compute()`
   - Add `time.perf_counter()` around `composer.compose()`
   - Add `time.perf_counter()` around telemetry dict construction
   - Add `time.perf_counter()` around torque rate limiting + apply
   - Emit per-step timing breakdown to a JSONL log when `--profile-controller` flag is set
   - Do NOT change any control logic

2. **Identify duplicate estimator calls:**
   - Search for all call sites of `centroidal_estimator.estimate()` and `capture_estimator.update()`
   - Verify if the two calls per step produce identical results (same inputs, same return value)
   - Check for side effects (state mutation in estimator objects)

3. **Remove duplicates** (if safe):
   - Call each estimator once per step, reuse the result
   - If a second call has side effects, preserve those side effects in a separate state update
   - Apply to the Python path (not JAX-specific — this benefits both backends)

4. **Benchmark savings:**
   - Run `--profile-controller` before and after duplicate removal
   - Compare per-step controller time
   - Record in `docs/validation/stage1_estimator_profile_report.md`

5. **Profile the full controller breakdown:**
   - Run `scripts/simulate_hierarchical_controller.py` with `--profile-controller` for:
     - fixed high_0p480, 2000 steps
     - fixed low_0p330, 2000 steps
     - dynamic ramp_up, 2000 steps
   - Produce `docs/validation/stage1_controller_profile_breakdown.json`

### Tests to Add/Run

- `tests/test_stage1_no_duplicate_calls.py`: assert `estimate()` and `update()` called exactly once per step
- `tests/test_stage1_behavior_unchanged.py`: run Step C (7 cases) with profiled Python path, assert 0 falls, same classifications as K2 baseline
- Existing K2 tests: `tests/test_k2_best_current_promotion.py`, `tests/test_current_best_controller_profile.py`

### Files Modified

- `scripts/simulate_hierarchical_controller.py` — add `--profile-controller` flag, timing instrumentation, remove duplicate estimator calls

### Files Created

- `tests/test_stage1_no_duplicate_calls.py`
- `tests/test_stage1_behavior_unchanged.py`
- `docs/validation/stage1_estimator_profile_report.md`
- `docs/validation/stage1_controller_profile_breakdown.json`

### Acceptance Gate

- [ ] Duplicate estimator calls removed (or documented as necessary with side-effect justification)
- [ ] Python K2 behavior unchanged (Step C: 7 cases, 0 falls)
- [ ] Per-step controller time measured before and after
- [ ] Full per-component profile breakdown published
- [ ] `--profile-controller` flag works
- [ ] All existing K2 tests pass

### Rollback/Blocker Condition

- If duplicate removal causes any behavior change → revert and document the side effect
- If profiling reveals a single component dominates >80% of time → flag for targeted optimization

### Expected Output Artifact

`docs/validation/stage1_controller_profile_breakdown.json` — per-component timing for the Python K2 controller across 3 scenarios.

---

## Stage 2: JAX Notch + Torque Limiter Parity

**Goal:** Port the two most time-critical numeric components (notch filter + torque
composer) to JAX and verify exact parity.

### Implementation Tasks

1. **Add JAX-compatible pure functions to `signal_filters.py`:**
   - `biquad_notch_coefficients(fs_hz, fc_hz, Q) -> tuple[b0, b1, b2, a1, a2]`
   - `biquad_notch_update(x, x1, x2, y1, y2, b0, b1, b2, a1, a2) -> (y, x1_new, x2_new, y1_new, y2_new)`
   - `smoothstep_gate_jax(value, start, end) -> float`
   - All pure functions, no class instances, no mutable state
   - Match the existing `BiquadNotchFilter.update()` exactly (DF2T form)
   - Use `jnp` math operations for JAX compatibility (but keep callable from Python too)

2. **Port torque composer to JAX:**
   - Create `_torque_composer_jax(tau_sum, tau_prev, torque_limits, rate_limits, smoothing_alpha) -> (tau_final, clip_flags)` in `k2_jax_controller.py`
   - Match existing `BalanceCoreTorqueComposer.compose()` logic exactly
   - Include: per-joint clip, per-joint rate limit, smoothing

3. **Create stub `k2_jax_controller.py`:**
   - Define `K2_JAX_PARAMS_FIELDS` (provisional, notch + composer params only)
   - Define `pack_notch_params(...)` and `pack_composer_params(...)` helpers
   - Define `K2_JAX_STATE_FIELDS` (provisional, notch state + prev_tau only)
   - Define `pack_state(...)` / `unpack_state(...)` helpers

4. **Component parity tests:**
   - Random input sweep: 10,000 random (pitch_rate, notch_state) pairs
   - Compare `BiquadNotchFilter.update()` vs `biquad_notch_update()` output
   - Compare `BalanceCoreTorqueComposer.compose()` vs JAX composer output
   - Assert max absolute difference < 1e-10 for float64

### Tests to Add/Run

- `tests/test_k2_jax_component_parity.py::TestNotchCoefficientParity` — verify JAX coefficients match Python coefficients (fs=100, fc=2.5, Q=2.0)
- `tests/test_k2_jax_component_parity.py::TestNotchUpdateParity` — 10k random inputs, max diff < 1e-10
- `tests/test_k2_jax_component_parity.py::TestNotchStreamParity` — 1000-step stream with identical input sequence, final state diff < 1e-10
- `tests/test_k2_jax_component_parity.py::TestSmoothstepGateParity` — boundary and random values
- `tests/test_k2_jax_component_parity.py::TestTorqueComposerParity` — random 10-dim torque inputs, per-joint diff < 1e-10
- `tests/test_k2_jax_component_parity.py::TestStatePackUnpackRoundtrip` — pack→unpack preserves all values
- `python -m py_compile wheeled_biped/controllers/k2_jax_controller.py`

### Files Modified

- `wheeled_biped/controllers/signal_filters.py` — add pure JAX-compatible functions (no class changes)

### Files Created

- `wheeled_biped/controllers/k2_jax_controller.py` — stub with params/state helpers, notch + composer JAX implementations
- `tests/test_k2_jax_component_parity.py` — notch + composer parity tests (additional tests added in later stages)

### Acceptance Gate

- [ ] `biquad_notch_update()` output matches `BiquadNotchFilter.update()` to < 1e-10 for 10k random inputs
- [ ] Notch stream parity: 1000-step state evolution identical to < 1e-10
- [ ] Smoothstep gate parity: all boundary values match
- [ ] Torque composer parity: per-joint diff < 1e-10 for random inputs
- [ ] State pack/unpack roundtrip preserves all values
- [ ] `py_compile` passes on new file
- [ ] Existing `BiquadNotchFilter` and `BalanceCoreTorqueComposer` unchanged

### Rollback/Blocker Condition

- If any parity test fails at 1e-10 → fix before proceeding. Do not relax tolerance.
- If JAX function signature cannot match Python semantics exactly → document the difference and get approval.

### Expected Output Artifact

`tests/test_k2_jax_component_parity.py` — passing with notch and composer test classes.

---

## Stage 3: JAX Sagittal/Support/Posture Active Path Parity

**Goal:** Port every K2-active sub-component to JAX and verify component-level parity.

### Implementation Tasks

1. **Port K2-active sagittal torque assembly to JAX:**
   - Height scheduling: `smoothstep`-based `scheduled_k_position()`, `scheduled_k_wheel_velocity()`, `scheduled_kd_pitch()`, `scheduled_max_position_tau()`, `scheduled_k_velocity()`, `scheduled_k_support_velocity()`, `scheduled_kp_pitch()`, `scheduled_max_tau_wheel()`
   - Pitch reference: `interpolate_pitch_ref_offset()` (piecewise-linear, K2 uses height schedule)
   - Support-position outer loop: `compute_outer_loop_pitch_ref()` (PD with deadband, rate limit, low-pass)
   - Sagittal torque assembly: tau_pitch, tau_pitch_rate, tau_position, tau_sagittal_velocity, tau_support_velocity, wheel velocity damping
   - All in `k2_jax_controller.py` as pure JAX functions

2. **Port calibrated outer loop to JAX:**
   - Pre-evaluate SciPy PCHIP functions on fine grid (start 1000 pts)
   - Pack grid values + heights into params_flat
   - Implement JAX piecewise-linear interpolation with clamping
   - Verify max error < 1e-6 at 10,000 random heights (empirical)

3. **Port physics equilibrium feedforward to JAX:**
   - Same grid-based approach as calibrated outer loop
   - Height lookup → per-wheel equilibrium torque

4. **Port low-band support to JAX:**
   - Gaussian height gate, Kp compute, theta_ref clamp
   - Verify gate behavior: active only near 0.320 m

5. **Port shape_posture controller to JAX:**
   - PD control on hip-yaw, hip-pitch, knee joints
   - Hip-yaw divergence profile (HY2-DIV)
   - Support feedforward computation
   - Pure JAX function operating on joint position/velocity arrays

6. **Port lateral roll balance to JAX:**
   - Hip roll PD + smoothing
   - Verify smoothing state exists in Python path; if not, remove from state

7. **Port yaw controller to JAX:**
   - Yaw PD + differential wheel torque

8. **Port mode-hip-yaw-divergence to JAX:**
   - Kp=10.0, Kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80
   - Audit which state fields actually exist in Python path
   - Remove `[AUDIT]` fields that lack Python counterparts

9. **Component parity tests** for each item above:
   - Random input sweep per component
   - Compare Python output vs JAX output
   - Assert max diff < 1e-8 for float64

### Tests to Add/Run

Extend `tests/test_k2_jax_component_parity.py`:
- `TestHeightSchedulingParity` — scheduled gains match Python
- `TestPitchRefOffsetParity` — height schedule interpolation match
- `TestOuterLoopParity` — PD + deadband + rate limit + low-pass match
- `TestCalibratedOuterLoopParity` — grid interpolation vs PCHIP, max error < 1e-6 at 10k points
- `TestPhysicsFFParity` — grid interpolation vs PCHIP, max error < 1e-6
- `TestLowBandSupportParity` — gate + Kp match
- `TestShapePostureParity` — PD torque match per joint
- `TestLateralRollParity` — torque match
- `TestYawControllerParity` — torque match
- `TestModeDivParity` — torque match, state audit
- `TestSagittalTorqueAssemblyParity` — combined torque match

### Files Modified

- `wheeled_biped/controllers/k2_jax_controller.py` — add all sub-component JAX implementations, extend params/state fields
- `tests/test_k2_jax_component_parity.py` — add all component parity test classes

### Files Created

- `docs/validation/k2_jax_pchip_grid_verification.json` — empirical grid error measurements

### Acceptance Gate

- [ ] Every K2-active sub-component has a passing parity test (max diff < 1e-8)
- [ ] PCHIP grid verification passes (max error ≤ 1e-6 at chosen grid resolution)
- [ ] State `[AUDIT]` fields resolved: each field either confirmed or removed
- [ ] State layout stabilized: `K2_JAX_STATE_FIELDS` finalized (no more `[AUDIT]` markers)
- [ ] All existing Python controller files unchanged

### Rollback/Blocker Condition

- If any sub-component parity test fails at 1e-8 tolerance → fix before proceeding
- If PCHIP grid fails at 5000 points → implement exact Hermite port
- If a state field cannot be confirmed or removed → escalate for design review

### Expected Output Artifact

`docs/validation/k2_jax_pchip_grid_verification.json` — per-function grid resolution and max error.

---

## Stage 4: Full Legacy K2 Torque Assembly Parity

**Goal:** Integrate all sub-components into `k2_jax_controller_step()` and verify full-step
and multi-step parity against the Python K2 reference.

### Implementation Tasks

1. **Assemble `k2_jax_controller_step()`:**
   - Compose all Stage 3 sub-functions in sequence (see spec §3.4)
   - Implement `pack_input_flat()` from MuJoCo extracts
   - Implement `pack_params_flat()` from `SagittalAuthoritySchedule` + constructor args
   - Implement `pack_state_flat()` / `unpack_state_flat()`
   - Implement `pack_diag_flat()` inside JIT
   - Implement `k2_jax_diag_flat_to_dict()` outside JIT (Python mapper)
   - Define `K2_JAX_DIAG_FIELDS` (full field list)
   - Wire `@jax.jit` with `static_argnames=()` (all params in params_flat)

2. **Full-step parity harness:**
   - Create `scripts/compare_k2_python_vs_jax_step.py`
   - Run Python K2 controller for N steps, capturing:
     - All inputs to `sagittal_wheel_balance.compute()` before each call
     - All inputs to `shape_posture.compute()` before each call
     - All sub-component inputs
     - Controller state before each step
   - Replay captured inputs through `k2_jax_controller_step()`
   - Compare outputs step-by-step:
     - tau: max_abs_diff, RMS diff per joint
     - state: max_abs_diff per field
     - diag: max_abs_diff per field
     - clipping/saturation flag mismatch count
   - Save comparison to `outputs/k2_jax_parity/comparison.csv`

3. **Multi-step parity:**
   - Run Python and JAX in lockstep from identical initial state
   - Feed identical MuJoCo-extracted inputs each step
   - Track state divergence over 1000+ steps
   - Assert per-field drift < 1e-6 per 1000 steps
   - If drift exceeds threshold → investigate and fix

4. **Run scenarios:**
   - fixed high_0p480 (1000 steps)
   - fixed low_0p330 (1000 steps)
   - push high_0p480 forward 90N (500 steps)
   - dynamic ramp_up 0.33→0.48 (1000 steps)
   - dynamic gate_chatter (1000 steps)

5. **Branch activity audit:**
   - Create `tests/test_k2_jax_branch_activity_audit.py`
   - Instrument Python K2 controller to log branch execution + output contribution
   - Run all 5 scenario categories
   - Classify every branch (ENABLED_ACTIVE / ENABLED_GATED_ZERO / DISABLED_INACTIVE)
   - Assert no UNEXPECTED_ACTIVE branches
   - Assert all DISABLED_INACTIVE branches never execute
   - Produce `docs/validation/k2_branch_activity_audit.json`

6. **State field audit:**
   - Map every `K2_JAX_STATE_FIELDS` entry to Python source (file:line:attribute)
   - Remove any field without a Python counterpart
   - Produce `docs/validation/k2_jax_state_field_audit.json`

7. **Telemetry field audit:**
   - Map every `K2_JAX_DIAG_FIELDS` entry to Python telemetry source
   - Identify Python-only fields (whitelist)
   - Ensure no silent field drops
   - Produce `docs/validation/k2_jax_telemetry_field_audit.json`

### Tests to Add/Run

- `tests/test_k2_jax_step_parity.py::TestFullStepParityFixedHigh` — tau diff < 1e-5 per joint
- `tests/test_k2_jax_step_parity.py::TestFullStepParityFixedLow` — tau diff < 1e-5 per joint
- `tests/test_k2_jax_step_parity.py::TestFullStepParityPush` — tau diff < 1e-5 per joint
- `tests/test_k2_jax_step_parity.py::TestFullStepParityDynamicRamp` — tau diff < 1e-5
- `tests/test_k2_jax_step_parity.py::TestFullStepParityGateChatter` — tau diff < 1e-5
- `tests/test_k2_jax_step_parity.py::TestMultiStepParity` — state drift < 1e-6 per field per 1000 steps
- `tests/test_k2_jax_step_parity.py::TestStateFieldAudit` — all fields have Python source
- `tests/test_k2_jax_step_parity.py::TestDiagFieldAudit` — all fields mapped or whitelisted
- `tests/test_k2_jax_step_parity.py::TestDiagFlatToDictRoundtrip` — reconstruction matches
- `tests/test_k2_jax_branch_activity_audit.py` — all tests (see §7 of spec)
- Existing K2 tests must still pass

### Files Modified

- `wheeled_biped/controllers/k2_jax_controller.py` — complete implementation with full step function, all pack/unpack, diag mapper

### Files Created

- `scripts/compare_k2_python_vs_jax_step.py`
- `tests/test_k2_jax_step_parity.py`
- `tests/test_k2_jax_branch_activity_audit.py`
- `docs/validation/k2_branch_activity_audit.json`
- `docs/validation/k2_jax_state_field_audit.json`
- `docs/validation/k2_jax_telemetry_field_audit.json`

### Acceptance Gate

- [ ] Full-step tau diff < 1e-5 per joint across all 5 scenarios
- [ ] Multi-step state drift < 1e-6 per field per 1000 steps
- [ ] Branch audit: 0 UNEXPECTED_ACTIVE, all DISABLED_INACTIVE confirmed
- [ ] State field audit: 0 unconfirmed fields
- [ ] Telemetry field audit: 0 silently dropped fields
- [ ] Diag flat→dict roundtrip preserves all values
- [ ] `k2_jax_controller_step()` compiles with `@jax.jit` without error
- [ ] All existing K2 tests pass

### Rollback/Blocker Condition

- If full-step tau diff exceeds 1e-5 for any joint → fix before proceeding
- If multi-step state drift exceeds 1e-6 → investigate and fix IIR/integral state
- If branch audit finds UNEXPECTED_ACTIVE → audit the branch and either fix classification or acknowledge unexpected behavior

### Expected Output Artifact

`outputs/k2_jax_parity/comparison.csv` — per-step Python vs JAX comparison for all scenarios.

---

## Stage 5: Integrate `--controller-backend jax`

**Goal:** Add CLI flag, JAX fast-path in simulation loop, and warmup compilation.

### Implementation Tasks

1. **Add `--controller-backend` argument:**
   - In `scripts/simulate_hierarchical_controller.py` argument parser
   - Choices: `python` (default), `jax`
   - Add `--profile-controller` flag (from Stage 1, make permanent)
   - Print selected backend at startup

2. **Implement JAX fast-path in `simulation_step()`:**
   - Initialize JAX params once at startup (compile `params_flat` from controller state)
   - Initialize JAX state once at startup (pack from initial Python state)
   - Compile `k2_jax_controller_step` once (`jax.jit`) at startup, print compile time
   - Per-step:
     - Extract numeric inputs from MuJoCo data (same extraction as Python path)
     - Pack `input_flat`
     - Call compiled `jax_step(jax_state, input_flat, params_flat)`
     - Unpack `tau` → `np.array` → `mj_data.ctrl[:]`
     - Unpack `jax_state` → update reference state
     - If telemetry enabled: map `diag_flat` → telemetry dict
   - Guard Python path with `if backend == "python":`
   - Guard JAX path with `if backend == "jax":`

3. **Warmup and timing:**
   - Run JAX step once before visual loop starts (warmup / trigger compilation)
   - Print: `JIT compile time: X.XX s`
   - Print: `JAX warmup step time: X.XX ms`
   - In timing output, separate: `controller_backend`, `jit_compile_time`, `per_step_controller_ms`

4. **Smoke test:**
   - Run with `--controller-backend jax` for 100 steps, headless
   - Assert no crash, no NaN, torques within limits
   - Assert rollout completes

5. **Python path verification:**
   - Run with `--controller-backend python` (default) — must produce identical behavior to pre-Stage-5 Python path
   - Run Step C (7 cases) with Python backend → 0 falls, same classifications

### Tests to Add/Run

- `tests/test_k2_jax_backend_cli.py::TestBackendFlagParses` — `--controller-backend python` and `--controller-backend jax` both parse
- `tests/test_k2_jax_backend_cli.py::TestBackendDefaultIsPython` — no flag → backend is `python`
- `tests/test_k2_jax_backend_cli.py::TestJaxBackendSmoke` — 100-step headless rollout, no crash, no NaN
- `tests/test_k2_jax_backend_cli.py::TestJaxCompilesOnce` — verify JIT called exactly once
- `tests/test_k2_jax_backend_cli.py::TestPythonBackendUnchanged` — Step C still passes with default backend
- `tests/test_k2_jax_backend_cli.py::TestProfileControllerFlag` — `--profile-controller` emits timing JSONL
- Existing K2 tests must pass with default backend

### Files Modified

- `scripts/simulate_hierarchical_controller.py` — add `--controller-backend`, `--profile-controller`, JAX fast-path, warmup

### Files Created

- `tests/test_k2_jax_backend_cli.py`

### Acceptance Gate

- [ ] `--controller-backend python` is the default (no flag = python)
- [ ] `--controller-backend jax` produces valid 100-step rollout (no crash, no NaN)
- [ ] JIT compiles exactly once at startup
- [ ] Compile time printed separately from per-step time
- [ ] Python backend produces identical behavior to pre-Stage-5
- [ ] All existing K2 tests pass (backend=python)
- [ ] `--profile-controller` flag works for both backends

### Rollback/Blocker Condition

- If JAX path crashes on any scenario → fix before proceeding
- If Python path behavior changed → revert and fix guard
- If JIT recompiles during rollout → fix dynamic shape or branching

### Expected Output Artifact

JAX warmup output:
```
Controller backend: jax
JIT compile time: X.XX s
JAX warmup step: X.XX ms
```

---

## Stage 6: C/D/E + Push + Dynamic Validation on JAX Backend

**Goal:** Run the full K2 validation suite with `--controller-backend jax` and prove
equivalence or improvement vs Python K2.

### Implementation Tasks

1. **Step C validation (7 cases):**
   - Run each height case with `--controller-backend jax` for 2000 steps
   - Compare survival, fall rate, LF/WIP metrics vs Python K2 baseline
   - Classification: EQUIVALENT, BETTER, or REGRESSION
   - **Gate:** 0 falls, 0 REGRESSION classifications

2. **Step D push matrix (24 runs):**
   - 12 push conditions (3 heights × 2 directions × 2 magnitudes) × 2 backends = 24 runs
   - Backends: `--controller-backend python` and `--controller-backend jax`
   - Profile is the same in both: `--vd-sagittal-authority-profile k2_notch_low_q_v1`
   - Compare push recovery metrics: Python K2 backend vs JAX K2 backend
   - **Gate:** 0 falls on JAX backend, 0 REGRESSION classifications

3. **Step E validation (10 heights):**
   - Run each height with `--controller-backend jax`
   - **Gate:** 0 falls, 0 REGRESSION classifications

4. **Single-push visual/recovery:**
   - high_0p480 forward 90N with `--controller-backend jax`
   - high_0p480 backward 90N with `--controller-backend jax`
   - **Gate:** 0 falls, recovery metrics EQUIVALENT or BETTER

5. **Dynamic height validation (5 scenarios):**
   - ramp_up, ramp_down, up_down_cycle, gate_dwell, gate_chatter
   - All with `--controller-backend jax`
   - **Gate:** 0 falls, height tracking EQUIVALENT or BETTER

6. **Cross-backend comparison report:**
   - For each scenario, compare JAX vs Python on:
     - survival_rate, fall_rate
     - pitch_RMS_deg, roll_RMS_deg
     - height_RMSE_m
     - wheel_speed_RMS_rad_s
     - torque_RMS_Nm
     - hip_yaw_max_abs_rad
     - LF_pitch_power (where applicable)
   - Produce `docs/validation/k2_jax_validation_cross_backend_comparison.json`

### Tests to Add/Run

- **Discover and reuse existing validation runners first.** Known likely existing runners:
  - `scripts/validate_k2_step_c_e_fixed_height.py`
  - `scripts/validate_k2_step_d_push_matrix.py`
  - `scripts/validate_k2_dynamic_height_gate_crossing.py`
  - `scripts/validate_k2_post_promotion_long_run.py`
- If an existing runner does not accept `--controller-backend`, extend it without changing validation semantics
- If no runner exists for a suite, create a small backend-aware wrapper that reproduces the existing validated matrix exactly
- All existing K2 tests must pass with backend=python
- `tests/test_k2_jax_backend_cli.py::TestJaxStepCPasses` — 7 cases, 0 falls
- `tests/test_k2_jax_backend_cli.py::TestJaxStepDPasses` — push matrix subset
- `tests/test_k2_jax_backend_cli.py::TestJaxStepEPasses` — 10 heights, 0 falls

### Files Modified

- (potentially) validation scripts if they need `--controller-backend` flag

### Files Created

- `docs/validation/k2_jax_validation_cross_backend_comparison.json`
- `docs/validation/k2_jax_step_c_validation_report.md`
- `docs/validation/k2_jax_step_d_validation_report.md`
- `docs/validation/k2_jax_step_e_validation_report.md`
- `docs/validation/k2_jax_dynamic_height_validation_report.md`

### Acceptance Gate

- [ ] Step C: 7/7 cases pass, 0 falls, 0 REGRESSION
- [ ] Step D: 24/24 push runs pass, 0 falls, 0 REGRESSION
- [ ] Step E: 10/10 heights pass, 0 falls, 0 REGRESSION
- [ ] Single-push: both directions pass, 0 falls
- [ ] Dynamic height: 5/5 scenarios pass, 0 falls
- [ ] No hip-yaw violation >0.35 rad in any run
- [ ] No LF/WIP regression
- [ ] No hidden torque/WBC
- [ ] All classifications EQUIVALENT or BETTER (no REGRESSION)

### Rollback/Blocker Condition

- If ANY validation case shows fall → blocker. Fix parity before proceeding.
- If ANY validation case shows REGRESSION → blocker. Investigate and fix.
- If >0 hip-yaw violations → blocker.
- If any LF/WIP regression → blocker.
- **Backend default stays `python` regardless of outcome.**

### Expected Output Artifact

`docs/validation/k2_jax_validation_cross_backend_comparison.json` — per-scenario JAX vs Python comparison.

---

## Stage 7: Realtime Visual Benchmark

**Goal:** Measure JAX backend performance and classify realtime capability.

### Implementation Tasks

1. **Create benchmark script:**
   - `scripts/benchmark_k2_jax_controller.py`
   - Arg: `--backend {python,jax}`, `--scenarios [...]`, `--steps N`, `--warmup-steps N`
   - Headless by default (omit `--visual`); add `--visual` only for visual benchmark

2. **Benchmark scenarios:**
   - A. Python backend, headless (baseline)
   - B. JAX backend, cold (first-step compile included)
   - C. JAX backend, warm steady-state, headless
   - D. JAX backend, warm steady-state, visual (if GUI available)
   - E. Python backend, visual (if GUI available)

3. **Measure per-component timing (JAX path):**
   - MuJoCo state extraction: `time.perf_counter()` around `mj_data` → numpy
   - Input packing: numpy → jnp array construction
   - JAX controller step: `block_until_ready()` + timer
   - Tau conversion: jnp → numpy for `mj_data.ctrl`
   - Telemetry diag mapping: `k2_jax_diag_flat_to_dict()` time
   - Viewer sync: `viewer.sync()` time
   - Telemetry write: CSV/JSON write time

4. **Metrics collected:**
   - Controller ms/step: mean, p50, p95, p99
   - End-to-end ms/step: mean, p50, p95, p99
   - Realtime factor: control_dt / end_to_end_ms
   - JIT compile time (one-time)
   - Recompilation count (must be 0)
   - Memory usage at start and after 10k steps

5. **Classify outcome:**
   - `JAX_CONTROLLER_REALTIME_PASS` / `JAX_CONTROLLER_REALTIME_FAIL`
   - `END_TO_END_REALTIME_PASS` / `END_TO_END_REALTIME_BLOCKED_BY_*`
   - Produce `docs/validation/k2_jax_realtime_benchmark_report.md`

### Tests to Add/Run

- `tests/test_k2_jax_backend_cli.py::TestBenchmarkScriptRuns` — benchmark completes without error
- `tests/test_k2_jax_backend_cli.py::TestNoRecompilation` — `jax.jit` call count = 1
- `tests/test_k2_jax_backend_cli.py::TestNoMemoryLeak` — memory delta < 1 MB over 10k steps

### Files Created

- `scripts/benchmark_k2_jax_controller.py`
- `docs/validation/k2_jax_realtime_benchmark_report.md`

### Acceptance Gate

- [ ] JAX warm controller kernel measured (target ≤ 3 ms/step)
- [ ] All boundary costs measured and reported separately
- [ ] End-to-end timing classified (PASS or BLOCKED_BY_*)
- [ ] 0 JIT recompilations during benchmark
- [ ] Memory growth < 1 MB over 10k steps
- [ ] Python backend performance unchanged from Stage 1 baseline
- [ ] Report published with all timing breakdowns

### Rollback/Blocker Condition

- If JAX controller kernel > 3 ms/step → classify as `JAX_CONTROLLER_REALTIME_FAIL`, document bottleneck
- If end-to-end blocked by viewer/MuJoCo → classify as `END_TO_END_REALTIME_BLOCKED_BY_VIEWER_OR_MUJOCO`
- If end-to-end blocked by telemetry → classify as `END_TO_END_REALTIME_BLOCKED_BY_TELEMETRY_IO`
- **Do not claim realtime unless measured.** Do not change K2 behavior.
- **Backend default stays `python` regardless of benchmark outcome.**

### Expected Output Artifact

`docs/validation/k2_jax_realtime_benchmark_report.md` — full timing breakdown, classification, and recommended realtime visual command.

---

## Stage Gates Summary

| Stage | Key Gate | Blocker If |
|-------|----------|-----------|
| 1 | Duplicate calls removed, profile published | Behavior change from removing duplicates |
| 2 | Notch + composer parity < 1e-10 | Any parity test fails |
| 3 | All sub-components parity < 1e-8, PCHIP grid verified, state audited | Any component fails, PCHIP > 1e-6 at 5000 pts |
| 4 | Full-step tau < 1e-5, multi-step drift < 1e-6/1k steps, branch audit clean | Any parity failure, UNEXPECTED_ACTIVE branch |
| 5 | `--controller-backend jax` works, Python path unchanged | JAX crashes, Python regresses, JIT recompiles |
| 6 | Step C/D/E + push + dynamic: 0 falls, 0 REGRESSION | Any fall, any REGRESSION, any hip-yaw violation |
| 7 | Controller benchmark classified, report published | Cannot measure (no JAX), memory leak |

---

## Unresolved Risks Before Coding

1. **PCHIP grid resolution:** May need >1000 points or full Hermite port — determined empirically in Stage 3.
2. **State field count:** `[AUDIT]` fields may not exist in Python path — resolved in Stage 3/4.
3. **JIT compile time:** Unknown until measured in Stage 5. If >60s, may need to split or pre-compile.
4. **MuJoCo extraction overhead:** Unknown until benchmarked in Stage 7. May dominate if >5 ms.
5. **Telemetry diag mapper cost:** May be significant for full mapping in realtime — decimated mode may be required.
6. **Duplicate estimator calls:** May have undiscovered side effects — Stage 1 profiling will reveal.
7. **float64 vs float32:** float64 may be slower than needed for realtime. float32 variant only after parity passes.
8. **Viewer/render bottleneck:** External to controller — JAX can't fix this.

---

## File Manifest

### Created (11 files)

| Stage | File |
|-------|------|
| 1 | `tests/test_stage1_no_duplicate_calls.py` |
| 1 | `tests/test_stage1_behavior_unchanged.py` |
| 1 | `docs/validation/stage1_estimator_profile_report.md` |
| 1 | `docs/validation/stage1_controller_profile_breakdown.json` |
| 2 | `wheeled_biped/controllers/k2_jax_controller.py` |
| 2 | `tests/test_k2_jax_component_parity.py` |
| 4 | `scripts/compare_k2_python_vs_jax_step.py` |
| 4 | `tests/test_k2_jax_step_parity.py` |
| 4 | `tests/test_k2_jax_branch_activity_audit.py` |
| 5 | `tests/test_k2_jax_backend_cli.py` |
| 7 | `scripts/benchmark_k2_jax_controller.py` |

### Modified (2 files)

| Stage | File | Change |
|-------|------|--------|
| 1,5 | `scripts/simulate_hierarchical_controller.py` | `--profile-controller`, remove dup calls, `--controller-backend`, JAX fast-path |
| 2 | `wheeled_biped/controllers/signal_filters.py` | Add `biquad_notch_coefficients()`, `biquad_notch_update()`, `smoothstep_gate_jax()` |

### Validation Reports (10 files, Stages 3–7)

| Stage | File |
|-------|------|
| 3 | `docs/validation/k2_jax_pchip_grid_verification.json` |
| 4 | `docs/validation/k2_branch_activity_audit.json` |
| 4 | `docs/validation/k2_jax_state_field_audit.json` |
| 4 | `docs/validation/k2_jax_telemetry_field_audit.json` |
| 6 | `docs/validation/k2_jax_validation_cross_backend_comparison.json` |
| 6 | `docs/validation/k2_jax_step_c_validation_report.md` |
| 6 | `docs/validation/k2_jax_step_d_validation_report.md` |
| 6 | `docs/validation/k2_jax_step_e_validation_report.md` |
| 6 | `docs/validation/k2_jax_dynamic_height_validation_report.md` |
| 7 | `docs/validation/k2_jax_realtime_benchmark_report.md` |

### Phase 9 Final Report

| File |
|------|
| `docs/validation/k2_jax_jit_controller_parity_and_realtime_report.md` |

---

*Implementation plan complete. Ready for execution upon approval. No code has been written.*
*Backend default: `python`. K2 behavior: unchanged.*
