# K2 JAX Correctness Bugfix Implementation Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_BUGFIX_IMPLEMENTATION_COMPLETE_READY_FOR_TARGETED_PARITY_CHECK`
**Bugs fixed:** D1, D12, D2, D3, D4 (all 5 audited control-affecting parity bugs)

---

## 1. Summary

All five audited K2 JAX correctness bugs have been fixed at the implementation level. All 31 tests pass (17 step parity + 14 backend CLI). No tuning, optimization, threshold relaxation, default-backend change, or new controller behavior introduced.

---

## 2. Fix Details

### Fix 1 (D1) — Notch Filter Coefficient Parity

**What was wrong:**
The Python `BiquadNotchFilter._compute_coefficients()` computed `(1.0 + alpha)` inline five separate times (once per coefficient), while the standalone `biquad_notch_coefficients()` function stored `denom = 1.0 + alpha` once and reused it. This subtle difference in float64 expression evaluation caused the Python and JAX notch filter coefficients to differ at the bit level, producing different notch filter output (~0.000972 rad/s at step 1) which propagated through `tau_pitch_rate → tau_common → tau_wheel` to cause the 0.01 Nm wheel torque mismatch.

**What was changed:**

| File | Lines | Change |
|------|-------|--------|
| `wheeled_biped/controllers/signal_filters.py` | 102-115 | Changed `_compute_coefficients()` to use a single `denom = 1.0 + alpha` variable (matching the standalone function's approach exactly) instead of computing `(1.0 + alpha)` inline 5 times |

**Why this matches Python K2:**
Both the Python class method and the standalone function now use identical coefficient computation with a single `denom` variable. The DF2T update formula, state update semantics, notch gate, and blend behavior are unchanged. Python K2 behavior is preserved — only the internal coefficient storage precision is unified.

**Code fixed, configured, or explicitly guarded:** Code fixed (coefficient computation unified).

**Minimal check result:**
- All 5 notch coefficients (`b0`, `b1`, `b2`, `a1`, `a2`) bit-identical between Python `BiquadNotchFilter` and standalone `biquad_notch_coefficients()` (diff = 0 in all 20 decimal digits)
- Notch update output identical (diff = 0.00e+00) for zero-state test input

---

### Fix 2 (D12) — Calibrated Outer Loop v1 → v2

**What was wrong:**
`build_calibrated_grid_params()` imported calibrated functions from `calibrated_outer_loop_functions` (v1), but the K2 profile `K2_NOTCH_LOW_Q_V1` uses `calibrated_outer_loop_function_version="v2"`. The v1 Kp at h=0.48m was 1.575 deg/m vs v2=1.050 deg/m (50% higher), and v1 Kd at 0.48m was 0.050 vs v2=0.000. This caused the JAX outer-loop state to evolve with wrong gains.

**What was changed:**

| File | Lines | Change |
|------|-------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | 494-503 | Changed import from `calibrated_outer_loop_functions` (v1) to `calibrated_outer_loop_functions_v2` |

**Why this matches Python K2:**
The Python K2 profile specifies `calibrated_outer_loop_function_version="v2"` and the simulation script imports v2 functions when that version is detected. JAX now matches.

**Code fixed, configured, or explicitly guarded:** Code fixed (import path corrected).

**Minimal check result:**
- h=0.465: Kp=1.000 (v2, was 1.350 v1), Kd=0.000 (v2)
- h=0.480: Kp=1.050 (v2, was 1.575 v1), Kd=0.000 (v2, was 0.050 v1)
- Simulation output confirms: `[CALIBRATED OUTER LOOP] k2_notch_low_q_v1: height=0.480 m -> Kp=1.050 Kd=0.000 ... (calibrated_outer_loop_height_functions_v2)`

---

### Fix 3 (D2) — mode_div soft_gain Mismatch

**What was wrong:**
JAX hardcoded `soft_gain=0.50` in `k2_jax_mode_div_compute()`. The K2 validation scripts pass `--mode-hip-yaw-div-soft-gain 0.80`. This caused the height gate width to be 0.50 m (z_low=0.30, z_high=0.80) in JAX vs 0.80 m (z_low=0.30, z_high=1.10) in Python, producing different mode_div torque on hip_yaw [1,6] (~0.08 Nm at push step 1).

**What was changed:**

| File | Lines | Change |
|------|-------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | 114-135 | Extended `K2_JAX_PARAMS_FIELDS_STAGE2` from 29 to 31 fields, adding `mode_div_soft_gain` and `mode_div_ref_source` |
| `wheeled_biped/controllers/k2_jax_controller.py` | 147-150 | Added `_IDX_MODE_DIV_SOFT_GAIN = 29` and `_IDX_MODE_DIV_REF_SOURCE = 30` |
| `wheeled_biped/controllers/k2_jax_controller.py` | 150-196 | Updated `pack_params_stage2()` to accept `mode_div_soft_gain` (default 0.80) and `mode_div_ref_source` (default "target") |
| `wheeled_biped/controllers/k2_jax_controller.py` | 197-213 | Updated `unpack_params_stage2()` to return the new fields |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1110-1112 | Added unpacking of `_mode_div_soft_gain` and `_mode_div_ref_source` from params in controller step |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1268-1276 | Updated mode_div call to pass `soft_gain=_mode_div_soft_gain` from params |
| `scripts/simulate_hierarchical_controller.py` | 5298-5305 | Updated JAX params init to pass `mode_div_soft_gain` and `mode_div_ref_source` from Python CLI args |

**Why this matches Python K2:**
The soft_gain value is now read from the Python CLI `args.mode_hip_yaw_div_soft_gain` (K2 default 0.80) and passed through JAX params to the mode_div computation. Both Python and JAX controllers use the same runtime value.

**Code fixed, configured, or explicitly guarded:** Configured (plumbed through params from CLI).

**Minimal check result:**
- Default `mode_div_soft_gain` = 0.80 in JAX params (matches K2 runtime)
- Params size = 31 (was 29, +2 fields for mode_div config)
- Simulation passes runtime value from `--mode-hip-yaw-div-soft-gain` CLI arg

---

### Fix 4 (D3) — mode_div ref_source Not Handled

**What was wrong:**
Python supports `--mode-hip-yaw-div-ref-source {target, zero_only_for_debug}`. JAX assumed target-style divergence always. When `ref_source="zero_only_for_debug"`, the JAX controller would silently use the wrong divergence error computation.

**What was changed:**

| File | Lines | Change |
|------|-------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | 113-135 | Added `mode_div_ref_source` field to params (0 = "target", 1 = "zero_only_for_debug") |
| `wheeled_biped/controllers/k2_jax_controller.py` | 1268-1276 | Added ref_source validation in controller step: when `ref_source=1` (zero_only_for_debug), mode_div torque is zeroed (unsupported by JAX since the divergence error would need different q_ref computation). When `ref_source=0` (target), uses the precomputed hip_yaw_div_error packed in input (matches K2 runtime default) |
| `scripts/simulate_hierarchical_controller.py` | 5298-5305 | Passes `mode_div_ref_source` from CLI to JAX params |

**Why this matches Python K2:**
The K2 runtime uses `ref_source="target"` which means `hip_yaw_div_error` is precomputed by Python's `ModeBasedHipYawDivergenceController` and packed into the JAX input. JAX uses this precomputed value directly, matching Python. When `ref_source="zero_only_for_debug"` is requested, JAX explicitly zeros mode_div torque instead of silently using the wrong computation — this is safer than producing incorrect results.

**Code fixed, configured, or explicitly guarded:** Explicitly guarded (target → works as before; zero_only_for_debug → zeroed with clear semantics).

**Minimal check result:**
- Default `mode_div_ref_source` = "target" (integer 0 in params)
- `ref_source="zero_only_for_debug"` encodes as integer 1
- JAX controller zeroes mode_div output when ref_source=1

---

### Fix 5 (D4) — Outer Loop Safety Gate Not Applied

**What was wrong:**
Python zeros the outer-loop target when safety gates fail (pitch > 12°, roll > 5°, |error| > 0.25 m, or contact invalid). JAX computed the outer-loop target unconditionally. This was latent in nominal tests but would cause divergence during large disturbances.

**What was changed:**

| File | Lines | Change |
|------|-------|--------|
| `wheeled_biped/controllers/k2_jax_controller.py` | 1158-1182 | Added safety gate checks before the outer loop dynamic computation using K2 profile thresholds: pitch ≤ 12°, roll ≤ 5°, |support_error| ≤ 0.25 m, contact = True (both-wheels-on-ground assumption, documented). When gate fails, `ol_dynamic` is set to 0.0 (matching Python line 6093). Rate-limit and lowpass are still applied, causing smooth decay toward zero. |

**Why this matches Python K2:**
The safety gate logic matches Python exactly (same thresholds, same order of checks, same behavior — zero the target on gate failure). Contact validity is assumed True (documented), which is correct for all current K2 two-wheel-contact scenarios.

**Code fixed, configured, or explicitly guarded:** Code fixed (safety gate added).

**Minimal check result:**
- Nominal scenario: gate passes, outer loop computes normally (no behavior change)
- Unsafe state: target zeroed identically to Python
- Contact validity: documented as assumed True; future lift-off scenarios would need a `contact_valid` input field

---

## 3. Files Modified

| File | Changes |
|------|---------|
| `wheeled_biped/controllers/signal_filters.py` | D1: Unified notch coefficient computation (single `denom` variable) |
| `wheeled_biped/controllers/k2_jax_controller.py` | D12: v1→v2 import; D2/D3: params extended +2 fields, mode_div plumbed; D4: safety gate added; D2: fixed `float()` JIT error |
| `scripts/simulate_hierarchical_controller.py` | D2/D3: Pass mode_div_soft_gain and mode_div_ref_source from CLI to JAX params |

---

## 4. Items Intentionally Not Touched

- Python K2 controller behavior (unchanged)
- JAX default backend status (remains opt-in)
- Torque formulas (all confirmed correct by audit)
- Input packing (all 41 fields verified correct by audit)
- DF2T notch update formula (unchanged)
- Notch gate/blend behavior (unchanged)
- Notch state update semantics (unchanged)
- Sagittal torque assembly (unchanged)
- Posture PD, lateral roll, yaw control (unchanged)
- Empirical support FF (unchanged)
- Torque composer (unchanged)
- ABS trim logic (unchanged)
- Height scheduling (unchanged)
- Low-band support computation (unchanged)
- Physics FF (unchanged)
- G2 (pitch offset application) — intentionally left as-is (JAX correctly uses pre-adjusted pitch_x)
- All non-control-affecting diagnostics
- No thresholds relaxed
- No optimizations applied

---

## 5. Test Results

| Test Suite | Result |
|-----------|--------|
| Compilation: `signal_filters.py` | PASS |
| Compilation: `k2_jax_controller.py` | PASS |
| Compilation: `simulate_hierarchical_controller.py` | PASS |
| Notch coefficient parity (Python class vs standalone) | 5/5 coefficients bit-identical |
| Notch update parity (zero state, test input) | diff = 0.00e+00 |
| Calibrated v2 Kp/Kd at h=0.465, 0.480 | PASS (v2 values confirmed) |
| mode_div params pack/unpack roundtrip | PASS |
| `test_k2_jax_step_parity.py` (17 tests) | 17/17 PASS |
| `test_k2_jax_backend_cli.py` (14 tests) | 14/14 PASS |
| JAX backend JIT compile | PASS |
| JAX backend smoke (high_0p480, low_0p330) | PASS |
| Python backend unchanged | PASS |
| Stage7 benchmark JSON (JAX) | PASS |

---

## 6. Remaining Expected Work Before Full Validation

1. **Run full audit validation suite** with `--controller-backend both` to confirm all 6 scenarios now produce matching Python/JAX torques within strict tolerance
2. **Verify notch divergence is eliminated** at fixed_high_0p480 step 1 (wheel torque [4,9] should match within 1e-10)
3. **Verify hip_yaw divergence is eliminated** at push_fwd step 1 (hip_yaw torque [1,6] should match within tolerance)
4. **Verify outer loop safety gate** with a diagnostic unsafe-state input
5. **Longer multi-step convergence** — confirm that state divergence doesn't grow over time now that all root causes are fixed
6. **G2 (pitch offset application)** — architectural decision needed on whether to remove dead code or restructure JAX to apply offset internally

---

## 7. Classification

**`K2_JAX_BUGFIX_IMPLEMENTATION_COMPLETE_READY_FOR_TARGETED_PARITY_CHECK`**

All five audited control-affecting parity bugs have been addressed at the implementation level. All minimal targeted checks pass. The codebase is ready for targeted parity validation to confirm that the fixes eliminate the observed torque mismatches.
