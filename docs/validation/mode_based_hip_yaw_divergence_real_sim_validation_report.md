# Mode-Based Hip-Yaw Divergence — Real Simulation Validation Report

**Status:** Real simulation COMPLETE for D4/D5, real-simulation sweep COMPLETE for D4. Step C recheck and full Step D matrix not yet run for the new candidate.
**Date:** 2026-06-22
**Branch:** repo-cleanup-t6j
**Classification:** `MODE_HIP_YAW_DIVERGENCE_REAL_D4_D5_FAIL` — sign is correct, controller is now active, but the antisymmetric hip-yaw torque is too small to bring `hip_yaw_abs_max_rad` below 0.35 at any tested kp/max_torque combination. The base profile (C: low-band v2) and all A/B baselines also exceed 0.35, so the failure is the same one the architecture fix was designed to address: the divergence mode needs much higher authority, and/or a different approach (e.g. wheel-yaw stabilizer combined with mode-based controller) is required.

## 1. Scope

This report covers the work performed in two waves of the
`mode_based_hip_yaw_divergence_ownership_fix` task series:

* **Wave 1 (Tasks 1-10):** Architecture scaffold. Mode math, ownership
  policy, opt-in mode-based hip-yaw divergence controller, telemetry
  fields, and 53 passing unit tests. Stub validators.
* **Wave 2 (this wave):** Replaced stub validators with real-CSV
  parsers. Wired the new opt-in CLI flags into
  `simulate_hierarchical_controller.py`. Added the candidate runner
  `scripts/run_d4_d5_hip_yaw_div_validation.py` and the real-simulation
  sweep runner `scripts/sweep_hip_yaw_divergence_params.py`. Added
  stub-rejection tests. Ran real D4/D5 simulation for the candidate and
  ran a 3-point real-simulation parameter sweep.

## 2. Files changed (this wave)

### Production code

* `scripts/simulate_hierarchical_controller.py`
  * New CLI flags: `--enable-mode-hip-yaw-divergence`,
    `--mode-hip-yaw-div-kp`, `--mode-hip-yaw-div-kd`,
    `--mode-hip-yaw-div-max-torque`, `--mode-hip-yaw-div-soft-limit-rad`,
    `--mode-hip-yaw-div-soft-gain`, `--mode-hip-yaw-div-ref-source`.
  * New telemetry columns: `mode_hip_yaw_div_enabled`, `..._kp`, `..._kd`,
    `..._max_torque`, `..._soft_limit_rad`, `..._soft_gain`,
    `..._ref_source`, `..._height_gate`, `..._tau_left`, `..._tau_right`,
    `..._tau_left_sat`, `..._tau_right_sat`, `..._error`, `..._rate`,
    `..._ref`, `hip_yaw_mode_ownership_violation`.
  * New runtime block in the balance-core loop: opt-in computation of
    antisymmetric hip-yaw torque from the divergence mode and
    injection into `tau_shape_posture_with_yaw` before the composer.

* `wheeled_biped/validation/d4_d5_validation.py`
  * Stub replaced with a CSV parser reading
    `outputs/hip_yaw_push_limit_architecture_fix/d4_d5_validation/d4_d5_metrics.csv`.
    The candidate runner writes to
    `outputs/mode_based_hip_yaw_divergence_real_sim_validation/d4_d5_metrics.csv`
    and that file is also readable through the same parser.

* `wheeled_biped/validation/full_step_d.py`
  * Stub replaced with a CSV parser reading
    `outputs/step_d_all/step_d_all_metrics.csv`.

* `wheeled_biped/validation/step_c_fixed_height_recheck.py`
  * Stub replaced with a CSV parser reading the Step C / fixed-height
    summary CSVs.

* `wheeled_biped/validation/sweep_hip_yaw_divergence_params.py`
  * Stub analytic adjustment removed. Now reads per-candidate
    `telemetry_*.csv` under
    `outputs/mode_based_hip_yaw_divergence_sweep/sweep_*/`.

### Scripts

* `scripts/run_d4_d5_hip_yaw_div_validation.py` (new)
  * D4/D5 push battery for profiles A/B/C/D. Profile D enables
    `--enable-mode-hip-yaw-divergence` and invokes the simulator
    directly with the new opt-in flags.
* `scripts/sweep_hip_yaw_divergence_params.py` (new)
  * Real-simulation sweep over (kp, kd, max_torque, soft_gain) using
    the low-band v2 sagittal profile, D4 push case. Each candidate's
    telemetry lands in its own `sweep_*` directory.

### Tests

* `tests/test_d4_d5_validation.py` — updated for real-simulation parser.
* `tests/test_full_step_d_validation.py` — same.
* `tests/test_step_c_fixed_height_recheck_candidate.py` — same.
* `tests/test_sweep_hip_yaw_divergence_params.py` — replaced analytic
  stub test with missing-directory test.
* `tests/test_final_validation_rejects_stub_source.py` (new) — 9 tests
  that ensure no validator can return a stub.

## 3. Real-simulation results

### 3.1 D4_medium_push_low (low_0p330, 60N, 1000 steps)

| Profile | hip_yaw_abs_max_rad | fell | pitch max deg | roll_rms deg | wbc | hidden | ownership |
|---|---:|---|---:|---:|---:|---:|---:|
| A (B2v2) | 0.4074 | False | 13.84 | 0.94 | 0 | 0 | 0 |
| B (PFF) | 0.4048 | False | 13.63 | — | 0 | 0 | 0 |
| C (low-band v2) | 0.4076 | False | 13.62 | — | 0 | 0 | 0 |
| **D (mode_hip_yaw_div v1)** | **0.4045** | False | 13.14 | 0.92 | 0 | 0 | 0 |

* D4 result for D: `hip_yaw_abs_max_rad = 0.4045` — **FAIL** (>= 0.35).
* The mode-based controller was active (gate=1.0, tau_left up to 2.0
  Nm at saturation, 471/999 samples saturated at max_torque), but the
  antisymmetric hip-yaw torque is still too small to bring the
  divergence mode under control in 1000 steps.
* No WBC, no hidden torque, no ownership violation, no fall.
* Support recovery (max_abs) improved vs C: 0.272 m vs 0.318 m.

### 3.2 D5_large_push_high (high_0p480, 90N, 1000 steps)

| Profile | hip_yaw_abs_max_rad | fell | pitch max deg | roll_rms deg | wbc | hidden | ownership |
|---|---:|---|---:|---:|---:|---:|---:|
| A (B2v2) | 0.4018 | False | 13.85 | 6.84 | 0 | 0 | 0 |
| B (PFF) | 0.4030 | False | 14.94 | — | 0 | 0 | 0 |
| C (low-band v2) | 0.4030 | False | 14.94 | — | 0 | 0 | 0 |
| **D (mode_hip_yaw_div v1)** | **0.3803** | False | 14.90 | 1.83 | 0 | 0 | 0 |

* D5 result for D: `hip_yaw_abs_max_rad = 0.3803` — **FAIL** (>= 0.35).
* D is the best of the four profiles on D5 but still above the gate.
* No WBC, no hidden torque, no ownership violation, no fall.

### 3.3 Real-simulation parameter sweep (D4 only)

| kp | kd | max_torque | soft_gain | hip_yaw_abs_max_rad | sat rows | fell |
|---:|---:|---:|---:|---:|---:|---|
| 1.0 | 0.10 | 1.0 | 0.20 | 0.4054 | 0/999 | False |
| 2.0 | 0.10 | 1.5 | 0.25 | 0.4044 | 338/999 | False |
| 5.0 | 0.20 | 2.0 | 0.25 | 0.4045 | 471/999 | False |

Increasing kp from 1.0 to 5.0 and max_torque from 1.0 to 2.0 did not
reduce `hip_yaw_abs_max_rad` below 0.40. The controller is active and
saturates, but the divergence is not corrected.

## 4. Failure analysis

The mode-based controller is correctly wired and active:

* `mode_hip_yaw_div_enabled = True` for profile D.
* Height gate activates at low height (gate=1.0 at h=0.335).
* Reference source is `target` (the posture target).
* Antisymmetric torque is computed with the correct sign (positive
  `div_error` -> left negative, right positive).
* Torque is clipped to `max_torque`.

But the antisymmetric hip-yaw torque alone cannot cancel the
divergence mode under sustained push. Possible reasons:

1. **Authority too low.** At kp=5.0, max_torque=2.0 Nm, the controller
   saturates for 47% of the run, yet the divergence still drifts to
   ~0.4 rad. The hip-yaw joint authority may need to be much larger
   (e.g. 5-10 Nm) — but the existing `YawController` already has
   5 Nm max, and the gate also caps the wheel-yaw stabilizer.
2. **Wrong actuator for divergence.** The wheel-yaw stabilizer applies
   an antisymmetric wheel torque, which is much more effective at
   producing yaw rotation than hip-yaw torque (mechanical advantage).
   Combining the mode-based controller with the wheel-yaw stabilizer
   may be required, but that is the "next architecture" not the
   current fix.
3. **Push cadence not matched to controller bandwidth.** With 60N
   pushes every 150 steps, the system has 150 steps to recover before
   the next push. The mode-based PD may not be tuned aggressively
   enough to converge in that window.

The D4 sweep shows that the response plateaus around 0.404 rad — i.e.
the controller has done all it can with hip-yaw authority at this
height. The next step is to combine the mode-based controller with the
wheel-yaw stabilizer, but that is outside the scope of this fix.

## 5. Existing real-simulation data

The following real-simulation outputs are available on disk and have
been used by the parsers in this wave:

* `outputs/hip_yaw_push_limit_architecture_fix/d4_d5_validation/d4_d5_metrics.csv`
  * 8 rows: D4 and D5 for profiles A, B, C, D (D = old wheel-yaw
    stabilizer candidate, NOT the new divergence candidate).
* `outputs/step_d_all/step_d_all_metrics.csv`
  * 18 rows: D1-D6 for profiles A, B, C. No D row.
* `outputs/physics_ff_step_c_low_band_support_v1_full_step_c/`
  * Step C + fixed-height summary for A, B, C. No D row.
* `outputs/mode_based_hip_yaw_divergence_real_sim_validation/d4_d5_metrics.csv`
  * 8 rows: D4 and D5 for profiles A, B, C, D. D = new divergence
    candidate.
* `outputs/mode_based_hip_yaw_divergence_sweep/sweep_*/telemetry_*.csv`
  * 3 candidates with real telemetry.

## 6. Test results

```
tests/test_d4_d5_validation.py ....                                [4/4]
tests/test_full_step_d_validation.py .....                         [5/5]
tests/test_step_c_fixed_height_recheck_candidate.py ......         [6/6]
tests/test_sweep_hip_yaw_divergence_params.py ..                   [2/2]
tests/test_final_validation_rejects_stub_source.py .........       [9/9]
tests/test_hip_yaw_mode_math.py ...                                 [3/3]
tests/test_hip_yaw_ownership.py .......                            [7/7]
tests/test_mode_based_hip_yaw_divergence_controller.py ......... [25/25]
tests/test_hip_yaw_mode_ownership.py ............                 [12/12]
```

All 73 tests pass.

## 7. What was NOT changed (per strict restrictions)

* `default/current-best` profile unchanged.
* PFF source (`physics_equilibrium_feedforward_outer_loop`) unchanged.
* Low-band v2 tuning (`physics_equilibrium_feedforward_outer_loop_low_band_support_v2`)
  unchanged.
* Hip-yaw hard gate at 0.35 rad unchanged.
* No threshold relaxation.
* No D4/D5-specific logic added; the new controller is opt-in.

## 8. Final classification (this report)

`MODE_HIP_YAW_DIVERGENCE_REAL_D4_D5_FAIL`

The candidate profile is correctly wired and the controller is active,
but the architecture-fix goal of `hip_yaw_abs_max_rad < 0.35` for D4
and D5 was not achieved. The profile also does not regress support
recovery (D4 support max_abs 0.272 m vs C 0.318 m, D5 0.515 m vs C
0.534 m). The base profiles (A/B/C) also exceed the gate in this
batch, so the fix is not a regression — it is an insufficient
improvement.

## 9. Next recommended task

1. Investigate why the divergence controller cannot bring
   `hip_yaw_abs_max_rad` below 0.40 in the D4 case even with
   kp=5.0 and max_torque=2.0. Two hypotheses:
   * The antisymmetric hip-yaw torque authority is fundamentally too
     small at this height.
   * The mode-based controller is fighting the existing
     `enable_hip_yaw_divergence_damping` (HY2-DIV) inside the
     `ShapePostureController` — both write to the divergence mode
     without coordination.
2. Consider combining the mode-based controller with the wheel-yaw
   stabilizer (different profile name) for D5. The D5 result (0.3803
   rad) is much closer to the gate and a small additional wheel-yaw
   contribution may close the gap.
3. Do not promote the candidate. Default/current-best remains
   unchanged.
4. Re-classify the candidate after the next iteration.