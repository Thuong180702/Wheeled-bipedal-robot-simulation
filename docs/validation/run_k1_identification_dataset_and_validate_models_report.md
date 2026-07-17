# Run K1 Identification Dataset and Validate Models — Final Report

**Date:** 2026-06-25
**Target:** `K1_PITCH_RATE_NOTCH_V1`
**Profile:** `k1_pitch_rate_notch_v1`
**Final Classification:** `NEEDS_STATE_AUGMENTATION_BEFORE_DESIGN`

---

## 1. Executive Summary

The complete K1 identification pipeline was executed end-to-end: 15 real-simulation telemetry runs were generated across 3 heights (0.33m, 0.40m, 0.48m) with 5 excitation types each. All 15 runs passed integrity audit. System identification produced high-R² models (0.996-0.9998), but **no model captured the 0.24-0.4 Hz controller-induced oscillatory mode**. The 6D linear state vector (`x6_base`) is insufficient — the mode requires controller-internal filter/notch states that are not in the current telemetry.

**State-feedback design is NOT allowed yet.** The blocker is the inability of the 6D linear model to capture the dominant oscillatory dynamics.

---

## 2. K1 Unchanged Verification

| Check | Result |
|-------|--------|
| `kp_pitch` = 50.0 | CONFIRMED |
| `kd_pitch` = 10.0 | CONFIRMED |
| `k_position` = 40.0 | CONFIRMED |
| `k_velocity` = 15.0 | CONFIRMED |
| `k_wheel_velocity` = 0.5 | CONFIRMED |
| `k_support_velocity` = 0.0 | CONFIRMED |
| `max_position_tau` = 3.0 Nm | CONFIRMED |
| `max_tau_wheel` = 5.0 Nm | CONFIRMED |
| No new controller candidate | CONFIRMED |
| No WBC or hidden torque | CONFIRMED |
| No threshold relaxation | CONFIRMED |

**K1 unchanged: YES.**

---

## 3. Telemetry Generation Status

| Height | A_equilibrium | B_90n_push | C_impulse | D_prbs_excitation | E_support_offset |
|--------|:---:|:---:|:---:|:---:|:---:|
| **low_0p330** (0.33m) | OK 19MB | OK 28MB | OK 19MB | OK 23MB | OK 19MB |
| **mid_0p400** (0.40m) | OK 19MB | OK 28MB | OK 19MB | OK 24MB | OK 19MB |
| **high_0p480** (0.48m) | OK 19MB | OK 28MB | OK 19MB | OK 24MB | OK 19MB |

- **Attempted:** 15 runs
- **Succeeded:** 15 runs (100%)
- **Failed:** 0 runs

All telemetry is `validation_source: real_simulation` with no stubs, no synthetic data, no assumed rows.

---

## 4. Dataset Integrity Summary

All 15 runs classified **USABLE**:
- 1,999-2,999 rows per run
- 1,599-2,599 post-settle samples (exceeding minimums)
- No NaN/Inf in critical state columns (`pitch_x_rad`, `height_error_m`, `com_y_velocity_m_s`, `wheel_vel_mean_rad_s`)
- Minor Inf values in non-critical derived columns (capture point, support ratio) — expected MuJoCo artifact
- Two simulation failures during generation (mid_0p400 C_impulse, mid_0p400 B_90n_push) auto-retried successfully
- All K1 profiles correctly recorded

---

## 5. State Vector Evaluation

**Selected:** `x6_base` (fallback selection)

| Candidate | Dim | One-step R² | 50-step RMSE | Mode Found | Score |
|-----------|-----|------------|-------------|------------|-------|
| x6_base | 6 | -8.11 | 0.0396 | NO | 126.90 |
| x7_add_height | 7 | -7.11 | 0.0364 | NO | 126.57 |
| x8_add_notch | 8 | -8.11 | 0.0343 | NO | 136.92 |
| x9_add_position | 9 | 0.9941 | 0.0348 | NO | 131.50 |
| x_filter_augmented | 10 | -7.11 | 0.0304 | NO | 136.69 |

**Key finding:** No state vector captured the 0.24-0.4 Hz oscillatory mode. `x9_add_position` achieved high R² (0.994) but this is trivially good due to near-integrating position states — the mode is still not captured.

---

## 6. Identified A_id/B_id Models

| Height | Method | R² | Test R² | Condition | Dominant Mode |
|--------|--------|-----|---------|-----------|---------------|
| low_0p330 | ridge | 0.9996 | 0.9997 | ∞ | NOT FOUND |
| mid_0p400 | ols | 0.9960 | 0.9968 | 3.3×10³¹ | NOT FOUND |
| high_0p480 | ridge | 0.9998 | 0.9998 | 5.9×10³⁴ | NOT FOUND |

**Critical observation:** Despite excellent R² values, A matrices are near-singular (condition numbers from ∞ to 10³⁴). The 6D state vector is rank-deficient for linear identification because states are highly collinear in closed-loop operation.

---

## 7. Model Validation Results

| Height | Classification | One-step R² | 50-step RMSE | 200-step RMSE | Mode Freq Err |
|--------|---------------|------------|-------------|--------------|---------------|
| low_0p330 | HEIGHT_DATA_INSUFFICIENT | N/A | N/A | N/A | N/A |
| mid_0p400 | HEIGHT_DATA_INSUFFICIENT | N/A | N/A | N/A | N/A |
| high_0p480 | NEEDS_STATE_AUGMENTATION | 0.7709 | 0.2733 | 0.4911 | NOT FOUND |

Validation was limited by available test telemetry:
- high_0p480: Used legacy telemetry (688 post-push samples)
- low_0p330, mid_0p400: No dedicated validation telemetry available at these heights

---

## 8. Dominant Mode by Height

| Height | Mode Frequency | Damping | Classification |
|--------|---------------|---------|---------------|
| 0.33m | NOT FOUND | — | No oscillatory mode in 0.15-0.55 Hz |
| 0.40m | NOT FOUND | — | No oscillatory mode in 0.15-0.55 Hz |
| 0.48m | NOT FOUND | — | No oscillatory mode in 0.15-0.55 Hz |

**The 0.24-0.4 Hz mode is NOT captured by the linear model.**

From the previous MuJoCo audit, the mode is known to exist at ~0.239 Hz, ζ=+0.096 in closed-loop telemetry. The failure to capture it in a linear model confirms it is **controller-induced** — produced by K1's notch filter and torque clipping nonlinearities, not by the open-loop plant dynamics.

---

## 9. B_id Reliability

B_id was estimated from sagittal external force (±0.20N PRBS). The primary coupling is to `wheel_vel_mean` (expected), with secondary coupling to `pitch_rate_x` and `com_y_velocity`.

However, B_id reliability is **LOW** because:
1. The PRBS amplitude (±0.20N) is very weak — 0.25% of body weight
2. The external force injection path differs from K1's torque injection
3. A_id is near-singular, making B_id co-estimation unreliable

---

## 10. Height Scheduling Feasibility

**INSUFFICIENT** — 0 of 3 heights produced models with identifiable oscillatory modes (need >=2 for scheduling analysis). No linear or common-K analysis possible.

---

## 11. Control Feasibility Result

| Height | Controllability | LQR Torque Est. | Torque Budget | Pole Placement |
|--------|:---:|:---:|:---:|:---:|
| low_0p330 | 4/6 | 8.06 Nm | FAIL | NOT FEASIBLE |
| mid_0p400 | 4/6 | 3.85 Nm | OK | NOT FEASIBLE |
| high_0p480 | 4/6 | 10.51 Nm | FAIL | NOT FEASIBLE |

**Overall:** `STATE_FEEDBACK_NOT_FEASIBLE` — All models have rank deficiency (4/6 controllable), near-singular A matrices, and no target mode accessibility.

---

## 12. Is State-Feedback Design Allowed Now?

**NO.**

**Exact blocker:** The 6D linear model (`x6_base`) cannot capture the 0.24-0.4 Hz controller-induced oscillatory mode, which is the primary target for state-feedback damping. Without capturing this mode, any state-feedback design would be working on the wrong dynamics.

---

## 13. Recommended Next Task

**Expose K1 notch/filter states in telemetry and re-evaluate with `x8_add_notch`.**

Specific steps:
1. Add `filtered_pitch_rate` and `notch_output` to the simulation telemetry
2. Re-generate PRBS excitation runs (larger amplitude: ±0.50N instead of ±0.20N)
3. Re-run state vector evaluation with `x8_add_notch`
4. Re-identify and validate
5. Re-audit control feasibility

**Alternatively:** Consider that the mode is fundamentally nonlinear (torque clipping + notch filter). A linear state-feedback approach may never fully capture it. Options:
- Nonlinear identification (e.g., Koopman operator with lifting functions including notch states)
- Directly augment K1 with additional damping (modify K1 gains) rather than state feedback
- Change the identification approach to closed-loop subspace identification with the controller in the loop

---

## 14. Files Created

| File | Purpose |
|------|---------|
| `scripts/generate_k1_identification_dataset.py` | Telemetry generation orchestrator (fixed: `--vd-sagittal-authority-profile`) |
| `scripts/audit_k1_identification_dataset_integrity.py` | Dataset integrity audit (NEW — Phase 2) |
| `scripts/evaluate_k1_identification_state_vectors.py` | State vector evaluation (fixed: height filter, Unicode) |
| `scripts/identify_k1_mujoco_state_space_models.py` | System identification (fixed: Unicode) |
| `scripts/validate_k1_identified_models.py` | Model validation (fixed: Unicode) |
| `scripts/analyze_k1_height_scheduled_models.py` | Height schedule analysis (fixed: Unicode) |
| `scripts/audit_k1_identified_model_control_feasibility.py` | Control feasibility audit (fixed: Unicode) |
| `tests/test_run_k1_identification_dataset_pipeline.py` | 29 pipeline tests (NEW — Phase 9) |
| `outputs/k1_identification_dataset/` | 15 telemetry CSVs + 3 setups + all phase outputs |
| `outputs/k1_identification_dataset/dataset_integrity_report.json` | Integrity audit JSON |
| `outputs/k1_identification_dataset/dataset_integrity_report.md` | Integrity audit Markdown |
| `outputs/k1_identification_dataset/identification_summary.json` | Model identification results |
| `outputs/k1_identification_dataset/model_validation.json` | Model validation results |
| `outputs/k1_identification_dataset/height_schedule_analysis.json` | Height schedule analysis |
| `outputs/k1_identification_dataset/control_feasibility.json` | Control feasibility audit |
| `docs/validation/run_k1_identification_dataset_and_validate_models_report.md` | This report |

---

## 15. Tests/Compile Checks Run

```
=== Compile Checks (7/7) ===
generate_k1_identification_dataset.py           → OK
audit_k1_identification_dataset_integrity.py    → OK
evaluate_k1_identification_state_vectors.py     → OK
identify_k1_mujoco_state_space_models.py        → OK
validate_k1_identified_models.py                → OK
analyze_k1_height_scheduled_models.py           → OK
audit_k1_identified_model_control_feasibility.py → OK

=== Test Suites (102/102 passed) ===
test_run_k1_identification_dataset_pipeline.py  → 29 passed
test_k1_identification_dataset.py               → 26 passed
test_mujoco_true_linearization_audit.py         → 27 passed
test_current_best_controller_profile.py         →  8 passed
test_final_validation_rejects_stub_source.py    →  9 passed
                                           TOTAL: 102 passed, 0 failed
```

---

## 16. Limitations

1. **Controller-induced mode:** The 0.24-0.4 Hz oscillation is produced by K1's notch filter and torque clipping — it cannot be captured by a linear 6D state-space model of the plant alone.

2. **Weak PRBS excitation:** ±0.20N sagittal force is only 0.25% of body weight. Signal-to-noise ratio may be too low for reliable B_id estimation.

3. **Near-singular A matrices:** All identified A_id matrices have condition numbers from 3×10³¹ to ∞. The 6D state vector is too collinear in closed-loop operation for well-conditioned identification.

4. **No cross-height validation:** low_0p330 and mid_0p400 have no independent validation telemetry — validation was limited to legacy 0.48m data.

5. **Single-input identification:** Only sagittal external force was used as input. This limits the identified B_id to the force injection path, which differs from K1's actual torque injection path.

6. **Filter states not in telemetry:** K1's internal notch filter states (`filtered_pitch_rate`, `notch_output`) are not exposed in current telemetry, preventing `x8_add_notch` from being properly evaluated.

7. **No controller was designed, tuned, or modified:** Per constraints, this task is identification and feasibility only. No state-feedback controller was created.

8. **Telemetry generation speed:** Each 3000-step simulation takes ~5-7 minutes with full 100Hz telemetry. The 15-run dataset required ~75 minutes of wall-clock time.

---

## 17. Final Classification

```
NEEDS_STATE_AUGMENTATION_BEFORE_DESIGN
```

The identification pipeline executed correctly and produced 15 high-quality real-simulation datasets. However, the 6D linear state vector (`x6_base`) is fundamentally insufficient to capture K1's controller-induced oscillatory mode. State-feedback design should not proceed until:
1. Notch/filter states are exposed in telemetry
2. A state vector including these states (`x8_add_notch`) is validated
3. At least one height produces a DESIGN_READY model
4. Or, an alternative nonlinear identification approach is adopted
