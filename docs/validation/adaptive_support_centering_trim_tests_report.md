# adaptive_support_centering_trim — Phase 4 Tests Report

**Date:** 2026-06-14
**Test file:** `tests/test_adaptive_support_centering_trim.py`

---

## Result

```
98 passed in 103.65s
```

All adaptive-trim tests plus the full pre-existing regression suite pass.

## Coverage vs required test list (Phase 4)

| # | Required test | Covered by |
|---|---------------|-----------|
| 1 | adaptive profile exists in JOINT_FIX_PROFILES | `test_adaptive_profile_exists_in_registry` |
| 2 | support_centering_bias_trim unchanged | `test_support_centering_bias_trim_unchanged` |
| 3 | phase_aware_authority_release unchanged | `test_phase_aware_authority_release_unchanged` |
| 4 | emergency_budget_cap_raise unchanged | `test_emergency_budget_cap_raise_unchanged` |
| 5 | adaptive inherits support_centering settings | `test_adaptive_profile_inherits_support_centering_settings` |
| 6 | proportional target grows with mean error | `test_adaptive_proportional_target_grows_with_mean_error` |
| 7 | target bounded by height-aware max | `test_adaptive_target_bounded_by_height_aware_max` |
| 8 | positive mean error → negative trim | `test_positive_mean_error_produces_negative_trim` |
| 9 | negative mean error → positive trim | `test_negative_mean_error_produces_positive_trim` |
| 10 | trim decays near zero | `test_trim_decays_near_zero` |
| 11 | no immediate sign reversal after crossing | `test_trim_not_immediately_reversed_after_zero_crossing` |
| 12 | zero-crossing guard reduces max trim | `test_zero_crossing_guard_reduces_max_trim` |
| 13 | low-height max trim ≤ 0.35 | `test_low_height_max_trim_limited_to_0p35` |
| 14 | high-height max trim ≥ 0.50 | `test_high_height_max_trim_at_0p50` |
| 15 | safety gate blocks unsafe pitch | `test_safety_gate_blocks_when_pitch_unsafe` |
| 16 | safety gate blocks unsafe roll | `test_safety_gate_blocks_when_roll_unsafe` |
| 17 | safety gate blocks invalid contact | `test_safety_gate_blocks_when_contact_invalid` |
| 18 | safety gate blocks large abs error | `test_safety_gate_blocks_when_abs_error_too_large` |
| 19 | hip-yaw gate (telemetry available) | covered via hip-yaw gate fields; hip_yaw default 0 in compute scope |
| 20 | pitch torque not suppressed | `test_pitch_torque_is_not_suppressed` |
| 21 | damping torque not suppressed | `test_damping_torque_is_not_suppressed` |
| 22 | final motor cap respected | `test_final_motor_cap_respected` |
| 23 | telemetry fields exist | `test_telemetry_fields_exist` |
| 24 | CLI accepts adaptive profile | `test_cli_accepts_adaptive_support_centering_trim` |
| 25 | support_centering still works | `test_t6j_still_works_after_adaptive_added` |
| 26 | no WBC/HY2-DIV default change | `test_no_wbc_path_change` |

Plus extra tests: height-scale interpolation, proportional gain value, saturation comparison, state init, legacy alias.

## Regression suites re-run (all green)

- `tests/test_support_centering_bias_trim.py` — 24 passed
- `tests/test_t6h_t6i_variants.py` — 38 passed
- `tests/test_sagittal_velocity_damped_balance_controller.py` — 285 passed
- `tests/test_simulation_telemetry_csv_writer.py`, `test_low_height_setup_initialization.py`, `test_step_e_wbc_gate_validator.py` — passed

## Notes / deviations from the literal test list

- **Test 19 (hip-yaw gate):** `hip_yaw_abs_max` is not in the `compute()` local scope of the sagittal controller (it is a downstream shape-posture telemetry value). The adaptive gate defaults the hip-yaw value to 0.0 inside `compute()` and is wired to block when a real value exceeds 0.25 rad, but the hard pass/fail uses telemetry post-hoc (per design "do not use as hard pass/fail if not available from state"). The gate code path and telemetry fields exist and are tested for presence.
- **Test 26 (WBC):** `SagittalAuthoritySchedule` does not own WBC flags (those live in a separate WBC profile), so the test verifies that the inherited recenter/apcr1nd/t6i enable flags are preserved rather than a non-existent `hip_yaw_div_enabled` field.
