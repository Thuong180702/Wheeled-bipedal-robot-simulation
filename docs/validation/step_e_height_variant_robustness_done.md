# Step E Height‑Variant Position Hold Robustness – Validation Complete

**Date:** 2026-06-02  
**Status:** **DONE**

## Summary

The height‑variant sagittal authority schedule `candidate_D2_wheel_velocity_damping_light` has been validated and selected as the final fix for Step E height‑variant robustness.

All five true Step B height variants pass the 5000‑step position hold audit with no regressions:

| Variant | Verdict | Support max (m) | HipYaw max (rad) | Pitch max (rad) | Wheel max (rad/s) |
|---------|---------|----------------|------------------|-----------------|-------------------|
| nominal | PASS | 0.106 | 0.056 | 0.071 | 3.87 |
| low_tiny | PASS | 0.110 | 0.042 | 0.073 | 4.04 |
| high_tiny | PASS | 0.124 | 0.038 | 0.092 | 4.12 |
| low_small | PASS | 0.106 | 0.057 | 0.071 | 3.99 |
| high_small | PASS | 0.135 | 0.030 | 0.096 | 4.77 |

**Invariants preserved:**
- WBC applied = `false`
- Hidden torque norm max = `0.0`
- Ownership violation count = `0`
- No torque overshoot beyond configured limits

## Selected Controller Profile

`candidate_D2_wheel_velocity_damping_light` (defined in `scripts/simulate_hierarchical_controller.py`):

```python
SagittalAuthoritySchedule(
    profile_name="candidate_D2_wheel_velocity_damping_light",
    applies_to_variants=("high_tiny", "high_small"),
    position_tau_cap_by_variant=(("high_tiny", 4.0), ("high_small", 4.0)),
    pitch_tau_scale=1.0,
    velocity_damping_scale=1.10,
)
```

This profile:
- Retains the 4.0 Nm position cap for both high‑height variants.
- Increases sagittal velocity damping by 10% for high‑height variants only.
- Leaves pitch gains unchanged.
- Does **not** modify low‑height or nominal behavior.

## Validation Artifacts

- Final audit output: `outputs/step_e_height_variant_position_hold_final/`
- Candidate telemetry: `outputs/step_e_height_variant_position_hold_final/candidate_telemetry/`
- Summary report: `outputs/step_e_height_variant_position_hold_final/step_e_hv_sagittal_schedule_fix_report.md`
- JSON summary: `outputs/step_e_height_variant_position_hold_final/step_e_hv_sagittal_schedule_fix_summary.json`

## Relation to Step C

Step C height recovery was blocked until Step E height‑variant robustness was confirmed. With this validation, Step C has been re‑run using the same `candidate_D2_wheel_velocity_damping_light` profile and now passes all height‑variant recovery cases. See `docs/validation/step_c_height_recovery_done.md` for details.

## Code Changes

Modified files:
- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` – added `support_velocity_gain` scheduling field.
- `scripts/simulate_hierarchical_controller.py` – added D1/D2 authority profiles.
- `scripts/run_step_c_height_recovery.py` – added `--vd-sagittal-authority-profile` argument.
- `tests/test_sagittal_velocity_damped_balance_controller.py` – added tests for damping schedules.
- `tests/test_step_c_height_recovery.py` – updated tests for new command argument.

## Next Steps

- Step C height recovery is now unblocked and validated.
- The selected profile should be used as the default sagittal authority schedule for all future balance‑core runs that require height‑variant robustness.
- The fix is ready for merge into main.

---
*Validated by systematic debugging + TDD + candidate evaluation protocol. No blind tuning or invasive architectural changes.*
