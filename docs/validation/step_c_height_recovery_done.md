# Step C Height Recovery – Validation Complete

**Date:** 2026-06-02  
**Status:** **DONE**

## Summary

Step C height recovery has been re‑validated after the Step E height‑variant robustness fix (`candidate_D2_wheel_velocity_damping_light`). All five height‑variant recovery cases now pass the 5000‑step diagnostic sweep.

| Case | Verdict | Recovery time (s) | Support max (m) | HipYaw max (rad) | Wheel max (rad/s) |
|------|---------|-------------------|-----------------|------------------|-------------------|
| nominal | PASS | 0.0 | 0.106 | 0.056 | 3.87 |
| low_tiny | PASS | 0.0 | 0.110 | 0.042 | 4.04 |
| high_tiny | PASS | 0.0 | 0.124 | 0.038 | 4.12 |
| low_small | PASS | 0.0 | 0.106 | 0.057 | 3.99 |
| high_small | PASS | 0.0 | 0.135 | 0.030 | 4.77 |

**Invariants preserved:**
- WBC applied = `false`
- Hidden torque norm max = `0.0`
- Ownership violation count = `0`
- Step E structural invariants preserved = `true`

## Validation Command

```bash
python scripts/run_step_c_height_recovery.py \
  --use-height-variants \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --output-dir outputs/step_c_height_recovery_after_step_e_hv_fix \
  --continue-after-failure
```

## Artifacts

- Output directory: `outputs/step_c_height_recovery_after_step_e_hv_fix/`
- Summary JSON: `step_c_pass_fail_summary.json`
- Metrics: `step_c_height_recovery_metrics.json`
- Report: `step_c_height_recovery_report.md`

## Relation to Step E‑HV

Step C was previously blocked because Step E height‑variant position hold was not robust. The fix implemented for Step E‑HV (profile `candidate_D2_wheel_velocity_damping_light`) simultaneously unblocked Step C. No additional changes to Step C logic were required.

## Next Steps

- Step C is ready for merge.
- The selected sagittal authority profile should be used as the default for all balance‑core height‑variant experiments.
- Step D (residual training) can now proceed with confidence that both Step E and Step C are validated across all height variants.

## Extreme-height follow-up validation

A later extreme-height campaign reused the same D2 profile and evaluated Step C only after Step E passed on a narrower controller-ready envelope. The broader static-only search range was **not** accepted as dynamic-ready; instead, the final extreme Step C pass was reported on the conservative dynamic envelope established by fresh Step E telemetry.

Final conservative **controller-ready** extreme heights:

- `min_operational_height = 0.3932865805 m` CoM
- `max_operational_height = 0.4128130092 m` CoM

Controller-ready Step C extreme results:

| Case | Verdict | Recovery time (s) | Support max (m) | HipYaw max (rad) | Pitch max (rad) | Roll max (rad) | Wheel max (rad/s) | Final height error (m) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| min_operational_height | PASS | 0.0 | 0.133 | 0.056 | 0.096 | 0.014 | 4.72 | 0.0035 |
| max_operational_height | PASS | 0.0 | 0.135 | 0.030 | 0.096 | 0.009 | 4.77 | 0.0047 |

Interpretation:

- Both cases started already within the accepted height band, so `recovery_time_s = 0.0` is valid as a hold/recovery sanity result.
- Local transition recovery around the extrema was **NOT RUN** in this pass.
- WBC remained off, hidden torque remained zero, and ownership violations remained zero.

See `docs/validation/operational_height_extreme_validation.md` for the full static search, failed broader static extrema, boundary probes, and final controller-ready envelope.

---
*Validated with the same candidate_D2_wheel_velocity_damping_light profile used for Step E‑HV final validation.*
