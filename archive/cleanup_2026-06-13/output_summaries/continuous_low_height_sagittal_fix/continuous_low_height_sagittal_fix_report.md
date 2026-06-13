# Continuous Low-Height Sagittal Authority Fix Evaluation Report

Generated: 2026-06-03 23:16:30

## Candidates Evaluated

- **baseline**: FAIL
- **candidate_E1_k60_continuous**: FAIL
- **candidate_E2_k80_continuous**: FAIL
- **candidate_E3_k100_continuous**: FAIL

## Detailed Results

### baseline -- FAIL

Label                                          Steps Verdict      Key Failures
----------------------------------------------------------------------------------------------------
stepE_low0p300_1000                             1000 FAIL         support_position_error max_abs <= 0.15 m; hip_yaw_abs_max...

### candidate_E1_k60_continuous -- FAIL

Label                                          Steps Verdict      Key Failures
----------------------------------------------------------------------------------------------------
stepE_low0p300_1000                             1000 FAIL         support_position_error max_abs <= 0.15 m; hip_yaw_abs_max...

### candidate_E2_k80_continuous -- FAIL

Label                                          Steps Verdict      Key Failures
----------------------------------------------------------------------------------------------------
stepE_low0p300_1000                             1000 FAIL         support_position_error max_abs <= 0.15 m; hip_yaw_abs_max...

### candidate_E3_k100_continuous -- FAIL

Label                                          Steps Verdict      Key Failures
----------------------------------------------------------------------------------------------------
stepE_low0p300_1000                             1000 FAIL         support_position_error max_abs <= 0.15 m; hip_yaw_abs_max...

## Acceptance Gates

| Gate | Threshold |
|------|-----------|
| support_position_error max_abs <= 0.15 m | 0.15 |
| hip_yaw_abs_max <= 0.07 rad | 0.07 |
| pitch_x max_abs <= 0.10 rad | 0.1 |
| roll_y max_abs <= 0.05 rad | 0.05 |
| final height error max_abs <= 0.02 m | 0.02 |
| non-wheel floor contacts = 0 | 0 |
| contact valid >= 99.9% | 0.999 |
| WBC applied = false | False |
| hidden torque = 0 | 0.0 |
| ownership violations = 0 | 0 |
