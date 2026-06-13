# Step C high_tiny failure audit

## Verdict

- Classification: `high_height_variant_static_pose_valid_but_dynamic_controller_not_robust`
- Controller behavior changed: `false`
- WBC applied: `false`
- Hidden torque / ownership: clean
- Step C DONE: `false`

## Event order
- `wheel_velocity_peak` at `16.82 s`
- `support_position_peak` at `17.39 s`
- `hip_yaw_peak` at `49.57 s`

## Key metrics
- Height recovery passed: `True`
- Final height error: `-0.017505 m`
- Support peak: row `1739`, time `17.39 s`, abs `0.156463 m`
- Wheel velocity peak: row `1682`, time `16.82 s`, abs `6.095879 rad/s`
- Hip-yaw peak: row `4957`, time `49.57 s`, abs `0.271901 rad`

## Causal interpretation

High_tiny starts from a valid Step B pose and height remains within the Step C band, but wheel velocity and support-position excursions occur around 16.82-17.39s and a large late hip-yaw drift grows by the final window. Shape hip-yaw torque sign opposes the error, WBC/hidden torque/ownership are clean, so current evidence indicates dynamic robustness/authority coupling rather than initialization or WBC ownership failure.

## Recommended next action

`Add more telemetry before fix` ? Existing telemetry can show event order and torque ownership, but does not expose enough decomposed sagittal internal terms or shape-posture reference error/saturation diagnostics to safely choose between hip-yaw authority, sagittal scheduling, or reference handling changes.

## Artifacts
- `outputs\step_c_high_tiny_failure_audit\high_tiny_failure_audit.json`
- `outputs\step_c_high_tiny_failure_audit\high_tiny_support_peak_window.csv`
- `outputs\step_c_high_tiny_failure_audit\high_tiny_wheel_velocity_peak_window.csv`
- `outputs\step_c_high_tiny_failure_audit\high_tiny_hip_yaw_peak_window.csv`
- `outputs\step_c_high_tiny_failure_audit\high_tiny_nominal_low_tiny_comparison.csv`
