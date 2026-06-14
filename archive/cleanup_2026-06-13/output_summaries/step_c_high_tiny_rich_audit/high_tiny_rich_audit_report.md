# High Tiny Rich Telemetry Audit

Classification: `sagittal_position_authority_insufficient_at_high_height`
Recommendation: `add sagittal scheduling for high-height variants`

## Event timing
- height_error_peak: row 4957, time 49.57 s, abs 0.018752
- hip_yaw_peak: row 4957, time 49.57 s, abs 0.271901
- pitch_peak: row 1724, time 17.24 s, abs 0.086820
- support_position_peak: row 1739, time 17.39 s, abs 0.156463
- wheel_velocity_peak: row 1682, time 16.82 s, abs 6.095879

## Sagittal root-cause summary
- Wheel peak dominant term: tau_sagittal_velocity
- Support peak dominant term: tau_pitch
- Wheel peak velocity damping effect: opposes

## Hip-yaw root-cause summary
- Final-window sign-correct fraction: 1.000
- Final-window saturation fraction: 0.000
- Drift likely secondary: True

## Reference consistency
- Nominal reference leak detected: False
