# Step E Height-Variant Position-Hold Audit v2

## Verdict

- Overall audit verdict: **STEP_E_HEIGHT_VARIANT_ROBUSTNESS_GAP**
- Step E nominal remains valid: **true**
- Step E across true height variants passes: **false**
- Controller behavior changed: `false` (baseline profile used for all variants)
- WBC applied: `false`

## Per-variant results

| Variant | Verdict | Support max abs (m) | Support final (m) | HipYaw max (rad) | Pitch max (rad) | WheelVel max (rad/s) | Height final vs target (m) |
|---|:---:|---:|---:|---:|---:|---:|---:|
| nominal | PASS | 0.106062 | 0.090206 | 0.056483 | 0.071130 | 3.867527 | 0.004186 |
| low_tiny | PASS | 0.109590 | -0.044979 | 0.042003 | 0.072812 | 4.036306 | 0.001441 |
| high_tiny | FAIL | 0.156463 | 0.047837 | 0.271901 | 0.086820 | 6.095879 | -0.017505 |
| low_small | PASS | 0.106200 | 0.092534 | 0.057385 | 0.071244 | 3.989667 | 0.004896 |
| high_small | FAIL | 0.154629 | -0.031510 | 0.124348 | 0.086340 | 6.041556 | -0.003249 |

## Structural invariants

| Variant | WBC applied | Hidden torque max | Ownership violations |
|---|:---:|:---:|:---:|
| nominal | false | 0.0 | 0 |
| low_tiny | false | 0.0 | 0 |
| high_tiny | false | 0.0 | 0 |
| low_small | false | 0.0 | 0 |
| high_small | false | 0.0 | 0 |

## Failure classifications

- **nominal**: PASS
- **low_tiny**: PASS
- **high_tiny**: support_max_abs=0.156463 > 0.15 (lead: position-led)
- **high_tiny**: hip_yaw_max=0.271901 > 0.07 (lead: position-led)
- **high_tiny**: hip_yaw_gt_0.10_percent=62.78% > 0% (lead: position-led)
- **high_tiny**: wheel_vel_max=6.095879 > 5.0 (lead: position-led)
- **low_small**: PASS
- **high_small**: support_max_abs=0.154629 > 0.15 (lead: position-led)
- **high_small**: hip_yaw_max=0.124348 > 0.07 (lead: position-led)
- **high_small**: hip_yaw_gt_0.10_percent=4.16% > 0% (lead: position-led)
- **high_small**: wheel_vel_max=6.041556 > 5.0 (lead: position-led)

## Comparison to Step C high-height failures

- Step C high_tiny (baseline): support_peak=0.156463m, wheel_vel_peak=6.10rad/s, hip_yaw_peak=0.271901rad
- Step C high_small (candidate_A): support_peak=0.157301m, wheel_vel_peak=5.51rad/s, pitch_peak=0.100129rad

## Final decision

- **STEP_E_HEIGHT_VARIANT_ROBUSTNESS_GAP**

- Recommended next action: Step E nominal remains DONE. Step E height-variant robustness is incomplete. Pause Step C fix work until Step E height-variant hold passes for high_tiny.