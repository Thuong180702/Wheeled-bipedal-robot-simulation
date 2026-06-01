# Step E Completion Archive — 2026-06-01

## Status

Step E status: **DONE**.

Step E passes nominal standing-position hold on the official production simulation path.

## Validation source

- Official telemetry file: `outputs/hierarchical_controller_sim/telemetry_1780289121.csv`
- Validation output directory: `outputs/step_e_official_validation_v2/`
- Commit hash: `9971615447f36e0127482db28c5b4139b742bc3b`
- Final verdict: `PASS`
- Final decision: `STEP_E_DONE`
- Can mark Step E done: `true`

## Functional metrics

Duration:

- Rows: `5000`
- Source step range: `0..4999`
- Survived expected steps: `true`

Position hold:

- Metric used: `support_position_error_m`
- Max absolute support-position error: `0.104456751 m`
- Final support-position error: `0.091351773 m`
- RMS support-position error: `0.057046557 m`
- Required threshold `<= 0.15 m`: passed
- Preferred max-abs threshold `<= 0.12 m`: passed
- Preferred final-abs threshold `<= 0.10 m`: passed

Posture validity:

- Hip-yaw max absolute error: `0.0567 rad`
- Hip-yaw RMS error: `0.022819449 rad`
- Percent time abs hip-yaw error > `0.10 rad`: `0.0%`

Balance stability:

- Pitch max absolute: `0.070771351 rad`
- Roll max absolute: `0.012998945 rad`
- Minimum CoM height: `0.403835297 m`
- Wheel mean velocity max absolute: `3.839568138 rad/s`
- Contact valid percent: `100.0%`

## Structural invariants

Structural invariants: **PASS**.

- Raw `tau_wbc_norm` max: `14.207267761`
- Applied WBC contribution norm max: `0.0`
- WBC applied: `false`
- WBC computed only as diagnostic: `true`
- WBC contributed to `tau_total_raw_per_joint`: `false`
- `active_torque_owner_per_joint` includes WBC: `false`
- Hidden torque zero: `true`
- Ownership violation count max: `0`
- Legacy torque paths off: `true`

The nonzero raw `tau_wbc_norm` is diagnostic-only. The applied WBC contribution is zero, and WBC remains off in the applied torque path.

## Decision

Step E is complete for nominal standing-position hold.

No further Step E tuning should be done unless future Step C, Step D, or Step F regressions identify a root cause that specifically requires revisiting Step E.

## Next roadmap step

Next roadmap step: **Step C — Height recovery to target height**.
