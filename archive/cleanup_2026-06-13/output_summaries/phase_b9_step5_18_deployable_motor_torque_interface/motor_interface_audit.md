# Step 5.18 Motor Interface Audit

MJCF defines ten `<motor>` actuators, one per action index. Actuator index equals action index.
All actuators have `gear=1`, explicit `ctrlrange`, and explicit `forcerange`, so a deployable simulation motor-torque path can write actuator `ctrl` directly.
The existing baseline remains position-PID for leg joints and velocity-PI for wheel joints; that path is unchanged unless `low_level_control.mode` is explicitly set.

| action | actuator | joint | ctrlrange | forcerange |
|---:|---|---|---|---|
| 0 | l_hip_roll_motor | l_hip_roll | [-15.0, 15.0] | [-22.0, 22.0] |
| 1 | l_hip_yaw_motor | l_hip_yaw | [-15.0, 15.0] | [-22.0, 22.0] |
| 2 | l_hip_pitch_motor | l_hip_pitch | [-30.0, 30.0] | [-44.0, 44.0] |
| 3 | l_knee_motor | l_knee | [-30.0, 30.0] | [-44.0, 44.0] |
| 4 | l_wheel_motor | l_wheel | [-15.0, 15.0] | [-22.0, 22.0] |
| 5 | r_hip_roll_motor | r_hip_roll | [-15.0, 15.0] | [-22.0, 22.0] |
| 6 | r_hip_yaw_motor | r_hip_yaw | [-15.0, 15.0] | [-22.0, 22.0] |
| 7 | r_hip_pitch_motor | r_hip_pitch | [-30.0, 30.0] | [-44.0, 44.0] |
| 8 | r_knee_motor | r_knee | [-30.0, 30.0] | [-44.0, 44.0] |
| 9 | r_wheel_motor | r_wheel | [-15.0, 15.0] | [-22.0, 22.0] |

Current best controller remains `outputs\phase_b9_lqr_gain_strengthening\best_lqr_config.yaml`.
Step 6 remains BLOCKED.
