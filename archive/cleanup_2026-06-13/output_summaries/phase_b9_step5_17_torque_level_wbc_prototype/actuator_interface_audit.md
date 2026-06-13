# Step 5.17 Actuator / Force Interface Audit

The MJCF defines torque-like `<motor>` actuators with ctrl/force ranges. The deployed baseline does not expose those motor torques as policy actions: normalized actions are still interpreted as leg position targets and wheel velocity targets, then converted to `ctrl` by the low-level PID path.

`qfrc_applied` and `xfrc_applied` are available for diagnostic simulation experiments. Step 5.17 therefore uses diagnostic-only `qfrc_applied` joint generalized-force injection while preserving the current PID baseline and residual PPO semantics.

Current best remains `outputs\phase_b9_lqr_gain_strengthening\best_lqr_config.yaml`. Step 6 remains BLOCKED.
