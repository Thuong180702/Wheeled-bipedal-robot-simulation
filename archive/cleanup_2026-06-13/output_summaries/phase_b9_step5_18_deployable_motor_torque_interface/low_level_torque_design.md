# Step 5.18 Low-Level Torque Mode Design

Three low-level modes are supported for diagnostics and controller candidates:

1. `pid_position_velocity`: default behavior. Leg actions are position PID targets and wheel actions are velocity PI targets.
2. `motor_torque`: opt-in deployable simulation torque path. Normalized action maps directly to MJCF motor actuator `ctrl`, bounded by `max_ctrl_fraction` and actuator ctrlrange.
3. `hybrid_pid_plus_torque`: opt-in hybrid path. The existing PID/PI ctrl is computed first, then a bounded normalized torque residual is added and clipped to actuator ctrlrange.

The action dimension and ordering remain unchanged. `configs/training/balance_residual.yaml` is not modified. Hip-yaw torque is disabled by default in torque mode.
