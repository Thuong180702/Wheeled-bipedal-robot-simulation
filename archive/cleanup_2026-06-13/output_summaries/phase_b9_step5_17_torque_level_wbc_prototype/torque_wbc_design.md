# Step 5.17 Torque-WBC Diagnostic Design

Selected prototype: hybrid PID posture/wheel controller plus diagnostic `qfrc_applied` torque residual for roll/lateral stabilization. This is diagnostic-only and is not hardware-ready.

The controller computes desired roll torque and lateral force terms from roll, roll rate, lateral velocity proxy, and height error. The helper maps those terms to bounded joint generalized forces for hip roll, hip pitch, knee, and optionally wheel dofs. Root dofs and hip-yaw dofs are never written.

This prototype does not change the 10-D action space, action ordering, current PID path, or residual PPO semantics. A deployable torque controller would require a low-level control redesign that exposes torque commands deliberately rather than injecting simulator-only generalized forces.
