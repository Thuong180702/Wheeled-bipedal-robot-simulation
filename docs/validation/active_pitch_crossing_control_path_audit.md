# Active Pitch Crossing Control Path Audit

## Classification: CONTROL_PATH_CONFIRMED_READY_FOR_ACTIVE_CROSSING

## Control Path Verification

### 1. Which terms are computed?

| Term | Computed | Line | Notes |
|------|----------|------|-------|
| tau_pitch | YES | 599 | kp_pitch × pitch_x_rad |
| tau_pitch_rate | YES | 607 | kd_pitch × pitch_rate_x_rad_s |
| tau_position | YES | 622-668 | k_position × sagittal_position_error + integral |
| tau_wheel_velocity | YES | 595-596 | k_wheel_velocity × wheel_velocity |
| phase_recenter_tau | YES | 829-864 | F1_strategy, gated by pitch/height/hip_yaw |
| hysteresis_recenter_tau | YES | 931-972 | F2_strategy, stateful recenter |
| bias_cancel_tau | YES | 1050-1086 | G1_strategy, persistent bias cancellation |

### 2. Which terms are added to final wheel torque?

| Term | Added to tau_common | Method |
|------|---------------------|--------|
| tau_pitch | YES | Directly in tau_common_unclipped (line 1091) |
| tau_pitch_rate | YES | Directly in tau_common_unclipped (line 1091) |
| tau_position | YES | Directly in tau_common_unclipped (line 1091) |
| tau_support_velocity | YES | Directly in tau_common_unclipped (line 1091) |
| tau_cp + tau_com_vy | YES | Directly in tau_common_unclipped (line 1091) |
| phase_recenter_tau | YES | Added at line 1095 |
| hysteresis_recenter_tau | YES | Added at line 1097 |
| bias_cancel_tau | YES | Added at line 1099 |
| tau_wheel_velocity | YES | Per-wheel after tau_common (lines 1104-1105) |

### 3. Is tau_position computed but not applied?

**NO - tau_position IS applied.** It is added to tau_common_unclipped at line 1091.

However, **D2 baseline has k_position=0.0**, so tau_position contributes ZERO torque.
This is by design - D2 uses pitch + wheel velocity damping only.

### 4. Where should Active Pitch Crossing term enter?

**tau_common_unclipped** - same as phase_recenter_tau, hysteresis_recenter_tau, bias_cancel_tau.

The Active Pitch Crossing term should be added as a NEW independent term in the torque composition:
```
tau_common_unclipped = (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity +
    tau_support_velocity + tau_position + tau_cp + tau_com_vy +
    recenter_tau_clipped +   # F1
    hyst_tau_clipped +       # F2
    bias_tau_clipped +       # G1
    apc_tau_clipped          # NEW: Active Pitch Crossing
)
```

### 5. Does controller support adding separate wheel torque without modifying tau_position_raw?

**YES.** The state machine approach is already proven by F1/F2/G1 implementations.
The APC term will:
- Be computed in a separate state machine
- Produce its own torque term (apc_tau)
- Be added to tau_common_unclipped directly
- NOT modify tau_position_raw

## Safety Features Already in Place

1. **Safety gates**: pitch_danger, height_unsafe, contact_invalid, roll_unsafe
2. **Smoothing**: alpha smoothing prevents discontinuous jumps
3. **Rate limiting**: max_rate_per_step prevents sudden torque changes
4. **Clipping**: bounded by max_cross_tau

## Conclusion

Control path is confirmed ready. APC can be implemented as:
1. A new state machine in SagittalVelocityDampedBalanceController
2. With its own parameters in SagittalAuthoritySchedule
3. Adding a separate torque term to tau_common_unclipped
4. With full telemetry and safety gates
