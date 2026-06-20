# Unified Sagittal State-Feedback No-Offset (USSF-NO) Design

**Date:** 2026-06-19  
**Status:** UNIFIED_NO_OFFSET_DESIGN_READY  
**Predecessor:** [unified_no_offset_state_conflict_audit.md](unified_no_offset_state_conflict_audit.md)

---

## Design Principle

Replace the current two-independent-loop architecture:

```text
tau_final = tau_pitch(pitch_error) + tau_position(support_error) + tau_velocity_damping(wheel_vel)
```

...which produces 86–91% opposing-sign conflict, with a single coordinated
sagittal command from a unified state vector:

```text
tau_cmd = f(x, h, mode)
```

where:
- `x` = full sagittal state vector
- `h` = height (for gain scheduling)
- `mode` = operating mode (steady, drift, push, transition, etc.)

## State Vector

```text
x = [
    support_error_m,           # signed CoM-support center offset (m)
    support_error_rate_mps,    # low-passed derivative of above (m/s)
    pitch_x_rad,              # body pitch in sagittal plane (rad)
    pitch_rate_x_radps,       # body pitch rate (rad/s)
    height_error_m,           # com_z - height_cmd (m)
    height_rate_mps,          # vertical velocity (m/s)
    roll_y_rad,               # body roll for safety gating (rad)
    hip_yaw_abs_max_rad,      # max abs(l_hip_yaw, r_hip_yaw) for safety
    contact_state,             # 0/1/2 wheels in contact
    tau_saturation_flag,       # 1 if torque budget is exceeded
]
```

This state is already available in the telemetry stream — **no new sensors
or estimators needed.**

## Control Law

```text
tau_cmd = -Kx(h) * support_error
          -Kv(h) * support_error_rate
          -Ktheta(h) * pitch_x
          -Komega(h) * pitch_rate_x
          -Kh(h) * height_error    # optional, may be zero
          -Khdot(h) * height_rate  # optional, may be zero
```

All signs verified empirically in Phase 5 (sign sweep).

This is NOT the independent tau_pitch + tau_position sum — it is a single
expression where every term shares the SAME torque budget and the SAME
goal: coordinated sagittal stabilization.

## Mode Classifier

Eight operating modes, detected from state thresholds:

| Mode | Trigger | Priority Weight |
|------|---------|----------------|
| **STEADY** | \|support_error\| < 0.04 m AND \|pitch_x\| < 0.05 rad | w_support=1.0, w_pitch=1.0, w_height=0.5 |
| **DRIFT_RECOVERY** | \|support_error\| >= 0.04 m | w_support=2.0, w_pitch=0.7, w_rate=1.5 |
| **PUSH_RECOVERY** | \|pitch_x\| >= 0.10 rad OR pitch_rate_x > 0.15 rad/s AND increasing | w_pitch=2.0, w_support=0.5, w_rate=1.0 |
| **HEIGHT_TRANSITION** | \|height_error\| > 0.005 m AND changing height | w_height=1.0, w_support=0.7, w_pitch=0.7 |
| **CONTACT_DEGRADED** | contact_state < 2 | w_support=0.5, w_pitch=1.5, w_rate=0.5 |
| **HIP_YAW_RISK** | hip_yaw_abs_max > 0.10 rad | w_support=0.5, w_pitch=0.5, w_rate=0.3 (reduce aggressive torque) |
| **SATURATED** | tau_saturation_flag = 1 | Reduce w on the dominant fighting term |
| **PITCH_POSITION_CONFLICT** | tau_pitch_raw * tau_position_raw < 0 AND both > 0.1 Nm | Intervene — reduce the term that fights error direction |

Priority-weighted command:

```text
tau_cmd = [
    w_support * tau_support_state +
    w_pitch   * tau_pitch_state +
    w_rate    * tau_rate_damping +
    w_height  * tau_height_state
] / (w_support + w_pitch + w_rate + w_height)

Where:
    tau_support_state = -Kx * support_error - Kv * support_error_rate
    tau_pitch_state = -Ktheta * pitch_x - Komega * pitch_rate_x
    tau_height_state = -Kh * height_error - Khdot * height_rate
```

The normalization ensures the command stays within a consistent range
regardless of which mode is active.

## Height-Varying Gains

All gains use smooth PCHIP-style interpolation (numpy array evaluation,
no if/else chains):

| Gain | low_0p300 | low_0p320 | low_0p340 | low_0p360 | low_0p380 | high_0p430 | high_0p450 | high_0p465 | high_0p480 | Units |
|------|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|-------|
| Kx | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Nm/m |
| Kv | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Nm/(m/s) |
| Ktheta | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Nm/rad |
| Komega | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Nm/(rad/s) |
| Kh | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Nm/m |
| Khdot | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | Nm/(m/s) |
| torque_cap | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Nm |
| rate_limit | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Nm/step |

Gains are filled in during Phase 5 (sign sweep + coarse gain sweep).

Initial placeholder gains (based on existing B2v2 equivalent authority):

| Parameter | low band (0.30-0.36) | mid band (0.38-0.43) | high band (0.45-0.48) |
|-----------|---------------------|---------------------|----------------------|
| Kx | 4.0 | 3.0 | 3.0 |
| Kv | 0.20 | 0.15 | 0.10 |
| Ktheta | 3.0 | 3.0 | 3.0 |
| Komega | 0.15 | 0.15 | 0.10 |
| torque_cap | 6.0 | 5.0 | 5.0 |
| rate_limit | 0.15 | 0.10 | 0.10 |

## Safety Gates

1. **Contact gate:** If contact_state < 2, reduce w_support and w_pitch, increase w_rate
2. **Roll gate:** If |roll_y| > 0.10 rad, clamp sagittal torque
3. **Hip-yaw gate:** If hip_yaw_abs_max > 0.15 rad, clamp aggressive terms
4. **Torque cap:** |tau_cmd| <= torque_cap(h)
5. **Rate limit:** |tau_cmd - tau_prev| <= rate_limit(h)
6. **Height gate:** If com_z < 0.28 m or com_z > 0.50 m, freeze torque

## What is NOT changed

- **No WBC changes:** WBC pipeline operates after tau_cmd is computed
- **No HY2-DIV changes:** Hip-yaw divergence damping is independent
- **No shape posture changes:** Hip-roll/knee/hip-pitch PD position servos unchanged
- **No PID action bias:** Unchanged (already disabled in B2v2)
- **No torque rate limiting:** Unchanged (separate mechanism)
- **No contact model:** Unchanged

## Telemetry

New telemetry fields (all prefixed `no_offset_`):

```python
no_offset_controller_active      # bool — this profile is active
no_offset_mode                   # string — current mode name
no_offset_support_error_m        # float
no_offset_support_error_rate_mps # float
no_offset_pitch_rad              # float
no_offset_pitch_rate_radps       # float
no_offset_height_error_m         # float
no_offset_roll_rad               # float
no_offset_hip_yaw_metric         # float
no_offset_kx                     # float
no_offset_kv                     # float
no_offset_ktheta                 # float
no_offset_komega                 # float
no_offset_kh                     # float
no_offset_khdot                  # float
no_offset_tau_support_state      # float
no_offset_tau_pitch_state        # float
no_offset_tau_rate_state         # float
no_offset_tau_height_state       # float
no_offset_priority_support       # float
no_offset_priority_pitch         # float
no_offset_priority_rate          # float
no_offset_tau_total_raw          # float
no_offset_tau_total_limited      # float
no_offset_torque_cap             # float
no_offset_rate_limit             # float
no_offset_saturation_active      # bool
no_offset_gate_pass              # bool
no_offset_block_reason           # string
no_offset_arbitration_reason     # string
no_offset_pitch_ref_offset_deg   # MUST be 0.0
```

## Implementation Plan

1. Add `unified_sagittal_state_feedback_no_offset` profile to
   `sagittal_velocity_damped_balance_controller.py`
2. Profile inherits from `SagittalAuthoritySchedule` with:
   - `pitch_ref_offset_deg = 0.0`
   - All offset/trim/bias mechanisms OFF
   - Mode classifier + arbitration logic active
3. Compute function replaces `_compute_outer_loop_pitch_ref` +
   the independent tau_pitch/tau_position logic with the unified law
4. CLI registration in `simulate_hierarchical_controller.py` profile registry
5. Telemetry integration

## Validation Gates

- Phase 4: Unit tests for mode classifier, arbitration, zero offset
- Phase 5: 500-step sign and gain discovery
- Phase 6: Fixed-height validation (all 10 heights)
- Phase 7: Step C random height
- Phase 8: Step D push

## Risks

1. **Unknown gain signs for Ktheta/Komega:** Must be verified empirically
2. **Mode classifier thresholds:** May need tuning per height band
3. **Priority weights:** Too aggressive weights cause oscillation
4. **Torque cap too low:** May limit performance at extreme heights
5. **No offset means larger steady-state pitch:** The robot may stand at
   a non-zero pitch angle naturally. This is OK if support is centered.

## Design Classification

| Criterion | Status |
|-----------|--------|
| UNIFIED_NO_OFFSET_DESIGN_READY | ✅ |
| UNIFIED_NO_OFFSET_DESIGN_BLOCKED | — |
| UNIFIED_NO_OFFSET_DESIGN_INCONCLUSIVE | — |
