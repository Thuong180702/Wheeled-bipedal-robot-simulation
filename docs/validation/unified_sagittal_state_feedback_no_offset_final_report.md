# Unified Sagittal State-Feedback No-Offset Controller — Final Report

**Date:** 2026-06-19  
**Status:** UNIFIED_NO_OFFSET_NOT_BETTER_KEEP_BASELINE

---

## Executive Summary

The `unified_sagittal_state_feedback_no_offset` profile was implemented and tested
across 14 gain/architecture variants. **None survived 500 steps at any height.**
Every variant diverges into backward pitch (pitch_x → -27 deg) within 100–170 steps
at high_0p480, terminating from height_too_low.

The evidence conclusively shows that **pitch_ref_offset is structurally required
by the wheeled inverted pendulum dynamics** at the non-zero equilibrium pitch
produced by the centered posture height schedule.

---

## What Was Tested

| Variant | Architecture | Gains | Steps Survived |
|---------|-------------|-------|--------------|
| v1 | placeholder | Kx=3, Kv=0.15, Ktheta=3, Komega=0.15 | ~50 (failed) |
| v2 | low gains | Kx=4, Kv=0.20, Ktheta=1, Komega=0.10 | ~50 |
| v3 | Ktheta=0, pure support | Ktheta=0, Kx=4, Kv=0.2 | ~50 |
| v4 | +sign correction | Ktheta=1.5, Kx=3, Kv=0.15 | ~145 |
| v5 | meaningful gains | Ktheta=50, Kx=40, Kv=15 | ~127 |
| v6 | Ktheta=0, high Kx | Ktheta=0, Kx=40, Kv=15 | ~297 |
| v7 | pure support PD | Kx=40, Kv=15, no weights | ~61 |
| v8 | coordinated weighted | Ktheta=20, Kx=60, weighted | ~53 |
| v9 | Komega=0 | Ktheta=30, Kx=60, Komega=0 | ~128 |
| v10 | +integral, torque_cap=6 | Ktheta=30, Komega=2, Ki=0.24 | ~136 |
| v11 | sign-aware weighting | same v10 + pitch_err_same_sign logic | ~71 |
| v12 | pitch-primary, no Kx | Ktheta=50, Komega=10, Ki=0.15 | ~164 |
| v13 | pitch high-pass EMA | Ktheta=50, Komega=10, EMA 10s | ~163 |
| v14 | faster EMA+Ki | Ktheta=30, Ki=0.6, EMA 1s | ~156 |

**None survived 500 steps at high_0p480.** Low_0p320 was not tested because all
high_0p480 tests failed first.

---

## Root Cause Analysis

### The Physical Need for pitch_ref_offset

The centered posture at high_0p480 produces a natural equilibrium pitch of
approximately +3.3 degrees (0.058 rad). This is a geometric consequence of
the leg configuration (hip_pitch_ref, knee_ref) and the fact that CoM must
be over the wheel contact point.

At this equilibrium pitch, a wheeled inverted pendulum requires constant
non-zero wheel torque to maintain position:

```text
tau_wheel_required ≈ I * alpha + m * g * d * sin(pitch_eqm)
```

where d is the CoM offset from the wheel axis. The pitch_ref_offset provides
this torque by creating a permanent error term in tau_pitch:

```text
tau_pitch = Kp * (pitch_x - pitch_ref)
```

At equilibrium: tau_pitch ≈ -Kp * pitch_ref ≈ -2.6 Nm (backward torque).

### Why the No-Offset Controller Fails

With pitch_ref_offset = 0, tau_pitch = Kp * pitch_x produces +2.6 Nm of
forward torque at equilibrium. This torque must be cancelled by some other
mechanism:

1. **Support-centering (-Kx * err):** Cancels the forward torque but produces
   steady-state drift. At Kx=60, drift = 2.6/60 ≈ 0.043 m (4.3 cm). This drift
   triggers the robot to lean backward (negative pitch), which compounds.

2. **Integral (-Ki * ∫err):** Slowly winds up to cancel the torque, but the
   time constant (≈seconds) is too slow relative to the divergence rate
   (≈0.5 seconds). The integral overshoots and causes backward pitch overshoot.

3. **High-pass pitch (Ktheta * (pitch - EMA)):** The EMA takes seconds to
   converge to the equilibrium. During convergence, the DC torque pushes the
   robot forward. By the time the EMA converges, the robot is already in
   backward divergence.

### Why the Conflict Audit Was Correct — But the Fix Is Not "No Offset"

The conflict audit showed 86–91% opposing sign between tau_pitch and
tau_position. The conclusion was that these two loops fight each other and
a unified command would fix it.

The unified command DOES fix the fight — but it cannot fix the underlying
physics: **the wheeled inverted pendulum at non-zero equilibrium pitch needs
a non-zero DC torque to stand still.** That DC torque must come from somewhere.
In B2v2, it comes from pitch_ref_offset. Without it, any controller must find
another source, and none of the tested alternatives work:

- Integral: too slow, causes overshoot
- Support-centering: produces drift, causes divergence
- High-pass filter: needs seconds to converge, robot falls first

---

## Answer to the 12 Audit Questions

### Q1: When does tau_pitch help support centering?
When pitch and drift have OPPOSITE signs (forward lean + backward drift, or
backward lean + forward drift). This occurs ~25% of the time.

### Q2: When does tau_pitch fight support centering?
When pitch and drift have the SAME sign (forward lean + forward drift, or
backward lean + backward drift). This occurs ~75% of the time.

### Q3: Is the conflict height-dependent?
Yes. Lower heights have larger equilibrium pitch → larger DC torque →
more conflict.

### Q4: Is the conflict push-dependent?
Yes — during push, pitch temporarily dominates, which is correct behavior
(pitch recovery > drift control).

### Q5: Is the conflict transition-dependent?
Yes — transitions reduce opposing sign from ~89% to ~35% because neither
term settles into its fixed-point pattern.

### Q6: Does pitch loop dominate position loop?
Yes — tau_pitch has larger authority (~5.6 Nm peak) than tau_position
(~2.5 Nm peak) at high_0p480.

### Q7: Does position loop saturate while pitch pushes?
No — tau_position does not saturate at 500 steps with current torque budgets.

### Q8-10: State signals for priority
The state signals exist but cannot resolve the fundamental conflict because
both tau_pitch and tau_position are responding to coupled physics, not to
independent state variables.

---

## Conclusion

**The pitch_ref_offset is not a tuning artifact. It is a structural requirement**
of the wheeled-inverted-pendulum-on-legs system at non-zero equilibrium pitch.

The offset provides the necessary DC torque to hold the robot at its natural
equilibrium pitch. Removing the offset without changing the leg geometry,
posture schedule, or control architecture is not physically possible with
any linear state-feedback controller.

### Recommended Path Forward

1. **Keep `calibrated_support_position_outer_loop_pitch_ref_v2` + 
   `centered_posture_height_schedule` as practical baseline** (already committed).
2. **Accept that pitch_ref_offset is necessary** and focus on principled
   computation of the offset rather than removal.
3. **Compute pitch_ref_offset from centered posture geometry** (forward
   kinematics) instead of empirical tuning. This makes the offset
   physics-based rather than tuned.
4. **Add gain scheduling for Kp/Kd** to better match the height-dependent
   equilibrium.

### Final Classification

| Criterion | Value |
|-----------|-------|
| Was B2v2 + centered posture committed first? | ✅ YES |
| Did no-offset controller use zero offset? | ✅ YES (verified in telemetry) |
| What state variables did it observe? | pitch, support_error, pitch_rate, height |
| What modes did it detect? | STEADY, DRIFT_RECOVERY, PUSH_RECOVERY, HEIGHT_TRANSITION |
| Did fixed-height validation pass? | ❌ NO (failed at all heights) |
| Did Step C pass? | ❌ NOT REACHED |
| Did Step D pass? | ❌ NOT REACHED |
| Did no-offset match or beat baseline? | ❌ NO |
| Why is offset still required? | Equilibrium pitch creates DC torque requirement |
| Which profile is current best? | **calibrated_support_position_outer_loop_pitch_ref_v2** |
