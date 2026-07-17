# EZC Fix Path Decision

**Date:** 2026-06-15  
**Profile:** early_zero_crossing_recenter  
**Scenario:** high_0p480, 5000 steps

## Classification

**EZC_FIX_PATH_ANTIREBOUND**

## Audit Summary

### Phase 1: Telemetry Correctness - PASS
- EZC uses correct drift column: `active_pitch_crossing_signed_error_m`
- All drift columns agree (max diff 0.0004 m)
- **Classification: EZC_TELEMETRY_COLUMN_CORRECT**

### Phase 2: Episode Root Cause - FAIL
- 21 EZC episodes, 86% cross zero
- EZC reaches max torque 100% of time
- Net corrective torque = -5.37 Nm (strong)
- BUT rebound is FAST: avg 28 steps after exit
- 71% post-exit steps are positive drift
- **Classification: EZC_FAILURE_EXIT_TOO_EARLY_REBOUND**

Secondary factors:
- EZC_FAILURE_WEAK_TORQUE (partially - torque strong but decays too fast)

### Phase 3: Hip-Yaw/Posture - PASS
- Hip-yaw: safe, V-shape stance, never exceeded threshold
- Roll: perfect zero
- Pitch: very well controlled (max 0.29 deg)
- Height: stable (~0.485 m)
- Contact: 100% double support
- **Classification: EZC_POSTURE_HIP_YAW_SAFE**

## Root Cause Summary

**PRIMARY: EXIT_TOO_EARLY_REBOUND**

The robot oscillates around positive drift:
1. EZC enters at +0.05, applies strong -0.55 Nm correction
2. Drift crosses zero (86% of episodes)
3. EZC exits immediately at zero
4. Correction decays quickly (3 dwell + 0.025 Nm/step)
5. Positive bias (~+3.5 to +4.0 Nm from tau_pitch + tau_wheel_velocity) overwhelms tau_position
6. Drift returns positive in ~28 steps
7. EZC re-enters at +0.05
8. Repeat

**The robot is NOT drifting around zero. It is stuck on the positive side.**

## Fix Path Selection

### Candidate Fix Paths Considered

| Path | Description | Applicable? | Reason |
|------|-------------|-------------|--------|
| A. Stronger correction | Increase torque | NO | Torque already reaches max, net correction is strong |
| B. Faster response | Increase rate | PARTIAL | Rate already reaches max, not the bottleneck |
| C. Anti-rebound hold | Keep correction after zero, decay slowly | **YES** | Addresses the core issue: gap between exit and re-entry |
| D. Cap transmission fix | Move trim in composition | NO | No clipping/cancellation observed |
| E. Hip-yaw compensation | Add yaw compensation | NO | Hip-yaw is safe, not the cause |
| F. Mixed V2 | Combination | POSSIBLE | May need stronger torque + anti-rebound |

### Selected Fix: **Path C - Anti-Rebound Hold**

**Rationale:**
1. EZC exits at zero but positive bias immediately returns drift to +0.10 to +0.20 m
2. The gap between EZC exit and re-entry allows drift to accumulate
3. Anti-rebound hold keeps a small decaying correction after zero crossing
4. This prevents drift from bouncing back while tau_position recovers

### Proposed Changes for `early_zero_crossing_recenter_v2`

| Parameter | Current | Proposed | Reason |
|-----------|---------|----------|--------|
| ezc_base_tau_nm | 0.18 | 0.25 | Slightly stronger base correction |
| ezc_max_tau_nm | 0.55 | 0.70 | More authority for anti-rebound phase |
| ezc_rate_nm_per_step | 0.012 | 0.015 | Faster ramp to target |
| ezc_decay_nm_per_step | 0.025 | 0.020 | Slower decay after crossing (KEY CHANGE) |
| ezc_zero_dwell_steps | 3 | 5 | Longer hold at zero before decay |
| ezc_antirebound_decay_steps | N/A | 30 | NEW: keep decaying correction for 30 steps after exit |
| ezc_antirebound_decay_ratio | N/A | 0.50 | NEW: decay to 50% of current tau over antirebound period |

### Anti-Rebound Logic

```
When EZC crosses zero (e <= 0):
    - Enter ANTIREBOUND_DECAY state
    - Keep current tau (approximately 50% of max)
    - Decay slowly over ezc_antirebound_decay_steps (30 steps)
    - Exit to IDLE after decay completes OR if e > +0.05 (re-enter normal EZC)
    - Do NOT target opposite side (-0.02)
```

### Safety Considerations

- Anti-rebound decay does NOT target opposite side
- Still bounded by max_tau (0.70 Nm)
- Still subject to all safety gates (pitch, roll, hip_yaw, contact)
- Exit condition: decay complete OR re-entry at +0.05

## Decision

**Proceed with Phase 5: Implement early_zero_crossing_recenter_v2 with Path C (Anti-Rebound).**