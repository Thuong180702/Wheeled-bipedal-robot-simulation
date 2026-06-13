# APCR1c_active_pitch_crossing_early_activation Profile Design

## Overview

**Profile**: `APCR1c_active_pitch_crossing_early_activation`
**Base**: APCR1b_active_pitch_crossing_early_release
**Purpose**: Earlier APCR entry to correct signed support drift before it reaches the ±0.15 m band

## Hypothesis

APCR1b still enters too late because `outer_enter_m = 0.10` allows drift to approach the +0.15 band before recovery starts.

**APCR1b at 500-step**:
- outside ±0.15 = 13.8% (unchanged vs APCR1)
- zero crossings = 5 (reduced from APCR1's 8)
- inner_exit_m = 0.07 works correctly
- APCR activates correctly

APCR1b reduced oscillation amplitude (zero crossings 8→5) but did not reduce band violations.

**APCR1c approach**: Lower `outer_enter_m` from 0.10 to 0.08 so recovery starts earlier, before drift can accumulate toward the ±0.15 band.

## Parameter Changes vs APCR1b

| Parameter | APCR1b | APCR1c | Change |
|-----------|--------|--------|--------|
| `apc_outer_enter_m` | 0.10 m | 0.08 m | Enter earlier |
| `apc_inner_exit_m` | 0.07 m | 0.07 m | Keep |
| `apc_opposite_overshoot_m` | 0.00 m | 0.00 m | Keep |
| `apc_max_cross_tau` | 1.0 Nm | 1.0 Nm | Keep |
| `apc_max_rate_per_step` | 0.4 Nm/step | 0.4 Nm/step | Keep |

## Profile Definition

```python
"APCR1c_active_pitch_crossing_early_activation": SagittalAuthoritySchedule(
    profile_name="APCR1c_active_pitch_crossing_early_activation",
    applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,  # low_0p300, high_0p480
    # D2 baseline: position cap 4.0 Nm, velocity damping 1.10x
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    max_position_tau_low_max=4.0,
    velocity_damping_scale=1.10,
    # Active Pitch Recovery (APCR_strategy) - with recovery gate mode
    enable_active_pitch_crossing=True,
    active_pitch_crossing_recovery_gate_mode=True,  # same as APCR1/APCR1b
    apc_outer_enter_m=0.08,  # m - CHANGED from 0.10: enter earlier
    apc_inner_exit_m=0.07,  # m - same as APCR1b
    apc_opposite_overshoot_m=0.00,  # m - same as APCR1b
    apc_pitch_enter_rad=0.03,  # rad - same as APCR1b
    apc_pitch_safe_limit_rad=0.08,  # rad - same as APCR1b
    apc_max_cross_tau=1.0,  # Nm - same as APCR1b
    apc_smooth_alpha=0.10,
    apc_max_rate_per_step=0.4,  # Nm/step - same as APCR1b
    apc_contact_gate=True,
    apc_height_gate=True,
    apc_roll_gate=True,
    apc_min_com_z_m=0.28,  # m - same as APCR1b
    apc_max_com_z_m=0.50,  # m - same as APCR1b
    apc_pitch_safe_threshold_rad=0.05,  # rad - same as APCR1b
    apc_pitch_danger_threshold_rad=0.10,  # rad - same as APCR1b
    apc_roll_threshold_rad=0.15,  # rad - same as APCR1b
    # APCR recovery gate parameters - same as APCR1b
    apcr_pitch_hard_stop_rad=0.30,  # rad - hard stop, blocks APCR
    apcr_roll_hard_stop_rad=0.15,  # rad - lateral stability
    apcr_min_com_z_m=0.27,  # m - minimum safe height
    apcr_max_com_z_m=0.50,  # m - maximum operating height
),
```

## Entry/Exit Behavior

**Entry**: APCR activates when `|signed_error| > 0.08` m

**Exit**: APCR deactivates when `|signed_error| <= 0.07` m

**State Machine**:
- NEUTRAL → CROSS_FROM_POSITIVE: signed_error > 0.08 AND pitch > 0.03 AND safety gates pass
- NEUTRAL → CROSS_FROM_NEGATIVE: signed_error < -0.08 AND pitch < -0.03 AND safety gates pass
- CROSS_FROM_POSITIVE → NEUTRAL: signed_error <= 0.07 (exit earlier than APCR1b's 0.04)
- CROSS_FROM_NEGATIVE → NEUTRAL: signed_error >= -0.07 (exit earlier than APCR1b's -0.04)

## Expected Behavior

1. APCR enters earlier (at 0.08 vs 0.10)
2. Recovery torque starts before drift reaches the ±0.15 band
3. Band violations should decrease compared to APCR1b
4. Zero crossings may increase slightly due to earlier activation
5. Torque sign and magnitude unchanged from APCR1b

## Comparison with APCR1 and APCR1b

| Metric | D2 | APCR1 | APCR1b | APCR1c (expected) |
|--------|-----|-------|--------|-------------------|
| outer_enter_m | N/A | 0.10 | 0.10 | 0.08 |
| inner_exit_m | N/A | 0.05 | 0.07 | 0.07 |
| outside ±0.15 | 19.2% | 13.8% | 13.8% | <13.8% |
| zero crossings | 2 | 8 | 5 | ~6-7 |
| positive% | 93.2% | 79.4% | 79.2% | <79% |

## Do NOT Modify

- D2 baseline profile
- APCR1 profile
- APCR1b profile

## Validation Plan

1. Run APCR1c 500-step at low_0p300
2. Compare metrics against D2, APCR1, APCR1b
3. If outside ±0.15 < APCR1b and pass criteria met, run 2000-step
4. Otherwise, stop and document findings
