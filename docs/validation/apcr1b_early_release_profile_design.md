# APCR1b Active Pitch Crossing Early Release Profile Design

## Profile Name

`APCR1b_active_pitch_crossing_early_release`

## Base

APCR1 (`APCR1_active_pitch_crossing_recovery_moderate`)

## Design Rationale

APCR1 2000-step validation showed:
- **Positive bias reduced**: 98.3% (D2) → 72.7% (APCR1)
- **But excessive band violations**: 4.8% (D2) → 12.2% (APCR1)
- **Diagnosis**: APCR1 holds CROSS_FROM_POSITIVE too long, releasing too late

APCR1b aims to keep the positive bias reduction benefit while reducing oscillation amplitude and band violations.

## Parameter Changes from APCR1

| Parameter | APCR1 | APCR1b | Reason |
|-----------|-------|--------|--------|
| `apc_inner_exit_m` | 0.05 | **0.07** | Exit earlier, before support goes too far negative |
| `apc_opposite_overshoot_m` | 0.01 | **0.00** | No overshoot allowance into opposite direction |

## Parameters Kept Same as APCR1

| Parameter | Value | Note |
|-----------|-------|------|
| `apc_outer_enter_m` | 0.10 | Enter threshold unchanged |
| `apc_max_cross_tau` | 1.0 Nm | Torque unchanged |
| `apc_max_rate_per_step` | 0.4 Nm/step | Rate limit unchanged |
| `apcr_pitch_hard_stop_rad` | 0.30 | Hard safety unchanged |
| `apc_smooth_alpha` | 0.10 | Smoothing unchanged |

## Expected Behavior

1. APCR1b enters CROSS_FROM_POSITIVE at signed_error > 0.10 m (same as APCR1)
2. APCR1b applies negative torque to reverse pitch_rate (same as APCR1)
3. APCR1b exits CROSS_FROM_POSITIVE when signed_error <= 0.07 m (earlier than APCR1's 0.04 m exit target)
4. APCR1b exits CROSS_FROM_NEGATIVE when signed_error >= -0.07 m (earlier than APCR1's -0.04 m exit target)
5. APCR1b does NOT allow opposite overshoot (no 0.01 m buffer)

## Expected Outcome

| Metric | D2 | APCR1 | APCR1b Target |
|--------|-----|-------|---------------|
| Positive % | 98.3% | 72.7% | <75% (maintain APCR1 benefit) |
| Outside ±0.15 | 4.8% | 12.2% | <8% (improve vs APCR1) |
| Zero crossings | 5 | 19 | 8-12 (less oscillation than APCR1) |
| Final signed error | 0.0979 | 0.0047 | Close to zero |

## Validation Plan

1. **500-step smoke test**: Verify APCR1b survives and shows reduced oscillation vs APCR1
2. **2000-step full test** (only if 500-step passes): Verify band violations decrease while bias reduction maintained

## Do NOT Run

- 5000-step validation (not yet warranted)
- Step C or Step D