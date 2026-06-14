# APCR1b Threshold Candidate Design

## Classification: APCR1b_EARLY_RELEASE_CANDIDATE

## Summary

Based on the 2000-step validation, APCR1 achieves the primary goal (positive bias reduction: 98.3% → 72.7%) but introduces oscillation that causes more band violations (4.8% → 12.2%). This design targets reducing oscillation by releasing earlier.

## APCR1 Current Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `enable_active_pitch_crossing` | True | Enable APCR |
| `active_pitch_crossing_recovery_gate_mode` | True | Separate hard safety from recovery |
| `apc_pitch_safe_threshold_rad` | 0.030 (1.7°) | APCR activates when pitch < 1.7° AND signed error > outer |
| `apcr_pitch_hard_stop_rad` | 0.30 (17.2°) | Hard safety gate |
| `outer_enter_m` | 0.10 | Activate when signed error > 0.10 m |
| `inner_exit_m` | 0.05 | Exit when signed error ≤ 0.05 m |
| `opposite_overshoot_m` | 0.01 | Allow slight negative overshoot |
| `apc_max_cross_tau` | 1.0 Nm | Max torque during recovery |
| `apc_max_rate_per_step` | 0.4 Nm/step | Rate limit |

## APCR1b Proposed Parameters

| Parameter | APCR1 | APCR1b | Change |
|-----------|-------|---------|--------|
| `outer_enter_m` | 0.10 | 0.10 | Unchanged |
| `inner_exit_m` | 0.05 | **0.07** | Release earlier |
| `opposite_overshoot_m` | 0.01 | **0.00** | Prevent negative accumulation |
| `apc_max_cross_tau` | 1.0 | 1.0 | Unchanged |

## Rationale

### Why raise inner_exit_m from 0.05 to 0.07?

The current `inner_exit_m = 0.05` causes APCR to wait until signed error is very close to zero before releasing. By that time, momentum carries it past zero into negative territory. By raising to 0.07, APCR releases when signed error is still moderately positive (0.07 m), reducing negative overshoot.

### Why set opposite_overshoot_m from 0.01 to 0.00?

The current 0.01 m allows slight negative overshoot, which accumulates over time. By setting to 0.00, APCR enforces symmetric exit - it exits CROSS_FROM_POSITIVE the moment signed error crosses zero (or very close to zero). This prevents negative drift accumulation.

### Why not change other parameters?

- `outer_enter_m = 0.10`: The 0.10 m threshold seems appropriate for entry. The problem is not late entry but late release.
- `max_cross_tau = 1.0 Nm`: Increasing torque would make oscillation worse. The torque is sufficient (drift does reverse).
- `max_rate_per_step = 0.4 Nm/step`: The rate limit seems appropriate.

## Expected Behavior Change

### APCR1 (current):
1. Signed error exceeds 0.10 m → APCR enters CROSS_FROM_POSITIVE
2. APCR applies negative torque (wheel forward)
3. Signed error decreases toward zero
4. APCR waits until signed error ≤ 0.05 m
5. APCR releases → momentum carries past zero
6. Signed error becomes negative → APCR enters CROSS_FROM_NEGATIVE
7. Repeat with oscillation

### APCR1b (proposed):
1. Signed error exceeds 0.10 m → APCR enters CROSS_FROM_POSITIVE
2. APCR applies negative torque (wheel forward)
3. Signed error decreases toward zero
4. APCR releases when signed error ≤ 0.07 m (earlier)
5. Less momentum to overshoot negative
6. Signed error crosses zero but stays positive (or minimal negative)
7. Repeat with reduced oscillation amplitude

## 500-Step Validation Plan

If approved, run:
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1b_active_pitch_crossing_early_release \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 500 \
  --telemetry-decimation 1
```

Profile name: `APCR1b_active_pitch_crossing_early_release`

## Comparison Targets

| Metric | D2 | APCR1 | APCR1b Target |
|--------|-----|-------|---------------|
| Signed error mean | 0.0646 | 0.0616 | < 0.065 |
| Positive % | 98.3% | 72.7% | < 75% |
| Outside ±0.15 | 4.8% | 12.2% | < 8% |
| Zero crossings | 5 | 19 | 8-12 |

The goal is to maintain positive bias reduction while reducing oscillation magnitude.
