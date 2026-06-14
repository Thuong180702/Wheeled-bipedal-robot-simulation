# Phase-Aware Recenter Next Experiment Plan

**Date**: 2026-06-08
**Candidate**: F1_phase_aware_recenter_velocity_shaping
**Status**: READY FOR EXECUTION

## Experiment Overview

| Field | Value |
|-------|-------|
| Candidate | F1_phase_aware_recenter_velocity_shaping |
| Test Variant | low_0p300 |
| Duration | 500 steps |
| Controller | sagittal_velocity_damped_balance_controller |
| Config | Based on D2 baseline |

## Pass Criteria

| Metric | D2 Baseline | Target | Interpretation |
|--------|-------------|--------|----------------|
| support_position_error_m crossings >0.15 (norm-500) | 9.6 | < 9.6 | Reduced |
| hip_yaw_abs_max | 0.313 rad | < 0.130 rad | Better than E2 |
| support signed bias | 97.5% positive | < 90% | Reduced |
| wheel velocity std | 1.74 rad/s | < 2.0 rad/s | Not worsened |
| contact_valid | 100% | 100% | Maintained |
| hidden_torque_norm | 0 | 0 | Maintained |
| ownership_violation_count | 0 | 0 | Maintained |

## Implementation Details

### Code Location
- File: `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- Method: Add `compute_recenter_term()` function
- Integration: Add recenter term to wheel torque calculation

### Key Parameters

```python
# Phase detection thresholds
PITCH_SAFE_THRESHOLD = 0.05  # rad (~3°)
HIP_YAW_SAFE_THRESHOLD = 0.10  # rad

# Recenter gains
K_RECENTER = 10.0  # Nm/m
MAX_RECENTER_TAU = 1.0  # Nm
RECENTER_SMOOTH_ALPHA = 0.1  # Smoothing factor
```

### Sign Convention

```
signed_error = hip_yaw_comp_support_error_m (positive = forward drift)
recenter_tau = -k_recenter * signed_error

If signed_error > 0 (forward drift):
    recenter_tau < 0 (push wheels backward)
    ✓ Correct direction to recenter
```

## If F1 Passes 500 Steps

**Next task**: Run 2000-step evaluation

Pass criteria for 2000:
- All 500-step criteria maintained
- support_position_error_m mean < D2 baseline (0.058 m)
- survival rate maintained

## If F1 Fails

### Classification

| Failure Mode | Diagnosis | Next Action |
|--------------|------------|--------------|
| RECENTERING_TOO_WEAK | Recenter gain too low | Increase K_RECENTER or MAX_RECENTER_TAU |
| RECENTERING_TOO_AGGRESSIVE | Recenter competing with balance | Decrease MAX_RECENTER_TAU |
| PHASE_LOGIC_WRONG | Phase detection incorrect | Tune PITCH_SAFE_THRESHOLD |
| HIP_YAW_REGRESSION | Recenter term causes yaw | Add hip_yaw check to phase detection |

### Options

1. **Tune F1 parameters**: Adjust gains and thresholds
2. **Try Option A**: Modify tau_position gain scheduling
3. **Try Option B**: Hip-yaw-aware position correction
4. **Return to E2/E2b**: Accept trade-off, focus on other improvements

## Forbidden Actions

- Do NOT modify D2 baseline without explicit approval
- Do NOT increase max_position_tau beyond current value
- Do NOT enable HY2-DIV without explicit approval
- Do NOT add WBC without explicit approval
- Do NOT relax Step E gates
- Do NOT run 2000 steps without passing 500

## Evaluation Command

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller sagittal_velocity_damped_balance \
  --height-variant low_0p300 \
  --steps 500 \
  --output-dir outputs/step_e_extreme_support_fix_eval/f1_low_0p300_500 \
  --telemetry
```

## Expected Telemetry Fields

| Field | Expected |
|-------|----------|
| support_position_error_m | < 0.06 m mean |
| hip_yaw_abs_max | < 0.13 rad |
| recenter_phase_safe | True when pitch safe |
| recenter_tau | Within ±1 Nm |
| tau_position | Similar to D2 baseline |
| wheel_vel_mean | Within ±2 rad/s |
