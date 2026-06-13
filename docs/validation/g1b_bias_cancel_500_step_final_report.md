# G1b Bias Cancel Strong - 500-Step Final Report

## Summary

**Classification: G1B_IMPROVES_BUT_NOT_ENOUGH**

G1b shows modest improvement in positive bias reduction (+1.4% reduction in positive%) but overcorrects significantly, increasing time outside the ±0.15 band from 13.4% to 26.8%.

## Files Changed

None (G1b profile already existed in codebase).

## Tests Run

- `test_sagittal_velocity_damped_balance_controller.py`: 105 passed
- `test_step_e_wbc_gate_validator.py`: 4 passed
- `test_balance_core_height_variant_setup.py` + `test_balance_core_height_variant_setup_gates.py`: 26 passed
- `test_shape_posture_hip_yaw_sign.py`: 9 passed
- `test_simulation_telemetry_csv_writer.py`: 8 passed

## G1b Definition

```python
"G1b_bias_cancel_strong": SagittalAuthoritySchedule(
    profile_name="G1b_bias_cancel_strong",
    applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    max_position_tau_low_max=4.0,
    velocity_damping_scale=1.10,
    # Stronger bias cancellation
    enable_bias_cancel=True,
    bias_cancel_k=15.0,  # Higher than G1a (12.0)
    bias_cancel_max_tau=2.0,  # Higher than G1a (1.5)
    bias_cancel_filter_alpha=0.03,  # Faster than G1a (0.02)
    bias_cancel_deadband_m=0.02,  # Same as G1a
    bias_cancel_contact_gate=True,
    bias_cancel_height_gate=True,
    bias_cancel_roll_gate=True,
    bias_cancel_pitch_gate=False,
    bias_cancel_min_com_z_m=0.28,
    bias_cancel_max_com_z_m=0.50,
    bias_cancel_roll_threshold_rad=0.15,
)
```

## Exact Command

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile G1b_bias_cancel_strong \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 500 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar
```

## Telemetry Path

`outputs/step_e_extreme_support_fix_eval/g1b_low_0p300_500/g1b_low_0p300_500_telemetry.csv`

## Reference Telemetry Paths

- D2: `outputs/step_e_extreme_support_fix_eval/d2_low_0p300_500_after_csv_fix/d2_low_0p300_500_telemetry.csv`
- F1b: `outputs/step_e_extreme_support_fix_eval/f1b_low_0p300_500_after_csv_fix/f1b_low_0p300_500_telemetry.csv`
- G1a: `outputs/step_e_extreme_support_fix_eval/g1a_low_0p300_500/telemetry.csv`

## Comparison Table: D2 vs F1b vs G1a vs G1b

### Signed Support Error (hip_yaw_comp_support_error_m)

| Metric                  | D2      | F1b     | G1a     | G1b     | G1b vs G1a |
|-------------------------|---------|---------|---------|---------|------------|
| mean                    | 0.0823  | 0.0764  | 0.0724  | 0.0780  | +0.0056    |
| std                     | 0.0579  | 0.0651  | 0.0689  | 0.0798  | +0.0109    |
| min                     | -0.0035 | -0.0339 | -0.0530 | -0.0727 | -0.0197    |
| max                     | 0.1757  | 0.1695  | 0.1717  | 0.1855  | +0.0138    |
| RMS                     | 0.1007  | 0.1004  | 0.1000  | 0.1116  | +0.0116    |
| MAE                     | 0.0826  | 0.0832  | 0.0836  | 0.0944  | +0.0108    |
| final                   | 0.0593  | -0.0327 | -0.0045 | -0.0138 | -0.0093    |
| **positive%**           | **93.0%** | **82.8%** | **81.8%** | **80.4%** | **-1.4%** |
| negative%               | 6.6%    | 16.8%   | 17.8%   | 19.2%   | +1.4%      |
| **outside [-0.15,+0.15]%** | **19.2%** | **16.2%** | **13.4%** | **26.8%** | **+13.4%** |
| below -0.05             | 0       | 0       | 19      | 46      | +27        |
| below -0.10             | 0       | 0       | 0       | 0       | 0          |
| below -0.15             | 0       | 0       | 0       | 0       | 0          |
| zero crossings           | 5       | 6       | 6       | 6       | 0          |
| longest positive         | 316     | 256     | 228     | 211     | -17        |
| longest negative         | 28      | 66      | 81      | 85      | +4         |

### Tau Pitch (Bias Source)

| Metric           | D2      | F1b     | G1a     | G1b     |
|------------------|---------|---------|---------|---------|
| mean             | 2.5992  | 2.6976  | 2.8761  | 3.0774  |
| positive%        | 89.2%   | 82.8%   | 84.2%   | 83.0%   |
| max              | 5.5527  | 5.5158  | 5.9271  | 6.3341  |
| min              | -0.4146 | -1.6926 | -1.8547 | -2.7177 |
| RMS              | 3.1448  | 3.5211  | 3.7477  | 4.1699  |

### Tau Position (Counter-balance)

| Metric     | D2      | F1b     | G1a     | G1b     |
|------------|---------|---------|---------|---------|
| mean       | -2.6146 | -2.3618 | -2.2157 | -2.1613 |

### Pitch/Roll Stability

| Metric           | D2      | F1b     | G1a     | G1b     |
|------------------|---------|---------|---------|---------|
| pitch mean (deg) | 2.98    | 3.09    | 3.30    | 3.53    |
| pitch positive%  | 89.2%   | 82.8%   | 84.2%   | 83.0%   |
| pitch max (deg)  | 6.36    | 6.32    | 6.79    | 7.26    |
| pitch min (deg)  | -0.48   | -1.94   | -2.13   | -3.11   |
| pitch RMS (deg)  | 3.60    | 4.03    | 4.29    | 4.78    |
| roll max (deg)   | 0.76    | 0.75    | 0.67    | 0.66    |

### Hip Yaw Monitor

| Metric             | D2      | F1b     | G1a     | G1b     |
|--------------------|---------|---------|---------|---------|
| hip_yaw_abs_max mean | 0.0446 | 0.0399  | 0.0311  | 0.0267  |
| hip_yaw_abs_max max | 0.1018 | 0.1376  | 0.0935  | 0.0958  |
| hip_yaw_abs_max final | 0.0847 | 0.1376 | 0.0613  | 0.0456  |

## Analysis

### Positive Bias Reduction (Good)

G1b achieves the lowest positive% at 80.4%, down from G1a's 81.8% and F1b's 82.8%. This represents progress in reducing the one-sided positive drift.

### Overcorrection (Bad)

However, G1b significantly overcorrects:
- **outside band increases from 13.4% (G1a) to 26.8% (G1b)**
- **below -0.05 increases from 19 steps (G1a) to 46 steps (G1b)**
- Pitch min drops from -2.13° (G1a) to -3.11° (G1b)
- tau_pitch_max increases from 5.93 Nm (G1a) to 6.33 Nm (G1b)
- tau_pitch_min decreases from -1.85 Nm (G1a) to -2.72 Nm (G1b)

The stronger bias cancellation torque (2.0 Nm vs 1.5 Nm) and faster filter (0.03 vs 0.02) are pushing the system into negative territory too aggressively.

### Hip Yaw Monitor (Good)

Hip yaw remains stable across all profiles. G1b actually shows the lowest hip_yaw_abs_max mean (0.0267) compared to G1a (0.0311) and F1b (0.0399).

### Root Cause Insight

The persistent positive tau_pitch bias (2.6-3.1 Nm mean, 83-89% positive) is the root cause of the one-sided drift. The tau_pitch bias is stronger than what either G1a or G1b can cancel:

- tau_pitch mean: 3.08 Nm (G1b)
- bias_cancel_max_tau: 2.0 Nm (G1b)
- Ratio: 1.54x

The bias cancellation torque is fundamentally too weak to fully counteract the tau_pitch bias.

## Classification Rationale

**G1B_IMPROVES_BUT_NOT_ENOUGH** because:

1. **Positive% decreases**: 80.4% (G1b) vs 81.8% (G1a) vs 82.8% (F1b) - progress in correct direction
2. **But outside band increases significantly**: 26.8% (G1b) vs 13.4% (G1a) - overcorrecting
3. **Below -0.05 increases**: 46 steps (G1b) vs 19 steps (G1a) - too aggressive
4. **Root cause not addressed**: tau_pitch bias (3.08 Nm) still exceeds bias_cancel_max_tau (2.0 Nm)

## Recommendations

### Do NOT Continue Increasing Bias Cancellation Parameters

Further increasing bias_cancel_k or bias_cancel_max_tau will likely worsen overcorrection without fixing the root cause.

### Recommended Next Step: Direct Tau Pitch Bias Audit

The root cause is persistent positive tau_pitch bias around 2.6-3.1 Nm. The next investigation should target:

1. **Source of tau_pitch bias**: Why is tau_pitch persistently positive?
   - Is it from pitch_kp × pitch_error?
   - Is it from pitch_rate estimation?
   - Is it from the WBC or shape posture controller?

2. **Potential fixes at source**:
   - Reduce pitch_kp if it's too high
   - Add pitch bias compensation at source
   - Adjust pitch reference offset
   - Investigate pitch sensor or estimation issues

3. **Alternative approach**: Instead of downstream wheel bias cancellation, address the tau_pitch bias directly at the sagittal controller level.

## Conclusion

G1b_bias_cancel_strong shows that stronger persistent bias cancellation can reduce positive% (80.4% vs 81.8%) but overcorrects significantly (26.8% vs 13.4% outside band). The fundamental limitation is that the bias cancellation torque (max 2.0 Nm) is insufficient to counteract the persistent tau_pitch bias (~3.1 Nm). 

**Next step should be a direct tau_pitch bias audit to understand and address the root cause, rather than continuing to increase downstream bias cancellation torque.**

---

**Final Decision: G1B_IMPROVES_BUT_NOT_ENOUGH**

Do NOT proceed to 2000-step validation with G1b.
Do NOT increase bias cancellation parameters further.
Recommend direct tau_pitch bias source investigation.