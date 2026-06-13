# Step E Total-Torque-Bound Authority Fix Report

**Date:** 2026-05-31  
**Status:** Fix implemented and validated; Step C not approved yet  
**Scope:** Step E only

## 1. Root cause and fix

**Primary blocker:** `internal_position_authority_allocation_limit`

The previous position-authority cap clipped `tau_position` against a fixed internal budget. That was too conservative when `tau_position` opposed `tau_balance_before_position`, because it reduced the net wheel torque instead of pushing the final actuator torque toward saturation.

**New allocator formula**

For

- `tau_balance = tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_wheel_velocity + tau_support_velocity`
- `tau_position_raw = -k_position * support_position_error`
- `max_tau_wheel = wheel actuator limit`

clip position authority by final total torque bounds:

- `tau_position_lower_bound = -max_tau_wheel - tau_balance`
- `tau_position_upper_bound = +max_tau_wheel - tau_balance`
- `tau_position_clipped = clip(tau_position_raw, tau_position_lower_bound, tau_position_upper_bound)`

This preserves safety because final total torque still stays inside the wheel limit.

## 2. Validation inputs

- V1: 1000 steps
- V2: 2000 steps
- V3: 5000 steps
- Gains unchanged:
  - `k_position = 20.0`
  - `k_velocity = 15.0`
  - `k_support_velocity = 10.0`
  - `kp_cp = 0.0`
  - WBC off

Telemetry source files:

- `outputs/hierarchical_controller_sim/telemetry_1780226495.csv`
- `outputs/hierarchical_controller_sim/telemetry_1780226769.csv`
- `outputs/hierarchical_controller_sim/telemetry_1780227230.csv`

## 3. Test results

Passed:

- `pytest tests/test_torque_budget_aware_position.py -q`
- `pytest tests/test_simulate_hierarchical_controller_telemetry.py -q`
- `pytest tests/test_balance_core_validation_workflow.py -q`
- `pytest tests/test_balance_core_components.py -q`
- `pytest tests -q -k "torque_budget or position_authority or sagittal_velocity_damped or support_position"`

## 4. Validation summary

### V1 — 1000 steps

- survived_steps: `1000`
- termination_reason: `completed`
- support_position_error_m: min `-0.008281`, max `+0.121484`, final `+0.121484`
- max_abs_support_position_error_m: `0.121484`
- centering: one-sided / biased
- support_position_velocity_m_s RMS: `0.028481`
- pitch range: `[-0.001730, +0.054677] rad`
- roll range: `[-0.005822, +0.002934] rad`
- yaw range: `[-0.007622, +0.020493] rad`
- com_z range: `[0.403841, 0.408641] m`
- wheel velocity range: `[-2.054978, +1.320658] rad/s`
- tau_balance_before_position: `[-0.836988, +2.452997] Nm`
- tau_position_raw: `[-2.429681, +0.165623] Nm`
- tau_position_clipped: `[-2.429681, +0.165623] Nm`
- tau_position_lower_bound: `[-7.452997, -4.163012] Nm`
- tau_position_upper_bound: `[+2.547003, +5.836988] Nm`
- tau_total_before_final_clip: `[-0.831395, +1.293751] Nm`
- tau_total_after_final_clip: `[-0.831395, +1.293751] Nm`
- final_wheel_torque_margin: `[3.693829, +5.000000] Nm`
- actuator_ctrl_per_joint[4]: `[-0.847665, +1.281331] Nm`
- actuator_ctrl_per_joint[9]: `[-0.815125, +1.306171] Nm`
- torque saturation/rate saturation: `0`
- ownership_violation_count: `0`
- hidden_torque_norm: `0`
- tau_wbc_norm: `12.832582`

### V2 — 2000 steps

- survived_steps: `2000`
- termination_reason: `completed`
- support_position_error_m: min `-0.008281`, max `+0.385799`, final `+0.163960`
- max_abs_support_position_error_m: `0.385799`
- centering: one-sided / biased
- support_position_velocity_m_s RMS: `0.047942`
- pitch range: `[-0.001730, +0.181527] rad`
- roll range: `[-0.041832, +0.002934] rad`
- yaw range: `[-0.179219, +0.062378] rad`
- com_z range: `[0.363021, 0.408641] m`
- wheel velocity range: `[-4.753519, +1.320658] rad/s`
- tau_balance_before_position: `[-0.836988, +8.054951] Nm`
- tau_position_raw: `[-7.715974, +0.165623] Nm`
- tau_position_clipped: `[-7.715974, +0.165623] Nm`
- tau_position_lower_bound: `[-13.054951, -4.163012] Nm`
- tau_position_upper_bound: `[-3.054951, +5.836988] Nm`
- tau_total_before_final_clip: `[-1.074927, +1.293751] Nm`
- tau_total_after_final_clip: `[-1.074927, +1.293751] Nm`
- final_wheel_torque_margin: `[3.693829, +5.000000] Nm`
- actuator_ctrl_per_joint[4]: `[-1.035009, +1.281331] Nm`
- actuator_ctrl_per_joint[9]: `[-1.114846, +1.306171] Nm`
- torque saturation/rate saturation: `0`
- ownership_violation_count: `0`
- hidden_torque_norm: `0`
- tau_wbc_norm: `36.547367`

### V3 — 5000 steps

- survived_steps: `5000`
- termination_reason: `completed`
- support_position_error_m: min `-0.008281`, max `+0.385799`, final `+0.052668`
- max_abs_support_position_error_m: `0.385799`
- centering: one-sided / biased
- support_position_velocity_m_s RMS: `0.031804`
- pitch range: `[-0.001730, +0.181527] rad`
- roll range: `[-0.041832, +0.002934] rad`
- yaw range: `[-0.179219, +0.062378] rad`
- com_z range: `[0.363021, 0.408641] m`
- wheel velocity range: `[-4.753519, +1.320658] rad/s`
- tau_balance_before_position: `[-0.836988, +8.054951] Nm`
- tau_position_raw: `[-7.715974, +0.165623] Nm`
- tau_position_clipped: `[-7.715974, +0.165623] Nm`
- tau_position_lower_bound: `[-13.054951, -4.163012] Nm`
- tau_position_upper_bound: `[-3.054951, +5.836988] Nm`
- tau_total_before_final_clip: `[-1.074927, +1.293751] Nm`
- tau_total_after_final_clip: `[-1.074927, +1.293751] Nm`
- final_wheel_torque_margin: `[3.693829, +5.000000] Nm`
- actuator_ctrl_per_joint[4]: `[-1.035009, +1.281331] Nm`
- actuator_ctrl_per_joint[9]: `[-1.114846, +1.306171] Nm`
- torque saturation/rate saturation: `0`
- ownership_violation_count: `0`
- hidden_torque_norm: `0`
- tau_wbc_norm: `36.547367`

## 5. Gate checks

- ±0.10 m for 5000 steps: **failed**
- ±0.15 m for 5000 steps: **failed**
- hard minimum (`max_abs <= 0.30 m` and final abs error <= 0.10 m): **failed**

## 6. Before/after comparison

Previous Step E range: `[-0.0083, +0.4933]`  
New Step E range: `[-0.0083, +0.3858]`

The peak excursion improved, but the trajectory still does not meet the gate.

## 7. Interpretation

- final actuator saturation: **not the blocker**
- torque-rate saturation: **not the blocker**
- physical motor limit: **not proven**
- internal position authority is now less conservative, but still not enough for Step E gate closure
- steady-state bias remains present

## 8. Height variants

Not run.

Reason: V3 did not meet the hard minimum, so the requested high_5cm / low_5cm follow-up was not justified.

## 9. Confirmations

- no WBC: confirmed
- no E0b/E0c/E0d: confirmed
- kp_cp disabled: confirmed
- torque ownership unchanged: confirmed

## 10. Recommendation

**Do not start Step C yet.**

This allocator fix improved the authority model, but Step E still fails the acceptance gates. The next step should focus on the remaining steady-state bias / reference calibration path, not on Step C.
