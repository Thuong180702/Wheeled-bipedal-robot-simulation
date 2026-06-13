# Step E Equilibrium and Authority Root Cause Report

**Date:** 2026-05-31  
**Status:** Audit complete, no controller fix implemented  
**Scope:** Step E only — no Step C, no gain tuning, no integral action, no torque-limit increase

## 1. Verified telemetry source

This report uses regenerated 5000-step telemetry with the new audit-only fields present:

- Telemetry CSV: `outputs/hierarchical_controller_sim/telemetry_1780225291.csv`
- Rows: `5000`
- Verified audit columns present:
  - `tau_total_unclipped`
  - `tau_total_clipped`
  - `wheel_torque_saturation_left`
  - `wheel_torque_saturation_right`
  - `wheel_torque_rate_saturation_left`
  - `wheel_torque_rate_saturation_right`

**Reconstructed Step E command used for regeneration**

```bash
python scripts/simulate_hierarchical_controller.py \
  --steps 5000 \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-k-position 20.0 \
  --vd-k-velocity 15.0 \
  --vd-k-support-velocity 10.0 \
  --vd-max-position-tau 3.0 \
  --vd-enable-torque-budget-aware-position \
  --vd-position-tau-budget-cap 7.0 \
  --vd-pitch-reserve-tau 1.0 \
  --telemetry-decimation 1 \
  --write-run-summary-sidecar
```

**Important telemetry note**

The CSV field `controller_mode` is not a reliable proof of `balance-core`; in this file it contains `upright`, which is actually the control submode. Actual balance-core execution is verified by the command used, by the presence of balance-core-only per-joint fields (`tau_final_per_joint`, ownership and saturation masks), and by zero hidden/legacy torque.

## 2. Telemetry column mapping

### Direct mapping used in this report

| Report quantity | CSV column |
|---|---|
| `tau_pitch` | `sagittal_term_pitch` |
| `tau_pitch_rate` | `sagittal_term_pitch_rate` |
| `tau_sagittal_velocity` | `sagittal_term_com_vy` |
| `tau_wheel_velocity_left` | `sagittal_term_wheel_vel_left` |
| `tau_wheel_velocity_right` | `sagittal_term_wheel_vel_right` |
| `tau_support_velocity` | `tau_support_velocity` |
| `tau_position_raw` | `tau_position_raw` |
| `tau_position_clipped` | `tau_position_clipped` |
| `tau_balance_before_position` | `tau_balance_before_position` |
| `tau_total_unclipped` | `tau_total_unclipped` |
| `tau_total_clipped` | `tau_total_clipped` |
| final applied left wheel actuator command | `actuator_ctrl_per_joint[4]` |
| final applied right wheel actuator command | `actuator_ctrl_per_joint[9]` |
| final actuator torque saturation left | `wheel_torque_saturation_left` |
| final actuator torque saturation right | `wheel_torque_saturation_right` |
| final actuator torque-rate saturation left | `wheel_torque_rate_saturation_left` |
| final actuator torque-rate saturation right | `wheel_torque_rate_saturation_right` |

### Notes

- `tau_wheel_velocity_mean` is **not** reported as a primary field because averaging left/right damping terms can hide sign differences. Left and right wheel damping terms are reported separately.
- `actuator_ctrl_per_joint` is the final applied actuator command in balance-core mode via `tau_final_per_joint`, not the internal sagittal-controller torque before the composer.
- `tau_total_unclipped` and `tau_total_clipped` are controller-level common wheel-torque decomposition fields, not the final post-composer applied actuator commands.

## 3. Primary Step E metric

Using `support_position_error_m` as required:

- min: `-0.008281 m`
- max: `+0.459043 m`
- max abs: `0.459043 m`
- final: `+0.052668 m`

This confirms the one-sided positive support-position behavior remains present in the regenerated telemetry.

## 4. Steady-state decomposition (last 20% of run)

### State statistics

- `support_position_error_m`
  - mean: `+0.052668 m`
  - RMS: `0.052668 m`
- `support_position_velocity_m_s`
  - mean: `-8.45e-10 m/s`
  - RMS: `6.31e-08 m/s`
- `wheel_vel_mean_rad_s`
  - mean: `1.27e-08 rad/s`
  - RMS: `1.57e-06 rad/s`
- `pitch_x_error_rad`
  - mean: `+0.021067 rad`
  - RMS: `0.021067 rad`
- `pitch_rate_x`
  - mean: `1.04e-09 rad/s`
  - RMS: `1.59e-07 rad/s`
- `com_position_error_sagittal_m`
  - mean: `+0.055586 m`
- `com_z`
  - mean: `0.368027 m`

### Sagittal torque decomposition

- `tau_pitch`
  - mean: `+1.053357 Nm`
  - RMS: `1.053357 Nm`
- `tau_pitch_rate`
  - mean: `+1.04e-08 Nm`
  - RMS: `1.59e-06 Nm`
- `tau_sagittal_velocity`
  - mean: `+1.68e-08 Nm`
  - RMS: `8.47e-07 Nm`
- `tau_wheel_velocity_left`
  - mean: `-7.55e-08 Nm`
  - RMS: `7.86e-07 Nm`
- `tau_wheel_velocity_right`
  - mean: `+6.28e-08 Nm`
  - RMS: `7.88e-07 Nm`
- `tau_support_velocity`
  - mean: `+8.45e-09 Nm`
  - RMS: `6.31e-07 Nm`
- `tau_position_raw`
  - mean: `-1.053357 Nm`
  - RMS: `1.053357 Nm`
- `tau_position_clipped`
  - mean: `-1.053357 Nm`
  - RMS: `1.053357 Nm`
- `tau_balance_before_position`
  - mean: `+1.053357 Nm`
  - RMS: `1.053357 Nm`
- `tau_total_unclipped`
  - mean: `+5.90e-09 Nm`
  - RMS: `1.87e-06 Nm`
- `tau_total_clipped`
  - mean: `+5.90e-09 Nm`
  - RMS: `1.87e-06 Nm`

### Net torque balance

- `tau_balance_before_position_mean + tau_position_clipped_mean`
- result: `+5.90e-09 Nm` ≈ `0`

### Actuator torque margin

- `final_wheel_torque_margin`
  - mean: `4.999999 Nm`
  - min: `4.999990 Nm`
  - max: `4.99999996 Nm`

### Steady-state interpretation

This is a **biased equilibrium**, not continuous drift:

- support position velocity mean is effectively zero,
- support position error mean is positive and nonzero,
- balance torque mean is positive,
- position torque mean is negative,
- their sum is approximately zero.

This exactly matches:

**`steady_state_balance_torque_bias`**

and does **not** match:

- `physical_motor_limit`
- `sign_error`
- `continuous_position_drift`
- `WBC_active`
- `E0_logic_active`

## 5. Transient decomposition (steps 1300–1500)

### State statistics

- `support_position_error_m`
  - mean: `0.407910 m`
  - RMS: `0.411589 m`
  - min/max: `[0.278771, 0.459043] m`
- `support_position_velocity_m_s`
  - mean: `0.077488 m/s`
  - RMS: `0.123211 m/s`
  - min/max: `[-0.069052, 0.226609] m/s`
- `wheel_vel_mean_rad_s`
  - mean: `-1.367107 rad/s`
  - RMS: `2.192413 rad/s`
- `pitch_x_error_rad`
  - mean: `0.122675 rad`
  - RMS: `0.126129 rad`
- `pitch_rate_x`
  - mean: `-0.034020 rad/s`
  - RMS: `0.061109 rad/s`
- `com_z`
  - mean: `0.369584 m`

### Sagittal torque decomposition

- `tau_pitch`
  - mean: `+6.133756 Nm`
  - RMS: `6.306462 Nm`
- `tau_pitch_rate`
  - mean: `-0.340196 Nm`
  - RMS: `0.611092 Nm`
- `tau_sagittal_velocity`
  - mean: `-1.172093 Nm`
  - RMS: `1.735038 Nm`
- `tau_wheel_velocity_left`
  - mean: `+0.693048 Nm`
  - RMS: `1.113392 Nm`
- `tau_wheel_velocity_right`
  - mean: `+0.674059 Nm`
  - RMS: `1.079132 Nm`
- `tau_support_velocity`
  - mean: `-0.774875 Nm`
  - RMS: `1.232109 Nm`
- `tau_position_raw`
  - mean: `-8.158192 Nm`
  - RMS: `8.231786 Nm`
- `tau_position_clipped`
  - mean: `-4.000000 Nm`
  - RMS: `4.000000 Nm`
- `tau_balance_before_position`
  - mean: `+4.139448 Nm`
  - RMS: `4.139723 Nm`
- `tau_total_unclipped`
  - mean: `+0.139448 Nm`
  - RMS: `0.147387 Nm`
- `tau_total_clipped`
  - mean: `+0.139448 Nm`
  - RMS: `0.147387 Nm`

### Clipping and saturation facts

- `tau_position_raw` clipped count: `201 / 201`
- clipping fraction: `1.0`
- left final actuator torque saturation count: `0`
- right final actuator torque saturation count: `0`
- left final torque-rate saturation count: `0`
- right final torque-rate saturation count: `0`

### Sample transient steps

Representative samples:

- step `1300`
  - `tau_position_raw = -5.575 Nm`
  - `tau_position_clipped = -4.000 Nm`
  - `tau_balance_before_position = +4.064 Nm`
  - final actuator commands: `[4]=0.1050`, `[9]=0.0234`
- step `1411`
  - `tau_position_raw = -9.043 Nm`
  - `tau_position_clipped = -4.000 Nm`
  - `tau_balance_before_position = +4.218 Nm`
  - final actuator commands: `[4]=0.2159`, `[9]=0.2208`
- step `1435` (peak support error)
  - `tau_position_raw = -9.181 Nm`
  - `tau_position_clipped = -4.000 Nm`
  - `tau_balance_before_position = +4.177 Nm`
  - final actuator commands: `[4]=0.1724`, `[9]=0.1824`
- step `1500`
  - `tau_position_raw = -8.645 Nm`
  - `tau_position_clipped = -4.000 Nm`
  - `tau_balance_before_position = +4.083 Nm`
  - final actuator commands: `[4]=0.0780`, `[9]=0.0872`

## 6. Actuator torque margin proof

The old physical motor-limit conclusion is **not supported**.

### Evidence

During the transient:

- internal position term requests about `-8.2` to `-9.2 Nm`,
- internal position term is clipped to exactly `-4.0 Nm`,
- but the final applied wheel actuator commands remain small:
  - roughly `0.08` to `0.22` on joints 4 and 9 at representative peak steps,
- final actuator torque saturation flags remain `false` throughout the transient,
- final actuator torque-rate saturation flags remain `false` throughout the transient,
- final wheel torque margin remains large:
  - transient mean margin: `4.8467 Nm`
  - peak-step margins still about `4.78–4.91 Nm`.

### Conclusion

The limiter is **inside the internal authority allocation / controller composition path**, not at the final actuator command boundary.

## 7. Root-cause classification

### Primary blocker for Step E gate failure

**Primary classification: `internal_position_authority_allocation_limit`**

Reason:

- the hard-minimum gate failure is driven by the transient peak (`max_abs_support_position_error_m = 0.459 m`),
- throughout the transient, `tau_position_raw` is clipped 100% of the time,
- final actuator torque and torque-rate saturation never occur,
- therefore the main blocker is not physical actuator saturation but an internal position-authority cap that prevents the position term from using the available final actuator margin.

### Secondary contributing factors

1. **`steady_state_balance_torque_bias`**
   - real, persistent, and verified in the last 20% of the run,
   - explains why steady state settles near `+0.0527 m` instead of zero.

2. **Nonzero pitch equilibrium requirement**
   - steady-state pitch error remains positive (`~0.02107 rad`),
   - this creates a nonzero balance torque that the position term cancels.

3. **One-sided transient directionality**
   - support-position error grows mostly positive before recovering,
   - consistent with transient balance/authority allocation rather than symmetric oscillation around zero.

## 8. Rulings-out

### Not supported by evidence

- **`physical_motor_limit`**
  - ruled out by zero final actuator saturation and large remaining margin.
- **`software_torque_limit_too_low`** as final actuator bottleneck
  - not supported at the post-composer actuator layer.
- **`support_reference_pitch_reference_mismatch`**
  - references are captured from the same equilibrium snapshot.
- **`sign_error`**
  - steady-state and controller tests support correct sign behavior.
- **`continuous_position_drift`**
  - ruled out by near-zero support-position velocity mean in steady state.

## 9. Proposed minimal fix path (not implemented)

Because the primary blocker is `internal_position_authority_allocation_limit`, the minimal next fix should be:

**allow the position term to use available final actuator margin more faithfully during the transient, without stealing required pitch-balance authority.**

Concretely, the next fix should be in the family of:

- revising authority allocation so internal `tau_position_clipped` reflects true safe remaining authority at the actuator boundary,
- keeping pitch-balance protection explicit,
- not changing gains yet,
- not increasing actuator limits,
- not adding integral action yet.

For the steady-state secondary issue, a later follow-up may need either:

- pitch-reference calibration refinement, or
- slow bounded balance-bias compensation,

but only **after** the transient authority bottleneck is addressed.

## 10. Required confirmations

- **No WBC contribution in balance-core output path:** confirmed by `hidden_torque_norm = 0` and zero ownership violations.
- **No E0b/E0c/E0d reintroduced:** confirmed by active runtime path and current telemetry structure.
- **`kp_cp` disabled:** confirmed in the active velocity-damped controller defaults/runtime behavior.
- **Torque ownership unchanged:** confirmed by `ownership_violation_count = 0`.

## 11. Tests run

- `pytest tests/test_simulate_hierarchical_controller_telemetry.py -q` → pass
- `pytest tests/test_balance_core_validation_workflow.py -q` → pass
- `pytest tests/test_balance_core_components.py -q` → pass
- `pytest tests/test_torque_budget_aware_position.py -q` → pass
- `pytest tests -q -k "torque_budget or support_position or support_velocity or equilibrium or sagittal_velocity_damped"` → pass

## 12. Final audit verdict

Step E should **not** proceed to Step C.

The correct current interpretation is:

- **steady state:** biased equilibrium due to nonzero balance torque canceled by the position term,
- **gate failure blocker:** transient internal position-authority allocation limit,
- **not proven:** physical motor torque limit.
