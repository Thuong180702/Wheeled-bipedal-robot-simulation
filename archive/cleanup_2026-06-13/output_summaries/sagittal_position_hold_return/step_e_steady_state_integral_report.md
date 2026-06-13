# Step E Steady-State Integral Report

**Date:** 2026-06-01  
**Status:** integral safe but ineffective under strict gating  
**Scope:** Step E only; Step C remains blocked

## 1. Implemented mechanism

Added an optional steady-state-only centering integral to [sagittal_velocity_damped_balance_controller.py](wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py).

The integral is disabled by default and only runs when all safe steady-state gates pass:

- pitch error small
- pitch rate small
- support-position velocity small
- mean wheel velocity small
- COM height in safe range
- roll small
- contact valid

It is bounded and anti-windup safe: if any gate fails, the integral resets to zero and contributes no torque.

Telemetry added:

- `position_integral_error`
- `tau_position_integral`
- `integral_active`
- `integral_gate_reason`
- `integral_saturation_flag`
- `tau_position_p`
- `tau_position_i`
- `tau_position_total`
- `support_position_error_m`
- `support_position_velocity_m_s`
- `pitch_error_x_rad`
- `wheel_velocity_mean_rad_s`
- `com_z_m`

## 2. Validation setup

All runs used the restored no-pitch-reserve baseline plus:

```text
--vd-enable-position-integral
--vd-ki-position-integral 0.5
--vd-integral-max-abs 1.0
```

Common baseline gains:

```text
--vd-k-position 20.0
--vd-k-velocity 15.0
--vd-k-support-velocity 10.0
--vd-enable-torque-budget-aware-position
```

Telemetry files:

- V1 1000: `outputs/hierarchical_controller_sim/telemetry_1780278425.csv`
- V2 2000: `outputs/hierarchical_controller_sim/telemetry_1780278548.csv`
- V3 5000: `outputs/hierarchical_controller_sim/telemetry_1780278812.csv`

## 3. Results

| Run | Max abs support error | Final abs support error | Baseline max abs | Baseline final abs | Integral active rows |
|---|---:|---:|---:|---:|---:|
| V1 | `0.121484 m` | `0.121484 m` | `0.121484 m` | `0.121484 m` | `12 / 1000` |
| V2 | `0.385799 m` | `0.163959 m` | `0.385799 m` | `0.163960 m` | `12 / 2000` |
| V3 | `0.385799 m` | `0.052668 m` | `0.385799 m` | `0.052668 m` | `12 / 5000` |

## 4. Gate behavior

V3 integral gate reasons:

- `height_unsafe`: `2856`
- `pitch_error_large`: `2010`
- `support_velocity_large`: `86`
- `pitch_rate_large`: `35`
- `safe_steady_state`: `12`
- `contact_invalid`: `1`

The integral stayed inactive during transient and only activated during safe rows. It did not saturate.

## 5. Interpretation

The integral implementation behaved safely, but it did not correct the final bias because the final phase was not classified as safe steady state under the strict gates. The final phase is mostly gated by:

- low COM height (`height_unsafe`)
- pitch offset (`pitch_error_large`)

Therefore this run cannot claim steady-state centering solved. It also cannot claim transient solved: peak remains `0.3858 m`, above the hard minimum `0.30 m` gate.

## 6. Gate result

For V3:

- Preferred: **fail**
- Fallback: **fail**
- Hard minimum: **fail**
- Peak: `0.385799 m` > `0.30 m`
- Final: `0.052668 m` <= `0.10 m`

## 7. Classification

- pitch-reserve fix family: **rejected**
- no-pitch-reserve baseline: **restored**
- steady-state integral implementation: **safe but ineffective under strict gating**
- steady-state centering: **not solved**
- transient containment: **not solved**

## 8. Recommendation

Do **not** proceed to Step C.

Next Step E work should focus on transient authority sizing and/or review whether the final-phase height/pitch gates are correctly classifying the nominal low-height equilibrium as unsafe. Do not tune the integral blindly.
