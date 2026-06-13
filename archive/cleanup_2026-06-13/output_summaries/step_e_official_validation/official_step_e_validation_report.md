# Official Step E Validation Report

## 1. Executive summary

- Overall Step E verdict: **FAIL**.
- Final decision: **STEP_E_NOT_DONE_BALANCE_FAIL**.
- Can mark Step E DONE: **False**.
- Official production path does not confirm the controlled candidate_b result.

## 2. Input file

- CSV path: `F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation\outputs\hierarchical_controller_sim\telemetry_1780289121.csv`
- Row count: `5000`
- source_step_index range: `0` to `4999`
- Survived expected steps: `True`
- Final sim time: `49.990000` s

## 3. Structural invariant check

- Verdict: **FAIL**
- WBC off: `False`
- Hidden torque zero: `True`
- ownership_violation_count_max: `0`
- Legacy torque paths off: `True`

## 4. Position-hold check

- Verdict: **PASS**
- Metric used: `support_position_error_m`
- max_abs: `0.104457` m
- final: `0.091352` m
- RMS: `0.057047` m
- Required threshold: max_abs <= 0.15 m and final abs <= 0.15 m
- Preferred max_abs <= 0.12 m met: `True`
- Preferred final abs <= 0.10 m met: `True`

## 5. Posture validity check

- Verdict: **PASS**
- hip-yaw max_abs: `0.056700` rad
- hip-yaw RMS: `0.022819` rad
- percent abs hip-yaw error > 0.05 rad: `2.250000` %
- percent abs hip-yaw error > 0.07 rad: `0.000000` %
- percent abs hip-yaw error > 0.10 rad: `0.000000` %

## 6. Balance stability check

- Verdict: **PASS**
- pitch_x max_abs: `0.070771` rad
- roll_y max_abs: `0.012999` rad
- com_z min: `0.403835` m
- wheel_vel_mean max_abs: `3.839568` rad/s
- contact valid percent: `100.000000` %
- torque saturation max/RMS: `0.000000` / `0.000000`
- torque-rate saturation max/RMS: `missing` / `missing`

## 7. Peak window analysis

- Peak position step: `2573`
- Peak position value: `0.104457` m
- Window row indices: `2373` to `2773`
- Peak assessment: `benign`

## 8. Comparison with diagnostic candidate_b

| Metric | Official | candidate_b diagnostic |
|---|---:|---:|
| support max_abs m | 0.104457 | 0.104457 |
| hip-yaw max_abs rad | 0.056700 | 0.057555 |
| pitch max_abs rad | 0.070771 | 0.070771 |
| roll max_abs rad | 0.012999 | 0.012999 |
| com_z min m | 0.403835 | 0.403835 |
| wheel velocity max_abs rad/s | 3.839568 | 3.839568 |

## 9. Final decision

**STEP_E_NOT_DONE_BALANCE_FAIL**

## 10. Next action

Diagnose balance stability regression in official production path.

## Missing required metrics

None
