# Step C Contact Invalid Audit

- Input telemetry: `outputs\step_c_height_recovery_short\nominal_telemetry.csv`
- Row count: `500`
- Invalid contact row count: `1`
- Contact valid percent: `99.800000`
- Invalid source steps: `[0]`
- Invalid times: `[0.0]`
- Classification: **startup_contact_artifact**
- Recommended next action: Validator-only revision is appropriate: consider a small startup contact grace window before applying the strict 99.9% contact-valid threshold, and confirm with official 5000-step validation.
- Controller behavior changed: `False`
- WBC added: `False`
- WBC applied at invalid rows: `False`

## Contact pattern

- Only at startup: `True`
- Isolated single-sample blips: `True`
- Consecutive invalid rows: `False`
- Consecutive invalid groups: `[[0]]`
- Height recovery time: `0.0`
- Invalid rows after height recovery: `True`
- Wheel contacts true at invalid rows: `True`
- Non-wheel contacts zero at invalid rows: `True`
- Contact supervisor double_contact at invalid rows: `True`

## Signals at invalid rows

```json
{
  "source_step_index": 0,
  "time": 0.0,
  "step": 0,
  "sim_time_s": 0.0,
  "contact_force_valid": false,
  "left_wheel_contact": true,
  "right_wheel_contact": true,
  "left_wheel_floor_contact": true,
  "right_wheel_floor_contact": true,
  "left_contact_active": true,
  "right_contact_active": true,
  "contact_supervisor_state": "double_contact",
  "contact_previous_state": "none",
  "left_contact_force_world_z": 20.508811950683597,
  "right_contact_force_world_z": 22.049598693847656,
  "non_wheel_floor_contacts": 0,
  "total_wheel_floor_fz": 42.55841054498711,
  "total_contact_force_z": 42.55841064453125,
  "contact_duration_s": 0.0,
  "contact_transition_event": "initial_double_contact",
  "com_z_m": 0.4041118323802948,
  "pitch_x_rad": 0.0,
  "roll_y_rad": 0.0,
  "wheel_vel_mean_rad_s": 0.0,
  "support_position_error_m": 0.0,
  "hip_yaw_abs_max": 0.0,
  "tau_saturation_rate": 0.0,
  "torque_saturation_mask_per_joint": "False,False,False,False,False,False,False,False,False,False",
  "torque_rate_saturation_mask_per_joint": "False,False,False,True,False,False,False,False,True,False",
  "tau_wbc_correction": "0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000",
  "tau_wbc_norm": 11.987911224365234,
  "hidden_torque_norm": 0.0,
  "ownership_violation_count": 0
}
```

## Abnormal checks at invalid rows

```json
{
  "com_z_m": {
    "values_at_invalid_rows": [
      0.4041118323802948
    ],
    "min_at_invalid_rows": 0.4041118323802948,
    "limit_min": 0.38,
    "abnormal": false
  },
  "pitch_x_rad": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_abs_at_invalid_rows": 0.0,
    "limit": 0.1,
    "abnormal": false
  },
  "roll_y_rad": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_abs_at_invalid_rows": 0.0,
    "limit": 0.05,
    "abnormal": false
  },
  "wheel_vel_mean_rad_s": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_abs_at_invalid_rows": 0.0,
    "limit": 5.0,
    "abnormal": false
  },
  "support_position_error_m": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_abs_at_invalid_rows": 0.0,
    "limit": 0.15,
    "abnormal": false
  },
  "hip_yaw_abs_max": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_abs_at_invalid_rows": 0.0,
    "limit": 0.07,
    "abnormal": false
  },
  "tau_saturation_rate": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_at_invalid_rows": 0.0,
    "abnormal": false
  },
  "hidden_torque_norm": {
    "values_at_invalid_rows": [
      0.0
    ],
    "max_at_invalid_rows": 0.0,
    "abnormal": false
  }
}
```

## Nearby ?20-step window summary

```json
{
  "com_z_m": {
    "min": 0.4038352966308594,
    "max": 0.4066476821899414,
    "max_abs": 0.4066476821899414
  },
  "pitch_x_rad": {
    "min": 0.0,
    "max": 0.002604294206445,
    "max_abs": 0.002604294206445
  },
  "roll_y_rad": {
    "min": 0.0,
    "max": 0.0015193548215059,
    "max_abs": 0.0015193548215059
  },
  "wheel_vel_mean_rad_s": {
    "min": -0.0238454065984115,
    "max": 1.235780954360962,
    "max_abs": 1.235780954360962
  },
  "support_position_error_m": {
    "min": -0.0068471081097522,
    "max": 0.0,
    "max_abs": 0.0068471081097522
  },
  "hip_yaw_abs_max": {
    "min": 0.0,
    "max": 0.0023051933385431,
    "max_abs": 0.0023051933385431
  },
  "tau_saturation_rate": {
    "min": 0.0,
    "max": 0.0,
    "max_abs": 0.0
  },
  "torque_saturation_mask_per_joint": {
    "min": NaN,
    "max": NaN,
    "max_abs": NaN
  },
  "torque_rate_saturation_mask_per_joint": {
    "min": NaN,
    "max": NaN,
    "max_abs": NaN
  }
}
```

## Artifacts

- contact_invalid_rows_csv: `outputs\step_c_contact_invalid_audit\contact_invalid_rows.csv`
- contact_invalid_window_csv: `outputs\step_c_contact_invalid_audit\contact_invalid_window.csv`
- contact_invalid_audit_json: `outputs\step_c_contact_invalid_audit\contact_invalid_audit.json`
- contact_invalid_audit_report_md: `outputs\step_c_contact_invalid_audit\contact_invalid_audit_report.md`

## WBC numeric correction check

- tau_wbc_correction numeric norms at invalid rows: `[0.0]`
- Raw `tau_wbc_norm`, if present, is treated as diagnostic-only and not proof of applied WBC.
