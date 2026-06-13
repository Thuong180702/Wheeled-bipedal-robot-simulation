# Step E Extreme Height Requirement Definition

**Date:** 2026-06-07
**Scope:** Official Step E requirements + Extended-height monitoring for D2 baseline at 0.300m and 0.480m

---

## A. Official Step E Requirements

These are the validated gates from the baseline Step E specification.

### Survival
| Gate | Threshold | Source |
|------|-----------|--------|
| survived_5000_steps | True | Must complete 5000 steps without termination |
| termination_reason | null or "none" | No fall, no crash |

### Support Drift
| Gate | Threshold | Source |
|------|-----------|--------|
| support_position_error_max_abs | < 0.15 m | Max absolute support position error |
| support_position_error_final | < 0.15 m | Final support position error |

### Wheel Velocity
| Gate | Threshold | Source |
|------|-----------|--------|
| wheel_vel_mean_max_abs | < 5.0 rad/s | Max mean wheel velocity |

### Hip Yaw
| Gate | Threshold | Source |
|------|-----------|--------|
| hip_yaw_abs_max | < 0.10 rad | Max absolute hip yaw error |

### Contact Validity
| Gate | Threshold | Source |
|------|-----------|--------|
| contact_valid_percent_raw | >= 99.9% | Valid contact percentage |
| non_wheel_floor_contact_count | == 0 | No non-wheel floor contacts |
| left_wheel_contact_percent | > 0% | At least one wheel has contact |
| right_wheel_contact_percent | > 0% | At least one wheel has contact |

### Structural Invariants
| Gate | Threshold | Source |
|------|-----------|--------|
| wbc_applied | == false | No whole-body control applied |
| hidden_torque_norm_max | == 0 | No hidden torque |
| ownership_violation_count | == 0 | No ownership violations |

### Official Pass Criteria
**OFFICIAL_STEP_E_PASS** requires ALL of the above gates to pass.

---

## B. Extended-Height Monitoring Requirements

These are additional gates for extreme height targets (0.300m, 0.480m). They are monitoring gates only and do not affect official Step E status.

### Height Tracking
| Gate | Threshold | Type |
|------|-----------|------|
| target_com_z_m | 0.300 or 0.480 | Setup parameter |
| initial_com_z_m | ~target ± 0.01m | Sanity check |
| final_com_z_m | - | Report only |
| height_error_max_abs | - | Report only |
| height_error_final | - | Report only |
| height_error_RMS | - | Report only |
| final_height_error <= 0.02m | strict | Must pass for strict monitor |
| final_height_error <= 0.03m | relaxed | Must pass for relaxed monitor |

### Height Collapse Detection
| Gate | Threshold | Type |
|------|-----------|------|
| first_time_below_target_minus_1cm | - | Report step/time |
| first_time_below_target_minus_2cm | - | Report step/time |
| first_time_below_target_minus_3cm | - | Report step/time |
| no_height_collapse | true | Never drops below target - 3cm |

### Posture Stability
| Gate | Threshold | Source |
|------|-----------|--------|
| roll_y_max_abs | < 0.50 rad (~28.6°) | Lateral stability |
| pitch_x_max_abs | < 1.0 rad (~57°) | Forward stability |
| contact_remains_valid | true | Contact valid throughout |

### Extended-Height Monitor Classification
| Classification | Criteria |
|----------------|----------|
| HEIGHT_MONITOR_PASS_STRICT | final_height_error <= 0.02m AND no_height_collapse |
| HEIGHT_MONITOR_PASS_RELAXED | final_height_error <= 0.03m AND no_height_collapse |
| HEIGHT_MONITOR_FAIL_TARGET_TRACKING | final_height_error > 0.03m |
| HEIGHT_MONITOR_FAIL_COLLAPSE | Height dropped below target - 3cm |
| HEIGHT_MONITOR_INCONCLUSIVE | Missing telemetry |

---

## C. Combined Classification

| Classification | Official Step E | Extended-Height Monitor |
|----------------|-----------------|-------------------------|
| EXTREME_HEIGHT_STEP_E_FULL_PASS | PASS | PASS_STRICT |
| EXTREME_HEIGHT_STEP_E_OFFICIAL_PASS_HEIGHT_MONITORING_REQUIRED | PASS | FAIL_STRICT but PASS_RELAXED |
| EXTREME_HEIGHT_STEP_E_FAIL | FAIL | Any |
| EXTREME_HEIGHT_STEP_E_INCONCLUSIVE | INCONCLUSIVE | Any |

---

## D. Final Decision

| Decision | Meaning |
|----------|---------|
| D2_EXTREME_HEIGHTS_OFFICIAL_STEP_E_PASS | Both heights pass official Step E AND strict height monitoring |
| D2_EXTREME_HEIGHTS_OFFICIAL_PASS_HEIGHT_MONITORING_REQUIRED | Both heights pass official Step E but need height monitoring |
| D2_EXTREME_HEIGHTS_STEP_E_FAIL | One or both heights fail official Step E |
| D2_EXTREME_HEIGHTS_INCONCLUSIVE | Insufficient telemetry to determine |

---

## E. Command Template for Simulations

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/{variant}_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar
```

**Output directory:** `outputs/step_e_extreme_height_d2_official_check/`

**Required telemetry copies:**
- `low_0p300_5000_telemetry.csv`
- `high_0p480_5000_telemetry.csv`