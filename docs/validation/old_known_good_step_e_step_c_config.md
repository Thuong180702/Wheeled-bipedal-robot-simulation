# Old Known-Good Step E / Step C Configuration

**Date:** 2026-06-02
**Baseline ID:** `five_variant_step_e_step_c_candidate_D2_wheel_velocity_damping_light`

## 1. Profile That Passed

**Profile:** `candidate_D2_wheel_velocity_damping_light`

```python
SagittalAuthoritySchedule(
    profile_name="candidate_D2_wheel_velocity_damping_light",
    applies_to_variants=("high_tiny", "high_small"),
    position_tau_cap_by_variant=(("high_tiny", 4.0), ("high_small", 4.0)),
    pitch_tau_scale=1.0,
    velocity_damping_scale=1.10,
)
```

This profile:
- Retains 4.0 Nm position cap for high_tiny and high_small variants
- Increases sagittal velocity damping by 10% for high-height variants
- Leaves low-height and nominal behavior unchanged

## 2. Five Variants Mapping

| User Label | -5/-1/+1/+5 | Setup File | Target CoM (m) |
|------------|-------------|------------|-----------------|
| low_small | -5 | `variant_low_small/variant_setup.json` | 0.394 |
| low_tiny | -1 | `variant_low_tiny/variant_setup.json` | 0.399 |
| nominal | 0 | `variant_nominal/variant_setup.json` | 0.404 |
| high_tiny | +1 | `variant_high_tiny/variant_setup.json` | 0.409 |
| high_small | +5 | `variant_high_small/variant_setup.json` | 0.414 |

All setup files are in: `outputs/balance_core_true_height_variants/`

## 3. Exact Command Flags

### Step E Command
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --height-variant-setup outputs/balance_core_true_height_variants/variant_{name}/variant_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light
```

### Step C Command
```bash
python scripts/run_step_c_height_recovery.py \
  --use-height-variants \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --output-dir outputs/step_c_height_recovery_after_step_e_hv_fix \
  --continue-after-failure
```

## 4. HY2-DIV Status

**HY2-DIV was NOT enabled in the old pass.**

The old pass used:
- No `--enable-hip-yaw-divergence-damping` flag
- No HY2-DIV gains (k=0, kd=0)
- Default HY2-DIV gate (z_low=0.300, z_high=0.393)

This is a **baseline without hip-yaw divergence damping**.

## 5. Hip-Yaw Sign Fix

The old pass was performed **AFTER** the hip-yaw sign fix was applied.

Evidence:
- shape_posture_controller.py had correct sign conventions
- All five variants passed with hip_yaw_abs_max < 0.06 rad
- No sign convention issues in telemetry

## 6. WBC Status

**WBC was NOT applied in the old pass.**

Evidence:
- All runs: `wbc_applied = false`
- `hidden_torque_norm_max = 0.0`
- `ownership_violation_count = 0`

## 7. Step E Official Gates

| Gate | Threshold |
|------|-----------|
| support_position_error_max_abs | < 0.15 m |
| wheel_vel_mean_max_abs | < 5.0 rad/s |
| hip_yaw_abs_max (preferred) | < 0.10 rad |
| contact_valid_percent_raw | >= 99.9% |
| non_wheel_floor_contacts | = 0 |

## 8. Step C Official Gates

| Gate | Threshold |
|------|-----------|
| Height recovery within tolerance | < 0.05 m |
| Same structural invariants as Step E | true |

## 9. Old Results Summary

### Step E (5000 steps)

| Variant | Verdict | Support max | HipYaw max | Pitch max | Wheel max |
|---------|---------|-------------|------------|-----------|-----------|
| nominal | PASS | 0.106 m | 0.056 rad | 0.071 rad | 3.87 rad/s |
| low_tiny | PASS | 0.110 m | 0.042 rad | 0.073 rad | 4.04 rad/s |
| high_tiny | PASS | 0.124 m | 0.038 rad | 0.092 rad | 4.12 rad/s |
| low_small | PASS | 0.106 m | 0.057 rad | 0.071 rad | 3.99 rad/s |
| high_small | PASS | 0.135 m | 0.030 rad | 0.096 rad | 4.77 rad/s |

### Step C

| Variant | Verdict | Recovery time |
|---------|---------|---------------|
| nominal | PASS | 0.0 s |
| low_tiny | PASS | 0.0 s |
| high_tiny | PASS | 0.0 s |
| low_small | PASS | 0.0 s |
| high_small | PASS | 0.0 s |

## 10. Structural Invariants (Both Steps)

- WBC applied = false
- Hidden torque norm max = 0.0
- Ownership violation count = 0

## 11. Key Difference from Current HY2-DIV A0 Validation

The old pass used a **different height targeting**:
- Old variants: nominal=0.404m, low_tiny=0.399m, low_small=0.394m, high_tiny=0.409m, high_small=0.414m
- Current HY2-DIV A0 validation used: nominal=0.404m, low_0p300=0.300m, high_0p480=0.480m

The current validation tested **extreme heights (0.30m and 0.48m)** that are outside the validated envelope.
The old pass tested **modest height variants (±0.005m to ±0.01m)** within the validated envelope.