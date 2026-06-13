# Old Success Artifact Inventory
# Generated: 2026-06-06

## Step E Results

**File:** `docs/validation/step_e_height_variant_robustness_done.md`
**Date:** 2026-06-02
**Status:** DONE

### Controller Profile
`candidate_D2_wheel_velocity_damping_light`

```python
SagittalAuthoritySchedule(
    profile_name="candidate_D2_wheel_velocity_damping_light",
    applies_to_variants=("high_tiny", "high_small"),
    position_tau_cap_by_variant=(("high_tiny", 4.0), ("high_small", 4.0)),
    pitch_tau_scale=1.0,
    velocity_damping_scale=1.10,
)
```

### Five Variants - Old Results

| Variant | Verdict | Support max (m) | HipYaw max (rad) | Pitch max (rad) | Wheel max (rad/s) |
|---------|---------|----------------|------------------|-----------------|-------------------|
| nominal | PASS | 0.106 | 0.056 | 0.071 | 3.87 |
| low_tiny | PASS | 0.110 | 0.042 | 0.073 | 4.04 |
| high_tiny | PASS | 0.124 | 0.038 | 0.092 | 4.12 |
| low_small | PASS | 0.106 | 0.057 | 0.071 | 3.99 |
| high_small | PASS | 0.135 | 0.030 | 0.096 | 4.77 |

### Structural Invariants
- WBC applied = false
- Hidden torque norm max = 0.0
- Ownership violation count = 0

### Validation Artifacts
- Final audit: `outputs/step_e_height_variant_position_hold_final/`
- Candidate telemetry: `outputs/step_e_height_variant_position_hold_final/candidate_telemetry/`
- Summary JSON: `outputs/step_e_height_variant_position_hold_final/step_e_hv_sagittal_schedule_fix_summary.json`

---

## Step C Results

**File:** `docs/validation/step_c_height_recovery_done.md`
**Date:** 2026-06-02
**Status:** DONE

### Controller Profile
Same as Step E: `candidate_D2_wheel_velocity_damping_light`

### Five Variants - Old Results

| Case | Verdict | Recovery time (s) | Support max (m) | HipYaw max (rad) | Wheel max (rad/s) |
|------|---------|-------------------|-----------------|------------------|-------------------|
| nominal | PASS | 0.0 | 0.106 | 0.056 | 3.87 |
| low_tiny | PASS | 0.0 | 0.110 | 0.042 | 4.04 |
| high_tiny | PASS | 0.0 | 0.124 | 0.038 | 4.12 |
| low_small | PASS | 0.0 | 0.106 | 0.057 | 3.99 |
| high_small | PASS | 0.0 | 0.135 | 0.030 | 4.77 |

### Structural Invariants
- WBC applied = false
- Hidden torque norm max = 0.0
- Ownership violation count = 0
- Step E structural invariants preserved = true

### Validation Command
```bash
python scripts/run_step_c_height_recovery.py \
  --use-height-variants \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --output-dir outputs/step_c_height_recovery_after_step_e_hv_fix \
  --continue-after-failure
```

### Artifacts
- Output: `outputs/step_c_height_recovery_after_step_e_hv_fix/`
- Summary: `step_c_pass_fail_summary.json`
- Metrics: `step_c_height_recovery_metrics.json`

---

## Height Variant Setup Files

All five variants use setup files from:
`outputs/balance_core_true_height_variants/`

| Variant | Setup File | Target CoM (m) | Achieved CoM (m) | hip_pitch_ref | knee_ref |
|---------|-------------|-----------------|------------------|---------------|----------|
| nominal | variant_nominal/variant_setup.json | 0.404 | 0.404 | 0.926052 | 1.748364 |
| low_tiny | variant_low_tiny/variant_setup.json | 0.399 | 0.399 | 0.936578 | 1.779943 |
| high_tiny | variant_high_tiny/variant_setup.json | 0.409 | 0.409 | 0.915526 | 1.716785 |
| low_small | variant_low_small/variant_setup.json | 0.394 | 0.395 | 0.957631 | 1.800996 |
| high_small | variant_high_small/variant_setup.json | 0.414 | 0.413 | 0.894473 | 1.695732 |

---

## HY2-DIV Status in Old Pass

**HY2-DIV was NOT enabled in the old Step E/C pass.**

Evidence:
- `--enable-hip-yaw-divergence-damping` flag was not used in the old commands
- Old Step E summary shows no HY2-DIV telemetry fields
- Old reports do not mention HY2-DIV

---

## Hip-Yaw Sign Fix Status

The old pass was performed AFTER the hip-yaw sign fix was applied.

Evidence:
- The shape_posture_controller.py was updated with correct sign conventions
- All five variants passed with bounded hip-yaw values (< 0.06 rad)

---

## Gate Definitions (Old)

### Step E Gates
- support_position_error max < 0.15 m
- wheel_vel_mean_max_abs < 5.0 rad/s
- hip_yaw_abs_max < 0.10 rad (preferred)
- contact_valid_percent_raw >= 99.9
- non_wheel_floor_contacts = 0

### Step C Gates
- Height recovery within tolerance
- Same structural invariants as Step E

---

## Key Metrics Comparison

| Metric | nominal | low_tiny | high_tiny | low_small | high_small |
|--------|---------|----------|-----------|-----------|------------|
| Support max (m) | 0.106 | 0.110 | 0.124 | 0.106 | 0.135 |
| HipYaw max (rad) | 0.056 | 0.042 | 0.038 | 0.057 | 0.030 |
| Pitch max (rad) | 0.071 | 0.073 | 0.092 | 0.071 | 0.096 |
| Wheel max (rad/s) | 3.87 | 4.04 | 4.12 | 3.99 | 4.77 |