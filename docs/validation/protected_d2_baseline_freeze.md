# Protected D2 Baseline Freeze

**Date:** 2026-06-06
**Profile:** `candidate_D2_wheel_velocity_damping_light`
**Decision:** `BASELINE_LADDER_MAP_COMPLETE`

---

## Freeze Status

| Property | Value |
|----------|-------|
| Controller Mode | balance-core |
| Sagittal Controller | velocity-damped |
| Sagittal Authority Profile | candidate_D2_wheel_velocity_damping_light |
| HY2-DIV | **DISABLED** (must remain disabled by default) |
| WBC | balance-core four-source stack |
| Profile Status | **PROTECTED** - Do not modify unless root cause proven necessary |

---

## Protected Behaviors

The following are protected against modification without explicit approval:

1. **D2 profile gains** (`candidate_D2_wheel_velocity_damping_light`):
   - `k_pitch: 40.0`
   - `k_pitch_rate: 2.5`
   - `k_wheel_velocity: 1.5`
   - `k_position: 0.0`
   - `max_wheel_torque: 10.0`

2. **HY2-DIV gate**: Must remain `False` by default for balance-core D2 baseline.

3. **WBC configuration**: Current balance-core four-source stack must remain unchanged.

4. **Old five-variant Step E/C baseline**: Remains valid and protected.

---

## Rollback Command (D2 Baseline Only)

To reproduce the exact D2 baseline behavior at any height:

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant <variant_name> \
  --steps <steps>
```

Examples:

```bash
# 5000-step at nominal
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant nominal \
  --steps 5000

# 5000-step at low_0p300
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant low_0p300 \
  --steps 5000

# Smoke test at 0.380m
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant low_0p380 \
  --steps 100
```

---

## Extension Work Constraints

Any new work in this repository must:

1. **NOT modify** `candidate_D2_wheel_velocity_damping_light` profile defaults
2. **NOT enable HY2-DIV** by default
3. **NOT modify WBC** unless explicitly required for a proven fix
4. **NOT regress** the old five-variant baseline (low_small, low_tiny, nominal, high_tiny, high_small)

New fixes must be:
- Opt-in or profile-specific
- Proven necessary via root cause analysis
- Validated against regression tests before merging

---

## Experiment 0 Results Summary

| Height (m) | 100-step | 500-step | 2000-step | 5000-step |
|------------|----------|----------|-----------|-----------|
| 0.300 | PASS | PASS | PASS | **PASS** (CoM ~0.273m, degraded) |
| 0.320 | PASS | PASS | PASS | - |
| 0.340 | PASS | PASS | PASS | - |
| 0.360 | PASS | PASS | PASS | - |
| 0.380 | PASS | PASS | PASS | - |
| 0.430 | PASS | PASS | PASS | - |
| 0.450 | PASS | PASS | PASS | - |
| 0.465 | PASS | PASS | - | - |
| 0.480 | PASS | PASS | PASS | **PASS** (wide oscillation) |

**Key Finding:** Baseline D2 controller survives the full ladder (0.300m - 0.480m) without HY2-DIV.

**Remaining Degradation:**
- 0.300m: CoM collapses from 0.300m to ~0.273m (not a fall, but height tracking issue)
- 0.380m: hip-roll torque grows from 22 Nm (500-step) to 57 Nm (2000-step)

---

## Audit Trail

This freeze document was created as part of the D2 height tracking and hip-roll audit task.

Related files:
- `docs/validation/height_range_extension_experiment_0_baseline_ladder_report.md`
- `outputs/height_range_extension_experiment_0/experiment_0_baseline_ladder_summary.json`
- `scripts/generate_ladder_height_setups.py`
