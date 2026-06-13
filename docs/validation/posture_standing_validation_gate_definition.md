# Posture/Standing Validation Gate Definition

**Phase:** POSTURE_STANDING_VALIDATION
**Date:** 2026-06-06
**Profile:** HY2-DIV A0
**Steps:** 5000

## Objective

Validate posture and standing stability **before** support-position drift is addressed.
This phase prioritizes survival, contact, height, and hip-yaw divergence control.

**Explicitly NOT this phase:**
- Official Step E pass
- Support-position drift fix
- Step C or Step D implementation

## Priority Order

### Priority 1: Survival / Contact / Height

Required gates:
1. `survived_full_run` — robot survived all 5000 steps
2. `wbc_applied` — WBC applied = false throughout
3. `hidden_torque` — hidden torque = 0 throughout
4. `ownership_violations` — ownership violations = 0 throughout
5. `contact_valid` — contact validity acceptable throughout
6. `no_nonwheel_contacts` — no non-wheel floor contacts
7. `height_error_acceptable` — final height error within tolerance
8. `no_height_collapse` — no catastrophic height collapse

### Priority 2: Posture

Required gates:
9. `hip_yaw_divergence_bounded` — divergence RMS within target
10. `hip_yaw_abs_max_bounded` — hip_yaw_abs_max reported, preferably < 0.30 rad
11. `legs_not_twisted` — legs do not twist inward/outward
12. `roll_bounded` — roll does not collapse
13. `body_standing` — body remains physically standing

### Priority 3: Pitch (DEFERRED)

Recording only:
14. `pitch_reported` — pitch_x max/final/RMS recorded
15. `pitch_instability_classified` — classify as TASK_AWARE_PITCH_DEFERRED unless pitch causes instability

Pitch is **not** a primary fail gate in this phase.
Pitch will be converted to task-aware pitch control in a future phase.

### Priority 4: Support/Position Drift (DEFERRED)

Recording only:
16. `support_drift_reported` — support_position_error max/final/RMS recorded
17. `support_drift_classified` — classify as SUPPORT_DRIFT_DEFERRED unless drift causes contact/height/roll failure

Support drift is **not** a primary fail gate in this phase.
Support drift will be addressed after posture is stable.

## Posture Targets for This Phase

| Scenario | Divergence RMS Target |
|----------|----------------------|
| nominal | < 0.10 rad |
| low_0p300 | < 0.30 rad |
| high_0p480 | < 0.25 rad |

| Metric | Target |
|--------|--------|
| hip_yaw_abs_max | < 0.30 rad (preferred) |
| roll_y max_abs | bounded, no collapse |

## Fail Classifications

Allowed classifications if gate fails:

- `HY2_A0_INSUFFICIENT_DIVERGENCE_CONTROL` — divergence above target but robot survives
- `HY2_A0_CAUSES_CONTACT_OR_HEIGHT_FAILURE` — robot loses contact or collapses height
- `HY2_A0_CAUSES_ROLL_INSTABILITY` — roll becomes unstable
- `HY2_A0_SAFE_BUT_DIVERGENCE_REMAINS` — safe but divergence above target
- `HY2_A0_TIMEOUT_ONLY` — failure only due to timeout, not instability
- `POSTURE_REQUIRES_STRONGER_HY2_PROFILE` — A0 insufficient, need stronger profile
- `POSTURE_REQUIRES_MODE_BASED_HIP_YAW_CONTROLLER` — HY2-DIV alone insufficient

## Pass Classification

If all Priority 1 and Priority 2 gates pass:

```
POSTURE_STANDING_PASS_SUPPORT_DEFERRED
```

**Do NOT claim:**
- Official Step E pass
- Complete balance validation
- Ready for Step C or Step D

**May claim:**
- Posture-first standing validation pass
- Hip-yaw divergence controlled at all three heights
- Body remains physically standing
- Ready for support drift fix phase

## Required Telemetry

Per scenario (nominal, low_0p300, high_0p480):

### Survival/Contact/Height
- survived_5000 (bool)
- termination_reason
- contact_validity
- left/right_wheel_contact_percent
- nonwheel_floor_contact_count
- height_error max/final/RMS
- final_com_height
- target_com_height

### Hip-Yaw/Posture
- hip_yaw_abs_max max/final/RMS
- l_hip_yaw_error max/final/RMS
- r_hip_yaw_error max/final/RMS
- divergence max/final/RMS
- common_mode max/final/RMS
- hip_yaw_sign_correctness (left/right)
- hy2_div_enabled
- hy2_div_gate_active_percent
- hy2_div_gate mean/min/max
- hy2_div_effective_k/effective_kd
- hy2_div_torque max/final/RMS
- hy2_div_clipping_percent

### Roll
- roll_y max/final/RMS
- roll_collapse (bool)

### Pitch (DEFERRED)
- pitch_x max/final/RMS
- classification: TASK_AWARE_PITCH_DEFERRED unless instability

### Support (DEFERRED)
- support_position_error max/final/RMS
- classification: SUPPORT_DRIFT_DEFERRED unless contact/height/roll failure

### Structural Invariants
- wbc_diagnostic_vs_applied
- hidden_torque_max
- ownership_violation_max

## Scenario Configuration

### nominal
```
--controller-mode balance-core
--sagittal-controller velocity-damped
--vd-sagittal-authority-profile J3
HY2-DIV A0 profile (default height gate: z_low=0.50, z_high=0.65)
Steps: 5000
```

### low_0p300
```
--controller-mode balance-core
--sagittal-controller velocity-damped
--vd-sagittal-authority-profile J3
HY2-DIV A0 profile
--height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json
Steps: 5000
```

### high_0p480
```
--controller-mode balance-core
--sagittal-controller velocity-damped
--vd-sagittal-authority-profile J3
HY2-DIV A0 profile
--height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json
Steps: 5000
```

## Output Files

- `nominal_5000_telemetry.csv`
- `low_0p300_5000_telemetry.csv`
- `high_0p480_5000_telemetry.csv`
- `posture_standing_a0_5000_metrics.json`
- `posture_standing_a0_5000_summary.csv`
- `posture_standing_a0_5000_report.md`
- `posture_standing_a0_5000_pass_fail_summary.json`

## Explicit Statements

1. Support drift is **DEFERRED** — will be addressed after posture is stable
2. Pitch is **DEFERRED** to task-aware pitch control phase
3. This is NOT official Step E pass
4. This is NOT ready for Step C or Step D
5. Next phase should be support/position drift fix OR task-aware pitch control
