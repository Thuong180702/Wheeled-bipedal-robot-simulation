# Step E One-Sided Position Error Root Cause Report

## Executive Summary

The one-sided positive support_position_error_m is caused by **insufficient position authority during transients**. The default `max_position_tau = 3.0 Nm` is too low to overcome pitch balance torque (~4.6 Nm mean during transients), causing the robot to drift forward while recovering from pitch disturbances.

## Observed Behavior

With baseline configuration (3.0 Nm position authority):
- `support_position_error_m` range: [-0.038, +0.543] m
- Final error: -0.006 m
- Error is strongly one-sided positive (81.4% of steps have positive error)
- Peak error occurs at step 1520 (15.2 seconds)

## Root Cause Analysis

### Phase 1: Verify Active Configuration

Confirmed from telemetry:
- controller_mode = balance-core ✓
- sagittal_controller = velocity-damped ✓
- WBC torque not applied to wheels ✓
- kp_cp = 0.0 (disabled) ✓
- Support-center position error used as metric ✓

### Phase 2: Transient Phase Analysis (steps 1400-1600)

During the transient phase when error peaks:

| Metric | Value |
|--------|-------|
| support_position_error_m | mean=0.442, max=0.543 |
| tau_position | -3.0 Nm (100% saturated at cap) |
| tau_position_raw | mean=-17.7 Nm (controller wants much more) |
| sagittal_term_pitch | mean=+4.6 Nm |
| pitch_x_rad | mean=0.092 rad (5.3 deg) |

**Key Finding**: The controller is trying to apply -17.7 Nm of position correction but is clipped to -3.0 Nm. Meanwhile, pitch balance is applying +4.6 Nm. The net effect is positive torque, causing continued forward drift.

### Phase 3: Sign Analysis

The sign convention is correct:
- Positive position error → negative tau_position (correct)
- Positive pitch → positive tau_pitch (correct, to catch forward fall)

The conflict is physical, not a sign error: when the robot pitches forward, it needs to move forward to catch itself (tau_pitch > 0), but position hold wants to push it back (tau_position < 0).

### Phase 4: Authority Comparison

| Configuration | Transient Max | Final Error | Status |
|--------------|---------------|-------------|--------|
| 3.0 Nm (baseline) | +0.543 m | -0.006 m | FAIL |
| 4.0 Nm | +0.475 m | -0.017 m | FAIL |
| 5.0 Nm | +0.209 m | +0.186 m | PASS (hard min) |
| 5.0 Nm + k_sv=20 | +0.326 m | +0.026 m | FAIL |

### Phase 5: Steady-State Analysis

| Config | Pitch (deg) | tau_position | Wheel Vel (rad/s) |
|--------|-------------|--------------|-------------------|
| 3.0 Nm | 1.35 | -1.47 | +0.07 |
| 4.0 Nm | 0.71 | -0.61 | -0.02 |
| 5.0 Nm | 2.22 | -1.94 | -0.34 |

The 5.0 Nm case settles at a different equilibrium with higher pitch and negative wheel velocity.

## Root Cause Classification

**Primary Root Cause**: Position authority saturation during transients

The 3.0 Nm position authority cap is insufficient to overcome pitch balance torque during recovery. The controller correctly computes the needed correction (-17.7 Nm) but is clipped to -3.0 Nm, while pitch balance applies +4.6 Nm. The net positive torque causes forward drift.

**Secondary Contributor**: Pitch-position coupling

When the robot pitches forward, pitch balance and position hold have opposing goals. This is a fundamental physical constraint, not a bug. The solution is to provide sufficient position authority to eventually overcome pitch balance after the transient settles.

## Why Error is One-Sided

1. Initial disturbance causes forward pitch
2. Pitch balance applies positive torque to catch the fall
3. Position hold tries to apply negative torque but is saturated
4. Net positive torque → robot drifts forward
5. Forward drift increases position error
6. Position hold remains saturated, cannot catch up
7. Eventually pitch settles, position hold can recover
8. But by then, significant forward drift has accumulated

The error is one-sided because:
- Forward pitch → forward drift (positive error)
- Backward pitch → backward drift (negative error)
- But forward pitch is more common/larger in this system
- And position authority is insufficient to prevent drift during forward pitch recovery

## Recommended Fix

Increase `max_position_tau` from 3.0 Nm to 5.0 Nm.

**Rationale**:
- 5.0 Nm provides sufficient authority to limit transient peak to 0.209 m
- This passes the hard minimum requirement (max_abs <= 0.30 m)
- The worse final offset (0.186 m) is a tradeoff but acceptable for hard minimum

**Alternative**: Use 4.0 Nm for better final offset (-0.017 m) but worse transient (0.475 m).

## Validation Requirements

After fix:
1. Run 5000-step nominal simulation
2. Verify max_abs(support_position_error_m) <= 0.30 m (hard minimum)
3. Verify no posture/height/contact regression
4. Run height variants (high_5cm, low_5cm) if hard minimum passes
