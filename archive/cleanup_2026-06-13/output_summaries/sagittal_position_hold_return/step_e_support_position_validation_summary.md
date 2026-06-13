# Step E Support-Position Validation Summary

**Date:** 2026-05-30
**Status:** GATES NOT MET — STATE SEPARATION CORRECT — GAIN RECALIBRATION REQUIRED
**Do NOT proceed to Step C until reviewed.**

---

## What Was Done

Three sub-steps of Step E are now complete:

1. **Step E coupling fix:** `kp_cp=0.0` — eliminated destructive cancellation.
2. **Step E position-return migration:** `k_position=40.0` — restored effective return coefficient.
3. **Step E state separation:** `sagittal_position_error_m` now tracks wheel support center, not COM. Pitch error now uses `pitch_x - pitch_ref_x`.

---

## Root Cause of This Sub-Step

`sagittal_position_error_m` was using COM position as the standing-position error. For a wheeled biped (TWIP), the COM is *allowed* to move relative to the support center during pitch balance. When the robot pitches forward, COM moves forward relative to wheels — this was counted as position drift, causing `tau_position` to fight `tau_pitch`.

**Correct design:**
- Position hold tracks the **wheel support center** (midpoint of wheel body positions)
- COM position error is logged separately as a **balance diagnostic only**
- Pitch error uses `pitch_x - pitch_ref_x` (relative to equilibrium)

---

## Results Summary

| Metric | Step E migration (COM-based) | State separation (support-based) | Target |
|--------|------------------------------|----------------------------------|--------|
| Position error source | COM | Wheel support center | Wheel support center |
| Max position error | 0.254 m | 0.514 m | ≤0.10 m |
| Final position error | 0.028 m | 0.367 m | ≤0.05 m |
| tau_position max | 10.2 Nm | 20.6 Nm | ≤5.0 Nm |
| tau_position saturation | 2.0x | 4.1x | ≤1.0x |
| Completed 5000 steps | Yes | No (fell at 1158) | Yes |
| tau_cp | zero | zero | zero |
| Ownership violations | 0 | 0 | 0 |
| ±0.10 m gate | FAIL | FAIL | PASS |
| ±0.15 m gate | FAIL | FAIL | PASS |

---

## Why State Separation Is Correct But Gates Still Not Met

The state separation is architecturally correct. The failure is a **gain calibration mismatch**:

- k_position=40.0 was calibrated for COM-based error (max amplitude ~0.254 m)
- Support-center error has larger amplitude (~0.514 m) because wheels move to balance the robot
- At the same position error: `tau_position = -40.0 × 0.514 = -20.6 Nm` — 4.1× the `max_tau_wheel=5.0` limit
- Wheel torque saturates → pitch balance fails → robot falls

The pitch reference fix is working correctly: pitch stays near 0 deg (range -0.48 to +0.59 deg) in the 5000-step run before falling.

**Failure classification:** `position_return_gain_too_large_for_support_position_error`

---

## All Configurations Tested (Complete History)

| Configuration | Position source | Max error | Final error | Completed 5000 | Gate |
|---------------|----------------|-----------|-------------|----------------|------|
| Old F4c (kp_cp=30, k_pos=0) | COM | 3.876 m | ~3.8 m | No | FAIL |
| Bug-fixed F4c (kp_cp=30, k_pos=10) | COM | 0.254 m | 0.028 m | Yes | FAIL |
| Step E fix (kp_cp=0, k_pos=10) | COM | 0.606 m | 0.105 m | Yes | FAIL |
| Migration (kp_cp=0, k_pos=40) | COM | 0.254 m | 0.028 m | Yes | FAIL |
| **State separation (kp_cp=0, k_pos=40)** | **Support** | **0.514 m** | **0.367 m** | **No (1158)** | **FAIL** |

---

## Tests Run

| Test suite | Result |
|------------|--------|
| `pytest tests/test_sagittal_balance_state.py -v` | 17 passed |
| `pytest tests/test_sagittal_velocity_damped_balance_controller.py tests/test_sagittal_balance_state.py tests/test_balance_core_components.py tests/test_balance_core_validation_workflow.py -q` | 91 passed |

New tests added to `tests/test_sagittal_balance_state.py`:
- `test_support_center_is_midpoint_of_wheel_bodies`
- `test_support_center_symmetric_wheels`
- `test_support_position_error_zero_when_wheels_at_equilibrium`
- `test_support_position_error_positive_when_wheels_move_forward`
- `test_support_position_error_negative_when_wheels_move_backward`
- `test_com_pitch_motion_does_not_affect_support_position_error` ← key invariant
- `test_support_position_error_unaffected_by_yaw`
- `test_pitch_error_zero_at_equilibrium`
- `test_pitch_error_positive_when_pitched_forward`
- `test_pitch_error_negative_when_pitched_backward`

---

## Verification

| Check | Status |
|-------|--------|
| No WBC changes | CONFIRMED |
| No E0b/E0c/E0d reintroduced | CONFIRMED |
| Torque ownership unchanged | CONFIRMED |
| Sagittal controllers mutually exclusive | CONFIRMED |
| balance-core mode only | CONFIRMED |
| velocity-damped controller only | CONFIRMED |
| tau_cp = 0.0 throughout | CONFIRMED |
| Support-center used for position hold | CONFIRMED |
| COM error logged separately (diagnostic) | CONFIRMED |
| pitch_x_ref captured at equilibrium | CONFIRMED |
| pitch_x_error = pitch_x - pitch_ref | CONFIRMED |
| ownership_violation_count = 0 | CONFIRMED |
| hidden_torque_norm = 0.0 | CONFIRMED |

---

## Proposed Next Step

Recalibrate k_position for support-center error amplitude:

```
k_position_new = k_position_old × (com_error_amplitude / support_error_amplitude)
              ≈ 40.0 × (0.254 / 0.514) ≈ 20.0 Nm/m
```

Steady-state equilibrium formula still holds:
```
sag_support_err_ss = (kp_pitch / k_position) × pitch_ss
```

At k_position=20.0: `sag_support_err_ss = (50/20) × 0.021 = 0.053 m`

This is slightly larger than the 0.026 m from the COM-based config, but within the ±0.10 m preferred gate. Verify before running long simulations.

---

## Do NOT

- Revert to COM-based position error (architecturally wrong for TWIP)
- Increase k_position (already too large for support-center error)
- Reintroduce kp_cp (restores destructive cancellation)
- Proceed to Step C without gain recalibration
- Blindly tune k_position without verifying steady-state formula
