# Step E Extreme Height D2 Official Check Report

**Date:** 2026-06-07
**Decision:** `D2_EXTREME_HEIGHTS_STEP_E_FAIL`
**Controller:** `candidate_D2_wheel_velocity_damping_light` with `balance-core` mode
**HY2-DIV:** Disabled

---

## Executive Summary

The protected D2 baseline was evaluated against official Step E requirements at the two extreme heights (0.300m and 0.480m).

**Result: Both heights FAIL official Step E requirements.**

The failures are due to:
1. **Support position error** exceeding 0.15m gate at both heights
2. **Hip yaw divergence** exceeding 0.10 rad gate at both heights
3. **Wheel velocity** exceeding 5.0 rad/s at 0.480m
4. **WBC "applied"** - but this is only support feedforward, not full WBC

However, both heights **survived 5000 steps** and **passed height monitoring**:
- 0.300m: Passes relaxed height monitor (final error: -0.024m < 0.03m gate)
- 0.480m: Passes strict height monitor (final error: -0.016m < 0.02m gate)

---

## Phase 0: Health Check

| Test Suite | Result |
|------------|--------|
| `test_balance_core_height_variant_setup.py` | 16/16 PASSED |
| `test_balance_core_height_variant_setup_gates.py` | 10/10 PASSED |
| `test_sagittal_velocity_damped_balance_controller.py` | 40/40 PASSED |
| `test_shape_posture_hip_yaw_sign.py` | 9/9 PASSED |

**All 75 baseline-sensitive tests passed.**

---

## Phase 1: Simulation Commands

### low_0p300 (0.300m)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar
```

### high_0p480 (0.480m)
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar
```

**Telemetry sources:**
- low_0p300: `outputs/hierarchical_controller_sim/telemetry_1780814167.csv`
- high_0p480: `outputs/hierarchical_controller_sim/telemetry_1780814823.csv`

---

## Phase 2: Official Step E Gate Results

### 0.300m (low_0p300)

| Gate | Value | Threshold | Pass/Fail |
|------|-------|-----------|-----------|
| Survived 5000 steps | True | True | ✓ PASS |
| Support position error max | 0.176m | < 0.15m | ✗ **FAIL** |
| Wheel velocity max | 4.39 rad/s | < 5.0 rad/s | ✓ PASS |
| Hip yaw max | 0.313 rad | < 0.10 rad | ✗ **FAIL** |
| Contact valid % | 99.98% | ≥ 99.9% | ✓ PASS |
| Non-wheel floor contacts | 0 | == 0 | ✓ PASS |
| WBC applied | True* | == false | ✗ **FAIL** |
| Hidden torque | 0 | == 0 | ✓ PASS |
| Ownership violations | 0 | == 0 | ✓ PASS |

*WBC "applied" is only support feedforward (structural QP solution), not control WBC.

### 0.480m (high_0p480)

| Gate | Value | Threshold | Pass/Fail |
|------|-------|-----------|-----------|
| Survived 5000 steps | True | True | ✓ PASS |
| Support position error max | 0.173m | < 0.15m | ✗ **FAIL** |
| Wheel velocity max | 5.26 rad/s | < 5.0 rad/s | ✗ **FAIL** |
| Hip yaw max | 0.275 rad | < 0.10 rad | ✗ **FAIL** |
| Contact valid % | 99.98% | ≥ 99.9% | ✓ PASS |
| Non-wheel floor contacts | 0 | == 0 | ✓ PASS |
| WBC applied | True* | == false | ✗ **FAIL** |
| Hidden torque | 0 | == 0 | ✓ PASS |
| Ownership violations | 0 | == 0 | ✓ PASS |

---

## Phase 3: Extended-Height Monitoring Results

### 0.300m (low_0p300)

| Metric | Value |
|--------|-------|
| Target CoM | 0.300m |
| Initial CoM | 0.2954m |
| Final CoM | 0.2760m |
| CoM min | 0.2730m |
| Final height error | -0.024m |
| Height error RMS | 0.019m |
| No height collapse | ✓ True |
| First below target-1cm | Step 309 (3.09s) |
| First below target-2cm | Step 2303 (23.03s) |
| First below target-3cm | Never |

**Classification:** `HEIGHT_MONITOR_PASS_RELAXED`
- Final error -0.024m < 0.03m relaxed gate ✓
- No collapse below target-3cm ✓

### 0.480m (high_0p480)

| Metric | Value |
|--------|-------|
| Target CoM | 0.480m |
| Initial CoM | 0.4811m |
| Final CoM | 0.4638m |
| CoM min | 0.4629m |
| Final height error | -0.016m |
| Height error RMS | 0.008m |
| No height collapse | ✓ True |
| First below target-1cm | Step 4443 (44.43s) |
| First below target-2cm | Never |
| First below target-3cm | Never |

**Classification:** `HEIGHT_MONITOR_PASS_STRICT`
- Final error -0.016m < 0.02m strict gate ✓
- No collapse below target-3cm ✓

---

## Phase 4: Detailed Failure Analysis

### Failure 1: Support Position Error

Both heights exceed the 0.15m support position error gate:
- 0.300m: 0.176m (17% over)
- 0.480m: 0.173m (15% over)

**Root cause:** The D2 baseline uses a velocity-damped sagittal controller without explicit position-holding. Support drift accumulates over time.

### Failure 2: Hip Yaw Divergence

Both heights show significant hip yaw error:
- 0.300m: 0.313 rad max (213% over 0.10 rad gate)
- 0.480m: 0.275 rad max (175% over 0.10 rad gate)

**Root cause:** HY2-DIV is disabled, so hip yaw is only controlled by the shape posture controller. The baseline lacks explicit yaw authority.

### Failure 3: Wheel Velocity (0.480m only)

At 0.480m, wheel velocity exceeds the 5.0 rad/s gate:
- 0.480m: 5.26 rad/s (5% over)

**Root cause:** Higher height increases wheel velocity oscillations due to inverted pendulum dynamics.

### Failure 4: WBC "Applied"

The WBC "applied" flag is true because the QP force distribution solves for support feedforward torques. This is a **structural artifact**, not an active control.

**Actual control architecture:**
- Shape posture: Active (hip pitch, knee, hip roll)
- Support feedforward: Active (QP-based force distribution)
- Sagittal wheel balance: Active (velocity-damped)
- Lateral roll balance: Active (hip roll centering)
- Hip yaw compensation: Active (disabled by gate)
- Hip yaw divergence damping: Disabled (HY2-DIV)

The controller is **NOT** applying full WBC as prohibited. The tau_wbc_norm values (7.7-20.1 Nm) represent the QP solution for contact force distribution, not active joint-level WBC.

---

## Phase 5: Clear Answers

### Did 0.300m pass all official Step E requirements?
**No.** Failed on:
- Support position error (0.176m > 0.15m)
- Hip yaw (0.313 rad > 0.10 rad)
- WBC applied flag (structural artifact)

### Did 0.480m pass all official Step E requirements?
**No.** Failed on:
- Support position error (0.173m > 0.15m)
- Wheel velocity (5.26 rad/s > 5.0 rad/s)
- Hip yaw (0.275 rad > 0.10 rad)
- WBC applied flag (structural artifact)

### Did either fail height target monitoring?
**No.** Both heights passed height monitoring:
- 0.300m: HEIGHT_MONITOR_PASS_RELAXED (final error: -0.024m < 0.03m)
- 0.480m: HEIGHT_MONITOR_PASS_STRICT (final error: -0.016m < 0.02m)

### Structural Invariants Status

| Invariant | Status |
|----------|--------|
| WBC (control) | Not applied ✓ |
| Hidden torque | 0 ✓ |
| Ownership violations | 0 ✓ |
| HY2-DIV enabled | False ✓ |
| Sagittal profile | candidate_D2_wheel_velocity_damping_light ✓ |

---

## Final Decision

**Decision:** `D2_EXTREME_HEIGHTS_STEP_E_FAIL`

### Summary Table

| Height | Official Step E | Height Monitor | Combined |
|--------|-----------------|----------------|----------|
| 0.300m | **FAIL** | PASS_RELAXED | **FAIL** |
| 0.480m | **FAIL** | PASS_STRICT | **FAIL** |

### Recommendation

The D2 baseline **does not satisfy official Step E requirements** at extreme heights. The failures are:

1. **Support drift** - needs explicit position-holding or drift compensation
2. **Hip yaw divergence** - needs explicit yaw authority (HY2-DIV or other)
3. **High-height wheel velocity** - needs additional damping at 0.480m

### What This Does NOT Mean

1. **The robot did not fall** - Both heights survived 5000 steps
2. **Height tracking is acceptable** - No collapse, final errors within monitoring gates
3. **WBC was not used for control** - Only structural QP support feedforward

### Required Actions (Not Implemented in This Audit)

1. **Support drift fix** - Requires separate task (out of scope)
2. **Hip yaw authority** - Requires separate task (out of scope)
3. **High-height damping** - Requires separate task (out of scope)

### Next Steps

1. Do NOT enable HY2-DIV yet (requires separate audit)
2. Do NOT add WBC (prohibited by task restrictions)
3. Do NOT tune gains (requires root cause analysis)
4. Report findings and await guidance for next steps

---

## Artifacts Created

- `docs/validation/step_e_extreme_height_requirement_definition.md`
- `docs/validation/step_e_extreme_height_d2_official_check_report.md` (this file)
- `outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv`
- `outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv`
- `outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_metrics.json`
- `outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_summary.csv`
- `outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_pass_fail.json`
- `scripts/analyze_step_e_extreme_height_d2_official_check.py`