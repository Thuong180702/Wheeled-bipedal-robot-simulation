# E2b Support Integral Cap with Aligned Gate - 500-Step Report

**Date:** 2026-06-08
**Profile:** E2b_support_integral_higher_cap_aligned_gate
**Test:** low_0p300 at 500 steps

## Executive Summary

**Classification:** `E2B_500_NO_SUPPORT_IMPROVEMENT`

E2b (0.12 rad integral gate + 5.0 Nm cap) produces virtually identical results to E2 (0.03 rad gate + 5.0 Nm cap). Both improve support but regress hip_yaw compared to D2. The integral gate alignment does NOT fix the hip_yaw regression. This suggests the 5.0 Nm cap itself is the culprit, not the 0.03 rad threshold.

---

## Files Changed

1. `scripts/simulate_hierarchical_controller.py` - Added E2b profile definition
2. `tests/test_sagittal_velocity_damped_balance_controller.py` - Added E2b tests

## E2b Profile Definition

```python
"E2b_support_integral_higher_cap_aligned_gate": SagittalAuthoritySchedule(
    profile_name="E2b_support_integral_higher_cap_aligned_gate",
    applies_to_variants=BOUNDARY_HEIGHT_VARIANTS,
    # Enable position integral (same as E1/E2)
    enable_position_integral=True,
    ki_position_integral=2.0,
    integral_max_abs=1.0,
    integral_pitch_error_threshold_rad=0.12,  # KEY CHANGE: align to E1 value
    integral_support_velocity_threshold_m_s=0.03,
    integral_wheel_velocity_threshold_rad_s=1.0,
    integral_min_com_z_m=0.28,
    integral_max_com_z_m=0.50,
    # Keep position cap at E2 level (5.0 Nm)
    continuous_max_position_tau=True,
    max_position_tau_nominal=4.0,
    max_position_tau_low_max=5.0,
    # Keep velocity damping at D2 level
    velocity_damping_scale=1.10,
)
```

## Exact Command

```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile E2b_support_integral_higher_cap_aligned_gate \
  --height-variant-setup outputs/physical_target_height_setups/low_0p300_setup.json \
  --steps 500 \
  --telemetry-decimation 1 \
  --failure-window-steps 500 \
  --write-run-summary-sidecar
```

## Telemetry Path

```
outputs/step_e_extreme_support_fix_eval/e2b_low_0p300_500/e2b_low_0p300_500_telemetry.csv
```

---

## Comparison Table

| Metric | D2 | E1_before | E1_after | E2 | E2b |
|--------|------|-----------|----------|------|------|
| **Support Max (m)** | 0.1757 | 0.1757 | 0.1757 | 0.1703 | 0.1703 |
| **Support >0.15m** | 96 | 96 | 96 | 62 | 62 |
| **Support Mean (m)** | 0.0827 | 0.0827 | 0.0827 | 0.0677 | 0.0677 |
| **Support Final (m)** | 0.0580 | 0.0580 | 0.0579 | 0.0276 | 0.0276 |
| **Hip Yaw Max (rad)** | 0.1018 | 0.1018 | 0.1018 | 0.1304 | 0.1305 |
| **Hip Yaw Max (deg)** | 5.83° | 5.83° | 5.83° | 7.47° | 7.48° |
| **Hip Yaw >0.10rad** | 26 | 26 | 26 | 53 | 53 |
| **Divergence Max** | 0.1866 | 0.1866 | 0.1866 | 0.2434 | 0.2435 |
| **Tau Pos Raw Max (Nm)** | 7.0275 | 7.0275 | 7.0275 | 6.8111 | 6.8111 |
| **Integral Active %** | 0.0% | 4.4% | 7.8% | 6.2% | 9.0% |
| **Wheel Vel RMS (rad/s)** | 2.8207 | 2.8207 | 2.8214 | 3.2499 | 3.2509 |

---

## Key Observations

### Support Position Error
- E2b support metrics are **identical** to E2 (within numerical precision)
- Both E2 and E2b improve vs D2:
  - Max: 0.1757 → 0.1703 m (-3.1%)
  - Mean: 0.0827 → 0.0677 m (-18.1%)
  - Final: 0.0580 → 0.0276 m (-52.4%)
  - Crossings >0.15m: 96 → 62 (-35.4%)

### Hip Yaw
- E2b hip_yaw metrics are **identical** to E2 (within numerical precision)
- Both E2 and E2b **regress** vs D2:
  - Max: 0.1018 → 0.1304 rad (+28.1%)
  - Count >0.10 rad: 26 → 53 (+104%)
  - Divergence max: 0.1866 → 0.2434 (+30.4%)

### Integral Behavior
- E2b integral active: 9.0% vs E2: 6.2%
- The wider gate (0.12 rad) allows more integral accumulation, but this does NOT translate to improved hip_yaw
- The integral is not the root cause

### Tau Position
- E2b tau_position_raw_max: 6.8111 Nm (same as E2, lower than D2's 7.0275 Nm)
- The 5.0 Nm cap limits raw torque to similar levels regardless of gate threshold

---

## Hypothesis Evaluation

**Original Hypothesis:** E2's 0.03 rad threshold was too restrictive, causing tau_position to accumulate aggressively which drove hip_yaw divergence. By widening to 0.12 rad (E1's value), the integral would accumulate more naturally without windup-driven torque spikes.

**Result:** REJECTED. The gate alignment does NOT fix hip_yaw regression. The wider gate (E2b) allows more integral accumulation (9.0% vs 6.2%) but produces nearly identical hip_yaw behavior.

**New Hypothesis:** The 5.0 Nm cap itself (present in both E2 and E2b, absent in D2/E1) is the root cause of hip_yaw regression. The cap changes the torque distribution in a way that couples to hip_yaw through kinematic coupling.

---

## E2b vs E2 Delta (Gate Effect)

| Metric | E2 | E2b | Delta |
|--------|------|------|-------|
| Integral Active % | 6.2% | 9.0% | +2.8% |
| Hip Yaw Max | 0.1304 | 0.1305 | +0.0001 |
| Support >0.15m | 62 | 62 | 0 |

The only meaningful difference is integral activation rate. Hip_yaw is unchanged.

---

## Conclusion

**E2b confirms the integral gate is NOT the root cause of hip_yaw regression.**

The next candidate (E2c) should test whether the 5.0 Nm cap itself is the issue:
- max_position_tau_low_max = 4.5 Nm
- integral_pitch_error_threshold_rad = 0.12
- Hypothesis: Lowering the cap to 4.5 Nm while keeping the aligned gate may preserve some support improvement while reducing hip_yaw regression

---

## Decision

**Classification:** `E2B_500_NO_SUPPORT_IMPROVEMENT`

**Reasoning:**
- E2b does NOT improve hip_yaw vs E2 (0.1305 vs 0.1304 rad)
- E2b preserves support improvement (same as E2)
- The integral gate change has no effect on hip_yaw
- The 5.0 Nm cap is the common factor between E2 and E2b's hip_yaw regression

**Recommended Next Step:** E2c with max_position_tau_low_max = 4.5 Nm and integral_pitch_error_threshold_rad = 0.12

**Do NOT:**
- Run 2000-step validation (would show same regression)
- Run Step C or Step D (not ready)
- Commit E2b (it doesn't fix the problem)