# E1_support_integral Smoke Analysis — low_0p300

**Date:** 2026-06-07  
**Telemetry:** `outputs/hierarchical_controller_sim/telemetry_1780825597.csv`  
**Profile:** `E1_support_integral` (confirmed from `sagittal_schedule_profile` column)  
**Variant:** `low_0p300` (target_com_z = 0.2955 m)  
**Duration:** 100 steps (0.99 s)

---

## Decision

**E1_LOW_0P300_SMOKE_FAIL_DO_NOT_CONTINUE**

The E1_support_integral smoke run produced data that is **floating-point identical** to the D2 baseline first 100 rows. The integral term accumulated a maximum of 9.5e-06 Nm — effectively zero. The smoke test is inconclusive and should not proceed to 500-step validation.

---

## Profile Identity

| Field | Value |
|---|---|
| sagittal_schedule_profile | E1_support_integral ✓ |
| variant_name | low_0p300 ✓ |
| target_com_z_m | 0.2955 |
| control_mode | upright |
| sagittal_schedule_height_source | target_reference |
| low_height_sagittal_schedule_active | False |
| high_height_schedule_active | True |
| effective_k_position | 40.0 |
| effective_k_velocity | 15.0 |
| effective_max_position_tau | 4.0 |

Profile name confirmed from telemetry. However, the integral component produced zero effect (see below).

---

## Smoke Run Results

| Metric | E1 Value | D2 First 100 | Diff |
|---|---|---|---|
| support_position_error_m max | 0.1602 m | 0.1602 m | +0.0000 |
| support_position_error_m RMS | 0.0877 m | 0.0877 m | +0.0000 |
| support_position_error_m final | 0.1602 m | 0.1602 m | +0.0000 |
| hip_yaw_abs_max max | 0.0063 rad | 0.0063 rad | -0.0000 |
| hip_yaw_abs_max RMS | 0.0043 rad | 0.0043 rad | +0.0000 |
| wheel_vel_mean_rad_s max | 1.1276 rad/s | 1.1276 rad/s | +0.0000 |
| wheel_vel_mean_rad_s RMS | 3.0880 rad/s | 3.0880 rad/s | +0.0000 |
| height_error_m max | 0.0000 m | 0.0000 m | +0.0000 |
| height_error_m RMS | 0.0017 m | 0.0017 m | +0.0000 |
| pitch_x max | 0.1111 rad | 0.1111 rad | +0.0000 |
| roll_y max | 0.0044 rad | 0.0044 rad | -0.0000 |
| hidden_torque_norm max | 0.0000 Nm | 0.0000 Nm | +0.0000 |
| ownership_violation_count max | 0 | 0 | +0.0000 |
| non_wheel_floor_contacts max | 0 | 0 | +0.0000 |

All differences are < 1e-3, consistent with floating-point noise.

### Run integrity

- Survived requested steps: ✅ Yes (100/100)
- Terminated: ❌ No
- WBC feasible: ✅ Always
- Hidden torque: ✅ 0.0
- Ownership violations: ✅ 0

---

## E1 Integral Diagnostics

| Field | Value |
|---|---|
| integral_active | 94 steps False, **6 steps True** |
| tau_position_integral max | **9.5e-06 Nm** (effectively 0) |
| tau_position_integral final | 0.0 Nm |
| tau_position_integral RMS | 2.0e-06 Nm |
| tau_position_i max | 0.0 Nm (effectively 0) |
| position_integral_error max | 5e-06 m (effectively 0) |
| tau_position_raw max | 0.0190 Nm |
| tau_position final | -4.0 Nm (clipped) |

### Integral gate reasons

| Gate Reason | Count | % |
|---|---|---|
| **pitch_error_large** | **60** | **60%** |
| pitch_rate_large | 29 | 29% |
| safe_steady_state | 6 | 6% |
| support_velocity_large | 4 | 4% |
| contact_invalid | 1 | 1% |

### Integral activation moments (steps 3–9 only)

During the initial transient, the integral briefly activated 6 times with values on the order of 1e-06 Nm — **8 orders of magnitude smaller than the position P term (tau_position_p max = 0.019 Nm)**. After step 9, the `pitch_error_large` gate permanently blocks the integral for the remaining 91 steps.

---

## D2 Baseline Reference (5000-row full run)

| Metric | D2 Value |
|---|---|
| support_position_error_m max | 0.1757 m (step 91 crosses 0.15 m, 96 violations total) |
| hip_yaw_abs_max max | **0.3127 rad** (step 328 crosses 0.10 rad, 4419 violations total) |
| hip_yaw_abs_max final | 0.2812 rad |
| wheel_vel_mean_rad_s final | -2.4301 rad/s |
| height_error_m final | -0.0195 m |
| Terminated | No |

The D2 full run fails two Step E gates:
1. `support_position_error > 0.15 m` — first at step 91, 96 total violations
2. `hip_yaw_abs_max > 0.10 rad` — first at step 328, 4419 violations

E1 smoke exactly matches D2 first 100 rows for both metrics.

---

## Root Cause Analysis

### Why E1 produced no effect

**The integral gate uses `pitch_error` as the gating signal, but `pitch_error` IS the symptom of support drift.** This creates a self-defeating loop:

1. As support drifts, pitch error accumulates (pitch_x rises from 0.03 → 0.11 rad)
2. When |pitch_error| exceeds the gate threshold (~0.03 rad), the integral is **gated OFF**
3. The integral can only activate during the initial transient (steps 3–9) when pitch ≈ 0
4. At that point, position error is negligible (support_position_error < 0.001 m)
5. The integral therefore accumulates nothing meaningful

### Design flaw

```
pitch_error rises → integral gate activates → integral disabled → support drifts → pitch_error rises → ...
```

The integral gate must NOT gate on the same variable it is meant to correct. When the goal is to reduce support drift, gating the integral based on pitch error defeats the purpose.

### Additional concern

The E1 telemetry is floating-point identical to D2 first 100 rows for all 186 numeric columns that are identical. This raises the question of whether the E1 controller code was actually executed, or whether the simulation used D2 code with only the profile name changed.

---

## Classification

- **E1_SMOKE_NO_EFFECT**: ✅ Yes
- **E1_SMOKE_IMPROVES_SUPPORT**: ❌ No
- **E1_SMOKE_REGRESSES_HIP_YAW**: ❌ No (identical)
- **E1_SMOKE_REGRESSES_CONTACT_HEIGHT**: ❌ No (identical)
- **E1_SMOKE_TELEMETRY_INCONCLUSIVE**: ❌ Not inconclusive — conclusively no effect

---

## Stage 2 Decision

### ❌ FAIL — Do Not Proceed

**Pass criteria not met:**
- ✅ Survived smoke
- ✅ Contact valid
- ✅ Non-wheel contacts = 0
- ✅ WBC control gate pass
- ✅ Hidden torque = 0
- ✅ Ownership violations = 0
- ❌ **support_position_error NOT improved (identical to D2)**
- ❌ **hip_yaw NOT improved (identical to D2)**
- ❌ **Integral never activated meaningfully (tau_position_integral max = 9.5e-06)**

### Required actions before re-run

1. **Code audit**: Verify that the `E1_support_integral` code path actually modifies `tau_position` at runtime. The telemetry suggests it may not be wired in.
2. **Fix integral gate**: The `pitch_error_large` gate threshold must be raised significantly, or the gating condition changed to something that does not depend on pitch (e.g., time-based gate, or gate based on a separate error signal).
3. **Verify telemetry reflects E1 code**: Run the E1 code with a known perturbation and confirm telemetry changes relative to D2.

### Options for integral gate fix

| Option | Description | Risk |
|---|---|---|
| Raise pitch_error gate threshold | Increase from ~0.03 to ~0.10+ rad | May cause integral windup during large pitch excursions |
| Gate on time, not error | Activate integral after T seconds regardless of pitch | Simple but may not address the root cause |
| Use a separate error signal for integral | Integrate support_position_error directly, not pitch error | Cleanest but requires code change |
| Remove pitch_error gating entirely | Always allow integral to accumulate | Risk of integral windup but would at least test the concept |

### Next command after fix

```
python scripts/simulate_hierarchical_controller.py \
  --variant low_0p300 \
  --sagittal-profile E1_support_integral \
  --num-steps 100 \
  --output outputs/step_e_extreme_support_fix_eval/e1_low_0p300_smoke_v2
```

---

## Files Analyzed

- `outputs/hierarchical_controller_sim/telemetry_1780825597.csv` — E1 smoke telemetry
- `outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv` — D2 baseline