# Step E Position Regulator Root Cause Report

**Date:** 2026-05-31  
**Run:** telemetry_1780208317.csv (2000 steps, 20.0 seconds)  
**Mode:** balance-core (WBC disabled, verified)  
**Controller:** SagittalVelocityDampedBalanceController with smart position-hold capture gate

---

## Executive Summary

**Root Cause Identified:** The 0.595m forward drift is caused by **insufficient position control authority** (max_position_tau = 3.0 Nm) combined with **missing support-center velocity regulation**.

**Capture Gate Status:** The smart position-hold capture gate never activated because the capture point stayed within ±0.064m (below the 0.10m threshold). The gate logic is correct but irrelevant to this failure mode.

**Classification:** Primary cause **C** (position_gain_or_authority_insufficient) with secondary contributing factor **D** (missing_position_velocity_regulation).

---

## Task 1: Why Capture Gate Never Activated

### Capture Gate Logic Review

From `position_hold_capture_gate.py` lines 121-126:

```python
if capture_point_relative_to_support_m > 0.10:  # 10cm threshold
    required_capture_direction = 1.0  # forward
elif capture_point_relative_to_support_m < -0.10:
    required_capture_direction = -1.0  # backward
else:
    required_capture_direction = 0.0  # no capture needed
```

Gate activates only when:
1. `required_capture_direction != 0.0` (capture point exceeds ±0.10m)
2. `position_opposes_capture = True` (tau_position opposes capture direction)

### Telemetry Evidence

**Capture point behavior:**
- Min: -0.0330 m
- Max: +0.0640 m
- **Never exceeded ±0.10m threshold**

**Gate state throughout run:**
- `capture_gate_enabled`: True (all steps)
- `capture_gate_active`: False (all 2000 steps)
- `capture_gate_reason`: "capture_recovery" (1492 steps), "pitch_reversal" (408 steps), "warmup" (100 steps)
- `capture_gate_required_direction`: 0.0 (all steps)
- `capture_gate_position_opposes_capture`: False (all steps)
- `capture_gate_factor`: 1.0 (all steps, no gating applied)

**Around peak error (step 1360):**
- `support_position_error_m`: 0.595 m (forward drift)
- `capture_gate_cp_relative_to_support_m`: -0.0138 m (well within threshold)
- `capture_gate_required_direction`: 0.0 (no capture needed)
- `tau_position_raw`: -11.900 Nm (wanted to return backward)
- `tau_position_clipped`: -3.000 Nm (saturated at limit)

### Answer: Why Gate Never Activated

**The capture gate never activated because the capture point stayed within ±0.064m, below the 0.10m activation threshold.**

This reveals a fundamental mismatch:
- **Support-center position** drifted 0.595m forward
- **Capture point relative to support** stayed within ±0.064m
- **Capture point tracks CoM + velocity/omega**, not support-center position

The robot's support center (wheel contact) drifted forward while the CoM remained relatively centered over the support. This is a **support-center drift problem**, not a capture-point problem.

**Conclusion:** The capture gate is **not relevant** to this failure mode. It was designed to handle capture conflicts during transient pitch recovery, not steady-state support-center drift.

---

## Task 2: Root Cause Classification

**Primary Cause: C. position_gain_or_authority_insufficient**

### Evidence

1. **Position authority saturation:**
   - `tau_position_raw` wanted -11.900 Nm at peak error
   - `max_position_tau` limit: 3.0 Nm
   - Clipped to -3.000 Nm (74% reduction in desired torque)
   - Saturated for 558/2000 steps (27.9% of run)

2. **Insufficient gain-authority product:**
   - `k_position = 20.0 N/m`
   - At 0.595m error: desired torque = 20.0 × 0.595 = 11.9 Nm
   - Available torque: 3.0 Nm
   - **Effective gain at saturation: 3.0 / 0.595 = 5.04 N/m** (75% weaker than nominal)

3. **Saturation during growth phase:**
   - Steps 200-1400: position error grew from 0.06m to 0.595m
   - tau_position saturated for 312/1201 steps (26.0%) during growth
   - Controller could not generate sufficient restoring force

4. **Final error is acceptable:**
   - Final error: 0.039m (within 0.05m gate)
   - Final tau_position: not saturated
   - **This proves the controller CAN regulate position when error is small**
   - **But CANNOT prevent large transient excursions**

**Secondary Contributing Factor: D. missing_position_velocity_regulation**

### Evidence

1. **Support position velocity analysis:**
   - Peak velocity: +0.397 m/s at step 1289
   - Growth phase mean velocity: +0.0364 m/s (persistent forward drift)
   - Rapid growth phase (steps 1310-1360): velocity 0.0045 → 0.3522 m/s

2. **No explicit support velocity damping:**
   - Current controller has `k_velocity` term for CoM sagittal velocity
   - But CoM velocity ≠ support-center velocity
   - Support-center velocity is the rate of change of `support_position_error_m`
   - **No term directly opposes support-center velocity**

3. **Velocity allowed error to grow:**
   - Position error grew because support velocity was not directly damped
   - By the time position error became large enough to saturate tau_position, velocity was already high
   - High velocity + saturated position torque = continued drift

### Why Not Other Causes?

**A. capture_gate_condition_wrong:** Gate logic is correct. Capture point simply never exceeded threshold because this is a support-drift problem, not a capture problem.

**B. capture_point_diagnostic_not_relevant:** Correct assessment. Gate is irrelevant to this failure.

**E. missing_integral_or_bias_rejection:** Final error is small (0.039m), not persistent offset. No evidence of steady bias.

**F. torque_rate_or_saturation_limit:** Saturation is the symptom, not the root cause. The issue is insufficient authority, not rate limits.

**G. wheel_velocity_damping_conflict:** `k_wheel_velocity = 0.5` is weak. Not blocking position correction.

**H. frame_or_metric_error:** `support_position_error_m` correctly measures support-center drift in initial-heading frame.

---

## Task 3: Current Control Law Review

### From `sagittal_velocity_damped_balance_controller.py`

**Control law (lines 124-161):**

```python
tau_pitch = kp_pitch * pitch_x_rad                          # Line 124
tau_pitch_rate = kd_pitch * pitch_rate_x_rad_s              # Line 125
tau_sagittal_velocity = -k_velocity * sagittal_velocity_m_s # Line 126
tau_position_raw = -k_position * sagittal_position_error_m  # Line 129
tau_position = clip(tau_position_raw, -max_position_tau, max_position_tau)  # Line 148
tau_cp = -kp_cp * sagittal_position_error_m                 # Line 155
tau_com_vy = -kd_com_vy * sagittal_velocity_m_s             # Line 156
tau_common = wheel_torque_sign * (tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_position + tau_cp + tau_com_vy)
tau_left = tau_common + tau_wheel_vel_left
tau_right = tau_common + tau_wheel_vel_right
```

**Current gains (from telemetry/command):**
- `kp_pitch = 50.0` (default)
- `kd_pitch = 10.0` (default)
- `kp_cp = 0.0` (disabled)
- `kd_com_vy = 5.0` (default)
- `k_velocity = 15.0` (from --vd-k-velocity 15.0)
- `k_wheel_velocity = 0.5` (default)
- `k_position = 20.0` (from --vd-k-position 20.0)
- `max_position_tau = 3.0` (from --vd-max-position-tau 3.0)
- `max_tau_wheel = 5.0` (default)

### Why Final Error Is Small But Transient Error Is Large

**Final error (0.039m) is small because:**
1. Position error is below saturation threshold (3.0 / 20.0 = 0.15m)
2. tau_position = -20.0 × 0.039 = -0.78 Nm (not saturated)
3. Support velocity is near zero (steady state)
4. Controller can regulate small errors effectively

**Transient error (0.595m) is large because:**
1. Position error exceeded saturation threshold by 4×
2. tau_position saturated at -3.0 Nm (wanted -11.9 Nm)
3. Support velocity reached +0.397 m/s (no direct damping)
4. Insufficient restoring force to prevent drift during transient

**The controller is a good steady-state regulator but a poor transient limiter.**

---

## Task 4: Recommended Fix Family

### Selected Fix: **Fix B - Add Explicit Support-Position Velocity Regulation**

**Rationale:**

1. **Addresses root cause:** Directly damps support-center velocity, preventing error growth
2. **Preserves position authority:** Does not require increasing max_position_tau (which could destabilize pitch)
3. **Physics-based:** Support velocity damping is a standard control technique for position regulation
4. **Testable:** Clear input (support velocity) and output (damping torque) relationship

**Proposed control law addition:**

```python
# Compute support position velocity (rate of change of support_position_error_m)
support_position_velocity_m_s = (current_support_position_error_m - prev_support_position_error_m) / dt

# Add support velocity damping term
tau_support_velocity = -k_support_velocity * support_position_velocity_m_s

# Include in total torque
tau_common = wheel_torque_sign * (
    tau_pitch + tau_pitch_rate + tau_sagittal_velocity + 
    tau_position + tau_support_velocity +  # NEW TERM
    tau_cp + tau_com_vy
)
```

**Proposed gain:**
- Start with `k_support_velocity = 10.0 N·s/m` (conservative)
- At peak velocity (0.397 m/s): tau_support_velocity = -3.97 Nm
- This would have opposed the drift during growth phase

**Why not other fixes:**

- **Fix A (repair capture gate):** Gate logic is correct. Not relevant to this failure.
- **Fix C (integral term):** No persistent bias. Final error is small. Integral not needed.
- **Fix D (position reference governor):** More complex than direct velocity damping. Adds state and tuning complexity.
- **Fix E (increase position authority):** Risky. Could destabilize pitch balance. Velocity damping is safer.

---

## Task 5: Implementation Plan

### Step 1: Add support position velocity computation

**Location:** `sagittal_velocity_damped_balance_controller.py`

**Changes:**
1. Add `prev_support_position_error_m` to controller state
2. Compute `support_position_velocity_m_s` as numerical derivative
3. Add `k_support_velocity` parameter (default 10.0)
4. Add `tau_support_velocity` term to control law
5. Update diagnostics to log new term

### Step 2: Update telemetry

**Add columns:**
- `support_position_velocity_m_s`
- `tau_support_velocity`
- `k_support_velocity`

### Step 3: Update tests

**Add tests for:**
- Support position velocity sign under forward/backward motion
- Support velocity damping opposes drift
- tau_support_velocity = -k_support_velocity * support_position_velocity_m_s
- No WBC, no E0b/E0c/E0d, kp_cp = 0.0
- Ownership unchanged

### Step 4: Validation protocol

**Run sequence:**
1. V1: 500 steps nominal
2. V2: 1000 steps nominal
3. V3: 2000 steps nominal
4. V4: 5000 steps nominal

**Acceptance criteria:**
- Preferred: max |support_position_error_m| ≤ 0.10m, final ≤ 0.05m
- Fallback: max |support_position_error_m| ≤ 0.15m, final ≤ 0.10m
- Hard minimum: max |support_position_error_m| ≤ 0.30m, final ≤ 0.10m

---

## Summary

**Capture gate diagnosis:**
- Gate never activated because capture point stayed within ±0.064m (below 0.10m threshold)
- Gate logic is correct but irrelevant to support-center drift problem
- This is a support-drift failure, not a capture-conflict failure

**Root cause classification:**
- **Primary: C. position_gain_or_authority_insufficient**
  - tau_position saturated at 3.0 Nm (wanted 11.9 Nm)
  - Effective gain dropped from 20.0 to 5.04 N/m during saturation
- **Secondary: D. missing_position_velocity_regulation**
  - Support velocity reached 0.397 m/s with no direct damping
  - Allowed error to grow unchecked

**Recommended fix:**
- **Fix B: Add explicit support-position velocity regulation**
- Add `tau_support_velocity = -k_support_velocity * support_position_velocity_m_s`
- Start with `k_support_velocity = 10.0 N·s/m`
- Preserves position authority limit, adds velocity damping

**Next steps:**
1. Implement support velocity damping
2. Add tests
3. Run validation sequence
4. Generate fix report

---

## Files Generated

- `step_e_position_regulator_root_cause_report.md` (this file)
- `step_e_position_regulator_root_cause_report.json` (next)
