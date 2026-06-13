# E1 Wiring/Gate Audit After 500-Step

## Audit Classification: `E1_WIRING_OK_GATE_BLOCKS_EFFECT`

## Evidence Summary

The 500-step E1 vs D2 comparison showed:
- **E1 and D2 telemetry are nearly identical** (delta ≈ 0 on all metrics)
- E1 integral_active was only **22/500 steps (4.4%)**
- E1 tau_position_integral max was only **0.001001 Nm** (effectively zero)
- D2 support_position_error at step 500 was **0.004710 m** (well within bounds)

## Wiring Verification

| Check | Result | Evidence |
|-------|--------|---------|
| E1_support_integral profile exists | ✅ PASS | Line 197-215 in simulate_hierarchical_controller.py |
| E1 sets enable_position_integral=True | ✅ PASS | Line 201: `enable_position_integral=True` |
| E1 parameters ki=2.0, max=1.0 | ✅ PASS | Lines 202-203 |
| E1 profile passed to controller | ✅ PASS | Lines 2576-2589 extract profile params, line 2613 passes to constructor |
| Controller receives all integral params | ✅ PASS | Lines 1107-1111 pass to SagittalVelocityDampedBalanceController |
| tau_position_integral added to tau_position_raw | ✅ PASS | Line 607 in controller: `tau_position_raw = tau_position_p + tau_position_integral` |
| tau_position_integral is bounded | ✅ PASS | Lines 598-601 clip to integral_max_abs |
| Telemetry reports integral fields | ✅ PASS | Lines 764-767 in controller diagnostics |

## Root Cause

The **pitch_error_large gate blocks the integral for 349/500 steps (69.8%)** at low_0p300.

E1's `integral_pitch_error_threshold_rad=0.03 rad` is too restrictive for low_0p300:
- At low_0p300, robot pitch oscillates with max **0.111 rad (6.4 deg)**
- The integral threshold of 0.03 rad (1.7 deg) is exceeded for **349/500 steps (69.8%)**
- This means the integral is blocked during most of the oscillation cycle
- When pitch drops below 0.03 rad, other gates (support_velocity_large) also block it
- Only 22 steps (4.4%) reached `safe_steady_state`

The gate logic in `sagittal_velocity_damped_balance_controller.py` lines 574-590:
```python
if abs_pitch_error > self.integral_pitch_error_threshold_rad:
    integral_gate_reason = "pitch_error_large"
elif abs_pitch_rate > self.integral_pitch_rate_threshold_rad_s:
    integral_gate_reason = "pitch_rate_large"
elif abs_support_velocity > self.integral_support_velocity_threshold_m_s:
    integral_gate_reason = "support_velocity_large"
# ... etc
else:
    integral_active = True
    integral_gate_reason = "safe_steady_state"
```

**This is a gate design flaw, not a wiring flaw.** The pitch_error_large gate was intended to prevent the integral from winding up during large transients, but at low_0p300:
1. Pitch oscillations of 0.03-0.11 rad are *normal*, not transient
2. The integral cannot accumulate enough error to help with steady-state drift
3. Support drift at low_0p300 is the real problem the integral was meant to fix

## Why E1 Has No Effect

The integral's intended purpose is to correct steady-state **support drift** (cp_x error). But:
1. At step 0-500, support_position_error is only **0.002-0.005 m** (essentially zero drift)
2. The D2 baseline itself has **no support drift** in the first 500 steps
3. The integral was never given a chance to work because:
   - Pitch oscillations block it 70% of the time
   - When pitch drops, support_velocity also blocks it
   - Even when neither blocks it, the integral accumulates only ~22 steps worth

## Gate Fix Options

### Option A: Raise pitch_error threshold to 0.10 rad (Recommended)
- Keep pitch_error_large gate but raise threshold to 0.10 rad
- This allows the integral to activate during normal low_0p300 pitch oscillations
- Still protects against extreme pitch (>0.10 rad) which indicates fall

### Option B: Remove pitch_error_large gate entirely
- Let the integral accumulate during pitch oscillations
- Risk: integral may wind up during large transients
- Mitigation: keep integral_max_abs=1.0 cap, anti-windup resets on any gate failure

### Option C: Use anti-windup as the only gate
- Keep integral_max_abs=1.0
- Let integral accumulate whenever support_position_error exists
- Only reset on extreme conditions (contact_invalid, height_unsafe)

### Option D: Gate only on contact/termination safety
- Keep integral_min_com_z_m=0.28 and integral_max_com_z_m=0.50
- Keep contact_valid gate
- Remove all other gates (pitch, pitch_rate, support_velocity, wheel_velocity, roll)
- Let the integral work continuously within height bounds

## Recommendation

**Option A**: Raise `integral_pitch_error_threshold_rad` from 0.03 to **0.10 rad**.
- Rationale: 0.10 rad (5.7 deg) is a reasonable pitch safe threshold
- Below this, low_0p300 can safely accumulate integral action
- Above this, the robot is likely in a dangerous transient - block the integral
- This is the smallest safe fix that addresses the root cause

**Alternative Option D**: Gate only on contact_valid and height bounds.
- Rationale: The integral's purpose is to correct support drift, not pitch
- Pitch correction is handled by the PD controller
- Let the integral work continuously except for safety-critical conditions
- This maximizes integral availability while keeping safety guards

**Decision**: Proceed with Option A (raise threshold to 0.10 rad) as the primary fix, but also implement Option D as an alternative profile if Option A proves insufficient.
