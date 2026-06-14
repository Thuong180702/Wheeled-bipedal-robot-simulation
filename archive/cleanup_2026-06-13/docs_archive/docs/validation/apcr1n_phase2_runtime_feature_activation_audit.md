# APCR1n Phase 2 Runtime Feature Activation Audit

## Summary

**Classification: APCR1N_PHASE2_FEATURES_ELIGIBLE_BUT_NOT_ACTIVE**

APCR1n features did not activate during the 2000-step run. Root cause analysis reveals this is due to the Active Pitch Crossing (APC) system being disabled throughout the run, which prevents the drift priority mechanism from engaging.

## Telemetry Verification

All 16 APCR1n telemetry columns are present and populated:
- ✅ apcr1n_recenter_priority_active
- ✅ apcr1n_startup_guard_active
- ✅ apcr1n_wheel_damping_override_active
- ✅ apcr1n_wheel_damping_scale
- ✅ apcr1n_wheel_damping_before
- ✅ apcr1n_wheel_damping_after
- ✅ apcr1n_wheel_damping_fights_drift
- ✅ apcr1n_position_cap_boost_active
- ✅ apcr1n_position_cap_current
- ✅ apcr1n_tau_position_raw
- ✅ apcr1n_tau_position_after_cap
- ✅ apcr1n_position_saturated
- ✅ apcr1n_safety_gate_pass
- ✅ apcr1n_final_torque_direction_correct
- ✅ apcr1n_final_torque_fights_drift
- ✅ apcr1n_physical_drift_column_used

## Startup Guard Analysis

- Startup guard active count: 100/2000 (5.0%)
- Guarded steps 0-99: 100/100 ✅
- Guarded steps 100+: 0 ✅
- Wheel damping override during guard: 0 ✅
- Position cap boost during guard: 0 ✅

**Startup guard works correctly.**

## Feature Activation Analysis

### Feature 1: Recenter Priority
- Active count: 0/2000 (0.00%)
- **Root cause: active_pitch_crossing is DISABLED throughout the run**

### Feature 2: Wheel Damping Override
- Active count: 0/2000 (0.00%)
- **Blocked by drift priority not activating**

### Feature 3: Position Cap Boost
- Active count: 0/2000 (0.00%)
- Position cap stuck at 6.0 Nm (max) throughout
- Position saturated: 55/2000 (2.75%)
- **Blocked by drift priority not activating**

## Drift Statistics

- Error range: -0.0140 to +0.1714 m
- Mean error: +0.0600 m (positive bias)
- abs_max: 0.1714 m

### Drift Priority Eligibility (steps 100+)
- abs(error) > 0.08: 716/1900 (37.7%)
- Moving away from zero: 899/1900 (47.3%)
- **BOTH conditions: 337/1900 (17.7%)**

Drift DID reach levels that should trigger drift priority, but APC was disabled.

## APC Telemetry Analysis

- active_pitch_crossing_active: 0/2000 (0.00%)
- active_pitch_crossing_state: NEUTRAL throughout
- **active_pitch_crossing_gate_reason: "disabled" for all 2000 steps**

This is the root cause: APC is disabled, so drift priority never activates, so APCR1n features never activate.

## Safety Gate Analysis

- Safety gate pass: 0/2000 (0.00%)
- Safety conditions (steps 100+):
  - n_contacts >= 1: 1900/1900 ✅
  - com_z >= 0.25: 1900/1900 ✅
  - |pitch| <= 20: 1900/1900 ✅
  - |roll| <= 10: 1900/1900 ✅

Safety conditions are met, but safety gate pass = 0 because drift priority never activated.

## Torque Direction Analysis

- Final torque direction correct: 2000/2000 (100.00%) ✅
- Final torque fights drift: 0/2000 (0.00%) ✅

Torque direction is correct throughout, even without APCR1n features.

## Root Cause Analysis

The APCR1n features are designed to activate when drift priority is active. Drift priority activates when:
1. abs(error) > 0.08 m
2. Error is moving away from zero

However, drift priority depends on the Active Pitch Crossing (APC) system being enabled. The telemetry shows:
- `active_pitch_crossing_gate_reason = "disabled"` for all 2000 steps
- `active_pitch_crossing_active = 0` for all 2000 steps

This means the APC system is not engaged, so `_apc_drift_priority_active` never becomes True, so APCR1n features never activate.

## Why APC is Disabled

The APC system has multiple safety gates:
- pitch_safe: 1532/2000 (80.6%)
- pitch_danger: 79/2000 (4.2%)
- contact_safe: 1999/2000 (99.95%)
- height_safe: 2000/2000 (100%)
- roll_safe: 2000/2000 (100%)

The issue may be that pitch_danger (79 steps) or some other gate condition blocks APC from engaging. However, the gate_reason = "disabled" suggests the APC is explicitly disabled, not blocked by safety gates.

## Recommendation

This is NOT a failure of APCR1n feature code. The features are correctly wired and the telemetry is correct. The issue is that the APC system is not engaged in this configuration.

Options:
1. Investigate why APC is disabled (is this intentional for APCR1n?)
2. If APCR1n is meant to work without APC, the recenter priority detection logic needs to be decoupled from APC
3. If APCR1n requires APC, enable APC or provide an alternative recenter detection mechanism

## Classification Justification

**APCR1N_PHASE2_FEATURES_ELIGIBLE_BUT_NOT_ACTIVE**

- Features did not activate
- Eligibility conditions DID occur (337 steps had abs > 0.08 AND moving away)
- But features cannot activate because drift priority (their prerequisite) never activates
- Drift priority cannot activate because APC is disabled
- This is a configuration/architecture issue, not a code bug
