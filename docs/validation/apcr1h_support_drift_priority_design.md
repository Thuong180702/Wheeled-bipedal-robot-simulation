# APCR1h Support Drift Priority Design

## Date
2026-06-09

## Root Cause from Phase 1 and Phase 2

**APCR1g applies WRONG SIGN torque when drift exceeds threshold.**

- When drift > +0.10: APCR1g applies **positive** torque (should be **negative**)
- When drift > +0.10: APCR1f applies **negative** torque (correct)

This single issue causes APCR1g to accelerate drift rather than oppose it.

---

## Design Philosophy

APCR1h must:
1. **Base on APCR1f** (correct torque sign) not APCR1g (wrong torque sign)
2. **Prioritize support drift reduction** over pitch smoothing
3. **Allow higher wheel velocity** when needed to reduce drift
4. **Add drift-priority override** when drift is runaway

---

## APCR1h Profile: `APCR1h_support_drift_priority_fast_recenter`

### Entry and Bands

| Parameter | Value | Description |
|-----------|-------|-------------|
| `inner_deadband_m` | 0.015 | Within this, no APCR correction |
| `soft_enter_m` | 0.030 | Soft enter threshold |
| `target_band_m` | 0.08 | Target recovery band |
| `drift_priority_enter_m` | 0.08 | Drift priority activates |
| `emergency_drift_m` | 0.12 | Emergency clamp threshold |
| `hard_drift_m` | 0.15 | Hard safety threshold |

### Authority Levels

| Level | tau_max (Nm) | Description |
|-------|--------------|-------------|
| Base | 1.25 | Normal APCR authority |
| Drift Priority | 1.65 | When drift > 0.08 AND moving away |
| Emergency | 1.85 | When drift > 0.12 |
| Startup | 1.60 | First 500 steps, higher authority |

### Response Rates

| Level | rate_per_step (Nm/step) | Description |
|-------|------------------------|-------------|
| Normal | 0.55 | Normal APCR response |
| Drift Priority | 0.85 | Faster response when priority |
| Emergency | 1.00 | Maximum response for clamp |
| Decay | 0.55 | Torque decay rate |

### Phase Brake Rules

| Condition | Phase Brake |
|-----------|------------|
| `abs_e < 0.06` AND moving toward zero | Strong phase brake |
| `abs_e < 0.10` AND moving toward zero | Allow phase brake |
| `abs_e > 0.10` | **Disable phase brake** |
| `abs_e > 0.08` AND moving away | **Disable phase brake** |
| Drift priority active | **Disable phase brake** |

### Wheel Velocity Policy

- **Monitor-only** unless contact/height/roll destabilizes
- **Do NOT penalize** wheel velocity for reducing drift
- Allow wheel velocity to rise above APCR1g levels if needed
- APCR1g max wheel vel was 4.20 rad/s → APCR1h may need higher

### Torque Sign Convention

```
If drift > 0: apply NEGATIVE torque to reduce positive drift
If drift < 0: apply POSITIVE torque to reduce negative drift
```

This is the **same as APCR1f**, NOT APCR1g.

### Drift Clamp Behavior

When `abs_e > 0.15`:
1. Increase torque until e_dot stops moving away
2. Target e_dot sign reversal within N steps
3. Allow higher torque and faster rate
4. Log if clamp fails to reverse e_dot

---

## Telemetry Fields

Required telemetry for APCR1h:

| Field | Type | Description |
|-------|------|-------------|
| `drift_priority_enabled` | bool | Profile enables drift priority |
| `drift_priority_active` | bool | Currently in drift priority mode |
| `emergency_drift_clamp_active` | bool | Emergency clamp is active |
| `drift_priority_reason` | string | Why drift priority activated |
| `drift_priority_tau_limit` | float | Tau limit for this step |
| `selected_tau_limit` | float | Actual tau limit applied |
| `selected_rate_limit` | float | Actual rate limit applied |
| `support_priority_over_pitch` | bool | Support drift prioritized over pitch |
| `phase_brake_disabled_reason` | string | Why phase brake was disabled |
| `drift_clamp_success` | bool | Emergency clamp reversed e_dot |
| `steps_since_hard_drift` | int | Steps since abs_e > 0.15 |
| `error_rate_reversal_achieved` | bool | e_dot sign reversed |
| `physical_drift_column_used` | string | Column used for drift |
| `wheel_velocity_monitor_only` | bool | Wheel vel is monitor only |

---

## Implementation Notes

1. **Start from APCR1f code**, not APCR1g
2. **Copy APCR1f torque sign logic** exactly
3. **Add drift priority levels** above APCR1f base
4. **Modify phase brake** to disable when drift priority active
5. **Monitor wheel velocity** but don't restrict it for drift control

---

## Expected Behavior

| Scenario | APCR1f | APCR1g | APCR1h |
|----------|--------|--------|--------|
| drift > 0.08, moving away | Normal tau | Wrong sign | **Higher tau, correct sign** |
| drift > 0.12 | Normal tau | Wrong sign | **Emergency tau** |
| drift > 0.15 | Clamp | Wrong sign | **Emergency clamp, correct sign** |
| pitch danger | Reduce tau | Reduce tau | **Reduce tau if needed** |
| phase brake | Active | Active | **Disabled when drift priority** |

---

## Success Criteria

| Metric | Target | APCR1f Baseline |
|--------|--------|-----------------|
| max drift | < 0.14 m | 0.157 m |
| P2P | < 0.18 m | 0.207 m |
| outside ±0.15 | < 2.2% | 2.2% |
| outside ±0.10 | < 32.6% | 32.6% |
| wheel velocity | monitor | 5.44 rad/s max |
