# K2 Coordinated Stability Controller Design

**Phase:** 4 — Design Coordinated Control Architecture
**Date:** 2026-06-30
**Based on:** Phase 3 Controller Interaction Audit

## 1. Design Objective

Transform the K2 JAX controller from 10 independent, blindly-summed torque components
into a coordinated architecture where components share state, respect authority
limits, and do not fight each other.

The Phase 3 audit proved that the **#1 source of wasted control effort** is the
empirical support feedforward fighting the shape posture PD at knees and hip_pitch
joints (7.2 Nm and 4.4 Nm RMS cancellation respectively, 100% sign opposition).

## 2. Core Design Principle

**Route feedforward through posture targets, not direct torques.**

Instead of:
```
tau_knee = PD(q_ref, q_actual, q_vel) + FF_constant  ← THEY FIGHT
```

Use:
```
q_ref_biased = q_ref + delta_q_ref(FF_constant, kp)
tau_knee = PD(q_ref_biased, q_actual, q_vel)          ← THEY AGREE
```

The net torque at equilibrium is identical. The difference is that the PD's
feedback correction is now centered around the biased target — perturbations
are corrected *relative to* the feedforward-biased posture, not *against* it.

## 3. Candidate A: FF-to-Posture-Bias Conversion

### 3.1 Derivation

The shape posture PD law (from `k2_jax_shape_posture_compute`):
```
tau_pd = -kp * (q_actual - q_ref) - kd * q_vel
```

With a q_ref bias delta:
```
tau_pd_biased = -kp * (q_actual - (q_ref + delta)) - kd * q_vel
              = -kp * (q_actual - q_ref) - kd * q_vel + kp * delta
              = tau_pd_original + kp * delta
```

To replace empirical FF torque tau_ff with equivalent bias:
```
kp * delta = tau_ff  →  delta = tau_ff / kp
```

### 3.2 Per-Joint Bias Values

| Joint | kp | Empirical FF (Nm) | delta_q_ref (rad) | delta_q_ref (deg) |
|-------|-----|-------------------|-------------------|-------------------|
| Left knee [3] | 40.0 | -7.75 | -0.19375 | -11.10° |
| Right knee [8] | 40.0 | -7.90 | -0.19750 | -11.32° |
| Left hip_pitch [2] | 30.0 | +2.05 | +0.06833 | +3.92° |
| Right hip_pitch [7] | 30.0 | +1.60 | +0.05333 | +3.06° |

Sign convention: negative delta at knees means the biased target is more flexed
(less extended), matching the FF's intent to push knees toward extension. Positive
delta at hip_pitch means the biased target is more pitched forward.

### 3.3 Implementation Strategy

**Step 1:** Compute biased q_ref from empirical FF and posture gains:
```python
q_ref_biased = q_ref.copy()
q_ref_biased[3] += tau_ff_knee_l / kp_knee    # -7.75/40
q_ref_biased[8] += tau_ff_knee_r / kp_knee    # -7.90/40
q_ref_biased[2] += tau_ff_hp_l / kp_hip_pitch  # +2.05/30
q_ref_biased[7] += tau_ff_hp_r / kp_hip_pitch  # +1.60/30
```

**Step 2:** Pass biased q_ref to posture PD instead of original q_ref.

**Step 3:** Remove empirical FF from tau_sum (or fade it down with a continuous gate).

**Step 4:** Apply a smooth height-dependent blend to preserve behavior at heights
where the empirical FF was validated:
```python
# Blend factor: 1.0 = full bias, 0.0 = original FF
bias_weight = smoothstep(schedule_h, 0.30, 0.50)  # ramp bias in above 0.30m
ff_weight = 1.0 - bias_weight
```

The blend ensures:
- At very low heights (<0.30m): original empirical FF preserved (validated regime)
- At mid/high heights (>0.50m): full bias replacement (where conflicts are worst)
- Smooth transition in between

### 3.4 Expected Impact

- **Knee cancellation:** Should drop from 7.2 Nm RMS to near zero
- **Hip pitch cancellation:** Should drop from 4.4 Nm RMS to near zero
- **Posture:Balance power ratio:** Should drop from 4.3:1 toward 1:1
- **Pitch RMS:** Should improve as posture PD can now provide meaningful corrections
- **Zero net torque change at equilibrium** — the bias approach preserves the
  same steady-state torque as the original FF

## 4. Candidate B: Continuous Authority Allocator

### 4.1 Design

After FF-to-bias conversion removes the primary conflict source, add a continuous
authority allocator that dynamically adjusts component weights based on state.

### 4.2 Authority Priorities (descending)

1. **Balance/safety** (wheels, pitch stabilization) — never reduced
2. **Support/contact** (support correction) — yield only to balance
3. **Pitch/roll damping** — yield to balance and support
4. **Hip-yaw divergence** — yield to balance, support, pitch
5. **Posture comfort** (shape tracking) — yield to all above
6. **Torque smoothness** — passive (rate limit handles this)

### 4.3 Authority Weights

```python
# Balance authority: always 1.0 (protected)
w_balance = 1.0

# Posture authority: yields when pitch or support error is large
w_posture = smoothstep_gate(abs(pitch), 3.0, 8.0)  # deg, ramps down as pitch grows
           * smoothstep_gate(abs(support_error), 0.02, 0.08)  # m, ramps down as support drifts
           * contact_quality  # ramps down with poor contact

# Hip-yaw authority: increases with divergence, decreases with instability
w_hip_yaw = smoothstep_gate(abs(hip_yaw_div), 0.05, 0.20)  # rad, ramps up as divergence grows
          * contact_quality
          * (1.0 - 0.5 * (1.0 - w_posture))  # partial coupling to posture

# Mode-div authority: similar to hip-yaw but stronger height dependence
w_mode_div = smoothstep_gate(abs(hip_yaw_div), 0.08, 0.25)
           * smoothstep_gate(com_z, 0.30, 0.45)  # more active at taller heights
           * contact_quality

# Support correction authority
w_support = smoothstep_gate(abs(support_error), 0.01, 0.05)
          * contact_quality
          * w_balance  # never exceeds balance
```

### 4.4 Application

Authorities are applied as multiplicative weights on component torque outputs:
```python
tau_posture_weighted = w_posture * tau_posture
tau_yaw_weighted = w_balance * tau_yaw  # yaw is part of balance
tau_mode_div_weighted = w_mode_div * tau_mode_div
```

Balance torque (wheel sagittal) is never weighted down — it always operates at 1.0.

## 5. Candidate C: Hip-Yaw / Mode-Div Authority Scheduling

### 5.1 Problem

Phase 3 found: mode_div and yaw controllers produce near-zero torque because
their gains (kp=10, kp=8) are too low relative to posture hip_yaw (kp=15).
The height gate (0.30-0.80 smoothstep) is unnecessarily wide.

### 5.2 Design

1. **Narrow the mode-div height gate:** From [0.30, 0.80] to [0.30, 0.50].
   Above 0.50m, mode-div is fully active. Below 0.30m, fully off.
   This matches the physical observation that hip-yaw divergence is mostly
   a concern at taller heights where leg geometry changes.

2. **Dynamic mode-div gain based on divergence magnitude:**
```python
k_mode_div_effective = k_mode_div_nominal * (1.0 + 2.0 * smoothstep_gate(abs(div), 0.10, 0.25))
```
When divergence exceeds 0.10 rad, gain ramps from 1.0× to 3.0×. This makes
the controller responsive when needed without being aggressive at small errors.

3. **Reduce posture hip_yaw authority when mode-div is active:**
```python
w_posture_hy = 1.0 - 0.5 * w_mode_div  # posture yields 50% to mode-div
```

## 6. Candidate D: Dynamic Height Coordination

### 6.1 Problem

Dynamic height scenarios have the highest pitch RMS (4.46 deg mean) because:
1. q_ref changes discontinuously (or with linear ramp) while physics is continuous
2. Contact quality changes during transitions aren't accounted for
3. Pitch oscillations during transitions aren't damped

### 6.2 Design

1. **Continuous q_ref interpolation with contact-quality-aware rate limiting:**
```python
q_ref_rate_max = q_ref_rate_nominal * contact_quality * pitch_stability
```
When contact is poor or pitch is oscillating, slow down q_ref transitions.

2. **Height-tracking authority schedule:**
```python
w_height_track = smoothstep_gate(contact_quality, 0.5, 1.0)
               * smoothstep_gate(pitch_stability, 0.0, 1.0)
```
where `pitch_stability = 1.0 - smoothstep(abs(pitch_rms), 2.0, 6.0)`

3. **Pitch compensation during height transitions:**
```python
tau_pitch_comp = -k_comp * height_velocity * sign(pitch) * smoothstep(abs(pitch), 1.0, 5.0)
```
A small bounded term that anticipates the pitch effect of height changes.

## 7. Implementation Order

1. **Phase 5-7 (Candidate A):** FF-to-bias conversion — highest impact, lowest risk
2. **Phase 6 (Candidate B):** Authority allocator — structural improvement
3. **Phase 8 (Candidate C):** Hip-yaw scheduling — targeted fix
4. **Phase 9 (Candidate D):** Dynamic height coordination — quality improvement

Each candidate is independently testable and can be rolled back.

## 8. Safety Constraints

- All schedules are continuous (smoothstep, no discrete thresholds)
- No scenario-specific constants
- Balance authority never reduced below 1.0
- Zero net torque change at equilibrium (FF→bias conversion is torque-equivalent)
- All changes are explainable as control principles, not patches

## 9. Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| K2_STABILITY_SCORE | 0.683 | ≥ 0.80 |
| Knee cancellation RMS | 7.2 Nm | < 1.0 Nm |
| Hip pitch cancellation RMS | 4.4 Nm | < 0.5 Nm |
| Posture:Balance ratio | 4.3:1 | < 2.0:1 |
| Pitch RMS (problem cases) | 3.9° | < 3.0° |
| Dynamic pitch RMS | 4.5° | < 3.5° |
| SAFETY_FAIL | 0 | 0 |
| Performance | 147 Hz | ≥ 50 Hz |
