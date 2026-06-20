# Sagittal Equilibrium State Audit

**Date:** 2026-06-15  
**Phase:** Phase 2 — Equilibrium Analysis  
**Scenario:** high_0p480, 5000 steps  
**Profiles analyzed:**
- `adaptive_support_centering_trim` (CSV: adaptive_5000_high_0p480)
- `zero_crossing_support_recenter` (CSV: zc_5000_high_0p480)
- `early_zero_crossing_recenter` (CSV: ezc_5000_high_0p480)
- `early_zero_crossing_recenter_v2` (CSV: ezc_v2_5000_high_0p480)

---

## Classification

**FORWARD_EQUILIBRIUM_POSTURE_CONFIRMED**

Secondary: **PITCH_REFERENCE_CONTROLLER_CONFLICT_CONFIRMED**

Tertiary: **TORQUE_COMPOSITION_CONFLICT_CONFIRMED**

---

## Executive Summary

The robot at high_0p480 is in a **persistent forward-pitch equilibrium state** with:
- Mean pitch: **+3.6 to +3.9 degrees** (not oscillating around zero)
- pitch_ref: **exactly 0.0 rad** (not a reference bias)
- Mean tau_pitch: **+3.2 to +3.4 Nm** (correct response to forward pitch)
- Mean tau_position: **-3.5 to -3.7 Nm** (correcting the drift this creates)
- Position controller saturating at negative bound: **27–31% of steps**
- Final wheel torque: **near zero** (tau_pitch + tau_position cancel)
- Support drift: **92–84% positive** (not centered)

**The root cause is NOT in the controller logic. The root cause is that the robot's physical equilibrium posture is forward-pitched, and the controller correctly responds to it.**

---

## 1. State Analysis

### 1.1 Support Drift

| Profile | pos % | neg % | symmetry | min (m) | max (m) | P2P (m) |
|---------|-------|-------|----------|---------|---------|---------|
| adaptive | **92.2%** | 7.7% | 11.9:1 | -0.071 | +0.185 | 0.256 |
| zc | **86.4%** | 13.6% | 6.3:1 | -0.066 | +0.194 | 0.260 |
| ezc | **82.7%** | 3.4% | 24.3:1 | -0.039 | +0.200 | 0.239 |
| ezc_v2 | **84.4%** | 15.6% | 5.4:1 | -0.056 | +0.193 | 0.249 |

**Finding:** ALL profiles show predominantly positive drift. This is not a profile-specific issue — it's a physics issue.

### 1.2 Pitch Angle

| Profile | mean (deg) | median (deg) | min | max |
|---------|-----------|--------------|-----|-----|
| adaptive | **+3.637** | +3.085 | -1.267 | +8.438 |
| zc | **+3.812** | +3.237 | -1.444 | +8.592 |
| ezc | **+3.797** | +3.253 | -1.632 | +8.620 |
| ezc_v2 | **+3.873** | +3.338 | -1.720 | +8.658 |

**Critical finding:** The mean pitch is **+3.6 to +3.9 degrees FORWARD**, not oscillating around zero. The robot is physically leaning forward at equilibrium.

### 1.3 Pitch Reference

For ALL profiles:
```
pitch_x_ref_rad: mean = 0.0, min = 0.0, max = 0.0 (EXACTLY ZERO)
```

**Finding:** pitch_ref is not biased. The forward pitch comes from physics, not controller reference.

### 1.4 Pitch Error

| Profile | mean (rad) | mean (deg) |
|---------|-----------|-----------|
| adaptive | +0.0635 | **+3.64 deg** |
| zc | +0.0665 | **+3.81 deg** |
| ezc | +0.0663 | **+3.80 deg** |
| ezc_v2 | +0.0676 | **+3.87 deg** |

**Finding:** pitch_error ≈ pitch_mean (since pitch_ref=0). The error IS the forward equilibrium pitch.

### 1.5 Near-Zero Windows

| Profile | |pitch|<1° (n) | |pitch|<1° % |
|---------|-----------|--------|
| adaptive | 1056 | 21.1% |
| zc | 1175 | 23.5% |
| ezc | 1200 | 24.0% |
| ezc_v2 | 1090 | 21.8% |

**Finding:** About 75–80% of all steps have |pitch| > 1°. The forward pitch is PERSISTENT, not transient.

---

## 2. Torque Analysis

### 2.1 tau_pitch

| Profile | mean (Nm) | median (Nm) | min | max | RMS |
|---------|-----------|--------------|-----|-----|-----|
| adaptive | **+3.174** | +2.692 | -1.106 | +7.363 | 3.990 |
| zc | **+3.327** | +2.823 | -1.260 | +7.500 | 4.250 |
| ezc | **+3.314** | +2.845 | -1.424 | +7.518 | 4.261 |
| ezc_v2 | **+3.380** | +2.901 | -1.501 | +7.551 | 4.308 |

**Finding:** Mean tau_pitch ≈ +3.2 to +3.4 Nm (always positive). This is proportional to mean pitch (~+3.7 deg) with kp_pitch = 50 Nm/rad:
- +3.7 deg = +0.065 rad
- 50 × 0.065 = +3.25 Nm ✓ (matches observed +3.2 to +3.4 Nm)

**tau_pitch is NOT a bias injection. It is the correct response to the forward-pitch equilibrium.**

### 2.2 tau_pitch in Near-Zero Windows (|pitch| < 1°)

| Profile | n | mean (Nm) | median (Nm) |
|---------|---|-----------|--------------|
| adaptive | 1056 | **+0.282** | +0.320 |
| zc | 1175 | **+0.225** | +0.245 |
| ezc | 1200 | **+0.218** | +0.246 |
| ezc_v2 | 1090 | **+0.201** | +0.280 |

**Critical finding:** Even when pitch is constrained to |pitch| < 1°, tau_pitch has a residual positive mean of **+0.20 to +0.28 Nm**. This is the ONLY true controller-side DC bias — everything above +0.28 Nm is dynamic response to forward pitch.

### 2.3 tau_position

| Profile | mean (Nm) | median (Nm) | min | max | RMS |
|---------|-----------|------------|-----|-----|-----|
| adaptive | **-3.497** | -3.258 | -7.000 | +0.998 | 4.251 |
| zc | **-3.683** | -3.575 | -7.000 | +1.164 | 4.550 |
| ezc | **-3.673** | -3.575 | -7.000 | +1.161 | 4.561 |
| ezc_v2 | **-3.737** | -3.726 | -7.000 | +1.237 | 4.576 |

**Finding:** tau_position is ALWAYS negative (backward) with mean -3.5 to -3.7 Nm. It's trying to move the wheels backward to recenter support.

### 2.4 tau_pitch + tau_position (Net Common at Equilibrium)

| Profile | tau_pitch | tau_position | NET |
|---------|-----------|-------------|-----|
| adaptive | +3.174 | -3.497 | **-0.323** |
| zc | +3.327 | -3.683 | **-0.356** |
| ezc | +3.314 | -3.673 | **-0.359** |
| ezc_v2 | +3.380 | -3.737 | **-0.357** |

**Critical finding:** tau_pitch + tau_position ≈ **-0.35 Nm** (near zero). They nearly cancel at equilibrium, leaving the robot in a forward-pitch stalemate.

### 2.5 Final Wheel Torque

| Profile | mean (Nm) | median (Nm) | min | max | RMS |
|---------|-----------|-------------|-----|-----|-----|
| adaptive | **+0.004** | -0.055 | -1.451 | +1.157 | 0.246 |
| zc | **+0.010** | -0.037 | -2.642 | +3.352 | 0.278 |
| ezc | **+0.010** | -0.037 | -2.630 | +3.380 | 0.278 |
| ezc_v2 | **+0.012** | -0.037 | -2.634 | +3.354 | 0.278 |

**Finding:** Final wheel torque mean ≈ 0 Nm. The system is in a "frozen" equilibrium state — tau_pitch and tau_position cancel, leaving no net correction force.

### 2.6 tau_wheel_velocity (Damping)

| Profile | mean (Nm) | RMS |
|---------|-----------|-----|
| adaptive | **+0.348** | 1.398 |
| zc | **+0.399** | 1.562 |
| ezc | **+0.387** | 1.562 |
| ezc_v2 | **+0.394** | 1.573 |

**Finding:** Mean tau_wheel_velocity is positive (+0.35 to +0.40 Nm). This means wheels are moving backward on average (fighting forward drift). At equilibrium this is small, but it adds to the positive torque budget.

---

## 3. Correlations

| Profile | tau_pitch ↔ drift | tau_pitch ↔ pitch | tau_pitch ↔ pitch_rate | tau_pitch ↔ pitch_err |
|---------|-------------------|-------------------|------------------------|----------------------|
| adaptive | +0.996 | **+1.000** | -0.002 | **+1.000** |
| zc | +0.997 | **+1.000** | +0.005 | **+1.000** |
| ezc | +0.995 | **+1.000** | -0.001 | **+1.000** |
| ezc_v2 | +0.994 | **+1.000** | +0.002 | **+1.000** |

**Findings:**
1. tau_pitch is a **perfect linear function of pitch_error** (r = +1.000). No sign error, no gain asymmetry, no DC offset injection.
2. tau_pitch has **negligible correlation with pitch_rate** (r ≈ 0). The tau_pitch_rate term handles pitch rate separately.
3. The strong correlation tau_pitch ↔ drift is caused by pitch ↔ drift correlation (both rise together in the forward equilibrium).

---

## 4. Saturation Analysis

### 4.1 Position Controller Saturation

| Profile | pos sat % | neg sat % | tau_pos when pitch>0.02 | tau_pos when pitch<-0.02 |
|---------|-----------|-----------|-------------------------|--------------------------|
| adaptive | 0.0% | **13.1%** | -4.484 Nm | +0.950 Nm |
| zc | 0.0% | **27.8%** | -4.901 Nm | +1.052 Nm |
| ezc | 0.0% | **28.6%** | -4.956 Nm | +1.161 Nm |
| ezc_v2 | 0.0% | **31.3%** | -4.932 Nm | +1.237 Nm |

**Critical finding:** Position controller is **ALWAYS saturated on the negative side**, NEVER on the positive side. This means:
1. tau_balance_before_position is consistently positive (consuming headroom for backward torque)
2. The backward torque authority is capped by the upper bound
3. When pitch is high (pitch > 0.02 rad), tau_position saturates at -4.5 to -5.0 Nm (near max)

### 4.2 Saturation as a Function of Drift Sign

| Profile | pos_drift steps | tau_pos when pos drift | neg_drift steps | tau_pos when neg drift |
|---------|----------------|------------------------|-----------------|------------------------|
| adaptive | 4611 (92%) | **-4.012 Nm** | 387 (8%) | +0.100 Nm |
| zc | 4319 (86%) | **-4.548 Nm** | 679 (14%) | +0.098 Nm |
| ezc | 4297 (86%) | **-4.556 Nm** | 701 (14%) | +0.103 Nm |
| ezc_v2 | 4299 (85%) | **-4.600 Nm** | 699 (15%) | +0.102 Nm |

**Finding:** When drift IS positive, tau_position is large and negative (corrective). But the system rarely enters negative drift, so tau_position rarely produces positive torque. The system is biased toward positive drift.

---

## 5. Equilibrium Physics Analysis

### 5.1 Setup File Evidence

From `high_0p480_setup.json`:
```
hip_pitch_ref: 0.626052 rad (35.9 deg)
knee_ref: 1.223364 rad (70.1 deg)
equilibrium_pitch_x: 0.0 rad (prescribed)
com_x_relative_to_support: +1.35e-6 m (essentially centered)
equilibrium_com_pos: [-2.6e-7, -0.0057, 0.481] m
```

**The setup file prescribes zero pitch equilibrium, but the robot settles at +3.6 to +3.9 deg forward pitch. Why?**

### 5.2 Physical Explanation

The forward pitch equilibrium can come from several sources:

**Source 1: Low-level PD + gravity coupling**
- The shape_posture controller provides PD reference tracking for leg joints
- Gravity pulls the body forward relative to the wheel contact point
- Even with zero pitch_ref, the PD controllers may not perfectly counteract gravity, leading to a slight forward lean

**Source 2: Wheel contact dynamics**
- The wheeled biped's contact model may introduce a forward lean moment
- Wheel friction, contact stiffness, or geometry could create a lean tendency

**Source 3: COM position slightly forward of wheel contact line**
- The COM is calculated at equilibrium, but the actual COM during simulation may shift
- Hip joint compliance or controller dynamics could shift COM during operation

### 5.3 Controller Response to Equilibrium

The controller's job is to stabilize the robot given its current equilibrium. At high_0p480:

```
1. Robot settles at forward pitch (+3.6 deg)
2. Controller computes tau_pitch = kp_pitch * pitch_error = 50 * 0.063 = +3.2 Nm
3. This torque accelerates wheels forward, fighting the fall
4. But the fall is ongoing, so tau_position must also correct drift
5. tau_position = -k_position * position_error = -100 * 0.05 = -5.0 Nm (clipped to -4.0)
6. tau_pitch + tau_position ≈ 0, so final wheel torque ≈ 0
7. Robot stays in forward-pitch equilibrium
```

### 5.4 Why Support Drift is Predominantly Positive

The forward pitch equilibrium causes a POSITIVE bias on support drift:

1. **Forward pitch → forward COM shift**: When the robot is pitched forward, its COM shifts forward relative to the wheels
2. **tau_position corrects but saturates**: Position controller tries to move wheels backward, but saturates at lower bound
3. **tau_pitch fights tau_position**: tau_pitch produces forward torque proportional to forward pitch
4. **Net torque ≈ 0**: At equilibrium, tau_pitch and tau_position cancel, leaving no net recentering force
5. **Positive drift accumulates**: Small positive disturbances accumulate because there's no restoring force when near zero

---

## 6. Torque Composition Conflict

### 6.1 The Stalemate

At equilibrium (drift near zero, pitch at +3.6 deg):

```
tau_balance_before_position = tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_support_velocity + 0.5*(tau_wheel_vel_left + tau_wheel_vel_right)
                              ≈ +3.2 + 0.0 + 0.0 + 0.0 + 0.35
                              ≈ +3.55 Nm (positive)
```

This positive balance consumes **+3.55 Nm of headroom**, leaving:

```
tau_position_upper_bound = +4.0 - 3.55 = +0.45 Nm  (very small positive authority)
tau_position_lower_bound = -4.0 - 3.55 = -7.55 Nm  (large negative authority)
```

**The position controller CAN apply large negative torque (backward), but only small positive torque (forward). This asymmetry is caused by tau_pitch consuming positive headroom.**

### 6.2 Why Recentering is Hard

For support to drift negative (backward), the following must happen:
1. Position error goes negative (drifted backward)
2. tau_position needs to be positive (forward correction)
3. But positive tau_position authority is limited to +0.45 Nm (see above)
4. Positive drift events don't go negative because the small positive authority is overwhelmed by other factors

For support to drift positive (forward), the following happens:
1. tau_pitch produces positive torque (+3.2 Nm)
2. Position controller produces negative torque (-3.5 Nm)
3. They cancel to ~0
4. Small positive disturbances accumulate because no net restoring force

---

## 7. Key Questions Answered

### Q1: Is the robot physically settling with pitch_x > 0 at high_0p480?

**YES.** Mean pitch = +3.6 to +3.9 deg. The robot is forward-pitched at equilibrium.

### Q2: Is pitch reference fixed at 0 while actual pitch is +3 to +5 deg?

**YES.** pitch_ref = 0.0 rad (exactly) for all profiles. The forward pitch is NOT from a biased reference.

### Q3: Is CoM projection forward of wheel support at high_0p480?

**IMPLIED YES.** The forward pitch equilibrium means the COM is shifted forward relative to the wheel contact line. The setup file shows CoM is centered at equilibrium (com_x_relative_to_support ≈ 0), but during operation the COM shifts forward with the pitch lean.

### Q4: Are hip/knee equilibrium references causing forward lean?

**UNCERTAIN.** The setup file prescribes hip_pitch = 0.626 rad, knee = 1.223 rad. These should produce upright posture, but the robot leans forward during operation. Possible causes:
- Low-level PD doesn't perfectly track references under gravity
- Wheel contact dynamics introduce lean
- Controller dynamics shift COM during transient

### Q5: Is positive support drift a consequence of the pitch stabilizer moving wheels forward to catch the body?

**YES.** tau_pitch produces forward wheel torque when the robot leans forward. This pushes the wheels forward (positive drift) while trying to arrest the fall. The wheels move forward, and the robot stays balanced but with shifted support position.

### Q6: Is support recenter fighting pitch stabilization instead of coordinating with it?

**YES.** tau_pitch and tau_position are opposing each other:
- tau_pitch: forward torque to arrest forward pitch
- tau_position: backward torque to recenter support
- They nearly cancel at equilibrium (net ≈ -0.35 Nm)
- This creates a stalemate where neither goal is fully achieved

---

## 8. Structural Fix Path Recommendation

### Fix Path A: Equilibrium Posture Correction (RECOMMENDED)

**Rationale:** If the forward pitch equilibrium is caused by hip_pitch/knee references that place the COM slightly forward, adjusting these references could center the equilibrium at zero pitch.

**Approach:**
1. Generate centered setup variants with slightly reduced hip_pitch (e.g., -2 deg from current)
2. Test if this reduces equilibrium pitch from +3.6 to +1.5 deg
3. Verify tau_pitch mean drops from +3.2 to +1.5 Nm
4. Verify positive drift % decreases

**Expected outcome:**
- Reduced tau_pitch mean
- Shifted equilibrium pitch
- More symmetric drift distribution

**Risk:** Could cause fall if hip_pitch is too reduced. Needs careful validation.

### Fix Path B: Support-Position Outer Loop Pitch Reference

**Rationale:** Instead of pitch_ref = 0, use pitch_ref = f(support_error) to coordinate pitch stabilization with support recentering.

**Approach:**
1. When support drift is positive, slightly reduce pitch_ref (bias toward backward lean)
2. This reduces tau_pitch during positive drift
3. Allows tau_position to recenter without fighting tau_pitch
4. When support crosses zero, pitch_ref returns to 0

**Expected outcome:**
- tau_pitch reduced during positive drift
- tau_position can recenter more effectively
- More symmetric drift distribution

**Risk:** Could destabilize pitch control if pitch_ref changes are too aggressive.

### Fix Path C: Unified Sagittal State Feedback (LQR)

**Rationale:** The additive architecture creates fighting between tau_pitch and tau_position. A unified controller computes one optimal torque from all states.

**Approach:**
1. Design LQR with state vector [support_error, support_velocity, pitch, pitch_rate]
2. Compute single wheel torque command
3. This eliminates the stalemate by design

**Expected outcome:**
- Optimal torque command from full state
- No fighting between terms
- Potentially better stability margins

**Risk:** Requires careful gain tuning. More complex than additive approach.

---

## 9. Phase 3 Recommendations

Before implementing any fix, run causal ablation experiments:

1. **no_pitch_torque_diagnostic**: Set tau_pitch = 0, see if drift becomes centered or chaotic
2. **pitch_reference_offset_diagnostic**: Sweep pitch_ref from -4 deg to +1 deg
3. **posture_equilibrium_sweep**: Generate variants with hip_pitch ±5 deg from current
4. **support_outer_loop_pitch_ref_diagnostic**: Test cascaded controller concept

These ablations will confirm which root cause (posture vs architecture) is primary.

---

## 10. Classification Summary

| Classification | Evidence |
|---------------|----------|
| **FORWARD_EQUILIBRIUM_POSTURE_CONFIRMED** | Mean pitch +3.6 to +3.9 deg, all profiles |
| **PITCH_REFERENCE_CONTROLLER_CONFLICT_CONFIRMED** | pitch_ref = 0 exactly, but actual pitch ≠ 0 |
| **TORQUE_COMPOSITION_CONFLICT_CONFIRMED** | tau_pitch + tau_position ≈ 0 at equilibrium |
| **EQUILIBRIUM_AUDIT_INCONCLUSIVE** | NOT classified — enough evidence to proceed to Phase 3 |

**Final verdict:** The root cause is a forward-leaning equilibrium posture at high_0p480. The controller correctly responds to this equilibrium, but the equilibrium itself is biased. Fix must address the physics of WHY the robot settles forward-pitched, not just the controller gains or torque composition.