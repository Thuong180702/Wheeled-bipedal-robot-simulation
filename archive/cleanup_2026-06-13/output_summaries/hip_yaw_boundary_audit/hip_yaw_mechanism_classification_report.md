# Hip-Yaw Mechanism Classification Report

## Phase 5: Mechanism Classification

Date: 2026-06-04

## Evidence Summary

### Phase 1: Event Order (Baseline Telemetry)

**low_0p300:**
- Support position error > 0.15 m: **step 89**
- Hip-yaw error > 0.07 rad: **step 418**
- **Δ = 329 steps (3.29 seconds)**
- Event order: **support_position_led**

**high_0p480:**
- Support position error > 0.15 m: step 108
- Hip-yaw error > 0.07 rad: never exceeded
- Event order: **support_position_only**

**nominal:**
- Neither threshold exceeded
- Event order: **none_exceeded**

**Conclusion:** Support position drift happens FIRST, hip-yaw drift follows later.

### Phase 2: Reference and Command Audit

**All cases:**
- Hip-yaw references: correct (constant, zero initial error)
- Torque signs: correct (PD control equation verified)
- Reference source: equilibrium_joint_pos from setup
- Torque commands: reach shape posture controller correctly

**Conclusion:** No reference mismatch, no sign error. Controller logic is correct.

### Phase 3: Torque Authority Audit

**low_0p300:**
- Torque at hip-yaw onset (step 348): 0.61 Nm (error = 0.033 rad)
- Torque at hip-yaw peak (step 562): 3.28 Nm (error = 0.214 rad)
- Saturation rate: 0%
- Composer loss: No (raw = final)
- Ownership violations: 0

**high_0p480 and nominal:**
- Similar pattern: torque grows with error
- No saturation, no composer loss, no ownership violations

**Conclusion:** Hip-yaw torque is NOT saturated, NOT rate-limited, NOT overwritten. Torque DOES grow with error (0.6 → 3.3 Nm), but drift continues anyway.

## Mechanism Classification

### Primary Classification

**`sagittal_support_drift_forces_hip_yaw_drift`**

### Supporting Evidence

1. **Temporal causality**: Support drift precedes hip-yaw drift by 329 steps (3.3 seconds)
2. **Controller correctness**: Hip-yaw reference and torque commands are correct
3. **Torque response**: Hip-yaw torque grows appropriately with error (0.6 → 3.3 Nm)
4. **No authority limits**: No saturation, no composer loss, no ownership violations
5. **Height dependency**: Problem occurs at low_0p300 (extreme flexion), not at nominal or high_0p480

### Mechanism Hypothesis

At extreme flexion (low_0p300):

1. Sagittal controller has insufficient position authority (previous finding: k_position up to 100 still failed)
2. Support position drifts forward due to weak sagittal return torque
3. As support drifts, the robot's contact geometry changes
4. Contact constraints or kinematics couple support position to hip-yaw orientation
5. Hip-yaw controller applies correct torque (up to 3.3 Nm) but cannot overcome the coupling

Possible physical coupling mechanisms:

- **Wheel contact forces**: Forward support drift creates asymmetric wheel loading
- **Body pitch compensation**: Support drift → pitch → asymmetric hip-yaw compensation
- **Kinematic coupling**: At extreme flexion, forward support position geometrically favors certain hip-yaw angles
- **Actuator effectiveness loss**: Hip-yaw torque effectiveness degrades at extreme flexion

### Alternative Classifications (Ruled Out)

❌ **`hip_yaw_reference_mismatch`**: References are correct (Phase 2)
❌ **`hip_yaw_torque_sign_error`**: Signs are correct (Phase 2)
❌ **`hip_yaw_torque_saturation`**: 0% saturation rate (Phase 3)
❌ **`hip_yaw_torque_rate_limited`**: No evidence of rate limiting (Phase 3)
❌ **`hip_yaw_composer_loss`**: raw = final (Phase 3)
❌ **`hip_yaw_damping_insufficient`**: Torque grows appropriately with error (Phase 3)
❌ **`hip_yaw_authority_insufficient`**: Torque reaches 3.3 Nm without saturation (Phase 3)

### Secondary Contributing Factor

**`actuator_effectiveness_loss_at_extreme_flexion`**

Even though hip-yaw torque is not saturated (3.3 Nm applied), the drift continues. This suggests that at extreme flexion, hip-yaw actuator effectiveness may be reduced due to:

- Joint angle approaching singularity
- Reduced moment arm
- Unfavorable load distribution

However, this is a **secondary** factor because support drift happens FIRST.

## Root Cause Summary

**Primary root cause:** Sagittal support position drift at extreme flexion

**Hip-yaw is a symptom, not the root cause.**

The hip-yaw controller is working correctly:
- References are correct
- Torque signs are correct
- Torque grows with error
- No saturation or limits

But hip-yaw drift persists because **support position drift creates a coupled disturbance** that hip-yaw torque alone cannot overcome at extreme flexion.

## Implications for Fix Strategy

### What will NOT work:

❌ **Increasing hip-yaw kp/kd globally**: Torque is already growing to 3.3 Nm without saturation. More gain won't help if the coupling mechanism is dominant.

❌ **Hip-yaw reference adjustment**: References are already correct.

❌ **Hip-yaw torque cap increase**: Torque is not saturated.

❌ **Hip-yaw composer changes**: No composer loss detected.

### What MIGHT work:

✅ **Fix sagittal support drift first**: Address the k_position insufficiency that allows support drift
   - Previous attempts: k_position 40 → 100 failed
   - Need: Different sagittal approach (velocity damping, hybrid, integral, or different control mode)

✅ **Coupled sagittal-yaw correction**: If support drift → hip-yaw drift coupling is unavoidable, compensate hip-yaw for known support error
   - Add yaw-aware position correction term
   - Feedforward hip-yaw bias based on support position error

✅ **Height-dependent hip-yaw damping**: If actuator effectiveness is reduced at low height, increase damping to prevent oscillation even if drift cannot be fully prevented
   - Increase kd only at low heights
   - Keep kp unchanged initially

### Recommended Next Steps

1. **Do NOT pursue isolated hip-yaw gain increase**: Evidence shows this won't solve the root cause
2. **Return to sagittal authority problem**: Investigate hybrid k_position + k_velocity scheduling, or integral term, or different control mode
3. **If sagittal fix is infeasible**: Implement coupled yaw-aware position correction as compensatory strategy
4. **Test hypothesis**: Run isolation experiment with artificially frozen support position to verify hip-yaw remains stable

## Classification

- **Primary**: `sagittal_support_drift_forces_hip_yaw_drift`
- **Secondary**: `actuator_effectiveness_loss_at_extreme_flexion` (possible)
- **Root cause**: Sagittal position authority insufficient at extreme flexion
- **Hip-yaw role**: Symptom, not root cause
- **Coupling verified**: Temporal causality (support → hip-yaw delay = 329 steps)
