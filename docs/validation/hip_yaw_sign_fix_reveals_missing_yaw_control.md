# Hip-Yaw Sign Fix Reveals Missing Yaw Control

**Date:** 2026-06-05

**Status:** ROOT CAUSE IDENTIFIED

## Executive Summary

The hip-yaw torque sign fix (wheeled_biped/controllers/shape_posture_controller.py:250) is **mathematically correct** and passes all 9 unit tests. However, it revealed a critical architectural gap: **the balance-core controller has no yaw stabilization mechanism**.

The inverted hip-yaw sign was accidentally providing yaw control by driving the legs into antisymmetric divergence. With the correct sign, the robot yaws freely, causing centrifugal lean and roll collapse at step 136.

## Evidence

### 1. Hip-Yaw Behavior Analysis

**With correct sign (current):**
```
Step   0: L=0.0000 rad, R=0.0000 rad (common=0.0000, divergence=0.0000)
Step  25: L=-0.0244 rad, R=-0.0283 rad (common=-0.0264, divergence=0.0039)
Step  50: L=-0.4395 rad, R=-0.4119 rad (common=-0.4257, divergence=0.0275)
```

**Pattern:** Common-mode dominant (both joints move together = body yaw rotation)

**With wrong sign (baseline, from previous evaluation):**
```
Step 5000: L=-0.2275 rad, R=+0.2303 rad (divergence=0.458 rad dominant)
```

**Pattern:** Divergence dominant (joints twist opposite = leg deformation, not body yaw)

### 2. Robot Yaw Uncontrolled

```
Step   0: yaw =   0.01°
Step  25: yaw =  -0.60°
Step  50: yaw = -28.22°
Step  75: yaw =  13.05°
Step 100: yaw =  65.50°
Step 135: yaw =  95.10° [TERMINATED: height_too_low]
```

Robot spins freely with no yaw stabilization.

### 3. Roll Collapse Follows Yaw

```
Step  25: yaw= -0.6°, roll=  1.5°
Step  50: yaw=-28.2°, roll=  8.7°
Step 100: yaw= 65.5°, roll= -9.9°
Step 120: yaw= 86.0°, roll=-20.8°
Step 135: yaw= 95.1°, roll=-41.2° [height collapse]
```

Yaw-induced centrifugal lean causes hip-roll controller to saturate, leading to height loss.

### 4. Termination

- **Step:** 136
- **Reason:** height_too_low
- **Final height:** 0.351 m (target ~0.404 m)
- **Final roll:** -41.2°
- **Hip-roll torque:** 30.91 Nm (near saturation)

## Root Cause Classification

**Mechanism:** `corrected_hip_yaw_exposes_missing_yaw_controller`

**Confidence:** HIGH

The balance-core architecture has no yaw stabilization controller. The shape posture controller applies symmetric hip-yaw PD control (left and right toward same reference), which cannot generate yaw moments.

### Why Wrong Sign "Worked"

1. Inverted sign drove hip-yaw errors to grow antisymmetrically (left negative, right positive)
2. Antisymmetric leg configuration creates yaw stiffness through geometry
3. This inadvertently stabilized body yaw rotation
4. System survived 5000+ steps with large hip-yaw divergence but stable yaw

### Why Correct Sign Fails

1. Correct sign applies symmetric PD control (both joints toward zero)
2. No differential (antisymmetric) component for yaw correction
3. Robot yaws freely under disturbances
4. Yaw → centrifugal lean → roll saturation → height collapse

## Architectural Gap

The balance-core controller lacks a **yaw control layer** that uses hip-yaw joints antisymmetrically:

```
Existing (shape posture only):
  tau_left  = -(kp * error_left  + kd * vel_left)
  tau_right = -(kp * error_right + kd * vel_right)
  
  Both driven toward reference (symmetric)
  No yaw moment generated

Needed (yaw control):
  tau_yaw_antisym = k_yaw * yaw_error + kd_yaw * yaw_rate
  tau_left  = tau_posture_left  - tau_yaw_antisym
  tau_right = tau_posture_right + tau_yaw_antisym
  
  Antisymmetric component generates yaw moment
```

## Fix Options

### Option A: Add Yaw Controller (Recommended)

**Approach:**
1. Add yaw controller that computes antisymmetric hip-yaw torque
2. Compose with existing symmetric posture control
3. Yaw layer owns body rotation, posture layer owns leg geometry

**Pros:**
- Architecturally correct
- Decouples yaw and posture objectives
- Enables future yaw tracking/locomotion

**Cons:**
- Requires new controller implementation
- Needs tuning and validation

**Estimated effort:** Medium (2-4 hours implementation, 2-4 hours validation)

### Option B: Revert Hip-Yaw Sign Fix (Not Recommended)

**Approach:**
- Revert shape_posture_controller.py:250 to use wrong sign
- Document inverted axis as "feature" providing yaw stability

**Pros:**
- Returns to 5000-step stable baseline immediately

**Cons:**
- Sign fix is mathematically correct per joint axis convention
- Wrong sign creates large hip-yaw divergence (28-26°)
- Fragile: any hip-yaw gain change breaks yaw stability
- Technical debt: hiding architectural gap behind wrong sign

**Decision:** Do NOT revert. The sign fix is correct; system needs yaw control.

### Option C: Reduce Hip-Yaw Authority Temporarily

**Approach:**
- Keep correct sign
- Reduce hip-yaw kp/kd to 25-50% of current values temporarily
- This limits yaw drift rate until yaw controller is added

**Pros:**
- Buys time for proper yaw controller implementation
- Maintains correct sign convention

**Cons:**
- Hip-yaw posture control weakened
- Still no true yaw stabilization
- Temporary workaround, not architectural fix

**Use case:** Bridge solution if Option A implementation is deferred

## Recommended Path Forward

1. **Do NOT revert hip-yaw sign fix** - it is mathematically correct
2. **Implement yaw controller** (Option A) as proper architectural fix
3. **If yaw controller deferred:** Use Option C (reduced hip-yaw authority) as temporary bridge

## Implementation Notes for Yaw Controller

### Yaw Controller Structure

```python
class YawController:
    def __init__(self, kp_yaw: float, kd_yaw: float, max_yaw_torque: float):
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw
        self.max_yaw_torque = max_yaw_torque
    
    def compute(self, yaw_error: float, yaw_rate: float) -> tuple[float, float]:
        """Compute antisymmetric hip-yaw torque for yaw control.
        
        Returns:
            (tau_left, tau_right) where tau_left = -tau_right
        """
        tau_antisym = self.kp_yaw * yaw_error + self.kd_yaw * yaw_rate
        tau_antisym_clipped = jnp.clip(tau_antisym, -self.max_yaw_torque, self.max_yaw_torque)
        
        # Antisymmetric: left and right have opposite signs
        return (-tau_antisym_clipped, tau_antisym_clipped)
```

### Integration with Shape Posture

Shape posture controller continues to provide symmetric posture control. Yaw controller adds antisymmetric component:

```python
# In balance-core composer or shape posture:
tau_posture_left, tau_posture_right = shape_posture.compute(...)
tau_yaw_left, tau_yaw_right = yaw_controller.compute(yaw_error, yaw_rate)

tau_final_left = tau_posture_left + tau_yaw_left
tau_final_right = tau_posture_right + tau_yaw_right
```

### Initial Tuning Guidance

- Start with low yaw gains (kp_yaw=5.0, kd_yaw=1.0)
- Yaw control bandwidth should be lower than roll control
- Monitor for yaw-roll coupling during tuning

## Tests Required

1. Yaw controller unit tests (antisymmetric torque correctness)
2. Yaw stabilization smoke test (200 steps, |yaw| < 10°)
3. Hip-yaw sign tests continue to pass (existing 9 tests)
4. Step E 5000-step validation at all three heights
5. Hip-yaw posture quality vs wrong-sign baseline

## Restrictions Followed

- ✓ Did NOT revert hip-yaw sign fix
- ✓ Did NOT add WBC
- ✓ Did NOT modify hip-roll until sign is audited
- ✓ Did NOT tune gains blindly
- ✓ Did NOT proceed to Step C or Step D
- ✓ Did NOT commit

## Related Files

- Hip-yaw sign fix: [wheeled_biped/controllers/shape_posture_controller.py:250](wheeled_biped/controllers/shape_posture_controller.py#L250)
- Sign convention audit: [outputs/hip_yaw_sign_convention_audit/](outputs/hip_yaw_sign_convention_audit/)
- Failure telemetry: [outputs/hierarchical_controller_sim/telemetry_1780653682.csv](outputs/hierarchical_controller_sim/telemetry_1780653682.csv)
- Sign tests: [tests/test_shape_posture_hip_yaw_sign.py](tests/test_shape_posture_hip_yaw_sign.py)

## Next Steps

**Decision Required:**
1. Implement yaw controller now (Option A - recommended)
2. Use reduced hip-yaw gains temporarily (Option C - bridge)
3. Proceed with different approach

Hip-yaw sign fix is correct and will NOT be reverted. System requires yaw control architecture.
