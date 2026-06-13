# Hip-Yaw Support-Error Feedforward (HY-FF) - Phase 4 Implementation Plan

**Date:** 2026-06-04  
**Phase:** 4 (Implement HY-FF Candidate)  
**Status:** IN PROGRESS

---

## Implementation Requirements

### 1. Continuous Height Gating (NOT variant-based)

```python
def compute_height_gate(z_ref: float) -> float:
    """Smooth height-based activation gate.
    
    Returns 1.0 at z <= 0.300 (full compensation)
    Returns 0.0 at z >= 0.393 (no compensation)
    Smoothstep transition between.
    """
    z_low = 0.300
    z_high = 0.393
    
    u = jnp.clip((z_high - z_ref) / (z_high - z_low), 0.0, 1.0)
    s = 3 * u**2 - 2 * u**3  # smoothstep
    
    return s
```

### 2. Support-Error Compensation (Sign to be determined)

```python
# Candidate A: Direct sign
tau_comp_left  = +sign * k_support * support_error * height_gate
tau_comp_right = -sign * k_support * support_error * height_gate

# Candidate B: Opposite sign
tau_comp_left  = -sign * k_support * support_error * height_gate
tau_comp_right = +sign * k_support * support_error * height_gate
```

### 3. Compensation Clamping

```python
tau_comp_left_clamped = jnp.clip(tau_comp_left, -tau_max, tau_max)
tau_comp_right_clamped = jnp.clip(tau_comp_right, -tau_max, tau_max)
```

### 4. Required Telemetry

**New columns:**
- `hip_yaw_comp_active` (bool)
- `hip_yaw_comp_height_gate` (float)
- `hip_yaw_comp_support_error_m` (float)
- `hip_yaw_comp_tau_left` (float, pre-clamp)
- `hip_yaw_comp_tau_right` (float, pre-clamp)
- `hip_yaw_comp_tau_left_clipped` (bool)
- `hip_yaw_comp_tau_right_clipped` (bool)
- `hip_yaw_comp_sign` (float)
- `l_hip_yaw_tau_shape_final` (float, post-comp)
- `r_hip_yaw_tau_shape_final` (float, post-comp)

**Existing columns to preserve:**
- `hip_yaw_abs_max`
- `support_position_error_m`
- `pitch_x`
- `roll_y`
- All WBC/ownership diagnostics

---

## Candidate Evaluation Matrix

### Parameter Sweep

| Candidate | Sign | k_support | tau_max | Description |
|-----------|------|-----------|---------|-------------|
| A (baseline) | N/A | 0.0 | N/A | No compensation |
| B | +1 | 2.0 | 1.0 | Sign A, conservative gain |
| C | -1 | 2.0 | 1.0 | Sign B, conservative gain |
| D | best | 4.0 | 1.0 | Best sign, moderate gain |
| E | best | 6.0 | 2.0 | Best sign, higher gain |
| F | best | 8.0 | 2.0 | Best sign, aggressive gain |
| G | best | varies | varies | Optional velocity term if needed |

### Test Matrix per Candidate

**Short validation (1000 steps):**
- low_0p300
- high_0p480
- nominal

**Long validation (5000 steps, only if short passes):**
- low_0p300
- high_0p480
- nominal

**Regression suite (only if long passes):**
- nominal
- low_tiny
- high_tiny
- low_small
- high_small

---

## Acceptance Criteria

A candidate PASSES if ALL conditions met:

### Primary Gate
- `hip_yaw_abs_max <= 0.07 rad` ✅
- `percent(hip_yaw_abs_max > 0.10 rad) == 0%` ✅

### No Degradation
- `support_position_error_max` not worsened >10% vs baseline ✅
- `pitch_x_max <= 0.10 rad` ✅
- `roll_y_max <= 0.05 rad` ✅
- `height_error_final <= 0.02 m` ✅

### Safety Gates
- `contact_valid_percent >= 99.9%` ✅
- `non_wheel_contact_count == 0` ✅
- `wbc_applied_any == False` ✅
- `hidden_torque_detected == False` ✅
- `ownership_violations == 0` ✅

---

## Implementation Steps

### Step 4.1: Add HY-FF to ShapePostureController ✅ READY
- Add `k_support_hip_yaw`, `tau_max_support_comp` parameters
- Add `enable_support_feedforward` flag
- Implement height gate function
- Implement compensation computation
- Implement clamping
- Add to constructor and compute method

### Step 4.2: Add CLI Arguments ✅ READY
- `--enable-hip-yaw-support-feedforward`
- `--hip-yaw-support-k` (default: 0.0)
- `--hip-yaw-support-tau-max` (default: 1.0)
- `--hip-yaw-support-sign` (default: +1.0)

### Step 4.3: Add Telemetry ✅ READY
- Extend telemetry dict with 10 new columns
- Log raw compensation, clamped compensation, gate value
- Preserve all existing diagnostics

### Step 4.4: Create Evaluation Harness 🔄 IN PROGRESS
- `scripts/evaluate_hip_yaw_hy_ff_candidates.py`
- Automated sweep over candidates A-F
- Automatic pass/fail determination
- Generate comparison report

### Step 4.5: Run Short Validation 📋 PENDING
- Execute candidates B, C at low_0p300
- Determine correct sign
- Proceed to D-F with best sign

### Step 4.6: Run Long Validation 📋 PENDING
- Best candidate from short validation
- 5000-step runs at 3 heights

### Step 4.7: Run Regression Suite 📋 PENDING
- Best candidate across 5 variants
- Full acceptance criteria check

---

## Expected Outcomes

### Outcome 1: HY-FF Passes ✅
**Report:** `HIP_YAW_FIXED_SUPPORT_STILL_FAILS`  
**Action:** Return to sagittal support drift fix separately

### Outcome 2: HY-FF Worsens Support/Pitch ⚠️
**Report:** `HIP_YAW_FIX_CAUSED_POSITION_REGRESSION`  
**Action:** STOP. Do not stack more compensation.

### Outcome 3: No HY-FF Keeps hip-yaw <= 0.07 ❌
**Report:** `HIP_YAW_AND_SUPPORT_COUPLED_NEED_JOINT_FIX`  
**Action:** Next plan must be coupled sagittal-yaw controller.

---

## Files to Create/Modify

### New Files:
- `docs/validation/hip_yaw_hy_ff_implementation_plan.md` (this file)
- `scripts/evaluate_hip_yaw_hy_ff_candidates.py`
- `docs/validation/hip_yaw_hy_ff_evaluation_report.md` (after Phase 5)

### Modified Files:
- `wheeled_biped/controllers/shape_posture_controller.py`
- `scripts/simulate_hierarchical_controller.py`
- `tests/test_shape_posture_controller.py` (Phase 6)

---

## Restrictions Confirmed

❌ Do NOT add WBC  
❌ Do NOT enable legacy WBC  
❌ Do NOT modify hip-roll  
❌ Do NOT globally change hip-yaw gains  
❌ Do NOT use variant-name-only patches  
❌ Do NOT use discontinuous step schedules  
❌ Do NOT relax thresholds  
❌ Do NOT proceed to Step D  
❌ Do NOT make HY-FF permanent unless validation passes  

✅ Use continuous height gating  
✅ Implement as controlled candidate  
✅ Comprehensive telemetry  
✅ Systematic evaluation  
✅ Clear pass/fail criteria  

---

**Phase 4 Status:** Step 4.1 starting now
