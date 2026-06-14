# T6H/T6I Implementation Plan

**Date**: 2026-06-12  
**Task**: Implement T6H_soft_blend_arch_fix and T6I_phase_aware_release

---

## Implementation Steps

### Step 1: Add T6H fields to SagittalAuthoritySchedule class

Around line 448, after sign_fix fields, add:

```python
# T6H Soft Blend Arch Fix: Soft modulation instead of hard suppression
# Reduces pitch/damping authority by 50% (not 100%) during arch_fix
# Preserves partial stabilization while reducing fighting terms
t6h_enabled: bool = False
t6h_soft_pitch_blend_factor: float = 0.50  # Pitch scale during blend (0.5 = 50% reduction)
t6h_soft_damping_blend_factor: float = 0.50  # Damping scale during blend
t6h_pitch_error_threshold_m: float = 0.10  # Apply blend when |error| > this
t6h_pitch_safety_threshold_deg: float = 10.0  # Restore full pitch if |pitch| > this
t6h_wheel_velocity_safety_threshold_rad_s: float = 7.0  # Restore full damping if |wheel_vel| > this
```

### Step 2: Add T6I fields to SagittalAuthoritySchedule class

```python
# T6I Phase-Aware Release: Gradual cap decay when error converging
# Detects convergence and releases high authority smoothly
# Preserves full pitch/damping authority (no suppression)
t6i_enabled: bool = False
t6i_convergence_window_steps: int = 5  # Steps to track for convergence detection
t6i_convergence_threshold_m: float = 0.12  # Error must be below this
t6i_convergence_trend_threshold_m: float = 0.03  # Max error change to be converging
t6i_cap_decay_rate_nm_per_step: float = 0.10  # Cap decay rate when converging
t6i_cap_min_nm: float = 4.0  # Min cap (normal authority)
t6i_max_cap_delta_per_step_nm: float = 0.30  # Rate limit for cap transitions
```

### Step 3: Create T6H profile definition

After T6F_SIGN_CORRECTED definition (around line 1450), add T6H_SOFT_BLEND_ARCH_FIX.

### Step 4: Create T6I profile definition

After T6H definition, add T6I_PHASE_AWARE_RELEASE.

### Step 5: Register profiles in AUTHORITY_SCHEDULES dict

Add entries for T6H and T6I.

### Step 6: Implement T6H logic in compute_sagittal_balance_torques

Around line 2240 (after arch_fix_active is set), add T6H soft blend logic for pitch.
Around line 2590, add T6H soft blend logic for damping.

### Step 7: Implement T6I logic in compute_sagittal_balance_torques

After arch_fix logic (around line 2240), add T6I convergence detection and cap decay.

### Step 8: Add telemetry fields

Add T6H and T6I telemetry fields to the telemetry dict.

### Step 9: Add tests

Update test files to verify T6H and T6I behavior.

---

## Implementation Details

See design documents:
- docs/validation/t6h_t6i_safe_next_candidates_design.md
- docs/validation/post_t6f_signfix_next_step_recommendation.md

---

**End of Plan**
