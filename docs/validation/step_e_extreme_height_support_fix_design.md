# Step E Extreme Height Support/Wheel Fix Design

## Context

After WBC gate fix, official Step E failures remain at extreme heights:

**low_0p300 (0.300m)**:
- support_position_error = 0.176 m > 0.15 m (PRIORITY 1)
- hip_yaw = 0.313 rad > 0.10 rad (PRIORITY 3 - fix after support)
- wheel_velocity = 4.39 rad/s PASS

**high_0p480 (0.480m)**:
- support_position_error = 0.173 m > 0.15 m (PRIORITY 1)
- hip_yaw = 0.275 rad > 0.10 rad (PRIORITY 3)
- wheel_velocity = 5.26 rad/s > 5.0 rad/s (PRIORITY 2)

Event order evidence:
- low_0p300: support drift at step 91, hip-yaw at step 328
- high_0p480: wheel velocity spike at step 73, support drift at step 108, hip-yaw at step 2426

Root cause evidence:
- position authority saturates at 4.0 Nm
- position integral is disabled
- high_0p480 wheel velocity spike is transient (step 73)

## Design Principles

1. **Do NOT modify protected D2 default baseline**
2. **Fix support_position_error first** (Priority 1)
3. **Fix high-height wheel_velocity transient if needed** (Priority 2)
4. **Do NOT fix hip-yaw first** - it appears after support/wheel issues
5. **Opt-in profiles only** - candidates must be explicitly selected via `--vd-sagittal-authority-profile`
6. **No HY2-DIV** - not in this phase
7. **Continuous height scheduling** - preferred over variant-name patches

## Candidate Profiles

### Baseline: D2 (protected)
```yaml
profile_name: "candidate_D2_wheel_velocity_damping_light"
applies_to_variants: D2_HEIGHT_VARIANTS  # includes low_0p300, high_0p480
position_tau_cap_by_variant: 4.0 Nm for all D2 variants
velocity_damping_scale: 1.10
pitch_tau_scale: 1.0
support_velocity_gain: None
continuous_k_position: false
continuous_max_position_tau: false
continuous_k_wheel_velocity: false
enable_position_integral: false (default)
```

### Candidate E1: Support Integral Correction (minimal risk)
```yaml
profile_name: "candidate_E1_support_integral"
applies_to_variants: EXTREME_HEIGHT_VARIANTS  # low_0p300, high_0p480 only

# Enable position integral
enable_position_integral: true (via controller constructor)
ki_position_integral: 2.0  # Small integral gain
integral_max_abs: 1.0  # Nm, anti-windup limit
integral_pitch_error_threshold_rad: 0.03  # Gate: pitch must be small
integral_support_velocity_threshold_m_s: 0.03  # Gate: support must be slow
integral_wheel_velocity_threshold_rad_s: 1.0  # Gate: wheels must be slow
integral_min_com_z_m: 0.28  # Safety: height must be in range
integral_max_com_z_m: 0.50

# Keep position cap at 4.0 Nm
continuous_max_position_tau: true
max_position_tau_nominal: 4.0  # Same as D2
max_position_tau_low_max: 4.0  # No change for E1

# No wheel damping change
continuous_k_wheel_velocity: false
velocity_damping_scale: 1.10  # Same as D2
```

**Rationale**: Small integral term eliminates steady-state position error without changing cap. Gate conditions prevent integral windup during transients.

**Anti-windup rule**:
- Integral only accumulates when ALL gates pass (pitch small, support slow, wheels slow, height safe)
- Integral contribution clamped to ±1.0 Nm
- When gates fail, integral resets to 0.0

### Candidate E2: Support Integral + Increased Position Cap
```yaml
profile_name: "candidate_E2_support_integral_higher_cap"
applies_to_variants: EXTREME_HEIGHT_VARIANTS

# Enable position integral (same as E1)
enable_position_integral: true
ki_position_integral: 2.0
integral_max_abs: 1.0
integral_pitch_error_threshold_rad: 0.03
integral_support_velocity_threshold_m_s: 0.03
integral_wheel_velocity_threshold_rad_s: 1.0
integral_min_com_z_m: 0.28
integral_max_com_z_m: 0.50

# Increase position cap to 5.0 Nm
continuous_max_position_tau: true
max_position_tau_nominal: 4.0  # Same as D2 for nominal heights
max_position_tau_low_max: 5.0  # 25% increase for extreme heights

# No wheel damping change
continuous_k_wheel_velocity: false
velocity_damping_scale: 1.10
```

**Rationale**: 4.0 Nm cap may be insufficient for extreme heights. 5.0 Nm provides 25% more authority without aggressive change.

**Risk**: Higher cap could cause pitch overshoot. Mitigated by integral being the primary fix.

### Candidate E3: E2 + High-Height Wheel Damping
```yaml
profile_name: "candidate_E3_support_integral_cap_wheel_damping"
applies_to_variants: EXTREME_HEIGHT_VARIANTS

# Enable position integral (same as E1/E2)
enable_position_integral: true
ki_position_integral: 2.0
integral_max_abs: 1.0
integral_pitch_error_threshold_rad: 0.03
integral_support_velocity_threshold_m_s: 0.03
integral_wheel_velocity_threshold_rad_s: 1.0
integral_min_com_z_m: 0.28
integral_max_com_z_m: 0.50

# Increase position cap (same as E2)
continuous_max_position_tau: true
max_position_tau_nominal: 4.0
max_position_tau_low_max: 5.0

# Add high-height wheel damping
continuous_k_wheel_velocity: true
k_wheel_velocity_nominal: 0.5  # Current default (matches D2 effective: 0.5 * 1.10 = 0.55)
k_wheel_velocity_high_max: 0.75  # 50% increase for high heights
k_wheel_velocity_z_low: 0.45  # Start increasing above 0.45m
k_wheel_velocity_z_high: 0.52  # Full increase at 0.52m

velocity_damping_scale: 1.0  # Not used when continuous_k_wheel_velocity is true
```

**Rationale**: high_0p480 wheel velocity spike at step 73 is transient. Higher k_wheel_velocity at high heights will damp this spike.

**Height schedule logic**:
```
if z_ref >= k_wheel_velocity_z_high: k_wheel_velocity = k_wheel_velocity_high_max
elif z_ref <= k_wheel_velocity_z_low: k_wheel_velocity = k_wheel_velocity_nominal
else: interpolate smoothly between
```

### Candidate E4: Minimal Combined (deferred)
Only evaluate if E1/E2/E3 partially improve but don't fully pass.

## SagittalAuthoritySchedule Extensions

Need to add fields to `SagittalAuthoritySchedule`:

```python
@dataclass(frozen=True)
class SagittalAuthoritySchedule:
    # ... existing fields ...

    # Position integral settings (for opt-in profiles)
    enable_position_integral: bool = False
    ki_position_integral: float = 0.0
    integral_max_abs: float = 1.0
    integral_pitch_error_threshold_rad: float = 0.03
    integral_support_velocity_threshold_m_s: float = 0.03
    integral_wheel_velocity_threshold_rad_s: float = 1.0
    integral_min_com_z_m: float = 0.28
    integral_max_com_z_m: float = 0.50

    # High-height wheel velocity damping
    continuous_k_wheel_velocity: bool = False
    k_wheel_velocity_nominal: float = 0.5
    k_wheel_velocity_high_max: float = 0.75
    k_wheel_velocity_z_low: float = 0.45
    k_wheel_velocity_z_high: float = 0.52
```

## Telemetry Fields

### Position Integral
- `position_integral_enabled`: bool
- `position_integral_error`: float (accumulated error)
- `tau_position_integral`: float (integral torque contribution)
- `tau_position_integral_clipped`: float (after anti-windup)
- `position_integral_active_ratio`: float (0.0 to 1.0, fraction of steps integral was active)
- `integral_gate_reason`: str (why integral was/wasn't active)
- `integral_saturation_flag`: bool

### Position Cap
- `effective_max_position_tau`: float
- `tau_position_raw`: float (before cap)
- `tau_position_final`: float (after cap)
- `tau_position_saturated`: bool
- `tau_position_saturation_percent`: float (how close to saturation)

### High-Height Wheel Damping
- `high_height_wheel_damping_active`: bool
- `high_height_damping_gate`: str ("none", "active", "inactive")
- `effective_k_wheel_velocity`: float
- `wheel_velocity_damping_term`: float

## Pass/Fail Gates

### Stage 1 (100-step smoke)
- survived: True
- contact_valid_percent_raw >= 99.9%
- non_wheel_floor_contacts = 0
- hidden_torque = 0
- ownership = 0
- WBC gate pass
- no roll collapse (|roll_y| < 0.2 rad)
- no large height collapse (com_z > 0.25 m)

### Stage 2 (500-step validation)
- support_position_error does not worsen by >20%
- wheel_velocity does not worsen by >20%
- hip_yaw does not worsen by >50%
- no structural failures

### Stage 3 (2000-step screening)
- support_position_error improves toward <0.15 m
- high_0p480 wheel_velocity improves toward <5.0 rad/s
- hip_yaw does not worsen
- height/contact/roll remain valid

### Stage 4 (5000-step official Step E)
- support_position_error_max_abs < 0.15 m
- wheel_vel_mean_max_abs < 5.0 rad/s
- hip_yaw_abs_max < 0.10 rad
- contact_valid_percent_raw >= 99.9%
- non_wheel_floor_contacts = 0
- WBC gate pass
- hidden_torque = 0
- ownership = 0
- survived 5000

## Rollback Rule

If at any stage:
- WBC gate fails
- contact_valid drops below 99.9%
- non_wheel_floor_contacts > 0
- hidden_torque > 0
- ownership > 0
- roll collapse detected

Then: reject candidate, do not proceed to next stage.

## Regression Testing

For any candidate that passes or partially improves:
- Run old five variants at 5000-step: low_small, low_tiny, nominal, high_tiny, high_small
- Candidate must not regress:
  - support_position_error gate
  - hip_yaw gate
  - wheel_velocity gate
  - contact/WBC/hidden_torque/ownership

## Files to Modify

1. `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
   - Add fields to `SagittalAuthoritySchedule`
   - Add continuous k_wheel_velocity scheduling in `compute()`
   - Position integral already implemented (enable_position_integral in __init__)

2. `scripts/simulate_hierarchical_controller.py`
   - Add E1, E2, E3 profiles to `SAGITTAL_AUTHORITY_PROFILES`
   - Pass position integral parameters to controller constructor when profile selected

3. `tests/test_sagittal_velocity_damped_balance_controller.py`
   - Add tests for E1/E2/E3 profile behavior
   - Test integral anti-windup
   - Test high-height wheel damping

## Do NOT Include

- HY2-DIV (separate phase)
- hip_yaw fix (Priority 3, after support/wheel)
- WBC changes (already fixed)
- modifications to candidate_D2_wheel_velocity_damping_light
- changes to default controller behavior