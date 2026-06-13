# D2 Low 0p300 Initial Condition Fix 500-Step Comparison

## Summary

| Metric | Old D2 | New D2 | Improvement |
|--------|--------|--------|-------------|
| hip_pitch_error_max (rad) | 0.4500 | 0.0000 | 0.4500 |
| hip_pitch_error_left (rad) | -0.4500 | 0.0000 | -0.4500 |
| tau_pitch_mean (Nm) | 2.5992 | 2.5992 | 0.0000 |
| tau_pitch_positive_pct | 89.2% | 89.2% | 0.0% |
| tau_position_saturation_pct | 35.4% | 35.4% | 0.0% |
| survived_500 | True | True | - |
| pitch_x_max (rad) | 0.1111 | 0.1111 | 0.0000 |

## Initial State

### Before Fix (Old D2)
- hip_pitch_error_max: 0.4500 rad (25.79 deg)
- hip_pitch_error_left: -0.4500 rad
- hip_pitch_error_right: -0.4500 rad
- knee_error_max: 0.6000 rad
- pitch_x at step 0: 0.0000 rad
- com_z at step 0: 0.2955 m

### After Fix (New D2)
- hip_pitch_error_max: 0.0000 rad (0.00 deg)
- hip_pitch_error_left: 0.0000 rad
- hip_pitch_error_right: 0.0000 rad
- knee_error_max: 0.0000 rad
- pitch_x at step 0: 0.0000 rad
- com_z at step 0: 0.2955 m

## Tau Pitch

### Before Fix (Old D2)
- mean: 2.5992 Nm
- max: 5.5527 Nm
- positive%: 89.2%

### After Fix (New D2)
- mean: 2.5992 Nm
- max: 5.5527 Nm
- positive%: 89.2%

## Tau Position

### Before Fix (Old D2)
- mean: -2.6146 Nm
- saturation%: 35.4%

### After Fix (New D2)
- mean: -2.6146 Nm
- saturation%: 35.4%

## Stability

### Before Fix (Old D2)
- survived 500: True
- pitch_x_max: 0.1111 rad (6.36 deg)
- roll_y_max: 0.0133 rad

### After Fix (New D2)
- survived 500: True
- pitch_x_max: 0.1111 rad (6.36 deg)
- roll_y_max: 0.0133 rad

## Conclusion

The initialization fix successfully eliminates the initial hip_pitch_error mismatch:
- hip_pitch_error_max went from 0.4500 rad to 0.0000 rad
- This is a 100.0% reduction

The tau_pitch positive bias is unchanged/increased (0.0% change).
