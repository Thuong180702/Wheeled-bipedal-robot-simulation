# Hip-Yaw Disturbance Rejection Isolation Experiments - Results

**Date:** 2026-06-04
**Phase:** 2 (Isolation Experiments)

## Summary

- Total experiments: 21
- Successful: 21
- Passing hip-yaw gate (<= 0.07 rad): 8

## Best Hip-Yaw Result

- **Experiment:** D_damping_sweep
- **Variant:** high_0p480
- **Parameters:** kp=15.0, kd=9.0
- **hip_yaw_abs_max:** 0.0382 rad
- **support_error:** 0.2800 m
- **pitch_x:** 0.0929 rad

## Candidates Passing Hip-Yaw Gate ✅

| Experiment | Variant | kp | kd | hip_yaw | support | pitch |
|------------|---------|----|----|---------|---------|-------|
| baseline | high_0p480 | 15 | 3 | 0.0462 | 0.2336 | 0.0926 |
| baseline | nominal | 15 | 3 | 0.0392 | 0.1026 | 0.0706 |
| D_damping_sweep | high_0p480 | 15 | 5 | 0.0402 | 0.2520 | 0.0927 |
| D_damping_sweep | high_0p480 | 15 | 7 | 0.0388 | 0.2692 | 0.0928 |
| D_damping_sweep | high_0p480 | 15 | 9 | 0.0382 | 0.2800 | 0.0929 |
| D_damping_sweep | nominal | 15 | 5 | 0.0432 | 0.1025 | 0.0705 |
| D_damping_sweep | nominal | 15 | 7 | 0.0441 | 0.1025 | 0.0705 |
| D_damping_sweep | nominal | 15 | 9 | 0.0420 | 0.1025 | 0.0705 |

## Baseline Comparison

| Variant | hip_yaw | support | pitch | roll |
|---------|---------|---------|-------|------|
| low_0p300 | 0.2137 | 0.2430 | 0.0951 | 0.0150 |
| high_0p480 | 0.0462 | 0.2336 | 0.0926 | 0.0023 |
| nominal | 0.0392 | 0.1026 | 0.0706 | 0.0110 |

## Experiment D: Damping Sweep Results

### low_0p300

| kd | hip_yaw_abs_max | support_error | Status |
|----|----------------|---------------|--------|
| 5 | 0.2080 | 0.2425 | ❌ FAIL |
| 7 | 0.2037 | 0.2420 | ❌ FAIL |
| 9 | 0.2007 | 0.2416 | ❌ FAIL |
| 12 | 0.3786 | 0.3041 | ❌ FAIL |

### high_0p480

| kd | hip_yaw_abs_max | support_error | Status |
|----|----------------|---------------|--------|
| 5 | 0.0402 | 0.2520 | ✅ PASS |
| 7 | 0.0388 | 0.2692 | ✅ PASS |
| 9 | 0.0382 | 0.2800 | ✅ PASS |
| 12 | 0.0908 | 0.4191 | ❌ FAIL |

### nominal

| kd | hip_yaw_abs_max | support_error | Status |
|----|----------------|---------------|--------|
| 5 | 0.0432 | 0.1025 | ✅ PASS |
| 7 | 0.0441 | 0.1025 | ✅ PASS |
| 9 | 0.0420 | 0.1025 | ✅ PASS |
| 12 | 0.0994 | 0.1234 | ❌ FAIL |

## Experiment E: kp/kd Matrix (low_0p300)

| kp \ kd | 5 | 7 | 9 |
|---------|---|---|---|
| 15 | 0.208 | 0.204 | 0.201 |
| 20 | 0.180 | 0.180 | 0.177 |
| 25 | 0.162 | 0.162 | 0.162 |
