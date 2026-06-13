# E2b vs D2/E1_after/E2 Comparison (low_0p300, 500 steps)

## Summary

E2b (0.12 rad gate + 5.0 Nm cap) produces virtually identical results to E2 (0.03 rad gate + 5.0 Nm cap). Both improve support but regress hip_yaw. The integral gate alignment does NOT fix hip_yaw regression.

## Comparison Table

| Metric | D2 | E1_before | E1_after | E2 | E2b |
|--------|------|-----------|----------|------|------|
| **Support Max (m)** | 0.1757 | 0.1757 | 0.1757 | **0.1703** | **0.1703** |
| **Support >0.15m** | 96 | 96 | 96 | **62** | **62** |
| **Support Mean (m)** | 0.0827 | 0.0827 | 0.0827 | **0.0677** | **0.0677** |
| **Support Final (m)** | 0.0580 | 0.0580 | 0.0579 | **0.0276** | **0.0276** |
| **Hip Yaw Max (rad)** | 0.1018 | 0.1018 | 0.1018 | 0.1304 | 0.1305 |
| **Hip Yaw Max (deg)** | 5.83° | 5.83° | 5.83° | 7.47° | 7.48° |
| **Hip Yaw >0.10rad** | 26 | 26 | 26 | 53 | 53 |
| **Divergence Max** | 0.1866 | 0.1866 | 0.1866 | 0.2434 | 0.2435 |
| **Tau Pos Raw Max (Nm)** | 7.0275 | 7.0275 | 7.0275 | 6.8111 | 6.8111 |
| **Integral Active %** | 0.0% | 4.4% | 7.8% | 6.2% | 9.0% |
| **Wheel Vel RMS (rad/s)** | 2.8207 | 2.8207 | 2.8214 | 3.2509 | 3.2509 |

## Key Findings

### Support Improvement (E2/E2b vs D2)
- Max: 0.1757 → 0.1703 m (-3.1%)
- Mean: 0.0827 → 0.0677 m (-18.1%)
- Final: 0.0580 → 0.0276 m (-52.4%)
- Crossings >0.15m: 96 → 62 (-35.4%)

### Hip Yaw Regression (E2/E2b vs D2)
- Max: 0.1018 → 0.1304 rad (+28.1%)
- Count >0.10 rad: 26 → 53 (+104%)
- Divergence: 0.1866 → 0.2434 (+30.4%)

### E2b vs E2 (Gate Effect)
- Only difference: integral active 6.2% → 9.0%
- Hip yaw: identical (within numerical precision)
- Support: identical
- **Conclusion: Gate change has no effect on hip_yaw**

## Root Cause Analysis

The 5.0 Nm cap is the common factor between E2 and E2b's hip_yaw regression. The cap changes the torque distribution in a way that couples to hip_yaw through kinematic coupling.

## Next Candidate: E2c

Recommended parameters:
- max_position_tau_low_max = 4.5 Nm
- integral_pitch_error_threshold_rad = 0.12

Hypothesis: Lowering the cap while keeping the aligned gate may preserve some support improvement while reducing hip_yaw regression.
