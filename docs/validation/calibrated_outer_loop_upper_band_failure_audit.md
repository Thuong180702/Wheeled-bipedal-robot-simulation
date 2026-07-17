# Calibrated Outer Loop Upper-Band Failure Audit

## Scope
Audit of the failed `calibrated_support_position_outer_loop_pitch_ref` Phase 6 result at:
- `high_0p450`
- `high_0p465`
- `high_0p480`

Reference profiles:
- A = `height_scheduled_pitch_equilibrium_trim`
- B = `support_position_outer_loop_pitch_ref`
- B2 = `calibrated_support_position_outer_loop_pitch_ref`

## Summary
B2 fixed `high_0p450` relative to B, but it regressed at `high_0p465` and `high_0p480`. The upper-band pattern is consistent with an aggressive high-end proportional schedule rather than a damping or safety-limit failure.

### Key observations
- `high_0p450`:
  - B score: `2108.84`
  - B2 score: `713.89`
  - B2 is much better and remains close to A.
- `high_0p465`:
  - B score: `906.42`
  - B2 score: `1119.42`
  - B2 is worse than B.
- `high_0p480`:
  - B score: `1496.17`
  - B2 score: `1702.46`
  - B2 is worse than B.

### Upper-band gain shape
The fitted high-band gains jumped too quickly:
- `0.450 -> Kp 0.650`
- `0.465 -> Kp 1.350`
- `0.480 -> Kp 1.575`

This is the dominant suspect because:
1. the controller improves immediately after the lower-Kp point at `0.450`,
2. but then becomes substantially more aggressive at `0.465` and `0.480`, and
3. the regressions show up mainly as worse drift envelope and score, not as safety collapse.

### Evidence against other primary causes
- **Kd / filter:** no strong evidence of a derivative/filter mis-tuning. The failure pattern tracks the Kp jump more than damping-related instability.
- **Theta limit / deadband:** no evidence of a hard saturation or deadband issue causing the regression.
- **B already optimal:** B is clearly not optimal at `high_0p450`, because B2 substantially improves there.
- **Fit discontinuity:** likely contributes, but only as a mechanism for the Kp jump; the core problem is still the aggressive upper-band Kp curve.

## Classification
**HIGH_BAND_KP_TOO_AGGRESSIVE**

## Conclusion
The failed B2 should not replace B as current best. A revised v2 should preserve the `high_0p450` gain while smoothing the `0.465–0.480` band so Kp does not rise too sharply.
