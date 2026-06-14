# Phase 2 Addendum: Sign Error Investigation

## Finding

The "sign_error" classification for nominal was investigated and found to be a **false positive**.

## Root Cause

The sign correctness flag checks if torque opposes position error, but does not account for velocity damping.

At step 317 (nominal):
- Position error: +0.0050 rad
- Velocity: +0.0259 rad/s
- Position term: kp × error = 15.0 × 0.0050 = +0.074 Nm
- Damping term: -kd × vel = -3.0 × 0.0259 = -0.078 Nm
- **Total torque: +0.074 - 0.078 = -0.004 Nm**

The damping term slightly dominates the position term, producing negative total torque even though error is positive. This is **correct PD control behavior**, not a sign error.

## Verification

The actual torque matches the expected PD equation exactly:
- Expected: -0.003551 Nm
- Actual: -0.003551 Nm
- Match: ✅

## Revised Classification

- **low_0p300**: reference_correct ✅
- **high_0p480**: reference_correct ✅  
- **nominal**: reference_correct ✅ (sign_error was false positive due to damping dominance)

## Conclusion

All hip-yaw references and torque commands are correct. The controller is implementing proper PD control with correct signs.

**Ready for Phase 3: Hip-yaw torque authority audit.**
