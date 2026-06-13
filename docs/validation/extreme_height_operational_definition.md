# Extreme-Height Operational Standing Definition

**Date:** 2026-06-02

An operational standing height is a physically valid static standing pose, not a root-z displacement. A candidate height is valid only when the robot changes leg posture, calibrates root_z from wheel-floor contact, and passes all static geometry, posture, contact, and controller-readiness gates below.

## Required static gates

### Geometry and contact

- Left wheel is in floor contact.
- Right wheel is in floor contact.
- No non-wheel body part contacts the floor.
- Wheel contact force is positive and finite.
- Wheel contact distance is produced by root_z calibration, not by moving root_z alone.

### CoM and support region

- CoM XY projection remains close to the support center.
- Preferred support error norm is `<= 0.005 m`.
- Maximum support error norm is `<= 0.010 m`.
- Lateral offset remains near zero.
- Sagittal support reference can be captured from the pose.

### Posture

- `pitch_x`, `roll_y`, and `yaw_z` remain near equilibrium.
- Hip-yaw stays symmetric and near reference.
- Left and right hip-pitch/knee postures are symmetric.
- Hip-pitch and knee remain inside MJCF joint limits with explicit margin.
- Final extrema use an additional selection margin away from the first invalid or near-limit point.

### Height

- Achieved CoM height is close to requested target.
- Root_z is calibrated consistently with wheel contact.
- A pose is rejected if it is root-z-only, even if the achieved height is close.

### Controller readiness

- Equilibrium joint references are captured from the final static pose.
- Equilibrium CoM and support references are captured after applying the pose.
- WBC remains off.
- Hidden torque is zero.
- Ownership violation count is zero.

## Extrema interpretation

The selected minimum and maximum operational heights are conservative validated extrema. They are not claimed to be absolute mechanical limits, and they are not inferred from joint limits alone. Dynamic Step E and Step C validation must be run before claiming either extreme is operational under the controller.
