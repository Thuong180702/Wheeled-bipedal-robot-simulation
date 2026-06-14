# Phase B.9 Posture Geometry Inspection Report

## Static Posture Asymmetry Bug and Fix

This report covers B9 initial posture geometry only. It does not tune controller gains, train PPO, or proceed to fast-loop-only testing.

## Joint mapping check

The active real model is `F:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/assets/robot/wheeled_biped_real.xml`. Canonical action order is:

1. `l_hip_roll`
2. `l_hip_yaw`
3. `l_hip_pitch`
4. `l_knee`
5. `l_wheel`
6. `r_hip_roll`
7. `r_hip_yaw`
8. `r_hip_pitch`
9. `r_knee`
10. `r_wheel`

| joint | qpos idx | qvel idx | action idx | axis |
|---|---:|---:|---:|---|
| l_hip_roll | 7 | 6 | 0 | [0.000000, 1.000000, 0.000000] |
| l_hip_yaw | 8 | 7 | 1 | [0.000000, 0.999998, -0.002214] |
| l_hip_pitch | 9 | 8 | 2 | [0.000000, -1.000000, 0.000000] |
| l_knee | 10 | 9 | 3 | [0.000000, -1.000000, 0.000000] |
| l_wheel | 11 | 10 | 4 | [-1.000000, 0.000000, 0.000000] |
| r_hip_roll | 12 | 11 | 5 | [0.000000, 1.000000, 0.000000] |
| r_hip_yaw | 13 | 12 | 6 | [0.000000, 0.000000, -1.000000] |
| r_hip_pitch | 14 | 13 | 7 | [0.000000, 1.000000, 0.000000] |
| r_knee | 15 | 14 | 8 | [0.000000, 1.000000, 0.000000] |
| r_wheel | 16 | 15 | 9 | [1.000000, 0.000000, 0.000000] |

Key result:

- Left/right qpos indices are correct.
- Left/right qvel indices are correct.
- Same-sign hip_pitch/knee values are the correct symmetric construction in the corrected real XML.
- Opposite-sign hip_pitch/knee values break forward-knee symmetry.

## Root cause

The wheel misalignment came from a mechanical transform error in `assets/robot/wheeled_biped_real.xml`, not from camera angle, controller tuning, or qpos index order.

The right thigh local origin was offset relative to the left chain. Correcting `r_thigh` from `pos="-0.03 0 -0.01089"` to `pos="-0.03 0 0.0107637"` removes the nearly constant left/right wheel fore-aft mismatch across the B9 height grid.

Root height initialization now uses wheel collision geometry bottoms through `l_wheel_collision` and `r_wheel_collision`, not diagnostic contact sites. It anchors on the higher wheel bottom so both wheels are at-or-below the ground plane and both wheels produce contact force in MuJoCo.

The `l_wheel_contact` and `r_wheel_contact` sites are sensor/diagnostic frames attached to rotating wheel bodies. They are not reliable ground-contact markers for static visual inspection; collision-geometry contact points and wheel-bottom coordinates are authoritative. The rendered figures hide site visualization so the report cannot be misread as showing site positions as ground contact.

## Fixed posture table

| h | hip pitch L | hip pitch R | knee L | knee R | root pitch deg | root roll deg | root yaw deg | wheel clearance L | wheel clearance R | clearance diff | contact force L | contact force R |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.65 | 0.178 | 0.178 | 0.520 | 0.520 | -0.000 | 0.000 | 0.000 | -0.000049 | -0.000000 | 0.000049 | 33.774 | 18.451 |
| 0.60 | 0.484 | 0.484 | 0.942 | 0.942 | -0.000 | 0.000 | 0.000 | -0.000048 | -0.000000 | 0.000048 | 23.481 | 14.392 |
| 0.55 | 0.720 | 0.720 | 1.306 | 1.306 | -0.000 | 0.000 | 0.000 | -0.000048 | 0.000000 | 0.000048 | 18.429 | 11.602 |
| 0.50 | 0.885 | 0.885 | 1.614 | 1.614 | -0.000 | 0.000 | 0.000 | -0.000048 | 0.000000 | 0.000048 | 15.760 | 10.105 |
| 0.45 | 0.980 | 0.980 | 1.865 | 1.865 | -0.000 | 0.000 | 0.000 | -0.000048 | 0.000000 | 0.000048 | 14.029 | 9.075 |
| 0.40 | 1.005 | 1.005 | 2.059 | 2.059 | -0.000 | 0.000 | 0.000 | -0.000048 | 0.000000 | 0.000048 | 12.577 | 8.198 |

## Old vs fixed comparison

- h=0.65: old clearance L/R=(-0.000000, 0.000049) m, fixed clearance L/R=(-0.000049, -0.000000) m, old knee L/R=(0.520, 0.520), fixed knee L/R=(0.520, 0.520).
- h=0.60: old clearance L/R=(-0.000000, 0.000048) m, fixed clearance L/R=(-0.000048, -0.000000) m, old knee L/R=(0.942, 0.942), fixed knee L/R=(0.942, 0.942).
- h=0.55: old clearance L/R=(-0.000000, 0.000048) m, fixed clearance L/R=(-0.000048, 0.000000) m, old knee L/R=(1.306, 1.306), fixed knee L/R=(1.306, 1.306).
- h=0.50: old clearance L/R=(-0.000000, 0.000048) m, fixed clearance L/R=(-0.000048, 0.000000) m, old knee L/R=(1.614, 1.614), fixed knee L/R=(1.614, 1.614).
- h=0.45: old clearance L/R=(0.000000, 0.000048) m, fixed clearance L/R=(-0.000048, 0.000000) m, old knee L/R=(1.865, 1.865), fixed knee L/R=(1.865, 1.865).
- h=0.40: old clearance L/R=(0.000000, 0.000048) m, fixed clearance L/R=(-0.000048, 0.000000) m, old knee L/R=(2.059, 2.059), fixed knee L/R=(2.059, 2.059).

## Fixed root/orientation summary

- Root orientation remains upright: roll, pitch, and yaw stay near zero.
- Fixed wheel-bottom clearance difference stays under `1e-4` m for all tested heights.
- Both wheel clearances are non-positive after root correction, so both collision geoms touch or slightly penetrate the plane.
- Both wheels report finite positive contact force for all tested heights.
- Both knees bend forward for all tested heights.
- The fixed posture uses symmetric hip/knee scalar targets again; no per-side posture compensation is required after the XML correction.

## Fixed outputs

- Old repro CSV: `outputs/phase_b9_geometry_check/b9_postures.csv`
- Fixed images: `outputs/phase_b9_posture_fixed/`
- Fixed CSV: `F:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/phase_b9_posture_fixed/fixed_postures.csv`
- Coordinate CSV: `F:/ROBOTCUATAO/Wheeled-bipedal-robot-simulation/outputs/phase_b9_posture_fixed/fixed_posture_coordinates.csv`

## Answer

- Cause of asymmetry: mechanical XML transform error in the right thigh chain.
- Old left/right wheel clearance and knee values: see comparison list above.
- Fixed left/right wheel clearance, contact force, and joint values: see fixed table above.
- If a site marker appears offset, treat it as a diagnostic/sensor site artifact rather than wheel-ground contact.
- Fixed B9 posture safe for next fast-loop-only testing: yes, after visual inspection of the generated side/front/top renders.

## Next step

Inspect fixed side/front/top renders manually. Do not tune controller, train PPO, or start fast-loop-only testing until these renders look physically correct.
