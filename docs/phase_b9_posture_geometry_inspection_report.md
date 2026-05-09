# Phase B.9 Posture Geometry Inspection Report

## Scope

This report verifies B9 static initial posture geometry using `D:/2CHAN/Wheeled-bipedal-robot-simulation/assets/robot/wheeled_biped_real.xml`. It does not run residual PPO training, does not update residual training configs, and does not resume B9 tuning sweeps.

## Model frame and joint-axis inspection

The active model is `wheeled_biped_real.xml`. Its XML convention states:

- `X`: lateral, positive left.
- `Y`: backward when positive; robot forward is `-Y`.
- `Z`: up.

Joint axes from the active model:

- `l_hip_roll` axis=[0.0, 1.0, 0.0] range=[-0.7, 0.7]
- `l_hip_yaw` axis=[0.0, 0.9999975486805301, -0.0022141890006909444] range=[-0.4, 0.4]
- `l_hip_pitch` axis=[0.0, -1.0, 0.0] range=[-0.5, 1.8]
- `l_knee` axis=[0.0, -1.0, 0.0] range=[-0.5, 2.7]
- `l_wheel` axis=[-1.0, 0.0, 0.0] range=[0.0, 0.0]
- `r_hip_roll` axis=[0.0, 1.0, 0.0] range=[-0.7, 0.7]
- `r_hip_yaw` axis=[0.0, 0.0, -1.0] range=[-0.4, 0.4]
- `r_hip_pitch` axis=[0.0, 1.0, 0.0] range=[-0.5, 1.8]
- `r_knee` axis=[0.0, 1.0, 0.0] range=[-0.5, 2.7]
- `r_wheel` axis=[1.0, 0.0, 0.0] range=[0.0, 0.0]

Because the active robot faces `-Y`, knee-forward is measured as `hip_y - knee_y > 0` in world coordinates. The older simplified-model interpretation `knee_y - hip_y > 0` is not valid for this real model.

## B9 initial posture summary

| target h | hip pitch L | knee L | torso h | knee fwd L | knee fwd R | CoM lateral x | both wheels touch |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0.65 | 0.178 | 0.520 | 0.709 | 0.046 | 0.046 | 0.000 | True |
| 0.60 | 0.484 | 0.942 | 0.671 | 0.121 | 0.121 | 0.000 | True |
| 0.55 | 0.720 | 1.306 | 0.618 | 0.171 | 0.171 | 0.000 | True |
| 0.50 | 0.885 | 1.614 | 0.563 | 0.201 | 0.201 | 0.000 | True |
| 0.45 | 0.980 | 1.865 | 0.512 | 0.216 | 0.216 | 0.000 | True |
| 0.40 | 1.005 | 2.059 | 0.467 | 0.220 | 0.220 | 0.000 | True |

Summary:

- All B9 knees forward: `True`
- All B9 torsos near upright: `True`
- All B9 wheel collision bottoms grounded/symmetric: `True`
- Max absolute CoM lateral offset: `0.0000` m
- Max wheel collision-bottom height difference: `0.000049` m
- Max diagnostic site-height difference: `0.111561` m

Visual interpretation: the saved side/front/top images in `outputs/phase_b9_geometry_check/` should be used as the primary manual check. Numerically, B9 postures are forward-knee and symmetric under FK, but visual review is still required before using them as trusted equilibrium postures.

## Human-like candidate posture summary

| posture | hip pitch L | knee L | torso h | knee fwd L | knee fwd R | CoM lateral x |
|---|---:|---:|---:|---:|---:|---:|
| upright | 0.126 | 0.478 | 0.710 | 0.033 | 0.033 | 0.000 |
| mild_crouch | 0.527 | 1.020 | 0.661 | 0.131 | 0.131 | 0.000 |
| medium_crouch | 0.784 | 1.617 | 0.562 | 0.184 | 0.184 | 0.000 |
| deep_crouch | 1.067 | 2.092 | 0.460 | 0.228 | 0.228 | 0.000 |

Summary:

- All candidate knees forward: `True`
- All candidate torsos near upright: `True`
- All candidate wheel collision bottoms grounded/symmetric: `True`
- Max absolute candidate CoM lateral offset: `0.0000` m
- Max diagnostic candidate site-height difference: `0.110409` m

## B9 vs candidate comparison

- h=0.65: closest candidate by realized torso height is upright (B9 torso height 0.709 m, candidate torso height 0.710 m). B9 knees_forward=True, knee margins L/R=(0.046, 0.046) m, CoM lateral offset=0.000 m.
- h=0.60: closest candidate by realized torso height is mild_crouch (B9 torso height 0.671 m, candidate torso height 0.661 m). B9 knees_forward=True, knee margins L/R=(0.121, 0.121) m, CoM lateral offset=0.000 m.
- h=0.55: closest candidate by realized torso height is mild_crouch (B9 torso height 0.618 m, candidate torso height 0.661 m). B9 knees_forward=True, knee margins L/R=(0.171, 0.171) m, CoM lateral offset=0.000 m.
- h=0.50: closest candidate by realized torso height is medium_crouch (B9 torso height 0.563 m, candidate torso height 0.562 m). B9 knees_forward=True, knee margins L/R=(0.201, 0.201) m, CoM lateral offset=0.000 m.
- h=0.45: closest candidate by realized torso height is medium_crouch (B9 torso height 0.512 m, candidate torso height 0.562 m). B9 knees_forward=True, knee margins L/R=(0.216, 0.216) m, CoM lateral offset=0.000 m.
- h=0.40: closest candidate by realized torso height is deep_crouch (B9 torso height 0.467 m, candidate torso height 0.460 m). B9 knees_forward=True, knee margins L/R=(0.220, 0.220) m, CoM lateral offset=0.000 m.

## Answers to required questions

- Whether B9 currently uses human-like posture: B9 currently appears to use a human-like, symmetric forward-knee posture in the active real model; wheel-ground contact is grounded by wheel collision-bottom geometry, with contact-site asymmetry retained only as a diagnostic.
- Whether knees bend forward: B9 knee-forward margins are positive for all rendered target heights using the active `-Y` forward convention.
- Whether posture/contact symmetry is correct: B9 is symmetric by commanded joints and wheel collision-bottom height; CoM lateral offset is small. Contact-site heights are retained as diagnostics because the contact markers are not reliable ground-clearance proxies for this asymmetric real model. Contact forces may be `NaN` for static FK snapshots if MuJoCo does not report active contact forces without dynamic settling.
- Whether the mechanical limitation conclusion is justified: The mechanical-limitation conclusion remains premature; this report verifies static posture geometry only, not corrected-posture closed-loop roll recovery.
- What posture corrections should be tried next: use the rendered B9 and candidate images to choose the visually best height-indexed postures; then test those corrected postures in fast-loop-only B9 before changing LQR gains.
- Whether B9 can proceed to fast-loop-only testing: yes, after manual visual inspection confirms the rendered B9/candidate postures look correct; do not resume tuning sweeps yet.

## Recommended next posture/equilibrium actions

1. Visually inspect all side/front/top PNGs for knee direction, torso alignment, and wheel-ground contact.
2. If any B9 height looks visually folded incorrectly, replace that height's equilibrium with the closest visually correct candidate or a nearby FK-adjusted posture.
3. Build a corrected `equilibrium_posture_table_b9.yaml` only after choosing the visually preferred rows.
4. Use the selected postures for future numerical linearization and B9 fast-loop-only testing.
5. Only after corrected-posture fast-loop-only tests fail should roll authority or mechanical limitations be revisited.
