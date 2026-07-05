# Phase 2D.1 — Multi-Scenario Contact Dynamics Validation Audit Report

**Timestamp:** 2026-07-03T06:43:39.352772+00:00  
**Verdict:** `READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE`  
**Reason:** All validations pass with full coverage

## 1. Executive Summary

Phase 2D.1 expands Phase 2D contact dynamics validation across multiple
physically meaningful scenarios to harden readiness for Phase 3 QP-WBC prototyping.

- **12/12** scenarios produced valid wheel-floor contacts
- **44** total contacts validated
- **22** left wheel, **22** right wheel contacts
- All core validations: PASS

### Results Summary

| Validation | PASS | WARN | FAIL | Max Error |
|------------|------|------|------|-----------|
| Contact Point Reconstruction | 44 | 0 | 0 | 8.53e-08 |
| Jacobian Full (3×16) | 44 | 0 | 0 | 9.42e-08 |
| Jacobian Base Linear (cols 0:3) | 44 | 0 | 0 | 0.00e+00 |
| Jacobian Base Angular (cols 3:6) | 44 | 0 | 0 | 9.42e-08 |
| Jacobian Actuated (cols 6:16) | 44 | 0 | 0 | 7.91e-08 |
| QFRC Full (Path A) | 44 | 0 | 0 | 9.85e-05 |
| QFRC Free-Base | 44 | 0 | 0 | 8.76e-05 |
| QFRC Actuated | 44 | 0 | 0 | 9.85e-05 |

## 2. Controller Integrity Statement

- Controller code modified: **NO ✅**
- `K2_JAX_DEDICATED_DEFAULT_V3`: **unchanged**
- No controller files imported by contact dynamics module
- No QP solver, no WBC, no torque injection

## 3. Changed Files

| File | Status |
|------|--------|
| `scripts/phase2d1_contact_multiscenario_audit.py` | **new** — Phase 2D.1 audit |
| `tests/test_phase2d1_contact_multiscenario.py` | **new** — test suite |
| `docs/validation/k2_phase2d1_contact_multiscenario_audit.md` | **new** — this report |
| `docs/validation/k2_phase2d1_contact_multiscenario_audit.json` | **new** — JSON summary |
| `wheeled_biped/dynamics/jax_contact_dynamics.py` | **unchanged** |
| `wheeled_biped/controllers/*` | **unchanged** |

## 4. Phase 2C.5 Cleanup Recap

Phase 2C.5 remains READY_FOR_PHASE_2D_CONTACT_DYNAMICS_PORT:
- 23 tests passed, 0 failed, 0 xpassed
- Full bias: 35 PASS / 0 WARN / 0 FAIL
- Max actuated bias error: 4.60e-07
- JIT-compatible

## 5. Phase 2D Core Recap

Phase 2D core contact mapping tests: 30 passed, 0 failed, 0 xpassed
- Contact point reconstruction: 4 PASS, max error 6.06e-08 m
- Contact Jacobian: 4 PASS, max full error 4.76e-08
- Contact force mapping: 4 PASS, max qfrc error 1.27e-05

## 6. Scenario Generation Method

Scenarios generated deterministically using:
1. MuJoCo keyframe 0 as base pose
2. Symmetric hip_pitch/knee adjustments for height variations
3. Fixed-magnitude base velocity perturbations
4. `scipy.spatial.transform.Rotation` for orientation variations
5. Fixed-seed `np.random.default_rng()` for random perturbations

No controller execution, no QP/WBC calls, no random non-reproducible scenarios.
All scenarios use `mj_forward` for passive physics resolution only.

## 7. Scenario Inclusion / Skipping Table

| # | Scenario | Included | Contacts | Left | Right | Height | Velocity | Orientation | Skip Reason |
|---|----------|----------|----------|------|-------|--------|----------|-------------|-------------|
| 1 | keyframe_static | Yes | 4 | 2 | 2 | keyframe | No | No |  |
| 2 | passive_settle_keyframe | Yes | 4 | 2 | 2 | keyframe | No | No |  |
| 3 | low_height_settle | Yes | 4 | 2 | 2 | low | No | No |  |
| 4 | mid_height_settle | Yes | 4 | 2 | 2 | mid | No | No |  |
| 5 | high_height_settle | Yes | 4 | 2 | 2 | high | No | No |  |
| 6 | small_forward_velocity | Yes | 4 | 2 | 2 | -- | Yes | No |  |
| 7 | small_lateral_velocity | Yes | 4 | 2 | 2 | -- | Yes | No |  |
| 8 | small_yaw_rate | Yes | 4 | 2 | 2 | -- | Yes | No |  |
| 9 | small_roll_tilt | Yes | 4 | 2 | 2 | -- | No | Yes |  |
| 10 | small_pitch_tilt | Yes | 2 | 2 | 0 | -- | No | Yes |  |
| 11 | random_pose_small_perturbation_1 | Yes | 4 | 2 | 2 | -- | Yes | Yes |  |
| 12 | random_pose_small_perturbation_2 | Yes | 2 | 0 | 2 | -- | Yes | Yes |  |

## 8. Contact Filtering Method

Contacts filtered by: geom belongs to wheel body AND other geom belongs to floor/world body.
Non-wheel contacts (torso/thigh/knee ground collisions) excluded from readiness metrics.
Dynamic body = wheel body; Jacobian validated at contact point on wheel body.

## 9. Contact Detail Table

| Scenario | C ID | Geom1 | Geom2 | Dynamic Body | Side | Contact Pos (world) | Dist | Included |
|----------|------|-------|-------|-------------|------|---------------------|------|----------|
| keyframe_static | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| keyframe_static | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| keyframe_static | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| keyframe_static | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| passive_settle_keyframe | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| passive_settle_keyframe | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| passive_settle_keyframe | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| passive_settle_keyframe | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| low_height_settle | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0187, -0.0012] | -0.0024 | Yes |
| low_height_settle | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0187, -0.0012] | -0.0024 | Yes |
| low_height_settle | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0187, -0.0012] | -0.0023 | Yes |
| low_height_settle | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0187, -0.0012] | -0.0023 | Yes |
| mid_height_settle | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| mid_height_settle | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| mid_height_settle | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| mid_height_settle | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| high_height_settle | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, 0.0025, -0.0119] | -0.0237 | Yes |
| high_height_settle | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, 0.0025, -0.0119] | -0.0237 | Yes |
| high_height_settle | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, 0.0025, -0.0118] | -0.0237 | Yes |
| high_height_settle | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, 0.0025, -0.0118] | -0.0237 | Yes |
| small_forward_velocity | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| small_forward_velocity | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| small_forward_velocity | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| small_forward_velocity | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| small_lateral_velocity | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| small_lateral_velocity | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| small_lateral_velocity | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| small_lateral_velocity | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| small_yaw_rate | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| small_yaw_rate | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| small_yaw_rate | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, -0.0134, -0.0022] | -0.0044 | Yes |
| small_yaw_rate | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, -0.0134, -0.0022] | -0.0044 | Yes |
| small_roll_tilt | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1985, 0.0282, -0.0019] | -0.0038 | Yes |
| small_roll_tilt | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1485, 0.0282, -0.0019] | -0.0038 | Yes |
| small_roll_tilt | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1985, 0.0282, -0.0019] | -0.0037 | Yes |
| small_roll_tilt | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.1485, 0.0282, -0.0019] | -0.0037 | Yes |
| small_pitch_tilt | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1510, -0.0134, -0.0098] | -0.0197 | Yes |
| small_pitch_tilt | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1012, -0.0134, -0.0077] | -0.0153 | Yes |
| random_pose_small_perturbation_1 | 0 | floor | l_wheel_collision | l_wheel_link | left | [0.1836, -0.0179, -0.0154] | -0.0307 | Yes |
| random_pose_small_perturbation_1 | 1 | floor | l_wheel_collision | l_wheel_link | left | [0.1337, -0.0149, -0.0147] | -0.0294 | Yes |
| random_pose_small_perturbation_1 | 2 | floor | r_wheel_collision | r_wheel_link | right | [-0.1559, 0.0073, -0.0141] | -0.0283 | Yes |
| random_pose_small_perturbation_1 | 3 | floor | r_wheel_collision | r_wheel_link | right | [-0.2057, 0.0111, -0.0138] | -0.0277 | Yes |
| random_pose_small_perturbation_2 | 0 | floor | r_wheel_collision | r_wheel_link | right | [-0.1506, 0.0117, -0.0020] | -0.0040 | Yes |
| random_pose_small_perturbation_2 | 1 | floor | r_wheel_collision | r_wheel_link | right | [-0.2005, 0.0099, -0.0019] | -0.0038 | Yes |

## 10. Contact Point Reconstruction Validation

Threshold: PASS < 1e-06 m, WARN < 1e-05 m

| Scenario | Wheel | Point Error (m) | Verdict |
|----------|-------|-----------------|---------|
| keyframe_static | left_wheel | 5.22e-08 | PASS |
| keyframe_static | left_wheel | 5.16e-08 | PASS |
| keyframe_static | right_wheel | 6.06e-08 | PASS |
| keyframe_static | right_wheel | 5.46e-08 | PASS |
| passive_settle_keyframe | left_wheel | 5.22e-08 | PASS |
| passive_settle_keyframe | left_wheel | 5.16e-08 | PASS |
| passive_settle_keyframe | right_wheel | 6.06e-08 | PASS |
| passive_settle_keyframe | right_wheel | 5.46e-08 | PASS |
| low_height_settle | left_wheel | 8.53e-08 | PASS |
| low_height_settle | left_wheel | 5.25e-08 | PASS |
| low_height_settle | right_wheel | 3.47e-08 | PASS |
| low_height_settle | right_wheel | 3.77e-08 | PASS |
| mid_height_settle | left_wheel | 5.22e-08 | PASS |
| mid_height_settle | left_wheel | 5.16e-08 | PASS |
| mid_height_settle | right_wheel | 6.06e-08 | PASS |
| mid_height_settle | right_wheel | 5.46e-08 | PASS |
| high_height_settle | left_wheel | 5.68e-08 | PASS |
| high_height_settle | left_wheel | 5.86e-08 | PASS |
| high_height_settle | right_wheel | 5.35e-08 | PASS |
| high_height_settle | right_wheel | 5.64e-08 | PASS |
| small_forward_velocity | left_wheel | 5.22e-08 | PASS |
| small_forward_velocity | left_wheel | 5.16e-08 | PASS |
| small_forward_velocity | right_wheel | 6.06e-08 | PASS |
| small_forward_velocity | right_wheel | 5.46e-08 | PASS |
| small_lateral_velocity | left_wheel | 5.22e-08 | PASS |
| small_lateral_velocity | left_wheel | 5.16e-08 | PASS |
| small_lateral_velocity | right_wheel | 6.06e-08 | PASS |
| small_lateral_velocity | right_wheel | 5.46e-08 | PASS |
| small_yaw_rate | left_wheel | 5.22e-08 | PASS |
| small_yaw_rate | left_wheel | 5.16e-08 | PASS |
| small_yaw_rate | right_wheel | 6.06e-08 | PASS |
| small_yaw_rate | right_wheel | 5.46e-08 | PASS |
| small_roll_tilt | left_wheel | 3.49e-08 | PASS |
| small_roll_tilt | left_wheel | 4.12e-08 | PASS |
| small_roll_tilt | right_wheel | 1.71e-08 | PASS |
| small_roll_tilt | right_wheel | 2.00e-08 | PASS |
| small_pitch_tilt | left_wheel | 5.23e-08 | PASS |
| small_pitch_tilt | left_wheel | 5.08e-08 | PASS |
| random_pose_small_perturbation_1 | left_wheel | 3.95e-08 | PASS |
| random_pose_small_perturbation_1 | left_wheel | 4.40e-08 | PASS |
| random_pose_small_perturbation_1 | right_wheel | 2.87e-08 | PASS |
| random_pose_small_perturbation_1 | right_wheel | 2.92e-08 | PASS |
| random_pose_small_perturbation_2 | right_wheel | 8.19e-08 | PASS |
| random_pose_small_perturbation_2 | right_wheel | 8.07e-08 | PASS |

## 11. Contact Jacobian Validation

Thresholds: PASS < 1e-05, WARN < 1e-04, FAIL >= 1e-04

| Scenario | Wheel | Jp Full | Jp Base Lin | Jp Base Ang | Jp Act | Verdict |
|----------|-------|---------|-------------|-------------|--------|---------|
| keyframe_static | left_wheel | 4.76e-08 | 0.00e+00 | 4.76e-08 | 3.48e-08 | PASS |
| keyframe_static | left_wheel | 3.41e-08 | 0.00e+00 | 2.09e-08 | 3.41e-08 | PASS |
| keyframe_static | right_wheel | 4.15e-08 | 0.00e+00 | 3.87e-08 | 4.15e-08 | PASS |
| keyframe_static | right_wheel | 4.26e-08 | 0.00e+00 | 4.26e-08 | 4.17e-08 | PASS |
| passive_settle_keyframe | left_wheel | 4.76e-08 | 0.00e+00 | 4.76e-08 | 3.48e-08 | PASS |
| passive_settle_keyframe | left_wheel | 3.41e-08 | 0.00e+00 | 2.09e-08 | 3.41e-08 | PASS |
| passive_settle_keyframe | right_wheel | 4.15e-08 | 0.00e+00 | 3.87e-08 | 4.15e-08 | PASS |
| passive_settle_keyframe | right_wheel | 4.26e-08 | 0.00e+00 | 4.26e-08 | 4.17e-08 | PASS |
| low_height_settle | left_wheel | 8.53e-08 | 0.00e+00 | 8.53e-08 | 5.84e-08 | PASS |
| low_height_settle | left_wheel | 5.38e-08 | 0.00e+00 | 5.25e-08 | 5.38e-08 | PASS |
| low_height_settle | right_wheel | 7.05e-08 | 0.00e+00 | 3.66e-08 | 7.05e-08 | PASS |
| low_height_settle | right_wheel | 5.86e-08 | 0.00e+00 | 3.77e-08 | 5.86e-08 | PASS |
| mid_height_settle | left_wheel | 4.76e-08 | 0.00e+00 | 4.76e-08 | 3.48e-08 | PASS |
| mid_height_settle | left_wheel | 3.41e-08 | 0.00e+00 | 2.09e-08 | 3.41e-08 | PASS |
| mid_height_settle | right_wheel | 4.15e-08 | 0.00e+00 | 3.87e-08 | 4.15e-08 | PASS |
| mid_height_settle | right_wheel | 4.26e-08 | 0.00e+00 | 4.26e-08 | 4.17e-08 | PASS |
| high_height_settle | left_wheel | 5.94e-08 | 0.00e+00 | 4.15e-08 | 5.94e-08 | PASS |
| high_height_settle | left_wheel | 5.44e-08 | 0.00e+00 | 5.44e-08 | 4.85e-08 | PASS |
| high_height_settle | right_wheel | 7.06e-08 | 0.00e+00 | 5.35e-08 | 7.06e-08 | PASS |
| high_height_settle | right_wheel | 7.36e-08 | 0.00e+00 | 5.64e-08 | 7.36e-08 | PASS |
| small_forward_velocity | left_wheel | 4.76e-08 | 0.00e+00 | 4.76e-08 | 3.48e-08 | PASS |
| small_forward_velocity | left_wheel | 3.41e-08 | 0.00e+00 | 2.09e-08 | 3.41e-08 | PASS |
| small_forward_velocity | right_wheel | 4.15e-08 | 0.00e+00 | 3.87e-08 | 4.15e-08 | PASS |
| small_forward_velocity | right_wheel | 4.26e-08 | 0.00e+00 | 4.26e-08 | 4.17e-08 | PASS |
| small_lateral_velocity | left_wheel | 4.76e-08 | 0.00e+00 | 4.76e-08 | 3.48e-08 | PASS |
| small_lateral_velocity | left_wheel | 3.41e-08 | 0.00e+00 | 2.09e-08 | 3.41e-08 | PASS |
| small_lateral_velocity | right_wheel | 4.15e-08 | 0.00e+00 | 3.87e-08 | 4.15e-08 | PASS |
| small_lateral_velocity | right_wheel | 4.26e-08 | 0.00e+00 | 4.26e-08 | 4.17e-08 | PASS |
| small_yaw_rate | left_wheel | 4.76e-08 | 0.00e+00 | 4.76e-08 | 3.48e-08 | PASS |
| small_yaw_rate | left_wheel | 3.41e-08 | 0.00e+00 | 2.09e-08 | 3.41e-08 | PASS |
| small_yaw_rate | right_wheel | 4.15e-08 | 0.00e+00 | 3.87e-08 | 4.15e-08 | PASS |
| small_yaw_rate | right_wheel | 4.26e-08 | 0.00e+00 | 4.26e-08 | 4.17e-08 | PASS |
| small_roll_tilt | left_wheel | 5.98e-08 | 0.00e+00 | 5.12e-08 | 5.98e-08 | PASS |
| small_roll_tilt | left_wheel | 5.83e-08 | 0.00e+00 | 1.80e-08 | 5.83e-08 | PASS |
| small_roll_tilt | right_wheel | 5.25e-08 | 0.00e+00 | 3.00e-08 | 5.25e-08 | PASS |
| small_roll_tilt | right_wheel | 6.15e-08 | 0.00e+00 | 6.15e-08 | 5.25e-08 | PASS |
| small_pitch_tilt | left_wheel | 6.24e-08 | 0.00e+00 | 4.40e-08 | 6.24e-08 | PASS |
| small_pitch_tilt | left_wheel | 5.33e-08 | 0.00e+00 | 3.13e-08 | 5.33e-08 | PASS |
| random_pose_small_perturbation_1 | left_wheel | 6.73e-08 | 0.00e+00 | 4.00e-08 | 6.73e-08 | PASS |
| random_pose_small_perturbation_1 | left_wheel | 6.86e-08 | 0.00e+00 | 4.77e-08 | 6.86e-08 | PASS |
| random_pose_small_perturbation_1 | right_wheel | 7.32e-08 | 0.00e+00 | 7.32e-08 | 6.15e-08 | PASS |
| random_pose_small_perturbation_1 | right_wheel | 9.42e-08 | 0.00e+00 | 9.42e-08 | 6.27e-08 | PASS |
| random_pose_small_perturbation_2 | right_wheel | 6.89e-08 | 0.00e+00 | 5.72e-08 | 6.89e-08 | PASS |
| random_pose_small_perturbation_2 | right_wheel | 7.91e-08 | 0.00e+00 | 7.27e-08 | 7.91e-08 | PASS |

## 12. Free-Base Angular Convention Revalidation

Validates `Jp[:, 3:6] = -skew(r) @ R_base_world` at multiple base orientations.

| Orientation | RPY (deg) | Jp[:, 3:6] Error | Jp[:, 0:3] Identity Error | Verdict |
|-------------|-----------|------------------|---------------------------|---------|
| identity | [0, 0, 0] | 0.00e+00 | 0.00e+00 | PASS |
| roll_5deg | [5, 0, 0] | 4.13e-08 | 0.00e+00 | PASS |
| pitch_5deg | [0, 5, 0] | 5.96e-08 | 0.00e+00 | PASS |
| yaw_10deg | [0, 0, 10] | 2.85e-08 | 0.00e+00 | PASS |
| combined_small_rpy | [3, 4, 6] | 3.90e-08 | 0.00e+00 | PASS |

## 13. Contact Wrench / Frame Convention Validation

MuJoCo `contact.frame` is a 3×3 matrix where:
- `frame[:, 0]` = contact normal
- `frame[:, 1]` = first tangent
- `frame[:, 2]` = second tangent

World-frame force: `f_world = contact.frame @ f_contact_frame`
Extracted via `mj_contactForce` for each contact.

## 14. Contact QFRC Mapping Validation

Thresholds: PASS < 1e-04, WARN < 1e-03, FAIL >= 1e-03

CPU Path A: `qfrc_cpu = jacp_cpu^T @ force_world + jacr_cpu^T @ torque_world`

| Scenario | Wheel | QFRC Full | QFRC FB | QFRC Act | Verdict |
|----------|-------|-----------|---------|----------|---------|
| keyframe_static | left_wheel | 6.55e-06 | 4.70e-06 | 6.55e-06 | PASS |
| keyframe_static | left_wheel | 8.31e-06 | 3.57e-06 | 8.31e-06 | PASS |
| keyframe_static | right_wheel | 7.19e-06 | 7.00e-06 | 7.19e-06 | PASS |
| keyframe_static | right_wheel | 1.27e-05 | 1.27e-05 | 9.93e-06 | PASS |
| passive_settle_keyframe | left_wheel | 6.55e-06 | 4.70e-06 | 6.55e-06 | PASS |
| passive_settle_keyframe | left_wheel | 8.31e-06 | 3.57e-06 | 8.31e-06 | PASS |
| passive_settle_keyframe | right_wheel | 7.19e-06 | 7.00e-06 | 7.19e-06 | PASS |
| passive_settle_keyframe | right_wheel | 1.27e-05 | 1.27e-05 | 9.93e-06 | PASS |
| low_height_settle | left_wheel | 8.98e-06 | 8.98e-06 | 5.55e-06 | PASS |
| low_height_settle | left_wheel | 9.19e-06 | 6.19e-06 | 9.19e-06 | PASS |
| low_height_settle | right_wheel | 4.72e-06 | 2.55e-06 | 4.72e-06 | PASS |
| low_height_settle | right_wheel | 6.15e-06 | 4.21e-06 | 6.15e-06 | PASS |
| mid_height_settle | left_wheel | 6.55e-06 | 4.70e-06 | 6.55e-06 | PASS |
| mid_height_settle | left_wheel | 8.31e-06 | 3.57e-06 | 8.31e-06 | PASS |
| mid_height_settle | right_wheel | 7.19e-06 | 7.00e-06 | 7.19e-06 | PASS |
| mid_height_settle | right_wheel | 1.27e-05 | 1.27e-05 | 9.93e-06 | PASS |
| high_height_settle | left_wheel | 2.68e-05 | 8.10e-06 | 2.68e-05 | PASS |
| high_height_settle | left_wheel | 5.21e-05 | 5.21e-05 | 3.05e-05 | PASS |
| high_height_settle | right_wheel | 8.78e-05 | 6.14e-05 | 8.78e-05 | PASS |
| high_height_settle | right_wheel | 9.85e-05 | 6.00e-05 | 9.85e-05 | PASS |
| small_forward_velocity | left_wheel | 1.00e-05 | 5.44e-06 | 1.00e-05 | PASS |
| small_forward_velocity | left_wheel | 5.94e-06 | 5.94e-06 | 5.83e-06 | PASS |
| small_forward_velocity | right_wheel | 5.64e-06 | 5.64e-06 | 3.21e-06 | PASS |
| small_forward_velocity | right_wheel | 1.81e-05 | 1.81e-05 | 1.28e-05 | PASS |
| small_lateral_velocity | left_wheel | 7.37e-06 | 5.10e-06 | 7.37e-06 | PASS |
| small_lateral_velocity | left_wheel | 8.91e-06 | 2.77e-06 | 8.91e-06 | PASS |
| small_lateral_velocity | right_wheel | 7.85e-06 | 7.85e-06 | 6.93e-06 | PASS |
| small_lateral_velocity | right_wheel | 1.24e-05 | 1.24e-05 | 6.31e-06 | PASS |
| small_yaw_rate | left_wheel | 7.40e-06 | 6.29e-06 | 7.40e-06 | PASS |
| small_yaw_rate | left_wheel | 8.03e-06 | 1.90e-06 | 8.03e-06 | PASS |
| small_yaw_rate | right_wheel | 9.87e-06 | 9.87e-06 | 7.04e-06 | PASS |
| small_yaw_rate | right_wheel | 1.27e-05 | 1.27e-05 | 6.04e-06 | PASS |
| small_roll_tilt | left_wheel | 7.24e-06 | 5.48e-06 | 7.24e-06 | PASS |
| small_roll_tilt | left_wheel | 8.63e-06 | 2.53e-06 | 8.63e-06 | PASS |
| small_roll_tilt | right_wheel | 5.35e-06 | 5.35e-06 | 2.62e-06 | PASS |
| small_roll_tilt | right_wheel | 7.06e-06 | 7.06e-06 | 4.73e-06 | PASS |
| small_pitch_tilt | left_wheel | 5.60e-05 | 5.60e-05 | 3.43e-05 | PASS |
| small_pitch_tilt | left_wheel | 0.00e+00 | 0.00e+00 | 0.00e+00 | PASS |
| random_pose_small_perturbation_1 | left_wheel | 9.56e-05 | 8.76e-05 | 9.56e-05 | PASS |
| random_pose_small_perturbation_1 | left_wheel | 1.49e-05 | 1.49e-05 | 1.28e-05 | PASS |
| random_pose_small_perturbation_1 | right_wheel | 6.55e-05 | 5.74e-05 | 6.55e-05 | PASS |
| random_pose_small_perturbation_1 | right_wheel | 2.78e-05 | 2.62e-05 | 2.78e-05 | PASS |
| random_pose_small_perturbation_2 | right_wheel | 2.02e-05 | 1.47e-05 | 2.02e-05 | PASS |
| random_pose_small_perturbation_2 | right_wheel | 2.29e-06 | 9.33e-07 | 2.29e-06 | PASS |

## 15. Summed qfrc_constraint Validation

| Scenario | Applicable | Verdict | Error | Reason |
|----------|-----------|---------|-------|--------|
| keyframe_static | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| passive_settle_keyframe | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| low_height_settle | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| mid_height_settle | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| high_height_settle | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| small_forward_velocity | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| small_lateral_velocity | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| small_yaw_rate | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| small_roll_tilt | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| small_pitch_tilt | No | not_applicable | -- | Not applicable: joint limits active; nefc(20) > ncon(2), non-contact constraints present |
| random_pose_small_perturbation_1 | No | not_applicable | -- | Not applicable: joint limits active; nefc(40) > ncon(4), non-contact constraints present |
| random_pose_small_perturbation_2 | No | not_applicable | -- | Not applicable: joint limits active; nefc(20) > ncon(2), non-contact constraints present |

## 16. Aggregate Metrics

- Max contact point error: 8.53e-08 m
- Max Jacobian full error: 9.42e-08
- Max Jacobian base linear error: 0.00e+00
- Max Jacobian base angular error: 9.42e-08
- Max Jacobian actuated error: 7.91e-08
- Max QFRC full error: 9.85e-05
- Max QFRC free-base error: 8.76e-05
- Max QFRC actuated error: 9.85e-05

## 17. JIT Compatibility

JIT check: PASS ✅
All core contact functions JIT-compile and produce finite outputs.

## 18. Limitations

- Contact detection not implemented — CPU MuJoCo locates contacts; JAX validates mapping only.
- Summed qfrc_constraint validation may be inapplicable due to joint limits or non-contact constraints.
- No friction cone / QP / WBC integration — Phase 3 scope.
- No controller integration — pure dynamics validation layer.

## 19. Phase 3 Readiness Verdict

```text
READY_FOR_PHASE_3_OFFLINE_QP_WBC_PROTOTYPE
```
**Reason:** All validations pass with full coverage
