#!/usr/bin/env python3
"""Dose-response test for the chirality of ACC's push envelope.

The push envelope is not mirror-symmetric: fitting
``F(theta) = a0 + a1 cos t + b1 sin t + a2 cos 2t + b2 sin 2t`` to the measured
8-bearing sweeps gives ``b2 ~ -16..-29 N`` in every configuration, and a
sin(2 theta) term is exactly the harmonic a reflection about the sagittal
plane forbids.  The plant cannot supply it: the model's two legs agree to
0.05 mm in world position and 4e-8 in world-frame inertia (scripts/... see
paper appendix), and the nominal posture is identically mirrored.

The controller can.  ``_K2_EMPIRICAL_SUPPORT_FF`` is a constant torque vector
fitted empirically, and it is left/right asymmetric:

    [0, 0, 4.1, -15.5, 0,  0, 0, 3.2, -15.8, 0] * 0.5

i.e. a permanent 0.45 Nm hip-pitch and 0.15 Nm knee differential between two
otherwise identical legs.  This script scales that differential about its own
mean and measures the envelope at each scale:

    scale 0   -> both legs get the mean, controller is L/R symmetric
    scale 1   -> shipped ACC
    scale 3   -> differential tripled

If the constant is the source, |b2| grows monotonically with the scale and
falls to the noise floor at 0.  Measurement protocol is identical to
``scripts/replicate_ablation_n10.py``: fresh MjData and a fresh controller for
every bisection iteration, so no controller state crosses between trials.

Usage:
  .venv/bin/python scripts/sweep_push_chirality.py --scale 0 --reps 5 \
      --output outputs/push_chirality/ff_scale_0.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiplier on the L/R half-difference of the "
                         "support feedforward (1.0 = shipped ACC)")
    ap.add_argument("--mirror-controller", action="store_true",
                    help="run the EXACT mirror image of the controller, "
                         "tau~(x) = M tau(M x), by reflecting the state on the "
                         "way in and the torque on the way out. The plant is "
                         "mirror-symmetric, so if the chirality is entirely "
                         "controller-borne the envelope must satisfy "
                         "F~(t) = F(180-t) exactly; any leftover measures what "
                         "the plant and the harness contribute.")
    ap.add_argument("--sham-mirror", action="store_true",
                    help="control for --mirror-controller: route the "
                         "controller through the same second MjData and extra "
                         "mj_forward, but apply the IDENTITY instead of the "
                         "reflection. Isolates the cost of the proxy state "
                         "(a cold constraint solve every step) from the reflection.")
    ap.add_argument("--roll-sign", type=float, default=1.0,
                    help="hip_roll_torque_sign for the lateral channel. "
                         "-1 is the MIRROR-IMAGE controller: the plant is "
                         "mirror-symmetric, so it is equally stable, and if "
                         "this channel is what breaks the envelope's mirror "
                         "then the envelope must reflect, F'(t) = F(180-t).")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # Patch the feedforward BEFORE anything triggers a JAX trace.
    import wheeled_biped.controllers.k2_jax_controller as K2
    import jax.numpy as jnp

    ff = np.array(K2._K2_EMPIRICAL_SUPPORT_FF, dtype=np.float64)
    legs = ff.reshape(2, 5)
    mean = legs.mean(axis=0)
    half = (legs[0] - legs[1]) / 2.0
    scaled = np.stack([mean + args.scale * half, mean - args.scale * half])
    K2._K2_EMPIRICAL_SUPPORT_FF = jnp.array(scaled.reshape(10),
                                            dtype=jnp.float64)
    print(f"support FF  shipped: {ff.tolist()}")
    print(f"support FF  scale={args.scale}: {scaled.reshape(10).tolist()}")

    if args.roll_sign != 1.0:
        _lateral = K2.k2_jax_lateral_roll_compute

        def _lateral_signed(*a, **kw):
            kw["hip_roll_torque_sign"] = (args.roll_sign
                                          * kw.get("hip_roll_torque_sign", 1.0))
            return _lateral(*a, **kw)

        K2.k2_jax_lateral_roll_compute = _lateral_signed
    print(f"hip_roll_torque_sign: {args.roll_sign}")

    import mujoco as mj
    from wheeled_biped.wbc.offline_three_arm_counterfactual import (
        compute_v3_torque_for_state, init_v3_controller)
    from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
    from wheeled_biped.controllers.centroidal_state_estimator import (
        CentroidalStateEstimator, CentroidalStateEstimatorConfig)
    from wheeled_biped.utils.config import get_model_path

    DT, SUBSTEPS = 0.01, 5
    PUSH_DUR, PUSH_START = 7, 300
    POST_PUSH_STEPS = int(17.0 / DT)
    PITCH_LIMIT, HEIGHT_LIMIT = 0.8, 0.30
    FORCE_MIN, FORCE_MAX = 10.0, 160.0
    N_BISECT, TOLERANCE = 8, 5.0
    ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]
    PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
    JOINT_NAMES = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee",
                   "l_wheel", "r_hip_roll", "r_hip_yaw", "r_hip_pitch",
                   "r_knee", "r_wheel"]

    nom = json.load(open(
        "archive/cleanup_2026-06-13/output_summaries/"
        "balance_core_true_height_variants/variant_nominal__variant_setup.json"))
    H0 = float(nom["target_com_z_m"])
    ROOT_Z = float(nom["calibrated_root_z_m"])
    POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
                        nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
                        nom["hip_roll_right"], nom["hip_yaw_right"],
                        nom["hip_pitch_ref"], nom["knee_ref"], 0.0])

    model = mj.MjModel.from_xml_path(str(get_model_path()))
    torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso")
    cfg = CentroidalStateEstimatorConfig(
        robot_mass=float(np.sum(model.body_mass)),
        torso_inertia=np.array(model.body_inertia[1], dtype=np.float64))
    L_ID = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_wheel_link")
    R_ID = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "r_wheel_link")

    def make_ctx(eq_joint):
        """One context per trial.

        The controller MUTATES its context -- it keeps ``prev_com_pos`` and the
        ``airborne_mode``/``airborne_count``/``ground_count`` flight-mode latch
        there.  Sharing one context across bisection iterations lets a fallen
        trial hand the next trial a latched flight mode, so the trial starts in
        the air and dies instantly.
        """
        return {
            "centroidal_estimator": CentroidalStateEstimator(cfg,
                                                             mj_model=model),
            "initial_yaw_z": 0.0, "l_wheel_id": L_ID, "r_wheel_id": R_ID,
            "eq_joint": eq_joint, "height_ref": H0, "prev_com_pos": None,
        }

    # --- exact sagittal mirror of a state / a joint vector -------------------
    SWAP = np.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])
    # hip_roll and hip_yaw hinge about world z and world y, both of which the
    # x-mirror maps to themselves, so their angles reverse; hip_pitch, knee and
    # wheel hinge about world x, which the mirror reverses, so theirs do not.
    JSIGN = np.tile([-1.0, -1.0, 1.0, 1.0, 1.0], 2)

    def mirror_joints(q):
        return q[SWAP] * JSIGN

    # body ids are NOT swapped for the mirrored arm: mdata is itself a valid
    # configuration in which l_wheel_link already carries the mirrored left
    # wheel's state.
    PROXY = args.mirror_controller or args.sham_mirror
    if args.sham_mirror:                      # identity in place of the mirror
        SWAP = np.arange(10)
        JSIGN = np.ones(10)
        SGN = 1.0
    else:
        SGN = -1.0
    EQ = mirror_joints(POSTURE) if PROXY else POSTURE
    mdata = mj.MjData(model) if PROXY else None

    def step(data, v3):
        src, ctx = data, v3["ctx"]
        if PROXY:
            mdata.qpos[:] = data.qpos
            mdata.qvel[:] = data.qvel
            mdata.qpos[0] = SGN * data.qpos[0]
            w, x, y, z = data.qpos[3:7]
            mdata.qpos[3:7] = [w, x, SGN * y, SGN * z]
            mdata.qpos[7:17] = mirror_joints(data.qpos[7:17])
            mdata.qvel[0] = SGN * data.qvel[0]
            mdata.qvel[4] = SGN * data.qvel[4]  # omega is a pseudovector:
            mdata.qvel[5] = SGN * data.qvel[5]  # only the y,z components flip
            mdata.qvel[6:16] = mirror_joints(data.qvel[6:16])
            # The push itself must be mirrored too, or the mirrored controller
            # sees an unloaded robot for the 7 steps that matter most.
            mdata.xfrc_applied[:] = data.xfrc_applied
            mdata.xfrc_applied[:, 0] *= SGN           # force: f_x flips
            mdata.xfrc_applied[:, 4] *= SGN           # torque is a pseudovector:
            mdata.xfrc_applied[:, 5] *= SGN           # only t_y, t_z flip
            mdata.time = data.time
            mj.mj_forward(model, mdata)
            src = mdata
        r = compute_v3_torque_for_state(
            src, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], ctx, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        tau = np.array(r["tau_v3"])
        data.ctrl[:] = mirror_joints(tau) if PROXY else tau

    def survives(data, v3, force_N, angle_deg):
        a = np.deg2rad(angle_deg)
        force = np.array([force_N * np.cos(a), force_N * np.sin(a), 0.0])
        for k in range(POST_PUSH_STEPS + PUSH_DUR):
            step(data, v3)
            data.xfrc_applied[torso_id, :3] = force if k < PUSH_DUR else 0.0
            for _ in range(SUBSTEPS):
                mj.mj_step(model, data)
            q = data.qpos[3:7]
            pitch = np.arcsin(-2 * (q[1] * q[3] - q[0] * q[2]))
            if abs(pitch) > PITCH_LIMIT or data.subtree_com[0][2] < HEIGHT_LIMIT:
                return False
        return True

    def bisect(angle_deg, seed):
        rng = np.random.default_rng(seed)
        lo, hi, best = FORCE_MIN, FORCE_MAX, FORCE_MIN
        for _ in range(N_BISECT):
            mid = (lo + hi) / 2
            data = mj.MjData(model)
            q = POSTURE + rng.normal(0.0, 0.005, size=10)
            for j, name in enumerate(JOINT_NAMES):
                jid = model.joint(name).id
                q[j] = float(np.clip(q[j], *model.jnt_range[jid]))
            data.qpos[7:17] = q
            data.qpos[2] = ROOT_Z + rng.normal(0.0, 0.001)
            mj.mj_forward(model, data)
            v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
            v3["jax_state"] = pack_state_k2()
            v3["ctx"] = make_ctx(EQ)
            for _ in range(PUSH_START):
                step(data, v3)
                for _ in range(SUBSTEPS):
                    mj.mj_step(model, data)
            if survives(data, v3, mid, angle_deg):
                best, lo = mid, mid + TOLERANCE / 2
            else:
                hi = mid - TOLERANCE / 2
        return best

    t0 = time.time()
    all_reps: dict[int, list[float]] = {a: [] for a in ANGLES}
    for rep in range(args.reps):
        for ang in ANGLES:
            all_reps[ang].append(bisect(ang, args.seed * 100 + rep * 1000 + ang))
        print(f"  rep {rep + 1}/{args.reps}  {(time.time() - t0) / 60:.1f} min",
              flush=True)

    th = np.deg2rad(ANGLES)
    F = np.array([np.mean(all_reps[a]) for a in ANGLES])
    A = np.column_stack([np.ones_like(th), np.cos(th), np.sin(th),
                         np.cos(2 * th), np.sin(2 * th)])
    coef, *_ = np.linalg.lstsq(A, F, rcond=None)
    a0, a1, b1, a2, b2 = (float(c) for c in coef)

    out = {
        "ff_scale": args.scale, "reps": args.reps, "seed": args.seed,
        "support_ff": scaled.reshape(10).tolist(),
        "angles_deg": ANGLES,
        "per_bearing_mean_N": F.tolist(),
        "per_bearing_sd_N": [float(np.std(all_reps[a], ddof=1))
                             if args.reps > 1 else 0.0 for a in ANGLES],
        "all_reps": {str(a): all_reps[a] for a in ANGLES},
        "F_min_N": float(F.min()), "F_med_N": float(np.median(F)),
        "fourier": {"a0": a0, "a1_cos": a1, "b1_sin": b1,
                    "a2_cos2": a2, "b2_sin2": b2},
        "mirror_breaking_N": float(np.hypot(a1, b2)),
        "elapsed_min": (time.time() - t0) / 60.0,
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"scale={args.scale}  a1={a1:+.2f}  b2={b2:+.2f}  "
          f"F_min={F.min():.1f}  -> {args.output}")


if __name__ == "__main__":
    main()
