"""Side-by-side visual (V3 | Assist) under a single push, from the validated
dual-arm rollout in promote_v3_vs_assist. Writes a GIF (no ffmpeg needed).

Usage:
  python scripts/viz_v3_vs_assist_push.py --force 90 --seed 90 --variant nominal
"""
import argparse
import numpy as np
import mujoco
from PIL import Image, ImageDraw

import scripts.promote_v3_vs_assist as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", type=float, default=90.0, help="push magnitude (N)")
    ap.add_argument("--seed", type=int, default=90, help="random direction seed")
    ap.add_argument("--variant", type=str, default="nominal")
    ap.add_argument("--post-push", type=int, default=240, help="steps after push")
    ap.add_argument("--push-duration", type=int, default=12, help="push length (steps)")
    ap.add_argument("--stride", type=int, default=1, help="render every Nth step")
    ap.add_argument("--setup", type=str, default=None,
                    help="height-setup JSON: start from this posture (e.g. the 0.40m nominal)")
    ap.add_argument("--warmup", type=int, default=None,
                    help="steps before push (settle time); default PUSH_WARMUP_STEPS")
    ap.add_argument("--push-body", type=str, default="torso")
    ap.add_argument("--push-dir", type=float, nargs=3, default=None,
                    help="explicit push direction x y z (overrides random)")
    ap.add_argument("--out", type=str,
                    default="outputs/visual/v3_vs_assist_push.gif")
    ap.add_argument("--assist-mode", choices=["posture_guided", "torque_blend"],
                    default="posture_guided")
    ap.add_argument("--profile", type=str, default="K2_JAX_DEDICATED_DEFAULT_V3",
                    help="V3 controller profile (e.g. K2_JAX_DEDICATED_DEFAULT_V3_HOMING)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    data = mujoco.MjData(model)
    qp_c = P.build_qp_wbc_constants(model)
    P._offline_qp_wbc._ensure_contact_constants(qp_c)
    rolling_c = P.build_wheel_rolling_constants(
        model, contact_constants=qp_c.get("_contact_constants"))
    constants = P.build_three_arm_eval_constants(
        model, qp_constants=qp_c, rolling_constants=rolling_c,
        task_mode="balanced_default", rolling_mode="full_rolling_soft")
    v3 = P.init_v3_controller(profile_name=args.profile, model=model)

    if args.setup:
        # Start from a height-setup posture (matches run_k2_jax_realtime exactly).
        import json
        hs = json.load(open(args.setup))
        d0 = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d0, 0)
        d0.qpos[7:17] = [
            hs.get("hip_roll_left", 0.0), hs.get("hip_yaw_left", 0.0),
            hs.get("hip_pitch_ref", 0.0), hs.get("knee_ref", 0.0), 0.0,
            hs.get("hip_roll_right", 0.0), hs.get("hip_yaw_right", 0.0),
            hs.get("hip_pitch_ref", 0.0), hs.get("knee_ref", 0.0), 0.0,
        ]
        if "calibrated_root_z_m" in hs:
            d0.qpos[2] = hs["calibrated_root_z_m"]
        mujoco.mj_forward(model, d0)
        # Settle under V3 and capture the SETTLED controller state (F13): the
        # rollout must restore it rather than reset+re-stabilize (which over-drives
        # the wheels). Mirrors generate_height_variant_state.
        _tcom = float(hs.get("target_com_z_m", P._com_z(d0)))
        from wheeled_biped.controllers.k2_jax_controller import pack_state_k2 as _psk
        v3["jax_state"] = _psk()
        _sctx = P._build_v3_controller_context(model, d0, v3, height_ref=_tcom)
        for _ in range(300):
            _t = P._compute_v3_torque_real(d0, model, v3, _sctx)
            d0.ctrl[:] = _t
            for _ in range(5):
                mujoco.mj_step(model, d0)
        # seed_qpos_z = base-z (assist g_height gate target); target_com_z = CoM
        # command for V3 (compared against com_z). See audit F3/F4 — never feed
        # base-z as V3's height ref.
        state = {"qpos": d0.qpos.copy(), "qvel": d0.qvel.copy(),
                 "v3_jax_state": np.asarray(v3["jax_state"]),
                 "meta": {"seed_qpos_z": float(d0.qpos[2]),
                          "target_com_z": _tcom,
                          "variant": hs.get("variant_name", "setup"),
                          "final_qpos_z": float(d0.qpos[2])}}
        print(f"setup settled: root_z={d0.qpos[2]:.4f} target_com={_tcom:.4f} "
              f"qvel_norm={np.linalg.norm(d0.qvel):.3f} wheel_vel={d0.qvel[10]:.2f}/{d0.qvel[15]:.2f}")
    else:
        state = P.generate_height_variant_state(model, data, args.variant, v3_ctrl=v3)
        print(f"{args.variant} settled: z={state['meta']['final_qpos_z']:.4f}")

    if args.push_dir is not None:
        d = np.array(args.push_dir, dtype=float)
    else:
        rng = np.random.default_rng(args.seed)
        d = rng.uniform(-1, 1, 3); d[2] *= 0.25
    d = d / np.linalg.norm(d)
    force = (d * args.force).tolist()
    print(f"push: |F|={args.force}N dir={np.round(d,3).tolist()} body={args.push_body}")
    push_config = {"body": args.push_body, "force": force,
                   "direction": "explicit" if args.push_dir else "random",
                   "magnitude": args.force}

    warmup = args.warmup if args.warmup is not None else P.PUSH_WARMUP_STEPS
    frames = []
    result = P.run_dual_arm_rollout(
        model, data,
        scenario_name=f"viz_push_{int(args.force)}N",
        scenario_qpos=state["qpos"], scenario_qvel=state["qvel"],
        scenario_meta=state["meta"], constants=constants,
        n_steps=warmup + args.push_duration + args.post_push,
        n_substeps=5,
        push_config=push_config, push_step_start=warmup,
        push_duration=args.push_duration, post_push_steps=args.post_push,
        adaptive_alpha_max=P.ADAPTIVE_ASSIST_ALPHA_MAX,
        assist_mode=args.assist_mode,
        v3_ctrl=v3, v3_jax_state=state.get("v3_jax_state"), warm_start=True,
        frame_capture=frames, frame_stride=args.stride,
    )
    comp = result["classification"]
    v3f = result["v3_metrics"]["falls"]; af = result["assist_metrics"]["falls"]
    print(f"class={comp}  V3 falls={v3f}  Assist falls={af}  frames={len(frames)}")

    # ── Assemble GIF: start ~25 steps before push, mark + slow-mo the push ──
    push_idx = [i for i, fr in enumerate(frames) if fr["push"]]
    start = max(0, (push_idx[0] - 25)) if push_idx else 0
    sel = frames[start:]
    W = sel[0]["img"].shape[1]

    imgs, durations = [], []
    for fr in sel:
        im = Image.fromarray(fr["img"]).convert("RGB")
        dr = ImageDraw.Draw(im)
        dr.text((W // 4 - 12, im.height - 16), "V3", fill=(200, 220, 255))
        dr.text((3 * W // 4 - 24, im.height - 16), "ASSIST", fill=(255, 230, 180))
        if fr["push"]:
            # Force arrow is drawn in-scene (3D) by the renderer; just slow-mo.
            durations.append(160)
        else:
            durations.append(45)
        imgs.append(im)
    if push_idx:  # hold a beat right as the push lands
        durations[push_idx[0] - start] = 500

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                 duration=durations, loop=0)
    print(f"WROTE {args.out}  ({len(imgs)} frames, push at frame {push_idx[0]-start if push_idx else '?'})")


if __name__ == "__main__":
    main()
