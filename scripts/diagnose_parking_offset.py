#!/usr/bin/env python3
"""Static torque-balance decomposition of ACC's -27 mm sagittal parking offset.

Runs the idle-standing protocol of scripts/_idle_ladder_worker.py (identical
seeds, settle, window) but logs the controller's *internal* sagittal torque
decomposition alongside the measured gravity torque about the wheel axles, so
the parking offset can be attributed to a specific term rather than a
hypothesis.

NOTE ON AXES: the sagittal axis is world **y** (the controller projects on
(-sin yaw, cos yaw) and trials start at yaw=0); world x is lateral. The idle
ladder's ``*_x_*`` columns are therefore the lateral ones, and the paper's
-27 mm sagittal parking offset is ``base_off_y_mm``.

Per step it records:
  anchor_integ     anchor integral state, pre-prox-gate       (Nm)
  tau_position     clip(-kpos*e + I, +-max_pos_tau) + trim    (Nm)
  ext_trim         ABS adaptive bias trim                     (Nm)
  tau_pitch        pitch stiffness term                       (Nm)
  tau_wheel_l/r    final per-wheel torque command             (Nm)
  grav_tau         m_total * g * (com_y - axle_y)             (Nm)
  sag_pos_err_inferred   (I + trim - tau_position) / kpos     (m)

The gravity torque is the sagittal moment of total weight about the wheel-axle
line, i.e. the torque the wheels must supply to hold the CoM off-centre.

sag_pos_err is not exported in the diag vector, so it is inverted from the
position-loop law tau_position = -kpos*e + I + trim (valid while unclipped,
which holds throughout quiet stance: |tau_position| ~ 1.3 Nm vs a 4 Nm cap).

Usage:
  .venv/bin/python scripts/diagnose_parking_offset.py [N] [--ki KI] [--leak L]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.promote_v3_vs_assist as P  # noqa: E402
from wheeled_biped.controllers.k2_jax_controller import (  # noqa: E402
    _D_TAU_PITCH,
    _D_TAU_POSITION,
    _D_TAU_WHEEL_L,
    _D_TAU_WHEEL_R,
    _D_EXTERNAL_POS_TRIM,
    _D_LOW_BAND,
    _D_PHYSICS_FF,
    _IDX_ANCHOR_KI,
    _IDX_ANCHOR_INTEG_CAP,
    _IDX_ANCHOR_LEAK,
    _S_ANCHOR_INTEG_TAU,
    pack_state_k2,
)
from wheeled_biped.wbc.offline_three_arm_counterfactual import (  # noqa: E402
    compute_v3_torque_for_state,
    init_v3_controller,
)

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
DT = 0.01
SUBSTEPS = 5
TOTAL_S = 25.0
SETTLE_S = 5.0
WINDOW_S = 20.0
BASE_SEED = 20260727

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
VARIANT = "nominal"
for _i, _a in enumerate(sys.argv):
    if _a == "--variant":
        VARIANT = sys.argv[_i + 1]
_nom = json.load(open(ROOT / DV / f"variant_{VARIANT}__variant_setup.json"))
H0 = float(_nom["target_com_z_m"])
POSTURE = np.array([
    _nom["hip_roll_left"], _nom["hip_yaw_left"], _nom["hip_pitch_ref"],
    _nom["knee_ref"], 0.0, _nom["hip_roll_right"], _nom["hip_yaw_right"],
    _nom["hip_pitch_ref"], _nom["knee_ref"], 0.0,
])
JOINTS = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
          "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]

# Signals pulled from the controller diag vector each step.
DIAG_COLS = {
    "tau_position": _D_TAU_POSITION,
    "ext_trim": _D_EXTERNAL_POS_TRIM,
    "tau_pitch": _D_TAU_PITCH,
    "tau_wheel_l": _D_TAU_WHEEL_L,
    "tau_wheel_r": _D_TAU_WHEEL_R,
    "lb_offset_deg": _D_LOW_BAND,
    "physics_ff_tau": _D_PHYSICS_FF,
}


KPOS = 40.0        # k2_jax_controller.py: kpos = 40.0 (K2 vd_k_position)
KP_PITCH = 50.0    # ACC idle value (anchor_kp_pitch_soft=35 only while recovering)


def _axle_y(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """Sagittal (world-y) midpoint of the two wheel axles."""
    return 0.5 * (float(data.xpos[model.body("l_wheel_link").id][1])
                  + float(data.xpos[model.body("r_wheel_link").id][1]))


def trial(model, seed, overrides=None):
    rng = np.random.default_rng(seed)
    v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
    v3["jax_state"] = pack_state_k2()
    p = v3["jax_params"]
    p = p.at[94].set(50.0)  # S1/L3 = ACC as measured in Table tab:standing
    for i, val in (overrides or {}).items():
        p = p.at[int(i)].set(float(val))
    v3["jax_params"] = p

    data = mujoco.MjData(model)
    q = POSTURE + rng.normal(0.0, 0.005, size=10)
    for j, jn in enumerate(JOINTS):
        lo, hi = model.jnt_range[model.joint(jn).id]
        q[j] = float(np.clip(q[j], lo, hi))
    data.qpos[7:17] = q
    data.qpos[2] = float(_nom["calibrated_root_z_m"]) + rng.normal(0.0, 0.001)
    mujoco.mj_forward(model, data)
    ctx = P._build_v3_controller_context(model, data, v3, eq_joint=POSTURE,
                                         height_ref=H0)

    m_total = float(sum(model.body_mass))
    g = float(abs(model.opt.gravity[2]))
    n = int(TOTAL_S / DT)
    log = {k: np.zeros(n) for k in
           list(DIAG_COLS) + ["anchor_integ", "sag_mm", "lat_mm", "grav_tau",
                              "com_lever_mm", "pitch_deg", "sag_pos_err_mm"]}
    by0 = float(data.qpos[1])
    bx0 = float(data.qpos[0])

    for k in range(n):
        # Integral value that THIS step's tau_position was built from (the
        # state array is updated in-place by the step, so read it first).
        integ_pre = float(np.asarray(v3["jax_state"])[_S_ANCHOR_INTEG_TAU])
        r = compute_v3_torque_for_state(data, model, v3["jax_step_fn"],
                                        v3["jax_state"], v3["jax_params"], ctx,
                                        teleop=None)
        diag = np.asarray(r["diagnostics"])
        for name, idx in DIAG_COLS.items():
            log[name][k] = float(diag[idx])
        v3["jax_state"] = r["next_jax_state"]
        log["anchor_integ"][k] = integ_pre

        # Sagittal gravity moment about the wheel-axle line, pre-step state.
        lever = float(data.subtree_com[0][1]) - _axle_y(model, data)
        log["com_lever_mm"][k] = lever * 1000.0
        log["grav_tau"][k] = m_total * g * lever
        log["sag_mm"][k] = (float(data.qpos[1]) - by0) * 1000.0
        log["lat_mm"][k] = (float(data.qpos[0]) - bx0) * 1000.0
        # Invert the position law to recover the controller's own error signal:
        #   tau_position = -kpos*e + I + trim   =>   e = (I + trim - tau_pos)/kpos
        log["sag_pos_err_mm"][k] = 1000.0 * (
            integ_pre + float(diag[_D_EXTERNAL_POS_TRIM])
            - float(diag[_D_TAU_POSITION])
        ) / KPOS

        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

        # Sagittal attitude is the controller's pitch_x = rotation about world x
        # (orientation_utils.compute_robot_frame_orientation_from_quaternion:
        #  body_pitch_x = roll). The ZYX "pitch" is the LATERAL axis here.
        w, x, y, z = data.qpos[3:7]
        pitch = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        log["pitch_deg"][k] = pitch
        if abs(pitch) > 46 or data.qpos[2] < 0.15:
            return None

    sl = slice(int(SETTLE_S / DT), int(SETTLE_S / DT) + int(WINDOW_S / DT))
    out = {k: float(np.mean(v[sl])) for k, v in log.items()}
    out["sag_rep_mm"] = float(np.std(log["sag_mm"][sl]))
    out["anchor_integ_max"] = float(np.max(np.abs(log["anchor_integ"][sl])))
    out["m_total_kg"] = m_total
    # Closed-form prediction: at DC the leaky integrator is a first-order lag
    # with finite gain k_I,dc = ki*dt*(1-leak)/leak, so the position loop's
    # total DC stiffness is kpos + k_I,dc and the offset it needs to cancel
    # the standing pitch torque is (|tau_pitch| - trim) / that stiffness.
    ki = float(np.asarray(v3["jax_params"])[_IDX_ANCHOR_KI])
    leak = float(np.asarray(v3["jax_params"])[_IDX_ANCHOR_LEAK])
    k_i_dc = ki * DT * (1.0 - leak) / leak if leak > 0 else float("inf")
    out["ki"] = ki
    out["leak"] = leak
    out["k_i_dc_nm_per_m"] = k_i_dc
    out["pred_err_mm"] = 1000.0 * (abs(out["tau_pitch"]) - out["ext_trim"]) / (
        KPOS + k_i_dc)
    out["pred_integ_nm"] = k_i_dc * abs(out["sag_pos_err_mm"]) / 1000.0
    out["integ_cap_nm"] = float(np.asarray(v3["jax_params"])[_IDX_ANCHOR_INTEG_CAP])
    out["integ_frac_of_cap"] = out["anchor_integ_max"] / out["integ_cap_nm"]

    # Pitch-reference decomposition. tau_pitch = kp_pitch * effective_pitch_x
    # (pitch_bias_comp_tau is 0 for this profile), and
    #   effective_pitch_x = pitch_raw - pitch_x_eq - (ol_ref + lb + physics_eq)
    # so the total commanded reference offset is recoverable from the torque.
    out["eff_pitch_deg"] = float(np.degrees(out["tau_pitch"] / KP_PITCH))
    out["pitch_ref_offset_deg"] = out["pitch_deg"] - out["eff_pitch_deg"]
    from wheeled_biped.controllers.physics_equilibrium_feedforward import (
        physics_equilibrium_pitch_eq_no_off_deg as _pitch_eq)
    out["physics_pitch_eq_deg"] = float(_pitch_eq(H0))
    out["ol_ref_deg_residual"] = (out["pitch_ref_offset_deg"]
                                  - out["lb_offset_deg"]
                                  - out["physics_pitch_eq_deg"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=3)
    ap.add_argument("--ki", type=float, default=None)
    ap.add_argument("--leak", type=float, default=None)
    ap.add_argument("--cap", type=float, default=None)
    ap.add_argument("--tag", default="acc")
    ap.add_argument("--variant", default="nominal")
    ap.add_argument("--pitch-eq-bias", type=float, default=None,
                    help="Add DEG to the physics-equilibrium pitch_eq feedforward "
                         "grid. Tests whether the parking offset is caused by a "
                         "miscalibrated pitch reference rather than by loop gain.")
    ap.add_argument("--out", default="outputs/paper_verification/parking_offset_diag.json")
    a = ap.parse_args()

    if a.pitch_eq_bias is not None:
        # Mutate the module-level grid before any JIT trace reads it as a closure
        # constant (k2_jax_controller:3133 `ff_grid = _physics_ff_grid_cache`).
        import wheeled_biped.controllers.k2_jax_controller as _k2
        _k2._physics_ff_grid_cache["pitch_eq_grid"] = (
            _k2._physics_ff_grid_cache["pitch_eq_grid"] + a.pitch_eq_bias)

    ov = {}
    if a.ki is not None:
        ov[_IDX_ANCHOR_KI] = a.ki
    if a.leak is not None:
        ov[_IDX_ANCHOR_LEAK] = a.leak
    if a.cap is not None:
        ov[_IDX_ANCHOR_INTEG_CAP] = a.cap

    model = mujoco.MjModel.from_xml_path(str(P.get_model_path()))
    trials = []
    for i in range(a.n):
        r = trial(model, BASE_SEED + i, ov)
        if r is None:
            print(f"# trial {i} FELL", file=sys.stderr)
            continue
        trials.append(r)
        print(f"# trial {i}  sag={r['sag_mm']:+7.2f}mm  e={r['sag_pos_err_mm']:+7.2f}mm "
              f"(pred {-r['pred_err_mm']:+7.2f})  I={r['anchor_integ']:.4f}Nm "
              f"(pred {r['pred_integ_nm']:.4f}, {100*r['integ_frac_of_cap']:.1f}% of cap)  "
              f"tau_pos={r['tau_position']:+.4f}  tau_pitch={r['tau_pitch']:+.4f}  "
              f"grav={r['grav_tau']:+.4f}Nm  pitch={r['pitch_deg']:+.3f}deg",
              file=sys.stderr, flush=True)

    agg = {"tag": a.tag, "overrides": {str(k): v for k, v in ov.items()},
           "n_trials": a.n, "n_survived": len(trials)}
    if trials:
        for k in trials[0]:
            arr = np.array([t[k] for t in trials])
            agg[k] = {"mean": float(np.mean(arr)),
                      "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}

    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    all_out = json.load(dest.open()) if dest.exists() else {}
    all_out[a.tag] = agg
    json.dump(all_out, dest.open("w"), indent=2)
    print("@@JSON@@" + json.dumps(agg))


if __name__ == "__main__":
    main()
