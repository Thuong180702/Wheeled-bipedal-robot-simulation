
import json, sys, time
import numpy as np
import mujoco as mj
from wheeled_biped.wbc.offline_three_arm_counterfactual import (
    compute_v3_torque_for_state, init_v3_controller)
from wheeled_biped.controllers.k2_jax_controller import pack_state_k2
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator, CentroidalStateEstimatorConfig)
from wheeled_biped.utils.config import get_model_path

DT = 0.01; SUBSTEPS = 5
PUSH_DUR = 7; PUSH_START = 300; POST_PUSH_S = 17.0
POST_PUSH_STEPS = int(POST_PUSH_S / DT)
PITCH_LIMIT = 0.8; HEIGHT_LIMIT = 0.30
FORCE_MIN, FORCE_MAX = 10.0, 160.0
N_BISECT = 8; TOLERANCE = 5.0
N_REPS = 10
ANGLES = [0, 45, 90, 135, 180, -135, -90, -45]

DV = "archive/cleanup_2026-06-13/output_summaries/balance_core_true_height_variants"
nom = json.load(open(f"{DV}/variant_nominal__variant_setup.json"))
H0 = float(nom["target_com_z_m"])
POSTURE = np.array([nom["hip_roll_left"], nom["hip_yaw_left"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0,
    nom["hip_roll_right"], nom["hip_yaw_right"],
    nom["hip_pitch_ref"], nom["knee_ref"], 0.0])
ROOT_Z = float(nom["calibrated_root_z_m"])
JOINT_NAMES = ["l_hip_roll","l_hip_yaw","l_hip_pitch","l_knee","l_wheel",
               "r_hip_roll","r_hip_yaw","r_hip_pitch","r_knee","r_wheel"]

PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
param_overrides = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
# argv[2] = degrees added to the physics-equilibrium pitch_eq feedforward grid.
# Mutated before any JIT trace reads it as a closure constant.
if len(sys.argv) > 2 and float(sys.argv[2]) != 0.0:
    import wheeled_biped.controllers.k2_jax_controller as _k2
    _k2._physics_ff_grid_cache["pitch_eq_grid"] = (
        _k2._physics_ff_grid_cache["pitch_eq_grid"] + float(sys.argv[2]))

model = mj.MjModel.from_xml_path(str(get_model_path()))
l_wheel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "l_wheel_link")
r_wheel_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "r_wheel_link")
robot_mass = float(np.sum(model.body_mass))
torso_inertia = np.array(model.body_inertia[1], dtype=np.float64)
cfg = CentroidalStateEstimatorConfig(robot_mass=robot_mass, torso_inertia=torso_inertia)
est = CentroidalStateEstimator(cfg, mj_model=model)
CTX = {"centroidal_estimator": est, "initial_yaw_z": 0.0,
       "l_wheel_id": l_wheel_id, "r_wheel_id": r_wheel_id,
       "eq_joint": POSTURE, "height_ref": H0, "prev_com_pos": None}

def settle(data, v3, steps=300):
    for _ in range(steps):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], CTX, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        for _ in range(SUBSTEPS):
            mj.mj_step(model, data)

def run_push(data, v3, force_N, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    force = np.array([force_N * np.cos(angle_rad), force_N * np.sin(angle_rad), 0.0])
    torso_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "torso")
    for step in range(POST_PUSH_STEPS + PUSH_DUR):
        r = compute_v3_torque_for_state(
            data, model, v3["jax_step_fn"], v3["jax_state"],
            v3["jax_params"], CTX, teleop=None)
        v3["jax_state"] = r["next_jax_state"]
        data.ctrl[:] = np.array(r["tau_v3"])
        data.xfrc_applied[torso_id, :3] = 0.0
        if step < PUSH_DUR:
            data.xfrc_applied[torso_id, :3] = force
        for _ in range(SUBSTEPS):
            mj.mj_step(model, data)
        quat = data.qpos[3:7].copy()
        pitch = np.arcsin(-2*(quat[1]*quat[3] - quat[0]*quat[2]))
        if abs(pitch) > PITCH_LIMIT or data.subtree_com[0][2] < HEIGHT_LIMIT:
            return False
    return True

def bisect_one_direction(angle_deg, seed):
    rng = np.random.default_rng(seed)
    lo, hi = FORCE_MIN, FORCE_MAX
    best = lo
    for i in range(N_BISECT):
        mid = (lo + hi) / 2
        data = mj.MjData(model)
        perturbed = POSTURE + rng.normal(0.0, 0.005, size=10)
        for j, jname in enumerate(JOINT_NAMES):
            jid = model.joint(jname).id
            lo_j, hi_j = model.jnt_range[jid]
            perturbed[j] = float(np.clip(perturbed[j], lo_j, hi_j))
        data.qpos[7:17] = perturbed
        data.qpos[2] = ROOT_Z + rng.normal(0.0, 0.001)
        mj.mj_forward(model, data)
        v3 = dict(init_v3_controller(profile_name=PROFILE, model=model))
        v3["jax_state"] = pack_state_k2()
        # Apply parameter overrides
        params = v3["jax_params"]
        for idx, val in param_overrides.items():
            params = params.at[int(idx)].set(float(val))
        v3["jax_params"] = params
        settle(data, v3, steps=PUSH_START)
        if run_push(data, v3, mid, angle_deg):
            best = mid
            lo = mid + TOLERANCE / 2
        else:
            hi = mid - TOLERANCE / 2
    return best

t0 = time.time()
all_reps = {ang: [] for ang in ANGLES}
for rep in range(N_REPS):
    seed = 20260728 * 100 + rep
    for ang in ANGLES:
        th = bisect_one_direction(ang, seed + int(ang))
        all_reps[ang].append(th)
    elapsed = time.time() - t0
    if rep < N_REPS - 1:
        eta = elapsed / (rep+1) * (N_REPS - rep - 1)

# Per-rep F_min and F_med
f_min_vals = []
f_med_vals = []
for rep in range(N_REPS):
    rep_vals = [all_reps[ang][rep] for ang in ANGLES]
    f_min_vals.append(min(rep_vals))
    f_med_vals.append(sorted(rep_vals)[len(rep_vals)//2])

f_min_mean = float(np.mean(f_min_vals))
f_min_std = float(np.std(f_min_vals, ddof=1))
f_med_mean = float(np.mean(f_med_vals))
f_med_std = float(np.std(f_med_vals, ddof=1))
ci95_min = float(2.262 * f_min_std / np.sqrt(N_REPS))  # t-dist for N=10, df=9
ci95_med = float(2.262 * f_med_std / np.sqrt(N_REPS))
elapsed = time.time() - t0

print(json.dumps({
    "n_reps": N_REPS, "n_directions": len(ANGLES),
    "F_min_mean": f_min_mean, "F_min_std": f_min_std,
    "F_med_mean": f_med_mean, "F_med_std": f_med_std,
    "F_min_ci95": ci95_min, "F_med_ci95": ci95_med,
    "f_min_per_rep": f_min_vals, "f_med_per_rep": f_med_vals,
    "all_reps": {str(k): v for k, v in all_reps.items()},
    "elapsed_min": elapsed / 60.0,
}))
