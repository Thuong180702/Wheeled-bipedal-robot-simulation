"""
Sweep roll PD gains for LQR controller to find the best possible performance.

Goal: demonstrate that even with aggressive tuning of the lateral balance
channel, a decoupled LQR+PD architecture cannot stabilize this 10-DOF
wheeled biped robot. This preempts the reviewer criticism "you didn't try
hard enough to make the baseline work."

Sweeps:
  kp_roll ∈ {0.4, 0.8, 1.2, 1.6, 2.0, 3.0, 5.0}
  kd_roll ∈ {0.08, 0.2, 0.5, 1.0, 2.0}

Plus: high-gain variants with wheel_vel_limit ∈ {20, 30, 40, 60} rad/s
to check if saturation is the bottleneck.
"""
import sys
from pathlib import Path
import json
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mujoco
from wheeled_biped.controllers.lqr_balance import LQRBalanceController
from wheeled_biped.utils.config import get_model_path
from wheeled_biped.utils.math_utils import (
    get_gravity_in_body_frame, quat_conjugate, quat_rotate, quat_to_euler,
)

MODEL_PATH = str(get_model_path())
NUM_EPISODES = 10
MAX_STEPS = 1000
CONTROL_DT = 0.02

# PID params (from baseline_lqr.yaml)
PID_KP = np.array([55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0])
PID_KI = np.array([0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1])
PID_KD = np.array([3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0])
WHEEL_MASK = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
I_LIMIT = 0.4
ALPHA = 0.4
DEFAULT_WHEEL_LIMIT = 20.0

JOINT_NAMES = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
]


def load_model_and_limits():
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    j_mins, j_maxs = [], []
    for n in JOINT_NAMES:
        jid = mj_model.joint(n).id
        jrange = mj_model.jnt_range[jid]
        j_mins.append(float(jrange[0]))
        j_maxs.append(float(jrange[1]))
    joint_mins = np.array(j_mins)
    joint_maxs = np.array(j_maxs)
    ctrl_range = mj_model.actuator_ctrlrange
    ctrl_min = np.array(ctrl_range[:, 0])
    ctrl_max = np.array(ctrl_range[:, 1])
    return mj_model, joint_mins, joint_maxs, ctrl_min, ctrl_max


def run_episode(mj_model, joint_mins, joint_maxs, ctrl_min, ctrl_max,
                controller, wheel_vel_limit, seed):
    """Run one episode, return (survival_steps, fell, pitch_rms, roll_rms)."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    for _ in range(200):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    rng = np.random.default_rng(seed)
    height_cmd = 0.65
    controller.reset(height_cmd_m=height_cmd)

    initial_yaw = float(quat_to_euler(np.array(mj_data.qpos[3:7]))[2])
    height_cmd_norm = np.array([(height_cmd - 0.40) / 0.30])
    prev_action = np.zeros(10)
    pid_integral = np.zeros(10)
    n_substeps = max(1, int(round(CONTROL_DT / float(mj_model.opt.timestep))))

    pitches, rolls = [], []
    fell = False
    step = 0

    for step in range(MAX_STEPS):
        torso_quat = np.array(mj_data.qpos[3:7])
        g_body = get_gravity_in_body_frame(torso_quat)
        quat_inv = quat_conjugate(torso_quat)
        body_lin_vel = quat_rotate(quat_inv, np.array(mj_data.qvel[:3]))
        body_ang_vel = quat_rotate(quat_inv, np.array(mj_data.qvel[3:6]))
        joint_pos = np.array(mj_data.qpos[7:17])
        joint_vel = np.array(mj_data.qvel[6:16])
        current_yaw = float(quat_to_euler(torso_quat)[2])
        yaw_error = float(current_yaw - initial_yaw)
        while yaw_error > np.pi:
            yaw_error -= 2*np.pi
        while yaw_error < -np.pi:
            yaw_error += 2*np.pi
        current_height_norm = np.clip((float(mj_data.qpos[2]) - 0.40) / 0.30, 0.0, 1.0)

        obs = np.concatenate([
            np.array(g_body), np.array(body_lin_vel), np.array(body_ang_vel),
            joint_pos, joint_vel, prev_action,
            height_cmd_norm, np.array([current_height_norm]), np.array([yaw_error]),
        ]).astype(np.float64)

        action = controller.compute_action(obs)
        smooth_action = ALPHA * prev_action + (1.0 - ALPHA) * action

        pos_target = joint_mins + (smooth_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
        vel_target_whl = smooth_action * wheel_vel_limit
        pos_err = pos_target - joint_pos
        error = (1.0 - WHEEL_MASK) * pos_err + WHEEL_MASK * (vel_target_whl - joint_vel)
        d_error = (1.0 - WHEEL_MASK) * (-joint_vel)
        pid_integral = np.clip(pid_integral + error * CONTROL_DT, -I_LIMIT, I_LIMIT)
        ctrl = np.clip(PID_KP * error + PID_KD * d_error + PID_KI * pid_integral, ctrl_min, ctrl_max)

        mj_data.ctrl[:] = ctrl
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        prev_action = smooth_action

        euler = quat_to_euler(torso_quat)
        pitches.append(float(np.degrees(euler[1])))
        rolls.append(float(np.degrees(euler[0])))

        # Termination check
        tilt = float(np.arccos(np.clip(-g_body[2], -1.0, 1.0)))
        if float(mj_data.qpos[2]) < 0.3 or tilt > 0.8:
            fell = True
            break

    step_count = step + 1
    pitch_arr = np.array(pitches)
    roll_arr = np.array(rolls)
    surv_time = step_count * CONTROL_DT

    return {
        "survival_s": surv_time,
        "fell": fell,
        "steps": step_count,
        "pitch_rms_deg": float(np.sqrt(np.mean(pitch_arr**2))) if len(pitch_arr) > 0 else 0.0,
        "roll_rms_deg": float(np.sqrt(np.mean(roll_arr**2))) if len(roll_arr) > 0 else 0.0,
    }


# =========================================================================
# SWEEP 1: Roll PD gains
# =========================================================================
print("=" * 72)
print("SWEEP 1: Roll PD gain sweep")
print("=" * 72)

kp_values = [0.4, 0.8, 1.2, 1.6, 2.0, 3.0, 5.0]
kd_values = [0.08, 0.2, 0.5, 1.0, 2.0]
seeds = [0, 42, 123]

mj_model, joint_mins, joint_maxs, ctrl_min, ctrl_max = load_model_and_limits()
results = []

for kp in kp_values:
    for kd in kd_values:
        controller = LQRBalanceController(
            model_path=MODEL_PATH,
            kp_roll=kp,
            kd_roll=kd,
        )

        ep_results = []
        for ep in range(NUM_EPISODES):
            seed = seeds[ep % len(seeds)] + ep * 100
            r = run_episode(mj_model, joint_mins, joint_maxs, ctrl_min, ctrl_max,
                           controller, DEFAULT_WHEEL_LIMIT, seed)
            ep_results.append(r)

        surv_mean = np.mean([r["survival_s"] for r in ep_results])
        surv_std = np.std([r["survival_s"] for r in ep_results])
        fall_rate = np.mean([r["fell"] for r in ep_results])
        roll_mean = np.mean([r["roll_rms_deg"] for r in ep_results])
        pitch_mean = np.mean([r["pitch_rms_deg"] for r in ep_results])

        results.append({
            "kp_roll": kp, "kd_roll": kd,
            "survival_s": round(surv_mean, 3),
            "survival_std": round(surv_std, 3),
            "fall_rate": round(fall_rate, 3),
            "roll_rms_deg": round(roll_mean, 2),
            "pitch_rms_deg": round(pitch_mean, 2),
        })

        status = "★ SURVIVES" if fall_rate < 1.0 else ""
        best_marker = ""
        print(f"  kp={kp:.1f} kd={kd:.2f} | surv={surv_mean:.3f}s ±{surv_std:.3f} | fall={fall_rate:.0%} | roll_rms={roll_mean:.1f}° | pitch_rms={pitch_mean:.2f}° {status}{best_marker}")

# Find best
best = max(results, key=lambda r: (1 - r["fall_rate"], r["survival_s"]))
print(f"\n  BEST: kp={best['kp_roll']:.1f} kd={best['kd_roll']:.2f} surv={best['survival_s']:.3f}s fall={best['fall_rate']:.0%}")

# =========================================================================
# SWEEP 2: Best roll gains + varied wheel_vel_limit
# =========================================================================
print("\n" + "=" * 72)
print("SWEEP 2: Best roll gains + wheel velocity limit sweep")
print("=" * 72)

best_kp = best["kp_roll"]
best_kd = best["kd_roll"]
wheel_limits = [20, 30, 40, 60, 100]

for wl in wheel_limits:
    controller = LQRBalanceController(
        model_path=MODEL_PATH,
        kp_roll=best_kp,
        kd_roll=best_kd,
    )
    # Override wheel vel limit
    controller._wheel_vel_limit = wl

    ep_results = []
    for ep in range(NUM_EPISODES):
        seed = seeds[ep % len(seeds)] + ep * 100
        r = run_episode(mj_model, joint_mins, joint_maxs, ctrl_min, ctrl_max,
                       controller, wl, seed)
        ep_results.append(r)

    surv_mean = np.mean([r["survival_s"] for r in ep_results])
    fall_rate = np.mean([r["fell"] for r in ep_results])
    roll_mean = np.mean([r["roll_rms_deg"] for r in ep_results])

    status = "★ SURVIVES" if fall_rate < 1.0 else ""
    print(f"  wl={wl:3.0f} rad/s | surv={surv_mean:.3f}s | fall={fall_rate:.0%} | roll_rms={roll_mean:.1f}° {status}")

# =========================================================================
# SWEEP 3: Roll→wheel cross-coupling (3D coordination)
# =========================================================================
print("\n" + "=" * 72)
print("SWEEP 3: Roll→wheel differential cross-coupling")
print("=" * 72)

# When robot leans left, right wheel speeds up to push that side up
# This couples roll into the wheel differential channel
# omega_diff += k_roll_cross * roll_error

# We need a custom controller for this — modify compute_action on the fly
class CrossCoupledLQRController(LQRBalanceController):
    def __init__(self, k_roll_cross=0.0, **kwargs):
        super().__init__(**kwargs)
        self.k_roll_cross = k_roll_cross

    def compute_action(self, obs):
        # Get base action from parent
        action = super().compute_action(obs)

        # Extract roll error (lean_left from gravity)
        lean_left = float(obs[0])  # g_body[0]
        lean_rate_left = float(obs[7])  # body_ang_vel[1]

        # Cross-coupling: roll error → differential wheel
        # Positive lean_left (lean left) → speed up RIGHT wheel
        # This creates a moment that pushes the robot back upright
        roll_correction = self.k_roll_cross * lean_left + 0.05 * lean_rate_left

        # Modify wheel commands: add differential
        current_l = float(action[4])
        current_r = float(action[9])
        action[4] = np.clip(current_l - roll_correction, -1.0, 1.0)
        action[9] = np.clip(current_r + roll_correction, -1.0, 1.0)

        return action

k_cross_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]

for kc in k_cross_values:
    controller = CrossCoupledLQRController(
        model_path=MODEL_PATH,
        kp_roll=best_kp,
        kd_roll=best_kd,
        k_roll_cross=kc,
    )

    ep_results = []
    for ep in range(NUM_EPISODES):
        seed = seeds[ep % len(seeds)] + ep * 100
        r = run_episode(mj_model, joint_mins, joint_maxs, ctrl_min, ctrl_max,
                       controller, DEFAULT_WHEEL_LIMIT, seed)
        ep_results.append(r)

    surv_mean = np.mean([r["survival_s"] for r in ep_results])
    fall_rate = np.mean([r["fell"] for r in ep_results])
    roll_mean = np.mean([r["roll_rms_deg"] for r in ep_results])

    status = "★ SURVIVES" if fall_rate < 1.0 else ""
    print(f"  k_cross={kc:5.1f} | surv={surv_mean:.3f}s | fall={fall_rate:.0%} | roll_rms={roll_mean:.1f}° {status}")

# =========================================================================
# Save results
# =========================================================================
out_path = PROJECT_ROOT / "outputs" / "balance" / "lqr_sweep_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "roll_sweep": results,
        "best_kp": best["kp_roll"],
        "best_kd": best["kd_roll"],
    }, f, indent=2)
print(f"\nResults saved → {out_path}")
