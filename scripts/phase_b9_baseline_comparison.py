"""Phase B.9 Task 3: Fair baseline comparison of all controller candidates.

Compares:
1. geometric_lqr_ik (Phase B.5 baseline)
2. height_scheduled_dynamic_lqr_ik (Phase B.6 best prior)
3. height_ik_wheel_lqr_only_b8 (Phase B.8 candidate)
4. hierarchical_vmc_lqr (Phase B.7 full hierarchical)
5. hierarchical_vmc_lqr_v2 (Phase B.8 Option C)
6. hierarchical_vmc_lqr_v3 (Phase B.8 Option A)

Evaluation:
- Heights: 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40
- Episodes: 20 per height
- No push disturbances
- No domain randomization
- PID enabled, action bias disabled
- Fixed seeds for reproducibility

Metrics:
- survival_time, fall_rate
- pitch_RMS_deg, roll_RMS_deg
- height_RMSE, CoM_error_RMS
- wheel_speed_RMS, wheel_cmd_RMS
- action_saturation_rate, action_rate_RMS
"""

import json
from pathlib import Path
from typing import Dict, List

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd
from tqdm import tqdm

from wheeled_biped.controllers.lqr_ik_prior import create_lqr_ik_prior
from wheeled_biped.controllers.hierarchical_vmc_lqr import create_hierarchical_vmc_controller
from wheeled_biped.utils.config import get_model_path


def compute_metrics(
    obs_history: np.ndarray,
    action_history: np.ndarray,
    reward_history: np.ndarray,
    done_history: np.ndarray,
    height_cmd: float,
) -> Dict[str, float]:
    """Compute evaluation metrics from episode history.

    Args:
        obs_history: (T, obs_dim) observation history
        action_history: (T, 10) action history
        reward_history: (T,) reward history
        done_history: (T,) done flags
        height_cmd: Commanded height [m]

    Returns:
        Dictionary of metrics
    """
    # Find survival time
    fall_idx = np.where(done_history)[0]
    if len(fall_idx) > 0:
        survival_steps = fall_idx[0] + 1
        fell = True
    else:
        survival_steps = len(done_history)
        fell = False

    survival_time = survival_steps * 0.02  # 50 Hz control

    # Extract state variables from observations
    # obs format: [g_body(3), body_lin_vel(3), body_ang_vel(3), qpos(10), qvel(10),
    #              prev_action(10), height_cmd_norm(1), current_height(1), yaw_error(1)]
    g_body = obs_history[:survival_steps, 0:3]
    body_ang_vel = obs_history[:survival_steps, 6:9]
    qvel = obs_history[:survival_steps, 19:29]
    current_height = obs_history[:survival_steps, 40]

    # Compute pitch and roll from gravity vector
    pitch = -np.arcsin(np.clip(g_body[:, 1], -1.0, 1.0))
    roll = np.arcsin(np.clip(g_body[:, 0], -1.0, 1.0))

    # Pitch and roll RMS
    pitch_RMS_deg = np.sqrt(np.mean(pitch**2)) * 180.0 / np.pi
    roll_RMS_deg = np.sqrt(np.mean(roll**2)) * 180.0 / np.pi

    # Height tracking error
    height_error = current_height - height_cmd
    height_RMSE = np.sqrt(np.mean(height_error**2))

    # Wheel velocities (indices 4 and 9)
    wheel_vel = qvel[:, [4, 9]]
    wheel_speed_RMS = np.sqrt(np.mean(wheel_vel**2))

    # Wheel commands from actions (indices 4 and 9)
    wheel_cmd = action_history[:survival_steps, [4, 9]]
    wheel_cmd_RMS = np.sqrt(np.mean(wheel_cmd**2))

    # Action saturation rate
    action_abs = np.abs(action_history[:survival_steps])
    action_saturation_rate = np.mean(action_abs > 0.95)

    # Action rate
    if survival_steps > 1:
        action_diff = np.diff(action_history[:survival_steps], axis=0)
        action_rate_RMS = np.sqrt(np.mean(action_diff**2))
    else:
        action_rate_RMS = 0.0

    # CoM error (approximate from pitch - not exact but indicative)
    # For proper CoM error, would need to compute from full kinematics
    # Using pitch as proxy: larger pitch deviation suggests larger CoM error
    CoM_error_RMS = pitch_RMS_deg * 0.01  # Rough approximation

    return {
        "survival_time": float(survival_time),
        "fall_rate": float(fell),
        "pitch_RMS_deg": float(pitch_RMS_deg),
        "roll_RMS_deg": float(roll_RMS_deg),
        "height_RMSE": float(height_RMSE),
        "CoM_error_RMS": float(CoM_error_RMS),
        "wheel_speed_RMS": float(wheel_speed_RMS),
        "wheel_cmd_RMS": float(wheel_cmd_RMS),
        "action_saturation_rate": float(action_saturation_rate),
        "action_rate_RMS": float(action_rate_RMS),
    }


def _settle(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, n_steps: int = 200) -> None:
    """Damped settle from keyframe."""
    for _ in range(n_steps):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)


def _reset_to_keyframe(mj_model: mujoco.MjModel, mj_data: mujoco.MjData) -> None:
    mujoco.mj_resetData(mj_model, mj_data)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    _settle(mj_model, mj_data)


def _build_obs(
    mj_data: mujoco.MjData,
    prev_action: jnp.ndarray,
    height_cmd_norm: jnp.ndarray,
    initial_yaw: float,
) -> jnp.ndarray:
    """Build 42-dim observation from MuJoCo state."""
    from wheeled_biped.utils.math_utils import (
        get_gravity_in_body_frame,
        quat_conjugate,
        quat_rotate,
        quat_to_euler,
        wrap_angle,
    )

    torso_quat = jnp.array(mj_data.qpos[3:7])
    gravity_body = get_gravity_in_body_frame(torso_quat)
    quat_inv = quat_conjugate(torso_quat)
    body_lin_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[:3]))
    body_ang_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[3:6]))
    joint_pos = jnp.array(mj_data.qpos[7:17])
    joint_vel = jnp.array(mj_data.qvel[6:16])
    current_yaw = float(quat_to_euler(torso_quat)[2])
    yaw_error = jnp.array([wrap_angle(current_yaw - initial_yaw)])

    min_h, max_h = 0.40, 0.70
    current_height_norm = jnp.array(
        [float(jnp.clip((jnp.array(mj_data.qpos[2]) - min_h) / (max_h - min_h), 0.0, 1.0))]
    )

    obs = jnp.concatenate([
        gravity_body,
        body_lin_vel,
        body_ang_vel,
        joint_pos,
        joint_vel,
        prev_action,
        height_cmd_norm,
        current_height_norm,
        yaw_error,
    ])
    return obs


def _is_fallen(mj_data: mujoco.MjData, max_tilt_rad: float = 0.8, min_height: float = 0.3) -> bool:
    """Check termination using gravity-based tilt."""
    from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

    torso_z = float(mj_data.qpos[2])
    torso_quat = jnp.array(mj_data.qpos[3:7], dtype=jnp.float32)
    g_body = get_gravity_in_body_frame(torso_quat)
    tilt = float(jnp.arccos(jnp.clip(-g_body[2], -1.0, 1.0)))
    return torso_z < min_height or tilt > max_tilt_rad


def _compute_ctrl(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    control_action: jnp.ndarray,
    pid_integral: jnp.ndarray,
    kp: jnp.ndarray,
    ki: jnp.ndarray,
    kd: jnp.ndarray,
    joint_mins: jnp.ndarray,
    joint_maxs: jnp.ndarray,
    wheel_mask: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    wheel_vel_limit: float,
    i_limit: float,
    control_dt: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute actuator ctrl from normalized action via PID."""
    joint_pos = jnp.array(mj_data.qpos[7:17])
    joint_vel = jnp.array(mj_data.qvel[6:16])

    pos_target = joint_mins + (control_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
    vel_target_whl = control_action * wheel_vel_limit
    pos_err = pos_target - joint_pos
    error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_whl - joint_vel)
    d_error = (1.0 - wheel_mask) * (-joint_vel)

    pid_integral = jnp.clip(
        pid_integral + error * control_dt,
        -i_limit,
        i_limit,
    )
    ctrl = jnp.clip(
        kp * error + kd * d_error + ki * pid_integral,
        ctrl_min,
        ctrl_max,
    )
    return ctrl, pid_integral


def evaluate_controller(
    controller_name: str,
    controller_config_path: str,
    heights: List[float],
    episodes_per_height: int,
    seeds: List[int],
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate a controller across multiple heights using actual MuJoCo physics.

    Args:
        controller_name: Name of controller for logging
        controller_config_path: Path to controller config YAML
        heights: List of heights to evaluate [m]
        episodes_per_height: Number of episodes per height
        seeds: List of seeds to use
        output_dir: Output directory for results

    Returns:
        DataFrame with per-episode results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {controller_name}")
    print(f"Config: {controller_config_path}")
    print(f"{'='*60}\n")

    # Load MuJoCo model
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Create controller - detect type from config structure
    import yaml
    with open(controller_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    # Hierarchical configs have layer-specific keys (vmc_enabled, lqr_height_scheduled, etc.)
    is_hierarchical = any(key in config_data for key in ['vmc_enabled', 'lqr_height_scheduled', 'roll_yaw_enabled'])

    if is_hierarchical:
        controller = create_hierarchical_vmc_controller(controller_config_path, mj_model)
    else:
        controller = create_lqr_ik_prior(controller_config_path, mj_model)

    # PID parameters (matching balance_residual.yaml)
    kp = jnp.array([55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0], dtype=jnp.float32)
    ki = jnp.array([0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1], dtype=jnp.float32)
    kd = jnp.array([3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0], dtype=jnp.float32)
    wheel_vel_limit = 20.0
    i_limit = 0.4
    alpha = 0.5
    control_dt = 0.02
    n_substeps = max(1, int(round(control_dt / float(mj_model.opt.timestep))))

    # Joint limits
    joint_names = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                   "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]
    j_mins, j_maxs = [], []
    for n in joint_names:
        jid = mj_model.joint(n).id
        jrange = mj_model.jnt_range[jid]
        j_mins.append(float(jrange[0]))
        j_maxs.append(float(jrange[1]))
    joint_mins = jnp.array(j_mins, dtype=jnp.float32)
    joint_maxs = jnp.array(j_maxs, dtype=jnp.float32)
    wheel_mask = jnp.array([1.0 if "wheel" in n else 0.0 for n in joint_names], dtype=jnp.float32)

    ctrl_range = np.array(mj_model.actuator_ctrlrange)
    ctrl_min = jnp.array(ctrl_range[:, 0])
    ctrl_max = jnp.array(ctrl_range[:, 1])

    # Results storage
    results = []

    # Evaluate each height
    for height in heights:
        print(f"\nHeight: {height:.2f}m")

        for ep_idx in range(episodes_per_height):
            seed = seeds[ep_idx % len(seeds)]
            np.random.seed(seed)

            # Reset MuJoCo
            mj_data = mujoco.MjData(mj_model)
            _reset_to_keyframe(mj_model, mj_data)

            # Reset controller
            controller.reset(height_cmd_m=height)

            # Episode state
            prev_action = jnp.zeros(mj_model.nu)
            pid_integral = jnp.zeros(mj_model.nu)
            height_cmd_norm = jnp.array([(height - 0.40) / (0.70 - 0.40)])

            from wheeled_biped.utils.math_utils import quat_to_euler
            initial_yaw = float(quat_to_euler(jnp.array(mj_data.qpos[3:7]))[2])

            # Telemetry
            obs_history = []
            action_history = []
            fell = False
            max_steps = 1000

            for step in range(max_steps):
                # Build observation
                obs = _build_obs(mj_data, prev_action, height_cmd_norm, initial_yaw)

                # Controller action
                raw_action = jnp.array(controller.compute_action(np.array(obs)), dtype=jnp.float32)

                # Action smoothing
                smooth_action = alpha * prev_action + (1.0 - alpha) * raw_action

                # Store telemetry
                obs_history.append(np.array(obs))
                action_history.append(np.array(smooth_action))

                # Low-level control
                ctrl, pid_integral = _compute_ctrl(
                    mj_model, mj_data, smooth_action, pid_integral,
                    kp, ki, kd, joint_mins, joint_maxs, wheel_mask,
                    ctrl_min, ctrl_max, wheel_vel_limit, i_limit, control_dt
                )

                # Step physics
                mj_data.ctrl[:] = np.array(ctrl)
                for _ in range(n_substeps):
                    mujoco.mj_step(mj_model, mj_data)

                prev_action = smooth_action

                # Termination check
                if _is_fallen(mj_data):
                    fell = True
                    break

            # Compute metrics
            obs_history = np.array(obs_history)
            action_history = np.array(action_history)
            reward_history = np.zeros(len(obs_history))
            done_history = np.zeros(len(obs_history), dtype=bool)
            if fell:
                done_history[-1] = True

            metrics = compute_metrics(
                obs_history, action_history, reward_history, done_history, height
            )

            # Store results
            result = {
                "controller": controller_name,
                "height": height,
                "episode": ep_idx,
                "seed": seed,
                **metrics,
            }
            results.append(result)

            print(f"  Episode {ep_idx+1}/{episodes_per_height} (seed={seed}): "
                  f"survival={metrics['survival_time']:.3f}s, "
                  f"fell={metrics['fall_rate']}")

    return pd.DataFrame(results)


def main():
    """Run Phase B.9 Task 3: Fair baseline comparison."""
    # Configuration
    heights = [0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    episodes_per_height = 20
    seeds = list(range(42, 42 + episodes_per_height))

    output_dir = Path("outputs/phase_b9_baseline_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Controller configurations
    controllers = [
        ("geometric_lqr_ik", "configs/controllers/geometric_lqr_ik.yaml"),
        ("height_scheduled_dynamic_lqr_ik", "configs/controllers/height_scheduled_dynamic_lqr.yaml"),
        ("height_ik_wheel_lqr_only_b8", "configs/controllers/height_ik_wheel_lqr_only_b8.yaml"),
        ("hierarchical_vmc_lqr", "configs/controllers/hierarchical_vmc_lqr.yaml"),
        ("hierarchical_vmc_lqr_v2", "configs/controllers/hierarchical_vmc_lqr_v2.yaml"),
        ("hierarchical_vmc_lqr_v3", "configs/controllers/hierarchical_vmc_lqr_v3.yaml"),
    ]

    # Evaluate each controller
    all_results = []
    for controller_name, config_path in controllers:
        if not Path(config_path).exists():
            print(f"WARNING: Config not found: {config_path}, skipping {controller_name}")
            continue

        df = evaluate_controller(
            controller_name, config_path, heights, episodes_per_height, seeds, output_dir
        )
        all_results.append(df)

    # Combine results
    results_df = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    results_df.to_csv(output_dir / "comparison_per_episode.csv", index=False)

    # Compute per-height summary
    per_height_summary = results_df.groupby(["controller", "height"]).agg({
        "survival_time": ["mean", "std"],
        "fall_rate": "mean",
        "pitch_RMS_deg": ["mean", "std"],
        "roll_RMS_deg": ["mean", "std"],
        "height_RMSE": ["mean", "std"],
        "CoM_error_RMS": ["mean", "std"],
        "wheel_speed_RMS": ["mean", "std"],
        "wheel_cmd_RMS": ["mean", "std"],
        "action_saturation_rate": ["mean", "std"],
        "action_rate_RMS": ["mean", "std"],
    }).reset_index()
    # Flatten MultiIndex columns for JSON serialization
    per_height_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                   for col in per_height_summary.columns.values]
    per_height_summary.to_csv(output_dir / "comparison_per_height.csv", index=False)

    # Compute overall summary
    overall_summary = results_df.groupby("controller").agg({
        "survival_time": ["mean", "std"],
        "fall_rate": "mean",
        "pitch_RMS_deg": ["mean", "std"],
        "roll_RMS_deg": ["mean", "std"],
        "height_RMSE": ["mean", "std"],
        "CoM_error_RMS": ["mean", "std"],
        "wheel_speed_RMS": ["mean", "std"],
        "wheel_cmd_RMS": ["mean", "std"],
        "action_saturation_rate": ["mean", "std"],
        "action_rate_RMS": ["mean", "std"],
    }).reset_index()
    # Flatten MultiIndex columns for JSON serialization
    overall_summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                                for col in overall_summary.columns.values]
    overall_summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    # Save JSON results
    results_json = {
        "config": {
            "heights": heights,
            "episodes_per_height": episodes_per_height,
            "seeds": seeds,
        },
        "controllers": [c[0] for c in controllers if Path(c[1]).exists()],
        "per_height_summary": per_height_summary.to_dict(orient="records"),
        "overall_summary": overall_summary.to_dict(orient="records"),
    }
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(overall_summary.to_string())
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
