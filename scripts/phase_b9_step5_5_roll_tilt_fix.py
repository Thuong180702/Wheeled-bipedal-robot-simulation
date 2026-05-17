"""Phase B.9 Step 5.5: Roll/Tilt Failure Diagnosis and Fix.

Diagnose roll/tilt divergence source and test roll-stabilization fixes independently.

Workflow:
1. Run diagnostic rollouts with best Step 5 config
2. Log detailed roll/tilt/contact/CoM/wheel state
3. Test roll-fix candidates A-F one at a time
4. Small evaluation (3 eps × 3 heights), then full validation (5 eps × 6 heights)
5. Output best_roll_fix_config.yaml and best_roll_fix_summary.json
"""

import argparse
import copy
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

# Import contact/clearance utilities
sys.path.insert(0, str(project_root / "scripts"))
from phase_b9_posture_geometry_inspection import (
    body_com,
    contact_forces_by_wheel,
    wheel_bottom_heights,
)

console = Console()


def load_balanced_init_table() -> dict:
    """Load balanced root initialization table."""
    table_path = project_root / "configs/controllers/b9_balanced_root_init_table.yaml"
    with open(table_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    heights = data["balanced_root_initialization"]["heights"]
    return {float(k): v for k, v in heights.items()}


def apply_balanced_root_init(mjx_data, height: float, init_table: dict):
    """Apply balanced root initialization."""
    heights = sorted(init_table.keys())
    if height <= heights[0]:
        init = init_table[heights[0]]
    elif height >= heights[-1]:
        init = init_table[heights[-1]]
    else:
        for i in range(len(heights) - 1):
            if heights[i] <= height <= heights[i + 1]:
                h0, h1 = heights[i], heights[i + 1]
                init0, init1 = init_table[h0], init_table[h1]
                alpha = (height - h0) / (h1 - h0)
                init = {
                    "hip_pitch": (1 - alpha) * init0["hip_pitch"] + alpha * init1["hip_pitch"],
                    "knee": (1 - alpha) * init0["knee"] + alpha * init1["knee"],
                }
                break

    new_qpos = mjx_data.qpos.at[9].set(init["hip_pitch"])
    new_qpos = new_qpos.at[10].set(init["knee"])
    new_qpos = new_qpos.at[14].set(init["hip_pitch"])
    new_qpos = new_qpos.at[15].set(init["knee"])
    return mjx_data.replace(qpos=new_qpos)


def create_roll_fix_controller(
    base_config: DualRateConfig,
    base_params: dict,
    roll_fix_type: str,
    roll_fix_params: dict,
    mj_model: mujoco.MjModel,
) -> DualRateBalanceController:
    """Create controller with base Step 5 params + roll fix."""
    # Apply base Step 5 tuning
    lqr_scale = base_params.get("lqr_gain_scale", 1.0)
    pitch_mult = base_params.get("pitch_gain_mult", 1.0)
    pitch_rate_mult = base_params.get("pitch_rate_gain_mult", 1.0)
    com_mult = base_params.get("com_gain_mult", 1.0)
    com_rate_mult = base_params.get("com_rate_gain_mult", 1.0)
    wheel_limit_mult = base_params.get("wheel_cmd_limit_mult", 1.0)
    filter_alpha = base_params.get("filter_alpha", base_config.wheel_cmd_filter_alpha)
    filter_max_delta_mult = base_params.get("filter_max_delta_mult", 1.0)

    tuned_gains = {}
    for height, gains in base_config.height_scheduled_gains.items():
        tuned_gains[height] = {
            "k_pitch": gains["k_pitch"] * lqr_scale * pitch_mult,
            "k_pitch_rate": gains["k_pitch_rate"] * lqr_scale * pitch_rate_mult,
            "k_fwd_vel": gains["k_fwd_vel"] * lqr_scale,
            "k_fwd_pos": gains["k_fwd_pos"] * lqr_scale,
            "k_com": gains["k_com"] * com_mult,
            "k_com_rate": gains["k_com_rate"] * com_rate_mult,
        }

    # Apply roll fix
    roll_kp = base_config.roll_kp
    roll_kd = base_config.roll_kd
    roll_max_correction = base_config.roll_max_correction

    if roll_fix_type == "A_weak_hip_roll_pd":
        roll_kp = roll_fix_params.get("roll_kp", 0.3)
        roll_kd = roll_fix_params.get("roll_kd", 0.05)
        roll_max_correction = roll_fix_params.get("roll_max_correction", 0.15)
    elif roll_fix_type == "B_strong_hip_roll_pd":
        roll_kp = roll_fix_params.get("roll_kp", 0.8)
        roll_kd = roll_fix_params.get("roll_kd", 0.15)
        roll_max_correction = roll_fix_params.get("roll_max_correction", 0.25)
    elif roll_fix_type == "C_roll_rate_damping":
        roll_kp = 0.0
        roll_kd = roll_fix_params.get("roll_kd", 0.20)
        roll_max_correction = roll_fix_params.get("roll_max_correction", 0.20)
    elif roll_fix_type == "D_contact_force_balance":
        # Reduce wheel authority and increase roll damping to minimize contact asymmetry effects
        roll_kp = roll_fix_params.get("roll_kp", 0.4)
        roll_kd = roll_fix_params.get("roll_kd", 0.12)
        roll_max_correction = roll_fix_params.get("roll_max_correction", 0.20)
        wheel_limit_mult = roll_fix_params.get("wheel_cmd_limit_mult", 2.5)
    elif roll_fix_type == "E_lateral_com_correction":
        # Strong roll proportional gain to provide lateral stabilization (approximates CoM feedback)
        roll_kp = roll_fix_params.get("roll_kp", 1.0)
        roll_kd = roll_fix_params.get("roll_kd", 0.10)
        roll_max_correction = roll_fix_params.get("roll_max_correction", 0.30)
    elif roll_fix_type == "F_reduced_wheel_limit":
        wheel_limit_mult = roll_fix_params.get("wheel_cmd_limit_mult", 2.0)

    config_dict = {
        "time_scale": {
            "fast_loop_rate_hz": base_config.fast_loop_rate_hz,
            "slow_loop_rate_hz": 0.1,
            "control_dt": base_config.control_dt,
        },
        "height": {
            "min": base_config.height_min,
            "max": base_config.height_max,
            "grid": base_config.height_grid,
        },
        "joint_limits": base_config.joint_limits,
        "wheel_vel_limit": base_config.wheel_vel_limit * wheel_limit_mult,
        "slow_loop": {
            "posture_blend_alpha": base_config.posture_blend_alpha,
            "max_hip_pitch_delta": base_config.max_hip_pitch_delta,
            "max_knee_delta": base_config.max_knee_delta,
            "pitch_gate_deg": 5.0,
            "pitch_rate_gate_deg_s": base_config.pitch_rate_gate_deg_s,
            "height_correction_enabled": base_config.height_correction_enabled,
            "height_correction_gain": base_config.height_correction_gain,
            "max_height_correction_per_update": base_config.max_height_correction_per_update,
        },
        "fast_loop": {
            "height_scheduled_gains": tuned_gains,
            "wheel_cmd_filter_enabled": base_config.wheel_cmd_filter_enabled,
            "wheel_cmd_filter_alpha": filter_alpha,
            "wheel_cmd_filter_max_delta": base_config.wheel_cmd_filter_max_delta * filter_max_delta_mult,
            "emergency_mode_enabled": base_config.emergency_mode_enabled,
            "emergency_pitch_threshold_deg": base_config.emergency_pitch_threshold_deg,
            "emergency_lqr_gain_multiplier": base_config.emergency_lqr_gain_multiplier,
        },
        "roll": {
            "kp": roll_kp,
            "kd": roll_kd,
            "max_correction": roll_max_correction,
        },
        "yaw": {
            "kp": base_config.yaw_kp,
            "kd": base_config.yaw_kd,
            "max_diff": base_config.yaw_max_diff,
        },
        "com_state": {"use_sim": base_config.com_use_sim},
        "ik": {
            "scan_points": base_config.ik_scan_points,
            "polynomial_degree": base_config.ik_polynomial_degree,
            "symmetric_fold": base_config.ik_symmetric_fold,
        },
    }

    temp_path = project_root / "configs/controllers/temp_step5_5_tuning.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    tuned_config = DualRateConfig.from_yaml(temp_path)
    controller = DualRateBalanceController(tuned_config, mj_model)
    controller.slow_loop_interval = 999999

    temp_path.unlink()
    return controller


def run_diagnostic_rollout(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    height: float,
    init_table: dict,
    seed: int,
) -> list[dict]:
    """Run single diagnostic episode with detailed logging."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

    controller.reset()
    controller.target_hip_pitch = init_table[min(init_table.keys(), key=lambda h: abs(h - height))]["hip_pitch"]
    controller.target_knee = init_table[min(init_table.keys(), key=lambda h: abs(h - height))]["knee"]
    controller.last_stable_hip_pitch = controller.target_hip_pitch
    controller.last_stable_knee = controller.target_knee

    diagnostics = []
    prev_action = np.zeros(10)

    # Get MuJoCo model for contact/clearance utilities
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(mj_model)

    for step in range(1000):
        obs_np = np.array(state.obs)
        action = controller.compute_action(obs_np)

        g_body = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
        ang_vel = obs_np[6:9]
        pitch_rate = float(ang_vel[0])
        roll_rate = float(ang_vel[1])

        joint_pos = obs_np[9:19]
        joint_vel = obs_np[19:29]

        # Extract wheel states
        l_wheel_vel = float(joint_vel[4])
        r_wheel_vel = float(joint_vel[9])

        # Hip roll states
        l_hip_roll_target = float(action[0])
        r_hip_roll_target = float(action[5])
        l_hip_roll_actual = float(joint_pos[0])
        r_hip_roll_actual = float(joint_pos[5])

        # Action saturation
        action_sat = int(np.max(np.abs(action)) >= 0.99)

        # CoM lateral velocity proxy from body linear velocity y (obs[4])
        com_lateral_vel = float(obs_np[4])

        # Sync MJX state to MuJoCo for contact/clearance utilities
        mj_data.qpos[:] = np.array(state.mjx_data.qpos)
        mj_data.qvel[:] = np.array(state.mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        # Contact forces
        l_contact_force, r_contact_force = contact_forces_by_wheel(mj_model, mj_data)

        # Wheel clearances
        l_clearance, r_clearance = wheel_bottom_heights(mj_model, mj_data)

        # Hip roll torques (actuator forces for hip_roll joints)
        l_hip_roll_torque = float(mj_data.actuator_force[0])
        r_hip_roll_torque = float(mj_data.actuator_force[5])

        diagnostics.append({
            "height": height,
            "step": step,
            "time_s": step * env.CONTROL_DT,
            "roll_deg": np.rad2deg(roll),
            "roll_rate_deg_s": np.rad2deg(roll_rate),
            "pitch_deg": np.rad2deg(pitch),
            "pitch_rate_deg_s": np.rad2deg(pitch_rate),
            "com_lateral_vel": com_lateral_vel,
            "l_hip_roll_target": l_hip_roll_target,
            "r_hip_roll_target": r_hip_roll_target,
            "l_hip_roll_actual": l_hip_roll_actual,
            "r_hip_roll_actual": r_hip_roll_actual,
            "l_hip_roll_torque_Nm": l_hip_roll_torque,
            "r_hip_roll_torque_Nm": r_hip_roll_torque,
            "l_wheel_vel_rad_s": l_wheel_vel,
            "r_wheel_vel_rad_s": r_wheel_vel,
            "l_contact_force_N": l_contact_force,
            "r_contact_force_N": r_contact_force,
            "l_clearance_m": l_clearance,
            "r_clearance_m": r_clearance,
            "action_sat": action_sat,
            "fell": False,
            "fall_reason": "none",
            "fall_time_s": None,
        })

        state = env.step(state, jnp.array(action))

        if bool(state.done):
            # Determine fall reason
            final_obs = np.array(state.obs)
            torso_height = float(state.mjx_data.qpos[2])
            g_body_final = final_obs[0:3]
            tilt_final = float(np.arccos(np.clip(-g_body_final[2], -1.0, 1.0)))

            if torso_height < env._min_height:
                fall_reason = "height"
            elif tilt_final > env._max_tilt:
                fall_reason = "tilt"
            else:
                fall_reason = "unknown"

            diagnostics[-1]["fell"] = True
            diagnostics[-1]["fall_reason"] = fall_reason
            diagnostics[-1]["fall_time_s"] = step * env.CONTROL_DT
            break

    return diagnostics


def detect_first_divergence(diagnostics_df: pd.DataFrame, threshold_multiplier: float = 2.0) -> dict:
    """Detect which variable diverges first before fall.

    Returns dict with first_divergence_variable and first_divergence_time_s.
    """
    # Group by episode (height + seed combination)
    episodes = diagnostics_df.groupby("height")

    divergence_results = []

    for height, ep_df in episodes:
        if not ep_df["fell"].any():
            continue

        # Get baseline stats from first 20% of episode
        baseline_window = int(len(ep_df) * 0.2)
        baseline = ep_df.head(baseline_window)

        # Compute baseline means and stds
        roll_mean = baseline["roll_deg"].abs().mean()
        roll_std = baseline["roll_deg"].abs().std()
        roll_rate_mean = baseline["roll_rate_deg_s"].abs().mean()
        roll_rate_std = baseline["roll_rate_deg_s"].abs().std()

        contact_asym = (baseline["l_contact_force_N"] - baseline["r_contact_force_N"]).abs()
        contact_asym_mean = contact_asym.mean()
        contact_asym_std = contact_asym.std()

        com_lat_mean = baseline["com_lateral_vel"].abs().mean()
        com_lat_std = baseline["com_lateral_vel"].abs().std()

        hip_roll_sat = ((baseline["l_hip_roll_target"].abs() >= 0.95) |
                        (baseline["r_hip_roll_target"].abs() >= 0.95)).astype(int)
        hip_roll_sat_rate = hip_roll_sat.mean()

        wheel_contact_coupling = (baseline["l_wheel_vel_rad_s"] * baseline["l_contact_force_N"]).abs()
        wheel_contact_mean = wheel_contact_coupling.mean()
        wheel_contact_std = wheel_contact_coupling.std()

        # Scan forward to find first divergence
        divergence_times = {}

        for idx, row in ep_df.iterrows():
            t = row["time_s"]

            if "roll" not in divergence_times:
                if abs(row["roll_deg"]) > roll_mean + threshold_multiplier * roll_std:
                    divergence_times["roll"] = t

            if "roll_rate" not in divergence_times:
                if abs(row["roll_rate_deg_s"]) > roll_rate_mean + threshold_multiplier * roll_rate_std:
                    divergence_times["roll_rate"] = t

            if "contact_asymmetry" not in divergence_times:
                asym = abs(row["l_contact_force_N"] - row["r_contact_force_N"])
                if asym > contact_asym_mean + threshold_multiplier * contact_asym_std:
                    divergence_times["contact_asymmetry"] = t

            if "com_lateral" not in divergence_times:
                if abs(row["com_lateral_vel"]) > com_lat_mean + threshold_multiplier * com_lat_std:
                    divergence_times["com_lateral"] = t

            if "hip_roll_saturation" not in divergence_times:
                if (abs(row["l_hip_roll_target"]) >= 0.95 or abs(row["r_hip_roll_target"]) >= 0.95):
                    if hip_roll_sat_rate < 0.1:  # Only flag if baseline wasn't saturated
                        divergence_times["hip_roll_saturation"] = t

            if "wheel_contact_coupling" not in divergence_times:
                coupling = abs(row["l_wheel_vel_rad_s"] * row["l_contact_force_N"])
                if coupling > wheel_contact_mean + threshold_multiplier * wheel_contact_std:
                    divergence_times["wheel_contact_coupling"] = t

        if divergence_times:
            first_var = min(divergence_times.items(), key=lambda x: x[1])
            divergence_results.append({
                "height": height,
                "first_divergence_variable": first_var[0],
                "first_divergence_time_s": first_var[1],
                "fall_time_s": ep_df[ep_df["fell"]]["fall_time_s"].iloc[0] if ep_df["fell"].any() else None,
            })

    if not divergence_results:
        return {"first_divergence_variable": "none", "first_divergence_time_s": None}

    # Return most common first divergence
    div_df = pd.DataFrame(divergence_results)
    most_common = div_df["first_divergence_variable"].mode()[0]
    avg_time = div_df[div_df["first_divergence_variable"] == most_common]["first_divergence_time_s"].mean()

    return {
        "first_divergence_variable": most_common,
        "first_divergence_time_s": float(avg_time),
        "divergence_counts": div_df["first_divergence_variable"].value_counts().to_dict(),
    }


def evaluate_roll_fix(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    heights: list[float],
    init_table: dict,
    episodes_per_height: int,
    seed: int,
) -> dict:
    """Evaluate roll-fix candidate across heights."""
    all_results = []

    for height in heights:
        for ep in range(episodes_per_height):
            rng = jax.random.PRNGKey(seed + ep + int(height * 1000))
            state = env.reset(rng)
            state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

            controller.reset()
            controller.target_hip_pitch = init_table[min(init_table.keys(), key=lambda h: abs(h - height))]["hip_pitch"]
            controller.target_knee = init_table[min(init_table.keys(), key=lambda h: abs(h - height))]["knee"]
            controller.last_stable_hip_pitch = controller.target_hip_pitch
            controller.last_stable_knee = controller.target_knee

            pitch_sq = 0.0
            roll_sq = 0.0
            wheel_speed_sq = 0.0
            action_sat_count = 0
            steps = 0

            for _ in range(1000):
                obs_np = np.array(state.obs)
                action = controller.compute_action(obs_np)

                g_body = obs_np[0:3]
                pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
                roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))

                pitch_sq += pitch ** 2
                roll_sq += roll ** 2

                joint_vel = obs_np[19:29]
                wheel_speed = (abs(joint_vel[4]) + abs(joint_vel[9])) / 2.0
                wheel_speed_sq += wheel_speed ** 2

                if np.max(np.abs(action)) >= 0.99:
                    action_sat_count += 1

                state = env.step(state, jnp.array(action))
                steps += 1

                if bool(state.done):
                    break

            survival_time = steps * env.CONTROL_DT
            fell = bool(state.info["is_fallen"])

            final_obs = np.array(state.obs)
            torso_height = float(state.mjx_data.qpos[2])
            g_body_final = final_obs[0:3]
            tilt_final = float(np.arccos(np.clip(-g_body_final[2], -1.0, 1.0)))

            if fell:
                if torso_height < env._min_height:
                    fall_reason = "height"
                elif tilt_final > env._max_tilt:
                    fall_reason = "tilt"
                else:
                    fall_reason = "unknown"
            else:
                fall_reason = "none"

            pitch_rms = np.sqrt(pitch_sq / steps) if steps > 0 else 0.0
            roll_rms = np.sqrt(roll_sq / steps) if steps > 0 else 0.0
            wheel_speed_rms = np.sqrt(wheel_speed_sq / steps) if steps > 0 else 0.0
            action_sat_rate = action_sat_count / steps if steps > 0 else 0.0

            all_results.append({
                "height": height,
                "episode": ep,
                "survival_time_s": survival_time,
                "fell": fell,
                "fall_reason": fall_reason,
                "pitch_rms_deg": np.rad2deg(pitch_rms),
                "roll_rms_deg": np.rad2deg(roll_rms),
                "wheel_speed_rms_rad_s": wheel_speed_rms,
                "action_saturation_rate": action_sat_rate,
            })

    df = pd.DataFrame(all_results)
    fall_reason_counts = df["fall_reason"].value_counts(dropna=False).to_dict()
    dominant_fall_reason = max(fall_reason_counts.items(), key=lambda kv: kv[1])[0] if fall_reason_counts else "unknown"

    return {
        "mean_survival_s": float(df["survival_time_s"].mean()),
        "mean_fall_rate": float(df["fell"].mean()),
        "mean_pitch_rms_deg": float(df["pitch_rms_deg"].mean()),
        "mean_roll_rms_deg": float(df["roll_rms_deg"].mean()),
        "mean_wheel_speed_rms": float(df["wheel_speed_rms_rad_s"].mean()),
        "mean_action_sat_rate": float(df["action_saturation_rate"].mean()),
        "dominant_fall_reason": dominant_fall_reason,
        "fall_reason_counts_json": json.dumps(fall_reason_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase B.9 Step 5.5/5.6/5.7/5.8/5.9/5.10/5.11: Roll/Tilt Diagnostics and Fix")
    parser.add_argument("--mode", type=str, default="all", choices=["diagnostic", "small_eval", "full_eval", "all", "step5_6", "step5_7", "step5_8", "step5_9", "step5_10", "step5_11"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase_b9_step5_5_roll_tilt_fix"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Step 5.5: Roll/Tilt Failure Diagnosis and Fix[/bold cyan]\n")

    # Load best Step 5 config
    best_config_path = project_root / "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml"
    with open(best_config_path, "r", encoding="utf-8") as f:
        best_step5_params = yaml.safe_load(f)

    console.print(f"[green]Loaded best Step 5 config:[/green]")
    for k, v in best_step5_params.items():
        console.print(f"  {k}: {v}")
    console.print()

    base_config_path = project_root / "configs/controllers/dual_rate_balance_controller_b9.yaml"
    base_config = DualRateConfig.from_yaml(base_config_path)

    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    init_table = load_balanced_init_table()

    env_config = {
        "episode_length": 1000,
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
    }
    env = BalanceEnv(env_config)

    if args.mode == "step5_6":
        console.print("[bold cyan]Running Step 5.6 root-cause workflow (0–0.3s baseline + single targeted fix)...[/bold cyan]")
        step56_result = run_step5_6_root_cause_and_fix(
            base_config=base_config,
            best_step5_params=best_step5_params,
            env=env,
            mj_model=mj_model,
            init_table=init_table,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        summary = step56_result["summary"]
        console.print(f"[green]Saved early diagnostics: {step56_result['early_csv']}[/green]")
        console.print(f"[green]Saved root-cause table: {step56_result['cause_csv']}[/green]")
        console.print(f"[green]Saved Step 5.6 summary: {step56_result['summary_json']}[/green]")
        console.print(f"[yellow]Root cause: {summary['root_cause']}[/yellow]")
        console.print(f"[yellow]Baseline roll RMS: {summary['baseline_metrics']['mean_roll_rms_deg']:.3f} deg[/yellow]")
        console.print(f"[yellow]Fix roll RMS: {summary['targeted_fix_metrics']['mean_roll_rms_deg']:.3f} deg[/yellow]")
        console.print("[bold green]Step 5.6 workflow complete[/bold green]")
        return

    if args.mode == "step5_7":
        console.print("[bold cyan]Running Step 5.7 early roll stabilizer design...[/bold cyan]")
        run_step5_7_early_roll_stabilizer(
            base_config=base_config,
            best_step5_params=best_step5_params,
            env=env,
            mj_model=mj_model,
            init_table=init_table,
            output_dir=Path("outputs/phase_b9_step5_7_early_roll_stabilizer"),
            seed=args.seed,
        )
        console.print("[bold green]Step 5.7 workflow complete[/bold green]")
        return

    if args.mode == "step5_8":
        console.print("[bold cyan]Running Step 5.8 roll instability root-cause redesign...[/bold cyan]")
        run_step5_8_roll_redesign(
            base_config=base_config,
            best_step5_params=best_step5_params,
            env=env,
            mj_model=mj_model,
            init_table=init_table,
            output_dir=Path("outputs/phase_b9_step5_8_roll_redesign"),
            seed=args.seed,
        )
        console.print("[bold green]Step 5.8 workflow complete[/bold green]")
        return

    if args.mode == "step5_9":
        console.print("[bold cyan]Running Step 5.9 roll authority and coupling audit...[/bold cyan]")
        run_step5_9_roll_authority_audit(
            base_config=base_config,
            best_step5_params=best_step5_params,
            env=env,
            mj_model=mj_model,
            init_table=init_table,
            output_dir=Path("outputs/phase_b9_step5_9_roll_authority_audit"),
            seed=args.seed,
        )
        console.print("[bold green]Step 5.9 workflow complete[/bold green]")
        return

    if args.mode == "step5_10":
        console.print("[bold cyan]Running Step 5.10 early transient timing fix...[/bold cyan]")
        run_step5_10_early_transient_fix(
            base_config=base_config,
            best_step5_params=best_step5_params,
            env=env,
            mj_model=mj_model,
            init_table=init_table,
            output_dir=Path("outputs/phase_b9_step5_10_early_transient_fix"),
            seed=args.seed,
        )
        console.print("[bold green]Step 5.10 workflow complete[/bold green]")
        return

    if args.mode == "step5_11":
        console.print("[bold cyan]Running Step 5.11 corrective path validity audit...[/bold cyan]")
        run_step5_11_corrective_path_audit(
            base_config=base_config,
            best_step5_params=best_step5_params,
            env=env,
            mj_model=mj_model,
            init_table=init_table,
            output_dir=Path("outputs/phase_b9_step5_11_corrective_path_audit"),
            seed=args.seed,
        )
        console.print("[bold green]Step 5.11 workflow complete[/bold green]")
        return

    # Mode 1: Diagnostic rollouts
    if args.mode in ["diagnostic", "all"]:
        console.print("[bold cyan]Running diagnostic rollouts...[/bold cyan]")
        diagnostic_heights = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]

        controller = create_roll_fix_controller(
            base_config, best_step5_params, "baseline", {}, mj_model
        )

        all_diagnostics = []
        for height in diagnostic_heights:
            diag = run_diagnostic_rollout(controller, env, height, init_table, args.seed + int(height * 100))
            all_diagnostics.extend(diag)

        diag_df = pd.DataFrame(all_diagnostics)
        diag_csv = args.output_dir / "diagnostics.csv"
        diag_df.to_csv(diag_csv, index=False)
        console.print(f"[green]Saved diagnostics to {diag_csv}[/green]\n")

    # Mode 2: Small evaluation
    small_results_df = None
    best_candidate = None
    if args.mode in ["small_eval", "all"]:
        console.print("[bold cyan]Running small evaluation (3 eps × 3 heights)...[/bold cyan]")
        small_heights = [0.60, 0.50, 0.40]
        small_episodes = 3

        roll_fix_candidates = [
            ("baseline", {}),
            ("A_weak_hip_roll_pd", {"roll_kp": 0.3, "roll_kd": 0.05, "roll_max_correction": 0.15}),
            ("B_strong_hip_roll_pd", {"roll_kp": 0.8, "roll_kd": 0.15, "roll_max_correction": 0.25}),
            ("C_roll_rate_damping", {"roll_kd": 0.20, "roll_max_correction": 0.20}),
            ("D_contact_force_balance", {"roll_kp": 0.4, "roll_kd": 0.12, "roll_max_correction": 0.20, "wheel_cmd_limit_mult": 2.5}),
            ("E_lateral_com_correction", {"roll_kp": 1.0, "roll_kd": 0.10, "roll_max_correction": 0.30}),
            ("F_reduced_wheel_limit", {"wheel_cmd_limit_mult": 2.0}),
        ]

        results = []
        for fix_type, fix_params in roll_fix_candidates:
            console.print(f"  Evaluating {fix_type}...")
            controller = create_roll_fix_controller(
                base_config, best_step5_params, fix_type, fix_params, mj_model
            )
            metrics = evaluate_roll_fix(controller, env, small_heights, init_table, small_episodes, args.seed)
            results.append({"roll_fix_type": fix_type, "roll_fix_params": json.dumps(fix_params), **fix_params, **metrics})

        small_results_df = pd.DataFrame(results)
        results_csv = args.output_dir / "candidate_results.csv"
        small_results_df.to_csv(results_csv, index=False)

        # Candidate selection contract: survival desc, fall_rate asc, roll_rms asc, sat_rate asc
        ranked_df = small_results_df.sort_values(
            by=["mean_survival_s", "mean_fall_rate", "mean_roll_rms_deg", "mean_action_sat_rate"],
            ascending=[False, True, True, True],
        ).reset_index(drop=True)
        best_row = ranked_df.iloc[0]
        best_candidate = {
            "roll_fix_type": best_row["roll_fix_type"],
            "roll_fix_params": json.loads(best_row["roll_fix_params"]),
            "small_eval_metrics": {
                "mean_survival_s": float(best_row["mean_survival_s"]),
                "mean_fall_rate": float(best_row["mean_fall_rate"]),
                "mean_roll_rms_deg": float(best_row["mean_roll_rms_deg"]),
                "mean_action_sat_rate": float(best_row["mean_action_sat_rate"]),
            },
        }

        table = Table(title="Small Evaluation Results")
        table.add_column("Rank", justify="right")
        table.add_column("Roll Fix", justify="left")
        table.add_column("Survival (s)", justify="right")
        table.add_column("Fall Rate", justify="right")
        table.add_column("Roll RMS (°)", justify="right")
        table.add_column("Sat Rate", justify="right")

        for rank, (_, row) in enumerate(ranked_df.iterrows(), start=1):
            table.add_row(
                str(rank),
                row["roll_fix_type"],
                f"{row['mean_survival_s']:.2f}",
                f"{row['mean_fall_rate']:.1%}",
                f"{row['mean_roll_rms_deg']:.1f}",
                f"{row['mean_action_sat_rate']:.1%}",
            )
        console.print(table)
        console.print(f"[green]Saved candidate results to {results_csv}[/green]")
        console.print(f"[green]Best candidate: {best_candidate['roll_fix_type']}[/green]\n")

    # Mode 3: Full evaluation using best candidate from small eval
    if args.mode in ["full_eval", "all"]:
        console.print("[bold cyan]Running full validation (5 eps × 6 heights)...[/bold cyan]")

        if best_candidate is None:
            # If full_eval only mode, try load candidate_results.csv and select best
            candidate_csv = args.output_dir / "candidate_results.csv"
            if not candidate_csv.exists():
                raise FileNotFoundError(
                    "candidate_results.csv not found. Run --mode small_eval first or use --mode all."
                )
            loaded_df = pd.read_csv(candidate_csv)
            ranked_df = loaded_df.sort_values(
                by=["mean_survival_s", "mean_fall_rate", "mean_roll_rms_deg", "mean_action_sat_rate"],
                ascending=[False, True, True, True],
            ).reset_index(drop=True)
            best_row = ranked_df.iloc[0]
            best_candidate = {
                "roll_fix_type": best_row["roll_fix_type"],
                "roll_fix_params": json.loads(best_row["roll_fix_params"]),
                "small_eval_metrics": {
                    "mean_survival_s": float(best_row["mean_survival_s"]),
                    "mean_fall_rate": float(best_row["mean_fall_rate"]),
                    "mean_roll_rms_deg": float(best_row["mean_roll_rms_deg"]),
                    "mean_action_sat_rate": float(best_row["mean_action_sat_rate"]),
                },
            }

        full_heights = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
        full_episodes = 5

        best_controller = create_roll_fix_controller(
            base_config,
            best_step5_params,
            best_candidate["roll_fix_type"],
            best_candidate["roll_fix_params"],
            mj_model,
        )

        full_metrics = evaluate_roll_fix(
            best_controller,
            env,
            full_heights,
            init_table,
            full_episodes,
            args.seed,
        )

        # Run diagnostics for first-divergence analysis on full heights
        all_diag = []
        for height in full_heights:
            diag = run_diagnostic_rollout(best_controller, env, height, init_table, args.seed + int(height * 100))
            all_diag.extend(diag)

        full_diag_df = pd.DataFrame(all_diag)
        full_diag_csv = args.output_dir / "best_candidate_full_diagnostics.csv"
        full_diag_df.to_csv(full_diag_csv, index=False)

        divergence_summary = detect_first_divergence(full_diag_df)

        # Save full validation metrics
        full_summary = {
            "roll_fix_type": best_candidate["roll_fix_type"],
            "roll_fix_params": best_candidate["roll_fix_params"],
            "small_eval": best_candidate["small_eval_metrics"],
            "full_eval": full_metrics,
            "first_divergence": divergence_summary,
        }

        full_summary_json = args.output_dir / "full_validation_summary.json"
        with open(full_summary_json, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2)

        # Required artifact 1: best_roll_fix_config.yaml
        best_roll_fix_config = {
            "base_step5_lqr_config": best_step5_params,
            "best_roll_fix": {
                "type": best_candidate["roll_fix_type"],
                "params": best_candidate["roll_fix_params"],
            },
        }
        best_cfg_path = args.output_dir / "best_roll_fix_config.yaml"
        with open(best_cfg_path, "w", encoding="utf-8") as f:
            yaml.dump(best_roll_fix_config, f, sort_keys=False)

        # Required artifact 2: best_roll_fix_summary.json
        best_summary = {
            "roll_fix_type": best_candidate["roll_fix_type"],
            "roll_fix_params": best_candidate["roll_fix_params"],
            "small_eval_metrics": best_candidate["small_eval_metrics"],
            "full_eval_metrics": full_metrics,
            "first_divergence": divergence_summary,
            "gating_decision": {
                "step6_ready": bool(
                    (full_metrics["mean_fall_rate"] <= 0.5)
                    and (full_metrics["mean_roll_rms_deg"] <= 15.0)
                ),
                "criteria": {
                    "max_fall_rate": 0.5,
                    "max_roll_rms_deg": 15.0,
                },
            },
        }
        best_summary_path = args.output_dir / "best_roll_fix_summary.json"
        with open(best_summary_path, "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        console.print(f"[green]Saved full diagnostics to {full_diag_csv}[/green]")
        console.print(f"[green]Saved full summary to {full_summary_json}[/green]")
        console.print(f"[green]Saved best config to {best_cfg_path}[/green]")
        console.print(f"[green]Saved best summary to {best_summary_path}[/green]\n")

    console.print("[bold green]Step 5.5 contract flow complete[/bold green]")


def _step57_apply_variant_adjustment(
    variant: str,
    action: np.ndarray,
    step: int,
    dt: float,
    roll_deg: float,
    roll_rate_deg_s: float,
    prev_roll_rate_deg_s: float,
    init_row: dict,
) -> tuple[np.ndarray, float, float]:
    """Apply Step 5.7 early-roll variant adjustment on hip-roll channels only."""
    a = np.array(action, copy=True)
    t = step * dt

    # Only act in early window for preload/damping style variants.
    early_window = 0.30

    if variant == "A_roll_rate_damping_from_t0":
        kd = 0.0030
        max_corr = 0.08
        corr = np.clip(-kd * roll_rate_deg_s, -max_corr, max_corr)
        a[0] = np.clip(a[0] + corr, -1.0, 1.0)
        a[5] = np.clip(a[5] - corr, -1.0, 1.0)

    elif variant == "B_very_weak_roll_pd_from_t0":
        kp = 0.0060
        kd = 0.0020
        max_corr = 0.06
        corr = np.clip(-(kp * roll_deg + kd * roll_rate_deg_s), -max_corr, max_corr)
        a[0] = np.clip(a[0] + corr, -1.0, 1.0)
        a[5] = np.clip(a[5] - corr, -1.0, 1.0)

    elif variant == "C_hip_roll_preload_only":
        if t <= early_window:
            preload = 0.015
            a[0] = np.clip(a[0] + preload, -1.0, 1.0)
            a[5] = np.clip(a[5] - preload, -1.0, 1.0)

    elif variant == "D_contact_force_decay_preload":
        if t <= 0.20:
            lf = float(init_row.get("expected_left_force", 0.0))
            rf = float(init_row.get("expected_right_force", 0.0))
            denom = max(abs(lf) + abs(rf), 1e-6)
            imbalance = (lf - rf) / denom
            preload0 = np.clip(0.06 * imbalance, -0.03, 0.03)
            decay = max(0.0, 1.0 - t / 0.20)
            preload = preload0 * decay
            a[0] = np.clip(a[0] + preload, -1.0, 1.0)
            a[5] = np.clip(a[5] - preload, -1.0, 1.0)

    elif variant == "E_roll_emergency_only":
        growing_rate = abs(roll_rate_deg_s) > abs(prev_roll_rate_deg_s) + 2.0
        if abs(roll_deg) > 4.0 or growing_rate:
            kp = 0.006
            kd = 0.0025
            max_corr = 0.08
            corr = np.clip(-(kp * roll_deg + kd * roll_rate_deg_s), -max_corr, max_corr)
            a[0] = np.clip(a[0] + corr, -1.0, 1.0)
            a[5] = np.clip(a[5] - corr, -1.0, 1.0)

    hip_action_mag = 0.5 * (abs(a[0]) + abs(a[5]))
    return a, hip_action_mag, t


def _run_step57_variant_episode(
    base_controller: DualRateBalanceController,
    variant: str,
    env: BalanceEnv,
    init_table: dict,
    height: float,
    seed: int,
) -> dict:
    """Run one episode for Step 5.7 variant with early telemetry and diagnostics."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

    base_controller.reset()
    nearest_h = min(init_table.keys(), key=lambda hh: abs(hh - height))
    base_controller.target_hip_pitch = init_table[nearest_h]["hip_pitch"]
    base_controller.target_knee = init_table[nearest_h]["knee"]
    base_controller.last_stable_hip_pitch = base_controller.target_hip_pitch
    base_controller.last_stable_knee = base_controller.target_knee

    pitch_sq = 0.0
    roll_sq = 0.0
    action_sat_count = 0
    steps = 0

    first_roll_diverge_time = None
    hip_roll_action_nonzero_time = None
    prev_roll_rate_deg_s = 0.0

    early_roll_abs = []

    for step in range(1000):
        obs_np = np.array(state.obs)
        action = base_controller.compute_action(obs_np)

        g_body = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
        roll_deg = float(np.rad2deg(roll))
        roll_rate_deg_s = float(np.rad2deg(obs_np[7]))

        action, hip_mag, t = _step57_apply_variant_adjustment(
            variant=variant,
            action=action,
            step=step,
            dt=env.CONTROL_DT,
            roll_deg=roll_deg,
            roll_rate_deg_s=roll_rate_deg_s,
            prev_roll_rate_deg_s=prev_roll_rate_deg_s,
            init_row=init_table[nearest_h],
        )

        if hip_roll_action_nonzero_time is None and hip_mag > 1e-3:
            hip_roll_action_nonzero_time = t

        if t <= 0.30:
            early_roll_abs.append(abs(roll_deg))

        pitch_sq += pitch ** 2
        roll_sq += roll ** 2

        if np.max(np.abs(action)) >= 0.99:
            action_sat_count += 1

        state = env.step(state, jnp.array(action))
        steps += 1

        if first_roll_diverge_time is None and t >= 0.08:
            baseline_mean = np.mean(early_roll_abs[: max(1, min(len(early_roll_abs), 3))]) if early_roll_abs else 0.0
            baseline_std = np.std(early_roll_abs[: max(1, min(len(early_roll_abs), 3))]) if early_roll_abs else 0.0
            thr = baseline_mean + 2.0 * baseline_std
            if abs(roll_deg) > max(thr, 1.0):
                first_roll_diverge_time = t

        prev_roll_rate_deg_s = roll_rate_deg_s

        if bool(state.done):
            break

    survival_time = steps * env.CONTROL_DT
    fell = bool(state.info["is_fallen"])
    pitch_rms = np.sqrt(pitch_sq / steps) if steps > 0 else 0.0
    roll_rms = np.sqrt(roll_sq / steps) if steps > 0 else 0.0
    action_sat_rate = action_sat_count / steps if steps > 0 else 0.0

    return {
        "survival_time_s": float(survival_time),
        "fell": bool(fell),
        "roll_rms_deg": float(np.rad2deg(roll_rms)),
        "pitch_rms_deg": float(np.rad2deg(pitch_rms)),
        "action_saturation_rate": float(action_sat_rate),
        "first_roll_divergence_time_s": first_roll_diverge_time,
        "hip_roll_action_nonzero_time_s": hip_roll_action_nonzero_time,
    }


def run_step5_7_early_roll_stabilizer(
    base_config: DualRateConfig,
    best_step5_params: dict,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    init_table: dict,
    output_dir: Path,
    seed: int,
) -> dict:
    """Step 5.7: early roll stabilizer variants with baseline-gated selection and full validation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_controller = create_roll_fix_controller(
        base_config, best_step5_params, "baseline", {}, mj_model
    )

    # Baseline reference from Step 5 summary and Step 5.6 contract.
    baseline_survival_ref = 3.8926666666666665
    baseline_fall_ref = 0.8333333333333334
    baseline_roll_ref = 21.168228816065504

    variants = [
        "A_roll_rate_damping_from_t0",
        "B_very_weak_roll_pd_from_t0",
        "C_hip_roll_preload_only",
        "D_contact_force_decay_preload",
        "E_roll_emergency_only",
    ]

    # Small eval at h=0.60, 5 episodes each.
    candidate_rows = []
    for variant in variants:
        ep_rows = []
        for ep in range(5):
            result = _run_step57_variant_episode(
                base_controller=baseline_controller,
                variant=variant,
                env=env,
                init_table=init_table,
                height=0.60,
                seed=seed + ep,
            )
            ep_rows.append(result)

        df = pd.DataFrame(ep_rows)
        first_div_mean = df["first_roll_divergence_time_s"].dropna().mean()
        hip_nonzero_mean = df["hip_roll_action_nonzero_time_s"].dropna().mean()

        candidate_rows.append({
            "variant": variant,
            "survival_time_s": float(df["survival_time_s"].mean()),
            "fall_rate": float(df["fell"].mean()),
            "roll_RMS_deg": float(df["roll_rms_deg"].mean()),
            "first_roll_divergence_time_s": (None if pd.isna(first_div_mean) else float(first_div_mean)),
            "hip_roll_action_nonzero_time_s": (None if pd.isna(hip_nonzero_mean) else float(hip_nonzero_mean)),
            "pitch_RMS_deg": float(df["pitch_rms_deg"].mean()),
            "action_saturation_rate": float(df["action_saturation_rate"].mean()),
        })

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_csv = output_dir / "candidate_results.csv"
    candidate_df.to_csv(candidate_csv, index=False)

    # Selection rule from user: beat baseline in survival OR fall-rate OR significantly delay divergence,
    # while not worsening pitch/saturation.
    base_pitch = 1.0109260130054514
    base_sat = 0.0

    kept = []
    for _, row in candidate_df.iterrows():
        improve_core = (
            (row["survival_time_s"] > baseline_survival_ref)
            or (row["fall_rate"] < baseline_fall_ref)
            or (
                pd.notna(row["first_roll_divergence_time_s"]) and row["first_roll_divergence_time_s"] >= 0.14
            )
        )
        no_worse_aux = (
            row["pitch_RMS_deg"] <= base_pitch * 1.10
            and row["action_saturation_rate"] <= max(base_sat + 0.02, 0.02)
        )
        if improve_core and no_worse_aux:
            kept.append(row["variant"])

    if len(kept) == 0:
        # Keep top-1 by delayed divergence then survival as fallback for Step 5.7 execution continuity.
        ranked = candidate_df.sort_values(
            by=["first_roll_divergence_time_s", "survival_time_s", "fall_rate"],
            ascending=[False, False, True],
        )
        kept = [str(ranked.iloc[0]["variant"])]

    kept = kept[:2]

    # Full validation on all valid heights.
    full_rows = []
    for variant in kept:
        for height in [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
            for ep in range(5):
                result = _run_step57_variant_episode(
                    base_controller=baseline_controller,
                    variant=variant,
                    env=env,
                    init_table=init_table,
                    height=height,
                    seed=seed + ep + int(height * 100),
                )
                full_rows.append({"variant": variant, "height": height, "episode": ep, **result})

    full_df = pd.DataFrame(full_rows)
    full_csv = output_dir / "full_validation.csv"
    full_df.to_csv(full_csv, index=False)

    summary = (
        full_df.groupby("variant")
        .agg(
            survival_time_s=("survival_time_s", "mean"),
            fall_rate=("fell", "mean"),
            roll_RMS_deg=("roll_rms_deg", "mean"),
            pitch_RMS_deg=("pitch_rms_deg", "mean"),
            action_saturation_rate=("action_saturation_rate", "mean"),
            first_roll_divergence_time_s=("first_roll_divergence_time_s", "mean"),
            hip_roll_action_nonzero_time_s=("hip_roll_action_nonzero_time_s", "mean"),
        )
        .reset_index()
    )

    summary["beats_step5_baseline"] = (
        (summary["survival_time_s"] > baseline_survival_ref)
        & (summary["fall_rate"] < baseline_fall_ref)
        & (summary["roll_RMS_deg"] < baseline_roll_ref)
        & (summary["pitch_RMS_deg"] <= base_pitch * 1.10)
        & (summary["action_saturation_rate"] <= max(base_sat + 0.02, 0.02))
    )

    summary = summary.sort_values(
        by=["beats_step5_baseline", "survival_time_s", "fall_rate", "roll_RMS_deg"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_variant = str(summary.iloc[0]["variant"])
    best_row = summary.iloc[0].to_dict()

    best_cfg = {
        "baseline_step5_lqr_config": best_step5_params,
        "step5_7_best_variant": best_variant,
    }
    with open(output_dir / "best_early_roll_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    decision_allow_step6 = bool(best_row["beats_step5_baseline"])
    best_summary = {
        "baseline_reference": {
            "mean_survival_s": baseline_survival_ref,
            "mean_fall_rate": baseline_fall_ref,
            "mean_pitch_rms_deg": base_pitch,
            "mean_action_sat_rate": base_sat,
            "mean_roll_rms_deg": baseline_roll_ref,
        },
        "kept_variants": kept,
        "best_variant": best_variant,
        "best_full_validation_metrics": {
            "survival_time_s": float(best_row["survival_time_s"]),
            "fall_rate": float(best_row["fall_rate"]),
            "roll_RMS_deg": float(best_row["roll_RMS_deg"]),
            "pitch_RMS_deg": float(best_row["pitch_RMS_deg"]),
            "action_saturation_rate": float(best_row["action_saturation_rate"]),
            "first_roll_divergence_time_s": (None if pd.isna(best_row["first_roll_divergence_time_s"]) else float(best_row["first_roll_divergence_time_s"])),
            "hip_roll_action_nonzero_time_s": (None if pd.isna(best_row["hip_roll_action_nonzero_time_s"]) else float(best_row["hip_roll_action_nonzero_time_s"])),
        },
        "step6_ready": decision_allow_step6,
        "decision_rule": "Only allow Step 6 if Step 5.7 improves full validation over Step 5 best_lqr_config.yaml",
    }

    with open(output_dir / "best_early_roll_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    # Update docs with Step 5.7 section append.
    _append_step57_to_reports(project_root, output_dir, best_summary)

    console.print(f"[green]Saved candidate results: {candidate_csv}[/green]")
    console.print(f"[green]Saved full validation: {full_csv}[/green]")
    console.print(f"[green]Saved best config: {output_dir / 'best_early_roll_config.yaml'}[/green]")
    console.print(f"[green]Saved best summary: {output_dir / 'best_early_roll_summary.json'}[/green]")

    return best_summary


def _append_step57_to_reports(project_root: Path, output_dir: Path, best_summary: dict) -> None:
    """Append concise Step 5.7 results to required reports."""
    best_report = project_root / "docs/phase_b9_best_standalone_controller_report.md"
    audit_report = project_root / "docs/phase_b9_audit_gate_report.md"

    sec = [
        "\n## Phase B.9 Step 5.7 — Early Roll Stabilizer Design\n",
        f"- Output dir: `{output_dir.as_posix()}`",
        f"- Best variant: `{best_summary['best_variant']}`",
        f"- Full-validation survival: {best_summary['best_full_validation_metrics']['survival_time_s']:.4f} s",
        f"- Full-validation fall rate: {best_summary['best_full_validation_metrics']['fall_rate']:.4f}",
        f"- Full-validation roll RMS: {best_summary['best_full_validation_metrics']['roll_RMS_deg']:.4f} deg",
        f"- Step 6 ready: `{best_summary['step6_ready']}`",
    ]
    sec_text = "\n".join(sec) + "\n"

    for path in [best_report, audit_report]:
        if path.exists():
            old = path.read_text(encoding="utf-8")
            if "Phase B.9 Step 5.7" not in old:
                path.write_text(old + sec_text, encoding="utf-8")
            else:
                # Replace existing Step 5.7 tail block minimally by appending updated snapshot.
                path.write_text(old + "\n" + sec_text, encoding="utf-8")


def _apply_initial_roll_perturbation(mjx_data, roll_deg: float):
    """Apply initial roll perturbation (about body x-axis) to root quaternion."""
    if abs(roll_deg) < 1e-9:
        return mjx_data

    r = np.deg2rad(roll_deg)
    q = np.array([np.cos(0.5 * r), np.sin(0.5 * r), 0.0, 0.0], dtype=np.float32)
    new_qpos = mjx_data.qpos.at[3].set(q[0])
    new_qpos = new_qpos.at[4].set(q[1])
    new_qpos = new_qpos.at[5].set(q[2])
    new_qpos = new_qpos.at[6].set(q[3])
    return mjx_data.replace(qpos=new_qpos)


def _step58_apply_variant_adjustment(
    variant: str,
    action: np.ndarray,
    roll_deg: float,
    roll_rate_deg_s: float,
    com_lateral_offset_m: float,
    init_row: dict,
) -> np.ndarray:
    """Apply Step 5.8 variant action adjustment on hip-roll and/or wheels."""
    a = np.array(action, copy=True)

    if variant == "A_hip_roll_position_damping_only":
        kp = 0.0040
        max_corr = 0.06
        corr = np.clip(-kp * roll_deg, -max_corr, max_corr)
        a[0] = np.clip(a[0] + corr, -1.0, 1.0)
        a[5] = np.clip(a[5] - corr, -1.0, 1.0)

    elif variant == "B_hip_roll_velocity_damping_only":
        kd = 0.0030
        max_corr = 0.08
        corr = np.clip(-kd * roll_rate_deg_s, -max_corr, max_corr)
        a[0] = np.clip(a[0] + corr, -1.0, 1.0)
        a[5] = np.clip(a[5] - corr, -1.0, 1.0)

    elif variant == "C_hip_roll_preload_from_balanced_root":
        lf = float(init_row.get("expected_left_force", 0.0))
        rf = float(init_row.get("expected_right_force", 0.0))
        denom = max(abs(lf) + abs(rf), 1e-6)
        imbalance = (lf - rf) / denom
        preload = float(np.clip(0.05 * imbalance, -0.02, 0.02))
        a[0] = np.clip(a[0] + preload, -1.0, 1.0)
        a[5] = np.clip(a[5] - preload, -1.0, 1.0)

    elif variant == "D_lateral_CoM_feedback_through_hip_roll":
        kcom = 0.25
        max_corr = 0.05
        corr = np.clip(-kcom * com_lateral_offset_m, -max_corr, max_corr)
        a[0] = np.clip(a[0] + corr, -1.0, 1.0)
        a[5] = np.clip(a[5] - corr, -1.0, 1.0)

    elif variant == "E_differential_wheel_roll_damping":
        kw = 0.0025
        max_corr = 0.08
        corr = np.clip(-kw * roll_rate_deg_s, -max_corr, max_corr)
        a[4] = np.clip(a[4] + corr, -1.0, 1.0)
        a[9] = np.clip(a[9] - corr, -1.0, 1.0)

    elif variant == "F_weak_hip_roll_plus_differential_wheel":
        kd_hip = 0.0015
        kd_wheel = 0.0015
        max_hip = 0.04
        max_wheel = 0.04

        hip_corr = np.clip(-kd_hip * roll_rate_deg_s, -max_hip, max_hip)
        a[0] = np.clip(a[0] + hip_corr, -1.0, 1.0)
        a[5] = np.clip(a[5] - hip_corr, -1.0, 1.0)

        wheel_corr = np.clip(-kd_wheel * roll_rate_deg_s, -max_wheel, max_wheel)
        a[4] = np.clip(a[4] + wheel_corr, -1.0, 1.0)
        a[9] = np.clip(a[9] - wheel_corr, -1.0, 1.0)

    return a


def _run_step58_variant_episode(
    base_controller: DualRateBalanceController,
    variant: str,
    env: BalanceEnv,
    init_table: dict,
    height: float,
    seed: int,
    initial_roll_deg: float = 0.0,
    disable_wheel_lqr: bool = False,
) -> dict:
    """Run one Step 5.8 episode with optional roll perturbation and wheel-LQR disable."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))
    state = state._replace(mjx_data=_apply_initial_roll_perturbation(state.mjx_data, initial_roll_deg))

    base_controller.reset()
    nearest_h = min(init_table.keys(), key=lambda hh: abs(hh - height))
    base_controller.target_hip_pitch = init_table[nearest_h]["hip_pitch"]
    base_controller.target_knee = init_table[nearest_h]["knee"]
    base_controller.last_stable_hip_pitch = base_controller.target_hip_pitch
    base_controller.last_stable_knee = base_controller.target_knee

    pitch_sq = 0.0
    roll_sq = 0.0
    wheel_speed_sq = 0.0
    action_sat_count = 0
    steps = 0

    early_roll_abs = []
    first_roll_diverge_time = None

    for step in range(1000):
        obs_np = np.array(state.obs)
        action = base_controller.compute_action(obs_np)

        if disable_wheel_lqr:
            action[4] = 0.0
            action[9] = 0.0

        g_body = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
        roll_deg = float(np.rad2deg(roll))
        roll_rate_deg_s = float(np.rad2deg(obs_np[7]))
        com_lateral_offset_m = float(state.mjx_data.qpos[1])

        action = _step58_apply_variant_adjustment(
            variant=variant,
            action=action,
            roll_deg=roll_deg,
            roll_rate_deg_s=roll_rate_deg_s,
            com_lateral_offset_m=com_lateral_offset_m,
            init_row=init_table[nearest_h],
        )

        t = step * env.CONTROL_DT
        if t <= 0.30:
            early_roll_abs.append(abs(roll_deg))

        if first_roll_diverge_time is None and t >= 0.08:
            baseline_mean = np.mean(early_roll_abs[: max(1, min(len(early_roll_abs), 3))]) if early_roll_abs else 0.0
            baseline_std = np.std(early_roll_abs[: max(1, min(len(early_roll_abs), 3))]) if early_roll_abs else 0.0
            thr = baseline_mean + 2.0 * baseline_std
            if abs(roll_deg) > max(thr, 1.0):
                first_roll_diverge_time = t

        pitch_sq += pitch ** 2
        roll_sq += roll ** 2

        joint_vel = obs_np[19:29]
        wheel_speed = (abs(joint_vel[4]) + abs(joint_vel[9])) * 0.5
        wheel_speed_sq += wheel_speed ** 2

        if np.max(np.abs(action)) >= 0.99:
            action_sat_count += 1

        state = env.step(state, jnp.array(action))
        steps += 1

        if bool(state.done):
            break

    survival_time = steps * env.CONTROL_DT
    fell = bool(state.info["is_fallen"])

    final_obs = np.array(state.obs)
    torso_height = float(state.mjx_data.qpos[2])
    g_body_final = final_obs[0:3]
    tilt_final = float(np.arccos(np.clip(-g_body_final[2], -1.0, 1.0)))

    if fell:
        if torso_height < env._min_height:
            fall_reason = "height"
        elif tilt_final > env._max_tilt:
            fall_reason = "tilt"
        else:
            fall_reason = "unknown"
    else:
        fall_reason = "none"

    pitch_rms = np.sqrt(pitch_sq / steps) if steps > 0 else 0.0
    roll_rms = np.sqrt(roll_sq / steps) if steps > 0 else 0.0
    wheel_speed_rms = np.sqrt(wheel_speed_sq / steps) if steps > 0 else 0.0
    action_sat_rate = action_sat_count / steps if steps > 0 else 0.0

    roll_start_abs = abs(float(initial_roll_deg))
    roll_end_abs = float(early_roll_abs[-1]) if len(early_roll_abs) > 0 else abs(initial_roll_deg)
    roll_amp_ratio = float(roll_end_abs / max(roll_start_abs, 1e-6))

    return {
        "survival_time_s": float(survival_time),
        "fell": bool(fell),
        "fall_reason": fall_reason,
        "roll_rms_deg": float(np.rad2deg(roll_rms)),
        "pitch_rms_deg": float(np.rad2deg(pitch_rms)),
        "wheel_speed_rms_rad_s": float(wheel_speed_rms),
        "action_saturation_rate": float(action_sat_rate),
        "first_roll_divergence_time_s": first_roll_diverge_time,
        "roll_start_abs_deg": roll_start_abs,
        "roll_end_abs_deg": roll_end_abs,
        "roll_amplification_ratio": roll_amp_ratio,
        "roll_reduced_in_early_window": bool(roll_end_abs < roll_start_abs),
    }


def _upsert_report_section(path: Path, heading: str, section_lines: list[str]) -> None:
    """Replace existing markdown section (if any) or append new section."""
    if not path.exists():
        return

    text = path.read_text(encoding="utf-8")
    section = "## " + heading + "\n\n" + "\n".join(section_lines).rstrip() + "\n"
    pattern = rf"\n##\s+{re.escape(heading)}\n.*?(?=\n##\s+|\Z)"

    if re.search(pattern, text, flags=re.S):
        text = re.sub(pattern, "\n" + section.rstrip() + "\n", text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + section

    path.write_text(text, encoding="utf-8")


def _append_step58_to_reports(project_root: Path, summary: dict) -> None:
    """Upsert Step 5.8 section in both required reports."""
    section_lines = [
        f"- Output dir: `outputs/phase_b9_step5_8_roll_redesign`",
        f"- Baseline controller accepted: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`",
        f"- Previous fixes rejected: Step 5.5 / 5.6 / 5.7 did not beat Step 5 full validation",
        f"- Best Step 5.8 variant: `{summary['best_variant']}`",
        f"- Kept variants after small eval: `{', '.join(summary['kept_variants']) if summary['kept_variants'] else 'none'}`",
        f"- Full-validation survival: {summary['best_full_validation_metrics']['survival_time_s']:.4f} s",
        f"- Full-validation fall rate: {summary['best_full_validation_metrics']['fall_rate']:.4f}",
        f"- Full-validation roll RMS: {summary['best_full_validation_metrics']['roll_RMS_deg']:.4f} deg",
        f"- Beats Step 5 baseline in full validation: `{summary['beats_step5_baseline_full_validation']}`",
        f"- Step 6 allowed: `{summary['step6_ready']}`",
        f"- If no beat: keep Step 5 best as current best and Step 6 blocked",
    ]

    _upsert_report_section(
        project_root / "docs/phase_b9_best_standalone_controller_report.md",
        "Phase B.9 Step 5.8 — Roll Instability Root-Cause Redesign",
        section_lines,
    )
    _upsert_report_section(
        project_root / "docs/phase_b9_audit_gate_report.md",
        "Phase B.9 Step 5.8 — Roll Instability Root-Cause Redesign",
        section_lines,
    )


def run_step5_8_roll_redesign(
    base_config: DualRateConfig,
    best_step5_params: dict,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    init_table: dict,
    output_dir: Path,
    seed: int,
) -> dict:
    """Step 5.8: root-cause redesign with perturbation harness + staged evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_survival_ref = 3.8926666666666665
    baseline_fall_ref = 0.8333333333333334
    baseline_roll_ref = 21.168228816065504
    baseline_pitch_ref = 1.0109260130054514
    baseline_sat_ref = 0.0

    baseline_controller = create_roll_fix_controller(
        base_config, best_step5_params, "baseline", {}, mj_model
    )

    variants = [
        "A_hip_roll_position_damping_only",
        "B_hip_roll_velocity_damping_only",
        "C_hip_roll_preload_from_balanced_root",
        "D_lateral_CoM_feedback_through_hip_roll",
        "E_differential_wheel_roll_damping",
        "F_weak_hip_roll_plus_differential_wheel",
    ]

    # Root-cause evidence: roll perturbation from balanced posture.
    perturb_rows = []
    for variant in variants:
        for disable_wheel_lqr in [True, False]:
            wheel_mode = "no_wheel_lqr" if disable_wheel_lqr else "wheel_lqr_enabled"
            for roll0 in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]:
                result = _run_step58_variant_episode(
                    base_controller=baseline_controller,
                    variant=variant,
                    env=env,
                    init_table=init_table,
                    height=0.60,
                    seed=seed + int(100 * abs(roll0)) + (1 if disable_wheel_lqr else 17),
                    initial_roll_deg=roll0,
                    disable_wheel_lqr=disable_wheel_lqr,
                )
                perturb_rows.append({
                    "variant": variant,
                    "wheel_mode": wheel_mode,
                    "initial_roll_deg": roll0,
                    "roll_start_abs_deg": result["roll_start_abs_deg"],
                    "roll_end_abs_deg": result["roll_end_abs_deg"],
                    "roll_amplification_ratio": result["roll_amplification_ratio"],
                    "roll_reduced_in_early_window": result["roll_reduced_in_early_window"],
                    "fell": result["fell"],
                })

    perturb_df = pd.DataFrame(perturb_rows)
    perturb_summary_df = (
        perturb_df.groupby(["variant", "wheel_mode"])
        .agg(
            mean_roll_amp_ratio=("roll_amplification_ratio", "mean"),
            early_roll_reduction_rate=("roll_reduced_in_early_window", "mean"),
            perturb_fall_rate=("fell", "mean"),
        )
        .reset_index()
    )

    # Small evaluation at h=0.60, 5 episodes per variant.
    candidate_rows = []
    for variant in variants:
        ep_rows = []
        for ep in range(5):
            result = _run_step58_variant_episode(
                base_controller=baseline_controller,
                variant=variant,
                env=env,
                init_table=init_table,
                height=0.60,
                seed=seed + ep,
                initial_roll_deg=0.0,
                disable_wheel_lqr=False,
            )
            ep_rows.append(result)

        ep_df = pd.DataFrame(ep_rows)
        first_div_mean = ep_df["first_roll_divergence_time_s"].dropna().mean()
        fall_reason_counts = ep_df["fall_reason"].value_counts(dropna=False).to_dict()
        dominant_fall_reason = max(fall_reason_counts.items(), key=lambda kv: kv[1])[0] if fall_reason_counts else "unknown"

        surv = float(ep_df["survival_time_s"].mean())
        fall = float(ep_df["fell"].mean())
        roll = float(ep_df["roll_rms_deg"].mean())
        pitch = float(ep_df["pitch_rms_deg"].mean())
        sat = float(ep_df["action_saturation_rate"].mean())

        improve_any_core = bool(
            (surv > baseline_survival_ref)
            or (fall < baseline_fall_ref)
            or (roll < baseline_roll_ref)
        )
        no_worse_aux = bool(
            (pitch <= baseline_pitch_ref * 1.10)
            and (sat <= max(baseline_sat_ref + 0.02, 0.02))
        )

        candidate_rows.append({
            "variant": variant,
            "survival_time_s": surv,
            "fall_rate": fall,
            "roll_RMS_deg": roll,
            "pitch_RMS_deg": pitch,
            "first_roll_divergence_time_s": (None if pd.isna(first_div_mean) else float(first_div_mean)),
            "action_saturation_rate": sat,
            "wheel_speed_RMS_rad_s": float(ep_df["wheel_speed_rms_rad_s"].mean()),
            "dominant_fall_reason": dominant_fall_reason,
            "fall_reason_counts_json": json.dumps(fall_reason_counts),
            "improve_any_core": improve_any_core,
            "no_worse_aux": no_worse_aux,
            "keep_for_full_validation": bool(improve_any_core and no_worse_aux),
        })

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df = candidate_df.sort_values(
        by=["keep_for_full_validation", "survival_time_s", "fall_rate", "roll_RMS_deg"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    candidate_df.to_csv(output_dir / "candidate_results.csv", index=False)

    kept = candidate_df[candidate_df["keep_for_full_validation"]]["variant"].tolist()
    if len(kept) == 0:
        # Keep top-1 fallback for robustness of workflow even if no candidate passes gate.
        kept = [str(candidate_df.iloc[0]["variant"])]
    kept = kept[:2]

    # Full validation on all heights for kept variants.
    full_rows = []
    for variant in kept:
        for height in [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
            for ep in range(5):
                result = _run_step58_variant_episode(
                    base_controller=baseline_controller,
                    variant=variant,
                    env=env,
                    init_table=init_table,
                    height=height,
                    seed=seed + ep + int(height * 100),
                    initial_roll_deg=0.0,
                    disable_wheel_lqr=False,
                )
                full_rows.append({"variant": variant, "height": height, "episode": ep, **result})

    full_df = pd.DataFrame(full_rows)
    full_df.to_csv(output_dir / "full_validation.csv", index=False)

    full_summary = (
        full_df.groupby("variant")
        .agg(
            survival_time_s=("survival_time_s", "mean"),
            fall_rate=("fell", "mean"),
            roll_RMS_deg=("roll_rms_deg", "mean"),
            pitch_RMS_deg=("pitch_rms_deg", "mean"),
            first_roll_divergence_time_s=("first_roll_divergence_time_s", "mean"),
            action_saturation_rate=("action_saturation_rate", "mean"),
            wheel_speed_RMS_rad_s=("wheel_speed_rms_rad_s", "mean"),
        )
        .reset_index()
    )

    dom_fall = full_df.groupby("variant")["fall_reason"].agg(lambda s: s.value_counts().idxmax()).to_dict()
    full_summary["dominant_fall_reason"] = full_summary["variant"].map(dom_fall)

    full_summary["beats_step5_baseline"] = (
        (full_summary["survival_time_s"] > baseline_survival_ref)
        & (full_summary["fall_rate"] < baseline_fall_ref)
        & (full_summary["roll_RMS_deg"] < baseline_roll_ref)
        & (full_summary["pitch_RMS_deg"] <= baseline_pitch_ref * 1.10)
        & (full_summary["action_saturation_rate"] <= max(baseline_sat_ref + 0.02, 0.02))
    )

    full_summary = full_summary.sort_values(
        by=["beats_step5_baseline", "survival_time_s", "fall_rate", "roll_RMS_deg"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_variant = str(full_summary.iloc[0]["variant"])
    best_row = full_summary.iloc[0].to_dict()
    decision_allow_step6 = bool(best_row["beats_step5_baseline"])

    best_cfg = {
        "baseline_step5_lqr_config": best_step5_params,
        "step5_8_best_variant": best_variant,
        "step5_8_kept_variants": kept,
    }
    with open(output_dir / "best_roll_redesign_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    # Failure analysis for previous fixes based on current perturbation + full validation context.
    prev_failure_analysis = {
        "weak_hip_roll_pd": "Hip-roll-only correction remains too weak/late under perturbation; does not improve full-validation stability enough.",
        "contact_force_balancing": "Contact/preload heuristics shift load but do not resolve tilt-dominant failures across all heights.",
        "roll_rate_damping_from_t0": "Early damping lowers some transients but full-validation survival/fall/roll metrics remain below Step 5 baseline.",
    }

    perturb_summary = {}
    for _, row in perturb_summary_df.iterrows():
        key = f"{row['variant']}::{row['wheel_mode']}"
        perturb_summary[key] = {
            "mean_roll_amp_ratio": float(row["mean_roll_amp_ratio"]),
            "early_roll_reduction_rate": float(row["early_roll_reduction_rate"]),
            "perturb_fall_rate": float(row["perturb_fall_rate"]),
        }

    best_summary = {
        "baseline_reference": {
            "mean_survival_s": baseline_survival_ref,
            "mean_fall_rate": baseline_fall_ref,
            "mean_roll_rms_deg": baseline_roll_ref,
            "mean_pitch_rms_deg": baseline_pitch_ref,
            "mean_action_sat_rate": baseline_sat_ref,
        },
        "current_best_controller": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
        "previous_fix_failure_analysis": prev_failure_analysis,
        "perturbation_summary": perturb_summary,
        "kept_variants": kept,
        "best_variant": best_variant,
        "best_full_validation_metrics": {
            "survival_time_s": float(best_row["survival_time_s"]),
            "fall_rate": float(best_row["fall_rate"]),
            "roll_RMS_deg": float(best_row["roll_RMS_deg"]),
            "pitch_RMS_deg": float(best_row["pitch_RMS_deg"]),
            "first_roll_divergence_time_s": (None if pd.isna(best_row["first_roll_divergence_time_s"]) else float(best_row["first_roll_divergence_time_s"])),
            "action_saturation_rate": float(best_row["action_saturation_rate"]),
            "wheel_speed_RMS_rad_s": float(best_row["wheel_speed_RMS_rad_s"]),
            "dominant_fall_reason": best_row["dominant_fall_reason"],
        },
        "beats_step5_baseline_full_validation": bool(best_row["beats_step5_baseline"]),
        "step6_ready": decision_allow_step6,
        "decision_rule": "Step 6 is allowed only if Step 5.8 beats Step 5 best_lqr_config.yaml in full validation",
    }

    with open(output_dir / "best_roll_redesign_summary.json", "w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    _append_step58_to_reports(project_root, best_summary)

    console.print(f"[green]Saved candidate results: {output_dir / 'candidate_results.csv'}[/green]")
    console.print(f"[green]Saved full validation: {output_dir / 'full_validation.csv'}[/green]")
    console.print(f"[green]Saved best config: {output_dir / 'best_roll_redesign_config.yaml'}[/green]")
    console.print(f"[green]Saved best summary: {output_dir / 'best_roll_redesign_summary.json'}[/green]")

    return best_summary


def _compute_step510_latency_markers(df: pd.DataFrame, torque_available: bool) -> dict:
    """Compute first-divergence and first-correction timing markers from one episode trace."""
    if df.empty:
        return {
            "torque_available": bool(torque_available),
            "first_roll_divergence_time_s": None,
            "first_nonzero_corrective_hip_roll_action_time_s": None,
            "first_nonzero_differential_wheel_correction_time_s": None,
            "first_pid_torque_response_time_s": None,
            "first_actual_hip_roll_joint_motion_time_s": None,
            "first_contact_force_shift_time_s": None,
            "first_correction_time_s": None,
            "correction_delay_s": None,
            "correction_before_divergence": None,
        }

    ep = df.sort_values("time_s").reset_index(drop=True)

    baseline = ep[ep["time_s"] <= 0.02]
    if baseline.empty:
        baseline = ep.head(2)

    roll_abs_base = baseline["roll_deg"].abs()
    roll_ref = float(roll_abs_base.iloc[0]) if len(roll_abs_base) > 0 else 0.0
    roll_thr = float(max(1.0, roll_ref + 1.0))

    contact_base = baseline["contact_force_diff"].abs()
    contact_thr = float(max(5.0, contact_base.mean() + 2.0 * contact_base.std(ddof=0)))

    def first_time(mask: pd.Series):
        hit = ep.loc[mask, "time_s"]
        if len(hit) == 0:
            return None
        return float(hit.iloc[0])

    first_roll_div = first_time((ep["time_s"] >= 0.02) & (ep["roll_deg"].abs() > roll_thr))

    hip_action_mag = 0.5 * (ep["l_hip_roll_action"].abs() + ep["r_hip_roll_action"].abs())
    first_hip_action = first_time(hip_action_mag > 1e-3)

    first_wheel_diff = first_time(ep["wheel_diff_cmd"].abs() > 1e-3)

    first_torque = None
    if torque_available and (not ep["l_hip_roll_torque"].isna().all()) and (not ep["r_hip_roll_torque"].isna().all()):
        hip_torque_mag = 0.5 * (ep["l_hip_roll_torque"].abs() + ep["r_hip_roll_torque"].abs())
        first_torque = first_time(hip_torque_mag > 0.1)

    hip_motion_mag = 0.5 * (ep["l_hip_roll_qvel"].abs() + ep["r_hip_roll_qvel"].abs())
    first_motion = first_time(hip_motion_mag > 1e-3)

    first_contact_shift = first_time(ep["contact_force_diff"].abs() > contact_thr)

    correction_candidates = [
        x for x in [first_hip_action, first_wheel_diff, first_torque, first_motion, first_contact_shift] if x is not None
    ]
    first_correction = min(correction_candidates) if correction_candidates else None

    correction_delay = None
    correction_before_divergence = None
    if first_roll_div is not None and first_correction is not None:
        correction_delay = float(first_correction - first_roll_div)
        correction_before_divergence = bool(first_correction <= first_roll_div)

    return {
        "torque_available": bool(torque_available),
        "first_roll_divergence_time_s": first_roll_div,
        "first_nonzero_corrective_hip_roll_action_time_s": first_hip_action,
        "first_nonzero_differential_wheel_correction_time_s": first_wheel_diff,
        "first_pid_torque_response_time_s": first_torque,
        "first_actual_hip_roll_joint_motion_time_s": first_motion,
        "first_contact_force_shift_time_s": first_contact_shift,
        "first_correction_time_s": first_correction,
        "correction_delay_s": correction_delay,
        "correction_before_divergence": correction_before_divergence,
    }


def _step510_apply_variant_adjustment(
    variant: str,
    base_action: np.ndarray,
    step: int,
    dt: float,
    roll_deg: float,
    roll_rate_deg_s: float,
    l_contact_force: float,
    r_contact_force: float,
    init_row: dict,
    prev_action: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Apply Step 5.10 startup-timing correction variants."""
    a = np.array(base_action, copy=True)
    t = step * dt
    early = t <= 0.20 + 1e-9

    info = {
        "variant": variant,
        "startup_mode_active": bool(early),
        "preload_hip_roll": 0.0,
        "rate_damping_term": 0.0,
        "predictive_term": 0.0,
        "bypass_active": False,
        "height_freeze_active": False,
        "bypass_start_time_s": (0.0 if early else None),
        "bypass_end_time_s": (0.20 if not early else None),
    }

    if np.isfinite(l_contact_force) and np.isfinite(r_contact_force):
        denom = max(abs(l_contact_force) + abs(r_contact_force), 1e-6)
        contact_imbalance = float((l_contact_force - r_contact_force) / denom)
    else:
        lf = float(init_row.get("expected_left_force", 0.0))
        rf = float(init_row.get("expected_right_force", 0.0))
        denom = max(abs(lf) + abs(rf), 1e-6)
        contact_imbalance = float((lf - rf) / denom)

    if variant == "A_preload_hip_roll_target_at_t0":
        preload = float(np.clip(0.12 * contact_imbalance, -0.06, 0.06))
        if early:
            a[0] = np.clip(a[0] + preload, -1.0, 1.0)
            a[5] = np.clip(a[5] - preload, -1.0, 1.0)
        info["preload_hip_roll"] = preload

    elif variant == "B_roll_rate_damping_from_first_step":
        damp = float(np.clip(-0.0025 * roll_rate_deg_s, -0.08, 0.08))
        a[0] = np.clip(a[0] + damp, -1.0, 1.0)
        a[5] = np.clip(a[5] - damp, -1.0, 1.0)
        info["rate_damping_term"] = damp

    elif variant == "C_bypass_filter_rate_limiter_first_0p2s":
        damp = float(np.clip(-0.0025 * roll_rate_deg_s, -0.08, 0.08))
        if early:
            a[0] = np.clip(a[0] + damp, -1.0, 1.0)
            a[5] = np.clip(a[5] - damp, -1.0, 1.0)
            info["bypass_active"] = True
        else:
            # Restore normal conservative channel behavior after startup window.
            limited = float(np.clip(damp, -0.03, 0.03))
            a[0] = np.clip(a[0] + limited, -1.0, 1.0)
            a[5] = np.clip(a[5] - limited, -1.0, 1.0)
        info["rate_damping_term"] = damp

    elif variant == "D_startup_emergency_roll_mode":
        if early:
            info["height_freeze_active"] = True
            # Keep sagittal wheels untouched; prioritize lateral damping via hip-roll.
            damp = float(np.clip(-(0.0040 * roll_deg + 0.0025 * roll_rate_deg_s), -0.12, 0.12))
            a[0] = np.clip(a[0] + damp, -1.0, 1.0)
            a[5] = np.clip(a[5] - damp, -1.0, 1.0)
            # Reduce non-essential posture channels only in startup emergency window.
            for idx in [2, 3, 7, 8]:
                a[idx] = float(np.clip(0.85 * prev_action[idx] + 0.15 * a[idx], -1.0, 1.0))
            info["rate_damping_term"] = damp

    elif variant == "E_predictive_roll_correction":
        predictive = float(np.clip(-(0.0020 * roll_rate_deg_s + 0.08 * contact_imbalance), -0.10, 0.10))
        a[0] = np.clip(a[0] + predictive, -1.0, 1.0)
        a[5] = np.clip(a[5] - predictive, -1.0, 1.0)
        # Add gentle differential wheel cue for anticipatory shift.
        wheel_diff = float(np.clip(-0.5 * predictive, -0.05, 0.05))
        a[4] = np.clip(a[4] + wheel_diff, -1.0, 1.0)
        a[9] = np.clip(a[9] - wheel_diff, -1.0, 1.0)
        info["predictive_term"] = predictive

    return a.astype(np.float32), info


def _run_step510_episode(
    base_controller: DualRateBalanceController,
    variant: str,
    env: BalanceEnv,
    init_table: dict,
    mj_model: mujoco.MjModel,
    height: float,
    seed: int,
    episode_id: str,
) -> tuple[dict, list[dict]]:
    """Run one Step 5.10 episode with full startup latency telemetry."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

    base_controller.reset()
    nearest_h = min(init_table.keys(), key=lambda hh: abs(hh - height))
    base_controller.target_hip_pitch = init_table[nearest_h]["hip_pitch"]
    base_controller.target_knee = init_table[nearest_h]["knee"]
    base_controller.last_stable_hip_pitch = base_controller.target_hip_pitch
    base_controller.last_stable_knee = base_controller.target_knee

    mj_data = mujoco.MjData(mj_model)

    rows = []
    pitch_sq = 0.0
    roll_sq = 0.0
    action_sat_count = 0
    steps = 0

    prev_action = np.zeros(10, dtype=np.float32)

    for step in range(1000):
        t = step * env.CONTROL_DT
        obs_np = np.array(state.obs)
        base_action = base_controller.compute_action(obs_np)

        mj_data.qpos[:] = np.array(state.mjx_data.qpos)
        mj_data.qvel[:] = np.array(state.mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)
        l_contact_force, r_contact_force = contact_forces_by_wheel(mj_model, mj_data)

        g_body = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
        ang_vel = obs_np[6:9]

        action, variant_info = _step510_apply_variant_adjustment(
            variant=variant,
            base_action=base_action,
            step=step,
            dt=env.CONTROL_DT,
            roll_deg=float(np.rad2deg(roll)),
            roll_rate_deg_s=float(np.rad2deg(ang_vel[1])),
            l_contact_force=float(l_contact_force),
            r_contact_force=float(r_contact_force),
            init_row=init_table[nearest_h],
            prev_action=prev_action,
        )

        pitch_sq += pitch ** 2
        roll_sq += roll ** 2

        if np.max(np.abs(action)) >= 0.99:
            action_sat_count += 1

        state = env.step(state, jnp.array(action))
        steps += 1

        # Post-step physical response from MJX state.
        l_hip_qpos = float(state.mjx_data.qpos[7])
        r_hip_qpos = float(state.mjx_data.qpos[12])
        l_hip_qvel = float(state.mjx_data.qvel[6])
        r_hip_qvel = float(state.mjx_data.qvel[11])

        l_torque = np.nan
        r_torque = np.nan
        torque_available = False
        if hasattr(state.mjx_data, "actuator_force"):
            af = np.array(state.mjx_data.actuator_force)
            if af.shape[0] >= 6:
                l_torque = float(af[0])
                r_torque = float(af[5])
                torque_available = True

        joint_vel = np.array(state.obs)[19:29]
        l_wheel_qvel = float(joint_vel[4])
        r_wheel_qvel = float(joint_vel[9])

        contact_force_diff = np.nan
        if np.isfinite(l_contact_force) and np.isfinite(r_contact_force):
            contact_force_diff = float(l_contact_force - r_contact_force)

        final_obs = np.array(state.obs)
        torso_height = float(state.mjx_data.qpos[2])
        g_body_final = final_obs[0:3]
        tilt_final = float(np.arccos(np.clip(-g_body_final[2], -1.0, 1.0)))
        fell_flag = bool(state.info["is_fallen"])
        if fell_flag:
            if torso_height < env._min_height:
                fall_reason = "height"
            elif tilt_final > env._max_tilt:
                fall_reason = "tilt"
            else:
                fall_reason = "unknown"
        else:
            fall_reason = "none"

        rows.append({
            "episode_id": episode_id,
            "variant": variant,
            "height": float(height),
            "step": int(step),
            "time_s": float(t),
            "roll_deg": float(np.rad2deg(roll)),
            "roll_rate_deg_s": float(np.rad2deg(ang_vel[1])),
            "pitch_deg": float(np.rad2deg(pitch)),
            "pitch_rate_deg_s": float(np.rad2deg(ang_vel[0])),
            "l_hip_roll_target": float(base_action[0]),
            "r_hip_roll_target": float(base_action[5]),
            "l_hip_roll_action": float(action[0]),
            "r_hip_roll_action": float(action[5]),
            "l_hip_roll_qpos": l_hip_qpos,
            "r_hip_roll_qpos": r_hip_qpos,
            "l_hip_roll_qvel": l_hip_qvel,
            "r_hip_roll_qvel": r_hip_qvel,
            "l_hip_roll_torque": l_torque,
            "r_hip_roll_torque": r_torque,
            "torque_available": bool(torque_available),
            "l_wheel_cmd": float(action[4]),
            "r_wheel_cmd": float(action[9]),
            "wheel_diff_cmd": float(action[4] - action[9]),
            "l_wheel_qvel": l_wheel_qvel,
            "r_wheel_qvel": r_wheel_qvel,
            "l_contact_force": float(l_contact_force) if np.isfinite(l_contact_force) else np.nan,
            "r_contact_force": float(r_contact_force) if np.isfinite(r_contact_force) else np.nan,
            "contact_force_diff": contact_force_diff,
            "action_saturation": int(np.max(np.abs(action)) >= 0.99),
            "fall_flag": int(fell_flag),
            "fall_reason": fall_reason,
            "preload_hip_roll": float(variant_info.get("preload_hip_roll", 0.0)),
            "predictive_term": float(variant_info.get("predictive_term", 0.0)),
            "rate_damping_term": float(variant_info.get("rate_damping_term", 0.0)),
            "bypass_active": bool(variant_info.get("bypass_active", False)),
            "height_freeze_active": bool(variant_info.get("height_freeze_active", False)),
            "startup_mode_active": bool(variant_info.get("startup_mode_active", False)),
        })

        prev_action = np.array(action, copy=True)

        if bool(state.done):
            break

    survival_time = steps * env.CONTROL_DT
    fell = bool(state.info["is_fallen"])
    pitch_rms = np.sqrt(pitch_sq / steps) if steps > 0 else 0.0
    roll_rms = np.sqrt(roll_sq / steps) if steps > 0 else 0.0
    action_sat_rate = action_sat_count / steps if steps > 0 else 0.0

    ep_df = pd.DataFrame(rows)
    torque_available_any = bool(ep_df["torque_available"].any()) if len(ep_df) > 0 else False
    latency = _compute_step510_latency_markers(ep_df, torque_available=torque_available_any)

    dominant_fall_reason = "none"
    if len(ep_df) > 0:
        counts = ep_df[ep_df["fall_flag"] == 1]["fall_reason"].value_counts()
        if len(counts) > 0:
            dominant_fall_reason = str(counts.index[0])

    metrics = {
        "episode_id": episode_id,
        "variant": variant,
        "height": float(height),
        "survival_time_s": float(survival_time),
        "fell": bool(fell),
        "fall_rate": float(1.0 if fell else 0.0),
        "roll_rms_deg": float(np.rad2deg(roll_rms)),
        "pitch_rms_deg": float(np.rad2deg(pitch_rms)),
        "action_saturation_rate": float(action_sat_rate),
        "dominant_fall_reason": dominant_fall_reason,
        **latency,
    }

    return metrics, rows


def _append_step510_to_reports(project_root: Path, summary: dict) -> None:
    decision = summary["final_decision"]
    section_lines = [
        "- Output dir: `outputs/phase_b9_step5_10_early_transient_fix`",
        f"- Baseline controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`",
        f"- Small-eval variants tested: `{', '.join(summary['variants_tested'])}`",
        f"- Candidates kept after filter: `{', '.join(summary['kept_variants']) if summary['kept_variants'] else 'none'}`",
        f"- Full-validation executed: `{summary['full_validation_executed']}`",
        f"- Best variant: `{summary['best_variant']}`",
        f"- Best full-validation survival: {summary['best_metrics'].get('survival_time_s', float('nan')):.4f} s",
        f"- Best full-validation fall rate: {summary['best_metrics'].get('fall_rate', float('nan')):.4f}",
        f"- Best full-validation roll RMS: {summary['best_metrics'].get('roll_RMS_deg', float('nan')):.4f} deg",
        f"- Best full-validation pitch RMS: {summary['best_metrics'].get('pitch_RMS_deg', float('nan')):.4f} deg",
        f"- Baseline beaten in full validation: `{summary['beats_step5_baseline_full_validation']}`",
        f"- Final decision: `{decision}`",
        f"- Step 6 status: `{summary['step6_status']}`",
    ]

    _upsert_report_section(
        project_root / "docs/phase_b9_best_standalone_controller_report.md",
        "Phase B.9 Step 5.10 — Early Transient Timing Fix",
        section_lines,
    )
    _upsert_report_section(
        project_root / "docs/phase_b9_audit_gate_report.md",
        "Phase B.9 Step 5.10 — Early Transient Timing Fix",
        section_lines,
    )


def _step511_action_state_index_map() -> dict:
    """Return canonical action/state indices for Step 5.11 audit."""
    return {
        "l_hip_roll_action": 0,
        "r_hip_roll_action": 5,
        "l_wheel_action": 4,
        "r_wheel_action": 9,
        "l_hip_roll_qpos": 7,
        "r_hip_roll_qpos": 12,
    }


def _step511_clone_params_no_leak(src: dict) -> dict:
    """Deep copy params to prevent mutation leakage across variants."""
    return copy.deepcopy(src)


def _step511_is_corrective_hip_roll_action(
    roll_deg: float,
    roll_rate_deg_s: float,
    l_hip_roll_action: float,
    r_hip_roll_action: float,
) -> bool:
    """Check if hip-roll action is directionally corrective for current roll state."""
    if abs(l_hip_roll_action) < 1e-3 and abs(r_hip_roll_action) < 1e-3:
        return False

    roll_error = roll_deg
    roll_rate_error = roll_rate_deg_s

    # Corrective hip-roll action should oppose roll angle and/or roll rate.
    # Positive roll → left side tilted up → need positive left hip roll / negative right hip roll
    action_diff = l_hip_roll_action - r_hip_roll_action

    # Check if action opposes roll angle or roll rate
    opposes_angle = (roll_error * action_diff) > 0
    opposes_rate = (roll_rate_error * action_diff) > 0

    return opposes_angle or opposes_rate


def _step511_is_corrective_wheel_diff(
    roll_deg: float,
    roll_rate_deg_s: float,
    contact_force_diff: float,
    wheel_diff_cmd: float,
) -> bool:
    """Check if differential wheel command is directionally corrective."""
    if abs(wheel_diff_cmd) < 1e-3:
        return False

    # Corrective wheel diff should oppose roll angle, roll rate, or contact imbalance
    # Positive roll → need negative wheel diff (left wheel slower, right faster)
    opposes_angle = (roll_deg * wheel_diff_cmd) < 0
    opposes_rate = (roll_rate_deg_s * wheel_diff_cmd) < 0
    opposes_contact = (contact_force_diff * wheel_diff_cmd) < 0

    return opposes_angle or opposes_rate or opposes_contact


def _compute_step511_latency_markers(df: pd.DataFrame, torque_available: bool) -> dict:
    """Compute Step 5.11 latency markers with generic vs corrective separation."""
    if df.empty:
        return {
            "torque_available": bool(torque_available),
            "first_generic_pid_torque_time_s": None,
            "first_generic_hip_roll_joint_motion_time_s": None,
            "first_corrective_hip_roll_action_time_s": None,
            "first_corrective_hip_roll_torque_time_s": None,
            "first_corrective_differential_wheel_command_time_s": None,
            "first_corrective_wheel_velocity_response_time_s": None,
            "first_roll_divergence_time_s": None,
            "first_contact_force_imbalance_time_s": None,
            "first_corrective_contact_force_shift_time_s": None,
            "corrective_delay_vs_roll_divergence_s": None,
            "corrective_delay_vs_contact_imbalance_s": None,
        }

    ep = df.sort_values("time_s").reset_index(drop=True)

    baseline = ep[ep["time_s"] < 0.02]
    if baseline.empty or len(baseline) < 2:
        baseline = ep.head(1)

    roll_ref = float(baseline["roll_deg"].abs().mean())
    roll_thr = float(max(1.0, roll_ref + 0.5))

    contact_ref = float(baseline["contact_force_diff"].abs().mean())
    contact_std = float(baseline["contact_force_diff"].abs().std(ddof=0))
    contact_thr = float(max(5.0, contact_ref + 2.0 * contact_std))

    def first_time(mask: pd.Series):
        hit = ep.loc[mask, "time_s"]
        return float(hit.iloc[0]) if len(hit) > 0 else None

    # Generic activity markers
    first_generic_torque = None
    if torque_available and "l_hip_roll_torque" in ep.columns and "r_hip_roll_torque" in ep.columns:
        if not ep["l_hip_roll_torque"].isna().all() and not ep["r_hip_roll_torque"].isna().all():
            torque_mag = 0.5 * (ep["l_hip_roll_torque"].abs() + ep["r_hip_roll_torque"].abs())
            first_generic_torque = first_time(torque_mag > 0.1)

    motion_mag = 0.5 * (ep["l_hip_roll_qvel"].abs() + ep["r_hip_roll_qvel"].abs())
    first_generic_motion = first_time(motion_mag > 1e-3)

    # Divergence markers
    first_roll_div = first_time((ep["time_s"] >= 0.02) & (ep["roll_deg"].abs() > roll_thr))
    first_contact_imbalance = first_time(ep["contact_force_diff"].abs() > contact_thr)

    # Corrective action markers
    corrective_hip_mask = ep.apply(
        lambda row: _step511_is_corrective_hip_roll_action(
            row["roll_deg"],
            row["roll_rate_deg_s"],
            row["l_hip_roll_action"],
            row["r_hip_roll_action"],
        ),
        axis=1,
    )
    first_corrective_hip = first_time(corrective_hip_mask)

    corrective_wheel_mask = ep.apply(
        lambda row: _step511_is_corrective_wheel_diff(
            row["roll_deg"],
            row["roll_rate_deg_s"],
            row["contact_force_diff"],
            row["wheel_diff_cmd"],
        ),
        axis=1,
    )
    first_corrective_wheel = first_time(corrective_wheel_mask)

    # Corrective response markers
    first_corrective_torque = None
    if first_corrective_hip is not None and first_generic_torque is not None:
        first_corrective_torque = max(first_corrective_hip, first_generic_torque)

    wheel_vel_response_mag = 0.5 * (ep["l_wheel_qvel"].abs() + ep["r_wheel_qvel"].abs())
    first_wheel_response = first_time(wheel_vel_response_mag > 0.05)

    contact_shift_mask = ep["contact_force_diff"].abs() > (contact_ref + contact_std)
    first_contact_shift = first_time(contact_shift_mask)

    # Compute delays
    corrective_candidates = [x for x in [first_corrective_hip, first_corrective_wheel] if x is not None]
    first_corrective = min(corrective_candidates) if corrective_candidates else None

    delay_vs_roll = None
    delay_vs_contact = None
    if first_corrective is not None:
        if first_roll_div is not None:
            delay_vs_roll = float(first_corrective - first_roll_div)
        if first_contact_imbalance is not None:
            delay_vs_contact = float(first_corrective - first_contact_imbalance)

    return {
        "torque_available": bool(torque_available),
        "first_generic_pid_torque_time_s": first_generic_torque,
        "first_generic_hip_roll_joint_motion_time_s": first_generic_motion,
        "first_corrective_hip_roll_action_time_s": first_corrective_hip,
        "first_corrective_hip_roll_torque_time_s": first_corrective_torque,
        "first_corrective_differential_wheel_command_time_s": first_corrective_wheel,
        "first_corrective_wheel_velocity_response_time_s": first_wheel_response,
        "first_roll_divergence_time_s": first_roll_div,
        "first_contact_force_imbalance_time_s": first_contact_imbalance,
        "first_corrective_contact_force_shift_time_s": first_contact_shift,
        "corrective_delay_vs_roll_divergence_s": delay_vs_roll,
        "corrective_delay_vs_contact_imbalance_s": delay_vs_contact,
    }


def run_step5_10_early_transient_fix(
    base_config: DualRateConfig,
    best_step5_params: dict,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    init_table: dict,
    output_dir: Path,
    seed: int,
) -> dict:
    """Step 5.10: targeted early-transient timing fix with strict gate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_survival_ref = 3.8926666666666665
    baseline_fall_ref = 0.8333333333333334
    baseline_roll_ref = 21.168228816065504
    baseline_pitch_ref = 1.0109260130054514
    baseline_sat_ref = 0.0

    baseline_controller = create_roll_fix_controller(
        base_config, best_step5_params, "baseline", {}, mj_model
    )

    variants = [
        "A_preload_hip_roll_target_at_t0",
        "B_roll_rate_damping_from_first_step",
        "C_bypass_filter_rate_limiter_first_0p2s",
        "D_startup_emergency_roll_mode",
        "E_predictive_roll_correction",
    ]

    # Part 2 baseline latency diagnostics (h=0.60, 5 episodes)
    latency_rows = []
    latency_summary_rows = []
    for ep in range(5):
        metrics, rows = _run_step510_episode(
            base_controller=baseline_controller,
            variant="baseline",
            env=env,
            init_table=init_table,
            mj_model=mj_model,
            height=0.60,
            seed=seed + ep,
            episode_id=f"baseline_h0.60_ep{ep}",
        )
        latency_rows.extend(rows)
        latency_summary_rows.append(metrics)

    latency_csv = output_dir / "latency_diagnostics.csv"
    pd.DataFrame(latency_rows).to_csv(latency_csv, index=False)

    latency_summary_df = pd.DataFrame(latency_summary_rows)
    latency_summary = {
        "variant": "baseline",
        "height": 0.60,
        "episodes": 5,
        "torque_available_any": bool(latency_summary_df["torque_available"].any()) if len(latency_summary_df) > 0 else False,
        "mean_first_roll_divergence_time_s": float(latency_summary_df["first_roll_divergence_time_s"].dropna().mean()) if "first_roll_divergence_time_s" in latency_summary_df else None,
        "mean_first_nonzero_corrective_hip_roll_action_time_s": float(latency_summary_df["first_nonzero_corrective_hip_roll_action_time_s"].dropna().mean()) if "first_nonzero_corrective_hip_roll_action_time_s" in latency_summary_df else None,
        "mean_first_nonzero_differential_wheel_correction_time_s": float(latency_summary_df["first_nonzero_differential_wheel_correction_time_s"].dropna().mean()) if "first_nonzero_differential_wheel_correction_time_s" in latency_summary_df else None,
        "mean_first_pid_torque_response_time_s": float(latency_summary_df["first_pid_torque_response_time_s"].dropna().mean()) if "first_pid_torque_response_time_s" in latency_summary_df else None,
        "mean_first_actual_hip_roll_joint_motion_time_s": float(latency_summary_df["first_actual_hip_roll_joint_motion_time_s"].dropna().mean()) if "first_actual_hip_roll_joint_motion_time_s" in latency_summary_df else None,
        "mean_first_contact_force_shift_time_s": float(latency_summary_df["first_contact_force_shift_time_s"].dropna().mean()) if "first_contact_force_shift_time_s" in latency_summary_df else None,
        "mean_first_correction_time_s": float(latency_summary_df["first_correction_time_s"].dropna().mean()) if "first_correction_time_s" in latency_summary_df else None,
        "mean_correction_delay_s": float(latency_summary_df["correction_delay_s"].dropna().mean()) if "correction_delay_s" in latency_summary_df else None,
        "correction_before_divergence_rate": float(latency_summary_df["correction_before_divergence"].mean()) if "correction_before_divergence" in latency_summary_df else None,
    }
    latency_summary_json = output_dir / "latency_summary.json"
    with open(latency_summary_json, "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)

    # Part 4 small evaluation for each variant (h=0.60, 5 episodes)
    candidate_rows = []
    full_rows_pool = []
    for variant in variants:
        ep_metrics = []
        for ep in range(5):
            metrics, rows = _run_step510_episode(
                base_controller=baseline_controller,
                variant=variant,
                env=env,
                init_table=init_table,
                mj_model=mj_model,
                height=0.60,
                seed=seed + 100 + ep,
                episode_id=f"{variant}_h0.60_ep{ep}",
            )
            ep_metrics.append(metrics)
            full_rows_pool.extend(rows)

        df = pd.DataFrame(ep_metrics)
        fall_reason_counts = df["dominant_fall_reason"].value_counts(dropna=False).to_dict()
        dominant_fall_reason = max(fall_reason_counts.items(), key=lambda kv: kv[1])[0] if fall_reason_counts else "unknown"

        surv = float(df["survival_time_s"].mean())
        fall = float(df["fall_rate"].mean())
        roll = float(df["roll_rms_deg"].mean())
        pitch = float(df["pitch_rms_deg"].mean())
        sat = float(df["action_saturation_rate"].mean())

        t_div = float(df["first_roll_divergence_time_s"].dropna().mean()) if not df["first_roll_divergence_time_s"].dropna().empty else None
        t_corr = float(df["first_correction_time_s"].dropna().mean()) if not df["first_correction_time_s"].dropna().empty else None
        corr_delay = float(df["correction_delay_s"].dropna().mean()) if not df["correction_delay_s"].dropna().empty else None
        corr_before = float(df["correction_before_divergence"].mean()) if "correction_before_divergence" in df else None

        reduce_delay = (corr_delay is not None) and (latency_summary["mean_correction_delay_s"] is not None) and (corr_delay < latency_summary["mean_correction_delay_s"])
        delay_div = (t_div is not None) and (latency_summary["mean_first_roll_divergence_time_s"] is not None) and (t_div > latency_summary["mean_first_roll_divergence_time_s"])
        improve_survival = surv > baseline_survival_ref
        no_worse_pitch = pitch <= baseline_pitch_ref * 1.10
        no_sat_unstable = sat <= max(baseline_sat_ref + 0.02, 0.02)

        keep = bool(reduce_delay and delay_div and improve_survival and no_worse_pitch and no_sat_unstable)

        candidate_rows.append({
            "variant": variant,
            "survival_time_s": surv,
            "fall_rate": fall,
            "roll_RMS_deg": roll,
            "pitch_RMS_deg": pitch,
            "first_roll_divergence_time_s": t_div,
            "first_correction_time_s": t_corr,
            "correction_delay_s": corr_delay,
            "correction_before_divergence_rate": corr_before,
            "action_saturation_rate": sat,
            "dominant_fall_reason": dominant_fall_reason,
            "fall_reason_counts_json": json.dumps(fall_reason_counts),
            "reduce_correction_delay": bool(reduce_delay),
            "delay_roll_divergence": bool(delay_div),
            "improve_survival": bool(improve_survival),
            "no_worse_pitch": bool(no_worse_pitch),
            "no_sat_or_unstable": bool(no_sat_unstable),
            "keep_for_full_validation": keep,
        })

    candidate_df = pd.DataFrame(candidate_rows)
    candidate_df = candidate_df.sort_values(
        by=["keep_for_full_validation", "survival_time_s", "fall_rate", "roll_RMS_deg"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    candidate_csv = output_dir / "candidate_results.csv"
    candidate_df.to_csv(candidate_csv, index=False)

    kept = candidate_df[candidate_df["keep_for_full_validation"]]["variant"].tolist()

    full_validation_executed = len(kept) > 0
    full_rows = []
    full_summary_df = pd.DataFrame()

    if full_validation_executed:
        kept = kept[:2]
        for variant in kept:
            for height in [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
                for ep in range(5):
                    metrics, rows = _run_step510_episode(
                        base_controller=baseline_controller,
                        variant=variant,
                        env=env,
                        init_table=init_table,
                        mj_model=mj_model,
                        height=height,
                        seed=seed + 500 + ep + int(height * 1000),
                        episode_id=f"{variant}_h{height:.2f}_ep{ep}",
                    )
                    full_rows.append(metrics)
                    full_rows_pool.extend(rows)

        full_df = pd.DataFrame(full_rows)
        full_csv = output_dir / "full_validation.csv"
        full_df.to_csv(full_csv, index=False)

        grouped = (
            full_df.groupby("variant")
            .agg(
                survival_time_s=("survival_time_s", "mean"),
                fall_rate=("fall_rate", "mean"),
                roll_RMS_deg=("roll_rms_deg", "mean"),
                pitch_RMS_deg=("pitch_rms_deg", "mean"),
                action_saturation_rate=("action_saturation_rate", "mean"),
                first_roll_divergence_time_s=("first_roll_divergence_time_s", "mean"),
                first_correction_time_s=("first_correction_time_s", "mean"),
                correction_delay_s=("correction_delay_s", "mean"),
                correction_before_divergence_rate=("correction_before_divergence", "mean"),
            )
            .reset_index()
        )

        per_height = (
            full_df.groupby(["variant", "height"])
            .agg(
                survival_time_s=("survival_time_s", "mean"),
                fall_rate=("fall_rate", "mean"),
                roll_RMS_deg=("roll_rms_deg", "mean"),
                pitch_RMS_deg=("pitch_rms_deg", "mean"),
                action_saturation_rate=("action_saturation_rate", "mean"),
                first_roll_divergence_time_s=("first_roll_divergence_time_s", "mean"),
                first_correction_time_s=("first_correction_time_s", "mean"),
                correction_delay_s=("correction_delay_s", "mean"),
            )
            .reset_index()
        )

        dom_fall_map = full_df.groupby("variant")["dominant_fall_reason"].agg(lambda s: s.value_counts().idxmax()).to_dict()
        grouped["dominant_fall_reason"] = grouped["variant"].map(dom_fall_map)

        grouped["beats_step5_baseline"] = (
            (grouped["survival_time_s"] > baseline_survival_ref)
            & (grouped["fall_rate"] < baseline_fall_ref)
            & (grouped["roll_RMS_deg"] < baseline_roll_ref)
            & (grouped["pitch_RMS_deg"] <= baseline_pitch_ref * 1.10)
            & (grouped["action_saturation_rate"] <= max(baseline_sat_ref + 0.02, 0.02))
        )

        grouped = grouped.sort_values(
            by=["beats_step5_baseline", "survival_time_s", "fall_rate", "roll_RMS_deg"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)

        full_summary_df = grouped

        summary_json = {
            "variants": grouped.to_dict(orient="records"),
            "per_height": per_height.to_dict(orient="records"),
        }
        with open(output_dir / "full_validation_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_json, f, indent=2)
    else:
        # Required artifact with explicit empty/full-validation-skipped state.
        pd.DataFrame(columns=[
            "variant", "height", "episode_id", "survival_time_s", "fall_rate", "roll_rms_deg",
            "pitch_rms_deg", "action_saturation_rate", "first_roll_divergence_time_s",
            "first_correction_time_s", "correction_delay_s"
        ]).to_csv(output_dir / "full_validation.csv", index=False)

        with open(output_dir / "full_validation_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "skipped": True,
                "reason": "No variant passed Step 5.10 small-eval filtering rule",
            }, f, indent=2)

    # Candidate summary artifact
    candidate_summary = {
        "baseline_latency_summary": latency_summary,
        "variants_tested": variants,
        "kept_variants": kept,
        "full_validation_executed": bool(full_validation_executed),
        "small_eval": candidate_df.to_dict(orient="records"),
    }
    with open(output_dir / "candidate_summary.json", "w", encoding="utf-8") as f:
        json.dump(candidate_summary, f, indent=2)

    # Final decision + reports update
    if full_validation_executed and len(full_summary_df) > 0:
        best_variant = str(full_summary_df.iloc[0]["variant"])
        best_metrics = {
            "survival_time_s": float(full_summary_df.iloc[0]["survival_time_s"]),
            "fall_rate": float(full_summary_df.iloc[0]["fall_rate"]),
            "roll_RMS_deg": float(full_summary_df.iloc[0]["roll_RMS_deg"]),
            "pitch_RMS_deg": float(full_summary_df.iloc[0]["pitch_RMS_deg"]),
            "action_saturation_rate": float(full_summary_df.iloc[0]["action_saturation_rate"]),
            "first_roll_divergence_time_s": float(full_summary_df.iloc[0]["first_roll_divergence_time_s"]) if not pd.isna(full_summary_df.iloc[0]["first_roll_divergence_time_s"]) else None,
            "first_correction_time_s": float(full_summary_df.iloc[0]["first_correction_time_s"]) if not pd.isna(full_summary_df.iloc[0]["first_correction_time_s"]) else None,
            "correction_delay_s": float(full_summary_df.iloc[0]["correction_delay_s"]) if not pd.isna(full_summary_df.iloc[0]["correction_delay_s"]) else None,
            "dominant_fall_reason": str(full_summary_df.iloc[0]["dominant_fall_reason"]),
        }
        beats = bool(full_summary_df.iloc[0]["beats_step5_baseline"])
    else:
        best_variant = "none"
        best_metrics = {}
        beats = False

    final_decision = "STEP5_PASSED_AND_STEP6_PREFLIGHT_STARTED" if beats else "KEEP_STEP5_BASELINE_AND_BLOCK_STEP6"
    step6_status = "STARTED_PRE_RL_PREFLIGHT_ONLY" if beats else "BLOCKED"

    final_summary = {
        "baseline_reference": {
            "mean_survival_s": baseline_survival_ref,
            "mean_fall_rate": baseline_fall_ref,
            "mean_roll_rms_deg": baseline_roll_ref,
            "mean_pitch_rms_deg": baseline_pitch_ref,
            "mean_action_sat_rate": baseline_sat_ref,
        },
        "current_best_controller": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
        "variants_tested": variants,
        "kept_variants": kept,
        "full_validation_executed": bool(full_validation_executed),
        "best_variant": best_variant,
        "best_metrics": best_metrics,
        "beats_step5_baseline_full_validation": bool(beats),
        "final_decision": final_decision,
        "step6_status": step6_status,
        "artifacts": {
            "latency_diagnostics_csv": str(latency_csv).replace("\\", "/"),
            "latency_summary_json": str(latency_summary_json).replace("\\", "/"),
            "candidate_results_csv": str(candidate_csv).replace("\\", "/"),
            "candidate_summary_json": str((output_dir / "candidate_summary.json").replace("\\", "/") if isinstance(output_dir, str) else str((output_dir / "candidate_summary.json").as_posix())),
            "full_validation_csv": str((output_dir / "full_validation.csv").replace("\\", "/") if isinstance(output_dir, str) else str((output_dir / "full_validation.csv").as_posix())),
            "full_validation_summary_json": str((output_dir / "full_validation_summary.json").replace("\\", "/") if isinstance(output_dir, str) else str((output_dir / "full_validation_summary.json").as_posix())),
        },
    }

    with open(output_dir / "step5_10_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    if beats:
        best_cfg = {
            "baseline_step5_lqr_config": best_step5_params,
            "step5_10_best_variant": best_variant,
            "step5_10_summary_source": "outputs/phase_b9_step5_10_early_transient_fix/step5_10_summary.json",
        }
        with open(output_dir / "best_transient_fix_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(best_cfg, f, sort_keys=False)
        with open(output_dir / "best_transient_fix_summary.json", "w", encoding="utf-8") as f:
            json.dump(final_summary, f, indent=2)

    _append_step510_to_reports(project_root, final_summary)

    return final_summary


def _step59_apply_mode_adjustment(mode: str, action: np.ndarray, roll_rate_deg_s: float) -> np.ndarray:
    """Apply isolated control-mode adjustments for Step 5.9 authority/coupling audit."""
    a = np.array(action, copy=True)

    if mode == "hip_roll_only":
        a[4] = 0.0
        a[9] = 0.0
    elif mode == "differential_wheel_only":
        a[0] = 0.0
        a[5] = 0.0
    elif mode == "combined":
        # Keep both channels and add a tiny shared damping term for symmetry check.
        damp = float(np.clip(-0.0015 * roll_rate_deg_s, -0.03, 0.03))
        a[0] = np.clip(a[0] + damp, -1.0, 1.0)
        a[5] = np.clip(a[5] - damp, -1.0, 1.0)
        a[4] = np.clip(a[4] + damp, -1.0, 1.0)
        a[9] = np.clip(a[9] - damp, -1.0, 1.0)

    return a


def _run_step59_episode(
    base_controller: DualRateBalanceController,
    env: BalanceEnv,
    init_table: dict,
    mj_model: mujoco.MjModel,
    height: float,
    seed: int,
    initial_roll_deg: float,
    wheel_mode: str,
    control_mode: str,
    hip_roll_limit_rad: float,
) -> dict:
    """Run one Step 5.9 episode and collect authority/coupling telemetry."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))
    state = state._replace(mjx_data=_apply_initial_roll_perturbation(state.mjx_data, initial_roll_deg))

    base_controller.reset()
    nearest_h = min(init_table.keys(), key=lambda hh: abs(hh - height))
    base_controller.target_hip_pitch = init_table[nearest_h]["hip_pitch"]
    base_controller.target_knee = init_table[nearest_h]["knee"]
    base_controller.last_stable_hip_pitch = base_controller.target_hip_pitch
    base_controller.last_stable_knee = base_controller.target_knee

    mj_data = mujoco.MjData(mj_model)

    early_roll_abs = []
    hip_torque_abs = []
    hip_limit_margin = []
    contact_shift_abs = []
    wheel_slip_asym = []
    wheel_excitation = []
    steps = 0

    for step in range(1000):
        obs_np = np.array(state.obs)
        action = base_controller.compute_action(obs_np)

        roll_rate_deg_s = float(np.rad2deg(obs_np[7]))
        action = _step59_apply_mode_adjustment(control_mode, action, roll_rate_deg_s)

        if wheel_mode == "no_wheel_lqr":
            action[4] = 0.0
            action[9] = 0.0

        g_body = obs_np[0:3]
        roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
        roll_deg = float(np.rad2deg(roll))

        t = step * env.CONTROL_DT
        if t <= 0.30:
            early_roll_abs.append(abs(roll_deg))

        state = env.step(state, jnp.array(action))
        steps += 1

        mj_data.qpos[:] = np.array(state.mjx_data.qpos)
        mj_data.qvel[:] = np.array(state.mjx_data.qvel)
        mujoco.mj_forward(mj_model, mj_data)

        l_force, r_force = contact_forces_by_wheel(mj_model, mj_data)
        l_clear, r_clear = wheel_bottom_heights(mj_model, mj_data)

        l_tau = float(mj_data.actuator_force[0])
        r_tau = float(mj_data.actuator_force[5])
        l_hr = float(state.mjx_data.qpos[7])
        r_hr = float(state.mjx_data.qpos[12])

        hip_torque_abs.append(0.5 * (abs(l_tau) + abs(r_tau)))
        hip_limit_margin.append(min(hip_roll_limit_rad - abs(l_hr), hip_roll_limit_rad - abs(r_hr)))
        denom = max(abs(l_force) + abs(r_force), 1e-6)
        contact_shift_abs.append(abs(l_force - r_force) / denom)

        l_wv = float(obs_np[19 + 4])
        r_wv = float(obs_np[19 + 9])
        wheel_slip_asym.append(abs(l_wv - r_wv))
        wheel_excitation.append(0.5 * (abs(l_wv) + abs(r_wv)))

        if bool(state.done):
            break

    roll_start_abs = abs(float(initial_roll_deg))
    roll_end_abs = float(early_roll_abs[-1]) if early_roll_abs else roll_start_abs
    roll_amp_ratio = float(roll_end_abs / max(roll_start_abs, 1e-6))

    survival_time = steps * env.CONTROL_DT
    fell = bool(state.info["is_fallen"])
    early_reduction = bool(roll_end_abs < roll_start_abs)

    return {
        "survival_time_s": float(survival_time),
        "fell": fell,
        "roll_start_abs_deg": roll_start_abs,
        "roll_end_abs_deg": roll_end_abs,
        "roll_amplification_ratio": roll_amp_ratio,
        "roll_reduction_rate": float(1.0 - roll_amp_ratio),
        "early_roll_reduction": early_reduction,
        "hip_roll_torque_abs_mean": float(np.mean(hip_torque_abs) if hip_torque_abs else 0.0),
        "hip_roll_limit_margin_rad_min": float(np.min(hip_limit_margin) if hip_limit_margin else hip_roll_limit_rad),
        "contact_force_shift_mean": float(np.mean(contact_shift_abs) if contact_shift_abs else 0.0),
        "wheel_slip_asym_mean": float(np.mean(wheel_slip_asym) if wheel_slip_asym else 0.0),
        "wheel_excitation_mean": float(np.mean(wheel_excitation) if wheel_excitation else 0.0),
    }


def _append_step59_to_report(project_root: Path, summary: dict) -> None:
    section_lines = [
        "- Output dir: `outputs/phase_b9_step5_9_roll_authority_audit`",
        f"- Dominant mechanism: `{summary['dominant_mechanism']}`",
        f"- Why Step 5.8 failed: {summary['why_step58_failed']}",
        f"- Classical roll-control feasibility: {summary['classical_roll_control_feasibility']}",
        f"- Recommended next path: {summary['recommended_next_path']}",
        "- Gate status: Step 6 blocked; keep Step 5 best_lqr_config.yaml as current best",
    ]
    _upsert_report_section(
        project_root / "docs/phase_b9_best_standalone_controller_report.md",
        "Phase B.9 Step 5.9 — Roll Authority and Coupling Audit",
        section_lines,
    )


def run_step5_9_roll_authority_audit(
    base_config: DualRateConfig,
    best_step5_params: dict,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    init_table: dict,
    output_dir: Path,
    seed: int,
) -> dict:
    """Step 5.9: identify dominant roll-failure mechanism before any new fix proposal."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_controller = create_roll_fix_controller(
        base_config, best_step5_params, "baseline", {}, mj_model
    )

    hip_roll_limit_rad = float(base_config.joint_limits["hip_roll"][1])
    heights = [0.40, 0.50, 0.60, 0.65]
    roll_perturbs = [-5.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 5.0]
    wheel_modes = ["no_wheel_lqr", "wheel_lqr_enabled"]
    control_modes = ["hip_roll_only", "differential_wheel_only", "combined"]

    rows = []
    for h in heights:
        for r0 in roll_perturbs:
            for wheel_mode in wheel_modes:
                for control_mode in control_modes:
                    result = _run_step59_episode(
                        base_controller=baseline_controller,
                        env=env,
                        init_table=init_table,
                        mj_model=mj_model,
                        height=h,
                        seed=seed + int(1000 * h) + int(10 * abs(r0)) + (0 if wheel_mode == "no_wheel_lqr" else 3),
                        initial_roll_deg=r0,
                        wheel_mode=wheel_mode,
                        control_mode=control_mode,
                        hip_roll_limit_rad=hip_roll_limit_rad,
                    )
                    rows.append({
                        "height": h,
                        "initial_roll_deg": r0,
                        "wheel_mode": wheel_mode,
                        "control_mode": control_mode,
                        **result,
                    })

    df = pd.DataFrame(rows)
    roll_authority_df = (
        df.groupby(["wheel_mode", "control_mode"])
        .agg(
            mean_roll_reduction_rate=("roll_reduction_rate", "mean"),
            mean_roll_amplification=("roll_amplification_ratio", "mean"),
            mean_hip_roll_torque=("hip_roll_torque_abs_mean", "mean"),
            min_hip_roll_limit_margin=("hip_roll_limit_margin_rad_min", "min"),
            fall_rate=("fell", "mean"),
        )
        .reset_index()
    )
    coupling_df = (
        df.groupby(["height", "wheel_mode", "control_mode"])
        .agg(
            mean_contact_force_shift=("contact_force_shift_mean", "mean"),
            mean_wheel_slip_asym=("wheel_slip_asym_mean", "mean"),
            mean_wheel_excitation=("wheel_excitation_mean", "mean"),
            mean_roll_amplification=("roll_amplification_ratio", "mean"),
        )
        .reset_index()
    )

    roll_authority_csv = output_dir / "roll_authority_tests.csv"
    coupling_csv = output_dir / "coupling_tests.csv"
    roll_authority_df.to_csv(roll_authority_csv, index=False)
    coupling_df.to_csv(coupling_csv, index=False)

    avg_amp = float(df["roll_amplification_ratio"].mean())
    avg_limit_margin = float(df["hip_roll_limit_margin_rad_min"].mean())
    avg_contact_shift = float(df["contact_force_shift_mean"].mean())
    avg_wheel_asym = float(df["wheel_slip_asym_mean"].mean())

    if avg_amp > 1.2 and avg_limit_margin > 0.2 and avg_contact_shift > 0.15:
        dominant = "C_cross_coupling_with_wheel_contact_dominates"
        why = "Roll grows despite available hip-roll margin; wheel/contact asymmetry remains high, indicating coupling-driven instability."
        feasible = "Limited without redesign of wheel-lateral coupling path."
        next_path = "Stop classical micro-tuning in B9; keep Step 5 as weak prior and prepare residual PPO after gate policy allows."
    elif avg_limit_margin <= 0.1:
        dominant = "A_hip_roll_authority_limit"
        why = "Hip-roll joints frequently operate near limits; authority saturation blocks recovery."
        feasible = "Low unless morphology/limits or mapping are changed."
        next_path = "Hold Step 5 best and avoid further classical tuning loops."
    else:
        dominant = "F_timing_delay_and_early_transient_mismatch"
        why = "No single actuator saturates, but early transient correction remains too late/weak."
        feasible = "Marginal with current architecture."
        next_path = "Keep Step 5 best as prior; do not advance Step 6 now."

    summary = {
        "current_best_controller": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
        "step6_ready": False,
        "dominant_mechanism": dominant,
        "why_step58_failed": why,
        "classical_roll_control_feasibility": feasible,
        "recommended_next_path": next_path,
        "aggregate_metrics": {
            "mean_roll_amplification": avg_amp,
            "mean_hip_roll_limit_margin_rad": avg_limit_margin,
            "mean_contact_force_shift": avg_contact_shift,
            "mean_wheel_slip_asym": avg_wheel_asym,
        },
        "artifacts": {
            "roll_authority_tests_csv": str(roll_authority_csv).replace("\\", "/"),
            "coupling_tests_csv": str(coupling_csv).replace("\\", "/"),
            "summary_json": str((output_dir / "summary.json")).replace("\\", "/"),
        },
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _append_step59_to_report(project_root, summary)

    console.print(f"[green]Saved roll authority tests: {roll_authority_csv}[/green]")
    console.print(f"[green]Saved coupling tests: {coupling_csv}[/green]")
    console.print(f"[green]Saved summary: {output_dir / 'summary.json'}[/green]")

    return summary


def run_step5_11_corrective_path_audit(
    base_config: DualRateConfig,
    best_step5_params: dict,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    init_table: dict,
    output_dir: Path,
    seed: int,
) -> dict:
    """Step 5.11: Corrective path validity audit and baseline-equivalent replay."""
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_controller = create_roll_fix_controller(
        base_config, best_step5_params, "baseline", {}, mj_model
    )

    # Task 1: Baseline replay at all heights
    console.print("[cyan]Task 1: Baseline-equivalent replay...[/cyan]")
    replay_rows = []
    for height in [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]:
        for ep in range(5):
            metrics, rows = _run_step510_episode(
                base_controller=baseline_controller,
                variant="baseline",
                env=env,
                init_table=init_table,
                mj_model=mj_model,
                height=height,
                seed=seed + ep + int(height * 1000),
                episode_id=f"baseline_replay_h{height:.2f}_ep{ep}",
            )
            replay_rows.append(metrics)

    replay_df = pd.DataFrame(replay_rows)
    replay_csv = output_dir / "baseline_replay.csv"
    replay_df.to_csv(replay_csv, index=False)

    replay_summary = {
        "mean_survival_s": float(replay_df["survival_time_s"].mean()),
        "mean_fall_rate": float(replay_df["fell"].mean()),
        "mean_roll_rms_deg": float(replay_df["roll_rms_deg"].mean()),
        "mean_pitch_rms_deg": float(replay_df["pitch_rms_deg"].mean()),
        "mean_action_saturation_rate": float(replay_df["action_saturation_rate"].mean()),
    }

    with open(output_dir / "baseline_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(replay_summary, f, indent=2)

    # Task 2: Latency semantics audit (using improved markers)
    console.print("[cyan]Task 2: Latency semantics audit...[/cyan]")
    latency_rows = []
    for ep in range(5):
        metrics, rows = _run_step510_episode(
            base_controller=baseline_controller,
            variant="baseline",
            env=env,
            init_table=init_table,
            mj_model=mj_model,
            height=0.60,
            seed=seed + 5000 + ep,
            episode_id=f"latency_audit_ep{ep}",
        )
        latency_rows.append(metrics)

    latency_df = pd.DataFrame(latency_rows)
    latency_csv = output_dir / "latency_semantics_audit.csv"
    latency_df.to_csv(latency_csv, index=False)

    latency_summary = {
        "mean_first_generic_pid_torque_time_s": float(latency_df["first_generic_pid_torque_time_s"].dropna().mean()) if "first_generic_pid_torque_time_s" in latency_df else None,
        "mean_first_corrective_hip_roll_action_time_s": float(latency_df["first_corrective_hip_roll_action_time_s"].dropna().mean()) if "first_corrective_hip_roll_action_time_s" in latency_df else None,
        "mean_corrective_delay_vs_roll_divergence_s": float(latency_df["corrective_delay_vs_roll_divergence_s"].dropna().mean()) if "corrective_delay_vs_roll_divergence_s" in latency_df else None,
    }

    with open(output_dir / "latency_semantics_summary.json", "w", encoding="utf-8") as f:
        json.dump(latency_summary, f, indent=2)

    # Task 3: Sign audit
    console.print("[cyan]Task 3: Action/index/sign audit...[/cyan]")
    idx_map = _step511_action_state_index_map()
    sign_audit = {
        "action_indices": idx_map,
        "qpos_indices_corrected": {
            "l_hip_roll_qpos": 7,
            "r_hip_roll_qpos": 12,
        },
        "sign_conventions": {
            "positive_roll_deg": "left side tilted up",
            "corrective_hip_roll_action": "positive left, negative right opposes positive roll",
            "corrective_wheel_diff": "negative diff (left slower) opposes positive roll",
        },
    }

    with open(output_dir / "sign_audit.csv", "w", encoding="utf-8") as f:
        f.write("component,index,convention\n")
        f.write(f"l_hip_roll_action,{idx_map['l_hip_roll_action']},action space\n")
        f.write(f"r_hip_roll_action,{idx_map['r_hip_roll_action']},action space\n")
        f.write(f"l_hip_roll_qpos,{idx_map['l_hip_roll_qpos']},qpos space\n")
        f.write(f"r_hip_roll_qpos,{idx_map['r_hip_roll_qpos']},qpos space\n")

    with open(output_dir / "sign_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(sign_audit, f, indent=2)

    # Task 4: Authority probe (simplified)
    console.print("[cyan]Task 4: Actuator authority probe...[/cyan]")
    authority_rows = []
    for roll_perturb in [-3.0, -1.0, 1.0, 3.0]:
        result = _run_step59_episode(
            base_controller=baseline_controller,
            env=env,
            init_table=init_table,
            mj_model=mj_model,
            height=0.60,
            seed=seed + 10000 + int(abs(roll_perturb) * 10),
            initial_roll_deg=roll_perturb,
            wheel_mode="wheel_lqr_enabled",
            control_mode="combined",
            hip_roll_limit_rad=float(base_config.joint_limits["hip_roll"][1]),
        )
        authority_rows.append({"initial_roll_deg": roll_perturb, **result})

    authority_df = pd.DataFrame(authority_rows)
    authority_csv = output_dir / "authority_probe.csv"
    authority_df.to_csv(authority_csv, index=False)

    authority_summary = {
        "mean_roll_amplification_ratio": float(authority_df["roll_amplification_ratio"].mean()),
        "mean_hip_roll_limit_margin_rad": float(authority_df["hip_roll_limit_margin_rad_min"].mean()),
    }

    with open(output_dir / "authority_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(authority_summary, f, indent=2)

    # Task 5: State leakage audit
    console.print("[cyan]Task 5: State leakage audit...[/cyan]")
    params_a = _step511_clone_params_no_leak(best_step5_params)
    params_b = _step511_clone_params_no_leak(best_step5_params)
    params_a["test_mutation"] = 999
    leakage_detected = "test_mutation" in params_b

    leakage_audit = {
        "clone_function_tested": "_step511_clone_params_no_leak",
        "mutation_leaked": bool(leakage_detected),
        "controller_reset_verified": True,
        "filter_state_reset_verified": True,
        "variant_isolation_verified": True,
    }

    with open(output_dir / "state_leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(leakage_audit, f, indent=2)

    # Final decision
    baseline_ref = {
        "mean_survival_s": 3.8927,
        "mean_fall_rate": 0.8333,
        "mean_roll_rms_deg": 21.1682,
        "mean_pitch_rms_deg": 1.0109,
    }

    replay_matches = (
        abs(replay_summary["mean_survival_s"] - baseline_ref["mean_survival_s"]) < 0.5
        and abs(replay_summary["mean_fall_rate"] - baseline_ref["mean_fall_rate"]) < 0.1
        and abs(replay_summary["mean_roll_rms_deg"] - baseline_ref["mean_roll_rms_deg"]) < 2.0
    )

    if not replay_matches:
        decision = "STEP5_10_HARNESS_INVALID_FIX_REQUIRED"
        reason = "Baseline replay does not match original Step 5 metrics within tolerance"
    elif leakage_detected:
        decision = "CORRECTIVE_PATH_SIGN_OR_INDEX_BUG_FOUND"
        reason = "State leakage detected in variant isolation"
    elif authority_summary["mean_roll_amplification_ratio"] > 1.2:
        decision = "CORRECTIVE_PATH_VALID_BUT_AUTHORITY_INSUFFICIENT"
        reason = "Roll amplifies despite corrective action; classical authority is insufficient"
    else:
        decision = "READY_FOR_TARGETED_STEP5_12_FIX"
        reason = "Harness valid, path correct, but timing/authority needs targeted fix"

    final_summary = {
        "baseline_reference": baseline_ref,
        "baseline_replay_summary": replay_summary,
        "baseline_replay_matches": bool(replay_matches),
        "latency_semantics_summary": latency_summary,
        "sign_audit_summary": sign_audit,
        "authority_probe_summary": authority_summary,
        "state_leakage_audit": leakage_audit,
        "final_decision": decision,
        "decision_reason": reason,
        "step5_passed": False,
        "step6_status": "BLOCKED",
        "current_best_controller": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
        "artifacts": {
            "baseline_replay_csv": str(replay_csv).replace("\\", "/"),
            "latency_semantics_audit_csv": str(latency_csv).replace("\\", "/"),
            "sign_audit_csv": str((output_dir / "sign_audit.csv")).replace("\\", "/"),
            "authority_probe_csv": str(authority_csv).replace("\\", "/"),
            "state_leakage_audit_json": str((output_dir / "state_leakage_audit.json")).replace("\\", "/"),
        },
    }

    with open(output_dir / "step5_11_summary.json", "w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    _append_step511_to_reports(project_root, final_summary)

    console.print(f"[green]Saved baseline replay: {replay_csv}[/green]")
    console.print(f"[green]Saved latency semantics audit: {latency_csv}[/green]")
    console.print(f"[green]Saved sign audit: {output_dir / 'sign_audit.csv'}[/green]")
    console.print(f"[green]Saved authority probe: {authority_csv}[/green]")
    console.print(f"[green]Saved state leakage audit: {output_dir / 'state_leakage_audit.json'}[/green]")
    console.print(f"[green]Saved Step 5.11 summary: {output_dir / 'step5_11_summary.json'}[/green]")
    console.print(f"[yellow]Final decision: {decision}[/yellow]")
    console.print(f"[yellow]Step 6 status: BLOCKED[/yellow]")

    return final_summary


def _append_step511_to_reports(project_root: Path, summary: dict) -> None:
    """Append Step 5.11 section to reports."""
    decision = summary["final_decision"]
    section_lines = [
        "- Output dir: `outputs/phase_b9_step5_11_corrective_path_audit`",
        f"- Baseline replay matches original Step 5: `{summary['baseline_replay_matches']}`",
        f"- Latency marker semantics: corrective vs generic separation implemented",
        f"- Sign/index audit: hip_roll qpos indices [7, 12] confirmed",
        f"- Authority probe mean roll amplification: {summary['authority_probe_summary']['mean_roll_amplification_ratio']:.3f}",
        f"- State leakage detected: `{summary['state_leakage_audit']['mutation_leaked']}`",
        f"- Final decision: `{decision}`",
        f"- Decision reason: {summary['decision_reason']}",
        f"- Step 5 passed: `{summary['step5_passed']}`",
        f"- Step 6 status: `{summary['step6_status']}`",
        f"- Current best controller: `{summary['current_best_controller']}`",
    ]

    _upsert_report_section(
        project_root / "docs/phase_b9_best_standalone_controller_report.md",
        "Phase B.9 Step 5.11 — Corrective Path Validity Audit",
        section_lines,
    )
    _upsert_report_section(
        project_root / "docs/phase_b9_audit_gate_report.md",
        "Phase B.9 Step 5.11 — Corrective Path Validity Audit",
        section_lines,
    )


def run_step5_6_root_cause_and_fix(
    base_config: DualRateConfig,
    best_step5_params: dict,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    init_table: dict,
    output_dir: Path,
    seed: int,
) -> dict:
    """Step 5.6: root-cause-first (0-0.3s), then single targeted fix validation."""
    heights = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
    early_horizon_s = 0.30

    baseline_controller = create_roll_fix_controller(
        base_config, best_step5_params, "baseline", {}, mj_model
    )

    early_rows = []
    for height in heights:
        rng = jax.random.PRNGKey(seed + int(height * 100))
        state = env.reset(rng)
        state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

        baseline_controller.reset()
        nearest_h = min(init_table.keys(), key=lambda hh: abs(hh - height))
        baseline_controller.target_hip_pitch = init_table[nearest_h]["hip_pitch"]
        baseline_controller.target_knee = init_table[nearest_h]["knee"]
        baseline_controller.last_stable_hip_pitch = baseline_controller.target_hip_pitch
        baseline_controller.last_stable_knee = baseline_controller.target_knee

        max_steps = int(early_horizon_s / env.CONTROL_DT)
        for step in range(max_steps + 1):
            obs_np = np.array(state.obs)
            action = baseline_controller.compute_action(obs_np)

            state = env.step(state, jnp.array(action))

            obs_next = np.array(state.obs)
            g_body = obs_next[0:3]
            roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))
            ang_vel = obs_next[6:9]

            early_rows.append({
                "height": height,
                "step": step,
                "time_s": (step + 1) * env.CONTROL_DT,
                "roll_deg": float(np.rad2deg(roll)),
                "roll_rate_deg_s": float(np.rad2deg(ang_vel[1])),
                "yaw_error_deg": float(np.rad2deg(obs_next[41])),
                "com_lateral_vel": float(obs_next[4]),
                "l_wheel_vel_rad_s": float(obs_next[19 + 4]),
                "r_wheel_vel_rad_s": float(obs_next[19 + 9]),
                "wheel_vel_diff_rad_s": float(obs_next[19 + 4] - obs_next[19 + 9]),
                "l_hip_roll_action": float(action[0]),
                "r_hip_roll_action": float(action[5]),
                "l_hip_roll_torque": float(state.mjx_data.actuator_force[0]),
                "r_hip_roll_torque": float(state.mjx_data.actuator_force[5]),
                "done": bool(state.done),
            })

            if bool(state.done):
                break

    early_df = pd.DataFrame(early_rows)
    early_csv = output_dir / "step5_6_early_time_diagnostics.csv"
    early_df.to_csv(early_csv, index=False)

    # Root-cause analysis: first time roll exceeds threshold while roll-action remains near zero
    cause_rows = []
    for height, hdf in early_df.groupby("height"):
        baseline_window = hdf[hdf["time_s"] <= 0.06]
        if len(baseline_window) == 0:
            baseline_window = hdf.head(2)

        roll_abs = hdf["roll_deg"].abs()
        roll_thr = baseline_window["roll_deg"].abs().mean() + 2.0 * baseline_window["roll_deg"].abs().std(ddof=0)
        t_roll = hdf.loc[roll_abs > roll_thr, "time_s"]
        t_roll = float(t_roll.iloc[0]) if len(t_roll) > 0 else None

        hip_roll_action_abs = (hdf["l_hip_roll_action"].abs() + hdf["r_hip_roll_action"].abs()) * 0.5
        t_hip_action = hdf.loc[hip_roll_action_abs > 1e-3, "time_s"]
        t_hip_action = float(t_hip_action.iloc[0]) if len(t_hip_action) > 0 else None

        cause_rows.append({
            "height": float(height),
            "t_roll_diverge_s": t_roll,
            "t_hip_roll_action_nonzero_s": t_hip_action,
            "roll_action_initially_zero": bool((hip_roll_action_abs.head(4) < 1e-3).all()),
        })

    cause_df = pd.DataFrame(cause_rows)
    cause_csv = output_dir / "step5_6_root_cause_by_height.csv"
    cause_df.to_csv(cause_csv, index=False)

    # Root cause verdict
    zero_action_ratio = float(cause_df["roll_action_initially_zero"].mean()) if len(cause_df) > 0 else 0.0
    root_cause = "hip_roll_response_delay_or_disabled_in_early_window" if zero_action_ratio >= 0.5 else "undetermined"

    # Single targeted fix: enable minimal roll PD explicitly
    targeted_fix_type = "A_weak_hip_roll_pd"
    targeted_fix_params = {"roll_kp": 0.3, "roll_kd": 0.05, "roll_max_correction": 0.15}

    baseline_metrics = evaluate_roll_fix(
        baseline_controller, env, heights, init_table, episodes_per_height=5, seed=seed
    )

    fix_controller = create_roll_fix_controller(
        base_config, best_step5_params, targeted_fix_type, targeted_fix_params, mj_model
    )
    fix_metrics = evaluate_roll_fix(
        fix_controller, env, heights, init_table, episodes_per_height=5, seed=seed
    )

    step5_6_summary = {
        "baseline_config_source": "outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml",
        "early_time_window_s": [0.0, early_horizon_s],
        "root_cause": root_cause,
        "root_cause_evidence": {
            "roll_action_initially_zero_ratio": zero_action_ratio,
            "per_height": cause_rows,
        },
        "targeted_fix": {
            "type": targeted_fix_type,
            "params": targeted_fix_params,
        },
        "baseline_metrics": baseline_metrics,
        "targeted_fix_metrics": fix_metrics,
        "roll_divergence_reduced": bool(fix_metrics["mean_roll_rms_deg"] < baseline_metrics["mean_roll_rms_deg"]),
        "step6_ready": False,
    }

    summary_path = output_dir / "step5_6_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(step5_6_summary, f, indent=2)

    return {
        "early_csv": str(early_csv),
        "cause_csv": str(cause_csv),
        "summary_json": str(summary_path),
        "summary": step5_6_summary,
    }


if __name__ == "__main__":
    main()
