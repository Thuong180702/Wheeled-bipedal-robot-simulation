"""Phase B.9 Step 5: Strengthen fast wheel LQR gains safely.

Staged tuning to improve survival time by strengthening LQR gains
while monitoring for saturation, oscillation, and instability.

Stage A: Coarse sweep at h=0.60 (3 episodes)
Stage B: Top-5 from A, evaluate at h=[0.65, 0.60, 0.55] (5 episodes)
Stage C: Top-2 from B, full height sweep (5 episodes)

Keeps slow loop disabled (Step 4 baseline).
"""

import argparse
import json
import sys
from itertools import product
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

console = Console()


# Stage parameter grids
STAGE_PARAM_GRIDS = {
    "A": {
        "lqr_gain_scale": [1.0, 2.0, 3.0, 5.0],
        "pitch_gain_mult": [1.0, 2.0, 3.0, 5.0],
        "wheel_cmd_limit_mult": [1.0, 1.5, 2.0, 3.0],
    },
    "B": {
        "pitch_rate_gain_mult": [1.0, 2.0, 3.0],
        "com_gain_mult": [1.0, 2.0, 3.0],
        "filter_alpha": [0.0, 0.3, 0.5],
    },
    "C": {
        "com_rate_gain_mult": [1.0, 2.0, 3.0],
        "filter_max_delta_mult": [0.8, 1.0, 1.2],
    },
}


def load_balanced_init_table() -> dict:
    """Load balanced root initialization table."""
    table_path = project_root / "configs/controllers/b9_balanced_root_init_table.yaml"
    with open(table_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    heights = data["balanced_root_initialization"]["heights"]
    return {float(k): v for k, v in heights.items()}


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_euler2Quat(quat, np.array([roll, pitch, yaw]), b"xyz")
    return quat


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
                interpolated_keys = ["root_x", "root_z", "root_roll", "root_pitch", "hip_pitch", "knee"]
                init = {
                    key: (1 - alpha) * init0[key] + alpha * init1[key]
                    for key in interpolated_keys
                }
                if "root_y" in init0 and "root_y" in init1:
                    init["root_y"] = (1 - alpha) * init0["root_y"] + alpha * init1["root_y"]
                break

    hip_pitch = init["hip_pitch"]
    knee = init["knee"]
    joint_targets = jnp.array([
        0.0, 0.0, hip_pitch, knee, 0.0,
        0.0, 0.0, hip_pitch, knee, 0.0,
    ], dtype=mjx_data.qpos.dtype)
    quat = jnp.array(
        rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0),
        dtype=mjx_data.qpos.dtype,
    )

    new_qpos = mjx_data.qpos
    new_qpos = new_qpos.at[0].set(init["root_x"])
    if "root_y" in init:
        new_qpos = new_qpos.at[1].set(init["root_y"])
    new_qpos = new_qpos.at[2].set(init["root_z"])
    new_qpos = new_qpos.at[3:7].set(quat)
    new_qpos = new_qpos.at[7:17].set(joint_targets)
    new_qvel = jnp.zeros_like(mjx_data.qvel)
    return mjx_data.replace(qpos=new_qpos, qvel=new_qvel)


def create_tuned_controller(
    base_config: DualRateConfig,
    params: dict,
    mj_model: mujoco.MjModel,
) -> DualRateBalanceController:
    """Create controller with tuned LQR gains."""
    lqr_scale = params.get("lqr_gain_scale", 1.0)
    pitch_mult = params.get("pitch_gain_mult", 1.0)
    pitch_rate_mult = params.get("pitch_rate_gain_mult", 1.0)
    com_mult = params.get("com_gain_mult", 1.0)
    com_rate_mult = params.get("com_rate_gain_mult", 1.0)
    wheel_limit_mult = params.get("wheel_cmd_limit_mult", 1.0)
    filter_alpha = params.get("filter_alpha", base_config.wheel_cmd_filter_alpha)
    filter_max_delta_mult = params.get("filter_max_delta_mult", 1.0)

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
            "kp": base_config.roll_kp,
            "kd": base_config.roll_kd,
            "max_correction": base_config.roll_max_correction,
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

    temp_path = project_root / "configs/controllers/temp_step5_tuning.yaml"
    with open(temp_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    tuned_config = DualRateConfig.from_yaml(temp_path)
    controller = DualRateBalanceController(tuned_config, mj_model)
    controller.slow_loop_interval = 999999

    temp_path.unlink()
    return controller


def evaluate_config(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    heights: list[float],
    init_table: dict,
    episodes_per_height: int,
    seed: int,
) -> dict:
    """Evaluate controller across heights."""
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
            wheel_cmd_sq = 0.0
            wheel_speed_sq = 0.0
            com_error_sq = 0.0
            action_rate_sq = 0.0
            action_sat_count = 0
            steps = 0
            prev_action = np.zeros(10)

            for _ in range(1000):
                obs_np = np.array(state.obs)
                action = controller.compute_action(obs_np)
                telem = controller.get_telemetry()

                g_body = obs_np[0:3]
                pitch = float(np.arcsin(np.clip(-g_body[0], -1.0, 1.0)))
                roll = float(np.arcsin(np.clip(g_body[1], -1.0, 1.0)))

                pitch_sq += pitch ** 2
                roll_sq += roll ** 2
                wheel_cmd_sq += telem["filtered_wheel_cmd"] ** 2

                joint_vel = obs_np[19:29]
                wheel_speed = (abs(joint_vel[4]) + abs(joint_vel[9])) / 2.0
                wheel_speed_sq += wheel_speed ** 2

                # CoM error (forward velocity proxy, target = 0)
                com_y_dot = obs_np[3]
                com_error_sq += com_y_dot ** 2

                # Action rate
                if steps > 0:
                    action_diff = action - prev_action
                    action_rate_sq += np.sum(action_diff ** 2)
                prev_action = action.copy()

                if np.max(np.abs(action)) >= 0.99:
                    action_sat_count += 1

                state = env.step(state, jnp.array(action))
                steps += 1

                if bool(state.done):
                    break

            survival_time = steps * env.CONTROL_DT
            fell = bool(state.info["is_fallen"])

            # Determine fall reason from final state
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
            wheel_cmd_rms = np.sqrt(wheel_cmd_sq / steps) if steps > 0 else 0.0
            wheel_speed_rms = np.sqrt(wheel_speed_sq / steps) if steps > 0 else 0.0
            com_error_rms = np.sqrt(com_error_sq / steps) if steps > 0 else 0.0
            action_rate_rms = np.sqrt(action_rate_sq / steps) if steps > 0 else 0.0
            action_sat_rate = action_sat_count / steps if steps > 0 else 0.0

            all_results.append({
                "height": height,
                "episode": ep,
                "survival_time_s": survival_time,
                "fell": fell,
                "fall_reason": fall_reason,
                "pitch_rms_deg": np.rad2deg(pitch_rms),
                "roll_rms_deg": np.rad2deg(roll_rms),
                "wheel_cmd_rms": wheel_cmd_rms,
                "wheel_speed_rms_rad_s": wheel_speed_rms,
                "com_error_rms": com_error_rms,
                "action_rate_rms": action_rate_rms,
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
        "mean_wheel_cmd_rms": float(df["wheel_cmd_rms"].mean()),
        "mean_wheel_speed_rms": float(df["wheel_speed_rms_rad_s"].mean()),
        "mean_com_error_rms": float(df["com_error_rms"].mean()),
        "mean_action_rate_rms": float(df["action_rate_rms"].mean()),
        "mean_action_sat_rate": float(df["action_saturation_rate"].mean()),
        "dominant_fall_reason": dominant_fall_reason,
        "fall_reason_counts_json": json.dumps(fall_reason_counts),
        "success": True,
    }


def get_stage_csv_path(output_dir: Path, stage: str) -> Path:
    """Get exact required CSV path for each stage."""
    stage_names = {"A": "stage_a_trials.csv", "B": "stage_b_top5.csv", "C": "stage_c_final.csv"}
    return output_dir / stage_names[stage]


def load_completed_ids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "config_id" not in df.columns:
        return set()
    return set(df["config_id"].astype(int).tolist())


def append_result_row(csv_path: Path, row: dict):
    row_df = pd.DataFrame([row])
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def load_top_k_parents(output_dir: Path, parent_stage: str, top_k: int) -> list[dict]:
    parent_csv = get_stage_csv_path(output_dir, parent_stage)
    if not parent_csv.exists():
        raise FileNotFoundError(f"Missing stage {parent_stage} results")

    df = pd.read_csv(parent_csv)
    if df.empty:
        raise ValueError(f"Stage {parent_stage} results empty")

    if "success" in df.columns:
        df = df[df["success"] == True]
    if df.empty:
        raise ValueError(f"Stage {parent_stage} has no successful configs")

    df_sorted = df.sort_values("mean_survival_s", ascending=False).head(top_k)
    parents = []
    param_keys = list(STAGE_PARAM_GRIDS["A"].keys()) + list(STAGE_PARAM_GRIDS["B"].keys()) + list(STAGE_PARAM_GRIDS["C"].keys())
    for _, row in df_sorted.iterrows():
        parent_params = {}
        for key in param_keys:
            if key in row and not pd.isna(row[key]):
                parent_params[key] = row[key]
        parents.append(parent_params)

    return parents


def run_stage(
    stage: str,
    base_config: DualRateConfig,
    mj_model: mujoco.MjModel,
    env: BalanceEnv,
    init_table: dict,
    heights: list[float],
    episodes_per_height: int,
    seed: int,
    output_dir: Path,
    top_k: int,
    resume: bool,
):
    """Run one tuning stage."""
    if stage == "A":
        parents = [{}]
    elif stage == "B":
        parents = load_top_k_parents(output_dir, "A", 5)
    else:
        parents = load_top_k_parents(output_dir, "B", 2)

    stage_csv = get_stage_csv_path(output_dir, stage)
    completed_ids = load_completed_ids(stage_csv) if resume else set()

    param_grid = STAGE_PARAM_GRIDS[stage]
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    stage_combos = list(product(*param_values))
    total_configs = len(stage_combos) * len(parents)

    console.print(f"\n[bold cyan]Stage {stage}: LQR Gain Strengthening[/bold cyan]")
    console.print(f"Parents: {len(parents)} | Combinations: {len(stage_combos)} | Total: {total_configs}")
    console.print(f"Heights: {heights} | Episodes/height: {episodes_per_height}\n")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Evaluating...", total=total_configs)

        config_id = 0
        for parent_idx, parent_params in enumerate(parents):
            for combo in stage_combos:
                if config_id in completed_ids:
                    progress.update(task, advance=1, description=f"Skipping {config_id}/{total_configs-1}")
                    config_id += 1
                    continue

                stage_params = dict(zip(param_names, combo))
                merged_params = {**parent_params, **stage_params}

                controller = create_tuned_controller(base_config, merged_params, mj_model)
                metrics = evaluate_config(controller, env, heights, init_table, episodes_per_height, seed)

                row = {"config_id": config_id, "parent_idx": parent_idx, "stage": stage, **merged_params, **metrics}
                append_result_row(stage_csv, row)

                progress.update(
                    task,
                    advance=1,
                    description=f"Stage {stage} {config_id}/{total_configs-1} Survival: {metrics['mean_survival_s']:.2f}s",
                )
                config_id += 1

    df = pd.read_csv(stage_csv)
    if "success" in df.columns:
        df = df[df["success"] == True]
    if df.empty:
        raise RuntimeError(f"No successful configs in stage {stage}")

    df_sorted = df.sort_values("mean_survival_s", ascending=False)

    table = Table(title=f"Top 10 - Stage {stage}")
    table.add_column("Rank", justify="right")
    table.add_column("ID", justify="right")
    table.add_column("Survival (s)", justify="right")
    table.add_column("Fall Rate", justify="right")
    table.add_column("Roll RMS (°)", justify="right")
    table.add_column("Sat Rate", justify="right")

    for rank, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
        table.add_row(
            str(rank),
            str(int(row["config_id"])),
            f"{row['mean_survival_s']:.2f}",
            f"{row['mean_fall_rate']:.1%}",
            f"{row['mean_roll_rms_deg']:.1f}",
            f"{row['mean_action_sat_rate']:.1%}",
        )
    console.print(table)

    best_row = df_sorted.iloc[0]
    best_params = {}
    for key in param_names:
        if key in best_row and not pd.isna(best_row[key]):
            best_params[key] = best_row[key]

    summary = {
        "stage": stage,
        "best_config_id": int(best_row["config_id"]),
        "best_metrics": {
            "mean_survival_s": float(best_row["mean_survival_s"]),
            "mean_fall_rate": float(best_row["mean_fall_rate"]),
            "mean_pitch_rms_deg": float(best_row["mean_pitch_rms_deg"]),
            "mean_roll_rms_deg": float(best_row["mean_roll_rms_deg"]),
            "mean_action_sat_rate": float(best_row["mean_action_sat_rate"]),
        },
        "best_params": best_params,
    }

    summary_path = output_dir / f"stage_{stage}_best_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    console.print(f"[green]Saved stage {stage} to {stage_csv}[/green]")
    console.print(f"[green]Saved summary to {summary_path}[/green]\n")


def main():
    parser = argparse.ArgumentParser(description="Phase B.9 Step 5: LQR gain strengthening")
    parser.add_argument("--stage", type=str, default="all", choices=["A", "B", "C", "all"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase_b9_lqr_gain_strengthening"))
    parser.add_argument("--top-k", type=int, default=5, help="Top-k promoted to next stage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Step 5: LQR Gain Strengthening[/bold cyan]\n")

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

    resume = not args.no_resume

    if args.stage == "all":
        stages = ["A", "B", "C"]
    else:
        stages = [args.stage]

    stage_heights = {
        "A": [0.60],
        "B": [0.65, 0.60, 0.55],
        "C": [0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
    }
    stage_episodes = {
        "A": 3,
        "B": 5,
        "C": 5,
    }

    for stage in stages:
        run_stage(
            stage=stage,
            base_config=base_config,
            mj_model=mj_model,
            env=env,
            init_table=init_table,
            heights=stage_heights[stage],
            episodes_per_height=stage_episodes[stage],
            seed=args.seed,
            output_dir=args.output_dir,
            top_k=args.top_k,
            resume=resume,
        )

    # Generate final best config
    if "C" in stages or args.stage == "all":
        stage_c_csv = get_stage_csv_path(args.output_dir, "C")
        df_c = pd.read_csv(stage_c_csv)
        if "success" in df_c.columns:
            df_c = df_c[df_c["success"] == True]
        best_row = df_c.sort_values("mean_survival_s", ascending=False).iloc[0]

        all_param_keys = list(STAGE_PARAM_GRIDS["A"].keys()) + list(STAGE_PARAM_GRIDS["B"].keys()) + list(STAGE_PARAM_GRIDS["C"].keys())
        best_config = {}
        for key in all_param_keys:
            if key in best_row and not pd.isna(best_row[key]):
                best_config[key] = float(best_row[key])

        best_config_path = args.output_dir / "best_lqr_config.yaml"
        with open(best_config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(best_config, f)

        best_summary = {
            "best_config_id": int(best_row["config_id"]),
            "best_params": best_config,
            "metrics": {
                "mean_survival_s": float(best_row["mean_survival_s"]),
                "mean_fall_rate": float(best_row["mean_fall_rate"]),
                "mean_pitch_rms_deg": float(best_row["mean_pitch_rms_deg"]),
                "mean_roll_rms_deg": float(best_row["mean_roll_rms_deg"]),
                "mean_wheel_cmd_rms": float(best_row["mean_wheel_cmd_rms"]),
                "mean_wheel_speed_rms": float(best_row["mean_wheel_speed_rms"]),
                "mean_action_sat_rate": float(best_row["mean_action_sat_rate"]),
            },
        }

        best_summary_path = args.output_dir / "best_lqr_summary.json"
        with open(best_summary_path, "w", encoding="utf-8") as f:
            json.dump(best_summary, f, indent=2)

        console.print(f"\n[bold green]Best LQR config saved to {best_config_path}[/bold green]")
        console.print(f"[bold green]Best summary saved to {best_summary_path}[/bold green]\n")


if __name__ == "__main__":
    main()
