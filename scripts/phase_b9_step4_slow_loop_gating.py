"""Phase B.9 Step 4: Slow-loop rate and gating evaluation.

Tests whether weak, gated slow loop improves roll/posture stability
without fighting fast wheel LQR.

Variants:
1. slow_loop_disabled (same as Step 3 fast-only)
2. slow_loop_5Hz_weak
3. slow_loop_10Hz_weak
4. slow_loop_20Hz_weak
5. slow_loop_50Hz_very_weak

Settings:
- Fast wheel LQR active at 50Hz
- Slow loop only adjusts hip_pitch/knee
- No aggressive VMC, no CoM-to-hip correction
- Roll/yaw disabled unless explicitly tested
- Stability gating: disable slow loop when roll/pitch exceeds threshold
"""

import argparse
import csv
import sys
from dataclasses import dataclass
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
from rich.table import Table

from wheeled_biped.controllers.action_codec import (
    L_HIP_PITCH,
    L_KNEE,
    R_HIP_PITCH,
    R_KNEE,
    clip_normalized_action,
)
from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()

VALID_HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]


@dataclass
class SlowLoopVariant:
    """Slow-loop variant configuration."""
    name: str
    slow_loop_rate_hz: float
    roll_gate_deg: float
    pitch_gate_deg: float
    slow_loop_scale_when_unstable: float


VARIANTS = [
    SlowLoopVariant("slow_loop_disabled", 0.0, 5.0, 5.0, 0.0),
    SlowLoopVariant("slow_loop_5Hz_weak", 5.0, 5.0, 5.0, 0.0),
    SlowLoopVariant("slow_loop_10Hz_weak", 10.0, 5.0, 5.0, 0.0),
    SlowLoopVariant("slow_loop_20Hz_weak", 20.0, 5.0, 5.0, 0.0),
    SlowLoopVariant("slow_loop_50Hz_very_weak", 50.0, 5.0, 5.0, 0.0),
]


def select_variants(variant_names: list[str] | None) -> list[SlowLoopVariant]:
    """Select subset of variants by name, or all variants when omitted."""
    if not variant_names:
        return VARIANTS

    variant_map = {v.name: v for v in VARIANTS}
    selected = []
    for name in variant_names:
        if name not in variant_map:
            raise ValueError(f"Unknown variant '{name}'. Available: {list(variant_map.keys())}")
        selected.append(variant_map[name])
    return selected


def load_balanced_init_table():
    """Load balanced root initialization table."""
    config_path = project_root / "configs" / "controllers" / "b9_balanced_root_init_table.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["balanced_root_initialization"]["heights"]


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll, pitch, yaw to quaternion."""
    quat = np.zeros(4)
    euler = np.array([roll, pitch, yaw])
    mujoco.mju_euler2Quat(quat, euler, b"xyz")
    return quat


def apply_balanced_root_init(mjx_data, height: float, init_table: dict):
    """Apply balanced root initialization to MJX data."""
    height_key = f"{height:.2f}"
    if height_key not in init_table:
        raise ValueError(f"Height {height} not in balanced init table")

    init = init_table[height_key]

    # Set root pose
    new_qpos = mjx_data.qpos
    new_qpos = new_qpos.at[0].set(init["root_x"])
    new_qpos = new_qpos.at[2].set(init["root_z"])
    quat = rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0)
    new_qpos = new_qpos.at[3:7].set(quat)

    # Set joint positions
    hip_pitch = init["hip_pitch"]
    knee = init["knee"]
    joint_targets = jnp.array([
        0.0, 0.0, hip_pitch, knee, 0.0,
        0.0, 0.0, hip_pitch, knee, 0.0,
    ])
    new_qpos = new_qpos.at[7:17].set(joint_targets)

    # Zero velocities
    new_qvel = jnp.zeros_like(mjx_data.qvel)

    return mjx_data.replace(qpos=new_qpos, qvel=new_qvel)


def freeze_controller_posture(controller: DualRateBalanceController, height: float, init_table: dict, disable_slow_loop: bool = True) -> None:
    """Freeze controller leg targets to balanced init values (fast-only mode)."""
    height_key = f"{height:.2f}"
    init = init_table[height_key]

    # Lock leg targets to balanced init posture
    controller.target_hip_pitch = float(init["hip_pitch"])
    controller.target_knee = float(init["knee"])
    controller.last_stable_hip_pitch = float(init["hip_pitch"])
    controller.last_stable_knee = float(init["knee"])

    # Disable slow loop by setting interval to very large value
    if disable_slow_loop:
        controller.slow_loop_interval = 999999


def apply_slow_loop_correction(raw_correction: dict, gate_active: bool, slow_loop_scale: float) -> dict:
    """Apply gating to slow-loop correction.

    Args:
        raw_correction: Dict with hip_pitch_delta, knee_delta
        gate_active: Whether stability gate is active (roll/pitch exceeded threshold)
        slow_loop_scale: Scale factor when gate is active (0.0 = fully blocked)

    Returns:
        Dict with gate_active, slow_loop_scale, raw_slow_correction, applied_slow_correction
    """
    if gate_active:
        scale = slow_loop_scale
    else:
        scale = 1.0

    applied = {
        "hip_pitch_delta": raw_correction["hip_pitch_delta"] * scale,
        "knee_delta": raw_correction["knee_delta"] * scale,
    }

    return {
        "gate_active": gate_active,
        "slow_loop_scale": scale,
        "raw_slow_correction": raw_correction,
        "applied_slow_correction": applied,
    }


def to_jsonable_or_yamlable(obj):
    """Convert NumPy types to Python scalars for clean YAML/JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_jsonable_or_yamlable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable_or_yamlable(item) for item in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def inject_leg_targets_into_action(controller: DualRateBalanceController, action: np.ndarray) -> np.ndarray:
    """Rebuild leg targets in action from controller targets."""
    action_out = np.array(action, dtype=np.float32, copy=True)

    hip_pitch_range = controller.config.joint_limits["hip_pitch"]
    knee_range = controller.config.joint_limits["knee"]

    hip_pitch_norm = (
        2.0 * (controller.target_hip_pitch - hip_pitch_range[0]) / (hip_pitch_range[1] - hip_pitch_range[0]) - 1.0
    )
    knee_norm = 2.0 * (controller.target_knee - knee_range[0]) / (knee_range[1] - knee_range[0]) - 1.0

    action_out[L_HIP_PITCH] = hip_pitch_norm
    action_out[L_KNEE] = knee_norm
    action_out[R_HIP_PITCH] = hip_pitch_norm
    action_out[R_KNEE] = knee_norm

    return clip_normalized_action(action_out)


def apply_step4_slow_loop_gate(
    controller: DualRateBalanceController,
    action: np.ndarray,
    variant: SlowLoopVariant,
    roll_deg: float,
    pitch_deg: float,
    prev_target_hip_pitch: float,
    prev_target_knee: float,
) -> tuple[np.ndarray, dict]:
    """Apply Step 4 roll/pitch gate to real slow-loop correction path."""
    gate_active = bool(roll_deg > variant.roll_gate_deg or pitch_deg > variant.pitch_gate_deg)

    should_update_slow = bool(controller.last_should_update_slow)
    raw_correction = {
        "hip_pitch_delta": float(controller.target_hip_pitch - prev_target_hip_pitch),
        "knee_delta": float(controller.target_knee - prev_target_knee),
    }

    if should_update_slow:
        correction_info = apply_slow_loop_correction(
            raw_correction=raw_correction,
            gate_active=gate_active,
            slow_loop_scale=float(variant.slow_loop_scale_when_unstable),
        )

        applied = correction_info["applied_slow_correction"]
        controller.target_hip_pitch = float(prev_target_hip_pitch + applied["hip_pitch_delta"])
        controller.target_knee = float(prev_target_knee + applied["knee_delta"])

        if gate_active:
            controller.last_stable_hip_pitch = float(prev_target_hip_pitch)
            controller.last_stable_knee = float(prev_target_knee)
        else:
            controller.last_stable_hip_pitch = float(controller.target_hip_pitch)
            controller.last_stable_knee = float(controller.target_knee)

        action = inject_leg_targets_into_action(controller, action)
    else:
        correction_info = apply_slow_loop_correction(
            raw_correction={"hip_pitch_delta": 0.0, "knee_delta": 0.0},
            gate_active=gate_active,
            slow_loop_scale=float(variant.slow_loop_scale_when_unstable),
        )

    correction_info["should_update_slow"] = should_update_slow
    return action, correction_info


def create_variant_controller(
    base_config: DualRateConfig,
    variant: SlowLoopVariant,
    mj_model: mujoco.MjModel
) -> DualRateBalanceController:
    """Create controller with variant-specific slow-loop settings."""
    # Create modified config
    config_dict = {
        "time_scale": {
            "fast_loop_rate_hz": base_config.fast_loop_rate_hz,
            "slow_loop_rate_hz": variant.slow_loop_rate_hz if variant.slow_loop_rate_hz > 0 else 0.1,
            "control_dt": base_config.control_dt,
        },
        "height": {
            "min": base_config.height_min,
            "max": base_config.height_max,
            "grid": base_config.height_grid,
        },
        "joint_limits": base_config.joint_limits,
        "wheel_vel_limit": base_config.wheel_vel_limit,
        "slow_loop": {
            "posture_blend_alpha": base_config.posture_blend_alpha,
            "max_hip_pitch_delta": base_config.max_hip_pitch_delta,
            "max_knee_delta": base_config.max_knee_delta,
            "pitch_gate_deg": variant.pitch_gate_deg,
            "pitch_rate_gate_deg_s": base_config.pitch_rate_gate_deg_s,
            "height_correction_enabled": base_config.height_correction_enabled,
            "height_correction_gain": base_config.height_correction_gain,
            "max_height_correction_per_update": base_config.max_height_correction_per_update,
        },
        "fast_loop": {
            "height_scheduled_gains": base_config.height_scheduled_gains,
            "wheel_cmd_filter_enabled": base_config.wheel_cmd_filter_enabled,
            "wheel_cmd_filter_alpha": base_config.wheel_cmd_filter_alpha,
            "wheel_cmd_filter_max_delta": base_config.wheel_cmd_filter_max_delta,
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
        "com_state": {
            "use_sim": base_config.com_use_sim,
        },
        "ik": {
            "scan_points": base_config.ik_scan_points,
            "polynomial_degree": base_config.ik_polynomial_degree,
            "symmetric_fold": base_config.ik_symmetric_fold,
        },
    }

    # Save temp config
    temp_config_path = project_root / "configs" / "controllers" / f"temp_{variant.name}.yaml"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    # Load and create controller
    variant_config = DualRateConfig.from_yaml(temp_config_path)
    controller = DualRateBalanceController(variant_config, mj_model)

    # For disabled variant, set interval to very large value
    if variant.slow_loop_rate_hz == 0.0:
        controller.slow_loop_interval = 999999

    # Clean up temp config
    temp_config_path.unlink()

    return controller


def run_diagnostic_episode(
    controller: DualRateBalanceController,
    env: BalanceEnv,
    variant: SlowLoopVariant,
    height: float,
    init_table: dict,
    max_steps: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run one diagnostic episode with detailed logging."""
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    # Apply balanced root initialization
    state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

    # Reset controller
    controller.reset()

    # Freeze posture for disabled variant (Step 3 parity)
    if variant.slow_loop_rate_hz == 0.0:
        freeze_controller_posture(controller, height, init_table, disable_slow_loop=True)

    logs = []
    step = 0

    while not state.done and step < max_steps:
        obs_np = np.array(state.obs)

        # Track previous targets before compute_action
        prev_target_hip_pitch = float(controller.target_hip_pitch)
        prev_target_knee = float(controller.target_knee)

        action = controller.compute_action(obs_np)
        telem = controller.get_telemetry()

        # Extract state
        gravity_body = obs_np[0:3]
        pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
        roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
        pitch_rate = float(obs_np[6])
        roll_rate = float(obs_np[7])

        roll_deg = np.rad2deg(abs(roll))
        pitch_deg = np.rad2deg(abs(pitch))

        # Apply Step 4 gate to actual slow-loop correction path
        action, correction_info = apply_step4_slow_loop_gate(
            controller=controller,
            action=action,
            variant=variant,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            prev_target_hip_pitch=prev_target_hip_pitch,
            prev_target_knee=prev_target_knee,
        )

        joint_pos = obs_np[9:19]
        joint_vel = obs_np[19:29]
        hip_pitch_l = float(joint_pos[2])
        knee_l = float(joint_pos[3])
        wheel_vel_l = float(joint_vel[4])
        wheel_vel_r = float(joint_vel[9])

        # Step environment
        action_jax = jnp.array(action)
        state = env.step(state, action_jax)

        reward = float(state.reward)
        terminated = bool(state.info['is_fallen'])

        slow_loop_active = bool(
            correction_info['should_update_slow']
            and (
                (not correction_info['gate_active'])
                or correction_info['slow_loop_scale'] > 0.0
            )
        )

        # Log entry
        logs.append({
            'step': step,
            'time': step * controller.config.control_dt,
            'pitch_deg': np.rad2deg(pitch),
            'roll_deg': np.rad2deg(roll),
            'pitch_rate_deg_s': np.rad2deg(pitch_rate),
            'roll_rate_deg_s': np.rad2deg(roll_rate),
            'hip_pitch_target': controller.target_hip_pitch,
            'hip_pitch_actual': hip_pitch_l,
            'knee_target': controller.target_knee,
            'knee_actual': knee_l,
            'wheel_vel_l_rad_s': wheel_vel_l,
            'wheel_vel_r_rad_s': wheel_vel_r,
            'wheel_cmd_raw': telem['wheel_cmd_raw'],
            'wheel_cmd_filtered': telem['filtered_wheel_cmd'],
            'wheel_cmd_norm': telem['wheel_cmd_norm'],
            'slow_loop_active': slow_loop_active,
            'slow_loop_gated': correction_info['gate_active'],
            'roll_gated': bool(roll_deg > variant.roll_gate_deg),
            'pitch_gated': bool(pitch_deg > variant.pitch_gate_deg),
            'gate_active': correction_info['gate_active'],
            'slow_loop_scale': correction_info['slow_loop_scale'],
            'raw_slow_correction_hip_pitch_delta': correction_info['raw_slow_correction']['hip_pitch_delta'],
            'raw_slow_correction_knee_delta': correction_info['raw_slow_correction']['knee_delta'],
            'applied_slow_correction_hip_pitch_delta': correction_info['applied_slow_correction']['hip_pitch_delta'],
            'applied_slow_correction_knee_delta': correction_info['applied_slow_correction']['knee_delta'],
            'num_slow_updates': telem['num_slow_updates'],
            'num_frozen_updates': telem['num_frozen_updates'],
            'reward': reward,
            'terminated': terminated,
        })

        step += 1

    return pd.DataFrame(logs)


def run_batch_evaluation(
    base_config: DualRateConfig,
    env: BalanceEnv,
    mj_model: mujoco.MjModel,
    variant: SlowLoopVariant,
    heights: list[float],
    init_table: dict,
    episodes_per_height: int = 5,
    max_steps: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """Run batch evaluation for one variant across heights."""
    results = []

    console.print(f"\n[bold cyan]Variant: {variant.name}[/bold cyan]")

    for height in heights:
        console.print(f"\n[yellow]Height {height:.2f} m:[/yellow]")

        for ep in range(episodes_per_height):
            # Create fresh controller for each episode
            controller = create_variant_controller(base_config, variant, mj_model)

            rng = jax.random.PRNGKey(seed + ep)
            state = env.reset(rng)

            # Apply balanced root init
            state = state._replace(mjx_data=apply_balanced_root_init(state.mjx_data, height, init_table))

            controller.reset()

            # Freeze posture for disabled variant (Step 3 parity)
            if variant.slow_loop_rate_hz == 0.0:
                freeze_controller_posture(controller, height, init_table, disable_slow_loop=True)

            pitch_sq_sum = 0.0
            roll_sq_sum = 0.0
            wheel_cmd_sq_sum = 0.0
            wheel_speed_sq_sum = 0.0
            slow_loop_active_count = 0
            slow_loop_gated_count = 0
            steps = 0

            for _ in range(max_steps):
                obs_np = np.array(state.obs)

                # Track previous targets before compute_action
                prev_target_hip_pitch = float(controller.target_hip_pitch)
                prev_target_knee = float(controller.target_knee)

                action = controller.compute_action(obs_np)
                telem = controller.get_telemetry()

                gravity_body = obs_np[0:3]
                pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))
                roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))

                roll_deg = np.rad2deg(abs(roll))
                pitch_deg = np.rad2deg(abs(pitch))

                # Apply Step 4 gate to actual slow-loop correction path
                action, correction_info = apply_step4_slow_loop_gate(
                    controller=controller,
                    action=action,
                    variant=variant,
                    roll_deg=roll_deg,
                    pitch_deg=pitch_deg,
                    prev_target_hip_pitch=prev_target_hip_pitch,
                    prev_target_knee=prev_target_knee,
                )

                joint_vel = obs_np[19:29]
                wheel_vel_l = float(joint_vel[4])
                wheel_vel_r = float(joint_vel[9])

                pitch_sq_sum += pitch ** 2
                roll_sq_sum += roll ** 2
                wheel_cmd_sq_sum += telem['filtered_wheel_cmd'] ** 2
                wheel_speed_sq_sum += (wheel_vel_l ** 2 + wheel_vel_r ** 2) / 2

                slow_loop_active = bool(
                    correction_info['should_update_slow']
                    and (
                        (not correction_info['gate_active'])
                        or correction_info['slow_loop_scale'] > 0.0
                    )
                )

                if slow_loop_active:
                    slow_loop_active_count += 1
                if correction_info['gate_active']:
                    slow_loop_gated_count += 1

                action_jax = jnp.array(action)
                state = env.step(state, action_jax)

                steps += 1

                if bool(state.done):
                    break

            survival_time = steps * env.CONTROL_DT
            fell = bool(state.info['is_fallen'])
            pitch_rms = np.sqrt(pitch_sq_sum / steps) if steps > 0 else 0.0
            roll_rms = np.sqrt(roll_sq_sum / steps) if steps > 0 else 0.0
            wheel_cmd_rms = np.sqrt(wheel_cmd_sq_sum / steps) if steps > 0 else 0.0
            wheel_speed_rms = np.sqrt(wheel_speed_sq_sum / steps) if steps > 0 else 0.0
            slow_loop_active_ratio = slow_loop_active_count / steps if steps > 0 else 0.0
            slow_loop_gated_ratio = slow_loop_gated_count / steps if steps > 0 else 0.0

            results.append({
                'variant': variant.name,
                'height': height,
                'episode': ep,
                'survival_time_s': survival_time,
                'fell': fell,
                'pitch_rms_deg': np.rad2deg(pitch_rms),
                'roll_rms_deg': np.rad2deg(roll_rms),
                'wheel_cmd_rms': wheel_cmd_rms,
                'wheel_speed_rms_rad_s': wheel_speed_rms,
                'slow_loop_active_ratio': slow_loop_active_ratio,
                'slow_loop_gated_ratio': slow_loop_gated_ratio,
            })

            console.print(f"  Ep {ep}: {survival_time:.2f}s, fell={fell}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase B.9 Step 4: Slow-loop gating")
    parser.add_argument("--diagnostic-height", type=float, default=0.60, help="Height for diagnostic rollout")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--episodes-per-height", type=int, default=5, help="Episodes per height for batch eval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase_b9_slow_loop_gating_fixed"), help="Output directory")
    parser.add_argument("--variants", nargs="*", help="Variant names to run (default: all)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Phase B.9 Step 4: Slow-loop rate and gating[/bold cyan]\n")

    # Load config and model
    config_path = project_root / "configs/controllers/dual_rate_balance_controller_b9.yaml"
    base_config = DualRateConfig.from_yaml(config_path)
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))

    # Select variants to run
    selected_variants = select_variants(args.variants)
    console.print(f"Running {len(selected_variants)} variant(s): {[v.name for v in selected_variants]}")

    # Load balanced init table
    console.print(f"Loading balanced init table...")
    init_table = load_balanced_init_table()

    # Create environment
    env_config = {
        'episode_length': args.max_steps,
        'low_level_pid': {
            'enabled': True,
            'disable_pid_action_bias': True,
        },
        'domain_randomization': {
            'enabled': False,
        },
    }
    env = BalanceEnv(env_config)

    # Run diagnostics for each variant at h=0.60
    console.print(f"\n[yellow]Step 1: Diagnostic rollouts at h={args.diagnostic_height:.2f}[/yellow]")
    for variant in selected_variants:
        console.print(f"\n[cyan]Variant: {variant.name}[/cyan]")
        controller = create_variant_controller(base_config, variant, mj_model)
        df_diag = run_diagnostic_episode(
            controller, env, variant, args.diagnostic_height, init_table, args.max_steps, args.seed
        )

        diag_csv = args.output_dir / f"diagnostic_{variant.name}_h_{args.diagnostic_height:.2f}.csv"
        df_diag.to_csv(diag_csv, index=False)
        console.print(f"[green]Saved: {diag_csv}[/green]")

        # Summary
        fall_step = df_diag[df_diag['terminated']].index.min()
        if pd.isna(fall_step):
            fall_step = len(df_diag)
        survival_time = fall_step * base_config.control_dt

        console.print(f"  Survival: {survival_time:.2f}s ({fall_step} steps)")
        console.print(f"  Pitch RMS: {np.sqrt(np.mean(df_diag['pitch_deg']**2)):.2f}°")
        console.print(f"  Roll RMS: {np.sqrt(np.mean(df_diag['roll_deg']**2)):.2f}°")
        console.print(f"  Slow loop active: {df_diag['slow_loop_active'].sum()} / {len(df_diag)} steps")
        console.print(f"  Slow loop gated: {df_diag['slow_loop_gated'].sum()} / {len(df_diag)} steps")

    # Run batch evaluation for all variants
    console.print(f"\n[yellow]Step 2: Batch evaluation ({args.episodes_per_height} episodes per height)[/yellow]")
    all_results = []
    for variant in selected_variants:
        variant_results = run_batch_evaluation(
            base_config, env, mj_model, variant, VALID_HEIGHTS, init_table,
            args.episodes_per_height, args.max_steps, args.seed
        )
        all_results.extend(variant_results)

    # Save per-height results
    per_height_csv = args.output_dir / "slow_loop_per_height.csv"
    with open(per_height_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        writer.writeheader()
        writer.writerows(all_results)
    console.print(f"\n[green]Saved: {per_height_csv}[/green]")

    # Aggregate summary
    summary_rows = []
    for variant in selected_variants:
        for height in VALID_HEIGHTS:
            height_results = [r for r in all_results if r['variant'] == variant.name and r['height'] == height]
            survival_times = [r['survival_time_s'] for r in height_results]
            fall_rates = [r['fell'] for r in height_results]

            summary_rows.append({
                'variant': variant.name,
                'height': height,
                'survival_time_mean_s': np.mean(survival_times),
                'survival_time_std_s': np.std(survival_times),
                'fall_rate': np.mean(fall_rates),
                'pitch_rms_deg': np.mean([r['pitch_rms_deg'] for r in height_results]),
                'roll_rms_deg': np.mean([r['roll_rms_deg'] for r in height_results]),
                'wheel_cmd_rms': np.mean([r['wheel_cmd_rms'] for r in height_results]),
                'wheel_speed_rms_rad_s': np.mean([r['wheel_speed_rms_rad_s'] for r in height_results]),
                'slow_loop_active_ratio': np.mean([r['slow_loop_active_ratio'] for r in height_results]),
                'slow_loop_gated_ratio': np.mean([r['slow_loop_gated_ratio'] for r in height_results]),
            })

    summary_csv = args.output_dir / "slow_loop_summary.csv"
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    console.print(f"[green]Saved: {summary_csv}[/green]")

    # Display summary table
    table = Table(title="Step 4 Slow-Loop Gating Summary")
    table.add_column("Variant")
    table.add_column("Height (m)")
    table.add_column("Survival (s)")
    table.add_column("Fall Rate")
    table.add_column("Roll RMS (°)")

    for row in summary_rows:
        table.add_row(
            row['variant'],
            f"{row['height']:.2f}",
            f"{row['survival_time_mean_s']:.2f} ± {row['survival_time_std_s']:.2f}",
            f"{row['fall_rate']:.1%}",
            f"{row['roll_rms_deg']:.2f}",
        )

    console.print("\n")
    console.print(table)
    console.print()

    # Find best variant
    console.print("\n[yellow]Step 3: Selecting best variant[/yellow]")
    variant_scores = {}
    for variant in selected_variants:
        variant_rows = [r for r in summary_rows if r['variant'] == variant.name]
        mean_survival = np.mean([r['survival_time_mean_s'] for r in variant_rows])
        mean_fall_rate = np.mean([r['fall_rate'] for r in variant_rows])
        mean_roll_rms = np.mean([r['roll_rms_deg'] for r in variant_rows])

        # Score: prioritize survival time, penalize fall rate and roll RMS
        score = mean_survival - 10.0 * mean_fall_rate - 0.1 * mean_roll_rms
        variant_scores[variant.name] = score

        console.print(f"{variant.name}: survival={mean_survival:.2f}s, fall_rate={mean_fall_rate:.1%}, roll_rms={mean_roll_rms:.2f}°, score={score:.2f}")

    best_variant_name = max(variant_scores, key=variant_scores.get)
    best_variant = next(v for v in VARIANTS if v.name == best_variant_name)

    console.print(f"\n[bold green]Best variant: {best_variant_name}[/bold green]")

    # Save best config
    best_config_yaml = args.output_dir / "best_slow_loop_config.yaml"
    best_config_dict = {
        "best_variant": best_variant.name,
        "slow_loop_rate_hz": best_variant.slow_loop_rate_hz,
        "roll_gate_deg": best_variant.roll_gate_deg,
        "pitch_gate_deg": best_variant.pitch_gate_deg,
        "slow_loop_scale_when_unstable": best_variant.slow_loop_scale_when_unstable,
        "mean_survival_s": np.mean([r['survival_time_mean_s'] for r in summary_rows if r['variant'] == best_variant_name]),
        "mean_fall_rate": np.mean([r['fall_rate'] for r in summary_rows if r['variant'] == best_variant_name]),
        "mean_roll_rms_deg": np.mean([r['roll_rms_deg'] for r in summary_rows if r['variant'] == best_variant_name]),
    }
    with open(best_config_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(to_jsonable_or_yamlable(best_config_dict), f)
    console.print(f"[green]Saved: {best_config_yaml}[/green]")


if __name__ == "__main__":
    main()
