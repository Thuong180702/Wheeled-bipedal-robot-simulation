#!/usr/bin/env python3
"""
Phase B.9 Step 5.20: Low-Stiffness Dynamic Balance Evaluation (CORRECTED)

Tests whether reducing posture stiffness improves stability.

CRITICAL FIX: This version properly integrates with the environment's control pipeline
using the same pattern as Step 5.18b/5.18c, instead of bypassing it.

Core hypothesis:
The current controller is over-stiff and fighting natural balancing dynamics.

Evidence:
- Pure RL previously balanced successfully without persistent saturation
- Current PID saturates at ±30 Nm continuously
- Plant is stabilizable, but classical control structure may be inefficient

Evaluation:
Test soft mode candidates at h=0.60 and compare against baseline.

Usage:
    python scripts/phase_b9_step5_20_evaluation_corrected.py
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_step5_lqr_gain_strengthening import (
    apply_balanced_root_init,
    create_tuned_controller,
    load_balanced_init_table,
)
from wheeled_biped.controllers.action_codec import ACTION_DIM
from wheeled_biped.controllers.dual_rate_balance_controller import DualRateConfig
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

BEST_LQR_PATH = PROJECT_ROOT / "outputs" / "phase_b9_lqr_gain_strengthening" / "best_lqr_config.yaml"
CONTROLLER_CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"


@dataclass
class SoftModeCandidate:
    """Soft dynamic balance candidate."""
    name: str
    config_path: Path
    stiffness_reduction: float
    deadband_deg: float


def create_candidates(output_dir: Path) -> list[SoftModeCandidate]:
    """Create soft mode candidates."""
    return [
        SoftModeCandidate(
            name="baseline",
            config_path=output_dir / "soft_baseline.yaml",
            stiffness_reduction=1.0,
            deadband_deg=0.0,
        ),
        SoftModeCandidate(
            name="conservative",
            config_path=output_dir / "soft_conservative.yaml",
            stiffness_reduction=0.7,
            deadband_deg=1.0,
        ),
        SoftModeCandidate(
            name="moderate",
            config_path=output_dir / "soft_moderate.yaml",
            stiffness_reduction=0.5,
            deadband_deg=2.0,
        ),
        SoftModeCandidate(
            name="aggressive",
            config_path=output_dir / "soft_aggressive.yaml",
            stiffness_reduction=0.3,
            deadband_deg=3.0,
        ),
    ]


def load_best_lqr_params() -> dict[str, float]:
    """Load best LQR parameters from Step 5."""
    with BEST_LQR_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env_config(height: float = 0.60, episode_length: int = 250) -> dict:
    """Create base environment config."""
    return {
        "low_level_pid": {"enabled": True, "disable_pid_action_bias": True},
        "domain_randomization": {"enabled": False},
        "sensor_noise": {"enabled": False},
        "task": {"initial_min_height": height, "episode_length": episode_length},
        "termination": {"max_tilt_rad": 0.8, "min_height": 0.3},
    }


def activation_config(base_config: dict) -> dict:
    """Add low-level control config to enable hybrid_pid_plus_torque mode.

    Uses the same WBC torque configuration as Step 5.18c's "strong_k20" candidate
    to isolate the stiffness reduction variable.
    """
    cfg = {
        key: (value.copy() if isinstance(value, dict) else value)
        for key, value in base_config.items()
    }
    # Use Step 5.18c's "strong_k20" WBC configuration
    cfg["low_level_control"] = {
        "mode": "hybrid_pid_plus_torque",
        "torque_control": {
            "enabled": True,
            "max_ctrl_fraction": 0.5,  # strong_k20 value
            "allow_leg_torque": True,
            "allow_wheel_torque": False,  # strong_k20 disabled wheel torque
            "allow_hip_yaw_torque": False,
        },
    }
    return cfg


def make_controller(model: mujoco.MjModel, soft_config_path: Path):
    """Create controller with soft mode config."""
    # Load base controller config
    base_config = DualRateConfig.from_yaml(CONTROLLER_CONFIG_PATH)

    # Load soft mode config
    with open(soft_config_path) as f:
        soft_cfg = yaml.safe_load(f)

    # Merge soft mode config into base config
    # The DualRateConfig.from_yaml already handles soft_dynamic_balance section
    # We need to create a temporary merged config file
    with open(CONTROLLER_CONFIG_PATH) as f:
        full_cfg = yaml.safe_load(f)

    full_cfg["soft_dynamic_balance"] = soft_cfg["soft_dynamic_balance"]

    # Save temporary merged config
    temp_config_path = soft_config_path.parent / f"temp_{soft_config_path.stem}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(full_cfg, f)

    # Create controller with merged config
    merged_config = DualRateConfig.from_yaml(temp_config_path)
    controller = create_tuned_controller(merged_config, load_best_lqr_params(), model)

    # Clean up temp file
    temp_config_path.unlink()

    return controller


def compute_torque_residual_action(obs: np.ndarray) -> np.ndarray:
    """Compute WBC torque residual using Step 5.18c's "strong_k20" gains.

    This uses the exact same WBC configuration that achieved 0.86s survival
    in Step 5.18c, isolating the stiffness reduction variable.
    """
    # Extract state from observation
    gravity_body = obs[0:3]
    roll = float(np.arcsin(np.clip(gravity_body[1], -1.0, 1.0)))
    pitch = float(np.arcsin(np.clip(-gravity_body[0], -1.0, 1.0)))

    angular_vel = obs[6:9]
    pitch_rate = float(angular_vel[0])
    roll_rate = float(angular_vel[1])

    # Step 5.18c "strong_k20" gains
    k_roll = 20.0
    k_roll_rate = 2.0
    k_pitch = 5.0
    k_pitch_rate = 0.5

    # Compute torque commands
    roll_cmd = -k_roll * roll - k_roll_rate * roll_rate
    pitch_cmd = -k_pitch * pitch - k_pitch_rate * pitch_rate

    # Build residual action (normalized to [-1, 1])
    residual = np.zeros(ACTION_DIM, dtype=np.float32)

    # Hip roll (left/right mirrored)
    residual[0] = np.clip(roll_cmd, -1.0, 1.0)  # L_HIP_ROLL
    residual[5] = np.clip(-roll_cmd, -1.0, 1.0)  # R_HIP_ROLL

    # Hip pitch (symmetric)
    residual[2] = np.clip(pitch_cmd, -1.0, 1.0)  # L_HIP_PITCH
    residual[7] = np.clip(pitch_cmd, -1.0, 1.0)  # R_HIP_PITCH

    # Knee (half of pitch, symmetric)
    residual[3] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)  # L_KNEE
    residual[8] = np.clip(-0.5 * pitch_cmd, -1.0, 1.0)  # R_KNEE

    # Hip yaw (disabled)
    residual[1] = 0.0  # L_HIP_YAW
    residual[6] = 0.0  # R_HIP_YAW

    # Wheels (disabled in strong_k20)
    residual[4] = 0.0  # L_WHEEL
    residual[9] = 0.0  # R_WHEEL

    return residual


def set_height_and_roll(state, env: BalanceEnv, height: float, init_table: dict):
    """Initialize state at target height with balanced root."""
    mjx_data = apply_balanced_root_init(state.mjx_data, height, init_table)
    base_obs = env._extract_obs(mjx_data, jnp.zeros(env.num_actions), state.info["noise_rng"])
    height_norm = (height - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    current_height_norm = (mjx_data.qpos[2] - env.MIN_HEIGHT_CMD) / (env.MAX_HEIGHT_CMD - env.MIN_HEIGHT_CMD)
    obs = jnp.concatenate([base_obs, jnp.array([height_norm, current_height_norm, 0.0], dtype=base_obs.dtype)])
    info = {
        **state.info,
        "height_command": jnp.array(height, dtype=mjx_data.qpos.dtype),
        "initial_yaw": jnp.array(0.0, dtype=mjx_data.qpos.dtype),
    }
    return state._replace(mjx_data=mjx_data, obs=obs, info=info, prev_action=jnp.zeros(env.num_actions))


def evaluate_candidate(
    candidate: SoftModeCandidate,
    num_episodes: int,
    height: float,
    seed: int,
) -> dict:
    """Evaluate a single soft mode candidate using correct control pipeline."""

    # Create environment with proper control integration
    env_config = activation_config(make_env_config(height, 250))
    env = BalanceEnv(config=env_config)

    # Create model and controller
    model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    controller = make_controller(model, candidate.config_path)

    # Load balanced init table
    init_table = load_balanced_init_table()

    rng = jax.random.PRNGKey(seed)
    episode_results = []

    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)

        # Initialize at target height with balanced root
        state = set_height_and_roll(state, env, height, init_table)
        controller.reset()

        episode_steps = 0
        pitch_values = []
        roll_values = []
        torque_values = []
        saturation_count = 0

        for step in range(250):
            obs_np = np.array(state.obs)
            action = controller.compute_action(obs_np)

            # Compute WBC torque residual using Step 5.18c's "strong_k20" gains
            residual = compute_torque_residual_action(obs_np)
            state = state._replace(info={**state.info, "torque_residual_action": jnp.array(residual)})

            # Step environment (it will use hybrid_pid_plus_torque internally)
            state = env.step(state, jnp.array(action))

            # Extract metrics
            gravity = obs_np[0:3]
            pitch = float(np.arcsin(np.clip(-gravity[0], -1.0, 1.0)))
            roll = float(np.arcsin(np.clip(gravity[1], -1.0, 1.0)))

            pitch_values.append(np.rad2deg(pitch))
            roll_values.append(np.rad2deg(roll))

            # Extract actual torque from state.info if available
            if "final_actuator_ctrl" in state.info:
                final_ctrl = np.array(state.info["final_actuator_ctrl"])
                torque_values.append(np.abs(final_ctrl))

                # Check saturation
                if "actuator_saturation_flags" in state.info:
                    sat_flags = np.array(state.info["actuator_saturation_flags"])
                    if np.any(sat_flags):
                        saturation_count += 1

            episode_steps += 1

            if state.done:
                break

        survival_time = episode_steps * env.CONTROL_DT
        fell = bool(state.done)

        # Compute torque efficiency metrics
        if torque_values:
            torque_array = np.array(torque_values)
            mean_torque = float(np.mean(torque_array))
            rms_torque = float(np.sqrt(np.mean(torque_array**2)))
            saturation_rate = saturation_count / episode_steps if episode_steps > 0 else 0.0
        else:
            mean_torque = 0.0
            rms_torque = 0.0
            saturation_rate = 0.0

        episode_results.append({
            "episode": ep,
            "survival_time": float(survival_time),
            "fell": fell,
            "pitch_rms_deg": float(np.sqrt(np.mean(np.array(pitch_values)**2))),
            "roll_rms_deg": float(np.sqrt(np.mean(np.array(roll_values)**2))),
            "mean_torque": mean_torque,
            "rms_torque": rms_torque,
            "saturation_rate": saturation_rate,
        })

    summary = {
        "candidate": candidate.name,
        "stiffness_reduction": candidate.stiffness_reduction,
        "deadband_deg": candidate.deadband_deg,
        "height": height,
        "num_episodes": num_episodes,
        "mean_survival_time": float(np.mean([r["survival_time"] for r in episode_results])),
        "fall_rate": float(np.mean([r["fell"] for r in episode_results])),
        "mean_pitch_rms_deg": float(np.mean([r["pitch_rms_deg"] for r in episode_results])),
        "mean_roll_rms_deg": float(np.mean([r["roll_rms_deg"] for r in episode_results])),
        "mean_torque": float(np.mean([r["mean_torque"] for r in episode_results])),
        "rms_torque": float(np.mean([r["rms_torque"] for r in episode_results])),
        "mean_saturation_rate": float(np.mean([r["saturation_rate"] for r in episode_results])),
        "episodes": episode_results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                       default=Path("outputs/phase_b9_step5_20_low_stiffness_dynamic_balance"))
    parser.add_argument("--height", type=float, default=0.60)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase B.9 Step 5.20: Low-Stiffness Dynamic Balance Evaluation (CORRECTED)")
    print("=" * 80)
    print("\nHypothesis: Current controller is over-stiff and fighting natural dynamics")
    print("\nBaseline (Step 5.18c):")
    print("  h=0.60: survival=0.86s, fall_rate=0.80, roll_rms=15.9deg")
    print("  PID saturation: 93.75%")
    print("  RMS torque: ~30 Nm")
    print("\nCRITICAL FIX: Using correct control pipeline integration from Step 5.18b")

    # Create candidates
    candidates = create_candidates(args.output_dir)

    print(f"\nTesting {len(candidates)} candidates at h={args.height}m:")
    for c in candidates:
        print(f"  - {c.name}: stiffness={c.stiffness_reduction}, deadband={c.deadband_deg}deg")

    # Evaluate candidates
    results = []
    for candidate in candidates:
        print(f"\n{'='*80}")
        print(f"Evaluating: {candidate.name}")
        print(f"{'='*80}")

        result = evaluate_candidate(candidate, args.episodes, args.height, args.seed)
        results.append(result)

        print(f"\nResults:")
        print(f"  Survival: {result['mean_survival_time']:.2f}s")
        print(f"  Fall rate: {result['fall_rate']:.2f}")
        print(f"  Pitch RMS: {result['mean_pitch_rms_deg']:.2f}deg")
        print(f"  Roll RMS: {result['mean_roll_rms_deg']:.2f}deg")
        print(f"  Mean torque: {result['mean_torque']:.2f} Nm")
        print(f"  RMS torque: {result['rms_torque']:.2f} Nm")
        print(f"  Saturation rate: {result['mean_saturation_rate']:.2%}")

        # Compare to baseline
        baseline_survival = 0.86
        if result['mean_survival_time'] > baseline_survival:
            improvement = (result['mean_survival_time'] / baseline_survival - 1) * 100
            print(f"  [+] IMPROVEMENT: +{improvement:.1f}% vs baseline")
        else:
            degradation = (1 - result['mean_survival_time'] / baseline_survival) * 100
            print(f"  [-] DEGRADATION: -{degradation:.1f}% vs baseline")

    # Save results
    results_path = args.output_dir / "corrected_candidate_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    # Find best candidate
    best = max(results, key=lambda r: r["mean_survival_time"])
    print(f"\nBest candidate: {best['candidate']}")
    print(f"  Stiffness reduction: {best['stiffness_reduction']}")
    print(f"  Deadband: {best['deadband_deg']}deg")
    print(f"  Survival: {best['mean_survival_time']:.2f}s")
    print(f"  Fall rate: {best['fall_rate']:.2f}")
    print(f"  Roll RMS: {best['mean_roll_rms_deg']:.2f}deg")
    print(f"  RMS torque: {best['rms_torque']:.2f} Nm")
    print(f"  Saturation rate: {best['mean_saturation_rate']:.2%}")

    baseline_survival = 0.86
    baseline_torque = 30.0
    baseline_saturation = 0.9375

    # Torque efficiency analysis
    print(f"\nTorque Efficiency Analysis:")
    for result in results:
        torque_change = (result['rms_torque'] / baseline_torque - 1) * 100
        print(f"  {result['candidate']}: {result['rms_torque']:.1f} Nm ({torque_change:+.1f}%)")

    # Saturation analysis
    print(f"\nSaturation Analysis:")
    for result in results:
        sat_change = (result['mean_saturation_rate'] / baseline_saturation - 1) * 100
        print(f"  {result['candidate']}: {result['mean_saturation_rate']:.2%} ({sat_change:+.1f}%)")

    # Decision
    print(f"\n{'='*80}")
    print("Decision")
    print(f"{'='*80}")

    if best['mean_survival_time'] > baseline_survival * 1.2:
        print("\nSOFT_MODE_IMPROVES_STABILITY")
        print(f"Soft mode ({best['candidate']}) improves survival by ")
        print(f"{(best['mean_survival_time']/baseline_survival-1)*100:.1f}%")
        print(f"and reduces saturation by {(1-best['mean_saturation_rate']/baseline_saturation)*100:.1f}%.")
        print("\nConclusion: Controller was over-stiff. Soft mode is more efficient.")
        print("Recommendation: Adopt soft mode as new baseline, proceed to Step 6.")
    elif best['mean_survival_time'] > baseline_survival:
        print("\nMARGINAL_IMPROVEMENT")
        print(f"Soft mode shows small improvement (+{(best['mean_survival_time']/baseline_survival-1)*100:.1f}%).")
        print("May not be sufficient for Step 6 gate.")
        print("\nConclusion: Stiffness reduction helps slightly but not dramatically.")
        print("Recommendation: Consider hybrid approach or further investigation.")
    else:
        print("\nSOFT_MODE_DEGRADES_PERFORMANCE")
        print("Soft mode does not improve stability at h=0.60.")
        print("\nConclusion: Current stiffness is necessary, not excessive.")
        print("Classical control architecture may be fundamentally limited.")
        print("\nRecommendation: Consider alternative approaches:")
        print("  - Pure RL (no classical control)")
        print("  - Different hybrid architecture")
        print("  - Architectural redesign")

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
