#!/usr/bin/env python3
"""
Phase B.9 Step 5.20: Low-Stiffness Dynamic Balance Evaluation

Tests whether reducing posture stiffness improves stability.

Core hypothesis:
The current controller is over-stiff and fighting natural balancing dynamics.

Evidence:
- Pure RL previously balanced successfully without persistent saturation
- Current PID saturates at ±30 Nm continuously
- Plant is stabilizable, but classical control structure may be inefficient

Evaluation:
Test soft mode candidates at h=0.60 and compare against baseline.

Usage:
    python scripts/phase_b9_step5_20_evaluation.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
    DualRateConfig,
)
from wheeled_biped.envs.balance_env import BalanceEnv


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


def evaluate_candidate(
    candidate: SoftModeCandidate,
    num_episodes: int,
    height: float,
    seed: int,
) -> dict:
    """Evaluate a single soft mode candidate."""

    # Load base controller config
    base_config_path = Path("configs/controllers/dual_rate_balance_controller_b9.yaml")

    # Load soft mode config
    with open(candidate.config_path) as f:
        soft_cfg = yaml.safe_load(f)

    # Merge configs
    with open(base_config_path) as f:
        full_cfg = yaml.safe_load(f)

    full_cfg["soft_dynamic_balance"] = soft_cfg["soft_dynamic_balance"]

    # Save merged config temporarily
    temp_config_path = candidate.config_path.parent / f"temp_{candidate.name}.yaml"
    with open(temp_config_path, "w") as f:
        yaml.dump(full_cfg, f)

    # Create environment with proper config
    env_config = {
        "task": {
            "initial_min_height": height,
        },
        "domain_randomization": {
            "enabled": False,  # Disable DR for clean evaluation
        },
        "low_level_pid": {
            "enabled": False,  # Using controller directly
        },
    }
    env = BalanceEnv(config=env_config)

    # Create controller
    config = DualRateConfig.from_yaml(temp_config_path)
    controller = DualRateBalanceController(config, env.mj_model)

    rng = jax.random.PRNGKey(seed)
    episode_results = []

    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)

        episode_steps = 0
        pitch_values = []
        roll_values = []
        torque_values = []

        for step in range(500):
            obs = np.array(state.obs)
            action = controller.compute_action(obs)

            state = env.step(state, jnp.array(action))

            # Extract metrics
            gravity = obs[0:3]
            pitch = float(np.arcsin(np.clip(-gravity[0], -1.0, 1.0)))
            roll = float(np.arcsin(np.clip(gravity[1], -1.0, 1.0)))

            pitch_values.append(np.rad2deg(pitch))
            roll_values.append(np.rad2deg(roll))

            # Approximate torque from action (normalized action * max torque)
            # This is a rough proxy for actual torque
            torque_proxy = np.abs(action) * 30.0  # Assume max ~30 Nm
            torque_values.append(torque_proxy)

            episode_steps += 1

            if state.done:
                break

        survival_time = episode_steps * env.CONTROL_DT
        fell = bool(state.done)

        # Compute torque efficiency metrics
        torque_array = np.array(torque_values)
        mean_torque = float(np.mean(np.abs(torque_array)))
        rms_torque = float(np.sqrt(np.mean(torque_array**2)))

        episode_results.append({
            "episode": ep,
            "survival_time": float(survival_time),
            "fell": fell,
            "pitch_rms_deg": float(np.sqrt(np.mean(np.array(pitch_values)**2))),
            "roll_rms_deg": float(np.sqrt(np.mean(np.array(roll_values)**2))),
            "mean_torque": mean_torque,
            "rms_torque": rms_torque,
        })

    # Clean up temp config
    temp_config_path.unlink()

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
    print("Phase B.9 Step 5.20: Low-Stiffness Dynamic Balance Evaluation")
    print("=" * 80)
    print("\nHypothesis: Current controller is over-stiff and fighting natural dynamics")
    print("\nBaseline (Step 5.18c):")
    print("  h=0.60: survival=0.86s, fall_rate=0.80, roll_rms=15.9deg")
    print("  PID saturation: 93.75%")
    print("  RMS torque: ~30 Nm")

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

        # Compare to baseline
        baseline_survival = 0.86
        if result['mean_survival_time'] > baseline_survival:
            improvement = (result['mean_survival_time'] / baseline_survival - 1) * 100
            print(f"  [+] IMPROVEMENT: +{improvement:.1f}% vs baseline")
        else:
            degradation = (1 - result['mean_survival_time'] / baseline_survival) * 100
            print(f"  [-] DEGRADATION: -{degradation:.1f}% vs baseline")

    # Save results
    results_path = args.output_dir / "candidate_results.json"
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

    baseline_survival = 0.86
    baseline_torque = 30.0

    # Torque efficiency analysis
    print(f"\nTorque Efficiency Analysis:")
    for result in results:
        torque_reduction = (1 - result['rms_torque'] / baseline_torque) * 100
        print(f"  {result['candidate']}: {result['rms_torque']:.1f} Nm ({torque_reduction:+.1f}%)")

    # Decision
    print(f"\n{'='*80}")
    print("Decision")
    print(f"{'='*80}")

    if best['mean_survival_time'] > baseline_survival * 1.2:
        print("\nSOFT_MODE_IMPROVES_STABILITY")
        print(f"Soft mode ({best['candidate']}) improves survival by ")
        print(f"{(best['mean_survival_time']/baseline_survival-1)*100:.1f}%")
        print(f"and reduces torque by {(1-best['rms_torque']/baseline_torque)*100:.1f}%.")
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
