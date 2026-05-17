#!/usr/bin/env python3
"""
Phase B.9 Step 5.19: Authority Reallocation Evaluation

Tests PID output clamping to reserve actuator authority for WBC corrections.

Root cause (from Step 5.18c):
- PID outputs ~30 Nm and saturates actuators
- WBC torque residuals ~1 Nm
- Authority ratio: 1:30 (WBC:PID)
- PID saturation suppresses WBC corrections

Solution:
- Clamp PID output to fraction of actuator range
- Reserve headroom for WBC residuals
- Test pid_authority_fraction: 1.0, 0.9, 0.8, 0.7, 0.6, 0.5

Phases:
1. Static response validation (roll perturbations)
2. h=0.60 survival evaluation (5 episodes)
3. Full validation (all heights, best candidates only)

Usage:
    python scripts/phase_b9_step5_19_authority_reallocation_evaluation.py
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from mujoco import mjx

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
)
from wheeled_biped.envs.balance_env import BalanceEnv


@dataclass
class AuthorityCandidate:
    """Authority reallocation candidate."""
    name: str
    pid_authority_fraction: float
    config_path: Path


def create_candidates(output_dir: Path) -> list[AuthorityCandidate]:
    """Create authority reallocation candidates."""
    fractions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    candidates = []

    for frac in fractions:
        name = f"pid_authority_{frac}"
        config_path = output_dir / f"{name}.yaml"
        candidates.append(AuthorityCandidate(
            name=name,
            pid_authority_fraction=frac,
            config_path=config_path,
        ))

    return candidates


def run_static_response_validation(
    candidate: AuthorityCandidate,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Phase 3: Static response validation."""

    # Load config
    with open(candidate.config_path) as f:
        torque_cfg = yaml.safe_load(f)

    # Create environment with authority reallocation
    env_config = {
        "low_level_control": {
            "mode": "hybrid_pid_plus_torque",
            "torque_control": torque_cfg,
        }
    }

    env = BalanceEnv(
        episode_length=100,
        action_repeat=1,
        height_curriculum_mode="fixed",
        fixed_height=0.60,
        config=env_config,
    )

    # Create controller
    controller_config_path = Path("outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/best_torque_gain_config.yaml")
    with open(controller_config_path) as f:
        controller_cfg = yaml.safe_load(f)

    controller = DualRateBalanceController(controller_cfg)

    # Test perturbations
    perturbations = [
        {"name": "roll_pos_2deg", "roll_deg": 2.0},
        {"name": "roll_neg_2deg", "roll_deg": -2.0},
        {"name": "roll_pos_5deg", "roll_deg": 5.0},
        {"name": "roll_neg_5deg", "roll_deg": -5.0},
    ]

    results = []
    rng = jax.random.PRNGKey(seed)

    for pert in perturbations:
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)

        # Apply perturbation to initial state
        roll_rad = np.deg2rad(pert["roll_deg"])

        # Modify gravity vector to simulate roll
        obs = np.array(state.obs)
        obs[0] = -np.sin(0.0)  # pitch (keep zero)
        obs[1] = np.sin(roll_rad)  # roll
        obs[2] = np.cos(roll_rad) * np.cos(0.0)  # gravity_z

        # Get controller response
        action = controller.compute_action(obs)

        # Step once to see response
        rng, step_rng = jax.random.split(rng)
        state = env.step(state, jnp.array(action), step_rng)

        # Extract metrics
        gravity_after = np.array(state.obs[0:3])
        roll_after = float(np.arcsin(np.clip(gravity_after[1], -1.0, 1.0)))
        roll_after_deg = np.rad2deg(roll_after)

        # Check if response is stabilizing
        stabilizing = (np.sign(pert["roll_deg"]) != np.sign(roll_after_deg - pert["roll_deg"]))

        results.append({
            "perturbation": pert["name"],
            "roll_initial_deg": pert["roll_deg"],
            "roll_after_deg": float(roll_after_deg),
            "stabilizing": bool(stabilizing),
        })

    summary = {
        "candidate": candidate.name,
        "pid_authority_fraction": candidate.pid_authority_fraction,
        "perturbations": results,
        "stabilizing_count": sum(r["stabilizing"] for r in results),
        "total_count": len(results),
    }

    return summary


def run_h060_survival_evaluation(
    candidate: AuthorityCandidate,
    num_episodes: int,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Phase 4: h=0.60 survival evaluation."""

    # Load config
    with open(candidate.config_path) as f:
        torque_cfg = yaml.safe_load(f)

    # Create environment with authority reallocation
    env_config = {
        "low_level_control": {
            "mode": "hybrid_pid_plus_torque",
            "torque_control": torque_cfg,
        }
    }

    env = BalanceEnv(
        episode_length=500,
        action_repeat=1,
        height_curriculum_mode="fixed",
        fixed_height=0.60,
        config=env_config,
    )

    # Create controller
    controller_config_path = Path("outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/best_torque_gain_config.yaml")
    with open(controller_config_path) as f:
        controller_cfg = yaml.safe_load(f)

    controller = DualRateBalanceController(controller_cfg)

    rng = jax.random.PRNGKey(seed)
    episode_results = []

    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)

        episode_steps = 0
        pitch_values = []
        roll_values = []

        for step in range(500):
            obs = np.array(state.obs)
            action = controller.compute_action(obs)

            rng, step_rng = jax.random.split(rng)
            state = env.step(state, jnp.array(action), step_rng)

            # Extract pitch and roll
            gravity = obs[0:3]
            pitch = float(np.arcsin(np.clip(-gravity[0], -1.0, 1.0)))
            roll = float(np.arcsin(np.clip(gravity[1], -1.0, 1.0)))

            pitch_values.append(np.rad2deg(pitch))
            roll_values.append(np.rad2deg(roll))

            episode_steps += 1

            if state.done:
                break

        survival_time = episode_steps * env.dt
        fell = bool(state.done)

        episode_results.append({
            "episode": ep,
            "survival_time": float(survival_time),
            "fell": fell,
            "pitch_rms_deg": float(np.sqrt(np.mean(np.array(pitch_values)**2))),
            "roll_rms_deg": float(np.sqrt(np.mean(np.array(roll_values)**2))),
        })

        print(f"  Episode {ep}: survival={survival_time:.2f}s, fell={fell}, "
              f"roll_rms={episode_results[-1]['roll_rms_deg']:.1f}deg")

    summary = {
        "candidate": candidate.name,
        "pid_authority_fraction": candidate.pid_authority_fraction,
        "num_episodes": num_episodes,
        "mean_survival_time": float(np.mean([r["survival_time"] for r in episode_results])),
        "fall_rate": float(np.mean([r["fell"] for r in episode_results])),
        "mean_pitch_rms_deg": float(np.mean([r["pitch_rms_deg"] for r in episode_results])),
        "mean_roll_rms_deg": float(np.mean([r["roll_rms_deg"] for r in episode_results])),
        "episodes": episode_results,
    }

    return summary


def run_full_validation(
    candidate: AuthorityCandidate,
    heights: list[float],
    episodes_per_height: int,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Phase 5: Full validation across heights."""

    # Load config
    with open(candidate.config_path) as f:
        torque_cfg = yaml.safe_load(f)

    # Create controller
    controller_config_path = Path("outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/best_torque_gain_config.yaml")
    with open(controller_config_path) as f:
        controller_cfg = yaml.safe_load(f)

    controller = DualRateBalanceController(controller_cfg)

    rng = jax.random.PRNGKey(seed)
    height_results = []

    for height in heights:
        print(f"  Testing height {height:.2f}m...")

        # Create environment for this height
        env_config = {
            "low_level_control": {
                "mode": "hybrid_pid_plus_torque",
                "torque_control": torque_cfg,
            }
        }

        env = BalanceEnv(
            episode_length=500,
            action_repeat=1,
            height_curriculum_mode="fixed",
            fixed_height=height,
            config=env_config,
        )

        episode_results = []

        for ep in range(episodes_per_height):
            rng, reset_rng = jax.random.split(rng)
            state = env.reset(reset_rng)

            episode_steps = 0
            pitch_values = []
            roll_values = []

            for step in range(500):
                obs = np.array(state.obs)
                action = controller.compute_action(obs)

                rng, step_rng = jax.random.split(rng)
                state = env.step(state, jnp.array(action), step_rng)

                # Extract pitch and roll
                gravity = obs[0:3]
                pitch = float(np.arcsin(np.clip(-gravity[0], -1.0, 1.0)))
                roll = float(np.arcsin(np.clip(gravity[1], -1.0, 1.0)))

                pitch_values.append(np.rad2deg(pitch))
                roll_values.append(np.rad2deg(roll))

                episode_steps += 1

                if state.done:
                    break

            survival_time = episode_steps * env.dt
            fell = bool(state.done)

            episode_results.append({
                "episode": ep,
                "survival_time": float(survival_time),
                "fell": fell,
                "pitch_rms_deg": float(np.sqrt(np.mean(np.array(pitch_values)**2))),
                "roll_rms_deg": float(np.sqrt(np.mean(np.array(roll_values)**2))),
            })

        height_summary = {
            "height": height,
            "mean_survival_time": float(np.mean([r["survival_time"] for r in episode_results])),
            "fall_rate": float(np.mean([r["fell"] for r in episode_results])),
            "mean_pitch_rms_deg": float(np.mean([r["pitch_rms_deg"] for r in episode_results])),
            "mean_roll_rms_deg": float(np.mean([r["roll_rms_deg"] for r in episode_results])),
        }

        height_results.append(height_summary)
        print(f"    survival={height_summary['mean_survival_time']:.2f}s, "
              f"fall_rate={height_summary['fall_rate']:.2f}")

    summary = {
        "candidate": candidate.name,
        "pid_authority_fraction": candidate.pid_authority_fraction,
        "heights": height_results,
        "overall_mean_survival": float(np.mean([h["mean_survival_time"] for h in height_results])),
        "overall_fall_rate": float(np.mean([h["fall_rate"] for h in height_results])),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path,
                       default=Path("outputs/phase_b9_step5_19_controller_authority_reallocation"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-static", action="store_true", help="Skip static response validation")
    parser.add_argument("--skip-h060", action="store_true", help="Skip h=0.60 survival evaluation")
    parser.add_argument("--run-full", action="store_true", help="Run full validation for best candidates")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create candidates
    candidates = create_candidates(args.output_dir)

    print("=" * 80)
    print("Phase B.9 Step 5.19: Authority Reallocation Evaluation")
    print("=" * 80)
    print(f"\nCandidates: {len(candidates)}")
    for c in candidates:
        print(f"  - {c.name}: pid_authority_fraction={c.pid_authority_fraction}")

    # Phase 3: Static response validation
    if not args.skip_static:
        print("\n" + "=" * 80)
        print("Phase 3: Static Response Validation")
        print("=" * 80)

        static_results = []
        for candidate in candidates:
            print(f"\nTesting {candidate.name}...")
            result = run_static_response_validation(candidate, args.output_dir, args.seed)
            static_results.append(result)
            print(f"  Stabilizing: {result['stabilizing_count']}/{result['total_count']}")

        # Save results
        static_path = args.output_dir / "static_response_validation.json"
        with open(static_path, "w") as f:
            json.dump(static_results, f, indent=2)
        print(f"\nStatic response results saved to {static_path}")

    # Phase 4: h=0.60 survival evaluation
    if not args.skip_h060:
        print("\n" + "=" * 80)
        print("Phase 4: h=0.60 Survival Evaluation")
        print("=" * 80)

        h060_results = []
        for candidate in candidates:
            print(f"\nTesting {candidate.name}...")
            result = run_h060_survival_evaluation(candidate, 5, args.output_dir, args.seed)
            h060_results.append(result)
            print(f"  Mean survival: {result['mean_survival_time']:.2f}s")
            print(f"  Fall rate: {result['fall_rate']:.2f}")
            print(f"  Roll RMS: {result['mean_roll_rms_deg']:.1f}deg")

        # Save results
        h060_path = args.output_dir / "h060_survival_results.json"
        with open(h060_path, "w") as f:
            json.dump(h060_results, f, indent=2)
        print(f"\nh=0.60 survival results saved to {h060_path}")

        # Find best candidate
        best_candidate = max(h060_results, key=lambda r: r["mean_survival_time"])
        print(f"\nBest candidate: {best_candidate['candidate']}")
        print(f"  Survival: {best_candidate['mean_survival_time']:.2f}s")
        print(f"  Fall rate: {best_candidate['fall_rate']:.2f}")

    # Phase 5: Full validation
    if args.run_full:
        print("\n" + "=" * 80)
        print("Phase 5: Full Validation")
        print("=" * 80)

        # Select best 1-2 candidates from h=0.60 results
        if not args.skip_h060:
            sorted_candidates = sorted(h060_results, key=lambda r: r["mean_survival_time"], reverse=True)
            best_names = [sorted_candidates[0]["candidate"]]
            if len(sorted_candidates) > 1 and sorted_candidates[1]["mean_survival_time"] > 0.86:
                best_names.append(sorted_candidates[1]["candidate"])

            selected_candidates = [c for c in candidates if c.name in best_names]
        else:
            selected_candidates = candidates

        heights = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
        full_results = []

        for candidate in selected_candidates:
            print(f"\nFull validation: {candidate.name}...")
            result = run_full_validation(candidate, heights, 5, args.output_dir, args.seed)
            full_results.append(result)
            print(f"  Overall survival: {result['overall_mean_survival']:.2f}s")
            print(f"  Overall fall rate: {result['overall_fall_rate']:.2f}")

        # Save results
        full_path = args.output_dir / "full_validation_results.json"
        with open(full_path, "w") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nFull validation results saved to {full_path}")

        # Check if any candidate beats reset-fixed baseline
        baseline_survival = 3.8167
        for result in full_results:
            if result["overall_mean_survival"] > baseline_survival:
                print(f"\n{'='*80}")
                print(f"SUCCESS: {result['candidate']} beats reset-fixed baseline!")
                print(f"  Candidate: {result['overall_mean_survival']:.2f}s")
                print(f"  Baseline: {baseline_survival:.2f}s")
                print(f"  Improvement: +{(result['overall_mean_survival']/baseline_survival - 1)*100:.1f}%")
                print(f"{'='*80}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
