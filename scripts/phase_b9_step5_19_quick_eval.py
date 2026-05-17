#!/usr/bin/env python3
"""
Phase B.9 Step 5.19: Authority Reallocation Quick Evaluation

Simplified evaluation that works with the actual env/controller architecture.
Tests PID output clamping at h=0.60 to see if reserving actuator headroom
for WBC improves stability.

Usage:
    python scripts/phase_b9_step5_19_quick_eval.py
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from wheeled_biped.controllers.dual_rate_balance_controller import (
    DualRateBalanceController,
)
from wheeled_biped.envs.balance_env import BalanceEnv


def evaluate_candidate(
    pid_authority_fraction: float,
    num_episodes: int = 5,
    height: float = 0.60,
    seed: int = 42,
) -> dict:
    """Evaluate a single authority reallocation candidate."""

    # Load WBC gains from Step 5.18c best config
    wbc_config_path = Path("outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/best_torque_gain_config.yaml")
    with open(wbc_config_path) as f:
        wbc_cfg = yaml.safe_load(f)

    # Add pid_authority_fraction
    wbc_cfg["pid_authority_fraction"] = pid_authority_fraction

    # Create environment with authority reallocation
    env_config = {
        "low_level_control": {
            "mode": "hybrid_pid_plus_torque",
            "torque_control": wbc_cfg,
        }
    }

    env = BalanceEnv(
        episode_length=500,
        action_repeat=1,
        height_curriculum_mode="fixed",
        fixed_height=height,
        config=env_config,
    )

    # Load controller config
    controller_config_path = Path("configs/controllers/dual_rate_balance_controller_b9.yaml")
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

    summary = {
        "pid_authority_fraction": pid_authority_fraction,
        "height": height,
        "num_episodes": num_episodes,
        "mean_survival_time": float(np.mean([r["survival_time"] for r in episode_results])),
        "fall_rate": float(np.mean([r["fell"] for r in episode_results])),
        "mean_pitch_rms_deg": float(np.mean([r["pitch_rms_deg"] for r in episode_results])),
        "mean_roll_rms_deg": float(np.mean([r["roll_rms_deg"] for r in episode_results])),
        "episodes": episode_results,
    }

    return summary


def main():
    output_dir = Path("outputs/phase_b9_step5_19_controller_authority_reallocation")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Phase B.9 Step 5.19: Authority Reallocation Quick Evaluation")
    print("=" * 80)
    print("\nBaseline (Step 5.18c):")
    print("  h=0.60: survival=0.86s, fall_rate=0.80, roll_rms=15.9deg")
    print("  pid_authority_fraction=1.0 (no clamping)")
    print("\nTesting authority reallocation candidates...")

    # Test candidates
    candidates = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    results = []

    for frac in candidates:
        print(f"\n{'='*80}")
        print(f"Testing pid_authority_fraction={frac}")
        print(f"{'='*80}")

        result = evaluate_candidate(frac, num_episodes=5, height=0.60, seed=42)
        results.append(result)

        print(f"\nResults:")
        print(f"  Mean survival: {result['mean_survival_time']:.2f}s")
        print(f"  Fall rate: {result['fall_rate']:.2f}")
        print(f"  Pitch RMS: {result['mean_pitch_rms_deg']:.2f}deg")
        print(f"  Roll RMS: {result['mean_roll_rms_deg']:.2f}deg")

        # Compare to baseline
        baseline_survival = 0.86
        if result['mean_survival_time'] > baseline_survival:
            improvement = (result['mean_survival_time'] / baseline_survival - 1) * 100
            print(f"  ✓ IMPROVEMENT: +{improvement:.1f}% vs baseline")
        else:
            degradation = (1 - result['mean_survival_time'] / baseline_survival) * 100
            print(f"  ✗ DEGRADATION: -{degradation:.1f}% vs baseline")

    # Save results
    results_path = output_dir / "quick_eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")

    # Find best candidate
    best = max(results, key=lambda r: r["mean_survival_time"])
    print(f"\nBest candidate: pid_authority_fraction={best['pid_authority_fraction']}")
    print(f"  Survival: {best['mean_survival_time']:.2f}s")
    print(f"  Fall rate: {best['fall_rate']:.2f}")
    print(f"  Roll RMS: {best['mean_roll_rms_deg']:.2f}deg")

    baseline_survival = 0.86
    if best['mean_survival_time'] > baseline_survival:
        improvement = (best['mean_survival_time'] / baseline_survival - 1) * 100
        print(f"  Improvement: +{improvement:.1f}% vs baseline")
    else:
        print(f"  Did not beat baseline ({baseline_survival:.2f}s)")

    print(f"\nResults saved to {results_path}")

    # Decision
    print(f"\n{'='*80}")
    print("Decision")
    print(f"{'='*80}")

    if best['mean_survival_time'] > baseline_survival * 1.1:
        print("\nPID_AUTHORITY_CLAMPING_IMPROVES_STABILITY")
        print(f"Authority reallocation (fraction={best['pid_authority_fraction']}) ")
        print(f"improves h=0.60 survival by {(best['mean_survival_time']/baseline_survival-1)*100:.1f}%.")
        print("\nRecommendation: Proceed to full validation across all heights.")
    elif best['mean_survival_time'] > baseline_survival:
        print("\nMARGINAL_IMPROVEMENT")
        print(f"Authority reallocation shows small improvement (+{(best['mean_survival_time']/baseline_survival-1)*100:.1f}%).")
        print("May not be sufficient to pass Step 6 gate.")
    else:
        print("\nPID_AUTHORITY_CLAMPING_INEFFECTIVE")
        print("Authority reallocation does not improve stability at h=0.60.")
        print("\nRoot cause remains: PID authority dominates WBC corrections.")
        print("Clamping PID output alone is insufficient.")
        print("\nAlternative approaches needed:")
        print("  - Dynamic gain scheduling (reduce PID gains when WBC active)")
        print("  - Hierarchical arbitration (explicit WBC priority near falls)")
        print("  - Frequency separation (low-freq PID, high-freq WBC)")
        print("  - Architectural redesign (question PID+WBC hybrid viability)")


if __name__ == "__main__":
    main()
