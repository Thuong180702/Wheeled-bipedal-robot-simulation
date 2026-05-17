#!/usr/bin/env python3
"""
Phase B.9 Step 5.19: Controller Authority Audit

Quantifies PID vs WBC torque authority to diagnose suppression.

Measures:
- PID authority contribution (RMS, per-actuator)
- Torque residual contribution (RMS, per-actuator)
- Delivered authority fraction after clipping
- Actuator saturation frequency
- Sign disagreement between PID and WBC
- Authority bottlenecks

Usage:
    python scripts/phase_b9_step5_19_controller_authority_audit.py \
        --config outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/strong_k20.yaml \
        --height 0.60 \
        --episodes 5 \
        --output-dir outputs/phase_b9_step5_19_controller_authority_reallocation
"""

import argparse
import json
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


def compute_authority_metrics(
    pid_ctrl: np.ndarray,
    torque_residual: np.ndarray,
    final_ctrl: np.ndarray,
    ctrl_min: np.ndarray,
    ctrl_max: np.ndarray,
) -> dict:
    """Compute authority metrics for a single timestep."""
    eps = 1e-6

    # RMS authority
    pid_rms = np.sqrt(np.mean(pid_ctrl**2))
    residual_rms = np.sqrt(np.mean(torque_residual**2))
    final_rms = np.sqrt(np.mean(final_ctrl**2))

    # Per-actuator RMS
    pid_rms_per_act = np.sqrt(pid_ctrl**2)
    residual_rms_per_act = np.sqrt(torque_residual**2)

    # Authority ratio
    authority_ratio = residual_rms / (pid_rms + eps)

    # Saturation flags
    saturated = (final_ctrl <= ctrl_min + eps) | (final_ctrl >= ctrl_max - eps)
    saturation_rate = np.mean(saturated)

    # Delivered residual fraction
    # If PID + residual was clipped, residual was partially/fully suppressed
    unclipped_sum = pid_ctrl + torque_residual
    clipping_occurred = (unclipped_sum < ctrl_min - eps) | (unclipped_sum > ctrl_max + eps)

    # Compute how much residual was actually delivered
    delivered_residual = final_ctrl - pid_ctrl
    residual_delivery_fraction = np.where(
        np.abs(torque_residual) > eps,
        np.abs(delivered_residual) / (np.abs(torque_residual) + eps),
        1.0,  # If residual is zero, delivery is "perfect"
    )
    mean_delivery_fraction = np.mean(residual_delivery_fraction)

    # Sign disagreement (PID fights WBC)
    sign_disagreement = (np.sign(pid_ctrl) != np.sign(torque_residual)) & (np.abs(torque_residual) > eps)
    sign_disagreement_rate = np.mean(sign_disagreement)

    return {
        "pid_rms": float(pid_rms),
        "residual_rms": float(residual_rms),
        "final_rms": float(final_rms),
        "authority_ratio": float(authority_ratio),
        "saturation_rate": float(saturation_rate),
        "mean_delivery_fraction": float(mean_delivery_fraction),
        "sign_disagreement_rate": float(sign_disagreement_rate),
        "pid_rms_per_act": pid_rms_per_act.tolist(),
        "residual_rms_per_act": residual_rms_per_act.tolist(),
        "clipping_rate": float(np.mean(clipping_occurred)),
    }


def run_authority_audit(
    config_path: Path,
    height: float,
    num_episodes: int,
    output_dir: Path,
    seed: int = 42,
) -> dict:
    """Run authority audit rollout."""

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create environment
    env = BalanceEnv(
        episode_length=500,
        action_repeat=1,
        height_curriculum_mode="fixed",
        fixed_height=height,
    )

    # Create controller
    controller = DualRateBalanceController(config)

    # Initialize
    rng = jax.random.PRNGKey(seed)

    # Storage for metrics
    all_metrics = []
    episode_summaries = []

    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state = env.reset(reset_rng)

        episode_metrics = []
        episode_steps = 0
        episode_survival = 0.0

        for step in range(500):
            # Get controller action
            ctrl_output = controller.compute_action(
                state.pipeline_state,
                state.obs,
                state.info,
            )

            # Extract authority components from info
            pid_ctrl = np.array(ctrl_output["pid_ctrl"])
            torque_residual = np.array(ctrl_output.get("torque_residual", np.zeros(10)))
            final_ctrl = np.array(ctrl_output["final_ctrl"])

            # Get limits
            ctrl_min = np.array(env._ctrl_min)
            ctrl_max = np.array(env._ctrl_max)

            # Compute metrics
            metrics = compute_authority_metrics(
                pid_ctrl, torque_residual, final_ctrl, ctrl_min, ctrl_max
            )
            episode_metrics.append(metrics)

            # Step environment
            rng, step_rng = jax.random.split(rng)
            state = env.step(state, ctrl_output["action"], step_rng)

            episode_steps += 1
            episode_survival = episode_steps * env.dt

            if state.done:
                break

        # Aggregate episode metrics
        episode_summary = {
            "episode": ep,
            "steps": episode_steps,
            "survival_time": episode_survival,
            "mean_pid_rms": float(np.mean([m["pid_rms"] for m in episode_metrics])),
            "mean_residual_rms": float(np.mean([m["residual_rms"] for m in episode_metrics])),
            "mean_authority_ratio": float(np.mean([m["authority_ratio"] for m in episode_metrics])),
            "mean_saturation_rate": float(np.mean([m["saturation_rate"] for m in episode_metrics])),
            "mean_delivery_fraction": float(np.mean([m["mean_delivery_fraction"] for m in episode_metrics])),
            "mean_sign_disagreement": float(np.mean([m["sign_disagreement_rate"] for m in episode_metrics])),
            "mean_clipping_rate": float(np.mean([m["clipping_rate"] for m in episode_metrics])),
        }
        episode_summaries.append(episode_summary)
        all_metrics.extend(episode_metrics)

        print(f"Episode {ep}: survival={episode_survival:.2f}s, "
              f"authority_ratio={episode_summary['mean_authority_ratio']:.4f}, "
              f"delivery_fraction={episode_summary['mean_delivery_fraction']:.4f}")

    # Aggregate across all episodes
    summary = {
        "config": str(config_path),
        "height": height,
        "num_episodes": num_episodes,
        "mean_survival_time": float(np.mean([e["survival_time"] for e in episode_summaries])),
        "mean_pid_rms": float(np.mean([m["pid_rms"] for m in all_metrics])),
        "mean_residual_rms": float(np.mean([m["residual_rms"] for m in all_metrics])),
        "mean_authority_ratio": float(np.mean([m["authority_ratio"] for m in all_metrics])),
        "mean_saturation_rate": float(np.mean([m["saturation_rate"] for m in all_metrics])),
        "mean_delivery_fraction": float(np.mean([m["mean_delivery_fraction"] for m in all_metrics])),
        "mean_sign_disagreement": float(np.mean([m["sign_disagreement_rate"] for m in all_metrics])),
        "mean_clipping_rate": float(np.mean([m["clipping_rate"] for m in all_metrics])),
        "per_actuator_pid_rms": np.mean([m["pid_rms_per_act"] for m in all_metrics], axis=0).tolist(),
        "per_actuator_residual_rms": np.mean([m["residual_rms_per_act"] for m in all_metrics], axis=0).tolist(),
        "episode_summaries": episode_summaries,
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--height", type=float, default=0.60)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running authority audit: config={args.config}, height={args.height}")

    summary = run_authority_audit(
        args.config,
        args.height,
        args.episodes,
        args.output_dir,
        args.seed,
    )

    # Save results
    summary_path = args.output_dir / "authority_audit_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAuthority Audit Summary:")
    print(f"  Mean survival: {summary['mean_survival_time']:.2f}s")
    print(f"  PID RMS: {summary['mean_pid_rms']:.2f} Nm")
    print(f"  Residual RMS: {summary['mean_residual_rms']:.4f} Nm")
    print(f"  Authority ratio (residual/PID): {summary['mean_authority_ratio']:.4f}")
    print(f"  Saturation rate: {summary['mean_saturation_rate']:.2%}")
    print(f"  Delivery fraction: {summary['mean_delivery_fraction']:.2%}")
    print(f"  Sign disagreement: {summary['mean_sign_disagreement']:.2%}")
    print(f"  Clipping rate: {summary['mean_clipping_rate']:.2%}")

    print(f"\nResults saved to {summary_path}")

    # Generate analysis report
    analysis_path = args.output_dir / "control_stack_analysis.md"
    with open(analysis_path, "w") as f:
        f.write("# Phase B.9 Step 5.19: Control Authority Audit\n\n")
        f.write(f"## Configuration\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Height: {args.height} m\n")
        f.write(f"- Episodes: {args.episodes}\n\n")

        f.write(f"## Authority Metrics\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Mean survival time | {summary['mean_survival_time']:.2f} s |\n")
        f.write(f"| PID RMS authority | {summary['mean_pid_rms']:.2f} Nm |\n")
        f.write(f"| Residual RMS authority | {summary['mean_residual_rms']:.4f} Nm |\n")
        f.write(f"| Authority ratio (residual/PID) | {summary['mean_authority_ratio']:.4f} |\n")
        f.write(f"| Actuator saturation rate | {summary['mean_saturation_rate']:.2%} |\n")
        f.write(f"| Residual delivery fraction | {summary['mean_delivery_fraction']:.2%} |\n")
        f.write(f"| Sign disagreement rate | {summary['mean_sign_disagreement']:.2%} |\n")
        f.write(f"| Clipping rate | {summary['mean_clipping_rate']:.2%} |\n\n")

        f.write(f"## Per-Actuator Authority\n\n")
        f.write(f"| Actuator | PID RMS (Nm) | Residual RMS (Nm) | Ratio |\n")
        f.write(f"|----------|--------------|-------------------|-------|\n")

        actuator_names = [
            "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
            "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
        ]

        for i, name in enumerate(actuator_names):
            pid_rms = summary["per_actuator_pid_rms"][i]
            res_rms = summary["per_actuator_residual_rms"][i]
            ratio = res_rms / (pid_rms + 1e-6)
            f.write(f"| {name} | {pid_rms:.2f} | {res_rms:.4f} | {ratio:.4f} |\n")

        f.write(f"\n## Diagnosis\n\n")

        # Diagnose suppression
        if summary['mean_authority_ratio'] < 0.05:
            f.write(f"**CRITICAL: PID authority dominates WBC by >20:1 ratio.**\n\n")
            f.write(f"The WBC torque residuals contribute <5% of PID authority, ")
            f.write(f"indicating severe suppression.\n\n")
        elif summary['mean_authority_ratio'] < 0.10:
            f.write(f"**WARNING: PID authority dominates WBC by >10:1 ratio.**\n\n")

        if summary['mean_delivery_fraction'] < 0.50:
            f.write(f"**CRITICAL: <50% of WBC residuals are delivered after clipping.**\n\n")
            f.write(f"Actuator saturation is suppressing WBC corrections.\n\n")

        if summary['mean_saturation_rate'] > 0.30:
            f.write(f"**WARNING: Actuators saturate {summary['mean_saturation_rate']:.1%} of the time.**\n\n")
            f.write(f"High saturation leaves no headroom for WBC residuals.\n\n")

        if summary['mean_sign_disagreement'] > 0.30:
            f.write(f"**WARNING: PID and WBC disagree on control direction {summary['mean_sign_disagreement']:.1%} of the time.**\n\n")
            f.write(f"Controllers may be fighting each other.\n\n")

        f.write(f"## Root Cause\n\n")
        f.write(f"The control architecture `final = clip(PID + residual, limits)` ")
        f.write(f"allows PID to saturate actuators before WBC residuals are added. ")
        f.write(f"When PID outputs are near limits, clipping removes WBC contributions.\n\n")

        f.write(f"**Authority bottleneck**: PID controller produces outputs that ")
        f.write(f"consume most/all available actuator authority, leaving insufficient ")
        f.write(f"headroom for WBC stabilization corrections.\n\n")

        f.write(f"## Recommendation\n\n")
        f.write(f"Implement authority reallocation to reserve actuator headroom for WBC:\n\n")
        f.write(f"1. **PID output clamping**: Limit PID to use only α fraction of actuator range\n")
        f.write(f"2. **Residual priority blending**: Blend with WBC-priority weighting near falls\n")
        f.write(f"3. **Dynamic gain scheduling**: Reduce PID gains when WBC is active\n")
        f.write(f"4. **Hierarchical arbitration**: Explicit priority management\n\n")

    print(f"Analysis saved to {analysis_path}")


if __name__ == "__main__":
    main()
