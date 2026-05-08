"""Comprehensive evaluation protocol for Phase B.7 controller comparison.

Compares three controllers:
    1. Baseline: height_scheduled_dynamic_lqr (Phase B.6 adopted)
    2. Candidate: hierarchical_vmc_lqr (Phase B.7 new)
    3. Reference: geometric_lqr (Phase B.5 baseline)

Evaluation protocol:
    - Fixed-height tests: 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40 m
    - Multiple episodes per height for statistical significance
    - Comprehensive metrics: survival, pitch/roll RMS, CoM error, saturation
    - Failure mode classification
    - Decision rule: +20% survival OR -10pp fall rate OR -20% pitch RMS

Usage:
    python scripts/eval_phase_b7_comprehensive.py \
        --baseline height_scheduled_dynamic_lqr \
        --candidate hierarchical_vmc_lqr \
        --heights 0.70 0.65 0.60 0.55 0.50 0.45 0.40 \
        --episodes 20 \
        --output-dir outputs/phase_b7_eval
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import jax
import mujoco
import numpy as np
from rich.console import Console
from rich.table import Table

from wheeled_biped.controllers.lqr_ik_prior import LQRIKConfig, LQRIKPrior
from wheeled_biped.controllers.hierarchical_vmc_lqr import (
    HierarchicalVMCConfig,
    HierarchicalVMCController,
)
from wheeled_biped.envs.balance_env import BalanceEnv
from wheeled_biped.utils.config import get_model_path

console = Console()


def to_jsonable(obj):
    """Convert numpy/JAX types to JSON-serializable Python types."""
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return str(obj)


def evaluate_controller(
    controller,
    env: BalanceEnv,
    height: float,
    num_episodes: int,
    max_time: float = 10.0,
    rng_seed: int = 42,
) -> Dict:
    """Evaluate a controller at fixed height."""
    rng = jax.random.PRNGKey(rng_seed)
    max_steps = int(max_time / env.CONTROL_DT)

    survival_times = []
    pitch_rms_list = []
    roll_rms_list = []
    com_error_rms_list = []
    wheel_sat_durations = []
    falls = []

    for ep in range(num_episodes):
        rng, reset_key = jax.random.split(rng)
        state = env.reset(reset_key)

        # Override height
        obs = state.obs.at[39].set(height)
        state = state._replace(obs=obs)
        controller.reset(height_cmd_m=height)

        episode_pitches = []
        episode_rolls = []
        episode_com_errors = []
        episode_saturations = []

        for step in range(max_steps):
            obs_np = np.array(state.obs)
            action_np = controller.compute_action(obs_np)
            action = jax.numpy.array(action_np)

            state = env.step(state, action)

            # Metrics
            g_body = state.obs[0:3]
            pitch = -np.arcsin(np.clip(g_body[1], -1.0, 1.0))
            roll = np.arcsin(np.clip(g_body[0], -1.0, 1.0))

            episode_pitches.append(np.rad2deg(pitch))
            episode_rolls.append(np.rad2deg(roll))

            # CoM error (if controller supports it)
            if hasattr(controller, '_compute_com_y'):
                qpos = obs_np[9:19]
                com_y = controller._compute_com_y(qpos)
                wheel_y = controller._compute_wheel_contact_y(qpos)
                com_error = abs(com_y - wheel_y)
                episode_com_errors.append(com_error)

            # Saturation
            saturated = np.mean(np.abs(action) >= 0.999)
            episode_saturations.append(saturated)

            if state.done:
                break

        survival_time = (step + 1) * env.CONTROL_DT
        survival_times.append(survival_time)
        falls.append(survival_time < max_time - 0.01)

        if len(episode_pitches) > 0:
            pitch_rms_list.append(np.sqrt(np.mean(np.array(episode_pitches) ** 2)))
            roll_rms_list.append(np.sqrt(np.mean(np.array(episode_rolls) ** 2)))

            if episode_com_errors:
                com_error_rms_list.append(np.sqrt(np.mean(np.array(episode_com_errors) ** 2)))

            wheel_sat_duration = sum(episode_saturations) * env.CONTROL_DT
            wheel_sat_durations.append(wheel_sat_duration)

    # Aggregate
    metrics = {
        "survival_time_mean": float(np.mean(survival_times)),
        "survival_time_std": float(np.std(survival_times)),
        "fall_rate": float(np.mean(falls)),
        "pitch_rms_deg": float(np.mean(pitch_rms_list)) if pitch_rms_list else 999.0,
        "roll_rms_deg": float(np.mean(roll_rms_list)) if roll_rms_list else 999.0,
        "com_error_rms_m": float(np.mean(com_error_rms_list)) if com_error_rms_list else 999.0,
        "wheel_sat_duration_s": float(np.mean(wheel_sat_durations)) if wheel_sat_durations else 0.0,
    }

    return metrics


def compare_controllers(
    baseline_config_path: Path,
    candidate_config_path: Path,
    reference_config_path: Path,
    heights: List[float],
    num_episodes: int,
    output_dir: Path,
) -> Dict:
    """Compare baseline, candidate, and reference controllers."""
    console.print("[bold]Phase B.7 Comprehensive Evaluation[/bold]\n")

    # Create environment
    model_path = get_model_path()
    mj_model = mujoco.MjModel.from_xml_path(str(model_path))
    env_config = {
        "task": {"name": "balance"},
        "low_level_pid": {
            "enabled": True,
            "disable_pid_action_bias": True,
            "action_smoothing_alpha": 0.5,
            "anti_windup_limit": 0.4,
            "wheel_vel_limit": 20.0,
            "kp": [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0],
            "ki": [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1],
            "kd": [3.0, 2.0, 4.0, 4.0, 0.0, 3.0, 2.0, 4.0, 4.0, 0.0],
            "action_delay_steps": 0,
        },
    }
    env = BalanceEnv(config=env_config)

    # Load controllers
    console.print("[cyan]Loading baseline (height_scheduled_dynamic_lqr)...[/cyan]")
    baseline_config = LQRIKConfig.from_yaml(baseline_config_path)
    baseline = LQRIKPrior(baseline_config, mj_model)

    console.print("[cyan]Loading candidate (hierarchical_vmc_lqr)...[/cyan]")
    candidate_config = HierarchicalVMCConfig.from_yaml(candidate_config_path)
    candidate = HierarchicalVMCController(candidate_config, mj_model)

    console.print("[cyan]Loading reference (geometric_lqr)...[/cyan]")
    reference_config = LQRIKConfig.from_yaml(reference_config_path)
    reference = LQRIKPrior(reference_config, mj_model)

    # Evaluate all controllers
    baseline_results = {}
    candidate_results = {}
    reference_results = {}

    for height in heights:
        console.print(f"\n[yellow]Evaluating at h={height:.2f}m...[/yellow]")

        console.print("  Baseline...")
        baseline_results[height] = evaluate_controller(
            baseline, env, height, num_episodes
        )

        console.print("  Candidate...")
        candidate_results[height] = evaluate_controller(
            candidate, env, height, num_episodes
        )

        console.print("  Reference...")
        reference_results[height] = evaluate_controller(
            reference, env, height, num_episodes
        )

    # Aggregate across heights
    def aggregate(results):
        return {
            "survival_time_mean": np.mean([r["survival_time_mean"] for r in results.values()]),
            "fall_rate": np.mean([r["fall_rate"] for r in results.values()]),
            "pitch_rms_deg": np.mean([r["pitch_rms_deg"] for r in results.values()]),
            "com_error_rms_m": np.mean([r["com_error_rms_m"] for r in results.values()]),
        }

    baseline_agg = aggregate(baseline_results)
    candidate_agg = aggregate(candidate_results)
    reference_agg = aggregate(reference_results)

    # Compute improvements (candidate vs baseline)
    survival_improvement = (
        (candidate_agg["survival_time_mean"] - baseline_agg["survival_time_mean"])
        / baseline_agg["survival_time_mean"]
        * 100.0
    )
    pitch_improvement = (
        (baseline_agg["pitch_rms_deg"] - candidate_agg["pitch_rms_deg"])
        / baseline_agg["pitch_rms_deg"]
        * 100.0
    )
    fall_rate_improvement = (baseline_agg["fall_rate"] - candidate_agg["fall_rate"]) * 100.0

    # Decision logic
    meets_survival = survival_improvement >= 20.0
    meets_pitch = pitch_improvement >= 20.0
    meets_fall_rate = fall_rate_improvement >= 10.0
    decision = meets_survival or meets_pitch or meets_fall_rate

    # Prepare results dict
    results = {
        "baseline_config": str(baseline_config_path),
        "candidate_config": str(candidate_config_path),
        "reference_config": str(reference_config_path),
        "heights": heights,
        "num_episodes": num_episodes,
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "reference_results": reference_results,
        "baseline_aggregate": baseline_agg,
        "candidate_aggregate": candidate_agg,
        "reference_aggregate": reference_agg,
        "improvements": {
            "survival_time_pct": survival_improvement,
            "pitch_rms_pct": pitch_improvement,
            "fall_rate_pp": fall_rate_improvement,
        },
        "criteria_met": {
            "survival_time": meets_survival,
            "pitch_rms": meets_pitch,
            "fall_rate": meets_fall_rate,
        },
        "decision": "ADOPT" if decision else "REJECT",
    }

    # Save JSON results FIRST (before console printing that might crash)
    results_path = output_dir / "phase_b7_comparison.json"
    with open(results_path, "w") as f:
        json.dump(to_jsonable(results), f, indent=2)

    # Save summary CSV
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "Candidate", "Improvement", "Criterion", "Met"])
        writer.writerow([
            "Survival Time (s)",
            f"{baseline_agg['survival_time_mean']:.2f}",
            f"{candidate_agg['survival_time_mean']:.2f}",
            f"{survival_improvement:+.1f}%",
            ">= +20%",
            "Yes" if meets_survival else "No",
        ])
        writer.writerow([
            "Pitch RMS (deg)",
            f"{baseline_agg['pitch_rms_deg']:.1f}",
            f"{candidate_agg['pitch_rms_deg']:.1f}",
            f"{pitch_improvement:+.1f}%",
            ">= +20%",
            "Yes" if meets_pitch else "No",
        ])
        writer.writerow([
            "Fall Rate",
            f"{baseline_agg['fall_rate']:.1%}",
            f"{candidate_agg['fall_rate']:.1%}",
            f"{fall_rate_improvement:+.1f} pp",
            ">= +10 pp",
            "Yes" if meets_fall_rate else "No",
        ])

    # Save per-height comparison CSV
    per_height_path = output_dir / "per_height_comparison.csv"
    with open(per_height_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Height (m)",
            "Baseline Survival (s)",
            "Candidate Survival (s)",
            "Reference Survival (s)",
            "Baseline Pitch RMS (deg)",
            "Candidate Pitch RMS (deg)",
        ])
        for height in heights:
            b = baseline_results[height]
            c = candidate_results[height]
            r = reference_results[height]
            writer.writerow([
                f"{height:.2f}",
                f"{b['survival_time_mean']:.2f}",
                f"{c['survival_time_mean']:.2f}",
                f"{r['survival_time_mean']:.2f}",
                f"{b['pitch_rms_deg']:.1f}",
                f"{c['pitch_rms_deg']:.1f}",
            ])

    # Print results (after saving files)
    console.print("\n[bold green]Evaluation Results[/bold green]\n")

    # Per-height table
    table = Table(title="Per-Height Comparison")
    table.add_column("Height (m)", justify="right")
    table.add_column("Baseline\nSurvival (s)", justify="right")
    table.add_column("Candidate\nSurvival (s)", justify="right")
    table.add_column("Reference\nSurvival (s)", justify="right")
    table.add_column("Baseline\nPitch RMS (°)", justify="right")
    table.add_column("Candidate\nPitch RMS (°)", justify="right")

    for height in heights:
        b = baseline_results[height]
        c = candidate_results[height]
        r = reference_results[height]
        table.add_row(
            f"{height:.2f}",
            f"{b['survival_time_mean']:.2f}",
            f"{c['survival_time_mean']:.2f}",
            f"{r['survival_time_mean']:.2f}",
            f"{b['pitch_rms_deg']:.1f}",
            f"{c['pitch_rms_deg']:.1f}",
        )

    console.print(table)

    # Aggregate comparison
    console.print("\n[bold]Aggregate Comparison[/bold]\n")
    agg_table = Table()
    agg_table.add_column("Metric", justify="left")
    agg_table.add_column("Baseline", justify="right")
    agg_table.add_column("Candidate", justify="right")
    agg_table.add_column("Improvement", justify="right")
    agg_table.add_column("Criterion", justify="right")
    agg_table.add_column("Met?", justify="center")

    agg_table.add_row(
        "Survival Time (s)",
        f"{baseline_agg['survival_time_mean']:.2f}",
        f"{candidate_agg['survival_time_mean']:.2f}",
        f"{survival_improvement:+.1f}%",
        ">= +20%",
        "YES" if meets_survival else "NO",
    )
    agg_table.add_row(
        "Pitch RMS (deg)",
        f"{baseline_agg['pitch_rms_deg']:.1f}",
        f"{candidate_agg['pitch_rms_deg']:.1f}",
        f"{pitch_improvement:+.1f}%",
        ">= +20%",
        "YES" if meets_pitch else "NO",
    )
    agg_table.add_row(
        "Fall Rate",
        f"{baseline_agg['fall_rate']:.1%}",
        f"{candidate_agg['fall_rate']:.1%}",
        f"{fall_rate_improvement:+.1f} pp",
        ">= +10 pp",
        "YES" if meets_fall_rate else "NO",
    )

    console.print(agg_table)

    # Decision
    console.print(f"\n[bold]Decision: {'ADOPT' if decision else 'REJECT'}[/bold]")
    if decision:
        console.print("[green]Hierarchical VMC+LQR meets adoption threshold[/green]")
    else:
        console.print("[yellow]Hierarchical VMC+LQR does NOT meet threshold[/yellow]")

    # Save results
    results = {
        "baseline_config": str(baseline_config_path),
        "candidate_config": str(candidate_config_path),
        "reference_config": str(reference_config_path),
        "heights": heights,
        "num_episodes": num_episodes,
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "reference_results": reference_results,
        "baseline_aggregate": baseline_agg,
        "candidate_aggregate": candidate_agg,
        "reference_aggregate": reference_agg,
        "improvements": {
            "survival_time_pct": survival_improvement,
            "pitch_rms_pct": pitch_improvement,
            "fall_rate_pp": fall_rate_improvement,
        },
        "criteria_met": {
            "survival_time": meets_survival,
            "pitch_rms": meets_pitch,
            "fall_rate": meets_fall_rate,
        },
        "decision": "ADOPT" if decision else "REJECT",
    }

    # Save JSON results
    results_path = output_dir / "phase_b7_comparison.json"
    with open(results_path, "w") as f:
        json.dump(to_jsonable(results), f, indent=2)

    # Save summary CSV
    summary_path = output_dir / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", "Candidate", "Improvement", "Criterion", "Met"])
        writer.writerow([
            "Survival Time (s)",
            f"{baseline_agg['survival_time_mean']:.2f}",
            f"{candidate_agg['survival_time_mean']:.2f}",
            f"{survival_improvement:+.1f}%",
            ">= +20%",
            "Yes" if meets_survival else "No",
        ])
        writer.writerow([
            "Pitch RMS (deg)",
            f"{baseline_agg['pitch_rms_deg']:.1f}",
            f"{candidate_agg['pitch_rms_deg']:.1f}",
            f"{pitch_improvement:+.1f}%",
            ">= +20%",
            "Yes" if meets_pitch else "No",
        ])
        writer.writerow([
            "Fall Rate",
            f"{baseline_agg['fall_rate']:.1%}",
            f"{candidate_agg['fall_rate']:.1%}",
            f"{fall_rate_improvement:+.1f} pp",
            ">= +10 pp",
            "Yes" if meets_fall_rate else "No",
        ])

    # Save per-height comparison CSV
    per_height_path = output_dir / "per_height_comparison.csv"
    with open(per_height_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Height (m)",
            "Baseline Survival (s)",
            "Candidate Survival (s)",
            "Reference Survival (s)",
            "Baseline Pitch RMS (°)",
            "Candidate Pitch RMS (°)",
        ])
        for height in heights:
            b = baseline_results[height]
            c = candidate_results[height]
            r = reference_results[height]
            writer.writerow([
                f"{height:.2f}",
                f"{b['survival_time_mean']:.2f}",
                f"{c['survival_time_mean']:.2f}",
                f"{r['survival_time_mean']:.2f}",
                f"{b['pitch_rms_deg']:.1f}",
                f"{c['pitch_rms_deg']:.1f}",
            ])

    console.print(f"\n[cyan]Results saved to:[/cyan]")
    console.print(f"  - {results_path}")
    console.print(f"  - {summary_path}")
    console.print(f"  - {per_height_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation for Phase B.7"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="height_scheduled_dynamic_lqr",
        help="Baseline controller name",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default="hierarchical_vmc_lqr",
        help="Candidate controller name",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="gain_scheduled_lqr",
        help="Reference controller name (geometric baseline)",
    )
    parser.add_argument(
        "--heights",
        type=float,
        nargs="+",
        default=[0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40],
        help="Heights to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Episodes per height",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase_b7_eval"),
        help="Output directory",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Config paths
    baseline_config = Path(f"configs/controllers/{args.baseline}.yaml")
    candidate_config = Path(f"configs/controllers/{args.candidate}.yaml")
    reference_config = Path(f"configs/controllers/{args.reference}.yaml")

    # Check configs exist
    for config_path in [baseline_config, candidate_config, reference_config]:
        if not config_path.exists():
            console.print(f"[red]Error: Config not found: {config_path}[/red]")
            return

    # Run comparison
    results = compare_controllers(
        baseline_config,
        candidate_config,
        reference_config,
        args.heights,
        args.episodes,
        args.output_dir,
    )

    # Print next steps
    console.print("\n[bold]Next Steps:[/bold]")
    if results["decision"] == "ADOPT":
        console.print("1. Update balance_residual.yaml with hierarchical_vmc_lqr")
        console.print("2. Document Phase B.7 results in report")
        console.print("3. Proceed to Phase D: residual PPO training")
    else:
        console.print("1. Keep height_scheduled_dynamic_lqr as prior")
        console.print("2. Document Phase B.7 findings")
        console.print("3. Consider tuning or proceed to Phase D")


if __name__ == "__main__":
    main()
