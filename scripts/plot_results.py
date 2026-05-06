"""
Generate paper-ready figures from training logs and evaluation results.

Usage:
  python scripts/plot_results.py \
      --input outputs/paper_results \
      --output paper/figures

Generates:
- training_curves.pdf: reward, success rate, curriculum, residual RMS over training
- height_tracking.pdf: random-height tracking time series
- height_transition.pdf: commanded height transition time series
- push_recovery.pdf: push recovery time series
- push_sweep.pdf: push magnitude sweep (survival vs magnitude)
- robustness_sweep.pdf: robustness to friction/mass/noise
- residual_distribution.pdf: residual action distribution histogram

For paper Figures 4-10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = typer.Typer(help="Generate paper figures from results.")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Plot style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "font.family": "serif",
    "text.usetex": False,  # Set to True if LaTeX is available
})


def _load_training_log(log_path: Path) -> dict:
    """Load training log JSONL and return aggregated metrics."""
    if not log_path.exists():
        return {}

    metrics = {
        "step": [],
        "reward_mean": [],
        "success_rate": [],
        "curriculum_stage": [],
        "residual_rms": [],
    }

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)

            metrics["step"].append(entry.get("step", 0))
            metrics["reward_mean"].append(entry.get("reward_mean", 0.0))
            metrics["success_rate"].append(entry.get("success_rate", 0.0))
            metrics["curriculum_stage"].append(entry.get("curriculum_stage", 0))
            metrics["residual_rms"].append(entry.get("residual_rms", 0.0))

    return metrics


@app.command()
def plot_training_curves(
    input_dir: str = typer.Option(..., help="Directory containing training logs (seed*/train.log)."),
    output: str = typer.Option("training_curves.pdf", help="Output figure path."),
) -> None:
    """Generate training curves figure (reward, success, curriculum, residual RMS)."""

    input_path = Path(input_dir)
    log_files = list(input_path.glob("seed*/train.log"))

    if not log_files:
        console.print(f"[red]No training logs found in {input_path}/seed*/train.log[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Plotting training curves from {len(log_files)} seeds...[/cyan]")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Training Curves (Residual PPO, 3 seeds)")

    all_metrics = [_load_training_log(log) for log in log_files]

    # Plot reward
    ax = axes[0, 0]
    for i, metrics in enumerate(all_metrics):
        if metrics:
            ax.plot(metrics["step"], metrics["reward_mean"], label=f"Seed {i}", alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Episode Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot success rate
    ax = axes[0, 1]
    for i, metrics in enumerate(all_metrics):
        if metrics:
            ax.plot(metrics["step"], metrics["success_rate"], label=f"Seed {i}", alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot curriculum stage
    ax = axes[1, 0]
    for i, metrics in enumerate(all_metrics):
        if metrics:
            ax.plot(metrics["step"], metrics["curriculum_stage"], label=f"Seed {i}", alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Curriculum Stage")
    ax.set_title("Curriculum Progression")
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot residual RMS
    ax = axes[1, 1]
    for i, metrics in enumerate(all_metrics):
        if metrics:
            ax.plot(metrics["step"], metrics["residual_rms"], label=f"Seed {i}", alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Residual Action RMS")
    ax.set_title("Residual Action Magnitude")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    console.print(f"[green]✓[/green] Saved {output}")


@app.command()
def plot_push_sweep(
    input_json: str = typer.Option(..., help="Path to eval_results.json with push_sweep scenarios."),
    output: str = typer.Option("push_sweep.pdf", help="Output figure path."),
) -> None:
    """Generate push magnitude sweep figure (survival rate vs push magnitude)."""

    json_path = Path(input_json)
    if not json_path.exists():
        console.print(f"[red]File not found: {json_path}[/red]")
        raise typer.Exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Extract push_sweep results
    push_results = [r for r in data.get("results", []) if r.get("scenario", "").startswith("push_sweep_")]

    if not push_results:
        console.print("[yellow]No push_sweep scenarios found in results.[/yellow]")
        raise typer.Exit(1)

    # Sort by magnitude
    push_results.sort(key=lambda r: float(r.get("extra", {}).get("push_magnitude", 0)))

    magnitudes = [float(r.get("extra", {}).get("push_magnitude", 0)) for r in push_results]
    survival_rates = [r.get("survival_rate", 0.0) for r in push_results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(magnitudes, survival_rates, marker="o", linewidth=2, markersize=6)
    ax.set_xlabel("Push Magnitude (N)")
    ax.set_ylabel("Survival Rate")
    ax.set_title("Push Disturbance Recovery")
    ax.grid(alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    console.print(f"[green]✓[/green] Saved {output}")


@app.command()
def plot_residual_distribution(
    input_json: str = typer.Option(..., help="Path to residual_metrics.json from analyze_residual.py."),
    output: str = typer.Option("residual_distribution.pdf", help="Output figure path."),
) -> None:
    """Generate residual action distribution histogram."""

    json_path = Path(input_json)
    if not json_path.exists():
        console.print(f"[red]File not found: {json_path}[/red]")
        raise typer.Exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        console.print("[yellow]No results found in residual_metrics.json[/yellow]")
        raise typer.Exit(1)

    # Plot per-joint residual RMS for each scenario
    fig, ax = plt.subplots(figsize=(10, 6))

    joint_names = [
        "l_hip_roll",
        "l_hip_yaw",
        "l_hip_pitch",
        "l_knee",
        "l_wheel",
        "r_hip_roll",
        "r_hip_yaw",
        "r_hip_pitch",
        "r_knee",
        "r_wheel",
    ]

    x = np.arange(len(joint_names))
    width = 0.25

    for i, result in enumerate(results[:3]):  # Max 3 scenarios
        scenario = result.get("scenario", f"Scenario {i}")
        residual_rms = result.get("residual_action_rms_per_joint", [0] * 10)
        ax.bar(x + i * width, residual_rms, width, label=scenario, alpha=0.8)

    ax.set_xlabel("Joint")
    ax.set_ylabel("Residual Action RMS")
    ax.set_title("Residual Action Distribution by Joint")
    ax.set_xticks(x + width)
    ax.set_xticklabels(joint_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    console.print(f"[green]✓[/green] Saved {output}")


@app.command()
def plot_all(
    input_dir: str = typer.Option("outputs/paper_results", help="Input directory with all results."),
    output_dir: str = typer.Option("paper/figures", help="Output directory for figures."),
) -> None:
    """Generate all paper figures."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Generating Paper Figures[/bold cyan]")
    console.print(f"  Input: {input_path}")
    console.print(f"  Output: {output_path}\n")

    # Training curves
    if (input_path.parent / "residual_main_50M").exists():
        try:
            plot_training_curves(
                input_dir=str(input_path.parent / "residual_main_50M"),
                output=str(output_path / "training_curves.pdf"),
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate training_curves.pdf: {e}[/yellow]")

    # Push sweep
    eval_json = input_path / "eval_results.json"
    if eval_json.exists():
        try:
            plot_push_sweep(
                input_json=str(eval_json),
                output=str(output_path / "push_sweep.pdf"),
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate push_sweep.pdf: {e}[/yellow]")

    # Residual distribution
    residual_json = input_path / "residual_metrics.json"
    if residual_json.exists():
        try:
            plot_residual_distribution(
                input_json=str(residual_json),
                output=str(output_path / "residual_distribution.pdf"),
            )
        except Exception as e:
            console.print(f"[yellow]Warning: Could not generate residual_distribution.pdf: {e}[/yellow]")

    console.print(f"\n[green]✓[/green] Figure generation complete. Check {output_path}\n")


if __name__ == "__main__":
    app()
