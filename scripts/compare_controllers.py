"""
Compare multiple controllers (LQR/IK, pure PPO, residual PPO) across scenarios.

Usage:
  python scripts/compare_controllers.py \
      --lqr-ik outputs/balance/lqr_ik \
      --pure-ppo outputs/pure_ppo_baseline_50M/seed42/eval \
      --residual-ppo outputs/residual_main_50M/seed*/eval \
      --output-dir outputs/paper_results

Aggregates eval_results.json from multiple checkpoints and produces:
- controller_comparison.csv: side-by-side comparison table
- mean_std_tables.csv: mean ± std across seeds for residual PPO
- scenario_breakdown.csv: per-scenario detailed metrics

For paper Tables II, V-XI.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Compare controllers for paper tables.")
console = Console()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ControllerResult:
    """Results for one controller across scenarios."""

    name: str
    scenarios: dict[str, dict] = field(default_factory=dict)  # scenario -> metrics


def _load_eval_results(path: Path) -> dict | None:
    """Load eval_results.json from a directory."""
    json_path = path / "eval_results.json"
    if not json_path.exists():
        return None
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _aggregate_seeds(seed_dirs: list[Path], scenario: str) -> dict:
    """Aggregate metrics across multiple seeds for one scenario."""
    all_metrics = []

    for seed_dir in seed_dirs:
        data = _load_eval_results(seed_dir)
        if not data:
            continue

        # Find matching scenario in results
        for result in data.get("results", []):
            if result.get("scenario") == scenario:
                all_metrics.append(result)
                break

    if not all_metrics:
        return {}

    # Compute mean ± std for each metric
    def _mean_std(key: str) -> tuple[float, float]:
        vals = [m.get(key, float("nan")) for m in all_metrics]
        vals = [v for v in vals if not math.isnan(v)]
        if not vals:
            return (float("nan"), float("nan"))
        return (float(np.mean(vals)), float(np.std(vals)))

    aggregated = {
        "scenario": scenario,
        "num_seeds": len(all_metrics),
    }

    # Aggregate all numeric metrics
    metric_keys = [
        "fall_rate",
        "survival_rate",
        "survival_time_mean_s",
        "pitch_rms_deg",
        "roll_rms_deg",
        "pitch_rate_rms_rads",
        "xy_drift_max_m",
        "height_rmse_m",
        "wheel_speed_rms_rads",
        "torque_rms_nm",
        "recovery_time_s",
        "max_recoverable_push_n",
        "base_action_rms",
        "residual_action_rms",
        "final_action_rms",
        "residual_norm_mean",
        "residual_saturation_rate",
        "residual_to_base_ratio",
    ]

    for key in metric_keys:
        mean, std = _mean_std(key)
        aggregated[f"{key}_mean"] = mean
        aggregated[f"{key}_std"] = std

    return aggregated


@app.command()
def compare(
    lqr_ik: str = typer.Option("", help="Path to LQR/IK eval results directory."),
    pure_ppo: str = typer.Option("", help="Path to pure PPO eval results directory."),
    residual_ppo: str = typer.Option(
        "",
        help="Path pattern to residual PPO eval results (supports glob, e.g., outputs/residual_main_50M/seed*/eval).",
    ),
    output_dir: str = typer.Option("outputs/paper_results", help="Output directory for comparison tables."),
    scenarios: list[str] = typer.Option(
        ["nominal", "random_height", "push_recovery", "friction_low", "friction_high"],
        help="Scenarios to compare. Repeat flag for multiple.",
    ),
) -> None:
    """Compare controllers and generate paper-ready tables."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Controller Comparison[/bold cyan]")
    console.print(f"  Output: {out_dir}\n")

    # Load controller results
    controllers: dict[str, ControllerResult] = {}

    # LQR/IK baseline
    if lqr_ik:
        lqr_ik_path = Path(lqr_ik)
        if lqr_ik_path.exists():
            data = _load_eval_results(lqr_ik_path)
            if data:
                ctrl = ControllerResult(name="LQR/IK Prior")
                for result in data.get("results", []):
                    scenario = result.get("scenario")
                    if scenario in scenarios:
                        ctrl.scenarios[scenario] = result
                controllers["lqr_ik"] = ctrl
                console.print(f"  [green]✓[/green] Loaded LQR/IK: {len(ctrl.scenarios)} scenarios")

    # Pure PPO baseline
    if pure_ppo:
        pure_ppo_path = Path(pure_ppo)
        if pure_ppo_path.exists():
            data = _load_eval_results(pure_ppo_path)
            if data:
                ctrl = ControllerResult(name="Pure PPO")
                for result in data.get("results", []):
                    scenario = result.get("scenario")
                    if scenario in scenarios:
                        ctrl.scenarios[scenario] = result
                controllers["pure_ppo"] = ctrl
                console.print(f"  [green]✓[/green] Loaded Pure PPO: {len(ctrl.scenarios)} scenarios")

    # Residual PPO (aggregate across seeds)
    if residual_ppo:
        import glob

        seed_dirs = [Path(p) for p in glob.glob(residual_ppo)]
        if seed_dirs:
            ctrl = ControllerResult(name="Residual PPO")
            for scenario in scenarios:
                aggregated = _aggregate_seeds(seed_dirs, scenario)
                if aggregated:
                    ctrl.scenarios[scenario] = aggregated
            controllers["residual_ppo"] = ctrl
            console.print(
                f"  [green]✓[/green] Loaded Residual PPO: {len(seed_dirs)} seeds, {len(ctrl.scenarios)} scenarios"
            )

    if not controllers:
        console.print("[red]No controller results found. Check paths.[/red]")
        raise typer.Exit(1)

    # ── Generate comparison table ─────────────────────────────────────────────

    comparison_rows = []
    for scenario in scenarios:
        row = {"scenario": scenario}

        for ctrl_key, ctrl in controllers.items():
            metrics = ctrl.scenarios.get(scenario, {})

            # Core metrics
            row[f"{ctrl_key}_survival_rate"] = metrics.get("survival_rate", float("nan"))
            row[f"{ctrl_key}_fall_rate"] = metrics.get("fall_rate", float("nan"))
            row[f"{ctrl_key}_height_rmse_m"] = metrics.get("height_rmse_m", float("nan"))
            row[f"{ctrl_key}_pitch_rms_deg"] = metrics.get("pitch_rms_deg", float("nan"))
            row[f"{ctrl_key}_roll_rms_deg"] = metrics.get("roll_rms_deg", float("nan"))
            row[f"{ctrl_key}_xy_drift_max_m"] = metrics.get("xy_drift_max_m", float("nan"))
            row[f"{ctrl_key}_torque_rms_nm"] = metrics.get("torque_rms_nm", float("nan"))
            row[f"{ctrl_key}_recovery_time_s"] = metrics.get("recovery_time_s", float("nan"))
            row[f"{ctrl_key}_max_recoverable_push_n"] = metrics.get("max_recoverable_push_n", float("nan"))

            # Residual-specific (only for residual PPO)
            if ctrl_key == "residual_ppo":
                row[f"{ctrl_key}_base_action_rms"] = metrics.get("base_action_rms_mean", float("nan"))
                row[f"{ctrl_key}_residual_action_rms"] = metrics.get("residual_action_rms_mean", float("nan"))
                row[f"{ctrl_key}_residual_to_base_ratio"] = metrics.get(
                    "residual_to_base_ratio_mean", float("nan")
                )
                row[f"{ctrl_key}_residual_saturation_rate"] = metrics.get(
                    "residual_saturation_rate_mean", float("nan")
                )

        comparison_rows.append(row)

    # Save comparison CSV
    csv_path = out_dir / "controller_comparison.csv"
    if comparison_rows:
        fieldnames = list(comparison_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in comparison_rows:
                # Format NaN as empty string
                for k, v in row.items():
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        row[k] = ""
                writer.writerow(row)
        console.print(f"\n[dim]CSV → {csv_path}[/dim]")

    # ── Generate mean ± std table for residual PPO ────────────────────────────

    if "residual_ppo" in controllers:
        mean_std_rows = []
        for scenario in scenarios:
            metrics = controllers["residual_ppo"].scenarios.get(scenario, {})
            if not metrics:
                continue

            row = {
                "scenario": scenario,
                "num_seeds": metrics.get("num_seeds", 0),
            }

            # Format mean ± std for key metrics
            def _fmt_mean_std(key: str) -> str:
                mean = metrics.get(f"{key}_mean", float("nan"))
                std = metrics.get(f"{key}_std", float("nan"))
                if math.isnan(mean):
                    return ""
                if math.isnan(std):
                    return f"{mean:.3f}"
                return f"{mean:.3f} ± {std:.3f}"

            row["survival_rate"] = _fmt_mean_std("survival_rate")
            row["height_rmse_m"] = _fmt_mean_std("height_rmse_m")
            row["pitch_rms_deg"] = _fmt_mean_std("pitch_rms_deg")
            row["roll_rms_deg"] = _fmt_mean_std("roll_rms_deg")
            row["xy_drift_max_m"] = _fmt_mean_std("xy_drift_max_m")
            row["torque_rms_nm"] = _fmt_mean_std("torque_rms_nm")
            row["recovery_time_s"] = _fmt_mean_std("recovery_time_s")
            row["max_recoverable_push_n"] = _fmt_mean_std("max_recoverable_push_n")
            row["residual_to_base_ratio"] = _fmt_mean_std("residual_to_base_ratio")
            row["residual_saturation_rate"] = _fmt_mean_std("residual_saturation_rate")

            mean_std_rows.append(row)

        mean_std_path = out_dir / "mean_std_tables.csv"
        if mean_std_rows:
            fieldnames = list(mean_std_rows[0].keys())
            with open(mean_std_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(mean_std_rows)
            console.print(f"[dim]CSV → {mean_std_path}[/dim]")

    # ── Display summary table ──────────────────────────────────────────────────

    table = Table(title="Controller Comparison Summary")
    table.add_column("Scenario", style="cyan")
    table.add_column("Controller", style="yellow")
    table.add_column("Survival", justify="right")
    table.add_column("H_RMSE(m)", justify="right")
    table.add_column("Pitch_RMS°", justify="right")
    table.add_column("MaxPush(N)", justify="right")

    for scenario in scenarios:
        for ctrl_key, ctrl in controllers.items():
            metrics = ctrl.scenarios.get(scenario, {})
            if not metrics:
                continue

            survival = metrics.get("survival_rate", metrics.get("survival_rate_mean", float("nan")))
            height_rmse = metrics.get("height_rmse_m", metrics.get("height_rmse_m_mean", float("nan")))
            pitch_rms = metrics.get("pitch_rms_deg", metrics.get("pitch_rms_deg_mean", float("nan")))
            max_push = metrics.get(
                "max_recoverable_push_n", metrics.get("max_recoverable_push_n_mean", float("nan"))
            )

            def _f(v: float, fmt: str = ".3f") -> str:
                if math.isnan(v) or math.isinf(v):
                    return "—"
                return format(v, fmt)

            table.add_row(
                scenario if ctrl_key == list(controllers.keys())[0] else "",
                ctrl.name,
                _f(survival, ".2%"),
                _f(height_rmse),
                _f(pitch_rms, ".2f"),
                _f(max_push, ".1f"),
            )

    console.print(table)
    console.print(f"\n[green]✓[/green] Comparison complete. Results saved to {out_dir}\n")


if __name__ == "__main__":
    app()
