"""
Compare a current benchmark eval result against a saved baseline.

Usage
-----
  python scripts/compare_baseline.py \\
      --baseline baselines/nominal_v1.json \\
      --current  outputs/checkpoints/balance/final/eval_results_nominal.json

  # Exit code 0 = passed, 1 = regressions detected
  python scripts/compare_baseline.py --baseline ... --current ... --save-json diff.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.eval.baseline import compare_files  # noqa: E402

app = typer.Typer(help="Compare benchmark results against a saved baseline.")
console = Console()


@app.command()
def compare(
    baseline: str = typer.Option(..., help="Path to baseline JSON file."),
    current: str = typer.Option(..., help="Path to current eval JSON file."),
    save_json: str = typer.Option("", help="Optional: save comparison JSON to this path."),
    fail_on_regression: bool = typer.Option(
        True, help="Exit with code 1 if any regression is detected (useful for CI)."
    ),
) -> None:
    """Compare two benchmark JSON files and report metric regressions."""
    baseline_path = Path(baseline)
    current_path = Path(current)

    for p, label in [(baseline_path, "baseline"), (current_path, "current")]:
        if not p.exists():
            console.print(f"[red]File not found ({label}): {p}[/red]")
            raise typer.Exit(2)

    result = compare_files(current_path, baseline_path)

    # --- Rich table display ---
    table = Table(
        title=f"Baseline Comparison  [mode={result.mode}]",
        show_lines=False,
    )
    table.add_column("", width=3)
    table.add_column("Metric", style="cyan", min_width=28)
    table.add_column("Baseline", justify="right", style="dim")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")

    # Sort: regressions first, then improvements, then ok
    all_deltas = sorted(
        result.deltas,
        key=lambda d: (not d.is_regression, not d.is_improvement, d.metric),
    )

    for d in all_deltas:
        if d.is_regression:
            sym, delta_style = "❌", "[bold red]"
        elif d.is_improvement:
            sym, delta_style = "⬆️", "[bold green]"
        else:
            sym, delta_style = "  ", ""

        table.add_row(
            sym,
            d.metric,
            f"{d.baseline_value:.4f}",
            f"{d.current_value:.4f}",
            f"{delta_style}{d.delta:+.4f}",
        )

    console.print()
    console.print(table)
    console.print(f"  Baseline: [dim]{baseline}[/dim]")
    console.print(f"  Current:  [dim]{current}[/dim]")

    if result.passed:
        console.print("\n[bold green]✅  PASSED — no regressions detected.[/bold green]")
    else:
        n = len(result.regressions)
        console.print(f"\n[bold red]❌  FAILED — {n} regression(s) detected.[/bold red]")

    # --- Optional JSON output ---
    if save_json:
        out_path = Path(save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[dim]Comparison saved → {out_path}[/dim]")

    if fail_on_regression and not result.passed:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
