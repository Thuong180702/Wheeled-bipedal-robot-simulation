"""
Export training logs and benchmark results into paper-ready formats.

Two sub-commands:

  curves   -- Read a JSONL metrics log → CSV + optional PNG figure.
  table    -- Read a benchmark/eval JSON → Markdown table.
  latex    -- Read eval_balance.py JSON → LaTeX booktabs table.

Usage examples:

  # Training curves from log file (new output layout)
  python scripts/export_results.py curves \\
      outputs/balance/rl/seed42/balance_seed42_metrics.jsonl \\
      --tags reward/mean curriculum/level curriculum/eval_per_step \\
      --output outputs/balance/rl/seed42/training_curves.png

  # Aggregate 3-seed training curves into one figure
  python scripts/export_results.py curves \\
      outputs/balance/rl/seed42/balance_seed42_metrics.jsonl \\
      --tags reward/mean --output outputs/balance/rl/paper/seed42_curves.png

  # Benchmark table from evaluate.py output
  python scripts/export_results.py table \\
      outputs/balance/rl/seed42/checkpoints/final/eval_results_command_tracking.json \\
      --output outputs/balance/rl/seed42/tables/height_tracking.md

  # LaTeX table from eval_balance.py output
  python scripts/export_results.py latex \\
      outputs/balance/rl/paper_eval/eval_results.json \\
      --output outputs/tables/balance_eval.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Avoid UnicodeEncodeError in Windows shells that default to cp1252.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL metrics file.  Each line is {step, tag, value}."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _pivot_records(records: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Group records by tag → list of (step, value) sorted by step."""
    by_tag: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for r in records:
        by_tag[r["tag"]].append((int(r["step"]), float(r["value"])))
    for tag in by_tag:
        by_tag[tag].sort()
    return dict(by_tag)


# ---------------------------------------------------------------------------
# Sub-command: curves
# ---------------------------------------------------------------------------


def cmd_curves(args: argparse.Namespace) -> None:
    """Export training curves to CSV and (optionally) a PNG figure."""
    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    records = _load_jsonl(src)
    by_tag = _pivot_records(records)

    # Resolve requested tags (default: reward/mean + curriculum metrics)
    default_tags = [
        "reward/mean",
        "curriculum/level",
        "curriculum/eval_per_step",
        "curriculum/eval_success_rate",
        "curriculum/min_height",
    ]
    tags = args.tags if args.tags else default_tags
    # Keep only tags that actually exist in the log
    available = [t for t in tags if t in by_tag]
    if not available:
        all_tags = sorted(by_tag.keys())
        print("WARNING: none of the requested tags found.  Available tags:")
        for t in all_tags:
            print(f"  {t}")
        sys.exit(1)

    # ── Write CSV ────────────────────────────────────────────────────────────
    # Merge all (step, tag, value) into one sorted table.
    csv_out = Path(args.output).with_suffix(".csv") if args.output else src.with_suffix(".csv")
    csv_out.parent.mkdir(parents=True, exist_ok=True)

    # Build step-indexed dict: step → {tag: value}
    step_data: dict[int, dict[str, float]] = defaultdict(dict)
    for tag in available:
        for step, val in by_tag[tag]:
            step_data[step][tag] = val

    sorted_steps = sorted(step_data.keys())
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step"] + available)
        writer.writeheader()
        for step in sorted_steps:
            row = {"step": step}
            row.update(step_data[step])
            writer.writerow(row)

    print(f"CSV written → {csv_out}  ({len(sorted_steps)} rows, {len(available)} tags)")

    # ── Write PNG figure ──────────────────────────────────────────────────────
    if args.no_plot:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure.  pip install matplotlib")
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, tag in zip(axes, available):
        steps, vals = zip(*by_tag[tag])
        ax.plot(steps, vals, linewidth=1.0)
        ax.set_ylabel(tag.replace("/", "\n"), fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Training step")
    fig.suptitle("Training curves", y=1.01)
    fig.tight_layout()

    fig_out = Path(args.output) if args.output else src.with_suffix(".png")
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure written → {fig_out}")


# ---------------------------------------------------------------------------
# Sub-command: table
# ---------------------------------------------------------------------------


def cmd_table(args: argparse.Namespace) -> None:
    """Export a benchmark/eval JSON result as a Markdown table."""
    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    mode = data.get("mode", "unknown")
    lines: list[str] = []

    lines.append(f"## Benchmark results — mode: `{mode}`\n")

    # ── Top-level scalar metrics ──────────────────────────────────────────────
    scalar_keys = [
        ("num_episodes", "Episodes"),
        ("reward_mean", "Reward mean"),
        ("reward_std", "Reward std"),
        ("reward_p5", "Reward p5"),
        ("reward_p50", "Reward p50 (median)"),
        ("reward_p95", "Reward p95"),
        ("episode_length_mean", "Episode length (mean)"),
        ("success_rate", "Success rate"),
        ("fall_rate", "Fall rate"),
        ("timeout_rate", "Timeout rate"),
    ]

    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for key, label in scalar_keys:
        if key in data:
            val = data[key]
            if isinstance(val, float):
                if key in ("success_rate", "fall_rate", "timeout_rate"):
                    fmt = f"{val:.1%}"
                else:
                    fmt = f"{val:.4f}"
            else:
                fmt = str(val)
            lines.append(f"| {label} | {fmt} |")

    # ── Mode-specific extras ──────────────────────────────────────────────────
    mode_metrics = data.get("mode_metrics", {})

    # command_tracking: per-command table
    if "per_command" in mode_metrics:
        lines.append("")
        lines.append("### Per-command height tracking\n")
        lines.append(  # noqa: E501
            "| Height command (m) | Height RMSE (m) | Success rate | Fall rate | Reward mean |"
        )
        lines.append(
            "|--------------------|-----------------|--------------|-----------|-------------|"
        )
        for row in mode_metrics["per_command"]:
            h = row.get("height_command", "—")
            rmse = row.get("height_rmse", float("nan"))
            sr = row.get("success_rate", float("nan"))
            fr = row.get("fall_rate", float("nan"))
            rm = row.get("reward_mean", float("nan"))
            lines.append(f"| {h:.3f} | {rmse:.4f} | {sr:.1%} | {fr:.1%} | {rm:.4f} |")
        if "overall_height_rmse" in mode_metrics:
            lines.append(f"\n**Overall height RMSE:** {mode_metrics['overall_height_rmse']:.4f} m")

    # push_recovery extras
    for key in ("push_magnitude_used", "fall_after_push_rate", "mean_steps_to_fall"):
        if key in mode_metrics:
            val = mode_metrics[key]
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                fmt = f"{val:.3f}" if key != "fall_after_push_rate" else f"{val:.1%}"
            else:
                fmt = str(val)
            lines.append(f"\n**{label}:** {fmt}")

    # domain_randomized extras
    for key in ("height_error_mean", "mass_perturb_pct", "friction_perturb_pct"):
        if key in mode_metrics:
            val = mode_metrics[key]
            label = key.replace("_", " ").title()
            lines.append(f"\n**{label}:** {val:.4f}")

    # Provenance
    if "checkpoint" in data:
        lines.append(f"\n*Checkpoint:* `{data['checkpoint']}`")
    if "stage" in data:
        lines.append(f"*Stage:* `{data['stage']}`")

    md = "\n".join(lines) + "\n"

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Markdown table written → {out}")
    else:
        print(md)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # curves
    p_curves = sub.add_parser(
        "curves",
        help="Export JSONL training log → CSV + PNG training curves.",
    )
    p_curves.add_argument("source", help="Path to *_metrics.jsonl log file.")
    p_curves.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Metric tags to plot (default: reward/mean + curriculum metrics).",
    )
    p_curves.add_argument(
        "--output",
        default=None,
        help="Output path (base name; .csv and .png are appended). "
        "Default: same directory as source.",
    )
    p_curves.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip PNG generation (write CSV only).",
    )

    # table
    p_table = sub.add_parser(
        "table",
        help="Export benchmark/eval JSON → Markdown table.",
    )
    p_table.add_argument("source", help="Path to eval_results_*.json file.")
    p_table.add_argument(
        "--output",
        default=None,
        help="Output .md file path. Default: print to stdout.",
    )

    # latex
    p_latex = sub.add_parser(
        "latex",
        help="Export eval_balance.py JSON → LaTeX booktabs table.",
    )
    p_latex.add_argument("source", help="Path to eval_results.json from eval_balance.py.")
    p_latex.add_argument(
        "--output",
        default=None,
        help="Output .tex file path. Default: print to stdout.",
    )
    p_latex.add_argument(
        "--caption",
        default=None,
        help="LaTeX table caption.",
    )
    p_latex.add_argument(
        "--label",
        default=None,
        help=r"LaTeX table label (for \ref{}).",
    )

    return parser


# ---------------------------------------------------------------------------
# Sub-command: latex
# ---------------------------------------------------------------------------


def cmd_latex(args: argparse.Namespace) -> None:
    """Export eval_balance.py JSON results as a LaTeX booktabs table."""
    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: file not found: {src}", file=sys.stderr)
        sys.exit(1)

    with open(src, encoding="utf-8") as f:
        data = json.load(f)

    # eval_balance.py JSON has {"results": [...]}
    # evaluate.py JSON is a flat dict — wrap in a list
    if "results" in data and isinstance(data["results"], list):
        results = data["results"]
    else:
        results = [data]

    from wheeled_biped.eval.latex_table import generate_latex_table

    caption = args.caption if args.caption else "Balance evaluation results."
    label = args.label if args.label else "tab:balance_eval"

    tex = generate_latex_table(results, caption=caption, label=label)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex, encoding="utf-8")
        print(f"LaTeX table written → {out}")
    else:
        print(tex)


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "curves":
        cmd_curves(args)
    elif args.command == "table":
        cmd_table(args)
    elif args.command == "latex":
        cmd_latex(args)
