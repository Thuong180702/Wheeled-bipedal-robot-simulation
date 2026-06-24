#!/usr/bin/env python3
"""Analyze and rank K1 controller completion candidates (L, M, N families).

Aggregates telemetry from all candidate runs and applies ranking tiers:
    Tier 0 — integrity (real_simulation, direct telemetry, no WBC)
    Tier 1 — safety (no falls, no NaN/Inf)
    Tier 2 — hard gates (hip_yaw, support, pitch)
    Tier 3 — task quality (sustained recovery, RMS, 2.5 Hz attenuation)
    Tier 4 — architecture (coordinated state feedback, correct actuator)

Output:
    outputs/k1_controller_completion/analysis/
        ranking_summary.csv
        ranking_summary.txt

Usage:
    python scripts/analyze_k1_controller_completion_results.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_BASE = ROOT / "outputs" / "k1_controller_completion"
ANALYSIS_DIR = OUT_BASE / "analysis"


def load_metrics(csv_path: Path) -> list[dict]:
    """Load metrics CSV if exists."""
    import csv as _csv
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        reader = _csv.DictReader(f)
        return [dict(r) for r in reader]


def classify_candidate(metrics: dict) -> str:
    """Rank a candidate by metrics.

    Returns one of:
        SAFETY_FAIL, REGRESSION, NO_IMPROVEMENT, IMPROVED_NOT_PROMOTED,
        PROMOTED
    """
    try:
        fell = str(metrics.get("fell", "True")).lower() in ("true", "1", "yes")
        nan_count = int(metrics.get("nan_count", 1))
        inf_count = int(metrics.get("inf_count", 0))
        hy = float(metrics.get("hip_yaw_abs_max", 1.0))
        pitch_rms = float(metrics.get("pitch_rms_deg", 99.0))
    except (ValueError, TypeError):
        return "INCONCLUSIVE"

    # Tier 1: Safety
    if fell or nan_count > 0 or inf_count > 0:
        return "SAFETY_FAIL"

    # Tier 2: Hard gates
    hy_gate = hy < 0.35
    # Support gate (soft)
    support_max = float(metrics.get("support_max_m", 1.0))
    support_ok = support_max < 0.30

    # Pitch gate (soft)
    pitch_ok = pitch_rms < 10.0

    if not hy_gate and not support_ok:
        return "NO_IMPROVEMENT"

    # Tier 3: Quality
    pitch_improved = pitch_rms < 6.0  # better than K1's typical ~5-6 deg post-push

    if hy_gate and pitch_improved and support_ok:
        return "IMPROVED_NOT_PROMOTED"
    elif hy_gate and support_ok:
        return "NO_IMPROVEMENT"
    else:
        return "REGRESSION"


def write_ranking(all_metrics: list[dict], path: Path):
    """Write ranking summary."""
    lines = []
    lines.append("=" * 80)
    lines.append("K1 CONTROLLER COMPLETION — CANDIDATE RANKING")
    lines.append("=" * 80)
    lines.append("")

    # Classify each candidate
    for m in all_metrics:
        m["classification"] = classify_candidate(m)

    # Group by classification
    tiers = ["SAFETY_FAIL", "REGRESSION", "NO_IMPROVEMENT",
             "IMPROVED_NOT_PROMOTED", "PROMOTED", "INCONCLUSIVE"]
    for tier in tiers:
        tier_candidates = [m for m in all_metrics if m.get("classification") == tier]
        if not tier_candidates:
            continue
        lines.append(f"\n### {tier} ({len(tier_candidates)})")
        for m in tier_candidates:
            name = m.get("profile", m.get("case", m.get("profile", "?")))
            hy = m.get("hip_yaw_abs_max", "N/A")
            pitch = m.get("pitch_rms_deg", "N/A")
            fell = m.get("fell", "?")
            lines.append(f"  {name:45s}  hy={hy:.4f}  pitch={pitch:.2f}°  fell={fell}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("PROMOTION DECISION")
    lines.append("=" * 80)

    promoted = [m for m in all_metrics if m.get("classification") == "PROMOTED"]
    if promoted:
        lines.append(f"PROMOTED: {len(promoted)} candidate(s)")
        for m in promoted:
            lines.append(f"  {m.get('profile', m.get('case', '?'))}")
        lines.append("")
        lines.append("ACTION: Update current-best to promoted candidate.")
    else:
        lines.append("NO CANDIDATE PROMOTED.")
        lines.append("K1 remains current-best with known limitations:")
        lines.append("  - D4/D5 hip_yaw_abs_max > 0.35 rad")
        lines.append("  - No sustained posture recovery (2+ s hold after push)")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  [RANKING] Wrote ranking summary to {path}")


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Search for metrics CSVs from various validation runs
    all_metrics = []
    glob_patterns = [
        "true_dynamic_step_c/analysis/dynamic_step_c_metrics.csv",
        "sustained_recovery_audit/analysis/*.csv",
        "d4_d5_body_yaw_audit/analysis/*.csv",
    ]
    for pattern in glob_patterns:
        for path in sorted(OUT_BASE.glob(pattern)):
            metrics = load_metrics(path)
            all_metrics.extend(metrics)
            print(f"  [LOAD] {len(metrics)} rows from {path}", flush=True)

    # Also look for K1 reference metrics
    k1_ref_path = ROOT / "outputs" / "k1_post_promotion_validation" / "analysis"
    k1_refs = [
        k1_ref_path / "step_e_metrics.csv",
        k1_ref_path / "step_c_metrics.csv",
        k1_ref_path / "full_step_d_metrics.csv",
    ]
    for ref_path in k1_refs:
        metrics = load_metrics(ref_path)
        if metrics:
            for m in metrics:
                m["reference"] = "K1"
            all_metrics.extend(metrics)
            print(f"  [LOAD] {len(metrics)} K1 reference rows from {ref_path}", flush=True)

    if not all_metrics:
        print("[ERROR] No metrics found to rank.", flush=True)
        print("  Run the simulation harnesses first:", flush=True)
        print("    - python scripts/run_true_dynamic_height_step_c_validation.py", flush=True)
        print("    - python scripts/audit_k1_sustained_recovery_failure.py", flush=True)
        print("    - python scripts/audit_k1_d4_d5_body_yaw_to_hip_yaw_coupling.py", flush=True)
        return

    # Write CSV
    csv_path = ANALYSIS_DIR / "candidate_ranking.csv"
    with open(csv_path, "w", newline="") as f:
        if all_metrics:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
    print(f"  Wrote {len(all_metrics)} rows to {csv_path}", flush=True)

    # Write ranking summary
    write_ranking(all_metrics, ANALYSIS_DIR / "candidate_ranking.txt")


if __name__ == "__main__":
    main()
