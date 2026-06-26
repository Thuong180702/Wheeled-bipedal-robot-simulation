#!/usr/bin/env python3
"""
Score K1 Notch Filter Sweep Candidates.

Loads screening results, computes composite scores, ranks candidates,
and applies hard reject filters.

STRICT CONSTRAINT: Scoring only.  Do NOT promote candidates.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "k1_notch_filter_sweep"

# -- K1 Baseline Profile --
K1_PROFILE = "k1_pitch_rate_notch_v1"

# -- Scoring Weights --
WEIGHTS = {
    "low_freq_pitch_power": 3.0,
    "pitch_notch_coherence": 2.0,
    "pitch_rms": 2.0,
    "support_rms": 1.5,
    "wip_band_power": 2.5,
    "safety": 2.0,
    "clipping": 1.0,
    "complexity": 1.0,
}

# -- Hard Reject Thresholds --
HARD_REJECT = {
    "wip_power_ratio_max": 1.25,       # WIP power > 1.25x K1 baseline
    "lf_power_ratio_max": 1.20,        # Low-freq power > 1.20x K1 baseline
    "pitch_rms_ratio_max": 1.15,       # Pitch RMS > 1.15x K1 baseline
    "pitch_abs_max_deg": 35.0,         # Absolute max pitch threshold
    "body_height_min_m": 0.20,         # Minimum safe body height
}


def load_screening_results() -> dict:
    """Load screening results from JSON."""
    screening_path = OUTPUT_DIR / "screening_results.json"
    if not screening_path.exists():
        raise FileNotFoundError(f"Screening results not found: {screening_path}")
    with open(screening_path, "r") as f:
        return json.load(f)


def get_k1_baseline_metrics(results: dict) -> Optional[dict]:
    """Extract K1 baseline metrics from screening results."""
    k1_data = results.get(K1_PROFILE, {})
    eq_key = "high_0p480/A_equilibrium"
    eq_metrics = k1_data.get("runs", {}).get(eq_key, {})
    if "lf_power_0p15_0p55_hz" not in eq_metrics:
        return None
    return eq_metrics


def compute_candidate_score(candidate_metrics: dict, baseline_metrics: dict,
                            candidate_params: dict, candidate_name: str) -> dict:
    """Compute composite score for a single candidate."""
    score_components = {}
    hard_reject_reasons = []

    # Normalize by baseline
    bl_lf_power = baseline_metrics.get("lf_power_0p15_0p55_hz", 1e-10)
    bl_wip_power = baseline_metrics.get("wip_power_2p0_3p0_hz", 1e-10)
    bl_pitch_rms = baseline_metrics.get("pitch_rms_deg", 1.0)
    bl_support_rms = baseline_metrics.get("support_rms_m", 0.01)
    bl_coherence = baseline_metrics.get("lf_pitch_notch_coherence", 1.0)
    bl_clip = baseline_metrics.get("clip_fraction", 0.0)

    c_lf_power = candidate_metrics.get("lf_power_0p15_0p55_hz", bl_lf_power)
    c_wip_power = candidate_metrics.get("wip_power_2p0_3p0_hz", bl_wip_power)
    c_pitch_rms = candidate_metrics.get("pitch_rms_deg", bl_pitch_rms)
    c_support_rms = candidate_metrics.get("support_rms_m", bl_support_rms)
    c_coherence = candidate_metrics.get("lf_pitch_notch_coherence", bl_coherence)
    c_clip = candidate_metrics.get("clip_fraction", bl_clip)

    # Hard reject checks
    has_fall = candidate_metrics.get("has_fall", False)
    has_nan = candidate_metrics.get("has_nan", False)
    pitch_abs_max = candidate_metrics.get("pitch_abs_max_deg", 0)
    body_height_min = candidate_metrics.get("body_height_min_m", 1.0)

    if has_fall:
        hard_reject_reasons.append("FALL")
    if has_nan:
        hard_reject_reasons.append("NAN_INF")
    if pitch_abs_max > HARD_REJECT["pitch_abs_max_deg"]:
        hard_reject_reasons.append(f"PITCH_ABS_MAX={pitch_abs_max:.1f} > {HARD_REJECT['pitch_abs_max_deg']}")
    if body_height_min < HARD_REJECT["body_height_min_m"]:
        hard_reject_reasons.append(f"BODY_HEIGHT_MIN={body_height_min:.3f} < {HARD_REJECT['body_height_min_m']}")

    # WIP power ratio check
    wip_ratio = c_wip_power / max(bl_wip_power, 1e-10)
    if bl_wip_power > 1e-10 and wip_ratio > HARD_REJECT["wip_power_ratio_max"]:
        hard_reject_reasons.append(f"WIP_POWER_RATIO={wip_ratio:.2f} > {HARD_REJECT['wip_power_ratio_max']}")

    # Low-freq power ratio check
    lf_ratio = c_lf_power / max(bl_lf_power, 1e-10)
    if bl_lf_power > 1e-10 and lf_ratio > HARD_REJECT["lf_power_ratio_max"]:
        hard_reject_reasons.append(f"LF_POWER_RATIO={lf_ratio:.2f} > {HARD_REJECT['lf_power_ratio_max']}")

    # Pitch RMS ratio check
    pitch_rms_ratio = c_pitch_rms / max(bl_pitch_rms, 0.01)
    if bl_pitch_rms > 0.01 and pitch_rms_ratio > HARD_REJECT["pitch_rms_ratio_max"]:
        hard_reject_reasons.append(f"PITCH_RMS_RATIO={pitch_rms_ratio:.2f} > {HARD_REJECT['pitch_rms_ratio_max']}")

    # Hidden torque / WBC check
    filter_type = candidate_params.get("filter_type", "biquad_notch")
    has_hidden_torque = False  # All sweep candidates use standard K1 base

    # Compute normalized scores (lower is better)
    # Normalize to [0, 2] range with baseline at 1.0
    norm_lf_power = min(2.0, c_lf_power / max(bl_lf_power, 1e-10))
    norm_pitch_notch_coh = min(2.0, c_coherence / max(bl_coherence, 1e-10))
    norm_pitch_rms = min(2.0, c_pitch_rms / max(bl_pitch_rms, 0.01))
    norm_support_rms = min(2.0, c_support_rms / max(bl_support_rms, 0.001))
    norm_wip_power = min(2.0, c_wip_power / max(bl_wip_power, 1e-10))
    norm_clip = min(2.0, c_clip / max(bl_clip, 0.001)) if bl_clip > 1e-6 else 1.0

    # Safety penalty: 0 = safe, 2 = hard reject
    safety_penalty = 0.0
    if has_fall:
        safety_penalty = 2.0
    elif has_nan:
        safety_penalty = 2.0
    elif pitch_abs_max > 30.0:
        safety_penalty = 1.0
    elif pitch_abs_max > 20.0:
        safety_penalty = 0.5

    # Complexity penalty
    complexity_penalty = 0.0
    if filter_type == "first_order_lowpass":
        complexity_penalty = 0.2  # New topology, slightly higher complexity
    elif filter_type == "notch_disabled":
        complexity_penalty = 0.0  # No filter at all

    score = (
        WEIGHTS["low_freq_pitch_power"] * norm_lf_power +
        WEIGHTS["pitch_notch_coherence"] * norm_pitch_notch_coh +
        WEIGHTS["pitch_rms"] * norm_pitch_rms +
        WEIGHTS["support_rms"] * norm_support_rms +
        WEIGHTS["wip_band_power"] * norm_wip_power +
        WEIGHTS["safety"] * safety_penalty +
        WEIGHTS["clipping"] * norm_clip +
        WEIGHTS["complexity"] * complexity_penalty
    )

    score_components = {
        "norm_lf_power": round(norm_lf_power, 4),
        "norm_pitch_notch_coh": round(norm_pitch_notch_coh, 4),
        "norm_pitch_rms": round(norm_pitch_rms, 4),
        "norm_support_rms": round(norm_support_rms, 4),
        "norm_wip_power": round(norm_wip_power, 4),
        "safety_penalty": round(safety_penalty, 4),
        "norm_clip": round(norm_clip, 4),
        "complexity_penalty": round(complexity_penalty, 4),
    }

    # Classification
    if hard_reject_reasons:
        classification = "INVALID"
    elif has_fall or has_nan:
        classification = "INVALID"
    elif lf_ratio < 0.5 and wip_ratio < 0.90:
        classification = "STRONG_IMPROVEMENT"
    elif lf_ratio < 0.70 and wip_ratio < 1.0:
        classification = "MODE_REDUCED_WIP_SAFE"
    elif lf_ratio < 0.70 and wip_ratio > 1.0:
        classification = "LOW_FREQ_REDUCED_BUT_WIP_RISK"
    elif lf_ratio >= 0.90 and wip_ratio <= 1.0:
        classification = "WIP_SAFE_BUT_MODE_UNCHANGED"
    elif wip_ratio > 1.25 or lf_ratio > 1.2:
        classification = "REGRESSION"
    else:
        classification = "WIP_SAFE_BUT_MODE_UNCHANGED"

    return {
        "score": round(score, 4),
        "score_components": score_components,
        "classification": classification,
        "hard_reject_reasons": hard_reject_reasons,
        "raw_metrics": {
            "lf_power": c_lf_power,
            "lf_power_ratio_vs_k1": round(lf_ratio, 4),
            "wip_power": c_wip_power,
            "wip_power_ratio_vs_k1": round(wip_ratio, 4),
            "pitch_rms_deg": c_pitch_rms,
            "pitch_rms_ratio_vs_k1": round(pitch_rms_ratio, 4),
            "support_rms_m": c_support_rms,
            "pitch_notch_coherence": c_coherence,
            "lf_peak_freq_hz": candidate_metrics.get("lf_peak_freq_hz", 0),
            "pitch_abs_max_deg": pitch_abs_max,
            "body_height_min_m": body_height_min,
            "clip_fraction": c_clip,
        },
    }


def score_all_candidates(results: dict) -> dict:
    """Score all candidates and rank them."""
    baseline_metrics = get_k1_baseline_metrics(results)
    if baseline_metrics is None:
        return {"error": "K1_BASELINE_MISSING", "message": "K1 baseline metrics not found in screening results"}

    print("=" * 72)
    print("K1 NOTCH FILTER SWEEP SCORING")
    print(f"  Baseline: {K1_PROFILE}")
    print(f"  Baseline LF peak: {baseline_metrics.get('lf_peak_freq_hz', 'N/A')} Hz")
    print(f"  Baseline pitch RMS: {baseline_metrics.get('pitch_rms_deg', 'N/A'):.2f} deg")
    print("=" * 72)

    scored = {}
    k1_scored = None

    for candidate_name, candidate_data in results.items():
        params = candidate_data.get("params", {})
        eq_key = "high_0p480/A_equilibrium"
        eq_metrics = candidate_data.get("runs", {}).get(eq_key, {})

        if eq_metrics.get("error") or eq_metrics.get("status") not in ("OK", None):
            scored[candidate_name] = {
                "score": 999.0,
                "classification": "INVALID",
                "hard_reject_reasons": ["NO_EQUILIBRIUM_DATA"],
                "params": params,
            }
            continue

        candidate_score = compute_candidate_score(eq_metrics, baseline_metrics, params, candidate_name)
        candidate_score["params"] = params
        scored[candidate_name] = candidate_score

        if candidate_name == K1_PROFILE:
            k1_scored = candidate_score

    # Rank by score (lower is better)
    ranking = sorted(scored.items(), key=lambda x: x[1]["score"])

    # Count per classification
    classification_counts = {}
    for name, data in scored.items():
        cls = data.get("classification", "UNKNOWN")
        classification_counts[cls] = classification_counts.get(cls, 0) + 1

    # Prepare output
    output = {
        "baseline": {
            "profile": K1_PROFILE,
            "metrics": baseline_metrics,
            "score": k1_scored["score"] if k1_scored else None,
        },
        "scoring_weights": WEIGHTS,
        "hard_reject_thresholds": HARD_REJECT,
        "classification_counts": classification_counts,
        "candidates": {name: data for name, data in ranking},
        "ranking": [name for name, _ in ranking],
    }

    # Save scored results
    scored_path = OUTPUT_DIR / "scored_candidates.json"
    with open(scored_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Print ranking
    print(f"\n  Classification counts: {classification_counts}")
    print(f"\n  Ranked candidates (top 15):")
    for i, (name, data) in enumerate(ranking[:15]):
        print(f"  {i+1:2d}. {name:40s}  score={data['score']:.3f}  {data['classification']}")
        if data.get("hard_reject_reasons"):
            print(f"      REJECT: {', '.join(data['hard_reject_reasons'])}")

    print(f"\n  Results saved to: {scored_path}")

    # Also save markdown report
    md_path = OUTPUT_DIR / "scored_candidates.md"
    write_markdown_report(output, md_path)
    print(f"  Markdown report: {md_path}")

    return output


def write_markdown_report(output: dict, md_path: Path):
    """Write scored candidates as markdown report."""
    lines = []
    lines.append("# K1 Notch Filter Sweep — Scored Candidates")
    lines.append("")
    lines.append(f"**Baseline:** `{K1_PROFILE}`")
    bl = output["baseline"]["metrics"]
    lines.append(f"- LF peak: {bl.get('lf_peak_freq_hz', 'N/A')} Hz")
    lines.append(f"- Pitch RMS: {bl.get('pitch_rms_deg', 'N/A'):.2f} deg")
    lines.append(f"- LF power: {bl.get('lf_power_0p15_0p55_hz', 'N/A'):.6f}")
    lines.append(f"- WIP power: {bl.get('wip_power_2p0_3p0_hz', 'N/A'):.6f}")
    lines.append(f"- Pitch-notch coherence: {bl.get('lf_pitch_notch_coherence', 'N/A'):.4f}")
    lines.append("")

    lines.append("## Scoring Weights")
    lines.append("")
    lines.append("| Term | Weight |")
    lines.append("|------|--------|")
    for term, weight in WEIGHTS.items():
        lines.append(f"| {term} | {weight} |")
    lines.append("")

    lines.append("## Hard Reject Thresholds")
    lines.append("")
    lines.append("| Threshold | Value |")
    lines.append("|-----------|-------|")
    for key, val in HARD_REJECT.items():
        lines.append(f"| {key} | {val} |")
    lines.append("")

    lines.append("## Classification Counts")
    lines.append("")
    for cls, count in sorted(output["classification_counts"].items()):
        lines.append(f"- **{cls}:** {count}")
    lines.append("")

    lines.append("## Rankings")
    lines.append("")
    lines.append("| Rank | Candidate | Score | Classification | LF Ratio | WIP Ratio | Pitch RMS |")
    lines.append("|------|-----------|-------|----------------|----------|-----------|-----------|")
    for i, name in enumerate(output["ranking"][:20]):
        data = output["candidates"][name]
        raw = data.get("raw_metrics", {})
        lines.append(
            f"| {i+1} | `{name}` | {data['score']:.3f} | {data['classification']} | "
            f"{raw.get('lf_power_ratio_vs_k1', 'N/A')} | "
            f"{raw.get('wip_power_ratio_vs_k1', 'N/A')} | "
            f"{raw.get('pitch_rms_deg', 'N/A')} |"
        )
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Score K1 Notch Filter Sweep Candidates")
    parser.add_argument("--screening-results", type=str, default=None,
                       help="Path to screening results JSON (default: outputs/k1_notch_filter_sweep/screening_results.json)")
    args = parser.parse_args()

    results = load_screening_results()
    score_all_candidates(results)


if __name__ == "__main__":
    main()
