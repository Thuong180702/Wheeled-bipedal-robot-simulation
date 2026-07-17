#!/usr/bin/env python3
"""
K2 Stability Improvement Evaluator
===================================

Evaluates K2 controller improvements against the K2_STABILITY_SCORE.
Compares baseline and candidate quality data, computes scores,
identifies regressions, and classifies outcomes.

Inputs:
  --baseline: Path to baseline quality JSON (from analyze_k2_behavior_quality.py)
  --candidate: Path to candidate quality JSON
  OR
  --baseline-dir / --candidate-dir: Directories to analyze first, then evaluate

Output:
  --output: Markdown evaluation report

Usage:
  python scripts/evaluate_k2_stability_improvement.py \
    --baseline docs/validation/k2_improvement_baseline_quality.json \
    --candidate docs/validation/k2_improvement_candidate_quality.json \
    --output docs/validation/k2_improvement_candidate_evaluation.md
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Smoothstep ─────────────────────────────────────────────────────────────────


def smoothstep(x, low, high):
    """Smoothstep from 0 at x<=low to 1 at x>=high."""
    if high - low < 1e-9:
        return 1.0 if x >= high else 0.0
    t = (x - low) / (high - low)
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def score_from_smoothstep(value, perfect, zero, invert=False):
    """Score a metric using smoothstep. 1.0 = perfect, 0.0 = worst."""
    if invert:
        return 1.0 - smoothstep(value, perfect, zero)
    return 1.0 - smoothstep(value, perfect, zero)


# ── Scoring Functions ──────────────────────────────────────────────────────────


def compute_safety_gate(scenario: Dict) -> Tuple[bool, List[str]]:
    """Check hard safety gates. Returns (passed, failures_list)."""
    failures = []
    safety = scenario.get("safety", {})

    if safety.get("fell", False):
        failures.append("FALL")
    if safety.get("nan_inf_detected", False):
        failures.append("NaN/Inf detected")
    if safety.get("hip_yaw_joint_max_rad", 0.0) > 0.35:
        failures.append(f"hip_yaw_joint_max_rad={safety['hip_yaw_joint_max_rad']:.4f} > 0.35")

    # Check for catastrophic instability from pitch/roll extremes
    pitch_max = abs(safety.get("pitch_max_deg", 0.0))
    roll_max = abs(safety.get("roll_max_deg", 0.0))
    if pitch_max > 45.0 or roll_max > 45.0:
        failures.append(f"Catastrophic orientation: pitch_max={pitch_max:.1f}, roll_max={roll_max:.1f}")

    return len(failures) == 0, failures


def compute_posture_score(scenario: Dict) -> float:
    """Compute posture stability score [0,1]."""
    p = scenario.get("posture", {})
    pitch_rms = p.get("pitch_rms_deg", 0.0)
    roll_rms = p.get("roll_rms_deg", 0.0)
    angvel_rms = p.get("angular_velocity_rms_deg_s", 0.0)
    peak_val = max(p.get("pitch_peak_deg", 0.0), p.get("roll_peak_deg", 0.0))

    s_pitch = score_from_smoothstep(pitch_rms, 1.5, 6.0)
    s_roll = score_from_smoothstep(roll_rms, 0.5, 3.0)
    s_angvel = score_from_smoothstep(angvel_rms, 5.0, 30.0)
    s_peak = score_from_smoothstep(peak_val, 3.0, 15.0)

    return 0.50 * s_pitch + 0.25 * s_roll + 0.15 * s_angvel + 0.10 * s_peak


def compute_support_score(scenario: Dict) -> float:
    """Compute support/drift score [0,1]."""
    sd = scenario.get("support_drift", {})
    support_rms = sd.get("support_rms_m", 0.0)
    displacement = sd.get("final_displacement_m", 0.0)
    sagittal = abs(sd.get("sagittal_drift_m", 0.0))
    lateral = abs(sd.get("lateral_drift_m", 0.0))

    s_support = score_from_smoothstep(support_rms, 0.005, 0.03)
    s_displacement = score_from_smoothstep(displacement, 0.02, 0.20)
    s_sagittal = score_from_smoothstep(sagittal, 0.02, 0.15)
    s_lateral = score_from_smoothstep(lateral, 0.01, 0.08)

    return 0.35 * s_support + 0.25 * s_displacement + 0.20 * s_sagittal + 0.20 * s_lateral


def compute_leg_score(scenario: Dict) -> float:
    """Compute leg health / hip-yaw score [0,1]."""
    ls = scenario.get("leg_symmetry", {})
    hy_max = ls.get("hip_yaw_joint_max_rad", 0.0)
    hy_rms = ls.get("hip_yaw_div_rms_rad", 0.0)
    symmetry = ls.get("leg_posture_error_rms", 0.0)
    hp_sym = ls.get("hip_pitch_symmetry_error_deg", 0.0)

    s_hy_max = score_from_smoothstep(hy_max, 0.05, 0.25)
    s_hy_rms = score_from_smoothstep(hy_rms, 0.02, 0.15)
    s_symmetry = score_from_smoothstep(symmetry, 0.05, 0.50)
    s_hp_sym = score_from_smoothstep(hp_sym, 1.0, 10.0)

    return 0.40 * s_hy_max + 0.25 * s_hy_rms + 0.20 * s_symmetry + 0.15 * s_hp_sym


def compute_dynamic_height_score(scenario: Dict) -> float:
    """Compute dynamic height score [0,1]."""
    dh = scenario.get("dynamic_height", {})
    rmse = dh.get("height_rmse_m", 0.0)
    overshoot = abs(dh.get("height_overshoot_m", 0.0))
    smoothness = dh.get("dynamic_transition_smoothness", 0.0)
    qref_err = dh.get("q_ref_tracking_error_rms", 0.0)

    s_rmse = score_from_smoothstep(rmse, 0.005, 0.04)
    s_overshoot = score_from_smoothstep(overshoot, 0.01, 0.06)
    s_smoothness = score_from_smoothstep(smoothness, 0.1, 2.0)
    s_qref = score_from_smoothstep(qref_err, 0.02, 0.20)

    return 0.40 * s_rmse + 0.30 * s_overshoot + 0.20 * s_smoothness + 0.10 * s_qref


def compute_torque_score(scenario: Dict) -> float:
    """Compute torque quality score [0,1]."""
    tq = scenario.get("torque_quality", {})
    # Pooled torque RMS: average of per-joint RMS values
    per_joint_rms = []
    for jn in ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
               "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]:
        key = f"torque_rms_{jn}_nm"
        if key in tq and tq[key] > 0:
            per_joint_rms.append(tq[key])
    tau_rms_pooled = np.mean(per_joint_rms) if per_joint_rms else tq.get("torque_peak_total_nm", 0.0) * 0.3

    tau_peak = tq.get("torque_peak_total_nm", 0.0)
    tau_rate = tq.get("torque_rate_rms_nm_s", 0.0)
    sat_count = tq.get("torque_saturation_count", 0)

    s_tau_rms = score_from_smoothstep(tau_rms_pooled, 1.0, 8.0)
    s_tau_peak = score_from_smoothstep(tau_peak, 5.0, 30.0)
    s_tau_rate = score_from_smoothstep(tau_rate, 50.0, 400.0)
    s_sat = score_from_smoothstep(float(sat_count), 5.0, 100.0)

    return 0.35 * s_tau_rms + 0.25 * s_tau_peak + 0.25 * s_tau_rate + 0.15 * s_sat


def compute_robustness_score(scenario: Dict) -> float:
    """Compute robustness score [0,1]."""
    rb = scenario.get("robustness", {})
    contact_loss = rb.get("contact_loss_frac", 0.0)
    drift_rate = rb.get("long_run_drift_rate_m_per_kstep", 0.0)
    post_push = rb.get("post_pitch_rms_500_deg", 0.0)
    stability = rb.get("stability_score_0_to_1", 1.0)

    s_contact = score_from_smoothstep(contact_loss, 0.001, 0.05)
    s_drift = score_from_smoothstep(drift_rate, 0.01, 0.20)
    s_post_push = score_from_smoothstep(post_push, 2.0, 10.0)
    s_stability = stability  # Already in [0,1]

    return 0.35 * s_contact + 0.30 * s_drift + 0.20 * s_post_push + 0.15 * s_stability


def compute_scenario_score(scenario: Dict) -> Dict:
    """Compute all scores for a single scenario."""
    safety_pass, safety_failures = compute_safety_gate(scenario)

    s_posture = compute_posture_score(scenario)
    s_support = compute_support_score(scenario)
    s_leg = compute_leg_score(scenario)

    scenario_type = scenario.get("scenario_type", "fixed_height")
    if scenario_type == "dynamic_height":
        s_dyn = compute_dynamic_height_score(scenario)
    else:
        # For non-dynamic scenarios, dynamic height score is N/A — use neutral 0.5
        s_dyn = 0.5

    s_torque = compute_torque_score(scenario)
    s_robust = compute_robustness_score(scenario)

    # K2_STABILITY_SCORE
    k2_score = (0.30 * s_posture + 0.20 * s_support + 0.15 * s_leg +
                0.15 * s_dyn + 0.10 * s_torque + 0.10 * s_robust)

    return {
        "scenario_id": scenario.get("scenario_id", "unknown"),
        "safety_pass": safety_pass,
        "safety_failures": safety_failures,
        "S_posture": s_posture,
        "S_support": s_support,
        "S_leg": s_leg,
        "S_dynamic_height": s_dyn,
        "S_torque": s_torque,
        "S_robustness": s_robust,
        "K2_STABILITY_SCORE": k2_score,
        "has_full_telemetry": scenario.get("has_full_telemetry", False),
    }


# ── Regression Detection ───────────────────────────────────────────────────────


REGRESSION_LIMITS = {
    "posture.pitch_rms_deg": {"abs_max_increase": 1.0, "rel_max_increase": 0.20},
    "leg_symmetry.hip_yaw_joint_max_rad": {"abs_max_increase": 0.05, "rel_max_increase": None},
    "support_drift.support_rms_m": {"abs_max_increase": 0.015, "rel_max_increase": None},
    "dynamic_height.height_rmse_m": {"abs_max_increase": 0.015, "rel_max_increase": None},
    "support_drift.final_displacement_m": {"abs_max_increase": 0.10, "rel_max_increase": None},
    "torque_quality.torque_peak_total_nm": {"abs_max_increase": 5.0, "rel_max_increase": None},
    "robustness.contact_loss_frac": {"abs_max_increase": 0.02, "rel_max_increase": None},
}

PERFORMANCE_MIN_HZ = 50.0


def get_nested(d: Dict, path: str, default=0.0):
    """Get nested dict value by dotted path."""
    keys = path.split(".")
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d if d is not None else default


def check_regressions(baseline_scenario: Dict, candidate_scenario: Dict) -> List[Dict]:
    """Check for regressions between baseline and candidate."""
    regressions = []
    for path, limits in REGRESSION_LIMITS.items():
        base_val = get_nested(baseline_scenario, path, 0.0)
        cand_val = get_nested(candidate_scenario, path, 0.0)

        if base_val == 0.0 and cand_val == 0.0:
            continue

        delta = cand_val - base_val

        # Absolute check
        if limits["abs_max_increase"] is not None and delta > limits["abs_max_increase"]:
            regressions.append({
                "metric": path,
                "baseline": base_val,
                "candidate": cand_val,
                "delta": delta,
                "limit": limits["abs_max_increase"],
                "type": "absolute",
                "severity": "MAJOR",
            })
        # Relative check
        elif limits["rel_max_increase"] is not None and base_val > 1e-9:
            rel_delta = delta / base_val
            if rel_delta > limits["rel_max_increase"]:
                regressions.append({
                    "metric": path,
                    "baseline": base_val,
                    "candidate": cand_val,
                    "delta": delta,
                    "delta_rel": rel_delta,
                    "limit": limits["rel_max_increase"],
                    "type": "relative",
                    "severity": "MAJOR",
                })

    return regressions


# ── Main Evaluation ────────────────────────────────────────────────────────────


def evaluate(baseline_data: Dict, candidate_data: Dict) -> Dict:
    """Run full evaluation comparing baseline vs candidate."""
    baseline_scenarios = {s["scenario_id"]: s for s in baseline_data.get("scenarios", [])}
    candidate_scenarios = {s["scenario_id"]: s for s in candidate_data.get("scenarios", [])}

    all_scenario_ids = sorted(set(baseline_scenarios.keys()) | set(candidate_scenarios.keys()))

    results = []
    total_safety_fails = 0
    total_regressions = 0
    baseline_scores = []
    candidate_scores = []

    for sid in all_scenario_ids:
        base_s = baseline_scenarios.get(sid)
        cand_s = candidate_scenarios.get(sid)

        if base_s is None:
            results.append({"scenario_id": sid, "status": "NOT_IN_BASELINE"})
            continue
        if cand_s is None:
            results.append({"scenario_id": sid, "status": "NOT_IN_CANDIDATE"})
            continue

        base_result = compute_scenario_score(base_s)
        cand_result = compute_scenario_score(cand_s)

        # Safety check
        if not cand_result["safety_pass"]:
            total_safety_fails += 1

        # Regression check
        regressions = check_regressions(base_s, cand_s)
        total_regressions += len([r for r in regressions if r["severity"] == "MAJOR"])

        # Score delta
        score_delta = cand_result["K2_STABILITY_SCORE"] - base_result["K2_STABILITY_SCORE"]

        baseline_scores.append(base_result["K2_STABILITY_SCORE"])
        candidate_scores.append(cand_result["K2_STABILITY_SCORE"])

        results.append({
            "scenario_id": sid,
            "status": "evaluated",
            "baseline_score": base_result["K2_STABILITY_SCORE"],
            "candidate_score": cand_result["K2_STABILITY_SCORE"],
            "score_delta": score_delta,
            "safety_pass": cand_result["safety_pass"],
            "safety_failures": cand_result["safety_failures"],
            "regressions": regressions,
            "baseline_detail": base_result,
            "candidate_detail": cand_result,
            "scenario_type": base_s.get("scenario_type", "unknown"),
            "height_region": base_s.get("height_region", "unknown"),
        })

    # Compute aggregate scores
    agg_baseline = np.mean(baseline_scores) if baseline_scores else 0.0
    agg_candidate = np.mean(candidate_scores) if candidate_scores else 0.0
    agg_delta = agg_candidate - agg_baseline

    # Check performance
    baseline_hz = baseline_data.get("aggregate", {}).get("performance", {}).get("mean_hz", 0.0)
    candidate_hz = candidate_data.get("aggregate", {}).get("performance", {}).get("mean_hz", 0.0)
    perf_regression = candidate_hz < PERFORMANCE_MIN_HZ

    # Classification
    if total_safety_fails > 0 or perf_regression:
        classification = "SAFETY_FAIL"
    elif agg_delta < 0 and agg_candidate < 0.40:
        classification = "SAFETY_FAIL"
    elif total_regressions > 0:
        classification = "STABILITY_REGRESSED"
    elif agg_candidate >= 0.80:
        classification = "STABILITY_IMPROVED_PASS"
    elif agg_candidate >= 0.60:
        classification = "STABILITY_PARTIAL"
    else:
        classification = "STABILITY_REGRESSED"

    return {
        "classification": classification,
        "baseline_aggregate_score": float(agg_baseline),
        "candidate_aggregate_score": float(agg_candidate),
        "score_delta": float(agg_delta),
        "total_scenarios": len(results),
        "evaluated_scenarios": len([r for r in results if r["status"] == "evaluated"]),
        "safety_fails": total_safety_fails,
        "total_regressions": total_regressions,
        "performance": {
            "baseline_hz": baseline_hz,
            "candidate_hz": candidate_hz,
            "perf_regression": perf_regression,
        },
        "scenario_results": results,
        "baseline_scores": [float(s) for s in baseline_scores],
        "candidate_scores": [float(s) for s in candidate_scores],
    }


# ── Report Generation ──────────────────────────────────────────────────────────


def generate_report(eval_result: Dict, output_path: Path, metadata: Optional[Dict] = None):
    """Generate Markdown evaluation report."""
    lines = []
    lines.append("# K2 Stability Improvement — Candidate Evaluation\n")
    lines.append(f"**Classification:** `{eval_result['classification']}`")
    if metadata:
        for k, v in metadata.items():
            lines.append(f"**{k}:** {v}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Classification | **{eval_result['classification']}** |")
    lines.append(f"| Baseline aggregate score | {eval_result['baseline_aggregate_score']:.4f} |")
    lines.append(f"| Candidate aggregate score | {eval_result['candidate_aggregate_score']:.4f} |")
    lines.append(f"| Score delta | {eval_result['score_delta']:+.4f} |")
    lines.append(f"| Evaluated scenarios | {eval_result['evaluated_scenarios']}/{eval_result['total_scenarios']} |")
    lines.append(f"| Safety fails | {eval_result['safety_fails']} |")
    lines.append(f"| Major regressions | {eval_result['total_regressions']} |")
    lines.append(f"| Baseline performance | {eval_result['performance']['baseline_hz']:.1f} Hz |")
    lines.append(f"| Candidate performance | {eval_result['performance']['candidate_hz']:.1f} Hz |")
    if eval_result['performance']['perf_regression']:
        lines.append(f"| **Performance regression!** | Below 50 Hz minimum |")
    lines.append("")

    # Per-scenario detail
    lines.append("## Per-Scenario Results\n")
    lines.append("| Scenario | Type | Base Score | Cand Score | Delta | Safety | Regressions |")
    lines.append("|----------|------|------------|------------|-------|--------|-------------|")

    for r in eval_result["scenario_results"]:
        if r["status"] != "evaluated":
            lines.append(f"| {r['scenario_id']} | — | — | — | — | {r['status']} | — |")
            continue
        safety_icon = "PASS" if r["safety_pass"] else f"FAIL({','.join(r['safety_failures'])})"
        reg_count = len(r.get("regressions", []))
        reg_icon = f"{reg_count} regressions" if reg_count > 0 else "OK"
        delta_str = f"{r['score_delta']:+.4f}"
        lines.append(f"| {r['scenario_id']} | {r['scenario_type']} | {r['baseline_score']:.4f} | "
                    f"{r['candidate_score']:.4f} | {delta_str} | {safety_icon} | {reg_icon} |")
    lines.append("")

    # Score breakdown by dimension
    lines.append("## Dimension Score Comparison\n")
    dims = ["S_posture", "S_support", "S_leg", "S_dynamic_height", "S_torque", "S_robustness"]
    dim_labels = ["Posture", "Support/Drift", "Leg Health", "Dynamic Height", "Torque Quality", "Robustness"]

    lines.append("| Dimension | Baseline Mean | Candidate Mean | Delta |")
    lines.append("|-----------|---------------|----------------|-------|")
    for dim, label in zip(dims, dim_labels):
        base_vals = [r["baseline_detail"][dim] for r in eval_result["scenario_results"]
                     if r["status"] == "evaluated"]
        cand_vals = [r["candidate_detail"][dim] for r in eval_result["scenario_results"]
                     if r["status"] == "evaluated"]
        if base_vals:
            base_mean = np.mean(base_vals)
            cand_mean = np.mean(cand_vals)
            delta = cand_mean - base_mean
            lines.append(f"| {label} | {base_mean:.4f} | {cand_mean:.4f} | {delta:+.4f} |")
    lines.append("")

    # Regression details
    all_regressions = []
    for r in eval_result["scenario_results"]:
        if r["status"] == "evaluated":
            for reg in r.get("regressions", []):
                reg["scenario_id"] = r["scenario_id"]
                all_regressions.append(reg)

    if all_regressions:
        lines.append("## Regression Details\n")
        lines.append("| Scenario | Metric | Baseline | Candidate | Delta | Limit |")
        lines.append("|----------|--------|----------|-----------|-------|-------|")
        for reg in all_regressions:
            lines.append(f"| {reg['scenario_id']} | {reg['metric']} | {reg['baseline']:.4f} | "
                        f"{reg['candidate']:.4f} | {reg['delta']:+.4f} | {reg['limit']:.4f} |")
        lines.append("")

    # Classification explanation
    lines.append("## Classification\n")
    lines.append(f"**Result:** `{eval_result['classification']}`\n")

    if eval_result['classification'] == "STABILITY_IMPROVED_PASS":
        lines.append("All criteria met:")
        lines.append("- [x] Zero safety fails")
        lines.append("- [x] No major regressions")
        lines.append("- [x] Aggregate score >= 0.80")
        lines.append("- [x] Performance >= 50 Hz")
    elif eval_result['classification'] == "SAFETY_FAIL":
        lines.append("Failure reasons:")
        if eval_result['safety_fails'] > 0:
            lines.append(f"- {eval_result['safety_fails']} safety gate violation(s)")
            for r in eval_result["scenario_results"]:
                if r["status"] == "evaluated" and not r["safety_pass"]:
                    lines.append(f"  - {r['scenario_id']}: {', '.join(r['safety_failures'])}")
        if eval_result['performance']['perf_regression']:
            lines.append("- Performance below 50 Hz minimum")
    elif eval_result['classification'] == "STABILITY_REGRESSED":
        lines.append(f"- {eval_result['total_regressions']} major regression(s) detected")
    elif eval_result['classification'] == "STABILITY_PARTIAL":
        lines.append("- Some improvements, but aggregate score below 0.80")
        lines.append(f"- Current score: {eval_result['candidate_aggregate_score']:.4f} (need >= 0.80)")

    # Write
    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="K2 Stability Improvement Evaluator")
    p.add_argument("--baseline", type=str, default=None,
                   help="Path to baseline quality JSON")
    p.add_argument("--candidate", type=str, default=None,
                   help="Path to candidate quality JSON")
    p.add_argument("--baseline-dir", type=str, default=None,
                   help="Alternatively: baseline directory to analyze first")
    p.add_argument("--candidate-dir", type=str, default=None,
                   help="Alternatively: candidate directory to analyze first")
    p.add_argument("--output", type=str, required=True,
                   help="Output Markdown report path")
    args = p.parse_args()

    # Load baseline data
    if args.baseline:
        with open(args.baseline, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    elif args.baseline_dir:
        # Run analyzer first
        from analyze_k2_behavior_quality import main as analyze_main
        import subprocess as sp
        json_path = str(Path(args.baseline_dir).parent / "baseline_quality.json")
        sp.run([sys.executable, "scripts/analyze_k2_behavior_quality.py",
                "--input-dir", args.baseline_dir, "--output",
                json_path.replace(".json", ".md")], check=True)
        with open(json_path, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
    else:
        print("ERROR: Must provide --baseline or --baseline-dir")
        return 1

    # Load candidate data
    if args.candidate:
        with open(args.candidate, "r", encoding="utf-8") as f:
            candidate_data = json.load(f)
    elif args.candidate_dir:
        import subprocess as sp
        json_path = str(Path(args.candidate_dir).parent / "candidate_quality.json")
        sp.run([sys.executable, "scripts/analyze_k2_behavior_quality.py",
                "--input-dir", args.candidate_dir, "--output",
                json_path.replace(".json", ".md")], check=True)
        with open(json_path, "r", encoding="utf-8") as f:
            candidate_data = json.load(f)
    else:
        print("ERROR: Must provide --candidate or --candidate-dir")
        return 1

    print(f"Baseline: {len(baseline_data.get('scenarios', []))} scenarios")
    print(f"Candidate: {len(candidate_data.get('scenarios', []))} scenarios")

    eval_result = evaluate(baseline_data, candidate_data)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "baseline_source": args.baseline or args.baseline_dir,
        "candidate_source": args.candidate or args.candidate_dir,
        "evaluator_version": "1.0.0",
    }
    report = generate_report(eval_result, output_path, metadata)

    # JSON output
    json_out = str(output_path).replace(".md", ".json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULT: {eval_result['classification']}")
    print(f"{'='*60}")
    print(f"Baseline score:  {eval_result['baseline_aggregate_score']:.4f}")
    print(f"Candidate score: {eval_result['candidate_aggregate_score']:.4f}")
    print(f"Delta:           {eval_result['score_delta']:+.4f}")
    print(f"Safety fails:    {eval_result['safety_fails']}")
    print(f"Regressions:     {eval_result['total_regressions']}")
    print(f"Report: {output_path}")
    print(f"JSON:   {json_out}")

    return 0 if eval_result["classification"] != "SAFETY_FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
