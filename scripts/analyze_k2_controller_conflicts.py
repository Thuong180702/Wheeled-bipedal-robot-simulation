#!/usr/bin/env python3
"""
K2 Controller Conflict & Cross-Coupling Analyzer (Phase 3)
===========================================================

Analyzes per-component torque telemetry from the instrumented K2 JAX controller
to identify controller conflicts, cross-coupling, and authority allocation issues.

Input:
  --input-dir: Directory containing full-telemetry CSV files from instrumented runs

Output:
  --output: Markdown report path

Usage:
  python scripts/analyze_k2_controller_conflicts.py \
    --input-dir outputs/k2_phase3_instrumented \
    --output docs/analysis/k2_current_controller_interaction_audit.md
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Constants ──────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
]

# Conflict-prone joint pairs (indices where multiple components write)
CONFLICT_JOINTS = {
    "hip_yaw": {"indices": [1, 6], "components": ["posture", "yaw", "mode_div"]},
    "hip_roll": {"indices": [0, 5], "components": ["posture", "lateral"]},
    "hip_pitch": {"indices": [2, 7], "components": ["posture", "emp_support_ff"]},
    "knee": {"indices": [3, 8], "components": ["posture", "emp_support_ff"]},
}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) > 0 else 0.0


def peak(x):
    return float(np.max(np.abs(x))) if len(x) > 0 else 0.0


# ── Telemetry Loading ──────────────────────────────────────────────────────────


def load_telemetry(csv_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load a full telemetry CSV."""
    if not csv_path.exists():
        return None
    with open(csv_path, "r") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}


def discover_runs(input_dir: Path) -> List[Tuple[str, Path]]:
    """Discover all CSV files in the input directory hierarchy."""
    runs = []
    for csv_path in sorted(input_dir.rglob("telemetry_*.csv")):
        # Determine scenario name from path
        rel = csv_path.relative_to(input_dir)
        scenario_name = str(rel.parent).replace("\\", "/")
        if scenario_name == ".":
            scenario_name = csv_path.stem
        runs.append((scenario_name, csv_path))
    return runs


# ── Conflict Analysis ──────────────────────────────────────────────────────────


def analyze_conflicts(tel: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze same-joint torque cancellation."""
    result = {}

    for joint_group, info in CONFLICT_JOINTS.items():
        idx_l, idx_r = info["indices"]
        jn_l = JOINT_NAMES[idx_l]
        jn_r = JOINT_NAMES[idx_r]

        # Load per-component torques for these joints
        comp_torques = {}
        for comp in info["components"]:
            # Map component name to CSV column prefix
            col_prefixes = {
                "posture": f"tau_posture_{joint_group[:2]}",  # "tau_posture_hy", "tau_posture_hr", etc.
                "yaw": "tau_yaw",
                "mode_div": "tau_mode_div",
                "lateral": "tau_lateral",
                "emp_support_ff": "tau_emp_support",
            }
            prefix = col_prefixes.get(comp, f"tau_{comp}")
            # Try to find matching columns
            l_key = f"{prefix}_l"
            r_key = f"{prefix}_r"
            # Handle special naming
            if comp == "emp_support_ff":
                if joint_group == "hip_pitch":
                    l_key = "tau_emp_support_hp_l"
                    r_key = "tau_emp_support_hp_r"
                elif joint_group == "knee":
                    l_key = "tau_emp_support_kn_l"
                    r_key = "tau_emp_support_kn_r"
            elif comp == "posture":
                if joint_group == "hip_yaw":
                    l_key = "tau_posture_hy_l"
                    r_key = "tau_posture_hy_r"
                elif joint_group == "hip_roll":
                    l_key = "tau_posture_hr_l"
                    r_key = "tau_posture_hr_r"
                elif joint_group == "hip_pitch":
                    l_key = "tau_posture_hp_l"
                    r_key = "tau_posture_hp_r"
                elif joint_group == "knee":
                    l_key = "tau_posture_kn_l"
                    r_key = "tau_posture_kn_r"

            if l_key in tel and r_key in tel:
                comp_torques[comp] = {
                    "left": tel[l_key],
                    "right": tel[r_key],
                    "combined": tel[l_key] + tel[r_key],
                }

        if len(comp_torques) < 2:
            continue

        # Compute cancellation: sum(|individual|) - |sum of all|
        abs_sum = np.zeros(len(next(iter(comp_torques.values()))["combined"]))
        sum_torques = np.zeros_like(abs_sum)
        for comp_data in comp_torques.values():
            abs_sum += np.abs(comp_data["combined"])
            sum_torques += comp_data["combined"]

        cancellation = abs_sum - np.abs(sum_torques)
        cancel_rms = rms(cancellation)
        cancel_peak = peak(cancellation)
        cancel_total = float(np.sum(cancellation))

        # Per-component authority share
        authority = {}
        total_rms_abs = sum(rms(np.abs(cd["combined"])) for cd in comp_torques.values())
        for comp, cd in comp_torques.items():
            authority[comp] = rms(np.abs(cd["combined"])) / max(total_rms_abs, 1e-9)

        # Sign opposition: fraction of steps where two largest components
        # have opposite signs on either left or right joint
        comp_list = list(comp_torques.items())
        sign_opposition = 0.0
        if len(comp_list) >= 2:
            # Find the two components with largest RMS
            rms_values = {c: rms(cd["combined"]) for c, cd in comp_torques.items()}
            sorted_comps = sorted(rms_values, key=rms_values.get, reverse=True)
            if len(sorted_comps) >= 2:
                c1, c2 = sorted_comps[0], sorted_comps[1]
                for side in ["left", "right"]:
                    s1 = np.sign(comp_torques[c1][side])
                    s2 = np.sign(comp_torques[c2][side])
                    # Opposite signs: s1 * s2 < 0
                    # Only count when both are non-zero (|tau| > 0.01 Nm)
                    active = (np.abs(comp_torques[c1][side]) > 0.01) & (np.abs(comp_torques[c2][side]) > 0.01)
                    if np.sum(active) > 0:
                        opposite = (s1 * s2 < 0) & active
                        sign_opposition = max(sign_opposition, float(np.sum(opposite) / np.sum(active)))

        result[joint_group] = {
            "joints": f"{jn_l}, {jn_r}",
            "components": info["components"],
            "cancellation_rms_nm": cancel_rms,
            "cancellation_peak_nm": cancel_peak,
            "cancellation_total_nm": cancel_total,
            "authority_share": authority,
            "sign_opposition_frac": sign_opposition,
            "cancellation_mean_nm": float(np.mean(cancellation)),
            "cancel_steps_frac": float(np.sum(cancellation > 0.01) / max(len(cancellation), 1)),
        }

    # Online cancellation metrics from diag
    online_metrics = {}
    for key in ["cancel_hip_yaw", "cancel_hip_roll", "cancel_hip_pitch", "cancel_knee", "cancel_total"]:
        if key in tel:
            online_metrics[key] = {
                "mean": float(np.mean(tel[key])),
                "rms": rms(tel[key]),
                "peak": peak(tel[key]),
                "total": float(np.sum(tel[key])),
            }
    result["online_cancellation"] = online_metrics

    return result


# ── Cross-Coupling Analysis ───────────────────────────────────────────────────


def analyze_cross_coupling(tel: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze cross-coupling between controller components via shared state."""
    result = {}

    # 1. Pitch torque vs support error: does sagittal balance fight support?
    if "tau_preclip_4" in tel and "height_error" in tel:
        tau_balance = tel["tau_preclip_4"] + tel["tau_preclip_9"]  # wheel torque
        support_err = np.abs(tel.get("height_error", np.zeros_like(tau_balance)))
        if len(tau_balance) > 10:
            # Correlation between balance torque magnitude and support error
            corr = np.corrcoef(np.abs(tau_balance), support_err)[0, 1]
            result["balance_vs_support_error_corr"] = float(corr) if not np.isnan(corr) else 0.0

    # 2. Support correction vs pitch phase
    if "tau_support_ff_hy_l" in tel and "pitch_deg" in tel and "pitch_rate_deg_s" in tel:
        sup_ff = tel["tau_support_ff_hy_l"] + tel["tau_support_ff_hy_r"]
        pitch = tel["pitch_deg"]
        pitch_rate = tel["pitch_rate_deg_s"]
        if len(pitch) > 10:
            # Cross-correlation at various lags
            corr_pitch = np.corrcoef(sup_ff, pitch)[0, 1] if len(sup_ff) == len(pitch) else 0
            corr_prate = np.corrcoef(sup_ff, pitch_rate)[0, 1] if len(sup_ff) == len(pitch_rate) else 0
            result["support_ff_vs_pitch_corr"] = float(corr_pitch) if not np.isnan(corr_pitch) else 0.0
            result["support_ff_vs_pitch_rate_corr"] = float(corr_prate) if not np.isnan(corr_prate) else 0.0

    # 3. Yaw/mode-div torque vs sagittal drift
    if "tau_yaw_l" in tel and "com_vx" in tel:
        yaw_tau = tel["tau_yaw_l"] + tel["tau_yaw_r"]
        mode_div_tau = tel.get("tau_mode_div_l", np.zeros_like(yaw_tau)) + tel.get("tau_mode_div_r", np.zeros_like(yaw_tau))
        sag_vel = tel["com_vx"]
        if len(yaw_tau) > 10:
            corr_yaw = np.corrcoef(yaw_tau, sag_vel)[0, 1]
            corr_md = np.corrcoef(mode_div_tau, sag_vel)[0, 1] if len(mode_div_tau) == len(sag_vel) else 0
            result["yaw_torque_vs_sagittal_vel_corr"] = float(corr_yaw) if not np.isnan(corr_yaw) else 0.0
            result["mode_div_vs_sagittal_vel_corr"] = float(corr_md) if not np.isnan(corr_md) else 0.0

    # 4. Lateral torque vs hip_yaw divergence
    if "tau_lateral_l" in tel and "hip_yaw_div_error" in tel:
        lat_tau = tel["tau_lateral_l"] + tel["tau_lateral_r"]
        hy_div = tel["hip_yaw_div_error"]
        if len(lat_tau) > 10 and len(lat_tau) == len(hy_div):
            corr = np.corrcoef(np.abs(lat_tau), np.abs(hy_div))[0, 1]
            result["lateral_vs_hip_yaw_div_corr"] = float(corr) if not np.isnan(corr) else 0.0

    # 5. Posture authority vs balance authority
    if all(k in tel for k in ["tau_posture_kn_l", "tau_preclip_4"]):
        posture_power = rms(tel["tau_posture_kn_l"] + tel["tau_posture_kn_r"] +
                           tel["tau_posture_hp_l"] + tel["tau_posture_hp_r"])
        balance_power = rms(tel["tau_preclip_4"] + tel["tau_preclip_9"])
        result["posture_balance_power_ratio"] = float(posture_power / max(balance_power, 1e-9))
        result["posture_power_nm"] = float(posture_power)
        result["balance_power_nm"] = float(balance_power)

    # 6. Total torque vs contact quality
    if "tau_preclip_4" in tel and "contact_valid" in tel:
        total_tau = np.sqrt(tel["tau_preclip_4"]**2 + tel["tau_preclip_9"]**2)
        contact = tel["contact_valid"]
        if len(total_tau) > 10:
            # Mean torque when contact is good vs poor
            good_contact = contact > 0.5
            poor_contact = ~good_contact
            tau_good = np.mean(np.abs(total_tau[good_contact])) if np.any(good_contact) else 0
            tau_poor = np.mean(np.abs(total_tau[poor_contact])) if np.any(poor_contact) else 0
            result["torque_when_good_contact"] = float(tau_good)
            result["torque_when_poor_contact"] = float(tau_poor)

    return result


# ── Saturation & Clipping Analysis ─────────────────────────────────────────────


def analyze_saturation(tel: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze torque saturation and clipping behavior."""
    result = {}

    # Pre-clip vs post-clip delta
    if "tau_preclip_4" in tel and "tau_postclip_4" in tel:
        clip_delta_4 = tel["tau_preclip_4"] - tel["tau_postclip_4"]
        clip_delta_9 = tel["tau_preclip_9"] - tel["tau_postclip_9"]
        result["wheel_clip_rms_nm"] = rms(np.concatenate([clip_delta_4, clip_delta_9]))
        result["wheel_clip_peak_nm"] = peak(np.concatenate([clip_delta_4, clip_delta_9]))
        result["wheel_clip_fraction"] = float(np.sum(np.abs(clip_delta_4) + np.abs(clip_delta_9) > 0.001) / max(len(clip_delta_4), 1))

    # Saturation attribution
    for attr in ["sat_attr_sagittal", "sat_attr_posture", "sat_attr_yaw", "sat_attr_lateral"]:
        if attr in tel:
            result[f"{attr}_total"] = float(np.sum(tel[attr]))
            result[f"{attr}_frac"] = float(np.mean(tel[attr] > 0.5))

    # Rate-limit attribution
    for attr in ["rate_attr_balance", "rate_attr_posture"]:
        if attr in tel:
            result[f"{attr}_total"] = float(np.sum(tel[attr]))
            result[f"{attr}_frac"] = float(np.mean(tel[attr] > 0.5))

    return result


# ── Height-Phase Analysis ──────────────────────────────────────────────────────


def analyze_by_height_phase(tel: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Analyze conflicts stratified by height region and pitch phase."""
    result = {}

    if "com_z" not in tel:
        return result

    com_z = tel["com_z"]
    pitch = tel.get("pitch_deg", np.zeros_like(com_z))

    # Height bins: low (<0.35), mid (0.35-0.43), high (>0.43)
    height_regions = {
        "low": com_z < 0.35,
        "mid": (com_z >= 0.35) & (com_z < 0.43),
        "high": com_z >= 0.43,
    }

    # Pitch phase bins: pitching forward (>0.5 deg), backward (<-0.5 deg), neutral
    pitch_phases = {
        "forward": pitch > 0.5,
        "backward": pitch < -0.5,
        "neutral": (pitch >= -0.5) & (pitch <= 0.5),
    }

    by_height = {}
    for region, mask in height_regions.items():
        if np.sum(mask) < 5:
            continue
        by_height[region] = {}
        for key in ["cancel_total", "cancel_hip_yaw", "cancel_hip_roll",
                     "cancel_hip_pitch", "cancel_knee"]:
            if key in tel:
                by_height[region][f"{key}_mean"] = float(np.mean(tel[key][mask]))

    result["by_height"] = by_height

    by_pitch_phase = {}
    for phase, mask in pitch_phases.items():
        if np.sum(mask) < 5:
            continue
        by_pitch_phase[phase] = {}
        for key in ["cancel_total", "cancel_hip_yaw", "cancel_hip_pitch"]:
            if key in tel:
                by_pitch_phase[phase][f"{key}_mean"] = float(np.mean(tel[key][mask]))

    result["by_pitch_phase"] = by_pitch_phase

    return result


# ── Report Generation ──────────────────────────────────────────────────────────


def generate_report(all_analyses: List[Dict], output_path: Path, metadata: Optional[Dict] = None):
    """Generate comprehensive audit report."""
    lines = []
    lines.append("# K2 Current Controller Interaction Audit\n")
    lines.append(f"**Phase:** 3 — System Architecture Audit")
    lines.append(f"**Scenarios analyzed:** {len(all_analyses)}")
    if metadata:
        for k, v in metadata.items():
            lines.append(f"**{k}:** {v}")
    lines.append("")

    # ── 1. Executive Summary ──
    lines.append("## 1. Executive Summary\n")

    # Aggregate across all scenarios
    agg_cancel = defaultdict(list)
    agg_authority = defaultdict(lambda: defaultdict(list))
    agg_cross = defaultdict(list)
    agg_sat = defaultdict(list)

    for a in all_analyses:
        conflicts = a.get("conflicts", {})
        for jg, cd in conflicts.items():
            agg_cancel[f"{jg}_rms"].append(cd.get("cancellation_rms_nm", 0))
            agg_cancel[f"{jg}_peak"].append(cd.get("cancellation_peak_nm", 0))
            for comp, share in cd.get("authority_share", {}).items():
                agg_authority[jg][comp].append(share)

        cross = a.get("cross_coupling", {})
        for k, v in cross.items():
            if isinstance(v, (int, float)):
                agg_cross[k].append(v)

        sat = a.get("saturation", {})
        for k, v in sat.items():
            if isinstance(v, (int, float)):
                agg_sat[k].append(v)

    lines.append("### Conflict Severity Ranking\n")
    lines.append("| Rank | Joint Group | Mean Cancel RMS (Nm) | Mean Cancel Peak (Nm) | Top Competing Components |")
    lines.append("|------|-------------|---------------------|----------------------|--------------------------|")
    rankings = []
    for jg in ["knee", "hip_pitch", "hip_yaw", "hip_roll"]:
        rms_key = f"{jg}_rms"
        peak_key = f"{jg}_peak"
        if rms_key in agg_cancel and agg_cancel[rms_key]:
            mean_rms = np.mean(agg_cancel[rms_key])
            mean_peak = np.mean(agg_cancel[peak_key]) if agg_cancel[peak_key] else 0
            top_comps = sorted(agg_authority.get(jg, {}).items(),
                              key=lambda x: np.mean(x[1]) if x[1] else 0, reverse=True)
            top_str = " vs ".join([f"{c}({np.mean(v):.0%})" for c, v in top_comps[:2]])
            rankings.append((mean_rms, jg, mean_rms, mean_peak, top_str))
    rankings.sort(reverse=True)
    for rank, (_, jg, rms_v, peak_v, comps) in enumerate(rankings):
        lines.append(f"| {rank+1} | **{jg}** | {rms_v:.3f} | {peak_v:.3f} | {comps} |")
    lines.append("")

    # ── 2. Per-Joint Conflict Detail ──
    lines.append("## 2. Same-Joint Torque Cancellation\n")

    for jg in ["knee", "hip_pitch", "hip_yaw", "hip_roll"]:
        lines.append(f"### 2.{['a','b','c','d'][['knee','hip_pitch','hip_yaw','hip_roll'].index(jg)]} {jg.replace('_', ' ').title()} Joints\n")

        # Per-scenario data
        lines.append("| Scenario | Cancel RMS (Nm) | Cancel Peak (Nm) | Sign Opposition | Active Steps % |")
        lines.append("|----------|----------------|-------------------|-----------------|----------------|")
        for a in all_analyses:
            cd = a.get("conflicts", {}).get(jg, {})
            if cd:
                sid = a.get("scenario", "unknown")[:40]
                lines.append(f"| {sid} | {cd.get('cancellation_rms_nm',0):.3f} | "
                           f"{cd.get('cancellation_peak_nm',0):.3f} | "
                           f"{cd.get('sign_opposition_frac',0):.1%} | "
                           f"{cd.get('cancel_steps_frac',0):.1%} |")
        lines.append("")

        # Authority share
        lines.append("**Authority Share by Component:**\n")
        lines.append("| Component | Mean Share | Min | Max |")
        lines.append("|-----------|------------|-----|-----|")
        for comp in sorted(agg_authority.get(jg, {}).keys()):
            shares = agg_authority[jg][comp]
            if shares:
                lines.append(f"| {comp} | {np.mean(shares):.1%} | {np.min(shares):.1%} | {np.max(shares):.1%} |")
        lines.append("")

    # ── 3. Cross-Coupling ──
    lines.append("## 3. Cross-Coupling Analysis\n")
    lines.append("| Metric | Mean | Std | Interpretation |")
    lines.append("|--------|------|-----|----------------|")
    cross_interpretations = {
        "balance_vs_support_error_corr": "Balance torque magnitude correlation with support error (↑ = balance fights support)",
        "support_ff_vs_pitch_corr": "Support FF correlation with pitch angle (↑ = support changes with pitch, may fight balance)",
        "yaw_torque_vs_sagittal_vel_corr": "Yaw torque correlation with sagittal velocity (↑ = yaw induces drift)",
        "mode_div_vs_sagittal_vel_corr": "Mode-div torque correlation with sagittal velocity",
        "lateral_vs_hip_yaw_div_corr": "Lateral torque correlation with hip-yaw divergence (↑ = roll control couples to yaw)",
        "posture_balance_power_ratio": "Posture:Balance power ratio (>1 = posture dominates, <1 = balance dominates)",
    }
    for metric, interpretation in cross_interpretations.items():
        if metric in agg_cross:
            vals = [v for v in agg_cross[metric] if not np.isnan(v)]
            if vals:
                lines.append(f"| {metric} | {np.mean(vals):.3f} | {np.std(vals):.3f} | {interpretation} |")
    lines.append("")

    # ── 4. Saturation & Clipping ──
    lines.append("## 4. Torque Saturation & Clipping\n")
    lines.append("| Metric | Mean | Max | Interpretation |")
    lines.append("|--------|------|-----|----------------|")
    sat_metrics = [
        ("wheel_clip_rms_nm", "Wheel torque clipping RMS (pre-clip - post-clip)"),
        ("wheel_clip_fraction", "Fraction of steps with wheel torque clipped"),
        ("sat_attr_sagittal_frac", "Fraction of steps with sagittal saturation"),
        ("sat_attr_posture_frac", "Fraction of steps with posture saturation"),
        ("sat_attr_yaw_frac", "Fraction of steps with yaw saturation"),
        ("rate_attr_balance_frac", "Fraction of steps with balance rate-limited"),
        ("rate_attr_posture_frac", "Fraction of steps with posture rate-limited"),
    ]
    for metric, desc in sat_metrics:
        if metric in agg_sat:
            vals = [v for v in agg_sat[metric] if not np.isnan(v)]
            if vals:
                lines.append(f"| {metric} | {np.mean(vals):.4f} | {np.max(vals):.4f} | {desc} |")
    lines.append("")

    # ── 5. Height-Phase Stratification ──
    lines.append("## 5. Conflict by Height Region & Pitch Phase\n")

    for a in all_analyses[:3]:  # First 3 scenarios
        sid = a.get("scenario", "unknown")
        strat = a.get("height_phase", {})
        by_h = strat.get("by_height", {})
        by_p = strat.get("by_pitch_phase", {})

        if by_h:
            lines.append(f"### {sid} — Cancellation by Height\n")
            lines.append("| Region | cancel_total | cancel_hip_yaw | cancel_hip_pitch | cancel_knee |")
            lines.append("|--------|-------------|----------------|------------------|-------------|")
            for region in ["low", "mid", "high"]:
                if region in by_h:
                    hd = by_h[region]
                    lines.append(f"| {region} | {hd.get('cancel_total_mean',0):.3f} | "
                               f"{hd.get('cancel_hip_yaw_mean',0):.3f} | "
                               f"{hd.get('cancel_hip_pitch_mean',0):.3f} | "
                               f"{hd.get('cancel_knee_mean',0):.3f} |")
            lines.append("")

        if by_p:
            lines.append(f"### {sid} — Cancellation by Pitch Phase\n")
            lines.append("| Phase | cancel_total | cancel_hip_yaw | cancel_hip_pitch |")
            lines.append("|-------|-------------|----------------|------------------|")
            for phase in ["forward", "neutral", "backward"]:
                if phase in by_p:
                    pd = by_p[phase]
                    lines.append(f"| {phase} | {pd.get('cancel_total_mean',0):.3f} | "
                               f"{pd.get('cancel_hip_yaw_mean',0):.3f} | "
                               f"{pd.get('cancel_hip_pitch_mean',0):.3f} |")
            lines.append("")

    # ── 6. Per-Scenario Summary ──
    lines.append("## 6. Per-Scenario Summary\n")
    lines.append("| Scenario | Cancel Total RMS | Cancel Peak | Worst Joint | Sig. Cross-Coupling |")
    lines.append("|----------|-----------------|-------------|-------------|---------------------|")
    for a in all_analyses:
        sid = a.get("scenario", "unknown")[:35]
        online = a.get("conflicts", {}).get("online_cancellation", {})
        ct = online.get("cancel_total", {})
        cancel_rms = ct.get("rms", 0)
        cancel_peak = ct.get("peak", 0)

        worst_joint = ""
        worst_rms = 0
        for jg in ["knee", "hip_pitch", "hip_yaw", "hip_roll"]:
            cd = a.get("conflicts", {}).get(jg, {})
            if cd.get("cancellation_rms_nm", 0) > worst_rms:
                worst_rms = cd["cancellation_rms_nm"]
                worst_joint = jg

        cross = a.get("cross_coupling", {})
        sig_cross = []
        for k, v in cross.items():
            if isinstance(v, float) and abs(v) > 0.3:
                sig_cross.append(f"{k}={v:.2f}")
        cross_str = ", ".join(sig_cross[:3]) if sig_cross else "—"

        lines.append(f"| {sid} | {cancel_rms:.2f} | {cancel_peak:.2f} | {worst_joint} | {cross_str} |")
    lines.append("")

    # ── 7. Findings & Recommendations ──
    lines.append("## 7. Ranked Findings\n")

    # Compute ranked findings from the data
    findings = []

    # Finding 1: Cancellation magnitude
    all_cancel_rms = []
    for a in all_analyses:
        online = a.get("conflicts", {}).get("online_cancellation", {})
        ct = online.get("cancel_total", {})
        if ct.get("rms", 0) > 0:
            all_cancel_rms.append(ct["rms"])
    if all_cancel_rms:
        mean_cancel = np.mean(all_cancel_rms)
        findings.append({
            "severity": "HIGH" if mean_cancel > 3 else "MEDIUM",
            "title": "Significant torque cancellation across all scenarios",
            "detail": f"Mean cancellation RMS: {mean_cancel:.2f} Nm. "
                     f"Components are independently summed with no coordination, "
                     f"resulting in wasted control effort.",
            "mechanism": "Authority allocation — add smooth weight scheduler",
        })

    # Finding 2: Worst conflict joint
    for jg in ["knee", "hip_pitch", "hip_yaw", "hip_roll"]:
        rms_key = f"{jg}_rms"
        if rms_key in agg_cancel:
            vals = [v for v in agg_cancel[rms_key] if v > 0]
            if vals and np.mean(vals) > 1.0:
                findings.append({
                    "severity": "HIGH" if np.mean(vals) > 2 else "MEDIUM",
                    "title": f"{jg.replace('_',' ').title()} joints: posture vs feedforward conflict",
                    "detail": f"Mean cancellation RMS: {np.mean(vals):.2f} Nm. "
                             f"The shape posture PD controller and empirical support "
                             f"feedforward write opposing torques to the same joints.",
                    "mechanism": f"Coordinate posture PD with support FF at {jg} joints",
                })

    # Finding 3: Cross-coupling
    for metric in ["balance_vs_support_error_corr", "yaw_torque_vs_sagittal_vel_corr"]:
        if metric in agg_cross:
            vals = [abs(v) for v in agg_cross[metric] if not np.isnan(v)]
            if vals and np.mean(vals) > 0.2:
                findings.append({
                    "severity": "MEDIUM",
                    "title": f"Cross-coupling detected: {metric}",
                    "detail": f"Mean |correlation|: {np.mean(vals):.2f}. "
                             f"Controller components affect shared state variables "
                             f"without awareness of each other's effects.",
                    "mechanism": "Add cross-coupling terms to shared state estimate",
                })

    for i, f in enumerate(sorted(findings, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["severity"]])):
        lines.append(f"### Finding {i+1}: [{f['severity']}] {f['title']}\n")
        lines.append(f"**Detail:** {f['detail']}\n")
        lines.append(f"**Recommended mechanism:** {f['mechanism']}\n")
        lines.append("")

    # ── 8. JSON Data ──
    lines.append("## 8. Data\n")
    json_path = str(output_path).replace(".md", ".json")
    lines.append(f"Full analysis data: `{json_path}`")

    # Write
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Write JSON
    json_data = []
    for a in all_analyses:
        clean = {}
        for k, v in a.items():
            if isinstance(v, dict):
                clean[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                           for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v
        json_data.append(clean)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(description="K2 Controller Conflict Analyzer")
    p.add_argument("--input-dir", type=str, required=True,
                   help="Directory containing instrumented telemetry CSVs")
    p.add_argument("--output", type=str, required=True,
                   help="Output Markdown report path")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    runs = discover_runs(input_dir)
    if not runs:
        print(f"ERROR: No telemetry CSV files found in {input_dir}")
        return 1

    print(f"Found {len(runs)} telemetry runs")

    all_analyses = []
    for scenario_name, csv_path in runs:
        print(f"  Analyzing {scenario_name} ...")
        tel = load_telemetry(csv_path)
        if tel is None:
            print(f"    SKIP: could not load CSV")
            continue

        analysis = {
            "scenario": scenario_name,
            "steps": len(next(iter(tel.values()))),
        }
        analysis["conflicts"] = analyze_conflicts(tel)
        analysis["cross_coupling"] = analyze_cross_coupling(tel)
        analysis["saturation"] = analyze_saturation(tel)
        analysis["height_phase"] = analyze_by_height_phase(tel)
        all_analyses.append(analysis)

    if not all_analyses:
        print("ERROR: No runs successfully analyzed")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "input_dir": str(input_dir),
        "runs_analyzed": len(all_analyses),
        "analyzer_version": "1.0.0",
    }
    report = generate_report(all_analyses, output_path, metadata)
    print(f"\nReport written to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
