#!/usr/bin/env python3
"""
Phase 1: K1 Torque Headroom Audit

Computes torque budget utilization for every timestep of the K1 focused recovery run.

Output: outputs/system_audit/torque_headroom/

Key question: Is K1 authority-starved?
"""

import csv
import json
import statistics
import sys
import os
from collections import defaultdict

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Paths ──────────────────────────────────────────────────────────────────
TELEMETRY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "d_baseline_single_90n_10step_push_step300_3000",
    "telemetry_1782262602.csv",
)
OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "system_audit", "torque_headroom",
)
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "torque_headroom_audit.json")
OUTPUT_REPORT = os.path.join(OUTPUT_DIR, "torque_headroom_report.md")

# ── Load telemetry ─────────────────────────────────────────────────────────
print("Loading K1 telemetry...")
with open(TELEMETRY_PATH, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

N = len(rows)
print(f"  {N} steps loaded")

# ── Determine push window ──────────────────────────────────────────────────
def _safe_bool(val):
    """Handle push_active which may be 'True'/'False' string or float."""
    if isinstance(val, str) and val in ("True", "False"):
        return 1.0 if val == "True" else 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

push_active_steps = [i for i, r in enumerate(rows) if _safe_bool(r.get("push_active", 0)) > 0.5]
if push_active_steps:
    push_start = push_active_steps[0]
    push_end = push_active_steps[-1]
else:
    push_start = push_end = None
    print("  WARNING: No push steps detected")

print(f"  Push window: step {push_start} to {push_end}" if push_start is not None else "  No push window")

# ── Extract torque data ────────────────────────────────────────────────────
max_tau_wheel = 5.0  # hardware limit per wheel from controller __init__

data = []
for i, r in enumerate(rows):
    tau_left_abs = abs(float(r["tau_wheel_total_clipped_left"]))
    tau_right_abs = abs(float(r["tau_wheel_total_clipped_right"]))
    tau_max_abs = max(tau_left_abs, tau_right_abs)
    tau_common_clipped = abs(float(r["tau_common_clipped"]))

    # Per-wheel headroom against 5.0 Nm hardware limit
    headroom_left = max_tau_wheel - tau_left_abs
    headroom_right = max_tau_wheel - tau_right_abs
    headroom_min = min(headroom_left, headroom_right)
    headroom_pct = 100.0 * headroom_min / max_tau_wheel if max_tau_wheel > 0 else 0.0

    # Utilization ratio: |tau| / max_tau_wheel
    util_ratio = tau_max_abs / max_tau_wheel

    # Saturation margin: how far from clipping
    sat_flag_l = _safe_bool(r.get("wheel_torque_saturation_left", 0))
    sat_flag_r = _safe_bool(r.get("wheel_torque_saturation_right", 0))

    # Common torque components
    tau_pitch_val = float(r.get("sagittal_term_pitch", 0))
    tau_pitch_rate_val = float(r.get("sagittal_term_pitch_rate", 0))
    tau_cp_val = float(r.get("sagittal_term_cp", 0))
    tau_com_vy_val = float(r.get("sagittal_term_com_vy", 0))
    tau_wheel_vel_l = float(r.get("sagittal_term_wheel_vel_left", 0))
    tau_wheel_vel_r = float(r.get("sagittal_term_wheel_vel_right", 0))
    tau_bal_raw = float(r.get("sagittal_balance_torque_raw", 0))
    tau_bal_clip = float(r.get("sagittal_balance_torque_clipped", 0))
    tau_bal_final = float(r.get("sagittal_balance_torque_final", 0))
    tau_pos_val = float(r.get("tau_position", 0))
    tau_pos_clip = float(r.get("tau_position_clipped", 0))
    tau_sup_vel = float(r.get("tau_support_velocity", 0))

    data.append({
        "step": i,
        "tau_left_abs": tau_left_abs,
        "tau_right_abs": tau_right_abs,
        "tau_max_abs": tau_max_abs,
        "tau_common_clipped": tau_common_clipped,
        "headroom_left": headroom_left,
        "headroom_right": headroom_right,
        "headroom_min": headroom_min,
        "headroom_pct": headroom_pct,
        "util_ratio": util_ratio,
        "sat_left": sat_flag_l,
        "sat_right": sat_flag_r,
        "is_push": push_start is not None and push_start <= i <= push_end,
        # Breakdown
        "tau_pitch": tau_pitch_val,
        "tau_pitch_rate": tau_pitch_rate_val,
        "tau_cp": tau_cp_val,
        "tau_com_vy": tau_com_vy_val,
        "tau_wheel_vel_l": tau_wheel_vel_l,
        "tau_wheel_vel_r": tau_wheel_vel_r,
        "tau_bal_raw": tau_bal_raw,
        "tau_bal_clipped": tau_bal_clip,
        "tau_bal_final": tau_bal_final,
        "tau_position": tau_pos_val,
        "tau_position_clipped": tau_pos_clip,
        "tau_support_velocity": tau_sup_vel,
    })

# ── Phase tagging ──────────────────────────────────────────────────────────
# pre_push: before push_start
# during_push: push_start to push_end
# post_push_early: push_end+1 to push_end+500
# post_push_late: remaining

for d in data:
    if push_start is None:
        d["phase"] = "pre_push"
    elif d["step"] < push_start:
        d["phase"] = "pre_push"
    elif d["step"] <= push_end:
        d["phase"] = "during_push"
    elif d["step"] <= push_end + 500:
        d["phase"] = "post_push_early"
    else:
        d["phase"] = "post_push_late"

# ── Bin utilization ratios ─────────────────────────────────────────────────
BINS = [
    (0.0, 0.10, "0-10%"),
    (0.10, 0.20, "10-20%"),
    (0.20, 0.40, "20-40%"),
    (0.40, 0.60, "40-60%"),
    (0.60, 0.80, "60-80%"),
    (0.80, 1.00, "80-100%"),
    (1.00, float("inf"), ">100%"),
]

def count_bins(dataset):
    counts = {label: 0 for _, _, label in BINS}
    for d_item in dataset:
        r = d_item["util_ratio"]
        for lo, hi, label in BINS:
            if lo <= r < hi:
                counts[label] += 1
                break
    return counts

phase_datasets = {
    "all": data,
    "pre_push": [d for d in data if d["phase"] == "pre_push"],
    "during_push": [d for d in data if d["phase"] == "during_push"],
    "post_push_early": [d for d in data if d["phase"] == "post_push_early"],
    "post_push_late": [d for d in data if d["phase"] == "post_push_late"],
}

distribution = {}
for phase_name, dataset in phase_datasets.items():
    counts = count_bins(dataset)
    total_steps = len(dataset)
    distribution[phase_name] = {
        "total_steps": total_steps,
        "counts": counts,
        "pct": {k: round(100.0 * v / total_steps, 2) if total_steps > 0 else 0.0 for k, v in counts.items()},
    }

# ── Headroom statistics by phase ───────────────────────────────────────────
def headroom_stats(dataset):
    if not dataset:
        return {}
    hrs = [d["headroom_min"] for d in dataset]
    pcts = [d["headroom_pct"] for d in dataset]
    utils = [d["util_ratio"] for d in dataset]
    sats = [1.0 if (d["sat_left"] or d["sat_right"]) else 0.0 for d in dataset]
    return {
        "count": len(dataset),
        "headroom_nm": {
            "min": round(min(hrs), 4),
            "max": round(max(hrs), 4),
            "mean": round(sum(hrs) / len(hrs), 4),
            "median": round(sorted(hrs)[len(hrs) // 2], 4),
        },
        "headroom_pct": {
            "min": round(min(pcts), 2),
            "max": round(max(pcts), 2),
            "mean": round(sum(pcts) / len(pcts), 2),
            "median": round(sorted(pcts)[len(pcts) // 2], 2),
        },
        "utilization_ratio": {
            "min": round(min(utils), 4),
            "max": round(max(utils), 4),
            "mean": round(sum(utils) / len(utils), 4),
            "mean_pct": round(100.0 * sum(utils) / len(utils), 2),
        },
        "saturation_rate_pct": round(100.0 * sum(sats) / len(sats), 2),
        "steps_above_80pct": sum(1 for u in utils if u > 0.80),
        "steps_above_90pct": sum(1 for u in utils if u > 0.90),
        "steps_above_100pct": sum(1 for u in utils if u > 1.00),
    }

phase_stats = {name: headroom_stats(ds) for name, ds in phase_datasets.items()}

# ── Torque component breakdown ─────────────────────────────────────────────
def component_breakdown(dataset):
    if not dataset:
        return {}
    keys = ["tau_pitch", "tau_pitch_rate", "tau_cp", "tau_com_vy",
            "tau_position", "tau_position_clipped", "tau_support_velocity",
            "tau_bal_raw", "tau_bal_clipped", "tau_bal_final"]
    result = {}
    for k in keys:
        vals = [abs(d[k]) for d in dataset]
        result[k] = {
            "mean_abs": round(sum(vals) / len(vals), 4),
            "max_abs": round(max(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0, 4),
        }
    return result

component_stats = {name: component_breakdown(ds) for name, ds in phase_datasets.items()}

# ── Authority starvation diagnosis ─────────────────────────────────────────
all_utils = [d["util_ratio"] for d in data]
all_headroom_pcts = [d["headroom_pct"] for d in data]

starved_steps = sum(1 for h in all_headroom_pcts if h < 10.0)
starved_pct = 100.0 * starved_steps / len(all_headroom_pcts)

constricted_steps = sum(1 for h in all_headroom_pcts if h < 5.0)
constricted_pct = 100.0 * constricted_steps / len(all_headroom_pcts)

# Post-push headroom analysis
post_push = [d for d in data if d["phase"] in ("post_push_early", "post_push_late")]
post_push_starved = sum(1 for d in post_push if d["headroom_pct"] < 10.0)
post_push_pct = 100.0 * post_push_starved / len(post_push) if post_push else 0

# ── Time-series headroom at key phases ─────────────────────────────────────
def headroom_at_steps(step_indices):
    """Get headroom stats at specific step ranges."""
    result = {}
    for label, start, end in step_indices:
        subset = [d for d in data if start <= d["step"] < end]
        if subset:
            hrs = [d["headroom_pct"] for d in subset]
            result[label] = {
                "steps": f"{start}-{end}",
                "mean_headroom_pct": round(sum(hrs) / len(hrs), 2),
                "min_headroom_pct": round(min(hrs), 2),
            }
    return result

key_windows = headroom_at_steps([
    ("pre_push_hold", 200, 300),
    ("during_push", push_start, push_end + 1) if push_start else ("during_push", 0, 0),
    ("immediate_post_push", push_end + 1, push_end + 50) if push_end else ("immediate_post_push", 0, 0),
    ("post_push_100", push_end + 50, push_end + 150) if push_end else ("post_push_100", 0, 0),
    ("post_push_500", push_end + 450, push_end + 550) if push_end else ("post_push_500", 0, 0),
    ("post_push_1000", push_end + 950, push_end + 1050) if push_end else ("post_push_1000", 0, 0),
    ("post_push_2000", push_end + 1950, push_end + 2050) if push_end else ("post_push_2000", 0, 0),
    ("late_phase", 2500, 3000),
])

# ── Answer the question ────────────────────────────────────────────────────
# Is K1 authority-starved?
# Starvation = mean headroom < 10% persistently
overall_mean_headroom = sum(all_headroom_pcts) / len(all_headroom_pcts)
is_starved = overall_mean_headroom < 10.0
is_constricted = overall_mean_headroom < 20.0

# During critical recovery (post_push_early), is authority sufficient?
pp_early_stats = phase_stats.get("post_push_early", {})
pp_early_mean_hr = pp_early_stats.get("headroom_pct", {}).get("mean", 100)
recovery_starved = pp_early_mean_hr < 10.0

# Is saturation actually occurring?
all_sats = sum(1 for d in data if d["sat_left"] or d["sat_right"])
sat_rate = 100.0 * all_sats / len(data)

verdict = "NOT_AUTHORITY_STARVED"
if recovery_starved and sat_rate > 1.0:
    verdict = "AUTHORITY_STARVED_DURING_RECOVERY"
elif sat_rate > 5.0:
    verdict = "AUTHORITY_CONSTRAINED_BUT_NOT_STARVED"
elif is_constricted:
    verdict = "AUTHORITY_COMFORTABLE_WITH_OCCASIONAL_SPIKES"
else:
    verdict = "AMPLE_AUTHORITY_HEADROOM"

# ── Compile audit dict ─────────────────────────────────────────────────────
audit = {
    "audit": "torque_headroom",
    "target": "K1_PITCH_RATE_NOTCH_V1",
    "run": "high_0p480_90N_push_step300_3000steps",
    "max_tau_wheel_nm": max_tau_wheel,
    "total_steps": N,
    "push_window": {"start_step": push_start, "end_step": push_end},
    "verdict": verdict,
    "overall": {
        "mean_headroom_nm": round(overall_mean_headroom * max_tau_wheel / 100.0, 4),
        "mean_headroom_pct": round(overall_mean_headroom, 2),
        "is_authority_starved": is_starved,
        "is_authority_constricted": is_constricted,
        "starved_steps_pct": round(starved_pct, 2),
        "constricted_steps_pct": round(constricted_pct, 2),
        "saturation_event_rate_pct": round(sat_rate, 2),
    },
    "utilization_distribution": distribution,
    "phase_statistics": phase_stats,
    "component_breakdown": component_stats,
    "key_windows": key_windows,
    "post_push_recovery": {
        "post_push_starved_pct": round(post_push_pct, 2),
        "recovery_phase_starved": recovery_starved,
    },
}

# ── Save JSON ─────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=2)
print(f"\nJSON audit saved to: {OUTPUT_JSON}")

# ── Print summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("K1 TORQUE HEADROOM AUDIT — SUMMARY")
print("=" * 70)
print(f"\nMax wheel torque budget: {max_tau_wheel} Nm per wheel")
print(f"Total steps: {N}")
print(f"Push window: step {push_start} to {push_end}")

print(f"\n── Overall Headroom ──")
print(f"  Mean headroom: {overall_mean_headroom:.1f}% ({overall_mean_headroom * max_tau_wheel / 100:.2f} Nm)")
print(f"  Steps < 10% headroom: {starved_steps}/{N} ({starved_pct:.1f}%)")
print(f"  Steps < 5% headroom: {constricted_steps}/{N} ({constricted_pct:.1f}%)")
print(f"  Saturation events: {all_sats}/{N} ({sat_rate:.2f}%)")

print(f"\n── Utilization Distribution ──")
for phase_name in ["all", "pre_push", "during_push", "post_push_early", "post_push_late"]:
    dist = distribution.get(phase_name, {})
    if not dist:
        continue
    print(f"  {phase_name} ({dist['total_steps']} steps):")
    for label in [b[2] for b in BINS]:
        cnt = dist["counts"].get(label, 0)
        pct = dist["pct"].get(label, 0.0)
        if pct > 0.05:
            print(f"    {label}: {cnt} ({pct:.1f}%)")

print(f"\n── Key Windows ──")
for label, info in key_windows.items():
    if info.get("steps"):
        print(f"  {label}: steps {info['steps']}, mean headroom {info['mean_headroom_pct']:.1f}%")

print(f"\n── Torque Component Breakdown (all steps) ──")
comp_all = component_stats.get("all", {})
for comp_name, stats in sorted(comp_all.items()):
    print(f"  {comp_name}: mean_abs={stats['mean_abs']:.3f}, max_abs={stats['max_abs']:.3f} Nm")

print(f"\n── VERDICT ──")
print(f"  {verdict}")

# ── Write report ──────────────────────────────────────────────────────────
report_lines = [
    "# K1 Torque Headroom Audit",
    "",
    f"**Verdict:** `{verdict}`",
    "",
    "## Setup",
    "",
    f"- **Controller:** K1_PITCH_RATE_NOTCH_V1",
    f"- **Scenario:** high_0p480, 90N sagittal push at step 300, 10-step duration, 3000 steps",
    f"- **Max wheel torque (hardware):** {max_tau_wheel} Nm per wheel",
    f"- **Total steps:** {N}",
    "",
    "## Overall Headroom",
    "",
    f"| Metric | Value |",
    f"|--------|-------|",
    f"| Mean headroom | {overall_mean_headroom:.1f}% ({overall_mean_headroom * max_tau_wheel / 100:.2f} Nm) |",
    f"| Steps < 10% headroom | {starved_pct:.1f}% |",
    f"| Steps < 5% headroom | {constricted_pct:.1f}% |",
    f"| Saturation event rate | {sat_rate:.2f}% |",
    f"| Authority starved | **{'YES' if is_starved else 'NO'}** |",
    f"| Authority constricted | **{'YES' if is_constricted else 'NO'}** |",
    "",
    "## Utilization Distribution by Phase",
    "",
    "| Phase | Steps | 0-10% | 10-20% | 20-40% | 40-60% | 60-80% | 80-100% | >100% |",
    "|-------|-------|-------|--------|--------|--------|--------|---------|-------|",
]

for phase_name in ["all", "pre_push", "during_push", "post_push_early", "post_push_late"]:
    dist = distribution.get(phase_name, {})
    if not dist:
        continue
    pcts = dist["pct"]
    row = f"| {phase_name} | {dist['total_steps']} |"
    for label in [b[2] for b in BINS]:
        row += f" {pcts.get(label, 0):.1f}% |"
    report_lines.append(row)

report_lines += [
    "",
    "## Headroom by Phase",
    "",
    "| Phase | Count | Mean HR (Nm) | Mean HR (%) | Max Util | Sat Rate |",
    "|-------|-------|-------------|------------|---------|---------|",
]

for phase_name, stats in phase_stats.items():
    if not stats:
        continue
    hr_nm = stats.get("headroom_nm", {})
    hr_pct = stats.get("headroom_pct", {})
    util = stats.get("utilization_ratio", {})
    sr = stats.get("saturation_rate_pct", 0)
    row = f"| {phase_name} | {stats.get('count', 0)} |"
    row += f" {hr_nm.get('mean', 0):.2f} | {hr_pct.get('mean', 0):.1f}% |"
    row += f" {util.get('mean', 0):.2f} ({util.get('max', 0):.2f} max) |"
    row += f" {sr:.1f}% |"
    report_lines.append(row)

report_lines += [
    "",
    "## Torque Component Breakdown (mean absolute values)",
    "",
    "| Component | All | Pre-Push | During Push | Post-Push Early | Post-Push Late |",
    "|-----------|-----|----------|-------------|-----------------|----------------|",
]

comp_names = ["tau_pitch", "tau_pitch_rate", "tau_cp", "tau_com_vy",
              "tau_position", "tau_position_clipped", "tau_support_velocity",
              "tau_bal_raw", "tau_bal_clipped", "tau_bal_final"]
for cn in comp_names:
    row = f"| {cn} |"
    for phase_name in ["all", "pre_push", "during_push", "post_push_early", "post_push_late"]:
        cs = component_stats.get(phase_name, {}).get(cn, {})
        row += f" {cs.get('mean_abs', 0):.3f} |"
    report_lines.append(row)

report_lines += [
    "",
    "## Key Windows Headroom",
    "",
    "| Window | Steps | Mean HR (%) | Min HR (%) |",
    "|--------|-------|------------|-----------|",
]

for label, info in key_windows.items():
    steps_str = info.get("steps", "")
    report_lines.append(f"| {label} | {steps_str} | {info.get('mean_headroom_pct', 0):.1f}% | {info.get('min_headroom_pct', 0):.1f}% |")

report_lines += [
    "",
    "## Diagnosis",
    "",
    f"### Is K1 authority-starved?",
    "",
    f"**{'YES' if is_starved else 'NO'}**",
    "",
    f"- Overall mean headroom: {overall_mean_headroom:.1f}%",
    f"- Steps with <10% headroom: {starved_pct:.1f}%",
    f"- Steps with <5% headroom: {constricted_pct:.1f}%",
    f"- Saturation event rate: {sat_rate:.2f}%",
    f"- Recovery phase mean headroom: {pp_early_mean_hr:.1f}%",
    "",
    f"### Interpretation",
    "",
]

if verdict == "NOT_AUTHORITY_STARVED":
    report_lines.append(
        "K1 has ample torque headroom. The 5.0 Nm per-wheel hardware limit is rarely approached. "
        "Mean utilization is very low (~{:.1f}% of max). Authority is NOT the bottleneck.".format(
            100 * sum(all_utils) / len(all_utils))
    )
elif verdict == "AUTHORITY_COMFORTABLE_WITH_OCCASIONAL_SPIKES":
    report_lines.append(
        "K1 has comfortable headroom with occasional utilization spikes. Authority is not the "
        "primary bottleneck, but transient saturation may occur during push events."
    )
elif verdict == "AUTHORITY_CONSTRAINED_BUT_NOT_STARVED":
    report_lines.append(
        "K1 operates with constrained authority. While not persistently starved, the torque budget "
        "is tight enough that any increase in demand (stronger push, lower height) could cause saturation."
    )
elif "RECOVERY" in verdict:
    report_lines.append(
        "K1 is authority-starved during the critical post-push recovery phase. While the mean headroom "
        "may appear comfortable, the recovery phase shows significant constraint that may limit performance."
    )

report_lines += [
    "",
    f"**Classification:** `{verdict}`",
]

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"Report saved to: {OUTPUT_REPORT}")
print("\nDone.")
