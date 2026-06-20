"""Adaptive Support Centering Trim — Phase 7 Height Ladder.

Runs adaptive_support_centering_trim across all available height setups,
2000 steps each, for height-ladder sanity check.
Covers: low_0p300, low_0p320, low_0p330, low_0p340, low_0p360, low_0p380,
        high_0p430, high_0p450, high_0p465, high_0p480.
"""
import csv, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"
MANIFEST_PATH = OUT_BASE / "adaptive_height_ladder_manifest.json"
STEPS = 2000
PROFILE = "adaptive_support_centering_trim"

DESIRED_LABELS = [
    "low_0p300", "low_0p320", "low_0p330", "low_0p340",
    "low_0p360", "low_0p380",
    "high_0p430", "high_0p450", "high_0p465", "high_0p480",
]


def discover_setups():
    setups = {}
    for p in sorted(SETUP_DIR.glob("*_setup.json")):
        label = p.stem.replace("_setup", "")
        setups[label] = p
    return setups


def run_variant(entry):
    label = entry["label"]
    setup_path = entry["setup_path"]
    out_dir = OUT_BASE / f"adaptive_height_ladder_2000_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE,
        "--height-variant-setup", str(setup_path),
        "--steps", str(STEPS),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(STEPS),
        "--write-run-summary-sidecar",
    ]

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    entry["return_code"] = result.returncode

    if result.returncode != 0:
        entry["launch_status"] = "failed"
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")
        entry["notes"] = f"rc={result.returncode}"
        print(f"  FAILED {label} rc={result.returncode}")
    else:
        entry["launch_status"] = "completed"
        sim_out = ROOT / "outputs" / "hierarchical_controller_sim"
        telemetry_files = sorted(sim_out.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        summary_files = sorted(sim_out.glob("run_summary_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        if telemetry_files:
            src = telemetry_files[0]
            dst = out_dir / "telemetry_2000.csv"
            shutil.copy2(src, dst)
            entry["telemetry_path"] = str(dst)
            try:
                src.unlink()
            except OSError:
                pass
        if summary_files:
            src = summary_files[0]
            dst = out_dir / "run_summary.json"
            shutil.copy2(src, dst)
            entry["summary_path"] = str(dst)
            try:
                src.unlink()
            except OSError:
                pass
        print(f"  OK {label} -> {out_dir.name}")


def build_manifest_entries(setups):
    entries = []
    for label in DESIRED_LABELS:
        setup_path = setups.get(label)
        entry = {
            "label": label,
            "setup_path": str(setup_path) if setup_path else "",
            "exists": setup_path is not None and setup_path.exists(),
            "launch_status": "pending",
            "return_code": None,
            "telemetry_path": "",
            "summary_path": "",
            "notes": "",
        }
        if not entry["exists"]:
            entry["launch_status"] = "skipped_missing"
            entry["notes"] = "Setup file not found"
        entries.append(entry)
    return entries


def safe_max(vals, default=0.0):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and x != x)]
    return max(v) if v else default


def analyze_telemetry(path):
    """Return per-variant summary metrics."""
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}

    n = len(rows)
    def get_col(key, default=0.0):
        return [float(r[key]) if r.get(key, "nan") not in ("", "nan", "None") else default for r in rows]
    def get_bool(key):
        return [str(r.get(key, "false")).lower().strip() in ("true", "1", "1.0") for r in rows]

    err = get_col("active_pitch_crossing_signed_error_m")
    abs_err = [abs(v) for v in err]
    sv = sorted(err); n4 = len(sv)
    ab_tau = get_col("adaptive_bias_tau_nm")
    ab_max = get_col("adaptive_bias_max_tau_current_nm")
    ab_act = get_bool("adaptive_bias_trim_active")
    ab_gat = get_bool("adaptive_bias_safety_gate_pass")
    ab_en = get_bool("adaptive_bias_trim_enabled")
    t6j_act = get_bool("t6j_bias_trim_active")
    t6j_tau = get_col("t6j_bias_trim_tau_nm")
    pitch = [float(r["robot_pitch_x"]) * 180 / 3.14159 for r in rows if r.get("robot_pitch_x", "nan") not in ("", "nan", "None")]
    pitch = [p for p in pitch if p == p]  # remove nan

    sat_n = sum(1 for i in range(n) if ab_max[i] > 0 and abs(ab_tau[i]) >= 0.99 * ab_max[i])

    return {
        "steps": n,
        "maxabs": round(safe_max(abs_err), 4),
        "p2p": round(max(err) - min(err), 4),
        "rms": round((sum(v * v for v in err) / n) ** 0.5, 4),
        "mae": round(sum(abs_err) / n, 4),
        "final": round(err[-1], 4) if err else 0,
        "mean": round(sum(err) / n, 4),
        "pos_pct": round(100 * sum(1 for v in err if v > 0) / n, 1),
        "neg_pct": round(100 * sum(1 for v in err if v < 0) / n, 1),
        "out10": round(100 * sum(1 for v in abs_err if v > 0.10) / n, 1),
        "out15": round(100 * sum(1 for v in abs_err if v > 0.15) / n, 1),
        "in3": round(100 * sum(1 for v in abs_err if v <= 0.03) / n, 1),
        "pitch_max": round(max(pitch), 2) if pitch else 0,
        "ab_en_pct": round(100 * sum(ab_en) / n, 1),
        "ab_act_pct": round(100 * sum(ab_act) / n, 1),
        "ab_sat_pct": round(100 * sat_n / n, 1),
        "ab_tau_range": [round(min(ab_tau), 3), round(max(ab_tau), 3)],
        "ab_gate_pct": round(100 * sum(ab_gat) / n, 1),
        "t6j_act_pct": round(100 * sum(t6j_act) / n, 1),
    }


def main():
    setups = discover_setups()
    entries = build_manifest_entries(setups)
    available = [e for e in entries if e["exists"] and e["launch_status"] == "pending"]
    skipped = [e for e in entries if not e["exists"]]

    print(f"Total: {len(entries)} labels | {len(available)} to run | {len(skipped)} skipped (no setup)")
    for e in skipped:
        print(f"  SKIP {e['label']}: no setup file")

    for entry in available:
        label = entry["label"]
        print(f"\n[{available.index(entry)+1}/{len(available)}] Running {label}...", end=" ", flush=True)
        start = time.time()
        run_variant(entry)
        elapsed = time.time() - start
        entry["elapsed_s"] = round(elapsed, 1)
        print(f"  ({elapsed:.0f}s)")

    # Analyze all completed
    for entry in available:
        if entry["telemetry_path"]:
            path = Path(entry["telemetry_path"])
            if path.exists():
                m = analyze_telemetry(path)
                entry["metrics"] = m
                print(f"  {entry['label']}: maxabs={m.get('maxabs','?')}  pos%={m.get('pos_pct','?')}  "
                      f"ab_sat={m.get('ab_sat_pct','?')}%  pitch_max={m.get('pitch_max','?')}°")

    # Save manifest
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2))
    print(f"\nManifest: {MANIFEST_PATH}")

    # Summary table
    print("\n=== Phase 7 Height Ladder Summary ===")
    print(f"{'Label':<16} {'maxabs':>7} {'mean':>7} {'pos%':>6} {'out10':>6} {'out15':>6} {'ab_sat%':>7} {'t6j%':>5} {'pitch_max':>9}")
    for e in entries:
        if "metrics" in e:
            m = e["metrics"]
            print(f"  {e['label']:<16} {m.get('maxabs','?'):>7} {m.get('mean','?'):>7} "
                  f"{m.get('pos_pct','?'):>6} {m.get('out10','?'):>6} {m.get('out15','?'):>6} "
                  f"{m.get('ab_sat_pct','?'):>7} {m.get('t6j_act_pct','?'):>5} {m.get('pitch_max','?'):>9}°")

    # Write summary table to markdown
    md = ["# Phase 7: Height Ladder — adaptive_support_centering_trim\n",
          f"**Steps:** {STEPS} | **Profile:** {PROFILE} | **Date:** 2026-06-14\n\n",
          "| Label | maxabs | mean | pos% | neg% | out10% | out15% | in3% | ab_sat% | pitch_max | ab_tau_range |\n",
          "|-------|--------|------|------|------|--------|--------|------|---------|-----------|---------------|\n"]
    for e in entries:
        if "metrics" in e:
            m = e["metrics"]
            md.append(f"| {e['label']} | {m.get('maxabs','?'):.4f} | {m.get('mean','?'):.4f} | "
                      f"{m.get('pos_pct','?')} | {m.get('neg_pct','?')} | {m.get('out10','?')} | "
                      f"{m.get('out15','?')} | {m.get('in3','?')} | {m.get('ab_sat_pct','?')} | "
                      f"{m.get('pitch_max','?'):.2f}° | {m.get('ab_tau_range','?')} |\n")
    md_path = OUT_BASE / "adaptive_height_ladder_summary.md"
    Path(md_path).write_text("".join(md))
    print(f"\nMarkdown: {md_path}")


if __name__ == "__main__":
    main()