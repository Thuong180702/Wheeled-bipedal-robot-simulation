"""Phase 6: pitch_bias_compensated_zero_crossing_recenter staged validation.

Runs high_0p480 for 1200, 2000, 5000 steps comparing:
- early_zero_crossing_recenter_v2 (baseline)
- pitch_bias_compensated_zero_crossing_recenter (new profile)
"""
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "step_e_extreme_support_fix_eval" / "active_pitch_crossing"

RUNS = [
    ("early_zero_crossing_recenter_v2", 1200, "ezc_v2"),
    ("pitch_bias_compensated_zero_crossing_recenter", 1200, "pbc_zc"),
    ("early_zero_crossing_recenter_v2", 2000, "ezc_v2"),
    ("pitch_bias_compensated_zero_crossing_recenter", 2000, "pbc_zc"),
    ("early_zero_crossing_recenter_v2", 5000, "ezc_v2"),
    ("pitch_bias_compensated_zero_crossing_recenter", 5000, "pbc_zc"),
]


def run_sim(profile, steps, label):
    setup_path = SETUP_DIR / "high_0p480_setup.json"
    out_dir = OUT_BASE / f"{label}_{steps}_high_0p480"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not setup_path.exists():
        print(f"ERROR: Setup not found: {setup_path}")
        return None

    args = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
    ]

    print(f"\n{'='*60}")
    print(f"Profile: {profile}")
    print(f"Steps: {steps}  Label: {label}")
    print(f"{'='*60}")
    sys.stdout.flush()

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print(f"ERROR (rc={result.returncode})")
        print(result.stderr[-2000:])
        return None

    csv_files = sorted(
        (ROOT / "outputs" / "hierarchical_controller_sim").glob("telemetry_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not csv_files:
        print("ERROR: No telemetry CSV found")
        return None

    target = out_dir / f"telemetry_{steps}.csv"
    shutil.copy(csv_files[0], target)
    print(f"Saved: {target}")
    return target


def f(r, c):
    v = r.get(c, "")
    try:
        return float(v)
    except Exception:
        return float("nan")


def analyze(csv_path, label, steps):
    if not csv_path or not csv_path.exists():
        return None

    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    drift_col = "active_pitch_crossing_signed_error_m"

    drift = [f(r, drift_col) for r in rows]
    drift_clean = [v for v in drift if not math.isnan(v)]
    n_drift = len(drift_clean)

    pos_pct = 100.0 * sum(1 for v in drift_clean if v > 0) / n_drift if n_drift else 0
    neg_pct = 100.0 * sum(1 for v in drift_clean if v < 0) / n_drift if n_drift else 0
    d_min = min(drift_clean) if drift_clean else float("nan")
    d_max = max(drift_clean) if drift_clean else float("nan")
    d_mean = sum(drift_clean) / len(drift_clean) if drift_clean else float("nan")
    d_p2p = d_max - d_min if drift_clean else float("nan")

    crossings = sum(
        1 for i in range(1, len(drift_clean))
        if drift_clean[i - 1] * drift_clean[i] < 0
    )

    ezc_active = [f(r, "ezc_active") for r in rows]
    ezc_enters = sum(
        1 for i in range(1, len(ezc_active))
        if ezc_active[i - 1] < 0.5 and ezc_active[i] >= 0.5
    )

    pbc_tau = [f(r, "pitch_bias_comp_tau_nm") for r in rows]
    pbc_clean = [v for v in pbc_tau if not math.isnan(v)]
    pbc_mean = sum(pbc_clean) / len(pbc_clean) if pbc_clean else 0.0
    pbc_max = max(pbc_clean) if pbc_clean else 0.0
    pbc_active = [f(r, "pitch_bias_comp_active") for r in rows]
    pbc_active_steps = sum(1 for v in pbc_active if v > 0.5 and not math.isnan(v))

    pbc_est = [f(r, "pitch_bias_estimate_nm") for r in rows]
    pbc_est_clean = [v for v in pbc_est if not math.isnan(v)]
    pbc_est_final = pbc_est_clean[-1] if pbc_est_clean else 0.0

    tp_before = [f(r, "tau_pitch_before_bias_comp") for r in rows]
    tp_after = [f(r, "tau_pitch_after_bias_comp") for r in rows]
    tp_before_clean = [v for v in tp_before if not math.isnan(v)]
    tp_after_clean = [v for v in tp_after if not math.isnan(v)]
    tp_before_mean = sum(tp_before_clean) / len(tp_before_clean) if tp_before_clean else float("nan")
    tp_after_mean = sum(tp_after_clean) / len(tp_after_clean) if tp_after_clean else float("nan")

    stats = {
        "label": label,
        "steps": steps,
        "n_rows": n,
        "drift_min": d_min,
        "drift_max": d_max,
        "drift_mean": d_mean,
        "drift_p2p": d_p2p,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "zero_crossings": crossings,
        "ezc_enters": ezc_enters,
        "tau_pitch_before_mean": tp_before_mean,
        "tau_pitch_after_mean": tp_after_mean,
        "pitch_bias_comp_tau_mean": pbc_mean,
        "pitch_bias_comp_tau_max": pbc_max,
        "pitch_bias_comp_active_steps": pbc_active_steps,
        "pitch_bias_estimate_final": pbc_est_final,
    }

    print(f"\n--- {label} {steps}-step ---")
    print(f"  drift min={d_min:+.4f}  max={d_max:+.4f}  P2P={d_p2p:.4f}  mean={d_mean:+.4f}")
    print(f"  pos%={pos_pct:.1f}  neg%={neg_pct:.1f}  crossings={crossings}  EZC enters={ezc_enters}")
    print(f"  tau_pitch before={tp_before_mean:+.3f}  after={tp_after_mean:+.3f}")
    print(f"  comp_tau mean={pbc_mean:+.4f}  max={pbc_max:+.4f}  active_steps={pbc_active_steps}")
    print(f"  bias_estimate final={pbc_est_final:+.4f}")

    return stats


def main():
    all_results = {}

    for profile, steps, label in RUNS:
        key = f"{label}_{steps}"
        csv_path = run_sim(profile, steps, label)
        stats = analyze(csv_path, label, steps)
        if stats:
            all_results[key] = stats

    # Save
    out_json = OUT_BASE / "pbc_zc_staged_validation.json"
    with open(out_json, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\nSaved: {out_json}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("STAGED VALIDATION SUMMARY: pbc_zc vs ezc_v2")
    print(f"{'='*80}")
    print(f"{'Profile':<12} {'Steps':>6} {'min':>8} {'max':>8} {'P2P':>7} {'pos%':>7} {'neg%':>7} {'X':>5} {'comp_max':>9}")
    print("-" * 80)
    for steps in [1200, 2000, 5000]:
        for label in ["ezc_v2", "pbc_zc"]:
            key = f"{label}_{steps}"
            if key in all_results:
                s = all_results[key]
                print(f"  {label:<12} {steps:>6} "
                      f"{s['drift_min']:>+8.4f} {s['drift_max']:>+8.4f} "
                      f"{s['drift_p2p']:>7.4f} "
                      f"{s['pos_pct']:>6.1f}% "
                      f"{s['neg_pct']:>6.1f}% "
                      f"{s['zero_crossings']:>5} "
                      f"{s['pitch_bias_comp_tau_max']:>9.4f}")

    # 5000-step pass check
    v2_5000 = all_results.get("ezc_v2_5000")
    pbc_5000 = all_results.get("pbc_zc_5000")
    if v2_5000 and pbc_5000:
        print(f"\n--- 5000-step pass criteria ---")
        criteria = {
            "pos% < 86% (V1 baseline)": pbc_5000["pos_pct"] < 86.0,
            "pos% lower than ezc_v2": pbc_5000["pos_pct"] < v2_5000["pos_pct"],
            "neg% higher than ezc_v2": pbc_5000["neg_pct"] > v2_5000["neg_pct"],
            "min drift more negative or same": pbc_5000["drift_min"] <= v2_5000["drift_min"] + 0.01,
            "max drift not worse by >0.02": pbc_5000["drift_max"] <= v2_5000["drift_max"] + 0.02,
            "P2P bounded (<0.30)": pbc_5000["drift_p2p"] < 0.30,
            "tau_pitch after < before": pbc_5000["tau_pitch_after_mean"] <= pbc_5000["tau_pitch_before_mean"],
            "no fall": pbc_5000["n_rows"] >= 4990,
        }
        all_pass = True
        for name, result in criteria.items():
            status = "PASS" if result else "FAIL"
            if not result:
                all_pass = False
            print(f"  {status}: {name}")
        print(f"\nOverall 5000-step: {'ALL PASS' if all_pass else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
