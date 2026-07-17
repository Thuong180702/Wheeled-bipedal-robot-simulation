"""Phase 5: pitch_bias_compensated_zero_crossing_recenter 500-step diagnostic.

Runs high_0p480 for 500 steps comparing:
- early_zero_crossing_recenter_v2 (baseline)
- pitch_bias_compensated_zero_crossing_recenter (new profile)

Measures: drift stats, tau_pitch mean, compensation activity.
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
    ("early_zero_crossing_recenter_v2", 500, "ezc_v2"),
    ("pitch_bias_compensated_zero_crossing_recenter", 500, "pbc_zc"),
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

    result = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"ERROR (rc={result.returncode})")
        print(result.stderr[-2000:])
        return None

    # Latest telemetry CSV
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


def analyze(csv_path, label):
    if not csv_path or not csv_path.exists():
        return None

    rows = list(csv.DictReader(open(csv_path)))
    n = len(rows)
    drift_col = "active_pitch_crossing_signed_error_m"

    drift = [f(r, drift_col) for r in rows]
    drift_clean = [v for v in drift if not math.isnan(v)]
    tau_pitch_vals = [f(r, "tau_pitch") for r in rows]
    tau_pitch_clean = [v for v in tau_pitch_vals if not math.isnan(v)]

    # Drift stats
    pos_n = sum(1 for v in drift_clean if v > 0)
    neg_n = sum(1 for v in drift_clean if v < 0)
    n_drift = len(drift_clean)
    pos_pct = 100.0 * pos_n / n_drift if n_drift else 0
    neg_pct = 100.0 * neg_n / n_drift if n_drift else 0

    d_min = min(drift_clean) if drift_clean else float("nan")
    d_max = max(drift_clean) if drift_clean else float("nan")
    d_mean = sum(drift_clean) / len(drift_clean) if drift_clean else float("nan")
    d_p2p = d_max - d_min if drift_clean else float("nan")

    # Zero crossings
    crossings = 0
    for i in range(1, len(drift_clean)):
        if drift_clean[i - 1] * drift_clean[i] < 0:
            crossings += 1

    # EZC activations
    ezc_active = [f(r, "ezc_active") for r in rows]
    ezc_enters = sum(
        1 for i in range(1, len(ezc_active))
        if ezc_active[i - 1] < 0.5 and ezc_active[i] >= 0.5
    )

    # Antirebound steps
    ar_steps = [f(r, "ezc_antirebound_steps") for r in rows]
    ar_total = sum(v for v in ar_steps if v > 0 and not math.isnan(v))

    # Pitch bias comp stats
    pbc_tau = [f(r, "pitch_bias_comp_tau_nm") for r in rows]
    pbc_clean = [v for v in pbc_tau if not math.isnan(v)]
    pbc_mean = sum(pbc_clean) / len(pbc_clean) if pbc_clean else 0.0
    pbc_max = max(pbc_clean) if pbc_clean else 0.0
    pbc_active = [f(r, "pitch_bias_comp_active") for r in rows]
    pbc_active_steps = sum(1 for v in pbc_active if v > 0.5 and not math.isnan(v))

    pbc_est = [f(r, "pitch_bias_estimate_nm") for r in rows]
    pbc_est_clean = [v for v in pbc_est if not math.isnan(v)]
    pbc_est_final = pbc_est_clean[-1] if pbc_est_clean else 0.0

    # tau_pitch stats
    tp_mean = sum(tau_pitch_clean) / len(tau_pitch_clean) if tau_pitch_clean else float("nan")
    tp_before = [f(r, "tau_pitch_before_bias_comp") for r in rows]
    tp_after = [f(r, "tau_pitch_after_bias_comp") for r in rows]
    tp_before_clean = [v for v in tp_before if not math.isnan(v)]
    tp_after_clean = [v for v in tp_after if not math.isnan(v)]
    tp_before_mean = sum(tp_before_clean) / len(tp_before_clean) if tp_before_clean else float("nan")
    tp_after_mean = sum(tp_after_clean) / len(tp_after_clean) if tp_after_clean else float("nan")

    stats = {
        "label": label,
        "n_rows": n,
        "drift_min": d_min,
        "drift_max": d_max,
        "drift_mean": d_mean,
        "drift_p2p": d_p2p,
        "pos_pct": pos_pct,
        "neg_pct": neg_pct,
        "zero_crossings": crossings,
        "ezc_enters": ezc_enters,
        "ezc_antirebound_total_steps": ar_total,
        "tau_pitch_mean": tp_mean,
        "tau_pitch_before_bias_comp_mean": tp_before_mean,
        "tau_pitch_after_bias_comp_mean": tp_after_mean,
        "pitch_bias_comp_tau_mean": pbc_mean,
        "pitch_bias_comp_tau_max": pbc_max,
        "pitch_bias_comp_active_steps": pbc_active_steps,
        "pitch_bias_estimate_nm_final": pbc_est_final,
    }

    print(f"\n--- {label} ---")
    print(f"  drift min={d_min:+.4f}  max={d_max:+.4f}  P2P={d_p2p:.4f}  mean={d_mean:+.4f}")
    print(f"  pos%={pos_pct:.1f}  neg%={neg_pct:.1f}  crossings={crossings}  EZC enters={ezc_enters}")
    print(f"  tau_pitch before={tp_before_mean:+.3f}  after={tp_after_mean:+.3f}")
    print(f"  pitch_bias_comp_tau mean={pbc_mean:+.4f}  max={pbc_max:+.4f}  active_steps={pbc_active_steps}")
    print(f"  pitch_bias_estimate final={pbc_est_final:+.4f}")

    return stats


def main():
    results = {}
    for profile, steps, label in RUNS:
        csv_path = run_sim(profile, steps, label)
        stats = analyze(csv_path, label)
        if stats:
            results[label] = stats

    # Save results
    out_json = OUT_BASE / "pbc_zc_500_diagnostic.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")

    # Comparison summary
    if "ezc_v2" in results and "pbc_zc" in results:
        v2 = results["ezc_v2"]
        pbc = results["pbc_zc"]
        print(f"\n{'='*60}")
        print("COMPARISON: pbc_zc vs ezc_v2 (500-step)")
        print(f"{'='*60}")
        print(f"  pos%:       {v2['pos_pct']:.1f}% -> {pbc['pos_pct']:.1f}% (delta={pbc['pos_pct']-v2['pos_pct']:+.1f}pp)")
        print(f"  neg%:       {v2['neg_pct']:.1f}% -> {pbc['neg_pct']:.1f}% (delta={pbc['neg_pct']-v2['neg_pct']:+.1f}pp)")
        print(f"  min drift:  {v2['drift_min']:+.4f} -> {pbc['drift_min']:+.4f}")
        print(f"  max drift:  {v2['drift_max']:+.4f} -> {pbc['drift_max']:+.4f}")
        print(f"  tau_pitch:  {v2['tau_pitch_mean']:+.3f} -> {pbc['tau_pitch_after_bias_comp_mean']:+.3f} (after comp)")
        print(f"  comp tau:   mean={pbc['pitch_bias_comp_tau_mean']:+.4f}  max={pbc['pitch_bias_comp_tau_max']:+.4f}  active_steps={pbc['pitch_bias_comp_active_steps']}")

        # Pass criteria
        print(f"\n--- Pass criteria check ---")
        criteria = {
            "no_fall (both have rows)": v2["n_rows"] > 0 and pbc["n_rows"] > 0,
            "max drift not worse by >0.02": pbc["drift_max"] <= v2["drift_max"] + 0.02,
            "pos% lower or equal": pbc["pos_pct"] <= v2["pos_pct"],
            "neg% higher or equal": pbc["neg_pct"] >= v2["neg_pct"],
            "zero crossings not lower": pbc["zero_crossings"] >= v2["zero_crossings"] - 1,
            "tau_pitch_after < tau_pitch_before": pbc["tau_pitch_after_bias_comp_mean"] <= pbc["tau_pitch_before_bias_comp_mean"],
            "comp_tau active (est converging)": pbc["pitch_bias_comp_active_steps"] > 0,
        }
        all_pass = True
        for name, result in criteria.items():
            status = "PASS" if result else "FAIL"
            if not result:
                all_pass = False
            print(f"  {status}: {name}")
        print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAIL'}")


if __name__ == "__main__":
    main()
