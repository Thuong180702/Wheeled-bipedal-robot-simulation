#!/usr/bin/env python3
"""Batch-run flight/terrain trials to N=10 for paper §VI statistical power.

Usage:
  .venv/bin/python scripts/collect_flight_terrain_n10.py
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import scripts.ramp_step_tests as ramp_mod

OUT_DIR = ROOT / "outputs" / "flight_terrain_n10"
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_TRIALS = 10


def run_ledge_n(n=N_TRIALS, h_m=0.50):
    """Run N ledge drive-off trials at height h_m."""
    print(f"\n{'='*60}")
    print(f"LEDGE {h_m*100:.0f}cm — N={n}")
    print(f"{'='*60}")
    results = []
    t0 = time.time()
    for i in range(n):
        r = ramp_mod.run_ramp_step(h_m, duration_s=30.0, course="up_off", seed=i)
        results.append(r)
        if r.get("fell"):
            print(f"  [{i+1}/{n}] FALL at {r['fall_t']:.1f}s")
        else:
            print(f"  [{i+1}/{n}] PASS  peak_pitch={r['peak_pitch_land']:.1f}°  settle={r['settle_s']:.1f}s")
    elapsed = time.time() - t0
    n_pass = sum(1 for r in results if r.get("verdict") == "PASS")
    n_fell = sum(1 for r in results if r.get("fell"))
    peaks = [r["peak_pitch_land"] for r in results if not r.get("fell")]
    print(f"  {n_pass}/{n} PASS ({n_fell} fell) in {elapsed:.0f}s")
    if peaks:
        print(f"  peak_pitch: {np.mean(peaks):.1f}±{np.std(peaks, ddof=1):.1f}°")
    return results, peaks


def run_curb_n(h_m, n=N_TRIALS):
    """Run N curb straddle trials at height h_m."""
    print(f"\n  Curb {h_m*100:.0f}cm — N={n}")
    results = []
    t0 = time.time()
    for i in range(n):
        r = ramp_mod.run_curb(h_m, duration_s=30.0, seed=i)
        results.append(r)
        if r.get("fell"):
            print(f"    [{i+1}/{n}] FALL at {r['fall_t']:.1f}s")
        else:
            off = r.get("off_curb_y")
            tail = f"  left the slab at {off:.2f} m" if off is not None else ""
            print(f"    [{i+1}/{n}] {r['verdict']}  roll_max={r['roll_curb_max']:.1f}°"
                  f"  settle={r['settle_s']:.1f}s{tail}")
    elapsed = time.time() - t0
    n_pass = sum(1 for r in results if r.get("verdict") == "PASS")
    n_fell = sum(1 for r in results if r.get("fell"))
    # Straddle roll is only defined for a trial that actually straddled the whole
    # curb, so the statistic is taken over PASS trials — an OFF_CURB run stops
    # accumulating roll the moment it leaves the slab and would bias it low.
    rolls = [r["roll_curb_max"] for r in results if r.get("verdict") == "PASS"]
    print(f"    {n_pass}/{n} PASS ({n_fell} fell) in {elapsed:.0f}s")
    if rolls:
        print(f"    roll_curb_max: {np.mean(rolls):.1f}±{np.std(rolls, ddof=1):.1f}°")
    return results, rolls


def main():
    t_start = time.time()
    output = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_trials": N_TRIALS}

    # ── Ledge 50cm ──
    ledge_results, ledge_peaks = run_ledge_n(N_TRIALS, 0.50)
    output["ledge_50cm"] = {
        "n_pass": sum(1 for r in ledge_results if r.get("verdict") == "PASS"),
        "n_total": N_TRIALS,
        "peak_pitch_mean": float(np.mean(ledge_peaks)) if ledge_peaks else None,
        "peak_pitch_std": float(np.std(ledge_peaks, ddof=1)) if len(ledge_peaks) > 1 else 0.0,
        "results": ledge_results,
    }

    # ── Curbs 10, 15, 20cm ──
    output["curbs"] = {}
    for h_cm in [10, 15, 20]:
        h_m = h_cm / 100.0
        curb_results, curb_rolls = run_curb_n(h_m, N_TRIALS)
        output["curbs"][f"{h_cm}cm"] = {
            "n_pass": sum(1 for r in curb_results if r.get("verdict") == "PASS"),
            "n_total": N_TRIALS,
            "roll_curb_max_mean": float(np.mean(curb_rolls)) if curb_rolls else None,
            "roll_curb_max_std": float(np.std(curb_rolls, ddof=1)) if len(curb_rolls) > 1 else 0.0,
            "results": curb_results,
        }

    # ── Summary ──
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY (N={N_TRIALS})")
    print(f"{'='*60}")
    ledge = output["ledge_50cm"]
    print(f"  Ledge 50cm: {ledge['n_pass']}/{N_TRIALS}  peak_pitch={ledge['peak_pitch_mean']:.1f}±{ledge['peak_pitch_std']:.1f}°")
    for h_cm in [10, 15, 20]:
        c = output["curbs"][f"{h_cm}cm"]
        print(f"  Curb {h_cm}cm:  {c['n_pass']}/{N_TRIALS}  roll_max={c['roll_curb_max_mean']:.1f}±{c['roll_curb_max_std']:.1f}°")

    # Clopper-Pearson 95% CI lower bound for k/N survival
    from scipy.stats import beta
    for label, k in [("Ledge 50cm", ledge["n_pass"]),
                     ("Curb 10cm", output["curbs"]["10cm"]["n_pass"]),
                     ("Curb 15cm", output["curbs"]["15cm"]["n_pass"]),
                     ("Curb 20cm", output["curbs"]["20cm"]["n_pass"])]:
        ci_lo = float(beta.ppf(0.025, k, N_TRIALS - k + 1)) if k > 0 else 0.0
        ci_hi = float(beta.ppf(0.975, k + 1, N_TRIALS - k)) if k < N_TRIALS else 1.0
        print(f"  {label}: {k}/{N_TRIALS}  95%CI [{ci_lo:.2f}, {ci_hi:.2f}]")

    print(f"\nTotal time: {elapsed:.0f}s")

    # Save
    out_path = OUT_DIR / "results.json"
    json.dump(output, out_path.open("w"), indent=2, default=str)
    print(f"Saved → {out_path}")
    return output


if __name__ == "__main__":
    main()
