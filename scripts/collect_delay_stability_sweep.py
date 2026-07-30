#!/usr/bin/env python3
"""Clean-delay stability threshold sweep.

Extends Table VII delay axis beyond 30ms (50, 100, 150ms) to find
the HARD STABILITY threshold where the robot actually falls from delay
alone — as opposed to the PERFORMANCE threshold (~10ms) where Fmax bifurcates.

Reuses the identical MEASURE_SCRIPT from collect_robustness_sweep.py.
Clean sensors only; N=5 per cell.

Usage:
  .venv/bin/mjpython scripts/collect_delay_stability_sweep.py
"""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Reuse identical MEASURE_SCRIPT from collect_robustness_sweep.py ──
# (import the other script's MEASURE_SCRIPT string)
from scripts.collect_robustness_sweep import MEASURE_SCRIPT

DELAY_CONFIGS = [
    # (label, delay_steps, delay_ms)
    ("clean_50ms", 5, 50),
    ("clean_100ms", 10, 100),
    ("clean_150ms", 15, 150),
]
NOISE_CFG = {"gyro": 0.0, "accel": 0.0, "joint": 0.0}
N_TRIALS = 5


def run_cell(label: str, delay_steps: int, n_trials: int) -> dict | None:
    cfg_json = json.dumps({
        "noise_name": "clean",
        "noise_cfg": NOISE_CFG,
        "delay_steps": delay_steps,
        "n_trials": n_trials,
    })
    try:
        result = subprocess.run(
            [sys.executable, "-c", MEASURE_SCRIPT, cfg_json],
            capture_output=True, text=True, timeout=7200,  # 2h — bisect at high delay may be slow
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"    FAILED (rc={result.returncode})")
            if result.stderr:
                lines = result.stderr.strip().split("\n")
                for line in lines[-5:]:
                    print(f"    stderr: {line[:200]}")
            return None
        for line in result.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith('{"'):
                return json.loads(line)
        print(f"    No JSON output. stdout tail: {result.stdout[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        print("    TIMEOUT (2h)")
        return None
    except Exception as e:
        print(f"    ERROR: {e}")
        return None


def main():
    # Also read existing 0/10/30ms data for context
    existing_path = OUT_DIR / "robustness_sweep.json"
    existing = {}
    if existing_path.exists():
        existing = json.load(existing_path.open())

    total = len(DELAY_CONFIGS)
    print("=" * 72)
    print(f"DELAY STABILITY THRESHOLD SWEEP — {total} cells × N={N_TRIALS}")
    print("Clean sensors only. Finding hard stability threshold.")
    print("Est. time: ~10-15 min per cell (~30-45 min total)")
    print("=" * 72)

    results = {}
    t_start = time.time()

    for idx, (label, delay_steps, delay_ms) in enumerate(DELAY_CONFIGS):
        print(f"\n[{idx+1}/{total}] {label} ({delay_ms}ms, {delay_steps} steps)...",
              end=" ", flush=True)
        t0 = time.time()
        data = run_cell(label, delay_steps, N_TRIALS)
        dt = time.time() - t0

        if data:
            idle_str = "---"
            if data["idle_rms_mm_mean"] is not None:
                idle_str = f"{data['idle_rms_mm_mean']:.2f}±{data['idle_rms_mm_std']:.2f}mm"
            print(f"OK ({dt:.0f}s) Idle={idle_str} "
                  f"Fmax={data['f_max_N_mean']:.0f}±{data['f_max_N_std']:.0f}N "
                  f"fell={data['n_fell']}")
            results[label] = data
        else:
            print(f"FAILED ({dt:.0f}s)")
            results[label] = {"error": "measurement_failed"}

        # Incremental save
        (OUT_DIR / "delay_stability_partial.json").write_text(
            json.dumps(results, indent=2))

    # ── Summary ──
    results["_metadata"] = {
        "n_trials": N_TRIALS,
        "noise": "clean",
        "delay_ms": [d[2] for d in DELAY_CONFIGS],
        "total_elapsed_min": (time.time() - t_start) / 60.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = OUT_DIR / "delay_stability_sweep.json"
    json.dump(results, out_path.open("w"), indent=2)

    print(f"\n{'='*72}")
    print("DELAY STABILITY RESULTS (clean sensors)")
    print(f"{'='*72}")
    print(f"{'Delay':<15} {'Idle RMS (mm)':>18} {'F_max (N)':>18} {'Fell':>6}")
    print("-" * 60)
    # Show existing 0/10/30 first
    for ek in ["clean_0ms", "clean_10ms", "clean_30ms"]:
        if ek in existing:
            r = existing[ek]
            im = f"{r['idle_rms_mm_mean']:.2f}±{r['idle_rms_mm_std']:.2f}" if r.get("idle_rms_mm_mean") is not None else "---"
            print(f"{ek:<15} {im:>18}  {r['f_max_N_mean']:>8.0f}±{r['f_max_N_std']:>3.0f}  {r['n_fell']:>4}")
    for label, _, delay_ms in DELAY_CONFIGS:
        r = results.get(label, {})
        if "error" in r:
            print(f"{label:<15} {'--- (error)':>18} {'---':>18}")
        else:
            im = f"{r['idle_rms_mm_mean']:.2f}±{r['idle_rms_mm_std']:.2f}" if r.get("idle_rms_mm_mean") is not None else "--- (fell)"
            print(f"{label:<15} {im:>18}  {r['f_max_N_mean']:>8.0f}±{r['f_max_N_std']:>3.0f}  {r['n_fell']:>4}")

    print(f"\nSaved → {out_path}")
    print(f"Total time: {results['_metadata']['total_elapsed_min']:.1f} min")
    print("Done.")


if __name__ == "__main__":
    main()
