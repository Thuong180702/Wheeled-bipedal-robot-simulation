#!/usr/bin/env python3
"""
N=30 replication of the factorial push ablation (Table: tab:push_ablation).

Answers the reviewer point that N=10 resolves only ~2 N, so "no single-mechanism
removal is significant" was an underpowered null rather than evidence of absence.

Reuses the measurement harness and patch definitions of
``replicate_ablation_n10.py`` verbatim (fresh controller context per bisection
trial, same seeds, same 8 bearings, same binary search) and adds two things:

  * reps 0..29 instead of 0..9. Seeds are ``20260728*100 + rep``, so the N=10
    reps are the first 10 of the N=30 set -- the old table is nested inside the
    new one, not replaced by a differently-seeded run.
  * rep-level sharding across worker processes. Each config patches
    ``k2_jax_controller.py`` once, then K workers import the patched module
    read-only and each run a contiguous slice of reps. ~1.8 min/rep serial,
    so 30 reps over 6 workers is ~10 min/config instead of ~55.

Rows L0 and L1 are profile-swaps rather than source patches, so they are
included here too; the whole table then comes from one harness instead of two.

Usage:
  .venv/bin/python scripts/replicate_ablation_n30.py                 # all rows
  .venv/bin/python scripts/replicate_ablation_n30.py --configs S1 S2 --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import replicate_ablation_n10 as n10  # noqa: E402  (patch defs + measure harness)

OUT_DIR = ROOT / "outputs" / "paper_statistics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REPS = 30
ANCHOR = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"
# t_{0.025, 29}; the N=10 harness used t_{0.025, 9} = 2.262
T_CRIT = 2.045

# ── Config table ────────────────────────────────────────────────────────────
# L0/L1 are profile swaps (no source patch); L2..S5 reuse the n10 definitions.
CONFIGS = [
    {"id": "L0", "name": "L0: P-only (PD, no pos, no I)",
     "profile": "K2_JAX_DEDICATED_DEFAULT_V3", "patches": [], "param_overrides": {}},
    {"id": "L1", "name": "L1: +Pos. homing",
     "profile": "K2_JAX_DEDICATED_DEFAULT_V3_HOMING", "patches": [], "param_overrides": {}},
]
for _c in n10.CONFIGS:
    CONFIGS.append({**_c, "profile": ANCHOR})


def build_worker_script() -> str:
    """Derive the sharded worker from the N=10 measurement script.

    Three edits: take the profile and the rep range from argv, and dump the raw
    per-rep thresholds instead of the N=10 summary statistics (the parent pools
    the shards and does the statistics once).
    """
    src = n10.MEASURE_SCRIPT

    src = src.replace(
        'PROFILE = "K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR"',
        'PROFILE = sys.argv[2]\nREP_START, REP_END = int(sys.argv[3]), int(sys.argv[4])',
    )
    src = src.replace(
        "for rep in range(N_REPS):\n    seed = 20260728 * 100 + rep",
        "for rep in range(REP_START, REP_END):\n    seed = 20260728 * 100 + rep",
    )
    # Drop the progress/ETA arithmetic, which references N_REPS as a rep count.
    src = src.replace(
        "    elapsed = time.time() - t0\n"
        "    if rep < N_REPS - 1:\n"
        "        eta = elapsed / (rep+1) * (N_REPS - rep - 1)\n",
        "",
    )
    # Replace the whole summary block with a raw shard dump.
    head, sep, _tail = src.partition("# Per-rep F_min and F_med")
    if not sep:
        raise RuntimeError("N=10 measure script changed shape; update the split marker")
    src = head + (
        "print(json.dumps({\n"
        '    "rep_start": REP_START, "rep_end": REP_END,\n'
        '    "all_reps": {str(k): v for k, v in all_reps.items()},\n'
        '    "elapsed_min": (time.time() - t0) / 60.0,\n'
        "}))\n"
    )

    for probe in ("REP_START", "sys.argv[2]", "range(REP_START, REP_END)"):
        if probe not in src:
            raise RuntimeError(f"worker script substitution failed: {probe!r} missing")
    return src


WORKER_SCRIPT = build_worker_script()


def shard_bounds(n: int, k: int) -> list[tuple[int, int]]:
    """Contiguous rep slices, remainder spread over the first shards."""
    base, extra = divmod(n, k)
    out, start = [], 0
    for i in range(k):
        stop = start + base + (1 if i < extra else 0)
        if stop > start:
            out.append((start, stop))
        start = stop
    return out


def run_config(cfg: dict, workers: int) -> dict | None:
    """Patch once, fan out rep shards, pool the thresholds."""
    param_json = json.dumps(cfg["param_overrides"])
    bounds = shard_bounds(N_REPS, workers)

    procs = []
    for lo, hi in bounds:
        procs.append((lo, hi, subprocess.Popen(
            [sys.executable, "-c", WORKER_SCRIPT, param_json, cfg["profile"], str(lo), str(hi)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=str(ROOT),
            # One BLAS/XLA thread per worker: 6 workers each grabbing 12 cores
            # thrashes and runs slower than the serial harness it replaces.
            env={**__import__("os").environ, "OMP_NUM_THREADS": "1",
                 "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                 "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false "
                              "intra_op_parallelism_threads=1"},
        )))

    shards = []
    for lo, hi, p in procs:
        out, err = p.communicate(timeout=14400)
        if p.returncode != 0:
            print(f"    shard {lo}:{hi} FAILED rc={p.returncode}\n    {err[-600:]}")
            for _, _, q in procs:
                q.poll() is None and q.kill()
            return None
        line = next((l for l in out.strip().split("\n") if l.strip().startswith('{"')), None)
        if line is None:
            print(f"    shard {lo}:{hi} produced no JSON. tail: {out[-300:]}")
            return None
        shards.append(json.loads(line))

    shards.sort(key=lambda s: s["rep_start"])
    angles = list(shards[0]["all_reps"].keys())
    all_reps = {a: [v for s in shards for v in s["all_reps"][a]] for a in angles}
    n = len(next(iter(all_reps.values())))
    if n != N_REPS:
        print(f"    pooled {n} reps, expected {N_REPS}")
        return None

    import numpy as np
    f_min = [min(all_reps[a][r] for a in angles) for r in range(n)]
    f_med = [sorted(all_reps[a][r] for a in angles)[len(angles) // 2] for r in range(n)]
    stats = {}
    for name, vals in (("F_min", f_min), ("F_med", f_med)):
        m, sd = float(np.mean(vals)), float(np.std(vals, ddof=1))
        stats[f"{name}_mean"] = m
        stats[f"{name}_std"] = sd
        stats[f"{name}_ci95"] = float(T_CRIT * sd / np.sqrt(n))
    return {
        "n_reps": n, "n_directions": len(angles), "profile": cfg["profile"],
        **stats,
        "f_min_per_rep": f_min, "f_med_per_rep": f_med,
        "all_reps": all_reps,
        "elapsed_min": max(s["elapsed_min"] for s in shards),
        "n_workers": len(shards),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="N=30 push-ablation replication")
    ap.add_argument("--configs", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    configs = [c for c in CONFIGS if args.configs is None or c["id"] in args.configs]
    out_path = OUT_DIR / f"ablation_n30_results{tag}.json"
    partial_path = OUT_DIR / f"ablation_n30_partial{tag}.json"

    print("=" * 72)
    print(f"N=30 ablation replication — {len(configs)} configs, {args.workers} workers")
    print("=" * 72)

    results: dict = {}
    t_start = time.time()

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {cfg['name']}  ({cfg['profile']})")
        if cfg["patches"]:
            try:
                n10.apply_patches(cfg["patches"])
                print(f"  {len(cfg['patches'])} patch(es) applied")
            except ValueError as e:
                print(f"  PATCH FAILED: {e}")
                n10.restore_original()
                results[cfg["id"]] = {"error": "patch_failed"}
                continue

        t0 = time.time()
        data = run_config(cfg, args.workers)
        if cfg["patches"]:
            n10.restore_original()

        if data:
            print(f"  OK ({(time.time()-t0)/60:.1f}m)  "
                  f"F_min = {data['F_min_mean']:.1f} ± {data['F_min_std']:.1f} N, "
                  f"F_med = {data['F_med_mean']:.1f} ± {data['F_med_std']:.1f} N")
            results[cfg["id"]] = data
        else:
            print(f"  FAILED ({(time.time()-t0)/60:.1f}m)")
            results[cfg["id"]] = {"error": "measurement_failed"}

        partial_path.write_text(json.dumps(results, indent=2, default=str))

    n10.restore_original()

    results["_metadata"] = {
        "n_reps": N_REPS, "n_directions": 8, "binary_search_tol_N": 5.0,
        "binary_search_iters": 8, "force_range_N": [10.0, 160.0],
        "t_crit_95": T_CRIT, "workers": args.workers,
        "seed_formula": "20260728*100 + rep, rep in [0,30)",
        "total_elapsed_min": (time.time() - t_start) / 60.0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'='*72}\n{'Config':<6} {'F_min':>8} {'±std':>7} {'F_med':>8} {'±std':>7}")
    print("-" * 72)
    for cfg in configs:
        r = results.get(cfg["id"], {})
        if "error" in r:
            print(f"{cfg['id']:<6} {'FAILED':>32}")
        else:
            print(f"{cfg['id']:<6} {r['F_min_mean']:>8.1f} {r['F_min_std']:>7.1f} "
                  f"{r['F_med_mean']:>8.1f} {r['F_med_std']:>7.1f}")
    print(f"\nSaved → {out_path}\nTotal: {results['_metadata']['total_elapsed_min']:.1f} min")


if __name__ == "__main__":
    main()
