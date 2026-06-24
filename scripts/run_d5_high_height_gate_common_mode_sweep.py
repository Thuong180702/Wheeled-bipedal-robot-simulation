"""Run D5 high-height G candidate sweep.

Sweeps G candidates (continuous gate and damping variants) on D4 and D5 cases.
Uses direct subprocess calls to simulate_hierarchical_controller.py.

G family candidates (base: F6 = kp=10, kd=0.50, mt=7.5, sl=0.30):

  G1_sg060: sg=0.60  (D5 gate≈0.78)
  G1_sg070: sg=0.70  (D5 gate≈0.84)
  G1_sg080: sg=0.80  (D5 gate≈0.87)
  G3_kd075: sg=0.70, kd=0.75  (higher damping)
  G3_kd100: sg=0.70, kd=1.00  (highest damping)

Each candidate runs:
  - D4_medium_push_low: low_0p330, 60N, 1000 steps, duration 5, interval 150
  - D5_large_push_high: high_0p480, 90N, 1000 steps, duration 5, interval 200
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

# G candidate grid
CANDIDATES: list[tuple[str, float, float, float, float, float]] = [
    # (name,           kp,   kd,   mt,    sl,   sg)
    ("D_baseline",     5.0,  0.20, 2.0,   0.30, 0.25),
    ("F6_reference",   10.0, 0.50, 7.5,   0.30, 0.25),
    ("F6_sg050_ref",   10.0, 0.50, 7.5,   0.30, 0.50),
    ("G1_sg060",       10.0, 0.50, 7.5,   0.30, 0.60),
    ("G1_sg070",       10.0, 0.50, 7.5,   0.30, 0.70),
    ("G1_sg080",       10.0, 0.50, 7.5,   0.30, 0.80),
    ("G3_kd075",       10.0, 0.75, 7.5,   0.30, 0.70),
    ("G3_kd100",       10.0, 1.00, 7.5,   0.30, 0.70),
]

CASES: list[tuple[str, str, int, int, int, int]] = [
    # (case, height_label, steps, push_mag, push_dur, push_int)
    ("D4_medium_push_low", "low_0p330", 1000, 60, 5, 150),
    ("D5_large_push_high", "high_0p480", 1000, 90, 5, 200),
]

OUTPUT_DIR = ROOT / "outputs" / "d5_high_height_mode_div_gate_and_common_mode_coupling_fix" / "sweep"

# Mode-div parameters that stay the same across all G candidates
SOFT_LIMIT_RAD = 0.30
REF_SOURCE = "target"


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    """Find the latest telemetry CSV in out_dir."""
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def build_cmd(
    profile: str,
    height_label: str,
    steps: int,
    push_mag: int,
    push_dur: int,
    push_int: int,
    kp: float,
    kd: float,
    mt: float,
    sl: float,
    sg: float,
    out_dir: Path,
) -> list[str]:
    """Build CLI command for simulate_hierarchical_controller.py."""
    setup_path = (
        ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    )
    cmd = [
        sys.executable, str(SIM_SCRIPT),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        # Mode-div flags
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", f"{kp}",
        "--mode-hip-yaw-div-kd", f"{kd}",
        "--mode-hip-yaw-div-max-torque", f"{mt}",
        "--mode-hip-yaw-div-soft-limit-rad", f"{sl}",
        "--mode-hip-yaw-div-soft-gain", f"{sg}",
        "--mode-hip-yaw-div-ref-source", REF_SOURCE,
        # Push flags
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_int),
    ]
    return cmd


def run_candidate(candidate_name: str, case_name: str, cmd: list[str], log_path: Path, out_dir: Path) -> bool:
    """Run one candidate+case and return True if successful."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  Running {candidate_name} / {case_name}...", end=" ", flush=True)
    t0 = time.time()

    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            timeout=600,
            cwd=ROOT,
        )

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED (rc={result.returncode}) in {elapsed:.0f}s")
        return False

    tele_path = _find_telemetry_csv(out_dir)
    if tele_path is None:
        print(f"NO TELEMETRY in {elapsed:.0f}s")
        return False

    # Read telemetry for quick metrics
    import csv
    with open(tele_path, newline="") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    hy = max(float(r["hip_yaw_abs_max"]) for r in rows) if n > 0 else -1
    sat = sum(1 for r in rows if r.get("mode_hip_yaw_div_tau_left_sat", "False") == "True") if n > 0 else 0
    print(f"done ({elapsed:.0f}s) hy={hy:.4f} rows={n} sat={sat}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Run only these candidate names")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Run only these case names (D4_medium_push_low, D5_large_push_high)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if telemetry CSV already exists")
    parser.add_argument("--profile", type=str,
                        default="physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
                        help="Sagittal authority profile")
    args = parser.parse_args()

    total = 0
    passed = 0
    failed = []

    for cand_name, kp, kd, mt, sl, sg in CANDIDATES:
        if args.candidates and cand_name not in args.candidates:
            continue

        for case_name, height_label, steps, push_mag, push_dur, push_int in CASES:
            if args.cases and case_name not in args.cases:
                continue

            total += 1
            out_dir = OUTPUT_DIR / case_name / cand_name
            log_path = out_dir / "sim.log"

            if args.skip_existing and _find_telemetry_csv(out_dir) is not None:
                print(f"  SKIP {cand_name}/{case_name}: telemetry exists")
                passed += 1
                continue

            cmd = build_cmd(
                profile=args.profile,
                height_label=height_label,
                steps=steps,
                push_mag=push_mag,
                push_dur=push_dur,
                push_int=push_int,
                kp=kp,
                kd=kd,
                mt=mt,
                sl=sl,
                sg=sg,
                out_dir=out_dir,
            )

            ok = run_candidate(cand_name, case_name, cmd, log_path, out_dir)
            if ok:
                passed += 1
            else:
                failed.append(f"{cand_name}/{case_name}")

    print(f"\n{'='*60}")
    print(f"  Sweep complete: {passed}/{total} passed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'='*60}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
