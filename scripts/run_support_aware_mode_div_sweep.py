"""Run support-aware mode-div authority schedule sweep (H candidates).

Sweeps H candidates (support-aware gating variants) on D4 and D5 cases.
Extends G1_sg080 with continuous support-aware modulation.

H family candidates (base: G1_sg080 = kp=10, kd=0.50, mt=7.5, sl=0.30, sg=0.80):

  H1_sXX_mYY_sgZZ: Support-error attenuation
    threshold_m = XX/100, width_m = YY/100, min_gate = ZZ/100

  H2_rXX_mYY_sgZZ: Support-rate attenuation
    rate_threshold_mps = XX/1000, rate_width_mps = YY/1000, min_gate = ZZ/100

  H3_combo_XX: Combined error + rate attenuation (min of both gates)
    threshold_m=XX, uses rate params too

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

# H candidate grid
# (name, support_threshold_m, support_width_m, support_min_gate,
#        rate_threshold_mps, rate_width_mps, rate_min_gate)
# None means use the same as the error param equivalent
CANDIDATES: list[tuple[str, float | None, float | None, float | None,
                       float | None, float | None, float | None]] = [
    # ---- H1: support-error attenuation ---- #
    ("H1_t25_w05_mg70", 0.25, 0.05, 0.70, None, None, None),
    ("H1_t25_w10_mg70", 0.25, 0.10, 0.70, None, None, None),
    ("H1_t30_w05_mg70", 0.30, 0.05, 0.70, None, None, None),
    ("H1_t30_w10_mg70", 0.30, 0.10, 0.70, None, None, None),
    ("H1_t30_w15_mg70", 0.30, 0.15, 0.70, None, None, None),
    # ---- H2: support-rate attenuation ---- #
    ("H2_r50_w30_mg70", None, None, None, 0.50, 0.30, 0.70),
    ("H2_r80_w40_mg70", None, None, None, 0.80, 0.40, 0.70),
    ("H2_r100_w50_mg60", None, None, None, 1.00, 0.50, 0.60),
    # ---- H3: combined error + rate ---- #
    ("H3_t30_w10_mg70_r80", 0.30, 0.10, 0.70, 0.80, 0.40, 0.70),
    ("H3_t30_w15_mg70_r100", 0.30, 0.15, 0.70, 1.00, 0.50, 0.60),
    ("H3_t35_w10_mg60_r80", 0.35, 0.10, 0.60, 0.80, 0.40, 0.60),
]

# Base mode-div parameters (G1_sg080)
KP = 10.0
KD = 0.50
MAX_TORQUE = 7.5
SOFT_LIMIT_RAD = 0.30
SOFT_GAIN = 0.80
REF_SOURCE = "target"

CASES: list[tuple[str, str, int, int, int, int]] = [
    ("D4_medium_push_low", "low_0p330", 1000, 60, 5, 150),
    ("D5_large_push_high", "high_0p480", 1000, 90, 5, 200),
]

OUTPUT_DIR = ROOT / "outputs" / "support_aware_mode_div_authority_schedule" / "sweep"


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
    cand_name: str,
    cand_params: tuple,
    out_dir: Path,
) -> list[str]:
    """Build CLI command for simulate_hierarchical_controller.py."""
    setup_path = (
        ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    )
    _, support_threshold_m, support_width_m, support_min_gate, \
        rate_threshold_mps, rate_width_mps, rate_min_gate = cand_params

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
        # Mode-div flags (G1_sg080 base)
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", f"{KP}",
        "--mode-hip-yaw-div-kd", f"{KD}",
        "--mode-hip-yaw-div-max-torque", f"{MAX_TORQUE}",
        "--mode-hip-yaw-div-soft-limit-rad", f"{SOFT_LIMIT_RAD}",
        "--mode-hip-yaw-div-soft-gain", f"{SOFT_GAIN}",
        "--mode-hip-yaw-div-ref-source", REF_SOURCE,
        # Support-aware flags
    ]
    # Only add support flags for H candidates (not baseline/G1_sg080 reference)
    if cand_name.startswith("H"):
        cmd += ["--mode-hip-yaw-div-support-enabled"]
        if support_threshold_m is not None:
            cmd += ["--mode-hip-yaw-div-support-threshold-m", f"{support_threshold_m}"]
        if support_width_m is not None:
            cmd += ["--mode-hip-yaw-div-support-width-m", f"{support_width_m}"]
        if support_min_gate is not None:
            cmd += ["--mode-hip-yaw-div-support-min-gate", f"{support_min_gate}"]
        if rate_threshold_mps is not None:
            cmd += ["--mode-hip-yaw-div-support-rate-threshold-mps", f"{rate_threshold_mps}"]
        if rate_width_mps is not None:
            cmd += ["--mode-hip-yaw-div-support-rate-width-mps", f"{rate_width_mps}"]
        if rate_min_gate is not None:
            cmd += ["--mode-hip-yaw-div-support-rate-min-gate", f"{rate_min_gate}"]

    # Push flags
    cmd += [
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
    sup = max(abs(float(r.get("support_position_error_m", 0.0))) for r in rows) if n > 0 else -1
    pitch = max(abs(float(r.get("pitch_error", 0.0))) for r in rows) if n > 0 else 0
    import math
    pitch_deg = pitch * 180 / math.pi if pitch else 0
    falls = sum(1 for r in rows if r.get("terminated", "False") == "True") if n > 0 else 0
    print(f"done ({elapsed:.0f}s) hy={hy:.4f} sup={sup:.4f} pitch={pitch_deg:.1f}° rows={n} falls={falls}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Run only these candidate names")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Run only these case names")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if telemetry CSV already exists")
    parser.add_argument("--profile", type=str,
                        default="physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
                        help="Sagittal authority profile")
    parser.add_argument("--only-references", action="store_true",
                        help="Only run D and G1_sg080 references (no H)")
    parser.add_argument("--only-h", action="store_true",
                        help="Only run H candidates (skip references)")
    args = parser.parse_args()

    total = 0
    passed = 0
    failed = []

    # Reference profiles (D baseline = no mode-div, G1_sg080 = mode-div gate widened)
    reference_profiles = [
        ("D_baseline", False, None),  # mode-div disabled
        ("G1_sg080_ref", True, None),  # G1_sg080 base, no support-aware
    ]

    for cand_name, support_threshold_m, support_width_m, support_min_gate, \
            rate_threshold_mps, rate_width_mps, rate_min_gate in CANDIDATES:
        if args.candidates and cand_name not in args.candidates:
            continue
        if args.only_references:
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

            cand_params = (cand_name, support_threshold_m, support_width_m,
                          support_min_gate, rate_threshold_mps, rate_width_mps, rate_min_gate)
            cmd = build_cmd(
                profile=args.profile,
                height_label=height_label,
                steps=steps,
                push_mag=push_mag,
                push_dur=push_dur,
                push_int=push_int,
                cand_name=cand_name,
                cand_params=cand_params,
                out_dir=out_dir,
            )

            ok = run_candidate(cand_name, case_name, cmd, log_path, out_dir)
            if ok:
                passed += 1
            else:
                failed.append(f"{cand_name}/{case_name}")

    # References
    if not args.only_h:
        for ref_name, mode_div_enabled, _ in reference_profiles:
            if args.candidates and ref_name not in args.candidates:
                continue

            for case_name, height_label, steps, push_mag, push_dur, push_int in CASES:
                if args.cases and case_name not in args.cases:
                    continue

                total += 1
                out_dir = OUTPUT_DIR / case_name / ref_name
                log_path = out_dir / "sim.log"

                if args.skip_existing and _find_telemetry_csv(out_dir) is not None:
                    print(f"  SKIP {ref_name}/{case_name}: telemetry exists")
                    passed += 1
                    continue

                setup_path = (
                    ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
                )
                cmd = [
                    sys.executable, str(SIM_SCRIPT),
                    "--controller-mode", "balance-core",
                    "--sagittal-controller", "velocity-damped",
                    "--vd-sagittal-authority-profile", args.profile,
                    "--height-variant-setup", str(setup_path),
                    "--steps", str(steps),
                    "--telemetry-decimation", "1",
                    "--failure-window-steps", str(steps),
                    "--write-run-summary-sidecar",
                    "--output-dir", str(out_dir),
                ]
                if mode_div_enabled:
                    cmd += [
                        "--enable-mode-hip-yaw-divergence",
                        "--mode-hip-yaw-div-kp", f"{KP}",
                        "--mode-hip-yaw-div-kd", f"{KD}",
                        "--mode-hip-yaw-div-max-torque", f"{MAX_TORQUE}",
                        "--mode-hip-yaw-div-soft-limit-rad", f"{SOFT_LIMIT_RAD}",
                        "--mode-hip-yaw-div-soft-gain", f"{SOFT_GAIN}",
                        "--mode-hip-yaw-div-ref-source", REF_SOURCE,
                    ]
                cmd += [
                    "--push-enabled",
                    "--push-magnitude-n", str(float(push_mag)),
                    "--push-duration-steps", str(push_dur),
                    "--push-interval-steps", str(push_int),
                ]

                ok = run_candidate(ref_name, case_name, cmd, log_path, out_dir)
                if ok:
                    passed += 1
                else:
                    failed.append(f"{ref_name}/{case_name}")

    print(f"\n{'='*60}")
    print(f"  Sweep complete: {passed}/{total} passed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'='*60}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
