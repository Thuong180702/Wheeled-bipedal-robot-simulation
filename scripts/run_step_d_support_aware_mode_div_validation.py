"""Run full Step D validation for selected support-aware mode-div candidate.

Runs D1 through D6 cases for a selected profile + mode-div parameters.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM_SCRIPT = ROOT / "scripts" / "simulate_hierarchical_controller.py"

# Step D cases: (name, height_label, steps, push_mag_N, push_dur_steps, push_int_steps)
STEP_D_CASES = [
    ("D1_small_push_high", "high_0p480", 1000, 30, 5, 150),
    ("D2_medium_push_high", "high_0p480", 1000, 60, 5, 150),
    ("D3_small_push_low", "low_0p330", 1000, 30, 5, 150),
    ("D4_medium_push_low", "low_0p330", 1000, 60, 5, 150),
    ("D5_large_push_high", "high_0p480", 1000, 90, 5, 200),
    ("D6_random_push_high", "high_0p480", 1000, 45, 5, 150),
]

OUTPUT_DIR = ROOT / "outputs" / "support_aware_mode_div_authority_schedule" / "step_d_validation"


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def build_mode_div_cmd(profile: str, height_label: str, steps: int,
                       push_mag: int, push_dur: int, push_int: int,
                       out_dir: Path, mode_div_params: dict | None = None,
                       support_params: dict | None = None) -> list[str]:
    """Build CLI command."""
    setup_path = ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
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
    ]
    # Mode-div flags
    if mode_div_params:
        cmd += [
            "--enable-mode-hip-yaw-divergence",
            "--mode-hip-yaw-div-kp", str(mode_div_params.get("kp", 10.0)),
            "--mode-hip-yaw-div-kd", str(mode_div_params.get("kd", 0.50)),
            "--mode-hip-yaw-div-max-torque", str(mode_div_params.get("max_torque", 7.5)),
            "--mode-hip-yaw-div-soft-limit-rad", str(mode_div_params.get("soft_limit_rad", 0.30)),
            "--mode-hip-yaw-div-soft-gain", str(mode_div_params.get("soft_gain", 0.80)),
            "--mode-hip-yaw-div-ref-source", "target",
        ]
        # Support-aware flags
        if support_params:
            cmd += ["--mode-hip-yaw-div-support-enabled"]
            for key, flag in [
                ("support_threshold_m", "--mode-hip-yaw-div-support-threshold-m"),
                ("support_width_m", "--mode-hip-yaw-div-support-width-m"),
                ("support_min_gate", "--mode-hip-yaw-div-support-min-gate"),
                ("support_rate_threshold_mps", "--mode-hip-yaw-div-support-rate-threshold-mps"),
                ("support_rate_width_mps", "--mode-hip-yaw-div-support-rate-width-mps"),
                ("support_rate_min_gate", "--mode-hip-yaw-div-support-rate-min-gate"),
            ]:
                if key in support_params:
                    cmd += [flag, str(support_params[key])]
    # Push flags
    cmd += [
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_int),
    ]
    return cmd


def run_candidate(candidate_name: str, case_name: str, cmd: list[str],
                  log_path: Path, out_dir: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  {candidate_name} / {case_name}...", end=" ", flush=True)
    t0 = time.time()

    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT,
                                timeout=600, cwd=ROOT)

    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"FAILED (rc={result.returncode}) in {elapsed:.0f}s")
        return False

    tele_path = _find_telemetry_csv(out_dir)
    if tele_path is None:
        print(f"NO TELEMETRY in {elapsed:.0f}s")
        return False

    import csv, math
    with open(tele_path, newline="") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    hy = max(float(r["hip_yaw_abs_max"]) for r in rows) if n > 0 else -1
    sup = max(abs(float(r.get("support_position_error_m", 0))) for r in rows) if n > 0 else -1
    pitch = max(abs(float(r.get("pitch_error", 0))) for r in rows) if n > 0 else 0
    pitch_deg = pitch * 180 / math.pi if pitch else 0
    falls = sum(1 for r in rows if r.get("terminated", "False") == "True")
    print(f"done ({elapsed:.0f}s) hy={hy:.4f} sup={sup:.4f} pitch={pitch_deg:.1f}° rows={n} falls={falls}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", default=None,
                        help="Candidate names to run")
    parser.add_argument("--cases", nargs="+", default=None,
                        help="Step D case names to run")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip if telemetry exists")
    parser.add_argument("--profile", type=str,
                        default="physics_equilibrium_feedforward_outer_loop_low_band_support_v2",
                        help="Sagittal authority profile")
    args = parser.parse_args()

    # Candidate definitions
    candidates = {}

    # D baseline (no mode-div)
    candidates["D_baseline"] = {
        "mode_div": None,
        "support": None,
    }

    # G1_sg080 reference
    candidates["G1_sg080"] = {
        "mode_div": {"kp": 10.0, "kd": 0.50, "max_torque": 7.5, "soft_limit_rad": 0.30, "soft_gain": 0.80},
        "support": None,
    }

    # Selected H candidates (from focused sweep)
    candidates["H1_t30_w10_mg70"] = {
        "mode_div": {"kp": 10.0, "kd": 0.50, "max_torque": 7.5, "soft_limit_rad": 0.30, "soft_gain": 0.80},
        "support": {"support_threshold_m": 0.30, "support_width_m": 0.10, "support_min_gate": 0.70},
    }
    candidates["H2_r80_w40_mg70"] = {
        "mode_div": {"kp": 10.0, "kd": 0.50, "max_torque": 7.5, "soft_limit_rad": 0.30, "soft_gain": 0.80},
        "support": {"support_rate_threshold_mps": 0.80, "support_rate_width_mps": 0.40, "support_rate_min_gate": 0.70},
    }

    total = 0
    passed = 0
    failed = []

    for cand_name, config in candidates.items():
        if args.candidates and cand_name not in args.candidates:
            continue

        md = config["mode_div"]
        sp = config["support"]

        for case_name, height_label, steps, push_mag, push_dur, push_int in STEP_D_CASES:
            if args.cases and case_name not in args.cases:
                continue

            total += 1
            out_dir = OUTPUT_DIR / case_name / cand_name
            log_path = out_dir / "sim.log"

            if args.skip_existing and _find_telemetry_csv(out_dir) is not None:
                print(f"  SKIP {cand_name}/{case_name}: exists")
                passed += 1
                continue

            cmd = build_mode_div_cmd(
                profile=args.profile,
                height_label=height_label,
                steps=steps,
                push_mag=push_mag,
                push_dur=push_dur,
                push_int=push_int,
                out_dir=out_dir,
                mode_div_params=md,
                support_params=sp,
            )

            ok = run_candidate(cand_name, case_name, cmd, log_path, out_dir)
            if ok:
                passed += 1
            else:
                failed.append(f"{cand_name}/{case_name}")

    print(f"\n{'='*60}")
    print(f"  Step D complete: {passed}/{total} passed")
    if failed:
        print(f"  FAILED: {', '.join(failed)}")
    print(f"{'='*60}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
