"""Run D4/D5 focused validation for the mode-based hip-yaw divergence candidate.

This runner is the candidate-side counterpart to
``scripts/run_d4_d5_hip_yaw_validation.py``. It runs the same D4/D5
push battery for profile D, but enables the new
``--enable-mode-hip-yaw-divergence`` CLI flag and reads the candidate
sagittal profile
``physics_equilibrium_feedforward_outer_loop_low_band_support_v2``.

Profiles:

    A — calibrated_support_position_outer_loop_pitch_ref_v2
    B — physics_equilibrium_feedforward_outer_loop
    C — physics_equilibrium_feedforward_outer_loop_low_band_support_v2
    D — physics_equilibrium_feedforward_outer_loop_low_band_support_v2
        + --enable-mode-hip-yaw-divergence
        + --enable-wheel-yaw-stabilizer (no-op for hip-yaw divergence)

Outputs:

    outputs/mode_based_hip_yaw_divergence_real_sim_validation/
        d4_d5_metrics.csv
        <case>/<tag>/telemetry_*.csv

Run:

    python scripts/run_d4_d5_hip_yaw_div_validation.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_outer_loop_step_d_push as runner

ROOT = Path(__file__).resolve().parent.parent

PROFILE_A = "calibrated_support_position_outer_loop_pitch_ref_v2"
PROFILE_B = "physics_equilibrium_feedforward_outer_loop"
PROFILE_C = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
PROFILE_D = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"

# Default opt-in tunings for the mode-based hip-yaw divergence candidate.
DIV_KP = 5.0
DIV_KD = 0.20
DIV_MAX_TORQUE = 2.0
# Soft-limit set so the controller is active across the protected height band
# (low_0p330 ~ 0.33 m). With soft_limit_rad=0.30 and soft_gain=0.25 the gate
# remains above 0.6 at h=0.40 and reaches zero at h=0.55.
DIV_SOFT_LIMIT_RAD = 0.30
DIV_SOFT_GAIN = 0.25

OUT_BASE = ROOT / "outputs" / "mode_based_hip_yaw_divergence_real_sim_validation"
D4_D5_CASES = [
    ("D4_medium_push_low", "low_0p330", 1000, 60, 5, 150),
    ("D5_large_push_high", "high_0p480", 1000, 90, 5, 200),
]


def _run(profile: str, tag: str, case_id: str, height_label: str, steps: int,
         push_mag: int, push_dur: int, push_int: int) -> tuple:
    """Run a single D4/D5 simulation segment for *profile*.

    Returns (telemetry_path, metrics_dict) like the inner run_sim wrapper.
    """
    out_dir = OUT_BASE / f"step_{case_id}_{tag}"
    is_div = tag == "D"
    if is_div:
        # Enable the new opt-in mode-based hip-yaw divergence controller.
        return _run_direct(
            profile=profile,
            tag=tag,
            case_id=case_id,
            height_label=height_label,
            steps=steps,
            push_mag=push_mag,
            push_dur=push_dur,
            push_int=push_int,
        )
    return runner.run_sim(
        label=height_label,
        steps=steps,
        profile=profile,
        out_dir=out_dir,
        push_magnitude=push_mag,
        push_duration=push_dur,
        push_interval=push_int,
        enable_wheel_yaw=False,
    )


def _run_direct(*, profile: str, tag: str, case_id: str, height_label: str,
                steps: int, push_mag: int, push_dur: int, push_int: int):
    """Call simulate_hierarchical_controller.py with the divergence flag."""
    import shutil
    import subprocess

    out_dir = OUT_BASE / f"step_{case_id}_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{steps}.csv"
    sum_dst = out_dir / "run_summary.json"
    if tel_dst.exists():
        return tel_dst, sum_dst if sum_dst.exists() else None

    setup_path = (
        ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    )
    if not setup_path.exists():
        print(f"  MISSING setup {height_label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode",
        "balance-core",
        "--sagittal-controller",
        "velocity-damped",
        "--vd-sagittal-authority-profile",
        profile,
        "--height-variant-setup",
        str(setup_path),
        "--steps",
        str(steps),
        "--telemetry-decimation",
        "1",
        "--failure-window-steps",
        str(steps),
        "--write-run-summary-sidecar",
        "--output-dir",
        str(out_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp",
        str(DIV_KP),
        "--mode-hip-yaw-div-kd",
        str(DIV_KD),
        "--mode-hip-yaw-div-max-torque",
        str(DIV_MAX_TORQUE),
        "--mode-hip-yaw-div-soft-limit-rad",
        str(DIV_SOFT_LIMIT_RAD),
        "--mode-hip-yaw-div-soft-gain",
        str(DIV_SOFT_GAIN),
        "--push-enabled",
        "--push-magnitude-n",
        str(float(push_mag)),
        "--push-duration-steps",
        str(push_dur),
        "--push-interval-steps",
        str(push_int),
    ]
    print(f"  [D] direct sim -> {case_id}", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {height_label}", flush=True)
        return None, None
    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
    return tel_dst, sum_dst if sum_dst.exists() else None


def main() -> int:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    profile_rows = [
        (PROFILE_A, "A", False),
        (PROFILE_B, "B", False),
        (PROFILE_C, "C", False),
        (PROFILE_D, "D", True),
    ]

    for case_id, height_label, steps, push_mag, push_dur, push_int in D4_D5_CASES:
        for profile, tag, _use_div in profile_rows:
            t0 = time.time()
            tel_path, _ = _run(
                profile=profile,
                tag=tag,
                case_id=case_id,
                height_label=height_label,
                steps=steps,
                push_mag=push_mag,
                push_dur=push_dur,
                push_int=push_int,
            )
            metrics = runner.analyze(tel_path) if tel_path else None
            row = {
                "case_id": case_id,
                "height": height_label,
                "steps": steps,
                "push_mag_N": push_mag,
                "push_dur": push_dur,
                "push_int": push_int,
                "profile": tag,
                "validation_source": "real_simulation",
            }
            if metrics:
                row.update({k: metrics.get(k) for k in (
                    "hip_yaw_abs_max_rad", "max_abs", "min_drift", "max_drift",
                    "p2p", "pitch_max_abs_deg", "roll_rms_deg", "yaw_drift_max_rad",
                    "wbc_authority_rows", "wbc_owner_rows", "hidden_torque_max",
                    "ownership_violation_max", "fell", "term_reason",
                )})
            all_rows.append(row)
            print(
                f"[{case_id}] [{tag}] hip_yaw={row.get('hip_yaw_abs_max_rad', 'NA')} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

    if all_rows:
        csv_path = OUT_BASE / "d4_d5_metrics.csv"
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nMetrics CSV: {csv_path}", flush=True)

    print("\nD4/D5 candidate real-simulation summary:")
    for r in all_rows:
        hy = r.get("hip_yaw_abs_max_rad", 0.0) or 0.0
        try:
            hy = float(hy)
        except (TypeError, ValueError):
            hy = 0.0
        verdict = "PASS" if hy < 0.35 else "FAIL"
        print(f"  [{r['profile']}] {r['case_id']}: hip_yaw={hy:.4f} {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())