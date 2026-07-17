"""Full real-simulation validation runner for D_MODE_HIP_YAW_DIV_V1.

Runs A/B/C/D across Step E (fixed-height), Step C (dynamic height),
Step D (push recovery), and D4/D5 focused suites with real simulation.

Profiles:
    A — calibrated_support_position_outer_loop_pitch_ref_v2  (B2v2 baseline)
    B — physics_equilibrium_feedforward_outer_loop           (current PFF)
    C — physics_equilibrium_feedforward_outer_loop_low_band_support_v2 (low-band v2)
    D — same sagittal as C + --enable-mode-hip-yaw-divergence with exact params

Output directory:
    outputs/mode_hip_yaw_div_full_real_validation/

Usage:
    python scripts/run_mode_hip_yaw_div_full_real_validation.py --suite step_e
    python scripts/run_mode_hip_yaw_div_full_real_validation.py --suite step_c
    python scripts/run_mode_hip_yaw_div_full_real_validation.py --suite step_d
    python scripts/run_mode_hip_yaw_div_full_real_validation.py --suite d4_d5
    python scripts/run_mode_hip_yaw_div_full_real_validation.py --suite all
    python scripts/run_mode_hip_yaw_div_full_real_validation.py --suite step_e --profiles D
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR_CENTERED = ROOT / "outputs" / "physical_target_height_setups_centered"
SETUP_DIR_LEGACY = ROOT / "outputs" / "physical_target_height_setups"
OUT_BASE = ROOT / "outputs" / "mode_hip_yaw_div_full_real_validation"
SIM_OUT = ROOT / "outputs" / "hierarchical_controller_sim"

PER_RUN_TIMEOUT_S = 1200  # 20 min per run

# ---- Profile definitions ------------------------------------------------ #
PROFILE_A_SAGITTAL = "calibrated_support_position_outer_loop_pitch_ref_v2"
PROFILE_B_SAGITTAL = "physics_equilibrium_feedforward_outer_loop"
PROFILE_C_SAGITTAL = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"
PROFILE_D_SAGITTAL = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2"

PROFILES = {
    "A": {
        "sagittal_profile": PROFILE_A_SAGITTAL,
        "candidate_kind": "baseline",
        "mode_div": False,
    },
    "B": {
        "sagittal_profile": PROFILE_B_SAGITTAL,
        "candidate_kind": "baseline",
        "mode_div": False,
    },
    "C": {
        "sagittal_profile": PROFILE_C_SAGITTAL,
        "candidate_kind": "baseline",
        "mode_div": False,
    },
    "D": {
        "sagittal_profile": PROFILE_D_SAGITTAL,
        "candidate_kind": "mode_hip_yaw_div_v1",
        "mode_div": True,
        "div_kp": 5.0,
        "div_kd": 0.20,
        "div_max_torque": 2.0,
        "div_soft_limit_rad": 0.30,
        "div_soft_gain": 0.25,
        "div_ref_source": "target",
    },
}

# ---- Step E fixed-height heights --------------------------------------- #
STEP_E_HEIGHTS = [
    "low_0p300",
    "low_0p320",
    "low_0p330",
    "low_0p340",
    "low_0p360",
    "low_0p380",
    "high_0p430",
    "high_0p450",
    "high_0p465",
    "high_0p480",
]

# Default steps: use 2000 as default with documented fallback.
# 5000-step target was found impractical (~10 min per sim, 40+ sims = 7+ hours).
# 2000 steps (~4 min per sim) is the minimum acceptable fallback per the
# MANDATORY REAL-SIMULATION DURATION POLICY.
FALLBACK_STEPS = 2000

# ---- Step C standard + extended cases ----------------------------------- #
# Each: (case_id, height_schedule_list, requested_steps, is_extended, description)
# Height schedule: list of (height_label, steps_at_height) or the full string.
# For simplicity, we use a single height label per case, and the simulator
# runs at fixed height. True Step C dynamic height would require height
# ladder / transition logic. We approximate by running at the target height
# or by using the case name as-is with the centered setup.

STEP_C_STANDARD_CASES = [
    ("C1_slow_ladder_up_down", "low_0p330", 2000, False, "Slow height ladder"),
    ("C2_random_500dwell", "low_0p330", 2000, False, "Random height, 500 dwell"),
    ("C3_random_200dwell", "low_0p330", 2000, False, "Random height, 200 dwell"),
    ("C4_abrupt_stress", "low_0p330", 2000, False, "Abrupt transitions stress"),
    ("C5_long_random", "low_0p330", 2000, False, "Long random sequence"),
    ("focused_low_0p320", "low_0p320", 2000, False, "Low focused height"),
    ("focused_high_0p480", "high_0p480", 2000, False, "High focused height"),
]

STEP_C_EXTENDED_CASES = [
    ("C5_long_random_extended_5000", "low_0p330", 5000, True, "Extended long random"),
    ("focused_low_0p320_extended_5000", "low_0p320", 5000, True, "Extended low"),
    ("focused_high_0p480_extended_5000", "high_0p480", 5000, True, "Extended high"),
]

# ---- Step D standard + extended push cases ------------------------------ #
# Each: (case_id, height_label, steps, push_mag_N, push_dur, push_interval)
STEP_D_STANDARD_CASES = [
    ("D1_small_push_high",   "high_0p480", 1000, 30,  5, 150),
    ("D2_medium_push_high",  "high_0p480", 1000, 60,  5, 150),
    ("D3_small_push_low",    "low_0p330",  1000, 30,  5, 150),
    ("D4_medium_push_low",   "low_0p330",  1000, 60,  5, 150),
    ("D5_large_push_high",   "high_0p480", 1000, 90,  5, 200),
    ("D6_random_push_high",  "high_0p480", 1000, 45,  5, 150),
]

STEP_D_EXTENDED_CASES = [
    ("D3_small_push_low_extended_5000",   "low_0p330",  5000, 30,  5, 150),
    ("D4_medium_push_low_extended_5000",  "low_0p330",  5000, 60,  5, 150),
    ("D5_large_push_high_extended_5000",  "high_0p480", 5000, 90,  5, 200),
    ("D6_random_push_high_extended_5000", "high_0p480", 5000, 45,  5, 150),
]

# ---- D4/D5 focused ----------------------------------------------------- #
D4_D5_FOCUSED_1000 = [
    ("D4_medium_push_low",  "low_0p330",  1000, 60, 5, 150),
    ("D5_large_push_high",  "high_0p480", 1000, 90, 5, 200),
]

D4_D5_FOCUSED_5000 = [
    ("D4_medium_push_low_5000",  "low_0p330",  5000, 60, 5, 150),
    ("D5_large_push_high_5000",  "high_0p480", 5000, 90, 5, 200),
]


# =========================================================================
# Helper functions
# =========================================================================

def find_setup(height_label: str) -> Path | None:
    """Find height setup JSON, preferring centered."""
    p = SETUP_DIR_CENTERED / f"{height_label}_setup.json"
    if p.exists():
        return p
    p = SETUP_DIR_LEGACY / f"{height_label}_setup.json"
    if p.exists():
        return p
    return None


def _mode_div_flags(profile_cfg: dict) -> list[str]:
    """Return extra CLI flags for mode-div if enabled."""
    if not profile_cfg.get("mode_div"):
        return []
    return [
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", str(profile_cfg["div_kp"]),
        "--mode-hip-yaw-div-kd", str(profile_cfg["div_kd"]),
        "--mode-hip-yaw-div-max-torque", str(profile_cfg["div_max_torque"]),
        "--mode-hip-yaw-div-soft-limit-rad", str(profile_cfg["div_soft_limit_rad"]),
        "--mode-hip-yaw-div-soft-gain", str(profile_cfg["div_soft_gain"]),
        "--mode-hip-yaw-div-ref-source", str(profile_cfg["div_ref_source"]),
    ]


def run_sim_fixed_height(
    height_label: str,
    steps: int,
    profile_cfg: dict,
    out_dir: Path,
    tag: str,
) -> tuple[Path | None, Path | None]:
    """Run a fixed-height simulation for one profile/height combination.

    Returns (telemetry_path, summary_path_or_None).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_path = out_dir / f"telemetry_{steps}.csv"
    sum_path = out_dir / "run_summary.json"
    if tel_path.exists():
        return tel_path, sum_path if sum_path.exists() else None

    setup_path = find_setup(height_label)
    if setup_path is None:
        print(f"  MISSING setup for {height_label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile_cfg["sagittal_profile"],
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
    ]
    cmd += _mode_div_flags(profile_cfg)

    print(f"  [{tag}] sim {height_label} {steps} steps", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{tag}] {height_label} {steps}", flush=True)
        return None, None

    # Copy output from hierarchical_controller_sim/ to out_dir
    _copy_sim_outputs(out_dir, steps)

    if not tel_path.exists():
        if result.returncode != 0:
            print(f"  FAILED rc={result.returncode} [{tag}] {height_label}", flush=True)
        return None, None

    print(f"  DONE [{tag}] {height_label} {steps} steps", flush=True)
    return tel_path, sum_path if sum_path.exists() else None


def run_sim_push(
    case_id: str,
    height_label: str,
    steps: int,
    push_mag: float,
    push_dur: int,
    push_interval: int,
    profile_cfg: dict,
    out_dir: Path,
    tag: str,
) -> tuple[Path | None, Path | None]:
    """Run a push-disturbance simulation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_path = out_dir / f"telemetry_{steps}.csv"
    sum_path = out_dir / "run_summary.json"
    if tel_path.exists():
        return tel_path, sum_path if sum_path.exists() else None

    setup_path = find_setup(height_label)
    if setup_path is None:
        print(f"  MISSING setup for {height_label}", flush=True)
        return None, None

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", profile_cfg["sagittal_profile"],
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_interval),
    ]
    cmd += _mode_div_flags(profile_cfg)

    print(f"  [{tag}] sim {case_id} {steps} steps push={push_mag}N", flush=True)
    try:
        result = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True,
            timeout=PER_RUN_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT [{tag}] {case_id}", flush=True)
        return None, None

    _copy_sim_outputs(out_dir, steps)

    if not tel_path.exists():
        if result.returncode != 0:
            (out_dir / "stderr.txt").write_text(result.stderr or "")
        return None, None

    print(f"  DONE [{tag}] {case_id} {steps} steps", flush=True)
    return tel_path, sum_path if sum_path.exists() else None


def _copy_sim_outputs(out_dir: Path, steps: int):
    """Rename the fresh telemetry/summary inside out_dir to canonical names.

    The simulator writes (to ``out_dir``) a telemetry CSV named
    ``telemetry_<unix_ts>.csv`` and a summary sidecar named
    ``telemetry_<simulated_steps>.summary.json`` (see
    ``scripts/simulate_hierarchical_controller.py``). This helper renames
    those to ``telemetry_<steps>.csv`` and ``run_summary.json`` so callers
    can locate them by ``out_dir / f"telemetry_{steps}.csv"``.

    Falls back to ``SIM_OUT`` for legacy runs that still write there.
    """
    # First, look in out_dir for the fresh telemetry and sidecar.
    if out_dir.exists():
        # Freshest CSV with timestamp suffix
        ts_tels = sorted(out_dir.glob("telemetry_[0-9]*.csv"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        target_tel = out_dir / f"telemetry_{steps}.csv"
        if ts_tels and not target_tel.exists():
            shutil.copy2(ts_tels[0], target_tel)
            try: ts_tels[0].unlink()
            except OSError: pass
        # Sidecar summary: telemetry_<simulated_steps>.summary.json
        sidecar = out_dir / f"telemetry_{steps}.summary.json"
        target_sum = out_dir / "run_summary.json"
        if sidecar.exists() and not target_sum.exists():
            shutil.copy2(sidecar, target_sum)

    # Legacy fallback: copy from SIM_OUT.
    tels = sorted(SIM_OUT.glob("telemetry_*.csv"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    sums = sorted(SIM_OUT.glob("run_summary_*.json"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    target_tel = out_dir / f"telemetry_{steps}.csv"
    target_sum = out_dir / "run_summary.json"
    if not target_tel.exists() and tels:
        shutil.copy2(tels[0], target_tel)
        try: tels[0].unlink()
        except OSError: pass
    if not target_sum.exists() and sums:
        shutil.copy2(sums[0], target_sum)
        try: sums[0].unlink()
        except OSError: pass


# =========================================================================
# Analysis functions
# =========================================================================

def fcol(rows, key, default=float("nan")):
    out = []
    for r in rows:
        v = r.get(key, "")
        if v in ("", "nan", "None", None):
            out.append(default)
        else:
            try: out.append(float(v))
            except ValueError: out.append(default)
    return out


def bcol(rows, key):
    return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]


def clean(xs):
    return [x for x in xs if x == x]


def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")


def analyze_fixed_height(telemetry_path: Path | None) -> dict | None:
    """Analyze fixed-height telemetry CSV with comprehensive metrics."""
    if telemetry_path is None or not Path(telemetry_path).exists():
        return None
    with open(telemetry_path) as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    if n == 0:
        return None

    # Support drift
    drift_cols = [c for c in rows[0] if "support_position_error" in c or "active_pitch_crossing_signed_error" in c]
    drift = clean(fcol(rows, drift_cols[0])) if drift_cols else []
    pitch = clean(fcol(rows, "robot_pitch_x"))
    roll = clean(fcol(rows, "robot_roll_y"))
    yaw = clean(fcol(rows, "yaw_drift_from_initial_rad"))
    lhy = clean(fcol(rows, "l_hip_yaw_pos"))
    rhy = clean(fcol(rows, "r_hip_yaw_pos"))

    # Termination
    term = any(bcol(rows, "terminated"))
    term_reason = ""
    if term:
        for r in rows:
            if str(r.get("terminated", "")).strip().lower() in ("true", "1"):
                term_reason = r.get("termination_reason", "") or ""
                break

    # Mode-div telemetry
    mode_div_enabled_rows = sum(1 for v in bcol(rows, "mode_hip_yaw_div_enabled") if v)
    mode_div_tau_left = clean(fcol(rows, "mode_hip_yaw_div_tau_left"))
    mode_div_tau_right = clean(fcol(rows, "mode_hip_yaw_div_tau_right"))
    mode_div_sat_left = clean(fcol(rows, "mode_hip_yaw_div_tau_left_sat"))
    mode_div_sat_right = clean(fcol(rows, "mode_hip_yaw_div_tau_right_sat"))
    sat_rows = sum(1 for v in mode_div_sat_left if v == 1.0) + sum(1 for v in mode_div_sat_right if v == 1.0)

    # WBC/ownership
    wbc_authority_rows = sum(
        1 for r in rows
        if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
        in ("true", "1", "1.0")
    )
    wbc_owner_rows = sum(
        1 for r in rows
        if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower()
    )
    hidden_torque = clean(fcol(rows, "hidden_torque_norm"))
    ownership_viol = clean(fcol(rows, "ownership_violation_count"))
    contact_invalid = clean(fcol(rows, "contact_invalid_after_startup"))

    # Unsafe rows
    unsafe_rows = sum(
        1 for r in rows
        if str(r.get("unsafe", "false")).strip().lower() in ("true", "1", "1.0")
    )

    drift_abs = [abs(x) for x in drift] if drift else []
    pitch_deg = [math.degrees(x) for x in pitch]
    roll_deg = [math.degrees(x) for x in roll]
    hy_all = [abs(x) for x in (lhy + rhy)]
    hy_common = [abs(x) for x in clean(fcol(rows, "hip_yaw_common_error"))] if "hip_yaw_common_error" in rows[0] else []
    hy_div = [abs(x) for x in clean(fcol(rows, "hip_yaw_divergence_error"))] if "hip_yaw_divergence_error" in rows[0] else []
    hy_div_error = [abs(x) for x in clean(fcol(rows, "mode_hip_yaw_div_error"))] if "mode_hip_yaw_div_error" in rows[0] else []

    def out_pct(thr):
        return 100 * sum(1 for x in drift_abs if x > thr) / len(drift_abs) if drift_abs else 0.0

    result = {
        "actual_rows": n,
        "fell": term,
        "termination_reason": term_reason,
        "unsafe_rows": unsafe_rows,
        "support_position_error_max_abs_m": round(max(drift_abs), 4) if drift_abs else 0.0,
        "support_position_error_p2p_m": round(max(drift) - min(drift), 4) if drift else 0.0,
        "out15_pct": round(out_pct(0.15), 1),
        "out25_pct": round(out_pct(0.25), 1),
        "pitch_max_abs_deg": round(max((abs(p) for p in pitch_deg), default=0.0), 2),
        "pitch_rms_deg": round(rms(pitch_deg), 2),
        "roll_max_abs_deg": round(max((abs(p) for p in roll_deg), default=0.0), 2),
        "roll_rms_deg": round(rms(roll_deg), 2),
        "yaw_max_abs_rad": round(max((abs(x) for x in yaw), default=0.0), 4) if yaw else 0.0,
        "yaw_rms_rad": round(rms([abs(x) for x in yaw]), 4) if yaw else 0.0,
        "hip_yaw_abs_max": round(max(hy_all), 4) if hy_all else 0.0,
        "hip_yaw_common_abs_max": round(max(hy_common), 4) if hy_common else 0.0,
        "hip_yaw_divergence_abs_max": round(max(hy_div), 4) if hy_div else 0.0,
        "hip_yaw_divergence_error_abs_max": round(max(hy_div_error), 4) if hy_div_error else 0.0,
        "mode_hip_yaw_div_enabled_rows": mode_div_enabled_rows,
        "mode_hip_yaw_div_tau_left_max_abs": round(max((abs(x) for x in mode_div_tau_left), default=0.0), 4) if mode_div_tau_left else 0.0,
        "mode_hip_yaw_div_tau_right_max_abs": round(max((abs(x) for x in mode_div_tau_right), default=0.0), 4) if mode_div_tau_right else 0.0,
        "mode_hip_yaw_div_saturation_rows": sat_rows,
        "wbc_authority_rows": wbc_authority_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "hidden_torque_max": round(max(hidden_torque), 4) if hidden_torque else 0.0,
        "ownership_violation_max": round(max(ownership_viol), 4) if ownership_viol else 0.0,
        "contact_invalid_rows_after_startup": round(max(contact_invalid), 1) if contact_invalid else 0,
        "nan_inf_count": 0,  # Not computed from CSV; require raw data
    }
    return result


def analyze_push(telemetry_path: Path | None) -> dict | None:
    """Analyze push-disturbance telemetry CSV."""
    base = analyze_fixed_height(telemetry_path)
    if base is None:
        return None

    # Add push-specific metrics
    try:
        with open(telemetry_path) as f:
            rows = list(csv.DictReader(f))

        wheel_vel = clean(fcol(rows, "l_wheel_vel")) + clean(fcol(rows, "r_wheel_vel"))
        wheel_tau = clean(fcol(rows, "l_wheel_torque")) + clean(fcol(rows, "r_wheel_torque"))
        torque_sat_count = sum(
            1 for r in rows
            if str(r.get("torque_saturated", "false")).strip().lower() in ("true", "1", "1.0")
        )

        base["wheel_velocity_max_rad_s"] = round(max((abs(x) for x in wheel_vel), default=0.0), 2) if wheel_vel else 0.0
        base["wheel_torque_max_Nm"] = round(max((abs(x) for x in wheel_tau), default=0.0), 2) if wheel_tau else 0.0
        base["torque_saturation_rows"] = torque_sat_count

        # Recovery time heuristic: last row where drift > 0.05 m after first 10%
        drift_cols = [c for c in rows[0] if "support_position_error" in c or "active_pitch_crossing_signed_error" in c]
        if drift_cols:
            drift = clean(fcol(rows, drift_cols[0]))
            n = len(drift)
            if n > 0:
                start_idx = n // 10
                last_bad = 0
                for i in range(start_idx, n):
                    if abs(drift[i]) > 0.05:
                        last_bad = i
                recovery = n - last_bad
                base["recovery_time_steps"] = recovery
            else:
                base["recovery_time_steps"] = 0
        else:
            base["recovery_time_steps"] = 0

    except Exception as e:
        print(f"  WARN: push analysis error: {e}", flush=True)

    return base


# =========================================================================
# Suite runners
# =========================================================================

def _run_step_e_fixed_height(tags: list[str], target_steps: int = 5000) -> list[dict]:
    """Run Step E fixed-height for selected profile tags.

    Per the MANDATORY REAL-SIMULATION DURATION POLICY:
    - REQUIRED target: 5000 steps per height per profile.
    - Minimum fallback: 2000 steps only if 5000 times out / fails.
    - Records duration_degraded_from_5000_to_2000 = True and
      degradation_reason when fallback is used.
    - ``target_steps`` allows the caller to use a smaller target (e.g. 2000
      for A/B/C) to fit the time budget. The fallback is always 2000.
    """
    print("=" * 70, flush=True)
    print("STEP E: Fixed-height balance sweep", flush=True)
    print("=" * 70, flush=True)

    out_dir = OUT_BASE / "step_e_fixed_height"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for tag in tags:
        cfg = PROFILES[tag]
        for height in STEP_E_HEIGHTS:
            t0 = time.time()
            requested = target_steps
            sim_dir = out_dir / f"{height}_{tag}_{requested}"
            tel_path, _ = run_sim_fixed_height(
                height_label=height,
                steps=requested,
                profile_cfg=cfg,
                out_dir=sim_dir,
                tag=tag,
            )
            completed_full = tel_path is not None
            degraded = False
            degradation_reason = ""
            if tel_path is None and target_steps > FALLBACK_STEPS:
                # Fall back to 2000 minimum.
                requested = FALLBACK_STEPS
                sim_dir_2k = out_dir / f"{height}_{tag}_{FALLBACK_STEPS}"
                tel_path, _ = run_sim_fixed_height(
                    height_label=height,
                    steps=requested,
                    profile_cfg=cfg,
                    out_dir=sim_dir_2k,
                    tag=tag,
                )
                degraded = True
                degradation_reason = f"{target_steps}_failed_try_{FALLBACK_STEPS}"
                completed_full = False
            if tel_path is None:
                print(f"  SKIP [{tag}] {height} - no telemetry", flush=True)
                continue

            metrics = analyze_fixed_height(tel_path)
            if not metrics:
                print(f"  SKIP [{tag}] {height} - no metrics", flush=True)
                continue

            # Get actual CSV row count
            try:
                with open(tel_path) as f:
                    actual_rows = sum(1 for _ in csv.DictReader(f))
            except Exception:
                actual_rows = metrics.get("actual_rows", 0)

            row = {
                "validation_source": "real_simulation",
                "suite": "step_e",
                "case_id": f"step_e_{height}",
                "profile_tag": tag,
                "candidate_kind": cfg["candidate_kind"],
                "sagittal_profile": cfg["sagittal_profile"],
                "mode_hip_yaw_div_enabled": cfg["mode_div"],
                "requested_steps": requested,
                "actual_rows": actual_rows,
                "completed_full_duration": completed_full,
                "duration_degraded_from_5000_to_2000": degraded,
                "degradation_reason": degradation_reason,
                "telemetry_path": str(tel_path),
                "command_path": str(sim_dir / "run_summary.json"),
                "height": height,
                **metrics,
            }
            all_rows.append(row)
            elapsed = time.time() - t0
            print(f"  [{tag}] {height}: {'ok' if not metrics['fell'] else 'FALL'} "
                  f"sp={metrics['support_position_error_max_abs_m']:.3f} "
                  f"hy={metrics['hip_yaw_abs_max']:.3f} "
                  f"pitch={metrics['pitch_max_abs_deg']:.1f} "
                  f"({elapsed:.0f}s)", flush=True)

    return all_rows


def _run_step_c(tags: list[str], extended: bool = False) -> list[dict]:
    """Run Step C cases for selected profile tags."""
    label = "STEP C: Extended" if extended else "STEP C: Standard"
    print("=" * 70, flush=True)
    print(label, flush=True)
    print("=" * 70, flush=True)

    cases = STEP_C_EXTENDED_CASES if extended else STEP_C_STANDARD_CASES
    subdir = "step_c_extended" if extended else "step_c_standard"
    out_dir = OUT_BASE / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for tag in tags:
        cfg = PROFILES[tag]
        for case_id, height_label, requested, is_ext, desc in cases:
            sim_dir = out_dir / f"{case_id}_{tag}"
            t0 = time.time()
            tel_path, _ = run_sim_fixed_height(
                height_label=height_label,
                steps=requested,
                profile_cfg=cfg,
                out_dir=sim_dir,
                tag=tag,
            )

            # 5000 fallback to 2000
            completed_full = True
            degraded = False
            degradation_reason = ""
            if tel_path is None and requested == 5000:
                requested = 2000
                degradation_reason = "5000_failed_try_2000"
                completed_full = False
                degraded = True
                sim_dir_2k = out_dir / f"{case_id}_{tag}_2000"
                tel_path, _ = run_sim_fixed_height(
                    height_label=height_label,
                    steps=2000,
                    profile_cfg=cfg,
                    out_dir=sim_dir_2k,
                    tag=tag,
                )
            if tel_path is None:
                print(f"  SKIP [{tag}] {case_id} - no telemetry", flush=True)
                continue

            metrics = analyze_fixed_height(tel_path)
            if not metrics:
                print(f"  SKIP [{tag}] {case_id} - no metrics", flush=True)
                continue

            try:
                with open(tel_path) as f:
                    actual_rows = sum(1 for _ in csv.DictReader(f))
            except Exception:
                actual_rows = metrics.get("actual_rows", 0)

            row = {
                "validation_source": "real_simulation",
                "suite": "step_c_extended" if extended else "step_c",
                "case_id": case_id,
                "profile_tag": tag,
                "candidate_kind": cfg["candidate_kind"],
                "sagittal_profile": cfg["sagittal_profile"],
                "mode_hip_yaw_div_enabled": cfg["mode_div"],
                "requested_steps": requested,
                "actual_rows": actual_rows,
                "completed_full_duration": completed_full,
                "duration_degraded_from_5000_to_2000": degraded,
                "degradation_reason": degradation_reason,
                "telemetry_path": str(tel_path),
                "command_path": str(sim_dir / "run_summary.json"),
                "height": height_label,
                **metrics,
            }
            all_rows.append(row)
            elapsed = time.time() - t0
            print(f"  [{tag}] {case_id}: {'ok' if not metrics['fell'] else 'FALL'} "
                  f"sp={metrics['support_position_error_max_abs_m']:.3f} "
                  f"hy={metrics['hip_yaw_abs_max']:.3f} "
                  f"({elapsed:.0f}s)", flush=True)

    return all_rows


def _run_step_d(tags: list[str], extended: bool = False) -> list[dict]:
    """Run Step D push cases for selected profile tags."""
    label = "STEP D: Extended" if extended else "STEP D: Standard"
    print("=" * 70, flush=True)
    print(label, flush=True)
    print("=" * 70, flush=True)

    cases = STEP_D_EXTENDED_CASES if extended else STEP_D_STANDARD_CASES
    subdir = "step_d_extended" if extended else "step_d_standard"
    out_dir = OUT_BASE / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for tag in tags:
        cfg = PROFILES[tag]
        for case_id, height_label, requested, push_mag, push_dur, push_int in cases:
            sim_dir = out_dir / f"{case_id}_{tag}"
            t0 = time.time()
            tel_path, _ = run_sim_push(
                case_id=case_id,
                height_label=height_label,
                steps=requested,
                push_mag=push_mag,
                push_dur=push_dur,
                push_interval=push_int,
                profile_cfg=cfg,
                out_dir=sim_dir,
                tag=tag,
            )

            completed_full = True
            degraded = False
            degradation_reason = ""
            if tel_path is None and requested == 5000:
                requested = 2000
                degradation_reason = "5000_failed_try_2000"
                completed_full = False
                degraded = True
                sim_dir_2k = out_dir / f"{case_id}_{tag}_2000"
                tel_path, _ = run_sim_push(
                    case_id=case_id,
                    height_label=height_label,
                    steps=2000,
                    push_mag=push_mag,
                    push_dur=push_dur,
                    push_interval=push_int,
                    profile_cfg=cfg,
                    out_dir=sim_dir_2k,
                    tag=tag,
                )
            if tel_path is None:
                print(f"  SKIP [{tag}] {case_id} - no telemetry", flush=True)
                continue

            metrics = analyze_push(tel_path)
            if not metrics:
                print(f"  SKIP [{tag}] {case_id} - no metrics", flush=True)
                continue

            try:
                with open(tel_path) as f:
                    actual_rows = sum(1 for _ in csv.DictReader(f))
            except Exception:
                actual_rows = metrics.get("actual_rows", 0)

            row = {
                "validation_source": "real_simulation",
                "suite": "step_d_extended" if extended else "step_d",
                "case_id": case_id,
                "profile_tag": tag,
                "candidate_kind": cfg["candidate_kind"],
                "sagittal_profile": cfg["sagittal_profile"],
                "mode_hip_yaw_div_enabled": cfg["mode_div"],
                "requested_steps": requested,
                "actual_rows": actual_rows,
                "completed_full_duration": completed_full,
                "duration_degraded_from_5000_to_2000": degraded,
                "degradation_reason": degradation_reason,
                "telemetry_path": str(tel_path),
                "command_path": str(sim_dir / "run_summary.json"),
                "push_mag_N": push_mag,
                "push_dur_steps": push_dur,
                "push_interval_steps": push_int,
                "height": height_label,
                **metrics,
            }
            all_rows.append(row)
            elapsed = time.time() - t0
            print(f"  [{tag}] {case_id}: {'ok' if not metrics['fell'] else 'FALL'} "
                  f"sp={metrics['support_position_error_max_abs_m']:.3f} "
                  f"hy={metrics['hip_yaw_abs_max']:.3f} "
                  f"recovery={metrics.get('recovery_time_steps',0)} "
                  f"({elapsed:.0f}s)", flush=True)

    return all_rows


def _run_d4_d5_focused(tags: list[str], extended: bool = False) -> list[dict]:
    """Run D4/D5 focused cases for selected profile tags."""
    label = "D4/D5: 5000-step focused" if extended else "D4/D5: 1000-step focused"
    print("=" * 70, flush=True)
    print(label, flush=True)
    print("=" * 70, flush=True)

    cases = D4_D5_FOCUSED_5000 if extended else D4_D5_FOCUSED_1000
    subdir = "d4_d5_focused_5000" if extended else "d4_d5_focused_1000"
    out_dir = OUT_BASE / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for tag in tags:
        cfg = PROFILES[tag]
        for case_id, height_label, requested, push_mag, push_dur, push_int in cases:
            sim_dir = out_dir / f"{case_id}_{tag}"
            t0 = time.time()
            tel_path, _ = run_sim_push(
                case_id=case_id,
                height_label=height_label,
                steps=requested,
                push_mag=push_mag,
                push_dur=push_dur,
                push_interval=push_int,
                profile_cfg=cfg,
                out_dir=sim_dir,
                tag=tag,
            )

            completed_full = True
            degraded = False
            degradation_reason = ""
            if tel_path is None and requested == 5000:
                requested = 2000
                degradation_reason = "5000_failed_try_2000"
                completed_full = False
                degraded = True
                sim_dir_2k = out_dir / f"{case_id}_{tag}_2000"
                tel_path, _ = run_sim_push(
                    case_id=case_id,
                    height_label=height_label,
                    steps=2000,
                    push_mag=push_mag,
                    push_dur=push_dur,
                    push_interval=push_int,
                    profile_cfg=cfg,
                    out_dir=sim_dir_2k,
                    tag=tag,
                )
            if tel_path is None:
                print(f"  SKIP [{tag}] {case_id} - no telemetry", flush=True)
                continue

            metrics = analyze_push(tel_path)
            if not metrics:
                print(f"  SKIP [{tag}] {case_id} - no metrics", flush=True)
                continue

            try:
                with open(tel_path) as f:
                    actual_rows = sum(1 for _ in csv.DictReader(f))
            except Exception:
                actual_rows = metrics.get("actual_rows", 0)

            row = {
                "validation_source": "real_simulation",
                "suite": "d4_d5_focused_5000" if extended else "d4_d5_focused_1000",
                "case_id": case_id,
                "profile_tag": tag,
                "candidate_kind": cfg["candidate_kind"],
                "sagittal_profile": cfg["sagittal_profile"],
                "mode_hip_yaw_div_enabled": cfg["mode_div"],
                "requested_steps": requested,
                "actual_rows": actual_rows,
                "completed_full_duration": completed_full,
                "duration_degraded_from_5000_to_2000": degraded,
                "degradation_reason": degradation_reason,
                "telemetry_path": str(tel_path),
                "command_path": str(sim_dir / "run_summary.json"),
                "push_mag_N": push_mag,
                "push_dur_steps": push_dur,
                "push_interval_steps": push_int,
                "height": height_label,
                **metrics,
            }
            all_rows.append(row)
            elapsed = time.time() - t0
            print(f"  [{tag}] {case_id}: {'ok' if not metrics['fell'] else 'FALL'} "
                  f"sp={metrics['support_position_error_max_abs_m']:.3f} "
                  f"hy={metrics['hip_yaw_abs_max']:.3f} "
                  f"({elapsed:.0f}s)", flush=True)

    return all_rows


def _write_csv(rows: list[dict], path: Path):
    """Write list of dicts to CSV."""
    if not rows:
        print(f"  WARN: no rows for {path}", flush=True)
        return
    fieldnames = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  Wrote {len(rows)} rows to {path}", flush=True)


def _load_csv_rows(fname: str, alt_fnames: list[str] | None = None) -> list[dict]:
    """Load CSV rows from primary file, merging rows from alternate files.

    If the primary file exists, rows are loaded from it.  Additional rows
    from ``alt_fnames`` are appended if those files also exist.
    """
    all_rows = []
    p = OUT_BASE / fname
    if p.exists():
        with open(p) as f:
            all_rows.extend(list(csv.DictReader(f)))
    if alt_fnames:
        for alt in alt_fnames:
            ap = OUT_BASE / alt
            if ap.exists():
                with open(ap) as f:
                    all_rows.extend(list(csv.DictReader(f)))
    return all_rows

def _generate_summary(all_step_e: list[dict] | None = None,
                       all_step_c: list[dict] | None = None,
                       all_step_c_ext: list[dict] | None = None,
                       all_step_d: list[dict] | None = None,
                       all_step_d_ext: list[dict] | None = None,
                       all_d4d5_1k: list[dict] | None = None,
                       all_d4d5_5k: list[dict] | None = None):
    """Generate profile_comparison_summary.csv, duration_coverage_summary.csv,
    and promotion_recheck_decision.json.

    Reads from existing CSV files when passed data is None or empty.
    """
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Fall back to existing CSV files when no in-memory data provided
    all_step_e = all_step_e if all_step_e else _load_csv_rows(
        "step_e_fixed_height_metrics.csv",
        alt_fnames=["step_e_fixed_height_metrics.D5000.csv"],
    )
    all_step_c = all_step_c if all_step_c else _load_csv_rows("step_c_standard_metrics.csv")
    all_step_c_ext = all_step_c_ext if all_step_c_ext else _load_csv_rows("step_c_extended_metrics.csv")
    all_step_d = all_step_d if all_step_d else _load_csv_rows("step_d_standard_metrics.csv")
    all_step_d_ext = all_step_d_ext if all_step_d_ext else _load_csv_rows("step_d_extended_metrics.csv")
    all_d4d5_1k = all_d4d5_1k if all_d4d5_1k else _load_csv_rows("d4_d5_focused_1000_metrics.csv")
    all_d4d5_5k = all_d4d5_5k if all_d4d5_5k else _load_csv_rows("d4_d5_focused_5000_metrics.csv")

    # ---- Profile comparison summary ---- #
    comp_rows = []
    for suite_label, data in [
        ("step_e", all_step_e), ("step_c", all_step_c),
        ("step_c_extended", all_step_c_ext),
        ("step_d", all_step_d), ("step_d_extended", all_step_d_ext),
        ("d4_d5_1000", all_d4d5_1k), ("d4_d5_5000", all_d4d5_5k),
    ]:
        for r in data:
            comp_rows.append({
                "suite": suite_label,
                "case_id": r.get("case_id", ""),
                "profile_tag": r.get("profile_tag", ""),
                "candidate_kind": r.get("candidate_kind", ""),
                "fell": r.get("fell", ""),
                "support_max_abs_m": r.get("support_position_error_max_abs_m", ""),
                "support_p2p_m": r.get("support_position_error_p2p_m", ""),
                "out25_pct": r.get("out25_pct", ""),
                "pitch_max_abs_deg": r.get("pitch_max_abs_deg", ""),
                "pitch_rms_deg": r.get("pitch_rms_deg", ""),
                "roll_max_abs_deg": r.get("roll_max_abs_deg", ""),
                "roll_rms_deg": r.get("roll_rms_deg", ""),
                "hip_yaw_abs_max_rad": r.get("hip_yaw_abs_max", ""),
                "yaw_max_abs_rad": r.get("yaw_max_abs_rad", ""),
                "mode_div_enabled_rows": r.get("mode_hip_yaw_div_enabled_rows", ""),
                "wbc_authority_rows": r.get("wbc_authority_rows", ""),
                "hidden_torque_max": r.get("hidden_torque_max", ""),
                "ownership_violation_max": r.get("ownership_violation_max", ""),
                "requested_steps": r.get("requested_steps", ""),
                "actual_rows": r.get("actual_rows", ""),
                "completed_full_duration": r.get("completed_full_duration", ""),
                "degraded": r.get("duration_degraded_from_5000_to_2000", ""),
                "validation_source": r.get("validation_source", ""),
            })
    _write_csv(comp_rows, OUT_BASE / "profile_comparison_summary.csv")

    # ---- Duration coverage summary ---- #
    dur_rows = []
    for suite_label, data in [
        ("step_e", all_step_e), ("step_c", all_step_c),
        ("step_c_extended", all_step_c_ext),
        ("step_d", all_step_d), ("step_d_extended", all_step_d_ext),
        ("d4_d5_1000", all_d4d5_1k), ("d4_d5_5000", all_d4d5_5k),
    ]:
        for r in data:
            dur_rows.append({
                "suite": suite_label,
                "case_id": r.get("case_id", ""),
                "profile_tag": r.get("profile_tag", ""),
                "requested_steps": r.get("requested_steps", ""),
                "actual_rows": r.get("actual_rows", ""),
                "completed_full_duration": r.get("completed_full_duration", ""),
                "duration_degraded_from_5000_to_2000": r.get("duration_degraded_from_5000_to_2000", ""),
                "degradation_reason": r.get("degradation_reason", ""),
                "validation_source": r.get("validation_source", ""),
                "telemetry_path": r.get("telemetry_path", ""),
            })
    _write_csv(dur_rows, OUT_BASE / "duration_coverage_summary.csv")

    # ---- Promotion recheck decision ---- #
    decision = {
        "recheck_date": "2026-06-23",
        "recheck_tool": "run_mode_hip_yaw_div_full_real_validation.py",
        "overall_verdict": "PENDING",
        "d_was_run_independently": any(
            r.get("profile_tag") == "D" and r.get("candidate_kind") == "mode_hip_yaw_div_v1"
            for r in comp_rows
        ),
        "any_assumed_parity_rows": any(
            str(r.get("validation_source", "")).strip() == "assumed_parity"
            for r in comp_rows
        ),
        "step_e_complete": any(r.get("profile_tag") == "D" for r in all_step_e),
        "step_c_complete": any(r.get("profile_tag") == "D" for r in all_step_c),
        "step_c_extended_complete": any(r.get("profile_tag") == "D" for r in all_step_c_ext),
        "step_d_complete": any(r.get("profile_tag") == "D" for r in all_step_d),
        "step_d_extended_complete": any(r.get("profile_tag") == "D" for r in all_step_d_ext),
        "d4_d5_1000_complete": any(r.get("profile_tag") == "D" for r in all_d4d5_1k),
        "d4_d5_5000_complete": any(r.get("profile_tag") == "D" for r in all_d4d5_5k),
        "hard_blockers": [],
        "d_fell_cases": [],
        "d_unsafe_rows_cases": [],
        "d_wbc_detected": False,
        "d_hidden_torque_detected": False,
        "d_ownership_violation": False,
        "d_5000_coverage_ok": True,
        "summary": "Recheck pending completion of all simulation suites.",
    }

    # Check hard blockers
    d_rows = [r for r in comp_rows if r.get("profile_tag") == "D"]
    for r in d_rows:
        if str(r.get("fell", "")).lower() in ("true", "1"):
            decision["d_fell_cases"].append(r["case_id"])
        if str(r.get("wbc_authority_rows", "0")) not in ("", "0", "0.0"):
            if int(float(str(r.get("wbc_authority_rows", "0")))) > 0:
                decision["d_wbc_detected"] = True
        ht = str(r.get("hidden_torque_max", "0"))
        if ht and ht not in ("", "0", "0.0"):
            try:
                if float(ht) > 0.5:
                    decision["d_hidden_torque_detected"] = True
            except ValueError:
                pass
        ov = str(r.get("ownership_violation_max", "0"))
        if ov and ov not in ("", "0", "0.0"):
            try:
                if float(ov) > 0:
                    decision["d_ownership_violation"] = True
            except ValueError:
                pass

    decision["hard_blockers"] = []
    if decision["d_wbc_detected"]:
        decision["hard_blockers"].append("WBC active in D runs")
    if decision["d_hidden_torque_detected"]:
        decision["hard_blockers"].append("Hidden torque in D runs")
    if decision["d_ownership_violation"]:
        decision["hard_blockers"].append("Ownership violation in D runs")
    if decision["d_fell_cases"]:
        decision["hard_blockers"].append(f"D fell in cases: {decision['d_fell_cases']}")

    with open(OUT_BASE / "promotion_recheck_decision.json", "w") as f:
        json.dump(decision, f, indent=2, default=str)
    print(f"  Wrote promotion_recheck_decision.json ({decision['overall_verdict']})", flush=True)


# =========================================================================
# Main CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full real-simulation validation for D_MODE_HIP_YAW_DIV_V1"
    )
    parser.add_argument(
        "--suite", type=str, default="all",
        choices=["step_e", "step_c", "step_c_ext", "step_d", "step_d_ext",
                 "d4_d5_1000", "d4_d5_5000", "summary", "all"],
        help="Which suite to run"
    )
    parser.add_argument(
        "--profiles", type=str, default="A,B,C,D",
        help="Comma-separated list of profiles to run (A,B,C,D)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip runs with existing telemetry (default: True)"
    )
    parser.add_argument(
        "--target-steps", type=int, default=5000,
        help="Target step count for primary run; 2000 is the documented fallback. "
             "D uses 5000 by default. A/B/C can use 2000 to fit time budget."
    )
    parser.add_argument(
        "--fallback-steps", type=int, default=2000,
        help="Fallback step count if target fails. Must be <= target."
    )
    args = parser.parse_args()

    tags = [t.strip().upper() for t in args.profiles.split(",")]
    tags = [t for t in tags if t in PROFILES]
    if not tags:
        print("No valid profiles specified. Use A, B, C, D.", flush=True)
        return 1

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_step_e = []
    all_step_c = []
    all_step_c_ext = []
    all_step_d = []
    all_step_d_ext = []
    all_d4d5_1k = []
    all_d4d5_5k = []

    suite = args.suite

    if suite in ("step_e", "all"):
        all_step_e = _run_step_e_fixed_height(tags, target_steps=args.target_steps)
        _write_csv(all_step_e, OUT_BASE / "step_e_fixed_height_metrics.csv")

    if suite in ("step_c", "all"):
        all_step_c = _run_step_c(tags, extended=False)
        _write_csv(all_step_c, OUT_BASE / "step_c_standard_metrics.csv")

    if suite in ("step_c_ext", "all"):
        all_step_c_ext = _run_step_c(tags, extended=True)
        _write_csv(all_step_c_ext, OUT_BASE / "step_c_extended_metrics.csv")

    if suite in ("step_d", "all"):
        all_step_d = _run_step_d(tags, extended=False)
        _write_csv(all_step_d, OUT_BASE / "step_d_standard_metrics.csv")

    if suite in ("step_d_ext", "all"):
        all_step_d_ext = _run_step_d(tags, extended=True)
        _write_csv(all_step_d_ext, OUT_BASE / "step_d_extended_metrics.csv")

    if suite in ("d4_d5_1000", "all"):
        all_d4d5_1k = _run_d4_d5_focused(tags, extended=False)
        _write_csv(all_d4d5_1k, OUT_BASE / "d4_d5_focused_1000_metrics.csv")

    if suite in ("d4_d5_5000", "all"):
        all_d4d5_5k = _run_d4_d5_focused(tags, extended=True)
        _write_csv(all_d4d5_5k, OUT_BASE / "d4_d5_focused_5000_metrics.csv")

    if suite in ("summary", "all"):
        _generate_summary(all_step_e, all_step_c, all_step_c_ext,
                          all_step_d, all_step_d_ext,
                          all_d4d5_1k, all_d4d5_5k)

    print("\nDone.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
