"""D4/D5 wheel-yaw correct actuator fix — parameter sweep.

Runs the D_MODE_HIP_YAW_DIV_V1 baseline and E_MODE_HIP_YAW_DIV_PLUS_WHEEL_YAW_V1
candidates across the D4 and D5 push cases. Each E candidate enables both

    --enable-mode-hip-yaw-divergence
    --enable-wheel-yaw-stabilizer

with a specific set of wheel-yaw parameters from the sweep grid.

Output:
    outputs/d4_d5_wheel_yaw_correct_actuator_fix/sweep/
        sweep_metrics.csv          — aggregate across candidates
        sign_verification.csv     — per-step sign correctness analysis
        <candidate>/telemetry_*.csv  — raw telemetry per run

Run:
    python scripts/run_d4_d5_wheel_yaw_correct_actuator_sweep.py
"""

from __future__ import annotations

import csv
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Canonical profile names
PROFILE_D = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_v1"
PROFILE_E = "physics_equilibrium_feedforward_outer_loop_low_band_support_v2_mode_hip_yaw_div_wheel_yaw_v1"

# D current-best flags (mode-div only)
D_DIV_KP = 5.0
D_DIV_KD = 0.20
D_DIV_MAX_TORQUE = 2.0
D_DIV_SOFT_LIMIT_RAD = 0.30
D_DIV_SOFT_GAIN = 0.25
D_DIV_REF_SOURCE = "target"

OUT_BASE = ROOT / "outputs" / "d4_d5_wheel_yaw_correct_actuator_fix" / "sweep"

# D4/D5 cases: (case_id, height_label, steps, push_mag_N, push_dur, push_int)
PUSH_CASES = [
    ("D4_medium_push_low", "low_0p330", 1000, 60, 5, 150),
    ("D5_large_push_high", "high_0p480", 1000, 90, 5, 200),
]

# Wheel-yaw parameter grid for E candidates.
# Each tuple: (kp, kd, max_torque, lowpass_alpha)
WHEEL_YAW_GRID = [
    # Conservative
    (0.25, 0.05, 1.0, 0.4),
    (0.50, 0.10, 1.0, 0.4),
    (0.50, 0.10, 2.0, 0.4),
    # Moderate
    (1.00, 0.10, 2.0, 0.4),
    (1.00, 0.20, 3.0, 0.4),
    (1.50, 0.10, 2.0, 0.4),
    (1.50, 0.20, 3.0, 0.4),
    # Higher authority
    (2.00, 0.10, 3.0, 0.4),
    (2.00, 0.20, 3.0, 0.4),
    (2.00, 0.35, 5.0, 0.4),
    # No-filter variants for comparison
    (1.00, 0.10, 2.0, 1.0),
    (2.00, 0.20, 3.0, 1.0),
]

WHEEL_YAW_HEIGHT_GATE_LOW = 0.250
WHEEL_YAW_HEIGHT_GATE_HIGH = 0.350


def _build_e_cmd(
    profile: str,
    height_label: str,
    steps: int,
    push_mag: int,
    push_dur: int,
    push_int: int,
    wy_kp: float,
    wy_kd: float,
    wy_max_torque: float,
    wy_lowpass_alpha: float,
    out_dir: Path,
) -> list[str]:
    """Build the simulation CLI command for an E candidate."""
    setup_path = (
        ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    )
    cmd = [
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
        "--output-dir", str(out_dir),
        # Mode-div flags (same as D)
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", str(D_DIV_KP),
        "--mode-hip-yaw-div-kd", str(D_DIV_KD),
        "--mode-hip-yaw-div-max-torque", str(D_DIV_MAX_TORQUE),
        "--mode-hip-yaw-div-soft-limit-rad", str(D_DIV_SOFT_LIMIT_RAD),
        "--mode-hip-yaw-div-soft-gain", str(D_DIV_SOFT_GAIN),
        "--mode-hip-yaw-div-ref-source", D_DIV_REF_SOURCE,
        # Wheel-yaw flags
        "--enable-wheel-yaw-stabilizer",
        "--wheel-yaw-kp", str(float(wy_kp)),
        "--wheel-yaw-kd", str(float(wy_kd)),
        "--wheel-yaw-max-torque", str(float(wy_max_torque)),
        "--wheel-yaw-lowpass-alpha", str(float(wy_lowpass_alpha)),
        "--wheel-yaw-height-gate-low", str(WHEEL_YAW_HEIGHT_GATE_LOW),
        "--wheel-yaw-height-gate-high", str(WHEEL_YAW_HEIGHT_GATE_HIGH),
        # Push flags
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_int),
    ]
    return cmd


def _find_telemetry_csv(out_dir: Path) -> Path | None:
    """Find the telemetry CSV in out_dir (simulator writes timestamped names)."""
    csvs = sorted(out_dir.glob("telemetry_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return csvs[0] if csvs else None


def _find_run_summary(out_dir: Path) -> Path | None:
    """Find the run summary JSON."""
    sums = sorted(out_dir.glob("*summary*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return sums[0] if sums else None


def _run_e_candidate(
    case_id: str,
    height_label: str,
    steps: int,
    push_mag: int,
    push_dur: int,
    push_int: int,
    wy_kp: float,
    wy_kd: float,
    wy_max_torque: float,
    wy_lowpass_alpha: float,
) -> tuple[Path | None, Path | None]:
    """Run one E candidate simulation and return telemetry/summary paths."""
    candidate_label = f"E_kp{wy_kp}_kd{wy_kd}_mt{wy_max_torque}_lp{wy_lowpass_alpha}"
    out_dir = OUT_BASE / f"{case_id}" / candidate_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    existing_tel = _find_telemetry_csv(out_dir)
    existing_sum = _find_run_summary(out_dir)
    if existing_tel is not None and existing_sum is not None:
        return existing_tel, existing_sum

    cmd = _build_e_cmd(
        profile=PROFILE_E,
        height_label=height_label,
        steps=steps,
        push_mag=push_mag,
        push_dur=push_dur,
        push_int=push_int,
        wy_kp=wy_kp,
        wy_kd=wy_kd,
        wy_max_torque=wy_max_torque,
        wy_lowpass_alpha=wy_lowpass_alpha,
        out_dir=out_dir,
    )

    print(f"  [E] {case_id}/{candidate_label} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("TIMEOUT", flush=True)
        return None, None
    elapsed = time.time() - t0

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")
        (out_dir / "stdout.txt").write_text(result.stdout or "")

    # Find the actual telemetry file (simulator writes timestamped names)
    actual_tel = _find_telemetry_csv(out_dir)
    actual_sum = _find_run_summary(out_dir)

    if actual_tel is None:
        print(f"FAILED rc={result.returncode} ({elapsed:.0f}s)", flush=True)
        return None, None

    print(f"done ({elapsed:.0f}s) rows={actual_tel.name}", flush=True)
    return actual_tel, actual_sum


def _run_d_baseline(
    case_id: str,
    height_label: str,
    steps: int,
    push_mag: int,
    push_dur: int,
    push_int: int,
) -> tuple[Path | None, Path | None]:
    """Run D current-best baseline (mode-div only, no wheel-yaw)."""
    out_dir = OUT_BASE / f"{case_id}" / "D_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_tel = _find_telemetry_csv(out_dir)
    existing_sum = _find_run_summary(out_dir)
    if existing_tel is not None and existing_sum is not None:
        return existing_tel, existing_sum

    setup_path = (
        ROOT / "outputs" / "physical_target_height_setups" / f"{height_label}_setup.json"
    )
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--vd-sagittal-authority-profile", PROFILE_D,
        "--height-variant-setup", str(setup_path),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", str(steps),
        "--write-run-summary-sidecar",
        "--output-dir", str(out_dir),
        "--enable-mode-hip-yaw-divergence",
        "--mode-hip-yaw-div-kp", str(D_DIV_KP),
        "--mode-hip-yaw-div-kd", str(D_DIV_KD),
        "--mode-hip-yaw-div-max-torque", str(D_DIV_MAX_TORQUE),
        "--mode-hip-yaw-div-soft-limit-rad", str(D_DIV_SOFT_LIMIT_RAD),
        "--mode-hip-yaw-div-soft-gain", str(D_DIV_SOFT_GAIN),
        "--mode-hip-yaw-div-ref-source", D_DIV_REF_SOURCE,
        "--push-enabled",
        "--push-magnitude-n", str(float(push_mag)),
        "--push-duration-steps", str(push_dur),
        "--push-interval-steps", str(push_int),
    ]

    print(f"  [D] {case_id} ...", end=" ", flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("TIMEOUT", flush=True)
        return None, None
    elapsed = time.time() - t0

    if result.returncode != 0:
        (out_dir / "stderr.txt").write_text(result.stderr or "")

    actual_tel = _find_telemetry_csv(out_dir)
    actual_sum = _find_run_summary(out_dir)

    if actual_tel is None:
        print(f"FAILED rc={result.returncode} ({elapsed:.0f}s)", flush=True)
        return None, None

    print(f"done ({elapsed:.0f}s) rows={actual_tel.name}", flush=True)
    return actual_tel, actual_sum



# ---- Sign verification helpers ---- #

def check_wheel_yaw_sign(rows: list[dict]) -> dict:
    """Verify that wheel-yaw torque reduces yaw error.

    Heuristic: look at consecutive steps. If |yaw_error| decreases from t to t+1,
    and wheel_yaw_tau_diff * yaw_error > 0 at time t (torque opposes error),
    count it as a "sign-correct" step.

    Returns:
        dict with correct_count, total_active_count, sign_correct_pct
    """
    correct = 0
    total = 0
    for i in range(len(rows) - 1):
        wy_enabled = str(rows[i].get("wheel_yaw_enabled", "false")).strip().lower() in ("true", "1", "1.0")
        if not wy_enabled:
            continue
        try:
            yaw_err = float(rows[i].get("wheel_yaw_error", 0.0))
            yaw_err_next = float(rows[i + 1].get("wheel_yaw_error", 0.0))
            tau_left = float(rows[i].get("wheel_yaw_tau_left", 0.0))
            tau_right = float(rows[i].get("wheel_yaw_tau_right", 0.0))
        except (ValueError, TypeError):
            continue

        tau_diff = tau_left - tau_right  # positive = CCW moment
        abs_err_decreased = abs(yaw_err_next) < abs(yaw_err)
        torque_opposes_error = (tau_diff * yaw_err) > 0  # tau_diff and yaw_err same sign = opposing

        if abs_err_decreased and torque_opposes_error:
            correct += 1
        total += 1

    return {
        "sign_correct_steps": correct,
        "sign_checked_steps": total,
        "sign_correct_pct": round(100.0 * correct / total, 1) if total > 0 else 0.0,
    }


# ---- Metric extraction ---- #

def extract_metrics(tel_path: Path | None, sum_path: Path | None) -> dict:
    """Extract D4/D5 metrics from telemetry CSV."""
    if tel_path is None or not tel_path.exists():
        return {}

    with open(tel_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    n = len(rows)
    if n == 0:
        return {}

    # Helper: float column
    def fcol(key, default=0.0):
        vals = []
        for r in rows:
            v = r.get(key, "")
            if v in ("", "nan", "None", None):
                vals.append(default)
            else:
                try:
                    vals.append(float(v))
                except (ValueError, TypeError):
                    vals.append(default)
        return vals

    def bcol(key):
        return [str(r.get(key, "false")).strip().lower() in ("true", "1", "1.0") for r in rows]

    def rms(xs):
        return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0

    # Core metrics
    l_hy = fcol("l_hip_yaw_pos")
    r_hy = fcol("r_hip_yaw_pos")
    hy_all = [abs(x) for x in (l_hy + r_hy)]

    pitch = fcol("robot_pitch_x")
    pitch_deg = [math.degrees(x) for x in pitch]
    roll = fcol("robot_roll_y")
    roll_deg = [math.degrees(x) for x in roll]

    yaw_err = fcol("yaw_error_from_equilibrium_rad")
    yaw_drift = fcol("yaw_drift_from_initial_rad")

    support_err = fcol("support_position_error_m")
    support_abs = [abs(x) for x in support_err]

    l_wheel_tau = fcol("l_wheel_tau")
    r_wheel_tau = fcol("r_wheel_tau")
    wheel_tau_all = [abs(x) for x in (l_wheel_tau + r_wheel_tau)]

    l_wheel_vel = fcol("l_wheel_vel")
    r_wheel_vel = fcol("r_wheel_vel")
    wheel_vel_all = [abs(x) for x in (l_wheel_vel + r_wheel_vel)]

    # Mode-div metrics
    md_tau_l = fcol("mode_hip_yaw_div_tau_left")
    md_tau_r = fcol("mode_hip_yaw_div_tau_right")
    md_sat_l = bcol("mode_hip_yaw_div_tau_left_sat")
    md_sat_r = bcol("mode_hip_yaw_div_tau_right_sat")

    # Wheel-yaw metrics
    wy_tau_l = fcol("wheel_yaw_tau_left")
    wy_tau_r = fcol("wheel_yaw_tau_right")
    wy_sat = bcol("wheel_yaw_saturated")

    # Yaw controller hip-yaw contribution
    yaw_ctrl_l = fcol("yaw_controller_tau_hip_yaw_left")
    yaw_ctrl_r = fcol("yaw_controller_tau_hip_yaw_right")
    yaw_ctrl_abs = [abs(x) for x in (yaw_ctrl_l + yaw_ctrl_r)]

    # Ownership
    body_yaw_owner = [r.get("body_yaw_owner", "") for r in rows]
    hy_div_owner = [r.get("hip_yaw_divergence_owner", "") for r in rows]

    # Safety
    wbc_rows = sum(1 for r in rows
                   if str(r.get("per_actuator_wbc_authority_enabled", "false")).strip().lower()
                   in ("true", "1", "1.0"))
    wbc_owner_rows = sum(1 for r in rows
                         if "wbc" in str(r.get("active_torque_owner_per_joint", "")).lower())
    hidden_torque = fcol("hidden_torque_norm")
    ownership_violation = fcol("ownership_violation_count")
    term = any(bcol("terminated"))
    nan_count = sum(
        1 for r in rows
        for k in ("l_hip_yaw_pos", "r_hip_yaw_pos", "robot_pitch_x", "yaw_error_from_equilibrium_rad")
        if r.get(k, "") in ("nan", "inf", "-inf")
    )
    inf_count = sum(
        1 for r in rows
        for k in ("l_wheel_tau", "r_wheel_tau", "l_hip_yaw_tau_shape_final", "r_hip_yaw_tau_shape_final")
        if r.get(k, "") in ("inf", "-inf")
    )

    # Hip-yaw divergence max
    hy_div_err = fcol("hip_yaw_divergence_error_rad")
    hy_div_abs = [abs(x) for x in hy_div_err]

    def out_pct(vals, thr):
        return 100.0 * sum(1 for x in vals if x > thr) / len(vals) if vals else 0.0

    # Recovery time: steps after push where |yaw_error| returns to < 0.05 rad
    # Find push intervals and check recovery
    recovery_time = n  # default: never recovered
    for push_on_col in ("push_active",):
        push_active = bcol(push_on_col) if push_on_col in rows[0] else None
        if push_active:
            # Find last push end
            last_push_end = 0
            for i in range(1, n):
                if push_active[i - 1] and not push_active[i]:
                    last_push_end = i
            if last_push_end > 0:
                for i in range(last_push_end, n):
                    if abs(yaw_err[i]) < 0.05:
                        recovery_time = i - last_push_end
                        break

    # Sign verification
    sign_check = check_wheel_yaw_sign(rows)

    # Ownership violation count
    ownership_violation_max = max(ownership_violation) if ownership_violation else 0.0

    return {
        "actual_rows": n,
        "fell": term,
        "hip_yaw_abs_max": round(max(hy_all), 4) if hy_all else 0.0,
        "hip_yaw_divergence_abs_max": round(max(hy_div_abs), 4) if hy_div_abs else 0.0,
        "hip_yaw_divergence_error_abs_max": round(max(hy_div_abs), 4) if hy_div_abs else 0.0,
        "yaw_error_max_rad": round(max(abs(x) for x in yaw_err), 4) if yaw_err else 0.0,
        "yaw_drift_max_rad": round(max(abs(x) for x in yaw_drift), 4) if yaw_drift else 0.0,
        "support_position_error_max_abs_m": round(max(support_abs), 4) if support_abs else 0.0,
        "support_position_error_p2p_m": round(max(support_err) - min(support_err), 4) if support_err else 0.0,
        "out15_pct": round(out_pct(support_abs, 0.15), 1),
        "out25_pct": round(out_pct(support_abs, 0.25), 1),
        "pitch_max_abs_deg": round(max(abs(x) for x in pitch_deg), 2) if pitch_deg else 0.0,
        "roll_rms_deg": round(rms(roll_deg), 2),
        "wheel_torque_max": round(max(wheel_tau_all), 4) if wheel_tau_all else 0.0,
        "wheel_velocity_max": round(max(wheel_vel_all), 4) if wheel_vel_all else 0.0,
        "wheel_yaw_saturation_rows": sum(wy_sat),
        "mode_div_saturation_rows": sum(1 for i in range(n) if md_sat_l[i] or md_sat_r[i]),
        "wheel_yaw_tau_abs_max": round(max([abs(x) for x in (wy_tau_l + wy_tau_r)]), 4) if (wy_tau_l + wy_tau_r) else 0.0,
        "yaw_controller_tau_hip_yaw_abs_max": round(max(yaw_ctrl_abs), 4) if yaw_ctrl_abs else 0.0,
        "recovery_time_steps": recovery_time,
        "wbc_authority_rows": wbc_rows,
        "wbc_owner_rows": wbc_owner_rows,
        "hidden_torque_max": round(max(hidden_torque), 4) if hidden_torque else 0.0,
        "ownership_violation_max": round(ownership_violation_max, 4),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "sign_correct_steps": sign_check["sign_correct_steps"],
        "sign_checked_steps": sign_check["sign_checked_steps"],
        "sign_correct_pct": sign_check["sign_correct_pct"],
    }


def main() -> int:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    # ---- Run D baselines ---- #
    for case_id, height_label, steps, push_mag, push_dur, push_int in PUSH_CASES:
        print(f"\n=== D baseline: {case_id} ===")
        tel_path, sum_path = _run_d_baseline(
            case_id, height_label, steps, push_mag, push_dur, push_int,
        )
        metrics = extract_metrics(tel_path, sum_path)
        row = {
            "case_id": case_id,
            "height_label": height_label,
            "candidate_kind": "mode_hip_yaw_div_v1",
            "candidate_params": "kp=5.0,kd=0.20,mt=2.0",
            "wheel_yaw_kp": 0.0,
            "wheel_yaw_kd": 0.0,
            "wheel_yaw_max_torque": 0.0,
            "wheel_yaw_lowpass_alpha": 0.0,
            "wheel_yaw_enabled": False,
            "mode_hip_yaw_div_enabled": True,
            "validation_source": "real_simulation",
            "requested_steps": steps,
        }
        row.update(metrics)
        all_rows.append(row)
        print(f"  D baseline: hy_max={row.get('hip_yaw_abs_max', 'N/A')}")

    # ---- Run E candidates ---- #
    for case_id, height_label, steps, push_mag, push_dur, push_int in PUSH_CASES:
        for wy_kp, wy_kd, wy_max_torque, wy_lp in WHEEL_YAW_GRID:
            print(f"\n=== E candidate: {case_id} kp={wy_kp} kd={wy_kd} mt={wy_max_torque} lp={wy_lp} ===")
            tel_path, sum_path = _run_e_candidate(
                case_id, height_label, steps, push_mag, push_dur, push_int,
                wy_kp, wy_kd, wy_max_torque, wy_lp,
            )
            metrics = extract_metrics(tel_path, sum_path) if tel_path else {}
            candidate_label = f"E_kp{wy_kp}_kd{wy_kd}_mt{wy_max_torque}_lp{wy_lp}"
            row = {
                "case_id": case_id,
                "height_label": height_label,
                "candidate_kind": "mode_hip_yaw_div_wheel_yaw_v1",
                "candidate_params": f"kp={wy_kp},kd={wy_kd},mt={wy_max_torque},lp={wy_lp}",
                "wheel_yaw_kp": wy_kp,
                "wheel_yaw_kd": wy_kd,
                "wheel_yaw_max_torque": wy_max_torque,
                "wheel_yaw_lowpass_alpha": wy_lp,
                "wheel_yaw_enabled": True,
                "mode_hip_yaw_div_enabled": True,
                "validation_source": "real_simulation",
                "requested_steps": steps,
                "telemetry_path": str(tel_path) if tel_path else "",
            }
            row.update(metrics)
            all_rows.append(row)

            if metrics:
                hy = metrics.get("hip_yaw_abs_max", 0.0)
                print(f"  hy_max={hy:.4f}  signed?={metrics.get('sign_correct_pct', 0.0):.1f}%")
            else:
                print(f"  FAILED — no metrics")

    # ---- Write metrics CSV ---- #
    if all_rows:
        csv_path = OUT_BASE / "sweep_metrics.csv"
        fieldnames = sorted({k for r in all_rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in all_rows:
                writer.writerow({k: r.get(k, "") for k in fieldnames})
        print(f"\nMetrics CSV: {csv_path}", flush=True)
    else:
        print("\nNo rows written!", flush=True)
        return 1

    # ---- Summary table ---- #
    print("\n" + "=" * 100)
    print(f"{'Case':20s} {'Candidate':45s} {'hy_max':>8s} {'pitch_max':>10s} {'support_abs':>12s} {'sign_ok':>8s} {'fell':>6s}")
    print("=" * 100)
    for r in all_rows:
        case = r.get("case_id", "")[:18]
        cand = r.get("candidate_params", "")[:43]
        hy = r.get("hip_yaw_abs_max", 0.0) or 0.0
        pitch = r.get("pitch_max_abs_deg", 0.0) or 0.0
        support = r.get("support_position_error_max_abs_m", 0.0) or 0.0
        sign = r.get("sign_correct_pct", 0.0) or 0.0
        fell = "FALL" if r.get("fell", False) else "OK"
        print(f"{case:20s} {cand:45s} {hy:>8.4f} {pitch:>10.2f} {support:>12.4f} {sign:>7.1f}% {fell:>6s}")

    # ---- Sign verification output ---- #
    sign_csv_path = OUT_BASE / "sign_verification.csv"
    sign_rows = [
        {
            "case_id": r.get("case_id", ""),
            "candidate": r.get("candidate_params", ""),
            "sign_correct_pct": r.get("sign_correct_pct", 0.0),
            "sign_checked_steps": r.get("sign_checked_steps", 0),
            "hip_yaw_abs_max": r.get("hip_yaw_abs_max", 0.0),
            "wheel_yaw_enabled": r.get("wheel_yaw_enabled", False),
        }
        for r in all_rows
    ]
    with sign_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sign_rows[0].keys() if sign_rows else [])
        writer.writeheader()
        for sr in sign_rows:
            writer.writerow(sr)
    print(f"\nSign verification: {sign_csv_path}")

    # ---- Selection recommendation ---- #
    print("\n--- Candidate Selection ---")
    # Find best E candidates per case
    for case_id in [c[0] for c in PUSH_CASES]:
        print(f"\n{case_id}:")
        d_rows = [r for r in all_rows if r.get("case_id") == case_id
                  and not r.get("wheel_yaw_enabled", False)]
        e_rows = [r for r in all_rows if r.get("case_id") == case_id
                  and r.get("wheel_yaw_enabled", False)]
        d_hy = d_rows[0].get("hip_yaw_abs_max", 0.0) if d_rows else 0.0
        print(f"  D baseline: hy_abs_max={d_hy:.4f}")

        # Rank by hy_abs_max (lower is better), then by sign correctness
        valid_e = [r for r in e_rows
                   if not r.get("fell", False)
                   and r.get("hip_yaw_abs_max", 0.0) is not None
                   and float(r.get("hip_yaw_abs_max", 0.0) or 0.0) < 0.50
                   and r.get("sign_correct_pct", 0.0) >= 50.0]
        valid_e.sort(key=lambda r: (float(r.get("hip_yaw_abs_max", 0.0) or 0.0), -float(r.get("sign_correct_pct", 0.0) or 0.0)))
        for i, r in enumerate(valid_e[:5]):
            hy = r.get("hip_yaw_abs_max", 0.0) or 0.0
            pitch = r.get("pitch_max_abs_deg", 0.0) or 0.0
            sign = r.get("sign_correct_pct", 0.0) or 0.0
            params = r.get("candidate_params", "")
            print(f"  [{i+1}] {params:45s} hy={hy:.4f} pitch={pitch:.2f} sign={sign:.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
