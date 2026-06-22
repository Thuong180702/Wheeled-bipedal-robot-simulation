"""Full Step C validation for the PFF low-band support v1 opt-in profile.

This runner intentionally leaves controller defaults untouched.  It runs the
project's segment-based Step C random/changing-height suite for three profiles:

  A: calibrated_support_position_outer_loop_pitch_ref_v2
  B: physics_equilibrium_feedforward_outer_loop
  C: physics_equilibrium_feedforward_outer_loop_low_band_support_v1

It also runs the 10 fixed-height validation at 2000 steps by default.  Outputs
are written under:

  outputs/physics_ff_step_c_low_band_support_v1_full_step_c/
  docs/validation/physics_ff_step_c_low_band_support_v1_full_step_c_report.md
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups_centered"
OUT_BASE = ROOT / "outputs" / "physics_ff_step_c_low_band_support_v1_full_step_c"
REPORT_PATH = ROOT / "docs" / "validation" / "physics_ff_step_c_low_band_support_v1_full_step_c_report.md"

PROFILES = [
    ("A_B2V2", "calibrated_support_position_outer_loop_pitch_ref_v2", "Baseline B2v2"),
    ("B_CURRENT_PFF", "physics_equilibrium_feedforward_outer_loop", "Current PFF"),
    (
        "C_LOW_BAND_V1",
        "physics_equilibrium_feedforward_outer_loop_low_band_support_v1",
        "Low-band support v1",
    ),
]

HEIGHTS = [
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

PROTECTED_HEIGHTS = {"low_0p320", "low_0p330", "low_0p360", "high_0p480"}
MAXABS_TOL_M = 0.02
P2P_FACTOR = 1.15
OUT15_TOL_PP = 3.0
HIGH_FOCUSED_MAXABS_TOL_M = 0.005
VECTOR_TOL = 1e-6


# This mirrors the existing Step C random-height artifact sequence.  The
# simulator is currently initialized per fixed-height dwell segment; it does not
# consume an in-run height sequence key.
STEP_C_CASES: dict[str, list[tuple[str, int]]] = {
    "C1_slow_ladder_up_down": [
        ("low_0p300", 300),
        ("low_0p320", 300),
        ("low_0p330", 300),
        ("low_0p340", 300),
        ("low_0p360", 300),
        ("low_0p380", 300),
        ("high_0p430", 300),
        ("high_0p450", 300),
        ("high_0p465", 300),
        ("high_0p480", 300),
        ("high_0p480", 300),
        ("high_0p465", 300),
        ("high_0p450", 300),
        ("high_0p430", 300),
        ("low_0p380", 300),
        ("low_0p360", 300),
        ("low_0p340", 300),
        ("low_0p330", 300),
        ("low_0p320", 300),
        ("low_0p300", 300),
    ],
    "C2_random_500dwell": [
        ("low_0p320", 406),
        ("low_0p380", 462),
        ("low_0p340", 435),
        ("low_0p320", 573),
        ("high_0p480", 422),
        ("high_0p430", 408),
        ("low_0p300", 423),
        ("low_0p360", 459),
        ("high_0p480", 554),
        ("low_0p300", 543),
    ],
    "C3_random_200dwell": [
        ("low_0p340", 241),
        ("high_0p480", 203),
        ("low_0p340", 207),
        ("low_0p380", 150),
        ("low_0p330", 239),
        ("high_0p450", 193),
        ("low_0p360", 169),
        ("low_0p340", 247),
        ("high_0p430", 163),
        ("low_0p320", 198),
        ("low_0p330", 195),
        ("high_0p430", 227),
        ("low_0p360", 155),
        ("high_0p465", 218),
        ("low_0p320", 198),
    ],
    "C4_abrupt_stress": [
        ("high_0p480", 300),
        ("low_0p300", 300),
        ("high_0p480", 300),
        ("low_0p330", 300),
        ("high_0p465", 300),
    ],
    "C5_long_random": [
        ("low_0p320", 382),
        ("low_0p380", 285),
        ("low_0p340", 135),
        ("low_0p300", 216),
        ("low_0p380", 140),
        ("low_0p340", 151),
        ("high_0p450", 242),
        ("high_0p465", 286),
        ("low_0p330", 289),
        ("high_0p430", 207),
        ("low_0p360", 136),
        ("low_0p330", 373),
        ("low_0p360", 183),
        ("high_0p465", 294),
        ("low_0p360", 385),
        ("low_0p340", 266),
        ("low_0p300", 217),
        ("low_0p320", 261),
        ("high_0p450", 237),
        ("low_0p320", 208),
    ],
    "focused_low_0p320": [("low_0p320", 300)],
    "focused_high_0p480": [("high_0p480", 300)],
}


@dataclass(frozen=True)
class RunSpec:
    suite: str
    case_name: str
    seg_idx: int
    height: str
    steps: int
    tag: str
    profile: str
    profile_label: str


def _run_dir(spec: RunSpec) -> Path:
    if spec.suite == "step_c":
        return OUT_BASE / "step_c" / spec.case_name / spec.tag / f"seg{spec.seg_idx:03d}_{spec.height}_{spec.steps}"
    return OUT_BASE / "fixed_height" / spec.tag / f"seg{spec.seg_idx:03d}_{spec.height}_{spec.steps}"


def _truthy(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"true", "1", "1.0", "yes", "y"}


def _vector_any_true(value: Any) -> bool:
    text = str(value).strip().lower()
    if not text:
        return False
    return any(token.strip() in {"true", "1", "1.0"} for token in text.replace("[", "").replace("]", "").split(","))


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_vector(value: Any) -> list[float]:
    if value is None:
        return []
    text = str(value).strip().strip('"').strip("'").replace("[", "").replace("]", "")
    if not text:
        return []
    out: list[float] = []
    for token in text.split(","):
        parsed = _parse_float(token)
        if parsed is not None:
            out.append(parsed)
    return out


def _vec_norm(value: Any) -> float:
    vals = _parse_vector(value)
    return math.sqrt(sum(v * v for v in vals)) if vals else 0.0


def _col(rows: list[dict[str, str]], key: str) -> list[float]:
    vals: list[float] = []
    for row in rows:
        parsed = _parse_float(row.get(key))
        if parsed is not None and math.isfinite(parsed):
            vals.append(parsed)
    return vals


def _first_existing(fieldnames: set[str], candidates: list[str]) -> str | None:
    for name in candidates:
        if name in fieldnames:
            return name
    return None


def _stats(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "max_abs": 0.0, "p2p": 0.0}
    mn = min(vals)
    mx = max(vals)
    return {
        "min": mn,
        "max": mx,
        "mean": sum(vals) / len(vals),
        "max_abs": max(abs(mn), abs(mx)),
        "p2p": mx - mn,
    }


def _max_abs(vals: list[float]) -> float:
    return max((abs(v) for v in vals), default=0.0)


def _max_abs_deg(rows: list[dict[str, str]], fieldnames: set[str], candidates: list[str]) -> tuple[float, str]:
    name = _first_existing(fieldnames, candidates)
    if name is None:
        return 0.0, ""
    vals = _col(rows, name)
    if name.endswith("_deg"):
        return _max_abs(vals), name
    return _max_abs([math.degrees(v) for v in vals]), name


def _resolve_support_error(rows: list[dict[str, str]], fieldnames: set[str]) -> tuple[list[float], str]:
    name = _first_existing(
        fieldnames,
        [
            "active_pitch_crossing_signed_error_m",
            "support_position_error_m",
            "sagittal_position_error_m",
            "com_position_error_sagittal_m",
        ],
    )
    if name is None:
        return [], ""
    return _col(rows, name), name


def _resolve_hip_yaw_abs(rows: list[dict[str, str]], fieldnames: set[str]) -> tuple[float, str]:
    # Corrected policy: prefer the tracking/gate metric when it exists.
    for name in ("hip_yaw_abs_max_tracking", "hip_yaw_abs_max"):
        if name in fieldnames:
            vals = _col(rows, name)
            if vals:
                return _max_abs(vals), name
    error_cols = [name for name in ("l_hip_yaw_error", "r_hip_yaw_error", "l_hip_yaw_error_rad", "r_hip_yaw_error_rad") if name in fieldnames]
    vals: list[float] = []
    for name in error_cols:
        vals.extend(_col(rows, name))
    if vals:
        return _max_abs(vals), "+".join(error_cols)
    pos_cols = [name for name in ("l_hip_yaw_pos", "r_hip_yaw_pos") if name in fieldnames]
    vals = []
    for name in pos_cols:
        vals.extend(_col(rows, name))
    return _max_abs(vals), "+".join(pos_cols)


def _resolve_hip_yaw_divergence(rows: list[dict[str, str]], fieldnames: set[str]) -> tuple[float, str]:
    for name in ("hip_yaw_divergence_error", "hip_yaw_divergence", "hip_yaw_asymmetry"):
        if name in fieldnames:
            vals = _col(rows, name)
            if vals:
                return _max_abs(vals), name
    if "l_hip_yaw_pos" in fieldnames and "r_hip_yaw_pos" in fieldnames:
        left = _col(rows, "l_hip_yaw_pos")
        right = _col(rows, "r_hip_yaw_pos")
        vals = [left[i] - right[i] for i in range(min(len(left), len(right)))]
        return _max_abs(vals), "l_hip_yaw_pos-r_hip_yaw_pos"
    return 0.0, ""


def run_sim(spec: RunSpec, *, force: bool, timeout_s: int) -> dict[str, Any]:
    out_dir = _run_dir(spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    tel_dst = out_dir / f"telemetry_{spec.steps}.csv"
    summary_dst = out_dir / f"telemetry_{spec.steps}.summary.json"

    if tel_dst.exists() and tel_dst.stat().st_size > 1000 and not force:
        return {
            "spec": spec,
            "telemetry_path": str(tel_dst),
            "summary_path": str(summary_dst) if summary_dst.exists() else "",
            "returncode": 0,
            "cached": True,
            "error": "",
        }

    setup_path = SETUP_DIR / f"{spec.height}_setup.json"
    if not setup_path.exists():
        return {
            "spec": spec,
            "telemetry_path": "",
            "summary_path": "",
            "returncode": None,
            "cached": False,
            "error": f"missing setup {setup_path}",
        }

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "simulate_hierarchical_controller.py"),
        "--controller-mode",
        "balance-core",
        "--sagittal-controller",
        "velocity-damped",
        "--vd-sagittal-authority-profile",
        spec.profile,
        "--height-variant-setup",
        str(setup_path),
        "--steps",
        str(spec.steps),
        "--telemetry-decimation",
        "1",
        "--failure-window-steps",
        str(spec.steps),
        "--write-run-summary-sidecar",
        "--output-dir",
        str(out_dir),
    ]
    (out_dir / "command.json").write_text(json.dumps(cmd, indent=2), encoding="utf-8")

    try:
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout_s)
        returncode: int | None = result.returncode
        (out_dir / "stdout.txt").write_text(result.stdout or "", encoding="utf-8", errors="replace")
        (out_dir / "stderr.txt").write_text(result.stderr or "", encoding="utf-8", errors="replace")
        error = "" if result.returncode == 0 else f"returncode {result.returncode}"
    except subprocess.TimeoutExpired as exc:
        returncode = None
        (out_dir / "stdout.txt").write_text(exc.stdout or "", encoding="utf-8", errors="replace")
        (out_dir / "stderr.txt").write_text((exc.stderr or "") + "\nTIMEOUT\n", encoding="utf-8", errors="replace")
        error = "timeout"

    telemetry_candidates = sorted(out_dir.glob("telemetry_*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if telemetry_candidates:
        newest = telemetry_candidates[0]
        if newest.resolve() != tel_dst.resolve():
            shutil.copy2(newest, tel_dst)
    sidecar_candidates = sorted(out_dir.glob("telemetry_*.summary.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    if sidecar_candidates:
        newest_sidecar = sidecar_candidates[0]
        if newest_sidecar.resolve() != summary_dst.resolve():
            shutil.copy2(newest_sidecar, summary_dst)

    return {
        "spec": spec,
        "telemetry_path": str(tel_dst) if tel_dst.exists() else "",
        "summary_path": str(summary_dst) if summary_dst.exists() else "",
        "returncode": returncode,
        "cached": False,
        "error": error,
    }


def analyze_telemetry(run_result: dict[str, Any]) -> dict[str, Any]:
    spec: RunSpec = run_result["spec"]
    path_text = run_result.get("telemetry_path") or ""
    path = Path(path_text) if path_text else None
    base: dict[str, Any] = {
        "suite": spec.suite,
        "case_name": spec.case_name,
        "seq_name": spec.case_name,
        "seg_idx": spec.seg_idx,
        "height": spec.height,
        "steps_nominal": spec.steps,
        "tag": spec.tag,
        "profile": spec.profile,
        "profile_label": spec.profile_label,
        "telemetry_path": path_text,
        "summary_path": run_result.get("summary_path", ""),
        "returncode": run_result.get("returncode"),
        "cached": bool(run_result.get("cached", False)),
        "run_error": run_result.get("error", ""),
    }

    if path is None or not path.exists():
        base.update(
            {
                "steps": 0,
                "any_fell": True,
                "any_unsafe": True,
                "unsafe_reasons": "missing_telemetry",
                "support_error_column": "",
                "support_position_error_max_abs_m": 0.0,
                "support_position_error_p2p_m": 0.0,
                "max_trans_m": 0.0,
                "out15_pct": 0.0,
                "pitch_max_abs_deg": 0.0,
                "roll_max_abs_deg": 0.0,
                "hip_yaw_abs_max": 0.0,
                "hip_yaw_abs_source": "",
                "hip_yaw_divergence_error": 0.0,
                "hip_yaw_divergence_source": "",
                "hidden_torque_max": 0.0,
                "ownership_violation_max": 0.0,
                "wbc_applied_rows": 0,
            }
        )
        return base

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])

    support_vals, support_source = _resolve_support_error(rows, fieldnames)
    support_stats = _stats(support_vals)
    trans_vals = support_vals[: min(30, len(support_vals))]
    out15_pct = 100.0 * sum(1 for value in support_vals if abs(value) > 0.15) / max(1, len(support_vals))

    pitch_max_abs_deg, pitch_source = _max_abs_deg(
        rows, fieldnames, ["pitch_x_rad", "robot_pitch_x", "pitch_x", "euler_pitch_y"]
    )
    roll_max_abs_deg, roll_source = _max_abs_deg(
        rows, fieldnames, ["roll_y_rad", "robot_roll_y", "roll_y", "euler_roll_x"]
    )
    hip_yaw_abs_max, hip_yaw_source = _resolve_hip_yaw_abs(rows, fieldnames)
    hip_yaw_divergence_error, hip_yaw_divergence_source = _resolve_hip_yaw_divergence(rows, fieldnames)

    hidden_torque_max = max(_col(rows, "hidden_torque_norm"), default=0.0) if "hidden_torque_norm" in fieldnames else 0.0
    ownership_violation_max = max(_col(rows, "ownership_violation_count"), default=0.0) if "ownership_violation_count" in fieldnames else 0.0
    terminated_rows = [row for row in rows if _truthy(row.get("terminated", ""))]
    termination_reason = ""
    if terminated_rows:
        termination_reason = terminated_rows[0].get("termination_reason", "") or ""
    fell_short = len(rows) < max(1, spec.steps - 1)
    any_fell = bool(terminated_rows) or fell_short or bool(run_result.get("error"))

    owner_rows = sum(1 for row in rows if "wbc" in str(row.get("active_torque_owner_per_joint", "")).lower())
    authority_rows = sum(1 for row in rows if _vector_any_true(row.get("per_actuator_wbc_authority_enabled", "")))
    after_clip_rows = sum(1 for row in rows if _vec_norm(row.get("tau_wbc_after_authority_clip", "")) > VECTOR_TOL)
    correction_rows = sum(1 for row in rows if _vec_norm(row.get("tau_wbc_correction", "")) > VECTOR_TOL)
    wbc_applied_rows = max(owner_rows, authority_rows, after_clip_rows)

    com_z_min = min(_col(rows, "com_z"), default=min(_col(rows, "com_z_m"), default=1.0))
    contact_invalid_after_startup = 0
    if "contact_force_valid" in fieldnames:
        for idx, row in enumerate(rows):
            if idx > 2 and not _truthy(row.get("contact_force_valid", "true")):
                contact_invalid_after_startup += 1
    if "left_wheel_contact" in fieldnames and "right_wheel_contact" in fieldnames:
        for idx, row in enumerate(rows):
            if idx > 2 and (not _truthy(row.get("left_wheel_contact", "true")) or not _truthy(row.get("right_wheel_contact", "true"))):
                contact_invalid_after_startup += 1

    low_band_scale = _col(rows, "support_outer_loop_height_scale")
    effective_kp = _col(rows, "support_outer_loop_kp_effective")
    pitch_trim = _col(rows, "support_outer_loop_pitch_ref_offset_deg")
    low_band_scale_stats = _stats(low_band_scale)
    effective_kp_stats = _stats(effective_kp)
    pitch_trim_stats = _stats(pitch_trim)
    low_band_active_rows = sum(1 for value in low_band_scale if abs(value) > 1e-6)

    unsafe_reasons: list[str] = []
    if any_fell:
        unsafe_reasons.append("fall_or_short_run")
    if hip_yaw_abs_max > 0.35:
        unsafe_reasons.append("hip_yaw_hard_fail")
    if pitch_max_abs_deg > 16.0:
        unsafe_reasons.append("pitch_hard_fail")
    if roll_max_abs_deg > 10.0:
        unsafe_reasons.append("roll_hard_fail")
    if com_z_min < 0.20:
        unsafe_reasons.append("com_z_floor")
    if hidden_torque_max > VECTOR_TOL:
        unsafe_reasons.append("hidden_torque")
    if ownership_violation_max > 0:
        unsafe_reasons.append("ownership_violation")
    if wbc_applied_rows > 0:
        unsafe_reasons.append("wbc_applied")

    base.update(
        {
            "steps": len(rows),
            "any_fell": any_fell,
            "fell_short": fell_short,
            "terminated_rows": len(terminated_rows),
            "termination_reason": termination_reason,
            "any_unsafe": bool(unsafe_reasons),
            "unsafe_reasons": ";".join(unsafe_reasons),
            "support_error_column": support_source,
            "support_position_error_min_m": support_stats["min"],
            "support_position_error_max_m": support_stats["max"],
            "support_position_error_mean_m": support_stats["mean"],
            "support_position_error_max_abs_m": support_stats["max_abs"],
            "support_position_error_p2p_m": support_stats["p2p"],
            "max_trans_m": _max_abs(trans_vals),
            "out15_pct": out15_pct,
            "pitch_max_abs_deg": pitch_max_abs_deg,
            "pitch_source": pitch_source,
            "roll_max_abs_deg": roll_max_abs_deg,
            "roll_source": roll_source,
            "com_z_min_m": com_z_min,
            "hip_yaw_abs_max": hip_yaw_abs_max,
            "hip_yaw_abs_source": hip_yaw_source,
            "hip_yaw_divergence_error": hip_yaw_divergence_error,
            "hip_yaw_divergence_source": hip_yaw_divergence_source,
            "hidden_torque_max": hidden_torque_max,
            "ownership_violation_max": ownership_violation_max,
            "wbc_owner_rows": owner_rows,
            "wbc_authority_rows": authority_rows,
            "wbc_after_authority_clip_rows": after_clip_rows,
            "wbc_correction_rows_diagnostic": correction_rows,
            "wbc_applied_rows": wbc_applied_rows,
            "contact_invalid_rows_after_startup": contact_invalid_after_startup,
            "low_band_scale_min": low_band_scale_stats["min"],
            "low_band_scale_mean": low_band_scale_stats["mean"],
            "low_band_scale_max": low_band_scale_stats["max"],
            "low_band_active_rows": low_band_active_rows,
            "low_band_active_pct": 100.0 * low_band_active_rows / max(1, len(low_band_scale)),
            "effective_support_kp_min": effective_kp_stats["min"],
            "effective_support_kp_mean": effective_kp_stats["mean"],
            "effective_support_kp_max": effective_kp_stats["max"],
            "pitch_trim_deg_mean": pitch_trim_stats["mean"],
            "pitch_trim_deg_max_abs": pitch_trim_stats["max_abs"],
        }
    )
    return base


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], *, suite: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["suite"] != suite:
            continue
        key = (row["case_name"], row["tag"]) if suite == "step_c" else ("fixed_height_10", row["tag"])
        groups.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (case_name, tag), group in sorted(groups.items()):
        profile = group[0]["profile"]
        profile_label = group[0]["profile_label"]
        out.append(
            {
                "suite": suite,
                "case_name": case_name,
                "tag": tag,
                "profile": profile,
                "profile_label": profile_label,
                "n_segments": len(group),
                "any_fell": any(bool(row["any_fell"]) for row in group),
                "any_unsafe": any(bool(row["any_unsafe"]) for row in group),
                "unsafe_reasons": ";".join(sorted({str(row.get("unsafe_reasons", "")) for row in group if row.get("unsafe_reasons")})),
                "max_maxabs": max(float(row["support_position_error_max_abs_m"]) for row in group),
                "max_trans": max(float(row["max_trans_m"]) for row in group),
                "support_position_error_max_abs_m": max(float(row["support_position_error_max_abs_m"]) for row in group),
                "support_position_error_p2p_m": max(float(row["support_position_error_p2p_m"]) for row in group),
                "out15_pct": max(float(row["out15_pct"]) for row in group),
                "pitch_max_abs_deg": max(float(row["pitch_max_abs_deg"]) for row in group),
                "roll_max_abs_deg": max(float(row["roll_max_abs_deg"]) for row in group),
                "hip_yaw_abs_max": max(float(row["hip_yaw_abs_max"]) for row in group),
                "hip_yaw_divergence_error": max(float(row["hip_yaw_divergence_error"]) for row in group),
                "hidden_torque_max": max(float(row["hidden_torque_max"]) for row in group),
                "ownership_violation_max": max(float(row["ownership_violation_max"]) for row in group),
                "wbc_applied_rows": sum(int(row["wbc_applied_rows"]) for row in group),
                "wbc_owner_rows": sum(int(row["wbc_owner_rows"]) for row in group),
                "wbc_authority_rows": sum(int(row["wbc_authority_rows"]) for row in group),
                "low_band_scale_mean": sum(float(row["low_band_scale_mean"]) for row in group) / len(group),
                "low_band_scale_max": max(float(row["low_band_scale_max"]) for row in group),
                "low_band_active_rows": sum(int(row["low_band_active_rows"]) for row in group),
                "effective_support_kp_mean": sum(float(row["effective_support_kp_mean"]) for row in group) / len(group),
                "effective_support_kp_max": max(float(row["effective_support_kp_max"]) for row in group),
                "pitch_trim_deg_max_abs": max(float(row["pitch_trim_deg_max_abs"]) for row in group),
            }
        )
    return out


def _by_key(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def _within_baseline(candidate: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(candidate["support_position_error_max_abs_m"]) > float(baseline["support_position_error_max_abs_m"]) + MAXABS_TOL_M:
        reasons.append("maxabs")
    if float(candidate["support_position_error_p2p_m"]) > float(baseline["support_position_error_p2p_m"]) * P2P_FACTOR:
        reasons.append("p2p")
    if float(candidate["out15_pct"]) > float(baseline["out15_pct"]) + OUT15_TOL_PP:
        reasons.append("out15")
    return not reasons, reasons


def _improves_or_matches(candidate: dict[str, Any], current_pff: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(candidate["support_position_error_max_abs_m"]) > float(current_pff["support_position_error_max_abs_m"]) + 1e-9:
        reasons.append("maxabs")
    if float(candidate["support_position_error_p2p_m"]) > float(current_pff["support_position_error_p2p_m"]) + 1e-9:
        reasons.append("p2p")
    return not reasons, reasons


def evaluate_decision(segment_rows: list[dict[str, Any]], step_c_summary: list[dict[str, Any]], fixed_summary: list[dict[str, Any]]) -> dict[str, Any]:
    seg = _by_key(segment_rows, "suite", "case_name", "seg_idx", "tag")
    summaries = _by_key(step_c_summary + fixed_summary, "suite", "case_name", "tag")
    failures: list[str] = []
    monitors: list[str] = []
    inconclusive: list[str] = []

    required_tags = {"A_B2V2", "B_CURRENT_PFF", "C_LOW_BAND_V1"}
    for row in segment_rows:
        if row["tag"] in required_tags and row.get("telemetry_path") == "":
            inconclusive.append(f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['tag']}:missing_telemetry")

    candidate_segments = [row for row in segment_rows if row["tag"] == "C_LOW_BAND_V1"]
    for row in candidate_segments:
        if bool(row["any_fell"]):
            failures.append(f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['height']}:fall")
        if bool(row["any_unsafe"]):
            failures.append(f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['height']}:{row['unsafe_reasons']}")

    for case_name in ["C1_slow_ladder_up_down", "C2_random_500dwell", "C3_random_200dwell", "C4_abrupt_stress", "C5_long_random"]:
        cand = summaries.get(("step_c", case_name, "C_LOW_BAND_V1"))
        if not cand:
            inconclusive.append(f"{case_name}:candidate_summary_missing")
        elif cand["any_fell"] or cand["any_unsafe"]:
            failures.append(f"{case_name}:candidate_not_ok:{cand.get('unsafe_reasons', '')}")

    focused_low = {
        tag: seg.get(("step_c", "focused_low_0p320", 0, tag))
        for tag in required_tags
    }
    focused_high = {
        tag: seg.get(("step_c", "focused_high_0p480", 0, tag))
        for tag in required_tags
    }
    if not all(focused_low.values()):
        inconclusive.append("focused_low_0p320:missing_profile")
    else:
        ok, reasons = _within_baseline(focused_low["C_LOW_BAND_V1"], focused_low["A_B2V2"])
        if not ok:
            failures.append(f"focused_low_0p320:outside_B2v2_tolerance:{','.join(reasons)}")
        ok, reasons = _improves_or_matches(focused_low["C_LOW_BAND_V1"], focused_low["B_CURRENT_PFF"])
        if not ok:
            failures.append(f"focused_low_0p320:not_improved_vs_current_PFF:{','.join(reasons)}")

    if not all(focused_high.values()):
        inconclusive.append("focused_high_0p480:missing_profile")
    else:
        cand = focused_high["C_LOW_BAND_V1"]
        pff = focused_high["B_CURRENT_PFF"]
        if float(cand["support_position_error_max_abs_m"]) > float(pff["support_position_error_max_abs_m"]) + HIGH_FOCUSED_MAXABS_TOL_M:
            failures.append("focused_high_0p480:maxabs_regressed_vs_current_PFF")
        if float(cand["support_position_error_p2p_m"]) > float(pff["support_position_error_p2p_m"]) * P2P_FACTOR:
            failures.append("focused_high_0p480:p2p_regressed_vs_current_PFF")

    fixed_cand = summaries.get(("fixed", "fixed_height_10", "C_LOW_BAND_V1"))
    if not fixed_cand:
        inconclusive.append("fixed_height_10:candidate_summary_missing")
        fixed_classification = "INCONCLUSIVE"
    elif fixed_cand["any_fell"] or fixed_cand["any_unsafe"]:
        failures.append(f"fixed_height_10:candidate_not_safe:{fixed_cand.get('unsafe_reasons', '')}")
        fixed_classification = "FAIL"
    else:
        fixed_classification = "PASS"

    # Protected regression checks on both fixed-height 2000-step rows and Step C
    # rows, compared against both B2v2 and current PFF for the same segment.
    protected_regressions: list[str] = []
    for row in candidate_segments:
        if row["height"] not in PROTECTED_HEIGHTS:
            continue
        key_a = (row["suite"], row["case_name"], row["seg_idx"], "A_B2V2")
        key_b = (row["suite"], row["case_name"], row["seg_idx"], "B_CURRENT_PFF")
        baseline = seg.get(key_a)
        current = seg.get(key_b)
        if baseline:
            ok, reasons = _within_baseline(row, baseline)
            if not ok:
                protected_regressions.append(
                    f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['height']}:vs_B2v2:{','.join(reasons)}"
                )
        if current:
            ok, reasons = _within_baseline(row, current)
            if not ok:
                protected_regressions.append(
                    f"{row['suite']}:{row['case_name']}:{row['seg_idx']}:{row['height']}:vs_current_PFF:{','.join(reasons)}"
                )
    if protected_regressions:
        failures.extend(protected_regressions)
        if fixed_classification == "PASS":
            fixed_classification = "FAIL"

    # Non-protected aggregate regressions are monitoring items, not hard fail,
    # provided all hard safety gates above passed.
    for row in step_c_summary:
        if row["tag"] != "C_LOW_BAND_V1":
            continue
        for ref_tag, ref_name in (("A_B2V2", "B2v2"), ("B_CURRENT_PFF", "current_PFF")):
            ref = summaries.get(("step_c", row["case_name"], ref_tag))
            if not ref:
                continue
            ok, reasons = _within_baseline(row, ref)
            if not ok:
                monitors.append(f"{row['case_name']}:candidate_vs_{ref_name}:{','.join(reasons)}")

    if inconclusive:
        classification = "PHYSICS_FF_LOW_BAND_V1_STEP_C_INCONCLUSIVE"
    elif failures:
        classification = "PHYSICS_FF_LOW_BAND_V1_STEP_C_FAIL"
    elif monitors:
        classification = "PHYSICS_FF_LOW_BAND_V1_STEP_C_PASS_WITH_MONITORING"
    else:
        classification = "PHYSICS_FF_LOW_BAND_V1_STEP_C_PASS"

    return {
        "classification": classification,
        "fixed_height_classification": fixed_classification,
        "failures": failures,
        "monitors": monitors,
        "inconclusive": inconclusive,
        "focused_low_0p320": focused_low,
        "focused_high_0p480": focused_high,
        "protected_regressions": protected_regressions,
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def render_report(
    *,
    segment_rows: list[dict[str, Any]],
    step_c_summary: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
    fixed_summary: list[dict[str, Any]],
    decision: dict[str, Any],
    elapsed_s: float,
    fixed_steps: int,
) -> str:
    summaries = _by_key(step_c_summary + fixed_summary, "suite", "case_name", "tag")
    seg = _by_key(segment_rows, "suite", "case_name", "seg_idx", "tag")
    classification = decision["classification"]

    lines: list[str] = [
        "# Physics FF Low-Band Support v1 Full Step C Report",
        "",
        "Date: 2026-06-21",
        "",
        f"Classification: `{classification}`",
        "",
        "## Scope",
        "",
        "This validation compares three opt-in/selected profiles without changing defaults:",
        "",
        "- A Baseline: `calibrated_support_position_outer_loop_pitch_ref_v2`",
        "- B Current PFF: `physics_equilibrium_feedforward_outer_loop`",
        "- C Candidate: `physics_equilibrium_feedforward_outer_loop_low_band_support_v1`",
        "",
        "The suite uses `outputs/physical_target_height_setups_centered` (`centered_posture_height_schedule`).",
        "The project simulator currently validates the random/changing-height cases as fixed-height dwell segments, matching the existing Step C random-height artifacts.",
        "",
        "Corrected hip-yaw policy was used: `hip_yaw_abs_max_tracking` is preferred, then `hip_yaw_abs_max`, then per-joint hip-yaw error/position fallbacks.",
        "`tau_wbc_norm` is treated as diagnostic only; WBC applied rows come from ownership, per-actuator authority, or nonzero post-authority WBC torque rows.",
        "",
        "## Artifacts",
        "",
        f"- Output directory: `{OUT_BASE.relative_to(ROOT).as_posix()}/`",
        f"- Segment metrics: `{(OUT_BASE / 'step_c_segment_metrics.csv').relative_to(ROOT).as_posix()}`",
        f"- Step C summary: `{(OUT_BASE / 'step_c_case_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- Fixed-height metrics: `{(OUT_BASE / 'fixed_height_metrics.csv').relative_to(ROOT).as_posix()}`",
        f"- Fixed-height summary: `{(OUT_BASE / 'fixed_height_summary.csv').relative_to(ROOT).as_posix()}`",
        f"- Decision JSON: `{(OUT_BASE / 'decision_summary.json').relative_to(ROOT).as_posix()}`",
        "",
        "## Gate Summary",
        "",
        f"- Elapsed wall time: {_fmt(elapsed_s, 1)} s",
        f"- Fixed-height dwell: {fixed_steps} steps per height/profile",
        f"- Fixed-height classification: `{decision['fixed_height_classification']}`",
        f"- Hard failures: {len(decision['failures'])}",
        f"- Monitoring items: {len(decision['monitors'])}",
        f"- Inconclusive items: {len(decision['inconclusive'])}",
    ]

    if decision["failures"]:
        lines += ["", "Hard failure details:"]
        for item in decision["failures"][:50]:
            lines.append(f"- {item}")
    if decision["monitors"]:
        lines += ["", "Monitoring details:"]
        for item in decision["monitors"][:50]:
            lines.append(f"- {item}")
    if decision["inconclusive"]:
        lines += ["", "Inconclusive details:"]
        for item in decision["inconclusive"][:50]:
            lines.append(f"- {item}")

    lines += [
        "",
        "## Step C Case Summary",
        "",
        "| Case | Profile | any_fell | any_unsafe | max_maxabs m | max_trans m | p2p m | out15% | pitch max deg | roll max deg | hip-yaw max | WBC rows | low-band scale max | Kp eff max | trim max deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name in STEP_C_CASES:
        for tag, _profile, label in PROFILES:
            row = summaries.get(("step_c", case_name, tag))
            if not row:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_name,
                        label,
                        str(row["any_fell"]),
                        str(row["any_unsafe"]),
                        _fmt(row["max_maxabs"], 4),
                        _fmt(row["max_trans"], 4),
                        _fmt(row["support_position_error_p2p_m"], 4),
                        _fmt(row["out15_pct"], 1),
                        _fmt(row["pitch_max_abs_deg"], 2),
                        _fmt(row["roll_max_abs_deg"], 2),
                        _fmt(row["hip_yaw_abs_max"], 4),
                        str(row["wbc_applied_rows"]),
                        _fmt(row["low_band_scale_max"], 4),
                        _fmt(row["effective_support_kp_max"], 3),
                        _fmt(row["pitch_trim_deg_max_abs"], 3),
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## Candidate Comparisons",
        "",
        "| Case | C maxabs vs A m | C maxabs vs B m | C p2p vs A % | C p2p vs B % | C out15 vs A pp | C out15 vs B pp |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name in STEP_C_CASES:
        cand = summaries.get(("step_c", case_name, "C_LOW_BAND_V1"))
        ref_a = summaries.get(("step_c", case_name, "A_B2V2"))
        ref_b = summaries.get(("step_c", case_name, "B_CURRENT_PFF"))
        if not (cand and ref_a and ref_b):
            continue
        p2p_a = 100.0 * (float(cand["support_position_error_p2p_m"]) / max(1e-12, float(ref_a["support_position_error_p2p_m"])) - 1.0)
        p2p_b = 100.0 * (float(cand["support_position_error_p2p_m"]) / max(1e-12, float(ref_b["support_position_error_p2p_m"])) - 1.0)
        lines.append(
            "| "
            + " | ".join(
                [
                    case_name,
                    _fmt(float(cand["max_maxabs"]) - float(ref_a["max_maxabs"]), 4),
                    _fmt(float(cand["max_maxabs"]) - float(ref_b["max_maxabs"]), 4),
                    _fmt(p2p_a, 2),
                    _fmt(p2p_b, 2),
                    _fmt(float(cand["out15_pct"]) - float(ref_a["out15_pct"]), 1),
                    _fmt(float(cand["out15_pct"]) - float(ref_b["out15_pct"]), 1),
                ]
            )
            + " |"
        )

    lines += [
        "",
        "## Focused Gates",
        "",
        "| Case | Profile | maxabs m | p2p m | out15% | pitch max deg | hip-yaw max | hidden max | WBC rows | low-band scale max | Kp eff max | trim max deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case_name in ("focused_low_0p320", "focused_high_0p480"):
        for tag, _profile, label in PROFILES:
            row = seg.get(("step_c", case_name, 0, tag))
            if not row:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        case_name,
                        label,
                        _fmt(row["support_position_error_max_abs_m"], 10),
                        _fmt(row["support_position_error_p2p_m"], 10),
                        _fmt(row["out15_pct"], 1),
                        _fmt(row["pitch_max_abs_deg"], 2),
                        _fmt(row["hip_yaw_abs_max"], 4),
                        _fmt(row["hidden_torque_max"], 4),
                        str(row["wbc_applied_rows"]),
                        _fmt(row["low_band_scale_max"], 4),
                        _fmt(row["effective_support_kp_max"], 3),
                        _fmt(row["pitch_trim_deg_max_abs"], 3),
                    ]
                )
                + " |"
            )

    lines += [
        "",
        "## Fixed-Height 10-Height Summary",
        "",
        "| Height | Profile | any_fell | any_unsafe | maxabs m | p2p m | out15% | pitch max deg | roll max deg | hip-yaw max | hidden max | WBC rows | low-band scale max | Kp eff max | trim max deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fixed_by = _by_key(fixed_rows, "height", "tag")
    for height in HEIGHTS:
        for tag, _profile, label in PROFILES:
            row = fixed_by.get((height, tag))
            if not row:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        height,
                        label,
                        str(row["any_fell"]),
                        str(row["any_unsafe"]),
                        _fmt(row["support_position_error_max_abs_m"], 10),
                        _fmt(row["support_position_error_p2p_m"], 10),
                        _fmt(row["out15_pct"], 1),
                        _fmt(row["pitch_max_abs_deg"], 2),
                        _fmt(row["roll_max_abs_deg"], 2),
                        _fmt(row["hip_yaw_abs_max"], 4),
                        _fmt(row["hidden_torque_max"], 4),
                        str(row["wbc_applied_rows"]),
                        _fmt(row["low_band_scale_max"], 4),
                        _fmt(row["effective_support_kp_max"], 3),
                        _fmt(row["pitch_trim_deg_max_abs"], 3),
                    ]
                )
                + " |"
            )

    if classification in {
        "PHYSICS_FF_LOW_BAND_V1_STEP_C_PASS",
        "PHYSICS_FF_LOW_BAND_V1_STEP_C_PASS_WITH_MONITORING",
    }:
        lines += [
            "",
            "## Decision",
            "",
            "Step D may be run next.",
            "Step D was not run in this task.",
            "The candidate remains opt-in; this report does not promote PFF or change defaults.",
        ]
    elif classification == "PHYSICS_FF_LOW_BAND_V1_STEP_C_FAIL":
        focused_low = decision.get("focused_low_0p320", {})
        cand = focused_low.get("C_LOW_BAND_V1") if isinstance(focused_low, dict) else None
        pff = focused_low.get("B_CURRENT_PFF") if isinstance(focused_low, dict) else None
        focused_worse = "unknown"
        if cand and pff:
            focused_worse = "yes" if float(cand["support_position_error_max_abs_m"]) > float(pff["support_position_error_max_abs_m"]) else "no"
        fixed_low_lines: list[str] = []
        fixed_low_c = fixed_by.get(("low_0p320", "C_LOW_BAND_V1"))
        fixed_low_b = fixed_by.get(("low_0p320", "B_CURRENT_PFF"))
        if fixed_low_c and fixed_low_b:
            candidate_p2p = float(fixed_low_c["support_position_error_p2p_m"])
            current_p2p = float(fixed_low_b["support_position_error_p2p_m"])
            threshold_p2p = current_p2p * P2P_FACTOR
            fixed_low_lines = [
                (
                    "Failing protected metric: fixed-height `low_0p320` P2P "
                    f"{candidate_p2p:.10f} m vs current PFF {current_p2p:.10f} m; "
                    f"15% threshold {threshold_p2p:.10f} m; exceeded by "
                    f"{candidate_p2p - threshold_p2p:.10f} m."
                ),
                (
                    "Worse than current/original PFF on the failing fixed-height P2P metric: "
                    f"{'yes' if candidate_p2p > current_p2p else 'no'}."
                ),
            ]
        lines += [
            "",
            "## Decision",
            "",
            f"Step C failed. Candidate worse than original/current PFF on focused low maxabs: {focused_worse}.",
            *fixed_low_lines,
            "Step D was not run.",
            "The candidate remains opt-in; this report does not promote PFF or change defaults.",
        ]
    else:
        lines += [
            "",
            "## Decision",
            "",
            "Step C is inconclusive because required telemetry or summaries were missing.",
            "Step D was not run.",
            "The candidate remains opt-in; this report does not promote PFF or change defaults.",
        ]

    return "\n".join(lines) + "\n"


def build_specs(*, fixed_steps: int, include_step_c: bool, include_fixed: bool) -> list[RunSpec]:
    specs: list[RunSpec] = []
    if include_step_c:
        for case_name, segments in STEP_C_CASES.items():
            for seg_idx, (height, steps) in enumerate(segments):
                for tag, profile, label in PROFILES:
                    specs.append(RunSpec("step_c", case_name, seg_idx, height, steps, tag, profile, label))
    if include_fixed:
        for seg_idx, height in enumerate(HEIGHTS):
            for tag, profile, label in PROFILES:
                specs.append(RunSpec("fixed", "fixed_height", seg_idx, height, fixed_steps, tag, profile, label))
    return specs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full Step C validation for PFF low-band support v1")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--fixed-steps", type=int, default=2000)
    parser.add_argument("--timeout-s", type=int, default=2400)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-step-c", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    args = parser.parse_args()

    if not SETUP_DIR.exists():
        raise FileNotFoundError(f"Missing centered setup directory: {SETUP_DIR}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    start = time.time()
    specs = build_specs(
        fixed_steps=args.fixed_steps,
        include_step_c=not args.skip_step_c,
        include_fixed=not args.skip_fixed,
    )
    (OUT_BASE / "run_matrix.json").write_text(
        json.dumps([spec.__dict__ for spec in specs], indent=2), encoding="utf-8"
    )

    print(f"Running {len(specs)} validation simulations with max_workers={args.max_workers}", flush=True)
    run_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        future_map = {
            executor.submit(run_sim, spec, force=args.force, timeout_s=args.timeout_s): spec for spec in specs
        }
        completed = 0
        for future in as_completed(future_map):
            spec = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - validation guard
                result = {
                    "spec": spec,
                    "telemetry_path": "",
                    "summary_path": "",
                    "returncode": None,
                    "cached": False,
                    "error": repr(exc),
                }
            run_results.append(result)
            completed += 1
            cached = "cached" if result.get("cached") else "ran"
            err = result.get("error") or "ok"
            print(
                f"[{completed:03d}/{len(specs):03d}] {spec.suite} {spec.case_name} "
                f"{spec.seg_idx:03d} {spec.height} {spec.tag}: {cached} {err}",
                flush=True,
            )

    segment_rows = [analyze_telemetry(result) for result in run_results]
    segment_rows.sort(key=lambda row: (row["suite"], row["case_name"], row["seg_idx"], row["tag"]))
    step_c_rows = [row for row in segment_rows if row["suite"] == "step_c"]
    fixed_rows = [row for row in segment_rows if row["suite"] == "fixed"]
    step_c_summary = _aggregate(segment_rows, suite="step_c")
    fixed_summary = _aggregate(segment_rows, suite="fixed")
    decision = evaluate_decision(segment_rows, step_c_summary, fixed_summary)

    _write_csv(OUT_BASE / "step_c_segment_metrics.csv", step_c_rows)
    _write_csv(OUT_BASE / "step_c_case_summary.csv", step_c_summary)
    _write_csv(OUT_BASE / "fixed_height_metrics.csv", fixed_rows)
    _write_csv(OUT_BASE / "fixed_height_summary.csv", fixed_summary)

    elapsed_s = time.time() - start
    decision_payload = {
        **decision,
        "elapsed_s": elapsed_s,
        "fixed_steps": args.fixed_steps,
        "profiles": [{"tag": tag, "profile": profile, "label": label} for tag, profile, label in PROFILES],
        "output_dir": str(OUT_BASE),
        "report_path": str(REPORT_PATH),
        "step_c_case_count": len(STEP_C_CASES),
        "step_c_segment_count": len(step_c_rows),
        "fixed_height_segment_count": len(fixed_rows),
    }
    # Remove row dictionaries embedded in focused entries from JSON readability.
    for key in ("focused_low_0p320", "focused_high_0p480"):
        focused = decision_payload.get(key)
        if isinstance(focused, dict):
            decision_payload[key] = {
                tag: {
                    "support_position_error_max_abs_m": row.get("support_position_error_max_abs_m"),
                    "support_position_error_p2p_m": row.get("support_position_error_p2p_m"),
                    "out15_pct": row.get("out15_pct"),
                    "any_fell": row.get("any_fell"),
                    "any_unsafe": row.get("any_unsafe"),
                }
                for tag, row in focused.items()
                if row
            }
    (OUT_BASE / "decision_summary.json").write_text(json.dumps(decision_payload, indent=2), encoding="utf-8")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        render_report(
            segment_rows=segment_rows,
            step_c_summary=step_c_summary,
            fixed_rows=fixed_rows,
            fixed_summary=fixed_summary,
            decision=decision,
            elapsed_s=elapsed_s,
            fixed_steps=args.fixed_steps,
        ),
        encoding="utf-8",
    )

    print(f"Classification: {decision['classification']}", flush=True)
    print(f"Report: {REPORT_PATH}", flush=True)
    return 0 if decision["classification"] != "PHYSICS_FF_LOW_BAND_V1_STEP_C_FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
