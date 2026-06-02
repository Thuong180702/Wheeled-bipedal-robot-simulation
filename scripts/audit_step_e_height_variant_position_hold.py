from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUT_DIR = Path("outputs/step_e_height_variant_position_hold_audit")
SETUP_REPORT = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
DYNAMIC_SUMMARY = Path("outputs/balance_core_true_height_variants/true_height_variant_full_validation_summary.json")
GATE_METRICS = OUTPUT_DIR / "gate_500" / "step_c_height_recovery_metrics.json"

VARIANTS = ["nominal", "low_tiny", "high_tiny", "low_small", "high_small"]
EXPECTED_STEPS = 5000

BASELINE = {
    "support_position_error_max_abs_m": 0.104456751,
    "support_position_error_final_abs_m": 0.091351773,
    "hip_yaw_max_abs_rad": 0.0567,
    "hip_yaw_rms_rad": 0.022819449,
    "pitch_x_max_abs_rad": 0.070771351,
    "roll_y_max_abs_rad": 0.012998945,
    "com_z_min_m": 0.403835297,
    "wheel_vel_mean_max_abs_rad_s": 3.839568138,
}


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def numeric_column(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return None


def bool_column(df: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in df.columns:
        return None
    return df[name].map(lambda value: str(value).lower() in {"true", "1", "yes"})


def rms(values: pd.Series) -> float | None:
    values = values.dropna()
    if len(values) == 0:
        return None
    return float(math.sqrt(float((values * values).mean())))


def max_abs(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    values = values.dropna()
    if len(values) == 0:
        return None
    return float(values.abs().max())


def final_value(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    values = values.dropna()
    if len(values) == 0:
        return None
    return float(values.iloc[-1])


def pct_gt_abs(values: pd.Series | None, threshold: float) -> float | None:
    if values is None:
        return None
    values = values.dropna()
    if len(values) == 0:
        return None
    return float(100.0 * (values.abs() > threshold).mean())


def longest_true_run(mask: pd.Series) -> int:
    longest = 0
    current = 0
    for value in mask.tolist():
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def within_10_percent(value: float | None, baseline: float) -> bool | None:
    if value is None:
        return None
    return abs(value) <= abs(baseline) * 1.10


def load_setup_variants() -> dict[str, dict[str, Any]]:
    report = json.loads(SETUP_REPORT.read_text(encoding="utf-8"))
    return {entry["variant_name"]: entry for entry in report["setup_results"]}


def load_dynamic_telemetry_sources() -> dict[str, str]:
    if not DYNAMIC_SUMMARY.exists():
        return {}
    summary = json.loads(DYNAMIC_SUMMARY.read_text(encoding="utf-8"))
    sources: dict[str, str] = {}
    for result in summary.get("results", []):
        if result.get("target_steps") == 1000 and result.get("success") and result.get("telemetry_path"):
            sources[result["variant_name"]] = result["telemetry_path"]
    return sources


def load_gate_status() -> dict[str, dict[str, Any]]:
    if not GATE_METRICS.exists():
        return {}
    return {entry["case_name"]: entry for entry in json.loads(GATE_METRICS.read_text(encoding="utf-8"))}


def resolve_existing_path(raw_path: str) -> Path | None:
    candidates = [Path(raw_path), Path(raw_path.replace("\\", "/"))]
    for marker in ("outputs/", "outputs\\"):
        if marker in raw_path:
            candidates.append(Path(raw_path[raw_path.index(marker):].replace("\\", "/")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def analyze_telemetry(path: Path, variant: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(path)
    source_step = numeric_column(df, "source_step_index", "step")
    time = numeric_column(df, "time", "sim_time_s")
    support = numeric_column(df, "support_position_error_m")
    hip_yaw = numeric_column(df, "hip_yaw_abs_max")
    pitch = numeric_column(df, "pitch_x_rad", "robot_pitch_x", "pitch_x")
    roll = numeric_column(df, "roll_y_rad", "robot_roll_y", "roll_y")
    yaw = numeric_column(df, "yaw_z_rad", "robot_yaw_z", "yaw_z")
    com_z = numeric_column(df, "com_z_m", "com_z")
    wheel = numeric_column(df, "wheel_vel_mean_rad_s")
    contact_force_valid = bool_column(df, "contact_force_valid")
    left_contact = bool_column(df, "left_wheel_contact")
    right_contact = bool_column(df, "right_wheel_contact")
    if contact_force_valid is not None and left_contact is not None and right_contact is not None:
        contact_valid = contact_force_valid & left_contact & right_contact
    else:
        contact_valid = None
    non_wheel = numeric_column(df, "non_wheel_floor_contacts")
    hidden = numeric_column(df, "hidden_torque_norm")
    ownership = numeric_column(df, "ownership_violation_count")
    applied_wbc = numeric_column(df, "applied_wbc_contribution_norm")

    owner_has_wbc = False
    if "active_torque_owner_per_joint" in df.columns:
        owner_has_wbc = bool(df["active_torque_owner_per_joint"].astype(str).str.lower().str.contains("wbc").any())

    support_abs = support.abs() if support is not None else None
    support_peak_idx = int(support_abs.idxmax()) if support_abs is not None and len(support_abs.dropna()) else None
    support_peak_time = float(time.iloc[support_peak_idx]) if support_peak_idx is not None and time is not None else None

    achieved = float(variant["achieved_com_z_m"])
    target = float(variant["target_com_z_m"])
    height_drift = (com_z - achieved) if com_z is not None else None
    target_error = (com_z - target) if com_z is not None else None
    achieved_error = (com_z - achieved) if com_z is not None else None

    raw_contact_percent = None
    invalid_rows = None
    longest_invalid = None
    if contact_valid is not None:
        raw_contact_percent = float(100.0 * contact_valid.mean())
        invalid_mask = ~contact_valid
        invalid_rows = int(invalid_mask.sum())
        longest_invalid = longest_true_run(invalid_mask)

    hidden_max = max_abs(hidden)
    ownership_max = int(ownership.max()) if ownership is not None and len(ownership.dropna()) else None
    applied_wbc_max = max_abs(applied_wbc)
    wbc_applied = bool(owner_has_wbc or (applied_wbc_max is not None and applied_wbc_max > 1e-9))

    metrics = {
        "variant_name": variant["variant_name"],
        "telemetry_source": str(path),
        "telemetry_context": "historical_step_b_1000_step_supplemental_not_official_5000_step_audit",
        "target_com_z_m": target,
        "achieved_initial_com_z_m": achieved,
        "calibrated_root_z_m": finite(variant.get("calibrated_root_z_m")),
        "hip_pitch_ref": finite(variant.get("hip_pitch_ref")),
        "knee_ref": finite(variant.get("knee_ref")),
        "setup_valid": bool(variant.get("setup_valid")),
        "initial_wheel_contacts": {
            "left": bool(variant.get("left_wheel_contact")),
            "right": bool(variant.get("right_wheel_contact")),
        },
        "initial_non_wheel_contacts": int(variant.get("non_wheel_floor_contact_count", 0)),
        "initial_support_error_m": finite(variant.get("com_support_error_norm_xy")),
        "row_count": int(len(df)),
        "source_step_index_min": int(source_step.min()) if source_step is not None else None,
        "source_step_index_max": int(source_step.max()) if source_step is not None else None,
        "final_time_s": final_value(time),
        "survived_5000_steps": len(df) >= EXPECTED_STEPS,
        "support_position_error_m": {
            "min": float(support.min()) if support is not None else None,
            "max": float(support.max()) if support is not None else None,
            "final": final_value(support),
            "rms": rms(support) if support is not None else None,
            "max_abs": max_abs(support),
            "peak_time_s": support_peak_time,
            "within_required_0p15_m": max_abs(support) is not None and max_abs(support) <= 0.15,
            "within_preferred_0p12_m": max_abs(support) is not None and max_abs(support) <= 0.12,
            "final_abs_within_preferred_0p10_m": final_value(support) is not None and abs(final_value(support)) <= 0.10,
        },
        "posture": {
            "hip_yaw_abs_max_max_rad": max_abs(hip_yaw),
            "hip_yaw_abs_max_final_rad": final_value(hip_yaw),
            "hip_yaw_abs_max_rms_rad": rms(hip_yaw) if hip_yaw is not None else None,
            "percent_hip_yaw_gt_0p07_rad": pct_gt_abs(hip_yaw, 0.07),
            "percent_hip_yaw_gt_0p10_rad": pct_gt_abs(hip_yaw, 0.10),
            "pitch_x_max_abs_rad": max_abs(pitch),
            "pitch_x_final_rad": final_value(pitch),
            "pitch_x_rms_rad": rms(pitch) if pitch is not None else None,
            "roll_y_max_abs_rad": max_abs(roll),
            "roll_y_final_rad": final_value(roll),
            "roll_y_rms_rad": rms(roll) if roll is not None else None,
            "yaw_z_max_abs_rad": max_abs(yaw),
            "yaw_z_final_rad": final_value(yaw),
            "yaw_z_rms_rad": rms(yaw) if yaw is not None else None,
        },
        "height": {
            "com_z_min_m": float(com_z.min()) if com_z is not None else None,
            "com_z_max_m": float(com_z.max()) if com_z is not None else None,
            "com_z_final_m": final_value(com_z),
            "com_z_rms_m": rms(com_z) if com_z is not None else None,
            "drift_from_achieved_initial_min_m": float(height_drift.min()) if height_drift is not None else None,
            "drift_from_achieved_initial_max_m": float(height_drift.max()) if height_drift is not None else None,
            "drift_from_achieved_initial_final_m": final_value(height_drift),
            "final_error_vs_target_m": final_value(target_error),
            "final_error_vs_achieved_initial_m": final_value(achieved_error),
            "within_0p02_m_of_target_or_achieved": (
                final_value(target_error) is not None
                and final_value(achieved_error) is not None
                and min(abs(final_value(target_error)), abs(final_value(achieved_error))) <= 0.02
            ),
        },
        "wheel_contact": {
            "wheel_vel_mean_max_abs_rad_s": max_abs(wheel),
            "wheel_vel_mean_final_rad_s": final_value(wheel),
            "wheel_vel_mean_rms_rad_s": rms(wheel) if wheel is not None else None,
            "contact_valid_percent_raw": raw_contact_percent,
            "contact_valid_percent_adjusted": raw_contact_percent,
            "invalid_contact_rows": invalid_rows,
            "consecutive_invalid_contact_run_max": longest_invalid,
            "non_wheel_floor_contacts_max": max_abs(non_wheel),
        },
        "structural_invariants": {
            "wbc_applied": wbc_applied,
            "applied_wbc_contribution_norm_max": applied_wbc_max,
            "tau_wbc_norm_is_diagnostic_only": "tau_wbc_norm" in df.columns and not wbc_applied,
            "hidden_torque_norm_max": hidden_max,
            "ownership_violation_count_max": ownership_max,
            "active_torque_owner_mentions_wbc": owner_has_wbc,
        },
        "baseline_10_percent_comparison": {
            "support_position_error_max_abs_within_10_percent": within_10_percent(max_abs(support), BASELINE["support_position_error_max_abs_m"]),
            "support_position_error_final_abs_within_10_percent": within_10_percent(abs(final_value(support)) if final_value(support) is not None else None, BASELINE["support_position_error_final_abs_m"]),
            "hip_yaw_max_abs_within_10_percent": within_10_percent(max_abs(hip_yaw), BASELINE["hip_yaw_max_abs_rad"]),
            "pitch_max_abs_within_10_percent": within_10_percent(max_abs(pitch), BASELINE["pitch_x_max_abs_rad"]),
            "roll_max_abs_within_10_percent": within_10_percent(max_abs(roll), BASELINE["roll_y_max_abs_rad"]),
            "wheel_velocity_max_abs_within_10_percent": within_10_percent(max_abs(wheel), BASELINE["wheel_vel_mean_max_abs_rad_s"]),
        },
    }
    return metrics


def official_case_result(variant: dict[str, Any], gate: dict[str, Any] | None, supplemental: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "variant_name": variant["variant_name"],
        "verdict": "INCONCLUSIVE",
        "primary_failure": "telemetry_missing",
        "failure_classifications": ["telemetry_missing", "unclear_requires_more_telemetry"],
        "failure_lead": "initialization-led",
        "official_5000_step_telemetry_available": False,
        "fresh_gate_500_status": None if gate is None else {
            "verdict": gate.get("verdict"),
            "simulation_returncode": gate.get("simulation_returncode"),
            "simulation_error": gate.get("simulation_error"),
            "telemetry_path": gate.get("telemetry_path"),
        },
        "simulation_blocker": {
            "type": "import_error_before_simulation_start",
            "file": "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
            "line": 43,
            "error": "IndentationError: unexpected indent",
        },
        "supplemental_historical_1000_step_metrics": supplemental,
    }


def write_report(summary: dict[str, Any], metrics: list[dict[str, Any]]) -> str:
    lines = [
        "# Step E Height-Variant Position-Hold Audit",
        "",
        "## Verdict",
        "",
        f"- Overall audit verdict: **{summary['overall_audit_verdict']}**",
        f"- Step E nominal remains valid: **{str(summary['step_e_nominal_remains_valid']).lower()}**",
        f"- Step E across true height variants passes: **{str(summary['step_e_height_variant_robust']).lower()}**",
        "- Controller behavior changed: `false`",
        "- WBC applied: `false` in available historical telemetry; fresh 5000-step audit unavailable",
        "",
        "## Root cause for inconclusive official audit",
        "",
        "Fresh simulation cannot start because the current working tree has an IndentationError in `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py:43`. The audit did not modify controller files or tune gains.",
        "",
        "## Per-variant official verdicts",
        "",
        "| Variant | Official 5000-step verdict | Supplemental rows | Support max abs (m) | Support final (m) | Pitch max abs (rad) | Roll max abs (rad) | Height final vs achieved (m) | WBC/ownership |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    by_variant = {m["variant_name"]: m for m in metrics}
    for name in VARIANTS:
        m = by_variant.get(name, {})
        support = m.get("support_position_error_m", {})
        posture = m.get("posture", {})
        height = m.get("height", {})
        structural = m.get("structural_invariants", {})
        lines.append(
            "| "
            + " | ".join([
                name,
                "INCONCLUSIVE",
                str(m.get("row_count")),
                fmt(support.get("max_abs")),
                fmt(support.get("final")),
                fmt(posture.get("pitch_x_max_abs_rad")),
                fmt(posture.get("roll_y_max_abs_rad")),
                fmt(height.get("final_error_vs_achieved_initial_m")),
                f"wbc={structural.get('wbc_applied')}, hidden={fmt(structural.get('hidden_torque_norm_max'))}, owner={structural.get('ownership_violation_count_max')}",
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The requested 5000-step Step E height-variant hold verdict is **INCONCLUSIVE**, not FAIL, because no fresh official telemetry could be produced.",
        "- Historical Step B 1000-step telemetry exists for all variants and is copied as supplemental evidence only; it cannot satisfy the required 5000-step final-verdict criterion.",
        "- No Step C DONE claim is made.",
        "",
        "## Recommended next action",
        "",
        "Resolve the separate uncommitted syntax error in the ongoing Step C work, then rerun the same diagnostic-only audit to produce fresh 5000-step telemetry for all variants.",
        "",
    ])
    return "\n".join(lines)


def fmt(value: Any) -> str:
    number = finite(value)
    if number is None:
        return "n/a"
    return f"{number:.9g}"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    variants = load_setup_variants()
    telemetry_sources = load_dynamic_telemetry_sources()
    gate_status = load_gate_status()

    case_matrix = []
    supplemental_metrics = []
    official_results = []

    for name in VARIANTS:
        variant = variants[name]
        case_matrix.append({
            "variant_name": name,
            "target_com_z_m": variant["target_com_z_m"],
            "achieved_initial_com_z_m": variant["achieved_com_z_m"],
            "calibrated_root_z_m": variant["calibrated_root_z_m"],
            "hip_pitch_ref": variant["hip_pitch_ref"],
            "knee_ref": variant["knee_ref"],
            "setup_valid": variant["setup_valid"],
            "initial_wheel_contacts": {
                "left": variant["left_wheel_contact"],
                "right": variant["right_wheel_contact"],
            },
            "initial_non_wheel_contacts": variant["non_wheel_floor_contact_count"],
            "initial_support_error_m": variant.get("com_support_error_norm_xy"),
            "official_steps_requested": EXPECTED_STEPS,
            "fresh_500_step_gate_attempted": name in gate_status,
        })

        source = telemetry_sources.get(name)
        supplemental = None
        source_path = resolve_existing_path(source) if source else None
        if source_path is not None:
            destination = OUTPUT_DIR / f"{name}_telemetry.csv"
            shutil.copy2(source_path, destination)
            supplemental = analyze_telemetry(destination, variant)
            supplemental_metrics.append(supplemental)
        official_results.append(official_case_result(variant, gate_status.get(name), supplemental))

    summary = {
        "overall_audit_verdict": "INCONCLUSIVE",
        "step_e_nominal_remains_valid": True,
        "step_e_nominal_basis": "official Step E v2 summary remains PASS; fresh nominal height-variant rerun blocked before simulation startup",
        "step_e_height_variant_robust": False,
        "step_e_height_variant_robustness_verdict": "INCONCLUSIVE",
        "controller_behavior_changed": False,
        "wbc_applied": False,
        "official_5000_step_cases_available": [],
        "official_5000_step_cases_missing": VARIANTS,
        "per_variant_verdicts": {name: "INCONCLUSIVE" for name in VARIANTS},
        "failures_match_current_step_c_high_tiny_issue": "INCONCLUSIVE: high_tiny could not be rerun; historical 1000-step Step B telemetry is supplemental only",
        "simulation_blocker": {
            "type": "import_error_before_simulation_start",
            "file": "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
            "line": 43,
            "error": "IndentationError: unexpected indent",
        },
        "recommended_next_action": "Resolve the separate uncommitted syntax error in the ongoing Step C work, then rerun the same diagnostic-only audit for fresh 5000-step data.",
    }

    (OUTPUT_DIR / "step_e_height_variant_case_matrix.json").write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step_e_height_variant_metrics.json").write_text(json.dumps(official_results, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step_e_height_variant_position_hold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step_e_height_variant_supplemental_1000_step_metrics.json").write_text(json.dumps(supplemental_metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step_e_height_variant_position_hold_report.md").write_text(write_report(summary, supplemental_metrics), encoding="utf-8")

    for result in official_results:
        audit_path = OUTPUT_DIR / f"{result['variant_name']}_failure_audit.json"
        audit_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
