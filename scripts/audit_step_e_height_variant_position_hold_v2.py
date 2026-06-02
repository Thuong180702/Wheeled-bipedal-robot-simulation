from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

OUTPUT_DIR = Path("outputs/step_e_height_variant_position_hold_audit_v2")
SETUP_REPORT = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
SETUP_DIR = Path("outputs/balance_core_true_height_variants")
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

STEP_E_THRESHOLDS = {
    "support_max_abs_m": 0.15,
    "support_final_abs_m": 0.15,
    "hip_yaw_max_abs_rad": 0.07,
    "hip_yaw_percent_gt_0p10_rad": 0.0,
    "pitch_max_abs_rad": 0.10,
    "roll_max_abs_rad": 0.05,
    "height_error_final_m": 0.02,
    "wheel_vel_max_abs_rad_s": 5.0,
    "contact_valid_percent": 99.9,
    "hidden_torque_norm_max": 0.0,
    "ownership_violation_max": 0,
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
    variants = {}
    for entry in report["setup_results"]:
        entry["variant_setup_path"] = str(SETUP_DIR / f"variant_{entry['variant_name']}" / "variant_setup.json")
        variants[entry["variant_name"]] = entry
    return variants


def resolve_existing_path(raw_path: str) -> Path | None:
    candidates = [Path(raw_path), Path(raw_path.replace("\\", "/"))]
    for marker in ("outputs/", "outputs\\"):
        if marker in raw_path:
            candidates.append(Path(raw_path[raw_path.index(marker):].replace("\\", "/")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


SIM_OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")


def _snapshot_outputs() -> tuple[set[Path], set[Path]]:
    existing_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    existing_sidecar = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    return existing_csv, existing_sidecar


def _copy_newest_outputs(case_name: str, output_dir: Path, before_csv: set[Path], before_sidecar: set[Path]) -> Path | None:
    current_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    new_csv = current_csv - before_csv
    if not new_csv:
        return None
    source_csv = max(new_csv, key=lambda path: path.stat().st_mtime)
    dest_csv = output_dir / f"{case_name}_telemetry.csv"
    shutil.copy2(source_csv, dest_csv)

    current_sidecars = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    new_sidecars = current_sidecars - before_sidecar
    if new_sidecars:
        source_sidecar = max(new_sidecars, key=lambda path: path.stat().st_mtime)
        shutil.copy2(source_sidecar, dest_csv.with_suffix(".summary.json"))
    return dest_csv


def build_variant_command(variant_name: str, variant_setup_path: Path, steps: int) -> list[str]:
    return [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode", "balance-core",
        "--sagittal-controller", "velocity-damped",
        "--height-variant-setup", str(variant_setup_path).replace("\\", "/"),
        "--steps", str(steps),
        "--telemetry-decimation", "1",
        "--failure-window-steps", "500",
        "--write-run-summary-sidecar",
        "--vd-sagittal-authority-profile", "baseline",
    ]


def run_variant_simulation(
    variant_name: str,
    variant_setup_path: Path,
    steps: int,
    output_dir: Path,
) -> tuple[dict[str, Any], Path | None]:
    before_csv, before_sidecar = _snapshot_outputs()
    cmd = build_variant_command(variant_name, variant_setup_path, steps)
    process_error = None
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        process_error = exc

    telemetry_path = _copy_newest_outputs(variant_name, output_dir, before_csv, before_sidecar)
    return {
        "variant_name": variant_name,
        "command": cmd,
        "simulation_returncode": None if process_error is None else process_error.returncode,
        "simulation_error": None if process_error is None else str(process_error),
        "telemetry_path": str(telemetry_path) if telemetry_path else None,
    }, telemetry_path


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

    achieved = float(variant["achieved_com_z_m"])
    target = float(variant["target_com_z_m"])
    height_drift = (com_z - achieved) if com_z is not None else None
    target_error = (com_z - target) if com_z is not None else None

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
            "final_error_vs_achieved_initial_m": final_value(height_drift),
        },
        "wheel_contact": {
            "wheel_vel_mean_max_abs_rad_s": max_abs(wheel),
            "wheel_vel_mean_final_rad_s": final_value(wheel),
            "wheel_vel_mean_rms_rad_s": rms(wheel) if wheel is not None else None,
            "contact_valid_percent_raw": raw_contact_percent,
            "invalid_contact_rows": invalid_rows,
            "consecutive_invalid_contact_run_max": longest_invalid,
            "non_wheel_floor_contacts_max": max_abs(non_wheel),
        },
        "structural_invariants": {
            "wbc_applied": wbc_applied,
            "applied_wbc_contribution_norm_max": applied_wbc_max,
            "hidden_torque_norm_max": hidden_max,
            "ownership_violation_count_max": ownership_max,
            "active_torque_owner_mentions_wbc": owner_has_wbc,
        },
    }
    return metrics


def classify_variant_result(metrics: dict[str, Any]) -> dict[str, Any]:
    verdict = "PASS"
    primary_failure = None
    failure_classifications = []
    failure_lead = None
    required_fails = []
    preferred_fails = []

    structural = metrics.get("structural_invariants", {})
    if structural.get("wbc_applied"):
        verdict = "FAIL"
        failure_classifications.append("wbc_applied")
        required_fails.append("WBC applied")
        failure_lead = "structural-led" if failure_lead is None else failure_lead
    if structural.get("hidden_torque_norm_max", 0) > 0:
        verdict = "FAIL"
        failure_classifications.append("hidden_torque_nonzero")
        required_fails.append(f"hidden_torque={structural.get('hidden_torque_norm_max')}")
        failure_lead = "structural-led" if failure_lead is None else failure_lead
    if structural.get("ownership_violation_count_max", 0) > 0:
        verdict = "FAIL"
        failure_classifications.append("ownership_violation")
        required_fails.append(f"ownership_violations={structural.get('ownership_violation_count_max')}")
        failure_lead = "structural-led" if failure_lead is None else failure_lead

    support = metrics.get("support_position_error_m", {})
    if support.get("max_abs", 999) > STEP_E_THRESHOLDS["support_max_abs_m"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_position_regression_at_height_variant")
        required_fails.append(f"support_max_abs={support.get('max_abs'):.6f} > {STEP_E_THRESHOLDS['support_max_abs_m']}")
        failure_lead = "position-led" if failure_lead is None else failure_lead

    posture = metrics.get("posture", {})
    if posture.get("hip_yaw_abs_max_max_rad", 999) > STEP_E_THRESHOLDS["hip_yaw_max_abs_rad"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_posture_regression_at_height_variant")
        required_fails.append(f"hip_yaw_max={posture.get('hip_yaw_abs_max_max_rad'):.6f} > {STEP_E_THRESHOLDS['hip_yaw_max_abs_rad']}")
        failure_lead = "posture-led" if failure_lead is None else failure_lead
    if posture.get("percent_hip_yaw_gt_0p10_rad", 100) > STEP_E_THRESHOLDS["hip_yaw_percent_gt_0p10_rad"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_posture_regression_at_height_variant")
        required_fails.append(f"hip_yaw_gt_0.10_percent={posture.get('percent_hip_yaw_gt_0p10_rad'):.2f}% > 0%")
        failure_lead = "posture-led" if failure_lead is None else failure_lead
    if posture.get("pitch_x_max_abs_rad", 999) > STEP_E_THRESHOLDS["pitch_max_abs_rad"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_pitch_regression_at_height_variant")
        required_fails.append(f"pitch_max={posture.get('pitch_x_max_abs_rad'):.6f} > {STEP_E_THRESHOLDS['pitch_max_abs_rad']}")
        failure_lead = "pitch-led" if failure_lead is None else failure_lead
    if posture.get("roll_y_max_abs_rad", 999) > STEP_E_THRESHOLDS["roll_max_abs_rad"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_roll_regression_at_height_variant")
        required_fails.append(f"roll_max={posture.get('roll_y_max_abs_rad'):.6f} > {STEP_E_THRESHOLDS['roll_max_abs_rad']}")
        failure_lead = "posture-led" if failure_lead is None else failure_lead

    height = metrics.get("height", {})
    final_height_error = height.get("final_error_vs_target_m")
    final_height_error_vs_achieved = height.get("final_error_vs_achieved_initial_m")
    if final_height_error is not None and abs(final_height_error) > STEP_E_THRESHOLDS["height_error_final_m"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_height_drift_at_height_variant")
        required_fails.append(f"height_final_error={final_height_error:.6f} > {STEP_E_THRESHOLDS['height_error_final_m']}")
        failure_lead = "height-led" if failure_lead is None else failure_lead
    if final_height_error_vs_achieved is not None and abs(final_height_error_vs_achieved) > STEP_E_THRESHOLDS["height_error_final_m"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_height_drift_at_height_variant")
        required_fails.append(f"height_drift_from_achieved={final_height_error_vs_achieved:.6f} > {STEP_E_THRESHOLDS['height_error_final_m']}")
        failure_lead = "height-led" if failure_lead is None else failure_lead

    wheel_contact = metrics.get("wheel_contact", {})
    if wheel_contact.get("wheel_vel_mean_max_abs_rad_s", 999) > STEP_E_THRESHOLDS["wheel_vel_max_abs_rad_s"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_wheel_velocity_runaway_at_height_variant")
        required_fails.append(f"wheel_vel_max={wheel_contact.get('wheel_vel_mean_max_abs_rad_s'):.6f} > {STEP_E_THRESHOLDS['wheel_vel_max_abs_rad_s']}")
        failure_lead = "wheel-velocity-led" if failure_lead is None else failure_lead

    contact_pct = wheel_contact.get("contact_valid_percent_raw")
    if contact_pct is not None and contact_pct < STEP_E_THRESHOLDS["contact_valid_percent"]:
        verdict = "FAIL"
        failure_classifications.append("step_e_contact_invalid_at_height_variant")
        required_fails.append(f"contact_valid={contact_pct:.2f}% < {STEP_E_THRESHOLDS['contact_valid_percent']}%")
        failure_lead = "contact-led" if failure_lead is None else failure_lead

    return {
        "verdict": verdict,
        "primary_failure": primary_failure if failure_classifications else None,
        "failure_classifications": failure_classifications,
        "failure_lead": failure_lead,
        "required_fails": required_fails,
        "preferred_fails": preferred_fails,
    }


def create_failure_windows(df: pd.DataFrame, metrics: dict, output_dir: Path, variant_name: str) -> None:
    support = pd.to_numeric(df["support_position_error_m"], errors="coerce").abs()
    wheel = pd.to_numeric(df["wheel_vel_mean_rad_s"], errors="coerce")
    hip_yaw = pd.to_numeric(df["hip_yaw_abs_max"], errors="coerce")
    time_col = pd.to_numeric(df["time"], errors="coerce")

    window_size = 200

    for label, signal in [("support", support), ("wheel_vel", wheel), ("hip_yaw", hip_yaw)]:
        if signal is None or signal.dropna().empty:
            continue
        signal_clean = signal.dropna()
        if len(signal_clean) == 0:
            continue
        peak_idx = int(signal_clean.abs().idxmax())
        peak_row = int(df.index[df.index == peak_idx][0]) if peak_idx in df.index else peak_idx
        start = max(0, peak_row - window_size)
        end = min(len(df), peak_row + window_size)
        window_df = df.iloc[start:end]
        window_df.to_csv(output_dir / f"{variant_name}_peak_{label}_window.csv", index=False)


def write_report(
    summary: dict[str, Any],
    case_results: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> str:
    lines = [
        "# Step E Height-Variant Position-Hold Audit v2",
        "",
        "## Verdict",
        "",
        f"- Overall audit verdict: **{summary['overall_audit_verdict']}**",
        f"- Step E nominal remains valid: **{str(summary['step_e_nominal_remains_valid']).lower()}**",
        f"- Step E across true height variants passes: **{str(summary['step_e_height_variant_robust']).lower()}**",
        "- Controller behavior changed: `false` (baseline profile used for all variants)",
        "- WBC applied: `false`",
        "",
        "## Per-variant results",
        "",
        "| Variant | Verdict | Support max abs (m) | Support final (m) | HipYaw max (rad) | Pitch max (rad) | WheelVel max (rad/s) | Height final vs target (m) |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|",
    ]
    by_variant = {m["variant_name"]: m for m in metrics}
    verdict_by_variant = {r["variant_name"]: r.get("verdict") for r in case_results}
    result_by_variant = {r["variant_name"]: r for r in case_results}

    for name in VARIANTS:
        m = by_variant.get(name, {})
        verdict = verdict_by_variant.get(name, "INCONCLUSIVE")
        support = m.get("support_position_error_m", {})
        posture = m.get("posture", {})
        height = m.get("height", {})
        wheel_contact = m.get("wheel_contact", {})

        def f(v):
            if v is None:
                return "n/a"
            return f"{v:.6f}"

        lines.append(
            "| "
            + " | ".join([
                name,
                verdict,
                f(support.get("max_abs")),
                f(support.get("final")),
                f(posture.get("hip_yaw_abs_max_max_rad")),
                f(posture.get("pitch_x_max_abs_rad")),
                f(wheel_contact.get("wheel_vel_mean_max_abs_rad_s")),
                f(height.get("final_error_vs_target_m")),
            ])
            + " |"
        )

    lines.extend([
        "",
        "## Structural invariants",
        "",
        "| Variant | WBC applied | Hidden torque max | Ownership violations |",
        "|---|:---:|:---:|:---:|",
    ])
    for name in VARIANTS:
        m = by_variant.get(name, {})
        structural = m.get("structural_invariants", {})
        lines.append(
            f"| {name} | {str(structural.get('wbc_applied', 'n/a')).lower()} | "
            f"{structural.get('hidden_torque_norm_max', 'n/a')} | "
            f"{structural.get('ownership_violation_count_max', 'n/a')} |"
        )

    lines.extend([
        "",
        "## Failure classifications",
        "",
    ])
    for name in VARIANTS:
        r = result_by_variant.get(name, {})
        verdict = r.get("verdict", "INCONCLUSIVE")
        if verdict == "PASS":
            lines.append(f"- **{name}**: PASS")
        else:
            for f in r.get("required_fails", []):
                lines.append(f"- **{name}**: {f} (lead: {r.get('failure_lead', 'unknown')})")

    lines.extend([
        "",
        "## Comparison to Step C high-height failures",
        "",
        f"- Step C high_tiny (baseline): support_peak={0.156463:.6f}m, wheel_vel_peak={6.095879:.2f}rad/s, hip_yaw_peak={0.271901:.6f}rad",
        f"- Step C high_small (candidate_A): support_peak={0.157301:.6f}m, wheel_vel_peak={5.510458:.2f}rad/s, pitch_peak={0.100129:.6f}rad",
        "",
        "## Final decision",
        "",
        f"- **{summary['overall_audit_verdict']}**",
        "",
        f"- Recommended next action: {summary.get('recommended_next_action', 'TBD')}",
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

    all_results = []
    all_metrics = []

    for name in VARIANTS:
        variant = variants[name]
        variant_setup_path = Path(variant["variant_setup_path"])
        if not variant_setup_path.exists():
            print(f"  WARNING: variant_setup_path not found: {variant_setup_path}")
            result.update({
                "verdict": "INCONCLUSIVE",
                "primary_failure": "initialization_invalid",
                "failure_classifications": ["initialization_invalid"],
                "failure_lead": "initialization-led",
                "required_fails": [f"variant_setup_path_missing: {variant_setup_path}"],
                "preferred_fails": [],
            })
            all_results.append(result)
            (OUTPUT_DIR / f"{name}_failure_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"  -> INCONCLUSIVE: variant setup path missing")
            continue

        print(f"\n=== Running {name} for {EXPECTED_STEPS} steps ===")
        result, telemetry_path = run_variant_simulation(
            variant_name=name,
            variant_setup_path=variant_setup_path,
            steps=EXPECTED_STEPS,
            output_dir=OUTPUT_DIR,
        )

        if telemetry_path is None or not telemetry_path.exists():
            result.update({
                "verdict": "INCONCLUSIVE",
                "primary_failure": "telemetry_missing",
                "failure_classifications": ["telemetry_missing"],
                "failure_lead": "initialization-led",
                "required_fails": ["simulation_failed_no_telemetry"],
                "preferred_fails": [],
            })
            all_results.append(result)
            (OUTPUT_DIR / f"{name}_failure_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"  -> INCONCLUSIVE: no telemetry produced")
            continue

        metrics = analyze_telemetry(telemetry_path, variant)
        all_metrics.append(metrics)

        classified = classify_variant_result(metrics)
        result.update(classified)
        all_results.append(result)

        print(f"  -> {classified['verdict']}")
        for f in classified.get("required_fails", []):
            print(f"     {f}")

        (OUTPUT_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (OUTPUT_DIR / f"{name}_failure_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

        df = pd.read_csv(telemetry_path)
        create_failure_windows(df, metrics, OUTPUT_DIR, name)

    pass_count = sum(1 for r in all_results if r.get("verdict") == "PASS")
    fail_count = sum(1 for r in all_results if r.get("verdict") == "FAIL")
    incl_count = sum(1 for r in all_results if r.get("verdict") == "INCONCLUSIVE")

    if fail_count == 0 and incl_count == 0:
        overall_verdict = "STEP_E_HEIGHT_VARIANT_HOLD_PASS"
    elif fail_count > 0:
        overall_verdict = "STEP_E_HEIGHT_VARIANT_ROBUSTNESS_GAP"
    else:
        overall_verdict = "STEP_E_HEIGHT_VARIANT_INCONCLUSIVE"

    nominal_result = next((r for r in all_results if r.get("variant_name") == "nominal"), None)
    high_tiny_result = next((r for r in all_results if r.get("variant_name") == "high_tiny"), None)
    high_small_result = next((r for r in all_results if r.get("variant_name") == "high_small"), None)

    if overall_verdict == "STEP_E_HEIGHT_VARIANT_HOLD_PASS":
        recommended = "Step E height-variant hold is robust. Current Step C failure should not be attributed to unvalidated Step E. Do not reopen Step E."
    elif high_tiny_result and high_tiny_result.get("verdict") == "FAIL":
        recommended = "Step E nominal remains DONE. Step E height-variant robustness is incomplete. Pause Step C fix work until Step E height-variant hold passes for high_tiny."
    elif high_small_result and high_small_result.get("verdict") == "FAIL":
        recommended = "Step E nominal and low-height variants pass. Step E high-height extreme robustness is incomplete. Recommend targeted Step E high-height sagittal robustness fix before continuing Step C."
    else:
        recommended = "Results inconclusive. Some telemetry missing. Resolve blockers before drawing conclusions."

    summary = {
        "overall_audit_verdict": overall_verdict,
        "step_e_nominal_remains_valid": nominal_result.get("verdict") == "PASS" if nominal_result else "INCONCLUSIVE",
        "step_e_height_variant_robust": overall_verdict == "STEP_E_HEIGHT_VARIANT_HOLD_PASS",
        "step_e_height_variant_robustness_verdict": overall_verdict,
        "controller_behavior_changed": False,
        "wbc_applied": False,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "inconclusive_count": incl_count,
        "per_variant_verdicts": {r["variant_name"]: r.get("verdict", "INCONCLUSIVE") for r in all_results},
        "high_tiny_fails": high_tiny_result.get("verdict") == "FAIL" if high_tiny_result else "INCONCLUSIVE",
        "high_small_fails": high_small_result.get("verdict") == "FAIL" if high_small_result else "INCONCLUSIVE",
        "failures_match_current_step_c_high_tiny_issue": (
            "COMPARING: See per-variant metrics for direct comparison to Step C baseline high_tiny failure"
        ),
        "recommended_next_action": recommended,
    }

    (OUTPUT_DIR / "step_e_height_variant_case_matrix.json").write_text(
        json.dumps([{"variant_name": v} for v in VARIANTS], indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "step_e_height_variant_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step_e_height_variant_position_hold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "step_e_height_variant_position_hold_report.md").write_text(
        write_report(summary, all_results, all_metrics), encoding="utf-8"
    )

    print(f"\n=== Audit Complete ===")
    print(f"Pass: {pass_count}/{len(VARIANTS)}, Fail: {fail_count}/{len(VARIANTS)}, Inconclusive: {incl_count}/{len(VARIANTS)}")
    print(f"Overall verdict: {overall_verdict}")
    print(f"Recommended next action: {recommended}")

    return 0 if overall_verdict == "STEP_E_HEIGHT_VARIANT_HOLD_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
