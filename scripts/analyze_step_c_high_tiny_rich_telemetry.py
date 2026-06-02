from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


TERM_COLUMNS = {
    "tau_position": "tau_position_clipped",
    "tau_pitch": "tau_pitch",
    "tau_pitch_rate": "tau_pitch_rate",
    "tau_sagittal_velocity": "tau_sagittal_velocity",
    "tau_support_velocity": "tau_support_velocity",
}

CLASSIFICATION_TO_RECOMMENDATION = {
    "sagittal_pitch_term_drives_wheel_velocity_peak": "add sagittal scheduling for high-height variants",
    "sagittal_position_authority_insufficient_at_high_height": "add sagittal scheduling for high-height variants",
    "sagittal_velocity_damping_insufficient_at_high_height": "add wheel velocity damping for high-height variants",
    "wheel_torque_rate_limit_causes_position_regression": "add sagittal scheduling for high-height variants",
    "hip_yaw_authority_insufficient_at_high_height": "increase hip-yaw authority for high-height variants",
    "hip_yaw_drift_secondary_to_sagittal_regression": "increase hip-yaw authority for high-height variants",
    "height_variant_reference_leak": "add height-variant-specific reference handling",
    "high_height_variant_dynamic_coupling_requires_scheduling": "add sagittal scheduling for high-height variants",
    "unclear_requires_more_telemetry": "collect more telemetry",
}


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([False] * len(df), index=df.index)
    values = df[column]
    if values.dtype == bool:
        return values
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([0.0] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _event(df: pd.DataFrame, column: str) -> dict[str, Any]:
    values = _numeric_series(df, column)
    idx = int(values.abs().idxmax())
    return {
        "row": idx,
        "time_s": float(_numeric_series(df, "time").loc[idx]) if "time" in df else float(idx) * 0.01,
        "value": float(values.loc[idx]),
        "abs_value": float(abs(values.loc[idx])),
    }


def _window(df: pd.DataFrame, center_row: int, radius: int = 50) -> pd.DataFrame:
    start = max(0, center_row - radius)
    end = min(len(df), center_row + radius + 1)
    return df.iloc[start:end].copy()


def _dominant_term(row: pd.Series) -> dict[str, Any]:
    terms = {
        label: float(abs(pd.to_numeric(row.get(column, 0.0), errors="coerce")))
        for label, column in TERM_COLUMNS.items()
    }
    name = max(terms, key=terms.get)
    return {"name": name, "abs_value": terms[name], "terms_abs": terms}


def _term_sign(row: pd.Series, term_column: str, wheel_velocity: float) -> str:
    term = float(pd.to_numeric(row.get(term_column, 0.0), errors="coerce"))
    if abs(term) < 1e-9 or abs(wheel_velocity) < 1e-9:
        return "neutral"
    return "opposes" if term * wheel_velocity < 0.0 else "amplifies"


def _row_bool(row: pd.Series, column: str) -> bool:
    value = row.get(column, False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).lower() in ["true", "1", "yes"]


def _references_consistent(df: pd.DataFrame) -> dict[str, Any]:
    shape_source = str(df.get("shape_posture_reference_source", pd.Series([""])).iloc[0])
    support_source = str(df.get("support_position_reference_source", pd.Series([""])).iloc[0])
    equilibrium_after_variant = bool(_bool_series(df, "equilibrium_capture_after_variant_applied").all())
    support_after_variant = bool(_bool_series(df, "support_reference_captured_after_variant").all())
    height_delta = float(abs(
        _numeric_series(df, "target_com_z_m").iloc[0]
        - _numeric_series(df, "height_variant_achieved_com_z_m").iloc[0]
    ))
    nominal_leak_detected = not (
        "variant" in shape_source
        and "variant" in support_source
        and equilibrium_after_variant
        and support_after_variant
        and height_delta < 1e-3
    )
    return {
        "shape_posture_reference_source": shape_source,
        "support_position_reference_source": support_source,
        "equilibrium_capture_after_variant_applied": equilibrium_after_variant,
        "support_reference_captured_after_variant": support_after_variant,
        "height_target_vs_variant_achieved_abs_delta_m": height_delta,
        "nominal_reference_leak_detected": nominal_leak_detected,
    }


def classify_rich_failure(df: pd.DataFrame, events: dict[str, Any], sagittal: dict[str, Any], hip_yaw: dict[str, Any], references: dict[str, Any]) -> str:
    if references["nominal_reference_leak_detected"]:
        return "height_variant_reference_leak"
    if sagittal["wheel_peak"]["torque_rate_limit_active"]:
        return "wheel_torque_rate_limit_causes_position_regression"
    if sagittal["wheel_peak"]["dominant_term"] == "tau_pitch" and sagittal["wheel_peak"]["wheel_torque_saturates"]:
        return "sagittal_pitch_term_drives_wheel_velocity_peak"
    if sagittal["support_peak"]["tau_position_saturates"] and sagittal["support_peak"]["dominant_term"] != "tau_position":
        return "sagittal_position_authority_insufficient_at_high_height"
    if sagittal["wheel_peak"]["velocity_damping_effect"] != "opposes":
        return "sagittal_velocity_damping_insufficient_at_high_height"
    if hip_yaw["sign_correct_final_500_fraction"] > 0.95 and hip_yaw["saturation_final_500_fraction"] > 0.5:
        return "hip_yaw_authority_insufficient_at_high_height"
    if events["hip_yaw_peak"]["time_s"] > events["support_position_peak"]["time_s"] and hip_yaw["sign_correct_final_500_fraction"] > 0.95:
        return "hip_yaw_drift_secondary_to_sagittal_regression"
    if events["wheel_velocity_peak"]["time_s"] < events["support_position_peak"]["time_s"] < events["hip_yaw_peak"]["time_s"]:
        return "high_height_variant_dynamic_coupling_requires_scheduling"
    return "unclear_requires_more_telemetry"


def analyze_high_tiny_rich_telemetry(csv_path: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = {
        "wheel_velocity_peak": _event(df, "wheel_vel_mean_rad_s"),
        "support_position_peak": _event(df, "support_position_error_m"),
        "hip_yaw_peak": _event(df, "hip_yaw_abs_max"),
        "height_error_peak": _event(df, "height_error_m"),
        "pitch_peak": _event(df, "pitch_x_rad"),
    }

    wheel_row = df.iloc[events["wheel_velocity_peak"]["row"]]
    support_row = df.iloc[events["support_position_peak"]["row"]]
    wheel_dominant = _dominant_term(wheel_row)
    support_dominant = _dominant_term(support_row)
    wheel_velocity = float(wheel_row.get("wheel_vel_mean_rad_s", 0.0))

    sagittal = {
        "wheel_peak": {
            "dominant_term": wheel_dominant["name"],
            "terms_abs": wheel_dominant["terms_abs"],
            "tau_position_saturates": _row_bool(wheel_row, "tau_position_saturation_flag"),
            "wheel_torque_saturates": _row_bool(wheel_row, "wheel_torque_saturation_left") or _row_bool(wheel_row, "wheel_torque_saturation_right"),
            "torque_rate_limit_active": _row_bool(wheel_row, "wheel_torque_rate_limit_active_left") or _row_bool(wheel_row, "wheel_torque_rate_limit_active_right"),
            "velocity_damping_effect": _term_sign(wheel_row, "tau_wheel_velocity_left", wheel_velocity),
        },
        "support_peak": {
            "dominant_term": support_dominant["name"],
            "terms_abs": support_dominant["terms_abs"],
            "tau_position_saturates": _row_bool(support_row, "tau_position_saturation_flag"),
            "wheel_torque_saturates": _row_bool(support_row, "wheel_torque_saturation_left") or _row_bool(support_row, "wheel_torque_saturation_right"),
            "torque_rate_limit_active": _row_bool(support_row, "wheel_torque_rate_limit_active_left") or _row_bool(support_row, "wheel_torque_rate_limit_active_right"),
        },
    }

    final_500 = df.tail(min(500, len(df)))
    sign_correct_left = _bool_series(final_500, "hip_yaw_torque_sign_correct_left")
    sign_correct_right = _bool_series(final_500, "hip_yaw_torque_sign_correct_right")
    sat_left = _bool_series(final_500, "hip_yaw_torque_saturation_flag_left")
    sat_right = _bool_series(final_500, "hip_yaw_torque_saturation_flag_right")
    hip_yaw_error_abs = np.maximum(
        _numeric_series(df, "l_hip_yaw_error").abs(),
        _numeric_series(df, "r_hip_yaw_error").abs(),
    )
    hip_yaw = {
        "sign_correct_at_peak_left": _row_bool(df.iloc[events["hip_yaw_peak"]["row"]], "hip_yaw_torque_sign_correct_left"),
        "sign_correct_at_peak_right": _row_bool(df.iloc[events["hip_yaw_peak"]["row"]], "hip_yaw_torque_sign_correct_right"),
        "sign_correct_final_500_fraction": float((sign_correct_left & sign_correct_right).mean()),
        "saturation_final_500_fraction": float((sat_left | sat_right).mean()),
        "error_final_abs": float(hip_yaw_error_abs.iloc[-1]),
        "error_peak_abs": float(hip_yaw_error_abs.max()),
        "error_grows_after_support_event": events["hip_yaw_peak"]["time_s"] > events["support_position_peak"]["time_s"],
        "drift_likely_secondary": events["hip_yaw_peak"]["time_s"] > events["support_position_peak"]["time_s"],
    }

    references = _references_consistent(df)
    classification = classify_rich_failure(df, events, sagittal, hip_yaw, references)
    recommendation = CLASSIFICATION_TO_RECOMMENDATION[classification]

    sagittal_window = pd.concat([
        _window(df, events["wheel_velocity_peak"]["row"]),
        _window(df, events["support_position_peak"]["row"]),
    ]).drop_duplicates()
    hip_yaw_window = _window(df, events["hip_yaw_peak"]["row"])
    reference_columns = [column for column in [
        "variant_name",
        "height_variant_target_com_z_m",
        "height_variant_achieved_com_z_m",
        "height_variant_root_z_m",
        "height_variant_hip_pitch_ref",
        "height_variant_knee_ref",
        "shape_posture_reference_source",
        "equilibrium_capture_after_variant_applied",
        "target_com_z_m",
        "current_com_z_m",
        "height_error_m",
        "root_z_m",
        "support_center_ref_x",
        "support_center_ref_y",
        "support_center_x",
        "support_center_y",
        "support_position_reference_source",
        "support_reference_captured_after_variant",
    ] if column in df.columns]

    sagittal_window.to_csv(output_dir / "high_tiny_sagittal_terms_peak_window.csv", index=False)
    hip_yaw_window.to_csv(output_dir / "high_tiny_hip_yaw_terms_peak_window.csv", index=False)
    df[reference_columns].head(20).to_csv(output_dir / "high_tiny_reference_consistency.csv", index=False)

    result = {
        "input_csv": str(csv_path),
        "events": events,
        "sagittal_root_cause": sagittal,
        "hip_yaw_root_cause": hip_yaw,
        "reference_consistency": references,
        "classification": classification,
        "recommendation": recommendation,
    }
    (output_dir / "high_tiny_rich_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output_dir / "high_tiny_rich_audit_report.md").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict[str, Any]) -> str:
    order = sorted((name, event["time_s"]) for name, event in result["events"].items())
    lines = [
        "# High Tiny Rich Telemetry Audit",
        "",
        f"Classification: `{result['classification']}`",
        f"Recommendation: `{result['recommendation']}`",
        "",
        "## Event timing",
    ]
    for name, time_s in order:
        event = result["events"][name]
        lines.append(f"- {name}: row {event['row']}, time {time_s:.2f} s, abs {event['abs_value']:.6f}")
    lines.extend([
        "",
        "## Sagittal root-cause summary",
        f"- Wheel peak dominant term: {result['sagittal_root_cause']['wheel_peak']['dominant_term']}",
        f"- Support peak dominant term: {result['sagittal_root_cause']['support_peak']['dominant_term']}",
        f"- Wheel peak velocity damping effect: {result['sagittal_root_cause']['wheel_peak']['velocity_damping_effect']}",
        "",
        "## Hip-yaw root-cause summary",
        f"- Final-window sign-correct fraction: {result['hip_yaw_root_cause']['sign_correct_final_500_fraction']:.3f}",
        f"- Final-window saturation fraction: {result['hip_yaw_root_cause']['saturation_final_500_fraction']:.3f}",
        f"- Drift likely secondary: {result['hip_yaw_root_cause']['drift_likely_secondary']}",
        "",
        "## Reference consistency",
        f"- Nominal reference leak detected: {result['reference_consistency']['nominal_reference_leak_detected']}",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze rich high_tiny Step C telemetry")
    parser.add_argument("--csv", type=Path, default=Path("outputs/step_c_height_recovery_rich/high_tiny_telemetry.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/step_c_high_tiny_rich_audit"))
    args = parser.parse_args()
    analyze_high_tiny_rich_telemetry(args.csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
