from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.run_step_c_height_recovery import (
    DEFAULT_STEP_E_TELEMETRY,
    build_simulation_command,
    evaluate_case_telemetry_or_failure,
    resolve_case_target_com_z,
)
from wheeled_biped.validation.step_c_height_recovery import (
    StepCThresholds,
    build_step_c_pass_fail_summary,
    build_step_c_variant_case_matrix,
    compute_height_reference,
    render_step_c_report,
)

OUTPUT_DIR = Path("outputs/step_c_sagittal_schedule_fix")
HEIGHT_VARIANT_SETUP_JSON = Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json")
CANDIDATES = [
    "baseline",
    "candidate_A_position_cap",
    "candidate_B_balanced",
    "candidate_C_stronger_position",
]


def build_candidate_simulation_command(*, candidate: str, steps: int, height_variant_setup: Path) -> list[str]:
    cmd = build_simulation_command(
        steps=steps,
        telemetry_decimation=1,
        failure_window_steps=500,
        height_variant_setup=height_variant_setup,
    )
    cmd.extend(["--vd-sagittal-authority-profile", candidate])
    return cmd


def _snapshot_outputs() -> tuple[set[Path], set[Path]]:
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    existing_csv = set(sim_output_dir.glob("telemetry_*.csv")) if sim_output_dir.exists() else set()
    existing_sidecar = set(sim_output_dir.glob("telemetry_*.summary.json")) if sim_output_dir.exists() else set()
    return existing_csv, existing_sidecar


def _copy_newest_outputs(destination_csv: Path, before_csv: set[Path], before_sidecar: set[Path]) -> Path | None:
    sim_output_dir = Path("outputs/hierarchical_controller_sim")
    current_csv = set(sim_output_dir.glob("telemetry_*.csv")) if sim_output_dir.exists() else set()
    new_csv = current_csv - before_csv
    if not new_csv:
        return None
    source_csv = max(new_csv, key=lambda path: path.stat().st_mtime)
    shutil.copy2(source_csv, destination_csv)

    current_sidecars = set(sim_output_dir.glob("telemetry_*.summary.json")) if sim_output_dir.exists() else set()
    new_sidecars = current_sidecars - before_sidecar
    if new_sidecars:
        source_sidecar = max(new_sidecars, key=lambda path: path.stat().st_mtime)
        shutil.copy2(source_sidecar, destination_csv.with_suffix(".summary.json"))
    return destination_csv


def _run_candidate_case(
    *,
    candidate: str,
    case: dict[str, Any],
    steps: int,
    output_dir: Path,
    reference_target_com_z_m: float,
    thresholds: StepCThresholds,
) -> dict[str, Any]:
    before_csv, before_sidecar = _snapshot_outputs()
    cmd = build_candidate_simulation_command(
        candidate=candidate,
        steps=steps,
        height_variant_setup=Path(case["variant_setup_path"]),
    )
    process_error = None
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        process_error = exc

    telemetry_path = _copy_newest_outputs(
        output_dir / f"{candidate}_{case['case_name']}_{steps}_telemetry.csv",
        before_csv,
        before_sidecar,
    )
    result = evaluate_case_telemetry_or_failure(
        telemetry_path=telemetry_path,
        case_name=case["case_name"],
        target_com_z_m=resolve_case_target_com_z(case, reference_target_com_z_m=reference_target_com_z_m),
        expected_steps=steps,
        thresholds=thresholds,
        process_error=process_error,
        variant_metadata={
            key: case[key]
            for key in [
                "initialization_method",
                "height_variant_name",
                "variant_setup_path",
                "target_com_z_m",
                "achieved_initial_com_z_m",
                "calibrated_root_z_m",
                "hip_pitch_ref",
                "knee_ref",
                "setup_valid",
                "left_wheel_contact",
                "right_wheel_contact",
                "non_wheel_floor_contact_count",
            ]
            if key in case
        },
    )
    result["candidate"] = candidate
    result["steps"] = steps
    result["command"] = cmd
    return result


def _case_by_name(case_matrix: list[dict[str, Any]], case_name: str) -> dict[str, Any]:
    return next(case for case in case_matrix if case["case_name"] == case_name)


def _regression_ok(result: dict[str, Any]) -> bool:
    return result.get("verdict") == "PASS"


def evaluate_candidates(
    *,
    output_dir: Path,
    height_variant_setup_json: Path,
    step_e_telemetry: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = compute_height_reference(pd.read_csv(step_e_telemetry), source_path=str(step_e_telemetry), tail_rows=500)
    case_matrix = build_step_c_variant_case_matrix(height_variant_setup_json)
    thresholds = StepCThresholds()

    high_tiny = _case_by_name(case_matrix, "high_tiny")
    nominal = _case_by_name(case_matrix, "nominal")
    low_tiny = _case_by_name(case_matrix, "low_tiny")
    rows: list[dict[str, Any]] = []
    selected_candidate = None
    selected_full_results: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        high_1000 = _run_candidate_case(
            candidate=candidate,
            case=high_tiny,
            steps=1000,
            output_dir=output_dir,
            reference_target_com_z_m=reference["target_com_z_m"],
            thresholds=thresholds,
        )
        rows.append(high_1000)
        if high_1000["verdict"] != "PASS":
            continue

        high_5000 = _run_candidate_case(
            candidate=candidate,
            case=high_tiny,
            steps=5000,
            output_dir=output_dir,
            reference_target_com_z_m=reference["target_com_z_m"],
            thresholds=thresholds,
        )
        rows.append(high_5000)
        if high_5000["verdict"] != "PASS":
            continue

        nominal_5000 = _run_candidate_case(
            candidate=candidate,
            case=nominal,
            steps=5000,
            output_dir=output_dir,
            reference_target_com_z_m=reference["target_com_z_m"],
            thresholds=thresholds,
        )
        low_tiny_5000 = _run_candidate_case(
            candidate=candidate,
            case=low_tiny,
            steps=5000,
            output_dir=output_dir,
            reference_target_com_z_m=reference["target_com_z_m"],
            thresholds=thresholds,
        )
        rows.extend([nominal_5000, low_tiny_5000])
        if not (_regression_ok(nominal_5000) and _regression_ok(low_tiny_5000)):
            continue

        full_results = []
        for case in case_matrix:
            result = _run_candidate_case(
                candidate=candidate,
                case=case,
                steps=5000,
                output_dir=output_dir,
                reference_target_com_z_m=reference["target_com_z_m"],
                thresholds=thresholds,
            )
            rows.append(result)
            full_results.append(result)
            if result["verdict"] != "PASS":
                break
        if all(result["verdict"] == "PASS" for result in full_results):
            selected_candidate = candidate
            selected_full_results = full_results
            high_telemetry = output_dir / f"{candidate}_high_tiny_5000_telemetry.csv"
            if high_telemetry.exists():
                shutil.copy2(high_telemetry, output_dir / "best_candidate_high_tiny_telemetry.csv")
            break

    summary = build_step_c_pass_fail_summary(
        selected_full_results,
        controller_behavior_changed=selected_candidate not in (None, "baseline"),
    ) if selected_full_results else {
        "overall_step_c_verdict": "FAIL",
        "final_decision": "STEP_C_FIX_REQUIRED",
        "controller_behavior_changed": False,
        "wbc_applied": any(bool(row.get("wbc_applied", False)) for row in rows),
        "step_e_invariants_preserved": False,
        "case_count": 0,
        "passed_cases": [],
        "failed_cases": [],
        "inconclusive_cases": [],
        "failure_classifications": sorted({failure for row in rows for failure in row.get("failure_classifications", [])}),
    }
    payload = {
        "selected_candidate": selected_candidate,
        "summary": summary,
        "results": rows,
    }
    _write_artifacts(output_dir, payload, selected_full_results)
    return payload


def _row_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": row.get("candidate"),
        "case_name": row.get("case_name"),
        "steps": row.get("steps"),
        "verdict": row.get("verdict"),
        "primary_failure": row.get("primary_failure"),
        "failure_classifications": ",".join(row.get("failure_classifications", [])),
        "height_final_abs_error_m": row.get("height_final_abs_error_m"),
        "support_position_error_max_abs_m": row.get("support_position_error_max_abs_m"),
        "hip_yaw_max_abs_rad": row.get("hip_yaw_max_abs_rad"),
        "wheel_vel_mean_max_abs_rad_s": row.get("wheel_vel_mean_max_abs_rad_s"),
        "wbc_applied": row.get("wbc_applied"),
        "hidden_torque_norm_max": row.get("hidden_torque_norm_max"),
        "ownership_violation_count_max": row.get("ownership_violation_count_max"),
        "telemetry_path": row.get("telemetry_path"),
    }


def _write_artifacts(output_dir: Path, payload: dict[str, Any], selected_full_results: list[dict[str, Any]]) -> None:
    rows = payload["results"]
    csv_path = output_dir / "sagittal_schedule_candidate_summary.csv"
    fieldnames = list(_row_for_csv({}).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_row_for_csv(row))

    (output_dir / "sagittal_schedule_candidate_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "sagittal_schedule_fix_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if selected_full_results:
        report = render_step_c_report(
            case_results=selected_full_results,
            summary=payload["summary"],
            artifact_paths={"summary": str(output_dir / "sagittal_schedule_fix_summary.json")},
        )
    else:
        report = "# Sagittal Schedule Fix Report\n\nNo candidate passed all required gates.\n"
    (output_dir / "sagittal_schedule_fix_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Step C sagittal authority schedule candidates")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--height-variant-setup-json", type=Path, default=HEIGHT_VARIANT_SETUP_JSON)
    parser.add_argument("--step-e-telemetry", type=Path, default=DEFAULT_STEP_E_TELEMETRY)
    args = parser.parse_args()
    payload = evaluate_candidates(
        output_dir=args.output_dir,
        height_variant_setup_json=args.height_variant_setup_json,
        step_e_telemetry=args.step_e_telemetry,
    )
    return 0 if payload["summary"]["overall_step_c_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
