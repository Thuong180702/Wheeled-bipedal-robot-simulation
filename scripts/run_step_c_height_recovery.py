from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from wheeled_biped.validation.step_c_height_recovery import (
    StepCThresholds,
    build_step_c_case_matrix,
    build_step_c_pass_fail_summary,
    build_step_c_variant_case_matrix,
    compute_height_reference,
    evaluate_step_c_case,
    render_step_c_report,
)


DEFAULT_OUTPUT_DIR = Path("outputs/step_c_height_recovery")
DEFAULT_STEP_E_TELEMETRY = Path("outputs/hierarchical_controller_sim/telemetry_1780289121.csv")
SIM_OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")


def build_simulation_command(
    *,
    steps: int,
    telemetry_decimation: int,
    failure_window_steps: int,
    perturbation_m: float | None = None,
    height_variant_setup: Path | None = None,
    vd_sagittal_authority_profile: str | None = None,
) -> list[str]:
    cmd = [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode",
        "balance-core",
        "--sagittal-controller",
        "velocity-damped",
        "--steps",
        str(steps),
    ]
    if height_variant_setup is not None:
        cmd.extend(["--height-variant-setup", str(height_variant_setup).replace("\\", "/")])
    else:
        cmd.extend(["--initial-root-z-perturbation", str(0.0 if perturbation_m is None else perturbation_m)])
    if vd_sagittal_authority_profile is not None:
        cmd.extend(["--vd-sagittal-authority-profile", vd_sagittal_authority_profile])
    cmd.extend([
        "--telemetry-decimation",
        str(telemetry_decimation),
        "--failure-window-steps",
        str(failure_window_steps),
        "--write-run-summary-sidecar",
    ])
    return cmd


def should_stop_after_case(case_result: dict) -> bool:
    return case_result.get("verdict") != "PASS"


def filter_case_matrix(case_matrix: list[dict], *, case_name: str | None) -> list[dict]:
    if case_name is None:
        return case_matrix
    filtered = [case for case in case_matrix if case["case_name"] == case_name]
    if not filtered:
        available = [case["case_name"] for case in case_matrix]
        raise ValueError(f"Requested Step C case '{case_name}' not found. Available cases: {available}")
    return filtered


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
        shutil.copy2(source_sidecar, output_dir / f"{case_name}_summary.json")
    return dest_csv


def resolve_case_target_com_z(case: dict, *, reference_target_com_z_m: float) -> float:
    if "achieved_initial_com_z_m" in case:
        return float(case["achieved_initial_com_z_m"])
    return reference_target_com_z_m


def evaluate_case_telemetry_or_failure(
    *,
    telemetry_path: Path | None,
    case_name: str,
    target_com_z_m: float,
    expected_steps: int,
    thresholds: StepCThresholds,
    process_error: subprocess.CalledProcessError | None,
    variant_metadata: dict | None = None,
) -> dict:
    if telemetry_path is None or not telemetry_path.exists():
        result = {
            "case_name": case_name,
            "verdict": "INCONCLUSIVE",
            "primary_failure": "unclear_requires_more_telemetry",
            "failure_classifications": ["unclear_requires_more_telemetry", "simulation_failed"],
            "telemetry_path": None,
            "simulation_returncode": None if process_error is None else process_error.returncode,
            "simulation_error": None if process_error is None else str(process_error),
            "wbc_applied": False,
            "step_e_invariants_preserved": False,
        }
        if variant_metadata:
            result.update(variant_metadata)
        return result

    df = pd.read_csv(telemetry_path)
    result = evaluate_step_c_case(
        df,
        case_name=case_name,
        target_com_z_m=target_com_z_m,
        expected_steps=expected_steps,
        thresholds=thresholds,
        simulation_returncode=None if process_error is None else process_error.returncode,
        simulation_error=None if process_error is None else str(process_error),
    )
    result["telemetry_path"] = str(telemetry_path)
    if variant_metadata:
        result.update(variant_metadata)
    return result


def run_case(
    case: dict,
    *,
    output_dir: Path,
    target_com_z_m: float,
    steps: int,
    thresholds: StepCThresholds,
    vd_sagittal_authority_profile: str | None = None,
) -> dict:
    before_csv, before_sidecar = _snapshot_outputs()
    if case.get("initialization_method") == "step_b_true_height_variant":
        cmd = build_simulation_command(
            steps=steps,
            height_variant_setup=Path(case["variant_setup_path"]),
            telemetry_decimation=1,
            failure_window_steps=500,
            vd_sagittal_authority_profile=vd_sagittal_authority_profile,
        )
    else:
        cmd = build_simulation_command(
            steps=steps,
            perturbation_m=float(case["initial_root_z_perturbation_m"]),
            telemetry_decimation=1,
            failure_window_steps=500,
            vd_sagittal_authority_profile=vd_sagittal_authority_profile,
        )
    process_error = None
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        process_error = exc

    telemetry_path = _copy_newest_outputs(case["case_name"], output_dir, before_csv, before_sidecar)
    result = evaluate_case_telemetry_or_failure(
        telemetry_path=telemetry_path,
        case_name=case["case_name"],
        target_com_z_m=resolve_case_target_com_z(case, reference_target_com_z_m=target_com_z_m),
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
    result["command"] = cmd
    return result


def write_artifacts(output_dir: Path, case_results: list[dict], reference: dict, case_matrix: list[dict]) -> dict[str, Path]:
    reference_path = output_dir / "step_c_height_reference.json"
    reference_path.write_text(json.dumps(reference, indent=2), encoding="utf-8")

    case_matrix_path = output_dir / "step_c_height_case_matrix.json"
    case_matrix_path.write_text(json.dumps(case_matrix, indent=2), encoding="utf-8")

    metrics_path = output_dir / "step_c_height_recovery_metrics.json"
    metrics_path.write_text(json.dumps(case_results, indent=2), encoding="utf-8")

    failure_payload = {result["case_name"]: result.get("failure_classifications", []) for result in case_results}
    failure_path = output_dir / "step_c_failure_classification.json"
    failure_path.write_text(json.dumps(failure_payload, indent=2), encoding="utf-8")

    summary = build_step_c_pass_fail_summary(case_results, controller_behavior_changed=False)
    summary_path = output_dir / "step_c_pass_fail_summary.json"
    summary["artifact_paths"] = {
        "height_reference": str(reference_path),
        "case_matrix": str(case_matrix_path),
        "metrics": str(metrics_path),
        "failure_classification": str(failure_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = output_dir / "step_c_height_recovery_report.md"
    report = render_step_c_report(
        case_results=case_results,
        summary=summary,
        artifact_paths={**summary["artifact_paths"], "summary": str(summary_path), "report": str(report_path)},
    )
    report_path.write_text(report, encoding="utf-8")

    return {
        "height_reference": reference_path,
        "case_matrix": case_matrix_path,
        "metrics": metrics_path,
        "failure_classification": failure_path,
        "summary": summary_path,
        "report": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Step C height recovery diagnostic sweep")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--step-e-telemetry", type=Path, default=DEFAULT_STEP_E_TELEMETRY)
    parser.add_argument("--continue-after-failure", action="store_true")
    parser.add_argument("--case", type=str, default=None, help="Run only one Step C case by case_name")
    parser.add_argument("--use-height-variants", action="store_true")
    parser.add_argument(
        "--height-variant-setup-json",
        type=Path,
        default=Path("outputs/balance_core_true_height_variants/true_height_variant_setup_report.json"),
    )
    parser.add_argument(
        "--vd-sagittal-authority-profile",
        type=str,
        default=None,
        help="Height-variant-aware sagittal authority profile. Default: None (baseline). Use 'candidate_D2_wheel_velocity_damping_light' for Step E-HV fix.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    step_e_df = pd.read_csv(args.step_e_telemetry)
    reference = compute_height_reference(step_e_df, source_path=str(args.step_e_telemetry), tail_rows=500)
    case_matrix = (
        build_step_c_variant_case_matrix(args.height_variant_setup_json)
        if args.use_height_variants
        else build_step_c_case_matrix()
    )
    case_matrix = filter_case_matrix(case_matrix, case_name=args.case)

    thresholds = StepCThresholds()
    case_results = []
    for case in case_matrix:
        result = run_case(
            case,
            output_dir=args.output_dir,
            target_com_z_m=reference["target_com_z_m"],
            steps=args.steps,
            thresholds=thresholds,
            vd_sagittal_authority_profile=args.vd_sagittal_authority_profile,
        )
        case_results.append(result)
        write_artifacts(args.output_dir, case_results, reference, case_matrix)
        if should_stop_after_case(result) and not args.continue_after_failure:
            break

    summary_path = args.output_dir / "step_c_pass_fail_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return 0 if summary["overall_step_c_verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
