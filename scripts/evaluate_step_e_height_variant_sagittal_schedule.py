from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_step_e_height_variant_position_hold_v2 import (
    SETUP_DIR,
    SETUP_REPORT,
    VARIANTS,
    analyze_telemetry,
    classify_variant_result,
    load_setup_variants,
    write_report,
)

OUTPUT_DIR = Path("outputs/step_e_height_variant_sagittal_schedule_fix")
SIM_OUTPUT_DIR = Path("outputs/hierarchical_controller_sim")
CANDIDATES = [
    "candidate_D1_support_velocity_light",
    "candidate_D2_wheel_velocity_damping_light",
]
EVAL_SEQUENCE = [
    ("high_small", 1000),
    ("high_small", 5000),
    ("high_tiny", 5000),
    ("nominal", 5000),
    ("low_tiny", 5000),
    ("low_small", 5000),
]


def build_candidate_command(*, variant_setup_path: Path, candidate: str, steps: int) -> list[str]:
    return [
        "python",
        "scripts/simulate_hierarchical_controller.py",
        "--controller-mode",
        "balance-core",
        "--sagittal-controller",
        "velocity-damped",
        "--height-variant-setup",
        str(variant_setup_path).replace("\\", "/"),
        "--steps",
        str(steps),
        "--telemetry-decimation",
        "1",
        "--failure-window-steps",
        "500",
        "--write-run-summary-sidecar",
        "--vd-sagittal-authority-profile",
        candidate,
    ]


def _snapshot_outputs() -> tuple[set[Path], set[Path]]:
    existing_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    existing_sidecar = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    return existing_csv, existing_sidecar


def _copy_newest_outputs(dest_csv: Path, before_csv: set[Path], before_sidecar: set[Path]) -> Path | None:
    current_csv = set(SIM_OUTPUT_DIR.glob("telemetry_*.csv")) if SIM_OUTPUT_DIR.exists() else set()
    new_csv = current_csv - before_csv
    if not new_csv:
        return None
    source_csv = max(new_csv, key=lambda path: path.stat().st_mtime)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, dest_csv)

    current_sidecars = set(SIM_OUTPUT_DIR.glob("telemetry_*.summary.json")) if SIM_OUTPUT_DIR.exists() else set()
    new_sidecars = current_sidecars - before_sidecar
    if new_sidecars:
        source_sidecar = max(new_sidecars, key=lambda path: path.stat().st_mtime)
        shutil.copy2(source_sidecar, dest_csv.with_suffix(".summary.json"))
    return dest_csv


def run_candidate_case(
    *,
    candidate: str,
    variant_name: str,
    steps: int,
    variant: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    before_csv, before_sidecar = _snapshot_outputs()
    variant_setup_path = Path(variant["variant_setup_path"])
    cmd = build_candidate_command(variant_setup_path=variant_setup_path, candidate=candidate, steps=steps)
    process_error = None
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        process_error = exc

    telemetry_path = _copy_newest_outputs(
        output_dir / "candidate_telemetry" / f"{candidate}_{variant_name}_{steps}_telemetry.csv",
        before_csv,
        before_sidecar,
    )
    result: dict[str, Any] = {
        "candidate": candidate,
        "variant_name": variant_name,
        "steps": steps,
        "command": cmd,
        "simulation_returncode": None if process_error is None else process_error.returncode,
        "simulation_error": None if process_error is None else str(process_error),
        "telemetry_path": str(telemetry_path) if telemetry_path else None,
    }
    if telemetry_path is None or not telemetry_path.exists():
        result.update({
            "verdict": "INCONCLUSIVE",
            "failure_classifications": ["telemetry_missing"],
            "failure_lead": "initialization-led",
            "required_fails": ["simulation_failed_no_telemetry"],
            "preferred_fails": [],
        })
        return result

    metrics = analyze_telemetry(telemetry_path, variant)
    classified = classify_variant_result(metrics)
    result.update(classified)
    result["metrics"] = metrics
    return result


def _result_row(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics", {})
    support = metrics.get("support_position_error_m", {})
    posture = metrics.get("posture", {})
    wheel = metrics.get("wheel_contact", {})
    structural = metrics.get("structural_invariants", {})
    return {
        "candidate": row.get("candidate"),
        "variant_name": row.get("variant_name"),
        "steps": row.get("steps"),
        "verdict": row.get("verdict"),
        "failure_lead": row.get("failure_lead"),
        "required_fails": "; ".join(row.get("required_fails", [])),
        "support_max_abs_m": support.get("max_abs"),
        "hip_yaw_max_abs_rad": posture.get("hip_yaw_abs_max_max_rad"),
        "pitch_max_abs_rad": posture.get("pitch_x_max_abs_rad"),
        "wheel_vel_max_abs_rad_s": wheel.get("wheel_vel_mean_max_abs_rad_s"),
        "wbc_applied": structural.get("wbc_applied"),
        "hidden_torque_norm_max": structural.get("hidden_torque_norm_max"),
        "ownership_violation_count_max": structural.get("ownership_violation_count_max"),
        "telemetry_path": row.get("telemetry_path"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(_result_row({}).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_result_row(row))


def _write_report(output_dir: Path, payload: dict[str, Any]) -> None:
    rows = payload["results"]
    lines = [
        "# Step E-HV Sagittal Schedule Fix Report",
        "",
        f"- Selected candidate: `{payload['selected_candidate']}`",
        f"- Final decision: **{payload['final_decision']}**",
        "- Controller behavior changed: `true` only when selected candidate is non-baseline",
        "- WBC remains disabled; hidden torque and ownership are checked per case.",
        "",
        "| Candidate | Variant | Steps | Verdict | Support max abs | HipYaw max | Pitch max | Wheel max | Required fails |",
        "|---|---|---:|:---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        r = _result_row(row)
        lines.append(
            f"| {r['candidate']} | {r['variant_name']} | {r['steps']} | {r['verdict']} | "
            f"{r['support_max_abs_m']} | {r['hip_yaw_max_abs_rad']} | {r['pitch_max_abs_rad']} | "
            f"{r['wheel_vel_max_abs_rad_s']} | {r['required_fails']} |"
        )
    (output_dir / "step_e_hv_sagittal_schedule_fix_report.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_candidates(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    variants = load_setup_variants()
    results: list[dict[str, Any]] = []
    selected_candidate = None
    selected_full_results: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        candidate_results: list[dict[str, Any]] = []
        for variant_name, steps in EVAL_SEQUENCE:
            result = run_candidate_case(
                candidate=candidate,
                variant_name=variant_name,
                steps=steps,
                variant=variants[variant_name],
                output_dir=output_dir,
            )
            results.append(result)
            candidate_results.append(result)
            if result["verdict"] != "PASS":
                break
        if len(candidate_results) == len(EVAL_SEQUENCE) and all(r["verdict"] == "PASS" for r in candidate_results):
            selected_candidate = candidate
            selected_full_results = [r for r in candidate_results if r["steps"] == 5000]
            best_dir = output_dir / "best_candidate_telemetry"
            best_dir.mkdir(parents=True, exist_ok=True)
            for row in selected_full_results:
                telemetry_path = Path(row["telemetry_path"])
                if telemetry_path.exists():
                    shutil.copy2(telemetry_path, best_dir / f"{row['variant_name']}_telemetry.csv")
            break

    if selected_candidate is None:
        final_decision = "STEP_E_HEIGHT_VARIANT_ROBUSTNESS_GAP"
    else:
        final_decision = "STEP_E_HEIGHT_VARIANT_HOLD_PASS"

    payload = {
        "selected_candidate": selected_candidate,
        "final_decision": final_decision,
        "results": results,
        "skipped_candidates": ["candidate_B_balanced", "candidate_C_stronger_position"],
        "skip_reason": "Previous diagnostics showed stronger/pitch-reduced profiles worsened high_tiny; this evaluator stops at smallest passing non-blind candidate family.",
    }
    _write_csv(output_dir / "step_e_hv_sagittal_schedule_candidate_summary.csv", results)
    (output_dir / "step_e_hv_sagittal_schedule_candidate_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "step_e_hv_sagittal_schedule_fix_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(output_dir, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Step E-HV sagittal schedule candidates")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    payload = evaluate_candidates(args.output_dir)
    return 0 if payload["final_decision"] == "STEP_E_HEIGHT_VARIANT_HOLD_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
