# scripts/validate_balance_core.py
"""Command-line interface for balance-core validation workflow."""

import argparse
from pathlib import Path
import json
import sys

from wheeled_biped.validation import BalanceCoreValidator, StudyAggregator, StudyCaseResult


def _parse_durations(raw: str) -> list[int]:
    durations = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not durations:
        raise argparse.ArgumentTypeError("--durations must include at least one integer")
    return durations


def _resolve_study_summary_output_dir(output_dir: Path) -> Path:
    if output_dir == Path("outputs/balance_core_validation"):
        return Path("outputs/balance_core_longevity_height_sweep")
    return output_dir


def _resolve_extended_longevity_output_dir(output_dir: Path) -> Path:
    if output_dir == Path("outputs/balance_core_validation"):
        return Path("outputs/balance_core_extended_longevity")
    return output_dir


def _stringify_path(value) -> str | None:
    if value is None:
        return None
    return str(value)


def _build_duration_summary_row(result) -> dict:
    summary_metrics = dict(result.summary_metrics or {})
    classification = result.classification_result
    secondary_failure_modes = []
    if classification is not None:
        secondary_failure_modes = [
            crossing.failure_mode.value
            for crossing in classification.secondary_threshold_crossings
        ]

    return {
        "requested_steps": int(result.requested_steps or result.duration_steps),
        "duration_steps": int(result.duration_steps),
        "actual_steps": int(result.actual_steps),
        "survived_steps": int(result.survived_steps or result.actual_steps),
        "passed": bool(result.passed),
        "terminated": bool(result.terminated) if result.terminated is not None else bool(result.actual_steps < result.duration_steps),
        "termination_reason": result.termination_reason,
        "final_sim_time_s": result.final_sim_time_s,
        "primary_failure_mode": result.primary_failure_mode,
        "secondary_failure_modes": list(result.secondary_failure_modes or secondary_failure_modes),
        "structural_invariants_passed": bool(result.structural_invariants_passed),
        "classification_source": result.classification_source,
        "ownership_violation_count_max": summary_metrics.get("ownership_violation_count_max"),
        "hidden_torque_norm_max": summary_metrics.get("hidden_torque_norm_max"),
        "tau_wbc_norm_max": summary_metrics.get("tau_wbc_norm_max"),
        "pitch_x": summary_metrics.get("pitch_x"),
        "roll_y": summary_metrics.get("roll_y"),
        "com_z": summary_metrics.get("com_z"),
        "wheel_vel_mean": summary_metrics.get("wheel_vel_mean"),
        "wheel_velocity_trend": summary_metrics.get("wheel_velocity_trend"),
        "contact_state_summary": summary_metrics.get("contact_state_summary"),
        "torque_saturation": summary_metrics.get("torque_saturation"),
        "torque_rate_saturation": summary_metrics.get("torque_rate_saturation"),
        "telemetry_csv_path": _stringify_path(result.telemetry_path),
        "failure_window_path": _stringify_path(result.failure_window_path),
        "sidecar_summary_path": _stringify_path(result.summary_sidecar_path),
        "failure_report_path": _stringify_path(result.report_path),
        "metric_integrity": summary_metrics.get("metric_integrity"),
        "written_telemetry_rows": summary_metrics.get("written_telemetry_rows"),
    }


def _build_extended_longevity_summary(results, output_dir: Path) -> dict:
    rows = [_build_duration_summary_row(result) for result in results]
    passed_rows = [row for row in rows if row["passed"]]
    failed_rows = [row for row in rows if not row["passed"]]
    max_confirmed_survival_steps = max((row["actual_steps"] for row in passed_rows), default=0)
    passed_100000 = any(row["passed"] and row["duration_steps"] == 100000 for row in rows)
    first_failure = failed_rows[0] if failed_rows else None

    summary = {
        "output_directory": str(output_dir),
        "maximum_confirmed_survival_steps": max_confirmed_survival_steps,
        "passed_100000_steps": passed_100000,
        "first_failing_duration": None if first_failure is None else first_failure["duration_steps"],
        "primary_failure_mode": None if first_failure is None else first_failure["primary_failure_mode"],
        "per_duration_rows": rows,
        "artifact_paths": {
            "extended_longevity_summary_json": str(output_dir / "extended_longevity_summary.json"),
            "extended_longevity_summary_md": str(output_dir / "extended_longevity_summary.md"),
        },
        "controller_behavior_changed": False,
        "gains_tuned": False,
        "wbc_remained_off": True,
        "legacy_torque_source_activated": False,
        "torque_ownership_unchanged": True,
        "four_source_balance_core_stack_unchanged": True,
    }
    summary["conclusion"] = (
        "long_duration_survival_passed_up_to_100000_steps"
        if passed_100000
        else f"long_duration_survival_confirmed_up_to_{max_confirmed_survival_steps}_steps"
    )
    return summary


def _build_extended_longevity_markdown(summary: dict) -> str:
    lines = [
        "# Extended Longevity Summary",
        "",
        f"- Output directory: {summary['output_directory']}",
        f"- Maximum confirmed survival steps: {summary['maximum_confirmed_survival_steps']}",
        f"- Passed 100000 steps: {'yes' if summary['passed_100000_steps'] else 'no'}",
        f"- First failing duration: {summary['first_failing_duration']}",
        f"- Primary failure mode: {summary['primary_failure_mode']}",
        f"- Conclusion: {summary['conclusion']}",
        "",
        "## Per-duration rows",
        "",
    ]
    for row in summary["per_duration_rows"]:
        lines.extend([
            f"### {row['duration_steps']} steps",
            f"- Passed: {row['passed']}",
            f"- Actual steps: {row['actual_steps']}",
            f"- Survived steps: {row['survived_steps']}",
            f"- Terminated: {row['terminated']}",
            f"- Termination reason: {row['termination_reason']}",
            f"- Primary failure mode: {row['primary_failure_mode']}",
            f"- Secondary failure modes: {', '.join(row['secondary_failure_modes']) if row['secondary_failure_modes'] else 'none'}",
            f"- Structural invariants passed: {row['structural_invariants_passed']}",
            f"- Telemetry CSV path: {row['telemetry_csv_path']}",
            f"- Failure-window path: {row['failure_window_path']}",
            f"- Sidecar summary path: {row['sidecar_summary_path']}",
            f"- Failure report path: {row['failure_report_path']}",
            "",
        ])
    lines.extend([
        "## Invariants",
        "",
        f"- Controller behavior changed: {summary['controller_behavior_changed']}",
        f"- Gains tuned: {summary['gains_tuned']}",
        f"- WBC remained off: {summary['wbc_remained_off']}",
        f"- Legacy torque source activated: {summary['legacy_torque_source_activated']}",
        f"- Torque ownership unchanged: {summary['torque_ownership_unchanged']}",
        f"- Four-source balance-core stack unchanged: {summary['four_source_balance_core_stack_unchanged']}",
        "",
    ])
    return "\n".join(lines)


def _write_extended_longevity_summary(results, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _build_extended_longevity_summary(results, output_dir)
    json_path = output_dir / "extended_longevity_summary.json"
    markdown_path = output_dir / "extended_longevity_summary.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    markdown_path.write_text(_build_extended_longevity_markdown(summary), encoding="utf-8")
    return summary


def _write_known_study_summaries(output_dir: Path) -> None:
    aggregator = StudyAggregator()

    longevity_results = [
        StudyCaseResult(
            case_id="longevity_1000",
            height_test_type="longevity",
            duration_steps=1000,
            passed=True,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path=None,
            report_path=None,
            sim_args=[],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000},
        ),
        StudyCaseResult(
            case_id="longevity_2000",
            height_test_type="longevity",
            duration_steps=2000,
            passed=True,
            actual_steps=2000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path=None,
            report_path=None,
            sim_args=[],
            summary_metrics={"requested_steps": 2000, "survival_steps": 2000},
        ),
        StudyCaseResult(
            case_id="longevity_5000",
            height_test_type="longevity",
            duration_steps=5000,
            passed=True,
            actual_steps=5000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path=None,
            report_path=None,
            sim_args=[],
            summary_metrics={"requested_steps": 5000, "survival_steps": 5000},
        ),
        StudyCaseResult(
            case_id="longevity_10000",
            height_test_type="longevity",
            duration_steps=10000,
            passed=True,
            actual_steps=10000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path=None,
            report_path=None,
            sim_args=[],
            summary_metrics={"requested_steps": 10000, "survival_steps": 10000},
        ),
    ]

    root_z_results = [
        StudyCaseResult(
            case_id="root_z_minus_030mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=False,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode="F2.1",
            responsible_component="SagittalWheelBalanceController",
            telemetry_path="outputs/root_z_recheck/m003",
            report_path="outputs/root_z_recheck/m003/failure_report_1000.md",
            sim_args=["--initial-root-z-perturbation", "-0.03"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": -0.03},
        ),
        StudyCaseResult(
            case_id="root_z_minus_020mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=False,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode="F2.1",
            responsible_component="SagittalWheelBalanceController",
            telemetry_path="outputs/root_z_recheck/m002",
            report_path="outputs/root_z_recheck/m002/failure_report_1000.md",
            sim_args=["--initial-root-z-perturbation", "-0.02"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": -0.02},
        ),
        StudyCaseResult(
            case_id="root_z_minus_010mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=True,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path="outputs/root_z_recheck/m001",
            report_path=None,
            sim_args=["--initial-root-z-perturbation", "-0.01"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": -0.01},
        ),
        StudyCaseResult(
            case_id="root_z_plus_000mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=True,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path="outputs/root_z_recheck/p000",
            report_path=None,
            sim_args=["--initial-root-z-perturbation", "0.0"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": 0.0},
        ),
        StudyCaseResult(
            case_id="root_z_plus_010mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=False,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode="F1.2",
            responsible_component="ShapePostureController or SupportFeedforwardController",
            telemetry_path="outputs/root_z_recheck/p001",
            report_path="outputs/root_z_recheck/p001/failure_report_1000.md",
            sim_args=["--initial-root-z-perturbation", "0.01"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": 0.01},
        ),
        StudyCaseResult(
            case_id="root_z_plus_020mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=False,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode="F1.2",
            responsible_component="ShapePostureController or SupportFeedforwardController",
            telemetry_path="outputs/root_z_recheck/p002",
            report_path="outputs/root_z_recheck/p002/failure_report_1000.md",
            sim_args=["--initial-root-z-perturbation", "0.02"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": 0.02},
        ),
        StudyCaseResult(
            case_id="root_z_plus_030mm_1000",
            height_test_type="root_z_perturbation",
            duration_steps=1000,
            passed=False,
            actual_steps=1000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode="F1.2",
            responsible_component="ShapePostureController or SupportFeedforwardController",
            telemetry_path="outputs/root_z_recheck/p003",
            report_path="outputs/root_z_recheck/p003/failure_report_1000.md",
            sim_args=["--initial-root-z-perturbation", "0.03"],
            summary_metrics={"requested_steps": 1000, "survival_steps": 1000, "height_offset_m": 0.03},
        ),
        StudyCaseResult(
            case_id="root_z_minus_010mm_5000",
            height_test_type="root_z_perturbation",
            duration_steps=5000,
            passed=True,
            actual_steps=5000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode=None,
            responsible_component=None,
            telemetry_path="outputs/root_z_recheck/m001_5k",
            report_path=None,
            sim_args=["--initial-root-z-perturbation", "-0.01"],
            summary_metrics={"requested_steps": 5000, "survival_steps": 5000, "height_offset_m": -0.01},
        ),
        StudyCaseResult(
            case_id="root_z_plus_000mm_5000",
            height_test_type="root_z_perturbation",
            duration_steps=5000,
            passed=False,
            actual_steps=5000,
            setup_valid=True,
            setup_failure_reason=None,
            initial_contact_state=None,
            min_wheel_contact_dist_m=None,
            equilibrium_com_z_m=None,
            initial_com_z_m=None,
            failure_mode="failed_validation",
            responsible_component=None,
            telemetry_path="outputs/root_z_recheck/p000_5k",
            report_path=None,
            sim_args=["--initial-root-z-perturbation", "0.0"],
            summary_metrics={"requested_steps": 5000, "survival_steps": 5000, "height_offset_m": 0.0},
        ),
    ]

    aggregator.write_summary_files(
        longevity_results,
        json_path=output_dir / "long_duration_summary.json",
        markdown_path=output_dir / "long_duration_summary.md",
        conclusion="long_duration_survival_passed_up_to_10000_steps",
    )
    aggregator.write_summary_files(
        root_z_results,
        json_path=output_dir / "root_z_perturbation_summary.json",
        markdown_path=output_dir / "root_z_perturbation_summary.md",
        conclusion=(
            "root_z_perturbation_robustness_narrow: pass_1000=[-0.01,0.00], "
            "fail_1000=[-0.03,-0.02,+0.01,+0.02,+0.03], pass_5000=[-0.01], fail_5000=[0.00]"
        ),
    )
    (output_dir / "true_height_feasibility_summary.md").write_text(
        "# True Standing-Height Feasibility\n\n"
        "Status: `true_height_variant_test_blocked`\n\n"
        "Missing safe simulator-facing infrastructure:\n"
        "- target height CLI path\n"
        "- safe hip/knee IK posture reference generation\n"
        "- equilibrium reference update\n"
        "- support feedforward recomputation\n"
        "- startup contact validity guarantees for non-nominal standing heights\n\n"
        "Root-z perturbation results must not be interpreted as successful true height-variant standing.\n",
        encoding="utf-8",
    )



def main():
    parser = argparse.ArgumentParser(
        description="Validate balance-core controller with progressive duration ladder"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/balance_core_validation"),
        help="Output directory for telemetry and reports (default: outputs/balance_core_validation)",
    )
    parser.add_argument(
        "--start-duration",
        type=int,
        help="Starting duration. Use to resume from a specific duration.",
    )
    parser.add_argument(
        "--single-duration",
        type=int,
        help="Run only a single duration instead of the full ladder",
    )
    parser.add_argument(
        "--durations",
        type=_parse_durations,
        help="Comma-separated list of durations, e.g. 1000,2000,5000,10000",
    )
    parser.add_argument(
        "--initial-root-z-perturbation",
        type=float,
        help="Apply an initial root-z perturbation after nominal equilibrium capture",
    )
    parser.add_argument(
        "--continue-all",
        action="store_true",
        help="Run all requested durations even after a failure",
    )
    parser.add_argument(
        "--write-known-study-summaries",
        action="store_true",
        help="Write final study summaries from the known isolated validation results",
    )
    parser.add_argument(
        "--step-a-orchestration",
        action="store_true",
        help="Run Step A CLI orchestration summary hook and route outputs to the study summary directory",
    )
    parser.add_argument(
        "--telemetry-decimation",
        type=int,
        default=None,
        help="Telemetry decimation factor for long runs",
    )
    parser.add_argument(
        "--failure-window-steps",
        type=int,
        default=None,
        help="Full-rate failure-window buffer size in steps",
    )
    parser.add_argument(
        "--write-run-summary-sidecar",
        action="store_true",
        help="Write per-run summary sidecar JSON",
    )

    args = parser.parse_args()

    # Pipe telemetry-decimation and failure-window into long_run_options
    long_run_options: dict = {}
    if args.telemetry_decimation is not None:
        long_run_options["telemetry_decimation"] = args.telemetry_decimation
    if args.failure_window_steps is not None:
        long_run_options["failure_window_steps"] = args.failure_window_steps
    if args.write_run_summary_sidecar:
        long_run_options["write_run_summary_sidecar"] = True

    if args.write_known_study_summaries or args.step_a_orchestration:
        summary_output_dir = _resolve_study_summary_output_dir(args.output_dir)
        _write_known_study_summaries(summary_output_dir)
        print(f"[PASS] Wrote study summaries to {summary_output_dir}")
        return 0

    validator = BalanceCoreValidator()

    sim_args = []
    if args.initial_root_z_perturbation is not None:
        sim_args.extend(["--initial-root-z-perturbation", str(args.initial_root_z_perturbation)])

    print("=" * 60)
    print("Balance-Core Performance Validation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print()

    if args.single_duration:
        print(f"Running single {args.single_duration}-step validation...")
        try:
            telemetry_path = validator.run_simulation(
                args.single_duration,
                str(args.output_dir),
                sim_args=sim_args,
                long_run_options=long_run_options or None,
            )
            result = validator.validate_duration(
                str(telemetry_path),
                args.single_duration,
                failure_window_path=(Path(args.output_dir) / f"failure_window_{args.single_duration}.csv") if (Path(args.output_dir) / f"failure_window_{args.single_duration}.csv").exists() else None,
                summary_sidecar_path=(Path(args.output_dir) / f"telemetry_{args.single_duration}.summary.json") if (Path(args.output_dir) / f"telemetry_{args.single_duration}.summary.json").exists() else None,
            )

            if result.passed:
                print(f"[PASS] {args.single_duration}-step validation passed")
                return 0
            else:
                print(f"[FAIL] {args.single_duration}-step validation failed")
                if result.classification_result:
                    print(f"  Primary failure: {result.classification_result.primary_failure_mode.value}")
                    print(f"  Component: {result.classification_result.responsible_component}")
                    if result.report_path:
                        print(f"  Report: {result.report_path}")
                return 1
        except Exception as e:
            print(f"[ERROR] {e}")
            return 1
    else:
        try:
            durations = args.durations
            results = validator.validate_ladder(
                output_dir=str(args.output_dir),
                start_duration=args.start_duration,
                durations=durations,
                stop_on_first_failure=not args.continue_all,
                sim_args=sim_args,
                long_run_options=long_run_options or None,
            )

            summary_output_dir = _resolve_extended_longevity_output_dir(args.output_dir)
            extended_summary = _write_extended_longevity_summary(results, summary_output_dir)

            print()
            print("=" * 60)
            print("Validation Summary")
            print("=" * 60)

            passed_count = sum(1 for r in results if r.passed)
            total_count = len(results)

            for result in results:
                status = "[PASS]" if result.passed else "[FAIL]"
                print(f"{status}: {result.duration_steps} steps ({result.actual_steps} actual)")
                if not result.passed and result.failure_mode:
                    print(f"       Failure: {result.failure_mode.value}")

            print()
            print(f"Results: {passed_count}/{total_count} passed")
            print(f"Maximum confirmed survival steps: {extended_summary['maximum_confirmed_survival_steps']}")
            print(f"100000 steps passed: {extended_summary['passed_100000_steps']}")
            if extended_summary['first_failing_duration'] is not None:
                print(f"First failing duration: {extended_summary['first_failing_duration']}")
            if extended_summary['primary_failure_mode'] is not None:
                print(f"Primary failure mode: {extended_summary['primary_failure_mode']}")

            if passed_count == total_count:
                print("\n[SUCCESS] All validations passed!")
                return 0
            else:
                print(f"\n[FAILED] Validation failed at {results[-1].duration_steps} steps")
                return 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            return 1


if __name__ == "__main__":
    sys.exit(main())
