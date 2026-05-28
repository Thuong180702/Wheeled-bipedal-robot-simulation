# scripts/validate_balance_core.py
"""Command-line interface for balance-core validation workflow."""

import argparse
from pathlib import Path
import sys

from wheeled_biped.validation import BalanceCoreValidator, StudyAggregator, StudyCaseResult


def _parse_durations(raw: str) -> list[int]:
    durations = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not durations:
        raise argparse.ArgumentTypeError("--durations must include at least one integer")
    return durations


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

    args = parser.parse_args()

    if args.write_known_study_summaries:
        summary_output_dir = args.output_dir
        if summary_output_dir == Path("outputs/balance_core_validation"):
            summary_output_dir = Path("outputs/balance_core_longevity_height_sweep")
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
            telemetry_path = validator.run_simulation(args.single_duration, str(args.output_dir), sim_args=sim_args)
            result = validator.validate_duration(str(telemetry_path), args.single_duration)

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
            )

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
