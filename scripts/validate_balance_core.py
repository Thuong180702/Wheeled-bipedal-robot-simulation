# scripts/validate_balance_core.py
"""Command-line interface for balance-core validation workflow."""

import argparse
from pathlib import Path
import sys

from wheeled_biped.validation import BalanceCoreValidator
def _parse_durations(raw: str) -> list[int]:
    durations = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not durations:
        raise argparse.ArgumentTypeError("--durations must include at least one integer")
    return durations


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

    args = parser.parse_args()

    validator = BalanceCoreValidator()

    # Build sim_args from explicit flags when provided
    sim_args = []
    if args.initial_root_z_perturbation is not None:
        sim_args.extend(["--initial-root-z-perturbation", str(args.initial_root_z_perturbation)])

    print("=" * 60)
    print("Balance-Core Performance Validation")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print()

    if args.single_duration:
        # Run single duration
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
        # Run duration ladder
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
