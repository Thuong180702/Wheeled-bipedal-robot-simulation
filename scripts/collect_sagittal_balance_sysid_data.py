"""Closed-loop sagittal balance system-identification data collection.

Runs the validated balance-core controller and records time-aligned
sagittal state + torque trajectories for local dynamics identification.

Does NOT use open-loop perturbations. Collects from stable closed-loop runs.

Usage:
    python scripts/collect_sagittal_balance_sysid_data.py \
        --scenario nominal --steps 5000 --output-root outputs
"""

import argparse
import json
from pathlib import Path


def resolve_sysid_output_dir(root: Path) -> Path:
    return root / "sagittal_position_aware_balance" / "sysid"


def build_sysid_run_metadata(
    scenario: str,
    duration_steps: int,
    controller_mode: str,
) -> dict:
    return {
        "collection_mode": "closed_loop",
        "scenario": scenario,
        "duration_steps": duration_steps,
        "controller_mode": controller_mode,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Collect closed-loop sagittal sysid data from balance-core runs"
    )
    parser.add_argument("--scenario", type=str, default="nominal")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output-root", type=str, default="outputs")
    parser.add_argument("--controller-mode", type=str, default="balance-core")
    args = parser.parse_args()

    output_dir = resolve_sysid_output_dir(Path(args.output_root))
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_sysid_run_metadata(
        scenario=args.scenario,
        duration_steps=args.steps,
        controller_mode=args.controller_mode,
    )

    metadata_path = output_dir / "collection_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SYSID] Collection metadata written to {metadata_path}")
    print(f"[SYSID] Scenario: {args.scenario}, Steps: {args.steps}")

    from scripts.validate_balance_core import main as validate_main
    import shutil

    telemetry_dir = output_dir / "telemetry"
    telemetry_dir.mkdir(parents=True, exist_ok=True)

    validate_argv = [
        "collect_sysid",
        "--single-duration", str(args.steps),
        "--output-dir", str(telemetry_dir),
    ]

    import sys
    sys.argv = validate_argv
    validate_main()

    print(f"[SYSID] Closed-loop telemetry collected under {telemetry_dir}")

    collected_files = list(telemetry_dir.glob("telemetry_*.csv"))
    if collected_files:
        latest = max(collected_files, key=lambda p: p.stat().st_mtime)
        dest = output_dir / "latest_telemetry.csv"
        shutil.copy2(latest, dest)
        print(f"[SYSID] Latest telemetry copied to {dest}")
    else:
        print("[SYSID] WARNING: No telemetry CSV found after collection")


if __name__ == "__main__":
    main()
