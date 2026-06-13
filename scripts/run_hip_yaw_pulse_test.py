"""Pulse test controller for measuring hip-yaw kinematic coupling.

Applies controlled torque pulses to hip-yaw joints in common-mode or
divergence-mode patterns to measure kinematic response:
- Common-mode pulse: tau_L = tau_R (symmetric)
- Divergence-mode pulse: tau_L = -tau_R (antisymmetric)

Measures response in:
- Body yaw angle and rate
- Hip-yaw common-mode and divergence-mode
- Roll, pitch, height, contact

Usage:
    python scripts/run_hip_yaw_pulse_test.py \\
        --pulse-mode common \\
        --pulse-magnitude 5.0 \\
        --pulse-duration 20 \\
        --steps 200

Output:
    outputs/hip_yaw_yaw_architecture_audit/isolation/pulse_tests/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def run_pulse_test(
    pulse_mode: str,
    pulse_magnitude: float,
    pulse_duration: int,
    total_steps: int,
    output_dir: str,
):
    """Run a hip-yaw pulse test experiment.

    Args:
        pulse_mode: "common" or "divergence"
        pulse_magnitude: Torque magnitude [Nm]
        pulse_duration: Pulse duration [steps]
        total_steps: Total simulation steps
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Hip-Yaw Pulse Test: {pulse_mode} mode")
    print(f"Pulse magnitude: {pulse_magnitude} Nm")
    print(f"Pulse duration: {pulse_duration} steps")
    print(f"Total steps: {total_steps}")
    print(f"{'='*80}\n")

    print("ERROR: Pulse test requires custom controller implementation")
    print("Current balance-core architecture does not support direct torque injection")
    print("\nRequired implementation:")
    print("1. Add pulse controller that bypasses normal control pipeline")
    print("2. Apply pulses directly to hip-yaw joints during specified window")
    print("3. Measure response with high-frequency telemetry")
    print("\nThis experiment requires architectural changes to simulate_hierarchical_controller.py")

    # Save placeholder results
    result = {
        "pulse_mode": pulse_mode,
        "pulse_magnitude": pulse_magnitude,
        "pulse_duration": pulse_duration,
        "total_steps": total_steps,
        "status": "NOT_IMPLEMENTED",
        "message": "Pulse test requires custom controller implementation not available in current architecture",
        "required_changes": [
            "Add pulse controller to balance-core",
            "Implement direct torque injection bypass",
            "Add high-frequency response telemetry",
        ],
    }

    result_path = output_path / f"pulse_test_{pulse_mode}_mode.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved result: {result_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run hip-yaw pulse test to measure kinematic coupling"
    )
    parser.add_argument(
        "--pulse-mode",
        choices=["common", "divergence"],
        required=True,
        help="Pulse mode: common (tau_L = tau_R) or divergence (tau_L = -tau_R)",
    )
    parser.add_argument(
        "--pulse-magnitude",
        type=float,
        default=5.0,
        help="Pulse torque magnitude [Nm]",
    )
    parser.add_argument(
        "--pulse-duration",
        type=int,
        default=20,
        help="Pulse duration [steps]",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Total simulation steps",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/hip_yaw_yaw_architecture_audit/isolation/pulse_tests",
        help="Output directory",
    )

    args = parser.parse_args()

    run_pulse_test(
        pulse_mode=args.pulse_mode,
        pulse_magnitude=args.pulse_magnitude,
        pulse_duration=args.pulse_duration,
        total_steps=args.steps,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
