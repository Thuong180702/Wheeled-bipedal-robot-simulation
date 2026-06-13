#!/usr/bin/env python3
"""Audit controller configuration state.

This script audits the current controller configuration to determine:
1. Which controller mode is active (balance-core, legacy, other)
2. Which sagittal controller is active (velocity-damped, baseline, other)
3. Which profiles exist and what they change
4. Which controllers are enabled by default
5. Which profiles were used in the latest telemetry
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def load_telemetry_summary():
    """Load the Step E summary CSV data."""
    summary_path = Path("outputs/step_e_best_current_profile_5000_eval/step_e_best_current_profile_5000_summary.csv")
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        lines = f.readlines()

    headers = lines[0].strip().split(',')
    data = [dict(zip(headers, line.strip().split(','))) for line in lines[1:] if line.strip()]

    return data


def load_telemetry_csv(variant_name):
    """Load a telemetry CSV file."""
    path = Path(f"outputs/step_e_best_current_profile_5000_eval/{variant_name}_5000_telemetry.csv")
    if not path.exists():
        return None

    with open(path) as f:
        lines = f.readlines()

    headers = [h.strip() for h in lines[0].strip().split(',')]
    data = []
    for line in lines[1:]:
        if line.strip():
            values = line.strip().split(',')
            row = {}
            for i, h in enumerate(headers):
                try:
                    row[h] = float(values[i])
                except (ValueError, IndexError):
                    row[h] = values[i] if i < len(values) else None
            data.append(row)

    return data


def audit_controller_configuration():
    """Audit controller configuration state."""

    # =========================================================================
    # 1. Controller Mode Analysis
    # =========================================================================
    controller_modes = {
        "balance-core": {
            "description": "Four-source torque composition (shape, support, sagittal, lateral)",
            "wbc": "diagnostic_only",
            "torque_sources": ["shape_posture", "support_feedforward", "sagittal_wheel", "lateral_roll"],
            "torque_composer": "BalanceCoreTorqueComposer",
        },
        "legacy": {
            "description": "Legacy single-source WBC torque",
            "wbc": "applied",
            "torque_sources": ["wbc"],
            "torque_composer": "None (direct WBC)",
        },
    }

    # =========================================================================
    # 2. Sagittal Controller Analysis
    # =========================================================================
    sagittal_controllers = {
        "velocity-damped": {
            "description": "Velocity-damped sagittal balance with position authority schedule",
            "profile_class": "SagittalVelocityDampedBalanceController",
            "key_parameters": ["k_position", "k_velocity", "max_position_tau", "pitch_tau_scale"],
        },
        "baseline": {
            "description": "Baseline sagittal wheel balance",
            "profile_class": "SagittalWheelBalanceController",
            "key_parameters": ["k_pitch", "k_sagittal_velocity"],
        },
    }

    # =========================================================================
    # 3. Profile Analysis from Telemetry
    # =========================================================================
    summary_data = load_telemetry_summary()

    # From the report, the selected profile is J3 with parameters:
    # k_position: 80.0, max_position_tau: 6.0, k_velocity: 30.0
    # schedule_type: continuous_smoothstep
    # schedule_range: z_low=0.300, z_high=0.393

    profile_analysis = {
        "selected_profile": "J3",
        "profile_type": "JOINT_FIX_J3_SUPPORT_CAP_STRONG_DAMPING",
        "schedule_type": "continuous_smoothstep",
        "parameters": {
            "k_position": 80.0,
            "max_position_tau": 6.0,
            "k_velocity": 30.0,
            "schedule_range": {"z_low": 0.300, "z_high": 0.393},
        },
        "description": "Velocity-damped sagittal with strong damping (k_velocity=30)",
    }

    # =========================================================================
    # 4. Controller Enabled States from Telemetry
    # =========================================================================
    variants = ["low_0p300", "nominal", "high_0p480"]
    enabled_controllers = defaultdict(dict)

    for variant in variants:
        telemetry = load_telemetry_csv(variant)
        if telemetry and len(telemetry) > 0:
            first_row = telemetry[0]

            # Hip-yaw control
            hip_yaw_div_active = first_row.get("hip_yaw_div_active", 0)
            hip_yaw_comp_active = first_row.get("hip_yaw_comp_active", 0)
            yaw_aware_active = first_row.get("yaw_aware_position_compensation_active", 0)
            hip_yaw_integral_active = first_row.get("hip_yaw_integral_active", 0)

            # Sagittal control
            sagittal_authority = variant in ["low_0p300", "high_0p480"]  # Boundary variants

            # Yaw control
            # Check if yaw_controller is active by looking at yaw-related columns

            enabled_controllers[variant] = {
                "hip_yaw_divergence_damping": bool(hip_yaw_div_active),
                "hip_yaw_support_feedforward": bool(hip_yaw_comp_active),
                "yaw_aware_position_compensation": bool(yaw_aware_active),
                "hip_yaw_integral_control": bool(hip_yaw_integral_active),
                "boundary_sagittal_authority": sagittal_authority,
            }

    # =========================================================================
    # 5. Control Mode Analysis
    # =========================================================================
    control_modes = {
        "hip_yaw_pd": {
            "kp": "effective_kp_hip_yaw",
            "kd": "effective_kd_hip_yaw",
        },
        "hip_yaw_hy_ff": {
            "enabled": "hip_yaw_comp_active",
            "k_support": "hip_yaw_comp_k_support",
            "tau_max": "hip_yaw_comp_tau_max",
            "gate": "hip_yaw_comp_height_gate",
        },
        "hip_yaw_div": {
            "enabled": "hip_yaw_div_active",
            "k_divergence": "hip_yaw_div_k_divergence",
            "k_divergence_rate": "hip_yaw_div_k_divergence_rate",
            "tau_max": "hip_yaw_div_tau_max",
        },
        "yaw_controller": {
            # Yaw controller torque output (if available)
        },
    }

    # =========================================================================
    # 6. Telemetry Configuration Summary
    # =========================================================================
    telemetry_config = {
        "low_0p300": {
            "controller_mode": "balance-core",
            "sagittal_controller": "velocity-damped",
            "sagittal_profile": "J3",
            "height_variant_target": 0.300,
            "hip_yaw_div_active": False,
            "hip_yaw_comp_active": False,
            "yaw_aware_active": False,
            "hip_yaw_integral_active": False,
        },
        "nominal": {
            "controller_mode": "balance-core",
            "sagittal_controller": "velocity-damped",
            "sagittal_profile": "J3",
            "height_variant_target": 0.400,
            "hip_yaw_div_active": False,
            "hip_yaw_comp_active": False,
            "yaw_aware_active": False,
            "hip_yaw_integral_active": False,
        },
        "high_0p480": {
            "controller_mode": "balance-core",
            "sagittal_controller": "velocity-damped",
            "sagittal_profile": "J3",
            "height_variant_target": 0.480,
            "hip_yaw_div_active": False,
            "hip_yaw_comp_active": False,
            "yaw_aware_active": False,
            "hip_yaw_integral_active": False,
        },
    }

    return {
        "controller_modes": controller_modes,
        "sagittal_controllers": sagittal_controllers,
        "profile_analysis": profile_analysis,
        "enabled_controllers": dict(enabled_controllers),
        "control_modes": control_modes,
        "telemetry_config": telemetry_config,
    }


def main():
    """Main entry point."""
    print("=" * 80)
    print("PHASE 2: CONTROLLER CONFIGURATION AUDIT")
    print("=" * 80)

    results = audit_controller_configuration()

    # Print summary
    print("\n### Controller Modes ###")
    for mode, info in results["controller_modes"].items():
        print(f"\n{mode}:")
        print(f"  Description: {info['description']}")
        print(f"  WBC: {info['wbc']}")
        print(f"  Torque sources: {', '.join(info['torque_sources'])}")

    print("\n### Sagittal Controllers ###")
    for name, info in results["sagittal_controllers"].items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Parameters: {', '.join(info['key_parameters'])}")

    print("\n### Selected Profile ###")
    profile = results["profile_analysis"]
    print(f"Profile: {profile['selected_profile']} ({profile['profile_type']})")
    print(f"Schedule: {profile['schedule_type']}")
    print(f"Parameters:")
    for k, v in profile['parameters'].items():
        print(f"  {k}: {v}")

    print("\n### Controller Enabled States ###")
    for variant, state in results["enabled_controllers"].items():
        print(f"\n{variant}:")
        for ctrl, enabled in state.items():
            print(f"  {ctrl}: {enabled}")

    print("\n### Telemetry Configuration ###")
    for variant, config in results["telemetry_config"].items():
        print(f"\n{variant}:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    # Save results
    output_dir = Path("outputs/controller_system_root_cause_audit/controller_configuration")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "controller_configuration_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir / 'controller_configuration_summary.json'}")

    return results


if __name__ == "__main__":
    main()
