#!/usr/bin/env python3
"""Tests for Step E WBC gate false-positive fix.

This test validates that the WBC gate uses per_actuator_wbc_authority_enabled
instead of tau_wbc_norm for determining if WBC is "applied".
"""

import pytest
import json
from pathlib import Path


def test_wbc_gate_uses_authority_flag_not_tau_norm():
    """Test that wbc_applied is False when per_actuator_wbc_authority_enabled=False
    even when tau_wbc_norm is nonzero (structural QP output)."""
    from scripts.analyze_step_e_extreme_height_d2_official_check import analyze_telemetry

    # Create synthetic telemetry with:
    # - tau_wbc_norm nonzero (structural QP output)
    # - per_actuator_wbc_authority_enabled = False
    # - active_torque_owner_per_joint = support_feedforward only

    import csv
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time', 'source_step_index', 'terminated', 'termination_reason',
            'support_center_x', 'support_center_ref_x',
            'support_center_y', 'support_center_ref_y',
            'wheel_vel_left_rad_s', 'wheel_vel_right_rad_s',
            'l_hip_yaw_error', 'r_hip_yaw_error',
            'contact_force_valid', 'left_wheel_contact', 'right_wheel_contact',
            'non_wheel_floor_contacts', 'hidden_torque_norm',
            'ownership_violation_count', 'tau_wbc_norm',
            'per_actuator_wbc_authority_enabled', 'active_torque_owner_per_joint',
            'com_z', 'euler_pitch_y', 'euler_roll_x', 'controller_mode'
        ])
        writer.writeheader()

        # Row with structural QP output but NO active WBC
        writer.writerow({
            'time': '0.0',
            'source_step_index': '0',
            'terminated': 'False',
            'termination_reason': '',
            'support_center_x': '0.0',
            'support_center_ref_x': '0.0',
            'support_center_y': '0.0',
            'support_center_ref_y': '0.0',
            'wheel_vel_left_rad_s': '0.0',
            'wheel_vel_right_rad_s': '0.0',
            'l_hip_yaw_error': '0.0',
            'r_hip_yaw_error': '0.0',
            'contact_force_valid': 'True',
            'left_wheel_contact': 'True',
            'right_wheel_contact': 'True',
            'non_wheel_floor_contacts': '0',
            'hidden_torque_norm': '0.0',
            'ownership_violation_count': '0',
            'tau_wbc_norm': '13.5',  # Nonzero structural QP output
            'per_actuator_wbc_authority_enabled': 'False',  # NO active WBC
            'active_torque_owner_per_joint': 'none,none,support_feedforward,support_feedforward,none,none,none,support_feedforward,support_feedforward,none',
            'com_z': '0.30',
            'euler_pitch_y': '0.0',
            'euler_roll_x': '0.0',
            'controller_mode': 'balance-core'
        })

        csv_path = f.name

    try:
        result = analyze_telemetry('test_case', csv_path, target_com_z=0.30)

        # Key assertions
        wbc_gate = result['official_step_e']['wbc_applied']

        # WBC gate should PASS (wbc_applied should be False)
        assert wbc_gate['pass'] is True, "WBC gate should PASS when no active WBC authority"

        # wbc_applied value should be False
        assert wbc_gate['value'] is False, "wbc_applied should be False when per_actuator_wbc_authority_enabled=False"

        # structural_qp_tau_norm should still be reported (nonzero)
        assert wbc_gate['structural_qp_tau_norm'] > 0, "structural_qp_tau_norm should be nonzero"

        # per_actuator_wbc_authority_enabled should be False
        assert wbc_gate['per_actuator_wbc_authority_enabled'] is False

        # active_wbc_owners_detected should be False (no WBC owners)
        assert wbc_gate['active_wbc_owners_detected'] is False

        print("Test PASSED: WBC gate correctly identifies structural QP output as NOT active WBC")

    finally:
        Path(csv_path).unlink()


def test_wbc_gate_fails_when_authority_enabled():
    """Test that wbc_applied is True when per_actuator_wbc_authority_enabled=True."""
    import csv
    import tempfile
    from scripts.analyze_step_e_extreme_height_d2_official_check import analyze_telemetry

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time', 'source_step_index', 'terminated', 'termination_reason',
            'support_center_x', 'support_center_ref_x',
            'support_center_y', 'support_center_ref_y',
            'wheel_vel_left_rad_s', 'wheel_vel_right_rad_s',
            'l_hip_yaw_error', 'r_hip_yaw_error',
            'contact_force_valid', 'left_wheel_contact', 'right_wheel_contact',
            'non_wheel_floor_contacts', 'hidden_torque_norm',
            'ownership_violation_count', 'tau_wbc_norm',
            'per_actuator_wbc_authority_enabled', 'active_torque_owner_per_joint',
            'com_z', 'euler_pitch_y', 'euler_roll_x', 'controller_mode'
        ])
        writer.writeheader()

        # Row with active WBC authority enabled
        writer.writerow({
            'time': '0.0',
            'source_step_index': '0',
            'terminated': 'False',
            'termination_reason': '',
            'support_center_x': '0.0',
            'support_center_ref_x': '0.0',
            'support_center_y': '0.0',
            'support_center_ref_y': '0.0',
            'wheel_vel_left_rad_s': '0.0',
            'wheel_vel_right_rad_s': '0.0',
            'l_hip_yaw_error': '0.0',
            'r_hip_yaw_error': '0.0',
            'contact_force_valid': 'True',
            'left_wheel_contact': 'True',
            'right_wheel_contact': 'True',
            'non_wheel_floor_contacts': '0',
            'hidden_torque_norm': '0.0',
            'ownership_violation_count': '0',
            'tau_wbc_norm': '15.0',
            'per_actuator_wbc_authority_enabled': 'True',  # ACTIVE WBC
            'active_torque_owner_per_joint': 'wbc,wbc,wbc,wbc,wbc,wbc,wbc,wbc,wbc,wbc',
            'com_z': '0.40',
            'euler_pitch_y': '0.0',
            'euler_roll_x': '0.0',
            'controller_mode': 'balance-core'
        })

        csv_path = f.name

    try:
        result = analyze_telemetry('test_case', csv_path, target_com_z=0.40)

        wbc_gate = result['official_step_e']['wbc_applied']

        # WBC gate should FAIL (wbc_applied should be True)
        assert wbc_gate['pass'] is False, "WBC gate should FAIL when active WBC authority"
        assert wbc_gate['value'] is True, "wbc_applied should be True when per_actuator_wbc_authority_enabled=True"

        print("Test PASSED: WBC gate correctly identifies active WBC authority")

    finally:
        Path(csv_path).unlink()


def test_wbc_gate_detects_wbc_owners():
    """Test that wbc_applied is True when active_torque_owner_per_joint includes WBC owners."""
    import csv
    import tempfile
    from scripts.analyze_step_e_extreme_height_d2_official_check import analyze_telemetry

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'time', 'source_step_index', 'terminated', 'termination_reason',
            'support_center_x', 'support_center_ref_x',
            'support_center_y', 'support_center_ref_y',
            'wheel_vel_left_rad_s', 'wheel_vel_right_rad_s',
            'l_hip_yaw_error', 'r_hip_yaw_error',
            'contact_force_valid', 'left_wheel_contact', 'right_wheel_contact',
            'non_wheel_floor_contacts', 'hidden_torque_norm',
            'ownership_violation_count', 'tau_wbc_norm',
            'per_actuator_wbc_authority_enabled', 'active_torque_owner_per_joint',
            'com_z', 'euler_pitch_y', 'euler_roll_x', 'controller_mode'
        ])
        writer.writeheader()

        # Row with WBC owner but no per_actuator flag (edge case detection)
        writer.writerow({
            'time': '0.0',
            'source_step_index': '0',
            'terminated': 'False',
            'termination_reason': '',
            'support_center_x': '0.0',
            'support_center_ref_x': '0.0',
            'support_center_y': '0.0',
            'support_center_ref_y': '0.0',
            'wheel_vel_left_rad_s': '0.0',
            'wheel_vel_right_rad_s': '0.0',
            'l_hip_yaw_error': '0.0',
            'r_hip_yaw_error': '0.0',
            'contact_force_valid': 'True',
            'left_wheel_contact': 'True',
            'right_wheel_contact': 'True',
            'non_wheel_floor_contacts': '0',
            'hidden_torque_norm': '0.0',
            'ownership_violation_count': '0',
            'tau_wbc_norm': '10.0',
            'per_actuator_wbc_authority_enabled': 'False',
            'active_torque_owner_per_joint': 'wbc,wbc,wbc,wbc,wbc,wbc,wbc,wbc,wbc,wbc',  # WBC owner detected
            'com_z': '0.40',
            'euler_pitch_y': '0.0',
            'euler_roll_x': '0.0',
            'controller_mode': 'balance-core'
        })

        csv_path = f.name

    try:
        result = analyze_telemetry('test_case', csv_path, target_com_z=0.40)

        wbc_gate = result['official_step_e']['wbc_applied']

        # WBC gate should FAIL (active_wbc_owners_detected should be True)
        assert wbc_gate['pass'] is False, "WBC gate should FAIL when WBC owners detected"
        assert wbc_gate['active_wbc_owners_detected'] is True, "Should detect WBC owners"

        print("Test PASSED: WBC gate correctly detects WBC owners in ownership")

    finally:
        Path(csv_path).unlink()


def test_extreme_height_telemetry_wbc_gate():
    """Test the actual extreme-height telemetry against the fixed WBC gate."""
    from scripts.analyze_step_e_extreme_height_d2_official_check import analyze_telemetry

    low_path = Path('outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv')
    high_path = Path('outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv')

    if not low_path.exists() or not high_path.exists():
        pytest.skip("Extreme height telemetry not available")

    # Test low_0p300
    low_result = analyze_telemetry('low_0p300', str(low_path), target_com_z=0.300)
    low_wbc_gate = low_result['official_step_e']['wbc_applied']

    print(f"\nlow_0p300 WBC gate:")
    print(f"  wbc_applied: {low_wbc_gate['value']}")
    print(f"  structural_qp_tau_norm: {low_wbc_gate['structural_qp_tau_norm']:.4f}")
    print(f"  per_actuator_wbc_authority_enabled: {low_wbc_gate['per_actuator_wbc_authority_enabled']}")
    print(f"  active_wbc_owners_detected: {low_wbc_gate['active_wbc_owners_detected']}")
    print(f"  gate PASS: {low_wbc_gate['pass']}")

    assert low_wbc_gate['pass'] is True, "low_0p300 WBC gate should PASS"
    assert low_wbc_gate['value'] is False, "low_0p300 should have no active WBC"
    assert low_wbc_gate['structural_qp_tau_norm'] > 0, "structural_qp_tau_norm should be nonzero"

    # Test high_0p480
    high_result = analyze_telemetry('high_0p480', str(high_path), target_com_z=0.480)
    high_wbc_gate = high_result['official_step_e']['wbc_applied']

    print(f"\nhigh_0p480 WBC gate:")
    print(f"  wbc_applied: {high_wbc_gate['value']}")
    print(f"  structural_qp_tau_norm: {high_wbc_gate['structural_qp_tau_norm']:.4f}")
    print(f"  per_actuator_wbc_authority_enabled: {high_wbc_gate['per_actuator_wbc_authority_enabled']}")
    print(f"  active_wbc_owners_detected: {high_wbc_gate['active_wbc_owners_detected']}")
    print(f"  gate PASS: {high_wbc_gate['pass']}")

    assert high_wbc_gate['pass'] is True, "high_0p480 WBC gate should PASS"
    assert high_wbc_gate['value'] is False, "high_0p480 should have no active WBC"
    assert high_wbc_gate['structural_qp_tau_norm'] > 0, "structural_qp_tau_norm should be nonzero"

    print("\nTest PASSED: Extreme height telemetry correctly passes WBC gate")


if __name__ == '__main__':
    test_wbc_gate_uses_authority_flag_not_tau_norm()
    test_wbc_gate_fails_when_authority_enabled()
    test_wbc_gate_detects_wbc_owners()
    test_extreme_height_telemetry_wbc_gate()
    print("\nAll tests passed!")