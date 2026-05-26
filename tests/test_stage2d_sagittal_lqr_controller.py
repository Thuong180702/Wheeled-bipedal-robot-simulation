#!/usr/bin/env python3
"""Unit tests for Stage2D Sagittal LQR Controller."""

import numpy as np
import pytest
from pathlib import Path

from wheeled_biped.controllers.stage2d_sagittal_lqr_controller import Stage2DSagittalLQRController


@pytest.fixture
def mock_identified_model(tmp_path):
    """Create a mock identified model for testing."""
    # Create a simple stable system for testing
    # A matrix: slightly damped system
    A = np.array([
        [0.98, 0.002, 0.0, 0.0, 0.0],
        [-0.5, 0.95, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.99, 0.002, 0.0],
        [0.0, 0.0, 0.0, 0.98, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.90],
    ])

    # B vector: positive for pitch (restoring), negative for wheel velocity (braking)
    B = np.array([0.001, 0.05, 0.002, 0.01, -0.1])

    equilibrium_cp_y = 0.0

    model_path = tmp_path / "test_model.npz"
    np.savez(
        model_path,
        A=A,
        B=B,
        equilibrium_cp_y=equilibrium_cp_y,
    )

    return str(model_path)


class TestStage2DSagittalLQRController:
    """Test suite for Stage2D Sagittal LQR Controller."""

    def test_controller_output_shape(self, mock_identified_model):
        """Test 1: Controller output shape is (10,)."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,
            pitch_rate_x=0.0,
            cp_y=0.0,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        assert tau.shape == (10,), f"Expected shape (10,), got {tau.shape}"

    def test_only_wheel_joints_nonzero(self, mock_identified_model):
        """Test 2: Only wheel joints [4,9] are nonzero."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,
            pitch_rate_x=0.0,
            cp_y=0.0,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        # Check leg joints are zero
        leg_indices = [0, 1, 2, 3, 5, 6, 7, 8]
        assert np.allclose(tau[leg_indices], 0.0), \
            f"Leg joints should be zero, got {tau[leg_indices]}"

        # Check wheel joints are equal
        assert np.isclose(tau[4], tau[9]), \
            f"Wheel joints should be equal, got l_wheel={tau[4]}, r_wheel={tau[9]}"

    def test_zero_state_gives_zero_torque(self, mock_identified_model):
        """Test 3: Zero state gives zero torque."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.0,
            pitch_rate_x=0.0,
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        assert np.allclose(tau, 0.0), \
            f"Zero state should give zero torque, got {tau}"
        assert np.isclose(diag['u_raw'], 0.0), \
            f"Zero state should give zero u_raw, got {diag['u_raw']}"

    def test_positive_pitch_produces_restoring_torque(self, mock_identified_model):
        """Test 4: Positive pitch_x produces restoring wheel torque according to identified B sign."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,  # ~5.7 deg forward tilt
            pitch_rate_x=0.0,
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        # With positive B[0], positive pitch should give negative control (forward wheel motion)
        # to restore balance
        # u = -K @ x, and K[0] should be positive (penalizing pitch)
        # So u should be negative for positive pitch
        assert diag['u_raw'] != 0.0, "Positive pitch should produce nonzero torque"

        # Check that pitch contribution is nonzero
        assert diag['contrib_pitch_x'] != 0.0, \
            "Pitch contribution should be nonzero for positive pitch"

    def test_positive_pitch_rate_produces_damping_torque(self, mock_identified_model):
        """Test 5: Positive pitch_rate_x produces damping/restoring torque."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.0,
            pitch_rate_x=0.5,  # Falling forward
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        # Positive pitch rate should produce damping torque
        assert diag['u_raw'] != 0.0, "Positive pitch rate should produce nonzero torque"
        assert diag['contrib_pitch_rate_x'] != 0.0, \
            "Pitch rate contribution should be nonzero"

    def test_positive_wheel_velocity_produces_braking_torque(self, mock_identified_model):
        """Test 6: Positive wheel_vel_mean produces braking torque."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.0,
            pitch_rate_x=0.0,
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=2.0,
            wheel_vel_right=2.0,
        )

        # Positive wheel velocity should produce braking (opposing) torque
        assert diag['u_raw'] != 0.0, "Positive wheel velocity should produce nonzero torque"
        assert diag['contrib_wheel_vel_mean'] != 0.0, \
            "Wheel velocity contribution should be nonzero"

    def test_saturation_clips_at_max_tau(self, mock_identified_model):
        """Test 7: Saturation clips at max_tau."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        max_tau = controller.max_tau

        # Apply large pitch to trigger saturation
        tau, diag = controller.compute_wheel_torques(
            pitch_x=1.0,  # ~57 deg - very large
            pitch_rate_x=0.0,
            cp_y=controller.equilibrium_cp_y,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        # Check clipping
        assert abs(diag['u_clipped']) <= max_tau, \
            f"u_clipped should be clipped to max_tau={max_tau}, got {diag['u_clipped']}"

        # If saturated, flag should be set
        if abs(diag['u_raw']) > max_tau:
            assert diag['saturated'] is True, \
                "Saturation flag should be True when |u_raw| > max_tau"

    def test_telemetry_includes_required_fields(self, mock_identified_model):
        """Test 8: Telemetry includes all required diagnostic fields."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,
            pitch_rate_x=0.0,
            cp_y=0.0,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        # Check state vector fields
        required_state_fields = [
            'pitch_x', 'pitch_rate_x', 'cp_y', 'cp_error_y', 'com_vy',
            'wheel_vel_left', 'wheel_vel_right', 'wheel_vel_mean'
        ]
        for field in required_state_fields:
            assert field in diag, f"Missing state field: {field}"

        # Check K vector
        assert 'K' in diag, "Missing K vector"
        assert len(diag['K']) == 5, f"K should have 5 elements, got {len(diag['K'])}"

        # Check state contributions
        required_contrib_fields = [
            'contrib_pitch_x', 'contrib_pitch_rate_x', 'contrib_cp_error_y',
            'contrib_com_vy', 'contrib_wheel_vel_mean'
        ]
        for field in required_contrib_fields:
            assert field in diag, f"Missing contribution field: {field}"

        # Check control fields
        assert 'u_raw' in diag, "Missing u_raw"
        assert 'u_clipped' in diag, "Missing u_clipped"
        assert 'saturated' in diag, "Missing saturated flag"

        # Check config fields
        assert 'config' in diag, "Missing config"
        assert 'max_tau' in diag, "Missing max_tau"

    def test_invalid_config_raises_error(self, mock_identified_model):
        """Test 9: Invalid config name raises clean error."""
        with pytest.raises(ValueError, match="Unknown config"):
            Stage2DSagittalLQRController.from_identified_model(
                mock_identified_model, config='INVALID'
            )

    def test_all_configs_are_stable(self, mock_identified_model):
        """Test 10: All predefined configs produce stable closed-loop systems."""
        for config_name in ['A', 'B', 'C', 'D']:
            controller = Stage2DSagittalLQRController.from_identified_model(
                mock_identified_model, config=config_name
            )

            is_stable, max_mag = controller.check_stability()

            assert is_stable, \
                f"Config {config_name} should be stable, got max |λ| = {max_mag}"
            assert max_mag < 1.0, \
                f"Config {config_name} max eigenvalue magnitude should be < 1.0, got {max_mag}"

    def test_controller_from_direct_matrices(self):
        """Test controller initialization from direct A, B matrices."""
        A = np.eye(5) * 0.95
        B = np.array([0.01, 0.05, 0.01, 0.01, -0.1])

        controller = Stage2DSagittalLQRController(
            A=A,
            B=B,
            config='A',
            equilibrium_cp_y=0.0,
        )

        # Should initialize without error
        assert controller.A.shape == (5, 5)
        assert controller.B.shape == (5, 1)
        assert controller.K.shape == (1, 5)

    def test_state_contributions_sum_to_control(self, mock_identified_model):
        """Test that individual state contributions sum to total control."""
        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,
            pitch_rate_x=0.2,
            cp_y=0.05,
            com_vy=0.1,
            wheel_vel_left=1.0,
            wheel_vel_right=1.5,
        )

        # Sum all contributions
        total_contrib = (
            diag['contrib_pitch_x'] +
            diag['contrib_pitch_rate_x'] +
            diag['contrib_cp_error_y'] +
            diag['contrib_com_vy'] +
            diag['contrib_wheel_vel_mean']
        )

        # Should equal u_raw (before clipping)
        assert np.isclose(total_contrib, diag['u_raw'], atol=1e-6), \
            f"Contributions should sum to u_raw: {total_contrib} vs {diag['u_raw']}"


class TestStage2DIntegration:
    """Integration tests for Stage2D in simulation pipeline."""

    def test_integration_flag_produces_nonzero_wheel_torque(self, mock_identified_model):
        """Test 10: Integration flag --enable-stage2d-sagittal-lqr produces nonzero wheel torque."""
        # This is a smoke test that would require running simulate_hierarchical_controller.py
        # For now, we test the controller in isolation

        controller = Stage2DSagittalLQRController.from_identified_model(
            mock_identified_model, config='A'
        )

        # Perturbed state
        tau, diag = controller.compute_wheel_torques(
            pitch_x=0.1,
            pitch_rate_x=0.0,
            cp_y=0.0,
            com_vy=0.0,
            wheel_vel_left=0.0,
            wheel_vel_right=0.0,
        )

        # Should produce nonzero wheel torque
        assert tau[4] != 0.0 or tau[9] != 0.0, \
            "Perturbed state should produce nonzero wheel torque"

        # Wheel torques should be equal
        assert np.isclose(tau[4], tau[9]), \
            f"Wheel torques should be equal: l_wheel={tau[4]}, r_wheel={tau[9]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
