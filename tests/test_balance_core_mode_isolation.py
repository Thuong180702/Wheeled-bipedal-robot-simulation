"""Test balance-core mode isolation from legacy flags.

Task 10: Ensure balance-core mode rejects incompatible legacy flags.
"""

import inspect
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import jax.numpy as jnp

from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer


def _args(**kwargs):
    """Helper to create mock args namespace for validation tests."""
    defaults = {
        "controller_mode": "balance-core",
        "enable_static_dynamics_wrapper": False,
        "enable_secondary_wheel_balance": False,
        "enable_stage2_static_posture_hold": False,
        "enable_stage2b_gravity_feedforward": False,
        "enable_stage2b_roll_direct": False,
        "enable_stage2b_sagittal_wheel": False,
        "enable_stage2c_sagittal_state_feedback": False,
        "enable_stage2d_sagittal_lqr": False,
        "initialize_tau_prev_from_wbc": False,
        "use_per_actuator_wbc_authority": False,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_legacy_mode_accepts_legacy_flags():
    """Legacy mode should accept all legacy flags without error."""
    script = Path("scripts/simulate_hierarchical_controller.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--controller-mode", "legacy",
            "--enable-static-dynamics-wrapper",
            "--enable-secondary-wheel-balance",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
    )
    # Should not fail validation (may fail later for other reasons, but not validation)
    assert "incompatible" not in result.stderr.lower(), f"Legacy mode rejected legacy flags: {result.stderr}"


def test_balance_core_mode_rejects_static_dynamics_wrapper():
    """Balance-core mode should reject --enable-static-dynamics-wrapper."""
    script = Path("scripts/simulate_hierarchical_controller.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--controller-mode", "balance-core",
            "--enable-static-dynamics-wrapper",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Should fail validation"
    assert "incompatible" in result.stderr.lower() or "incompatible" in result.stdout.lower(), \
        f"Should mention incompatibility: {result.stderr}"


def test_balance_core_mode_rejects_stage2_flags():
    """Balance-core mode should reject Stage 2 flags."""
    script = Path("scripts/simulate_hierarchical_controller.py")

    # Test Stage 2 static posture hold
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--controller-mode", "balance-core",
            "--enable-stage2-static-posture-hold",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Should fail validation for stage2-static-posture-hold"
    assert "incompatible" in result.stderr.lower() or "incompatible" in result.stdout.lower(), \
        f"Should mention incompatibility: {result.stderr}"

    # Test Stage 2B gravity feedforward
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--controller-mode", "balance-core",
            "--enable-stage2b-gravity-feedforward",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Should fail validation for stage2b-gravity-feedforward"
    assert "incompatible" in result.stderr.lower() or "incompatible" in result.stdout.lower(), \
        f"Should mention incompatibility: {result.stderr}"

    # Test Stage 2D sagittal LQR
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--controller-mode", "balance-core",
            "--enable-stage2d-sagittal-lqr",
            "--steps", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, "Should fail validation for stage2d-sagittal-lqr"
    assert "incompatible" in result.stderr.lower() or "incompatible" in result.stdout.lower(), \
        f"Should mention incompatibility: {result.stderr}"


def test_build_balance_core_controllers_returns_functional_components():
    """Test that build_balance_core_controllers returns all 6 components."""
    # Import the function from the script
    import sys
    import numpy as np
    sys.path.insert(0, str(Path("scripts")))
    from simulate_hierarchical_controller import build_balance_core_controllers

    control_dt = 0.01
    support_vector = np.array([0.0, 0.0, 0.0, -15.5, 0.0, 0.0, 0.0, 0.0, -15.8, 0.0])
    torque_limit = np.ones(10) * 60.0
    max_torque_rate = np.ones(10) * 400.0

    controllers = build_balance_core_controllers(
        control_dt=control_dt,
        support_feedforward_vector=support_vector,
        torque_limit=torque_limit,
        max_torque_rate=max_torque_rate,
    )

    # Check all 6 expected keys are present
    assert "contact_supervisor" in controllers
    assert "shape_posture" in controllers
    assert "support_feedforward" in controllers
    assert "sagittal_wheel_balance" in controllers
    assert "lateral_roll_balance" in controllers
    assert "composer" in controllers

    # Check that each component is not None
    assert controllers["contact_supervisor"] is not None
    assert controllers["shape_posture"] is not None
    assert controllers["support_feedforward"] is not None
    assert controllers["sagittal_wheel_balance"] is not None
    assert controllers["lateral_roll_balance"] is not None
    assert controllers["composer"] is not None


def test_resolve_support_feedforward_vector_returns_empirical_default():
    """Test that resolve_support_feedforward_vector returns empirical support vector."""
    import sys
    sys.path.insert(0, str(Path("scripts")))
    from simulate_hierarchical_controller import resolve_support_feedforward_vector

    support_vector = resolve_support_feedforward_vector()

    # Check it's a list or array with 10 elements
    assert len(support_vector) == 10

    # Check validated hip-pitch+knee support entries are present
    assert support_vector[2] == pytest.approx(4.1)
    assert support_vector[3] == pytest.approx(-15.5)
    assert support_vector[7] == pytest.approx(3.2)
    assert support_vector[8] == pytest.approx(-15.8)


def test_balance_core_support_feedforward_defaults_to_hip_pitch_knee_group():
    """Balance-core default support feedforward should cover both hip-pitch and knee joints."""
    import sys
    import numpy as np
    sys.path.insert(0, str(Path("scripts")))
    from simulate_hierarchical_controller import build_balance_core_controllers, resolve_support_feedforward_vector

    controllers = build_balance_core_controllers(
        control_dt=0.01,
        support_feedforward_vector=resolve_support_feedforward_vector(),
        torque_limit=np.ones(10) * 60.0,
        max_torque_rate=np.ones(10) * 400.0,
    )

    support_feedforward = controllers["support_feedforward"]
    assert support_feedforward.joint_group == "hip_pitch_knee"

    tau, diagnostics = support_feedforward.compute()
    assert tau[2] != 0.0
    assert tau[3] != 0.0
    assert tau[7] != 0.0
    assert tau[8] != 0.0


def test_balance_core_legacy_torque_sources_are_zeroed():
    """Test that zero_legacy_torque_sources_for_balance_core returns correct zero dict."""
    import sys
    sys.path.insert(0, str(Path("scripts")))
    from simulate_hierarchical_controller import zero_legacy_torque_sources_for_balance_core

    sources = zero_legacy_torque_sources_for_balance_core()

    assert sorted(sources) == [
        "tau_hip_roll_centering",
        "tau_inverse_dynamics",
        "tau_leg_position",
        "tau_posture",
        "tau_wbc_correction",
        "tau_wbc_scaled",
        "tau_wheel_balance",
    ]
    for value in sources.values():
        assert jnp.allclose(value, jnp.zeros(10))


def test_integrated_wbc_remains_available_but_composer_has_no_wbc_input():
    """Test that WBC is preserved but not used in balance-core composition.

    Task 16: Verify WBC-off-by-default behavior.
    - IntegratedWBC remains importable
    - BalanceCoreTorqueComposer.compose() has no WBC input parameters
    """
    # WBC should be importable (preserved)
    assert IntegratedWBC is not None

    # Balance-core composer should have no WBC parameters
    signature = inspect.signature(BalanceCoreTorqueComposer.compose)
    assert "tau_wbc" not in signature.parameters
    assert "tau_wbc_correction" not in signature.parameters
    assert "tau_inverse_dynamics" not in signature.parameters


def test_balance_core_rejects_simultaneous_experimental_sagittal_controllers():
    """Test that balance-core rejects all experimental sagittal controller flags.

    Task 17: Ensure balance-core cannot silently run simultaneous experimental
    sagittal controllers or any experimental sagittal controller by default.

    The functional SagittalWheelBalanceController is the only balance-core wheel owner.
    """
    # Import validation function
    sys.path.insert(0, str(Path("scripts")))
    from simulate_hierarchical_controller import validate_balance_core_mode_args

    args = _args(
        enable_stage2b_sagittal_wheel=True,
        enable_stage2c_sagittal_state_feedback=True,
        enable_stage2d_sagittal_lqr=True,
    )
    with pytest.raises(ValueError, match="enable-stage2b-sagittal-wheel"):
        validate_balance_core_mode_args(args)
