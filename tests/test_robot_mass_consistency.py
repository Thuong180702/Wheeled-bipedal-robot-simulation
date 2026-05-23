"""Regression tests for robot mass consistency.

Ensures that robot mass is consistently derived from MuJoCo model
and that no dangerous 15 kg defaults remain in production code.
"""

import numpy as np
import pytest
import mujoco

from wheeled_biped.controllers.robot_model_utils import (
    get_total_robot_mass,
    get_robot_weight,
)
from wheeled_biped.controllers.integrated_wbc import IntegratedWBC
from wheeled_biped.controllers.centroidal_state_estimator import (
    CentroidalStateEstimator,
    CentroidalStateEstimatorConfig,
)
import jax.numpy as jnp


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


@pytest.fixture
def mj_model():
    """Load MuJoCo model."""
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


@pytest.fixture
def mj_data(mj_model):
    """Create MuJoCo data."""
    return mujoco.MjData(mj_model)


def test_xml_mass_is_correct(mj_model):
    """Test 1: Verify XML robot mass is approximately 8.1 kg.

    The robot XML defines individual body masses that sum to ~8.1 kg:
    - torso: 2.5 kg
    - each leg: 0.5 + 0.8 + 0.8 + 0.6 + 0.1 = 2.8 kg
    - two legs: 5.6 kg
    - total: 8.1 kg

    This test ensures the XML hasn't been accidentally modified.
    """
    total_mass = np.sum(mj_model.body_mass)

    # Verify mass is in expected range (8.0 to 8.2 kg)
    assert 8.0 < total_mass < 8.2, (
        f"Expected robot mass ~8.1 kg, got {total_mass:.4f} kg. "
        f"Check assets/robot/wheeled_biped_real.xml for mass changes."
    )

    # Verify weight is in expected range (78.5 to 80.5 N)
    gravity = abs(float(mj_model.opt.gravity[2]))
    total_weight = total_mass * gravity
    assert 78.5 < total_weight < 80.5, (
        f"Expected robot weight ~79.5 N, got {total_weight:.4f} N"
    )


def test_robot_model_utils_helpers(mj_model):
    """Test 2: Verify robot_model_utils helpers return correct values."""
    robot_mass = get_total_robot_mass(mj_model)
    robot_weight = get_robot_weight(mj_model)

    # Verify mass
    expected_mass = float(np.sum(mj_model.body_mass))
    assert abs(robot_mass - expected_mass) < 1e-6, (
        f"get_total_robot_mass returned {robot_mass:.4f}, "
        f"expected {expected_mass:.4f}"
    )

    # Verify weight
    gravity = abs(float(mj_model.opt.gravity[2]))
    expected_weight = expected_mass * gravity
    assert abs(robot_weight - expected_weight) < 1e-6, (
        f"get_robot_weight returned {robot_weight:.4f}, "
        f"expected {expected_weight:.4f}"
    )


def test_integrated_wbc_derives_mass_from_model(mj_model):
    """Test 3: Verify IntegratedWBC derives robot_mass from model by default.

    When robot_mass is not provided, IntegratedWBC should automatically
    derive it from mj_model.body_mass, not default to 15 kg.
    """
    # Create WBC without explicit robot_mass
    wbc = IntegratedWBC(mj_model)

    # Verify it derived mass from model
    expected_mass = float(np.sum(mj_model.body_mass))
    assert abs(wbc.robot_mass - expected_mass) < 1e-6, (
        f"IntegratedWBC.robot_mass = {wbc.robot_mass:.4f}, "
        f"expected {expected_mass:.4f} (derived from model)"
    )

    # Verify wrench computer also has correct mass
    assert abs(wbc.wrench_computer.robot_mass - expected_mass) < 1e-6, (
        f"CentroidalWrenchComputer.robot_mass = {wbc.wrench_computer.robot_mass:.4f}, "
        f"expected {expected_mass:.4f}"
    )


def test_integrated_wbc_respects_explicit_mass(mj_model):
    """Test 4: Verify IntegratedWBC respects explicitly provided robot_mass."""
    explicit_mass = 10.0  # Arbitrary test value

    wbc = IntegratedWBC(mj_model, robot_mass=explicit_mass)

    assert abs(wbc.robot_mass - explicit_mass) < 1e-6, (
        f"IntegratedWBC.robot_mass = {wbc.robot_mass:.4f}, "
        f"expected {explicit_mass:.4f} (explicit value)"
    )

    assert abs(wbc.wrench_computer.robot_mass - explicit_mass) < 1e-6, (
        f"CentroidalWrenchComputer.robot_mass = {wbc.wrench_computer.robot_mass:.4f}, "
        f"expected {explicit_mass:.4f}"
    )


def test_wrench_computer_uses_correct_mass(mj_model):
    """Test 5: Verify CentroidalWrenchComputer uses correct robot mass.

    The wrench computer's gravity compensation force should be based on
    the actual robot mass (~8.1 kg → ~79.5 N), not 15 kg (→ 147 N).
    """
    # Create WBC with model-derived mass
    wbc = IntegratedWBC(mj_model)
    robot_mass = get_total_robot_mass(mj_model)
    gravity = abs(float(mj_model.opt.gravity[2]))

    # Verify wrench computer has correct mass
    assert abs(wbc.wrench_computer.robot_mass - robot_mass) < 1e-6, (
        f"CentroidalWrenchComputer.robot_mass = {wbc.wrench_computer.robot_mass:.4f}, "
        f"expected {robot_mass:.4f} (model-derived)"
    )

    # Verify gravity compensation would produce correct force
    expected_gravity_force = robot_mass * gravity
    assert 78.5 < expected_gravity_force < 80.5, (
        f"Expected gravity compensation ~79.5 N, got {expected_gravity_force:.2f} N"
    )

    # Verify it's NOT using 15 kg (which would give ~147 N)
    wrong_gravity_force = 15.0 * gravity
    assert abs(expected_gravity_force - wrong_gravity_force) > 50.0, (
        f"Gravity force {expected_gravity_force:.2f} N is suspiciously close to "
        f"15 kg * g = {wrong_gravity_force:.2f} N. Check for hardcoded 15 kg default."
    )


def test_no_15kg_hardcoded_in_controllers():
    """Test 6: Verify no 15 kg hardcoded defaults in controller production code.

    This is a source code audit test. It scans controller files for
    dangerous patterns like "robot_mass = 15.0" or "robot_mass: float = 15.0".

    Note: This test may have false positives (comments, test fixtures).
    Manual review is needed if this test fails.
    """
    import re
    from pathlib import Path

    # Controller files to audit
    controller_files = [
        "wheeled_biped/controllers/integrated_wbc.py",
        "wheeled_biped/controllers/centroidal_wrench_computer.py",
        "wheeled_biped/controllers/static_balance_controller.py",
    ]

    # Dangerous patterns
    patterns = [
        r"robot_mass\s*=\s*15\.0",  # robot_mass = 15.0
        r"robot_mass:\s*float\s*=\s*15\.0",  # robot_mass: float = 15.0
    ]

    violations = []

    for file_path in controller_files:
        path = Path(file_path)
        if not path.exists():
            continue

        content = path.read_text(encoding='utf-8')

        for pattern in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                # Get line number
                line_num = content[:match.start()].count('\n') + 1
                violations.append(f"{file_path}:{line_num}: {match.group()}")

    assert len(violations) == 0, (
        f"Found {len(violations)} dangerous 15 kg defaults in production code:\n" +
        "\n".join(violations) +
        "\n\nThese should use model-derived mass or require explicit parameter."
    )


def test_centroidal_state_estimator_requires_explicit_mass():
    """Test 7: Verify CentroidalStateEstimator requires explicit robot_mass.

    CentroidalStateEstimator should not have a default mass value.
    It should require the caller to provide robot_mass explicitly.
    """
    # This test verifies the API contract - robot_mass is required
    # If this test fails, it means someone added a dangerous default

    # Attempt to create without robot_mass should fail
    with pytest.raises(TypeError, match="robot_mass"):
        CentroidalStateEstimatorConfig(
            torso_inertia=jnp.array([0.1, 0.1, 0.05])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
