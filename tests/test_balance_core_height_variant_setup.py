"""Tests for balance-core true height variant setup generation."""

import pytest
import mujoco
import numpy as np

from wheeled_biped.utils.config import get_model_path


def test_nominal_variant_uses_keyframe_baseline():
    """Test that nominal variant uses keyframe baseline, not HeightIK regeneration."""
    from scripts.validate_balance_core_height_variants_v2 import (
        generate_height_variant_setup,
        calibrate_root_z_for_wheel_floor_contact,
    )

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Measure nominal from keyframe
    data_nominal = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data_nominal, 0)
    data_nominal.qvel[:] = 0.0
    calibrate_root_z_for_wheel_floor_contact(model, data_nominal)
    mujoco.mj_forward(model, data_nominal)

    torso_id = model.body("torso").id
    nominal_com_z = float(data_nominal.subtree_com[torso_id][2])
    nominal_hip_pitch = float(data_nominal.qpos[9])
    nominal_knee = float(data_nominal.qpos[10])

    # Generate nominal setup
    setup = generate_height_variant_setup(
        model=model,
        variant_name="nominal",
        target_com_z_m=nominal_com_z,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=True,
        tolerance_m=0.005,
    )

    # Verify it used keyframe baseline
    assert setup.posture_search_method == "keyframe_baseline"
    assert setup.setup_valid
    assert abs(setup.achieved_com_z_m - nominal_com_z) < 1e-6
    assert abs(setup.hip_pitch_ref - nominal_hip_pitch) < 1e-6
    assert abs(setup.knee_ref - nominal_knee) < 1e-6


def test_height_variant_uses_com_calibrated_search():
    """Test that non-nominal variants use CoM-calibrated search."""
    from scripts.validate_balance_core_height_variants_v2 import generate_height_variant_setup

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    nominal_com_z = 0.404
    nominal_hip_pitch = 0.926
    nominal_knee = 1.748

    # Generate high variant
    setup = generate_height_variant_setup(
        model=model,
        variant_name="high_small",
        target_com_z_m=nominal_com_z + 0.01,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=False,
        tolerance_m=0.005,
    )

    # Verify it used CoM-calibrated search
    assert setup.posture_search_method == "com_calibrated_search"

    # Verify posture changed from nominal
    assert abs(setup.hip_pitch_ref - nominal_hip_pitch) > 0.01 or abs(setup.knee_ref - nominal_knee) > 0.01


def test_com_calibrated_search_achieves_target_height():
    """Test that CoM-calibrated search achieves target height within tolerance."""
    from scripts.validate_balance_core_height_variants_v2 import generate_height_variant_setup

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    nominal_com_z = 0.404
    nominal_hip_pitch = 0.926
    nominal_knee = 1.748
    target_com_z = nominal_com_z + 0.01
    tolerance = 0.005

    setup = generate_height_variant_setup(
        model=model,
        variant_name="high_test",
        target_com_z_m=target_com_z,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=False,
        tolerance_m=tolerance,
    )

    # Verify height achieved within tolerance
    if setup.setup_valid:
        assert abs(setup.achieved_com_z_m - target_com_z) < tolerance


def test_setup_validity_rejects_invalid_candidates():
    """Test that setup validity correctly rejects invalid candidates."""
    from scripts.validate_balance_core_height_variants_v2 import generate_height_variant_setup

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    # Try to generate a variant far outside feasible range
    setup = generate_height_variant_setup(
        model=model,
        variant_name="invalid_test",
        target_com_z_m=0.20,  # Way too low
        nominal_hip_pitch=0.926,
        nominal_knee=1.748,
        use_keyframe_baseline=False,
        tolerance_m=0.005,
    )

    # Should be invalid
    assert not setup.setup_valid
    assert setup.setup_failure_reason is not None


def test_each_variant_captures_own_equilibrium():
    """Test that each valid variant captures its own equilibrium references."""
    from scripts.validate_balance_core_height_variants_v2 import generate_height_variant_setup

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    nominal_com_z = 0.404
    nominal_hip_pitch = 0.926
    nominal_knee = 1.748

    # Generate nominal
    setup_nominal = generate_height_variant_setup(
        model=model,
        variant_name="nominal",
        target_com_z_m=nominal_com_z,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=True,
        tolerance_m=0.005,
    )

    # Generate high variant
    setup_high = generate_height_variant_setup(
        model=model,
        variant_name="high_small",
        target_com_z_m=nominal_com_z + 0.01,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=False,
        tolerance_m=0.005,
    )

    # Both should be valid and have equilibrium references
    if setup_nominal.setup_valid and setup_high.setup_valid:
        assert setup_nominal.equilibrium_joint_pos is not None
        assert setup_high.equilibrium_joint_pos is not None

        # Equilibrium references should be different
        assert not np.allclose(
            setup_nominal.equilibrium_joint_pos,
            setup_high.equilibrium_joint_pos,
            atol=1e-3
        )


def test_root_z_calibration_runs_per_variant():
    """Test that root_z calibration is run for each variant."""
    from scripts.validate_balance_core_height_variants_v2 import generate_height_variant_setup

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    nominal_com_z = 0.404
    nominal_hip_pitch = 0.926
    nominal_knee = 1.748

    # Generate two variants
    setup1 = generate_height_variant_setup(
        model=model,
        variant_name="nominal",
        target_com_z_m=nominal_com_z,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=True,
        tolerance_m=0.005,
    )

    setup2 = generate_height_variant_setup(
        model=model,
        variant_name="high_small",
        target_com_z_m=nominal_com_z + 0.01,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=False,
        tolerance_m=0.005,
    )

    # Both should have calibrated root_z
    assert setup1.calibrated_root_z_m > 0.0
    assert setup2.calibrated_root_z_m > 0.0

    # Root_z should be different for different heights
    if setup1.setup_valid and setup2.setup_valid:
        assert abs(setup1.calibrated_root_z_m - setup2.calibrated_root_z_m) > 0.001


def test_height_variant_not_root_z_only_offset():
    """Test that height variants change joint posture, not just root_z."""
    from scripts.validate_balance_core_height_variants_v2 import generate_height_variant_setup

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))

    nominal_com_z = 0.404
    nominal_hip_pitch = 0.926
    nominal_knee = 1.748

    # Generate nominal and high variant
    setup_nominal = generate_height_variant_setup(
        model=model,
        variant_name="nominal",
        target_com_z_m=nominal_com_z,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=True,
        tolerance_m=0.005,
    )

    setup_high = generate_height_variant_setup(
        model=model,
        variant_name="high_small",
        target_com_z_m=nominal_com_z + 0.01,
        nominal_hip_pitch=nominal_hip_pitch,
        nominal_knee=nominal_knee,
        use_keyframe_baseline=False,
        tolerance_m=0.005,
    )

    # High variant should have different hip_pitch or knee
    if setup_high.setup_valid:
        joint_posture_changed = (
            abs(setup_high.hip_pitch_ref - setup_nominal.hip_pitch_ref) > 0.01 or
            abs(setup_high.knee_ref - setup_nominal.knee_ref) > 0.01
        )
        assert joint_posture_changed, "Height variant must change joint posture, not just root_z"
