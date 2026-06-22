"""Tests for hip‑yaw mode ownership validation and telemetry."""

import pytest

from wheeled_biped.controllers import hip_yaw_ownership
from wheeled_biped.controllers.hip_yaw_ownership import (
    HIP_YAW_MODE_OWNERS,
    OwnershipError,
    validate_ownership,
)


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset module‑level telemetry and internal owner tracking before each test."""
    hip_yaw_ownership._reset_telemetry()
    yield
    hip_yaw_ownership._reset_telemetry()


class TestHipYawModeOwnership:
    def test_single_controller_writes_common(self):
        """A single controller writing to 'common' updates telemetry and does not raise."""
        validate_ownership("controller_a", "common")
        assert hip_yaw_ownership.hip_yaw_common_owner == "controller_a"
        assert hip_yaw_ownership.hip_yaw_divergence_owner is None
        assert hip_yaw_ownership.hip_yaw_mode_ownership_violation is False

    def test_single_controller_writes_divergence(self):
        """A single controller writing to 'divergence' updates telemetry and does not raise."""
        validate_ownership("controller_b", "divergence")
        assert hip_yaw_ownership.hip_yaw_divergence_owner == "controller_b"
        assert hip_yaw_ownership.hip_yaw_common_owner is None
        assert hip_yaw_ownership.hip_yaw_mode_ownership_violation is False

    def test_conflicting_writes_divergence_raises(self):
        """Two dummy controllers writing to the same divergence mode raises OwnershipError."""
        validate_ownership("controller_a", "divergence")
        assert hip_yaw_ownership.hip_yaw_divergence_owner == "controller_a"

        with pytest.raises(OwnershipError) as exc_info:
            validate_ownership("controller_b", "divergence")

        # Telemetry flag must indicate the violation
        assert hip_yaw_ownership.hip_yaw_mode_ownership_violation is True
        assert hip_yaw_ownership.hip_yaw_divergence_owner == "controller_a"
        assert "controller_a" in str(exc_info.value)
        assert "controller_b" in str(exc_info.value)

    def test_conflicting_writes_common_raises(self):
        """Two controllers writing to the common mode raises OwnershipError."""
        validate_ownership("shape_posture", "common")
        with pytest.raises(OwnershipError):
            validate_ownership("yaw_controller", "common")

        assert hip_yaw_ownership.hip_yaw_mode_ownership_violation is True
        assert hip_yaw_ownership.hip_yaw_common_owner == "shape_posture"

    def test_independent_modes_do_not_conflict(self):
        """Writing to different modes is allowed (common and divergence are independent)."""
        validate_ownership("posture", "common")
        validate_ownership("divergence_controller", "divergence")
        assert hip_yaw_ownership.hip_yaw_common_owner == "posture"
        assert hip_yaw_ownership.hip_yaw_divergence_owner == "divergence_controller"
        assert hip_yaw_ownership.hip_yaw_mode_ownership_violation is False

    def test_unknown_mode_is_ignored(self):
        """An unknown mode string does not raise and does not affect telemetry."""
        validate_ownership("controller_x", "totally_unknown_mode")
        assert hip_yaw_ownership.hip_yaw_common_owner is None
        assert hip_yaw_ownership.hip_yaw_divergence_owner is None
        assert hip_yaw_ownership.hip_yaw_mode_ownership_violation is False

    def test_owners_constant_has_expected_keys(self):
        """HIP_YAW_MODE_OWNERS constant contains 'common' and 'divergence'."""
        assert "common" in HIP_YAW_MODE_OWNERS
        assert "divergence" in HIP_YAW_MODE_OWNERS
        assert HIP_YAW_MODE_OWNERS["divergence"] == "mode_based_divergence"
        assert HIP_YAW_MODE_OWNERS["common"] == "posture"
