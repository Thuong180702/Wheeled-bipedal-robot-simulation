import numpy as np
import pytest

from wheeled_biped.controllers.balance_core_types import ACTION_DIM
from wheeled_biped.controllers.torque_ownership_validator import (
    TorqueOwnershipValidator,
    TorqueSourceEntry,
)


def _entry(name, owned_indices, active_indices=None):
    tau = np.zeros(ACTION_DIM, dtype=np.float32)
    if active_indices is not None:
        tau[np.array(active_indices, dtype=np.int32)] = 1.0
    return TorqueSourceEntry(name=name, tau=tau, owned_indices=owned_indices)


def test_accepts_approved_balance_core_sources():
    sources = [
        _entry("tau_shape_posture", [2, 3, 7, 8], [2, 3]),
        _entry("tau_support_feedforward", [2, 3, 7, 8], [7, 8]),
        _entry("tau_sagittal_wheel_balance", [4, 9], [4]),
        _entry("tau_lateral_roll_balance", [0, 5], [0]),
    ]

    result = TorqueOwnershipValidator().validate(sources)

    assert result.ownership_violation_count == 0
    assert result.violations == []
    assert result.active_torque_owner_per_joint[0] == "tau_lateral_roll_balance"
    assert result.active_torque_owner_per_joint[2] == "tau_shape_posture"
    assert result.active_torque_owner_per_joint[4] == "tau_sagittal_wheel_balance"
    assert result.active_torque_owner_per_joint[7] == "tau_support_feedforward"


def test_rejects_torque_outside_owned_joint_group():
    sources = [_entry("tau_sagittal_wheel_balance", [4, 9], [4, 2])]

    with pytest.raises(ValueError, match="unowned joint"):
        TorqueOwnershipValidator().validate(sources)


def test_rejects_duplicate_source_name():
    sources = [
        _entry("tau_shape_posture", [2, 3, 7, 8], [2]),
        _entry("tau_shape_posture", [2, 3, 7, 8], [3]),
    ]

    with pytest.raises(ValueError, match="duplicate source name"):
        TorqueOwnershipValidator().validate(sources)


def test_rejects_exclusive_owner_conflict():
    sources = [
        _entry("tau_sagittal_wheel_balance", [4, 9], [4]),
        _entry("tau_lateral_roll_balance", [0, 5, 4], [4]),
    ]

    with pytest.raises(ValueError, match="exclusive owner conflict"):
        TorqueOwnershipValidator().validate(sources)


def test_allows_only_shape_and_support_sharing_on_support_joints():
    sources = [
        _entry("tau_shape_posture", [2, 3, 7, 8], [2]),
        _entry("tau_support_feedforward", [2, 3, 7, 8], [2]),
    ]

    result = TorqueOwnershipValidator().validate(sources)

    assert result.ownership_violation_count == 0
    assert result.violations == []
    assert result.active_torque_owner_per_joint[2] == "tau_shape_posture+tau_support_feedforward"


def test_rejects_out_of_range_owned_index():
    sources = [_entry("tau_sagittal_wheel_balance", [4, ACTION_DIM], [4])]

    with pytest.raises(ValueError, match="out-of-range indices"):
        TorqueOwnershipValidator().validate(sources)


def test_rejects_non_finite_tau():
    tau = np.zeros(ACTION_DIM, dtype=np.float32)
    tau[4] = np.inf
    sources = [
        TorqueSourceEntry(
            name="tau_sagittal_wheel_balance", tau=tau, owned_indices=[4, 9]
        )
    ]

    with pytest.raises(ValueError, match="must be finite"):
        TorqueOwnershipValidator().validate(sources)
