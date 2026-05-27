"""Torque ownership validation for balance-core source composition."""

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from wheeled_biped.controllers.balance_core_types import ACTION_DIM


_ALLOWED_SHARED_SOURCE_NAMES = {"tau_shape_posture", "tau_support_feedforward"}
_ALLOWED_SHARED_INDICES = {2, 3, 7, 8}


@dataclass(frozen=True)
class TorqueSourceEntry:
    """Single torque source command with ownership declaration."""

    name: str
    tau: np.ndarray
    owned_indices: Sequence[int]


@dataclass(frozen=True)
class TorqueOwnershipValidationResult:
    """Validation output for active source ownership across joints."""

    active_torque_owner_per_joint: List[str]
    ownership_violation_count: int
    violations: List[Dict[str, object]]


class TorqueOwnershipValidator:
    """Validates that active torque commands obey balance-core ownership rules."""

    def validate(
        self, sources: Sequence[TorqueSourceEntry], activity_tolerance: float = 1e-9
    ) -> TorqueOwnershipValidationResult:
        violations: List[Dict[str, object]] = []
        active_owners: List[str] = ["none"] * ACTION_DIM

        seen_names = set()
        active_sources_per_joint: Dict[int, List[str]] = {i: [] for i in range(ACTION_DIM)}

        for source in sources:
            if source.name in seen_names:
                raise ValueError(f"duplicate source name '{source.name}' is not allowed")
            seen_names.add(source.name)

            tau = np.asarray(source.tau)
            if tau.shape != (ACTION_DIM,):
                raise ValueError(f"tau must have shape ({ACTION_DIM},), got {tau.shape}")
            if not np.isfinite(tau).all():
                raise ValueError(f"tau for source '{source.name}' must be finite")

            owned = set(int(i) for i in source.owned_indices)
            invalid_owned = sorted(i for i in owned if i < 0 or i >= ACTION_DIM)
            if invalid_owned:
                raise ValueError(
                    f"owned_indices for source '{source.name}' contain out-of-range indices: {invalid_owned}"
                )

            active_indices = np.where(np.abs(tau) > activity_tolerance)[0]

            for joint_idx in active_indices:
                joint_idx = int(joint_idx)
                if joint_idx not in owned:
                    raise ValueError(
                        f"source '{source.name}' has active torque on unowned joint {joint_idx}"
                    )
                active_sources_per_joint[joint_idx].append(source.name)

        for joint_idx, owner_names in active_sources_per_joint.items():
            if not owner_names:
                continue
            if len(owner_names) == 1:
                active_owners[joint_idx] = owner_names[0]
                continue

            owner_set = set(owner_names)
            if (
                owner_set == _ALLOWED_SHARED_SOURCE_NAMES
                and joint_idx in _ALLOWED_SHARED_INDICES
            ):
                active_owners[joint_idx] = "tau_shape_posture+tau_support_feedforward"
            else:
                raise ValueError(
                    "exclusive owner conflict on joint "
                    f"{joint_idx}: active sources {sorted(owner_set)}"
                )

        return TorqueOwnershipValidationResult(
            active_torque_owner_per_joint=active_owners,
            ownership_violation_count=len(violations),
            violations=violations,
        )
