"""
Hip‑yaw mode ownership validation utilities.

Ensures that only one controller writes to a given hip‑yaw mode (common or divergence).
If multiple controllers attempt to write to the same mode, an ``OwnershipError`` is raised.
Telemetry fields are exposed at module level for inspection.
"""

from __future__ import annotations

from typing import Dict, Optional

# Mapping of hip‑yaw modes to their descriptive owners (for telemetry)
HIP_YAW_MODE_OWNERS: Dict[str, str] = {
    "common": "posture",
    "divergence": "mode_based_divergence",
}

# Telemetry variables (module level) – will be updated by ``validate_ownership``
hip_yaw_common_owner: Optional[str] = None
hip_yaw_divergence_owner: Optional[str] = None
hip_yaw_mode_ownership_violation: bool = False

# Internal tracking of which controller currently owns each mode
_current_owners: Dict[str, str] = {}


class OwnershipError(Exception):
    """Raised when more than one controller attempts to write to the same hip‑yaw mode."""


def _reset_telemetry() -> None:
    """Reset telemetry fields and internal owner tracking.
    Used in tests to ensure a clean state between runs.
    """
    global hip_yaw_common_owner, hip_yaw_divergence_owner, hip_yaw_mode_ownership_violation, _current_owners
    hip_yaw_common_owner = None
    hip_yaw_divergence_owner = None
    hip_yaw_mode_ownership_violation = False
    _current_owners = {}


def validate_ownership(controller_name: str, mode: str) -> None:
    """Validate that a controller can write to a given hip‑yaw mode.

    Args:
        controller_name: Identifier for the controller attempting the write.
        mode: One of the keys in ``HIP_YAW_MODE_OWNERS`` (e.g., ``"common"`` or ``"divergence"``).

    Raises:
        OwnershipError: If another controller already owns the requested mode.
    """
    global hip_yaw_common_owner, hip_yaw_divergence_owner, hip_yaw_mode_ownership_violation, _current_owners

    if mode not in HIP_YAW_MODE_OWNERS:
        # Unknown mode – ignore silently (could also raise, but spec does not require).
        return

    existing = _current_owners.get(mode)
    if existing is None:
        # First owner for this mode
        _current_owners[mode] = controller_name
        hip_yaw_mode_ownership_violation = False
        if mode == "common":
            hip_yaw_common_owner = controller_name
        elif mode == "divergence":
            hip_yaw_divergence_owner = controller_name
    else:
        # Conflict detected
        hip_yaw_mode_ownership_violation = True
        raise OwnershipError(
            f"Hip‑yaw mode '{mode}' already owned by '{existing}'. "
            f"Controller '{controller_name}' cannot also write to this mode."
        )
