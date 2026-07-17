"""
Simple composer that validates hip‑yaw mode ownership during torque composition.

This is a lightweight wrapper used by the balance‑core torque composer (or other
higher‑level composition code) to enforce that only one controller writes to a
given hip‑yaw mode (common or divergence). It records telemetry fields defined
in ``hip_yaw_ownership``.
"""

from __future__ import annotations

from typing import Mapping

# Import the ownership validation utilities
from .hip_yaw_ownership import validate_ownership, OwnershipError

# Telemetry fields are defined at module level in hip_yaw_ownership and are
# updated by ``validate_ownership``. They are re‑exported here for convenience.
from .hip_yaw_ownership import (
    hip_yaw_common_owner,
    hip_yaw_divergence_owner,
    hip_yaw_mode_ownership_violation,
    HIP_YAW_MODE_OWNERS,
)


def compose_torque(
    controller_writes: Mapping[str, str],
) -> None:
    """Validate hip‑yaw ownership for a set of controller writes.

    Args:
        controller_writes: Mapping from ``controller_name`` to ``mode`` string.
            ``mode`` must be a key in ``HIP_YAW_MODE_OWNERS`` (e.g. ``"common"``
            or ``"divergence"``). Controllers that do not write to hip‑yaw can
            simply omit an entry.

    Raises:
        OwnershipError: If more than one controller attempts to write to the same
            mode.
    """
    # Reset telemetry before a new composition step (the ownership module
    # provides a reset function for test isolation). In production this reset is
    # not strictly required because ownership is checked per step.
    try:
        # The ownership module defines a private reset helper; import it safely.
        from .hip_yaw_ownership import _reset_telemetry  # type: ignore
    except Exception:
        _reset_telemetry = lambda: None  # fallback no‑op

    _reset_telemetry()

    for controller_name, mode in controller_writes.items():
        # Validate each write – this will raise OwnershipError on conflict.
        validate_ownership(controller_name, mode)

    # No return value; telemetry variables are updated in the ownership module.
    return None
