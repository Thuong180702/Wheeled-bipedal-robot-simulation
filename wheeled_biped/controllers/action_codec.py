"""Canonical action composition and semantics for wheeled biped control.

This module defines the single source of truth for:
- Joint order and indices
- Action composition (base + residual)
- Action semantics (absolute vs pre-bias)
- Validation and clipping

Key concepts:
- **Absolute normalized action**: The action value after all composition, in [-1, 1].
  This is what the PID controller receives (after smoothing/delay/bias).
- **Pre-bias action**: The action before pid_action_bias is added.
  For pure PPO: policy output = pre-bias action.
  For residual PPO: base_action + residual_scale * residual_action = pre-bias action.
- **PID action bias**: A constant offset added to leg joints only, shifting action=0
  to the standing keyframe. Wheels have zero bias.

Action composition formula (residual mode):
    final_action_abs = clip(base_action_abs + residual_scale * residual_action, -1, 1)

where:
- base_action_abs: LQR/IK prior output (absolute normalized)
- residual_action: PPO policy output (bounded correction)
- residual_scale: scalar or per-joint scaling factor
- final_action_abs: composed action sent to env (absolute normalized)

CRITICAL: The residual composition pipeline does NOT add pid_action_bias. For residual
control, disable_pid_action_bias must be true to prevent double-addition of bias.
The base_action_abs from the LQR/IK prior is already in absolute normalized space.

Joint control modes:
- Leg joints (8): position targets
- Wheel joints (2): velocity targets
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import jax.numpy as jnp
import numpy as np


# ============================================================================
# Joint order and indices
# ============================================================================

JOINT_NAMES = [
    "L_hip_roll",
    "L_hip_yaw",
    "L_hip_pitch",
    "L_knee",
    "L_wheel",
    "R_hip_roll",
    "R_hip_yaw",
    "R_hip_pitch",
    "R_knee",
    "R_wheel",
]

# Individual joint indices
L_HIP_ROLL = 0
L_HIP_YAW = 1
L_HIP_PITCH = 2
L_KNEE = 3
L_WHEEL = 4
R_HIP_ROLL = 5
R_HIP_YAW = 6
R_HIP_PITCH = 7
R_KNEE = 8
R_WHEEL = 9

# Joint groups by control mode
LEG_POSITION_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]  # 8 leg joints
WHEEL_VELOCITY_INDICES = [4, 9]  # 2 wheel joints

# Joint groups by body side
LEFT_LEG_INDICES = [0, 1, 2, 3]
RIGHT_LEG_INDICES = [5, 6, 7, 8]
LEFT_WHEEL_INDEX = 4
RIGHT_WHEEL_INDEX = 9

# Joint groups by function
HIP_YAW_INDICES = [1, 6]
HIP_ROLL_INDICES = [0, 5]
HIP_PITCH_KNEE_INDICES = [2, 3, 7, 8]

ACTION_DIM = 10


# ============================================================================
# Enums
# ============================================================================

class ActionMode(Enum):
    """Action interpretation mode."""
    ABSOLUTE = "absolute"  # Policy outputs absolute normalized action
    RESIDUAL = "residual"  # Policy outputs bounded correction over base_action


class PolicyType(Enum):
    """Policy architecture type."""
    PURE_PPO = "pure_ppo"      # Standard PPO, outputs absolute action
    RESIDUAL_PPO = "residual_ppo"  # Residual PPO, outputs bounded correction


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class ControllerMetadata:
    """Metadata for controller configuration."""
    policy_type: PolicyType
    action_mode: ActionMode
    residual_scale: float | np.ndarray | jnp.ndarray
    obs_includes_base_action: bool
    base_controller_name: str | None = None  # e.g., "lqr_height_scheduled"


@dataclass
class ActionBreakdown:
    """Breakdown of action composition for logging and analysis.

    All action arrays are shape (10,) or (batch, 10).
    """
    base_action_abs: np.ndarray | jnp.ndarray  # Base controller output (absolute)
    residual_action: np.ndarray | jnp.ndarray  # Policy output (bounded correction)
    residual_scaled: np.ndarray | jnp.ndarray  # residual_scale * residual_action
    final_action_abs: np.ndarray | jnp.ndarray  # Composed action (absolute, clipped)
    residual_norm: float | np.ndarray | jnp.ndarray  # L2 norm of residual_scaled
    residual_saturation_rate: float | np.ndarray | jnp.ndarray  # Fraction of joints saturated


# ============================================================================
# Core composition function
# ============================================================================

def compose_residual_action(
    base_action_abs: np.ndarray | jnp.ndarray,
    residual_action: np.ndarray | jnp.ndarray,
    residual_scale: float | np.ndarray | jnp.ndarray,
    clip: bool = True,
) -> ActionBreakdown:
    """Compose residual action over base action.

    Implements the canonical formula:
        final_action_abs = clip(base_action_abs + residual_scale * residual_action, -1, 1)

    Args:
        base_action_abs: Base controller output, shape (..., 10), absolute normalized.
        residual_action: Policy output, shape (..., 10), bounded correction.
        residual_scale: Scalar or per-joint scaling factor, shape () or (10,).
        clip: Whether to clip final_action_abs to [-1, 1].

    Returns:
        ActionBreakdown with all intermediate values and diagnostics.

    Notes:
        - All inputs should be in [-1, 1] range.
        - residual_scale is typically in [0, 1] to bound the correction magnitude.
        - Clipping is recommended to ensure final_action_abs stays in valid range.
        - This function does NOT add pid_action_bias; that happens in the env.
    """
    # Validate shapes
    validate_action_shape(base_action_abs, "base_action_abs")
    validate_action_shape(residual_action, "residual_action")
    validate_residual_scale(residual_scale)

    # Compute scaled residual
    residual_scaled = residual_scale * residual_action

    # Compose
    final_action_abs = base_action_abs + residual_scaled

    # Clip if requested
    if clip:
        xp = jnp if isinstance(final_action_abs, jnp.ndarray) else np
        final_action_abs = xp.clip(final_action_abs, -1.0, 1.0)

    # Compute diagnostics
    xp = jnp if isinstance(residual_scaled, jnp.ndarray) else np
    residual_norm = xp.linalg.norm(residual_scaled, axis=-1)

    # Saturation rate: fraction of joints at ±1 after clipping
    if clip:
        saturated = xp.abs(final_action_abs) >= 0.999
        residual_saturation_rate = xp.mean(saturated.astype(xp.float32), axis=-1)
    else:
        residual_saturation_rate = xp.zeros_like(residual_norm)

    return ActionBreakdown(
        base_action_abs=base_action_abs,
        residual_action=residual_action,
        residual_scaled=residual_scaled,
        final_action_abs=final_action_abs,
        residual_norm=residual_norm,
        residual_saturation_rate=residual_saturation_rate,
    )


# ============================================================================
# Validation functions
# ============================================================================

def validate_action_shape(
    action: np.ndarray | jnp.ndarray,
    name: str = "action",
) -> None:
    """Validate action array shape.

    Args:
        action: Action array, shape (..., 10).
        name: Name for error messages.

    Raises:
        ValueError: If action shape is invalid.
    """
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"{name} must have last dimension {ACTION_DIM}, got shape {action.shape}"
        )


def validate_residual_scale(
    residual_scale: float | np.ndarray | jnp.ndarray,
) -> None:
    """Validate residual_scale shape and range.

    Args:
        residual_scale: Scalar or per-joint scaling factor.

    Raises:
        ValueError: If residual_scale shape or range is invalid.
    """
    if isinstance(residual_scale, (np.ndarray, jnp.ndarray)):
        if residual_scale.shape not in [(), (ACTION_DIM,)]:
            raise ValueError(
                f"residual_scale must be scalar or shape ({ACTION_DIM},), "
                f"got shape {residual_scale.shape}"
            )


def clip_normalized_action(
    action: np.ndarray | jnp.ndarray,
) -> np.ndarray | jnp.ndarray:
    """Clip action to [-1, 1] range.

    Args:
        action: Action array, shape (..., 10).

    Returns:
        Clipped action, same shape as input.
    """
    validate_action_shape(action)
    xp = jnp if isinstance(action, jnp.ndarray) else np
    return xp.clip(action, -1.0, 1.0)


# ============================================================================
# Helper functions
# ============================================================================

def action_group_stats(
    action: np.ndarray | jnp.ndarray,
    group_indices: list[int],
) -> Tuple[float, float, float]:
    """Compute mean, std, max absolute value for a joint group.

    Args:
        action: Action array, shape (..., 10).
        group_indices: List of joint indices to analyze.

    Returns:
        (mean, std, max_abs) for the specified joint group.
    """
    validate_action_shape(action)
    xp = jnp if isinstance(action, jnp.ndarray) else np

    group_action = action[..., group_indices]
    mean = float(xp.mean(group_action))
    std = float(xp.std(group_action))
    max_abs = float(xp.max(xp.abs(group_action)))

    return mean, std, max_abs


def compute_pid_action_bias(
    standing_keyframe: np.ndarray | jnp.ndarray,
) -> np.ndarray | jnp.ndarray:
    """Compute PID action bias from standing keyframe.

    The PID action bias shifts action=0 to the standing keyframe for leg joints.
    Wheel joints have zero bias (velocity targets, not position).

    Args:
        standing_keyframe: Joint positions at standing pose, shape (10,).
            Should be in the same units as qpos (radians for legs, rad/s for wheels).

    Returns:
        pid_action_bias: Bias to add to pre-bias action, shape (10,).
            Legs: normalized position offset to standing keyframe.
            Wheels: zero (velocity targets).

    Notes:
        - This function assumes the standing keyframe is already normalized to [-1, 1].
        - In practice, the standing keyframe is loaded from the MuJoCo model's keyframe.
        - The bias is added in the env after smoothing/delay: action_with_bias = action + bias.
        - This is a reference implementation; the actual bias is computed in BalanceEnv.
    """
    validate_action_shape(standing_keyframe, "standing_keyframe")
    xp = jnp if isinstance(standing_keyframe, jnp.ndarray) else np

    # Legs: use standing keyframe as bias
    # Wheels: zero bias (velocity targets)
    bias = xp.zeros_like(standing_keyframe)

    if isinstance(standing_keyframe, jnp.ndarray):
        # JAX: use functional update
        bias = bias.at[LEG_POSITION_INDICES].set(standing_keyframe[LEG_POSITION_INDICES])
    else:
        # NumPy: use standard indexing
        bias[LEG_POSITION_INDICES] = standing_keyframe[LEG_POSITION_INDICES]

    return bias


# ============================================================================
# Observation helpers
# ============================================================================

def obs_size_for_policy_type(
    base_obs_size: int,
    policy_type: PolicyType,
) -> int:
    """Compute observation size for a given policy type.

    Args:
        base_obs_size: Base observation size (e.g., 42 for BalanceEnv).
        policy_type: Policy architecture type.

    Returns:
        Total observation size including base_action if needed.

    Notes:
        - Pure PPO: obs_size = base_obs_size
        - Residual PPO: obs_size = base_obs_size + ACTION_DIM (includes base_action)
    """
    if policy_type == PolicyType.PURE_PPO:
        return base_obs_size
    elif policy_type == PolicyType.RESIDUAL_PPO:
        return base_obs_size + ACTION_DIM
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def extract_base_action_from_obs(
    obs: np.ndarray | jnp.ndarray,
    base_obs_size: int,
) -> np.ndarray | jnp.ndarray:
    """Extract base_action from residual PPO observation.

    Args:
        obs: Observation array, shape (..., base_obs_size + ACTION_DIM).
        base_obs_size: Base observation size (e.g., 42 for BalanceEnv).

    Returns:
        base_action: Extracted base action, shape (..., ACTION_DIM).

    Raises:
        ValueError: If obs size is too small to contain base_action.
    """
    expected_size = base_obs_size + ACTION_DIM
    if obs.shape[-1] < expected_size:
        raise ValueError(
            f"obs size {obs.shape[-1]} is too small to contain base_action. "
            f"Expected at least {expected_size} (base_obs_size={base_obs_size} + ACTION_DIM={ACTION_DIM})"
        )

    return obs[..., base_obs_size:base_obs_size + ACTION_DIM]


# ============================================================================
# Documentation strings
# ============================================================================

ACTION_SEMANTICS_DOC = """
Action semantics in wheeled biped control:

1. **Absolute normalized action**: The action value after all composition, in [-1, 1].
   This is what the PID controller receives (after smoothing/delay/bias).

2. **Pre-bias action**: The action before pid_action_bias is added.
   - Pure PPO: policy output = pre-bias action
   - Residual PPO: base_action + residual_scale * residual_action = pre-bias action

3. **PID action bias**: A constant offset added to leg joints only, shifting action=0
   to the standing keyframe. Wheels have zero bias.

4. **Action pipeline** (BalanceEnv):
   policy → clip → smooth → delay → add pid_action_bias → PID → torque

5. **Residual composition** (future ResidualBalanceEnv):
   base_action + residual_scale * residual_action → clip → smooth → delay → add pid_action_bias → PID → torque

Key invariants:
- Action dimension is always 10 (8 legs + 2 wheels).
- Leg joints use position targets, wheels use velocity targets.
- PID action bias is only added to leg joints, not wheels.
- Clipping happens before smoothing/delay to ensure valid range.
- The policy never sees pid_action_bias; it's an env-internal offset.
"""

RESIDUAL_CONTROL_DOC = """
Residual control architecture:

The residual control framework composes a structured base controller (LQR/IK)
with a learned residual policy (PPO) to achieve robust balance and transitions.

Components:
1. **Base controller**: Height-dependent gain-scheduled LQR/IK prior.
   - Outputs base_action_abs (absolute normalized action).
   - Provides nominal postural control for different heights.
   - Does not require training, uses classical control theory.

2. **Residual policy**: Bounded PPO policy.
   - Outputs residual_action (bounded correction in [-1, 1]).
   - Observes base_action_abs as part of observation.
   - Learns to correct base controller errors and handle disturbances.

3. **Composition**: final_action_abs = clip(base_action_abs + residual_scale * residual_action, -1, 1)
   - residual_scale bounds the correction magnitude (typically 0.1-0.5).
   - Clipping ensures final action stays in valid range.
   - Composition happens before smoothing/delay/bias.

Benefits:
- Structured prior reduces exploration burden.
- Bounded residual prevents policy from ignoring prior.
- Graceful degradation: if policy fails, base controller provides fallback.
- Interpretable: can analyze base vs residual contributions.

Observation design:
- Pure PPO: obs_size = base_obs_size (e.g., 42)
- Residual PPO: obs_size = base_obs_size + ACTION_DIM (e.g., 52)
  - Includes base_action_abs so policy can condition on prior.
"""
