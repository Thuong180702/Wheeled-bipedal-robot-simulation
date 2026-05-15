# wheeled_biped/controllers/posture_regularizer.py
"""Posture regularization for Level 3 stabilization.

Provides weak posture restoration with two-level gating (WBC error gate,
momentum coordinator gate) and per-joint deadbands with 20% authority budget.
"""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass(frozen=True)
class PostureRegularizerConfig:
    """Configuration for posture regularizer."""
    # Proportional gain
    k_posture: float = 2.0  # Weak compared to WBC gains

    # Per-joint deadbands (radians)
    hip_roll_deadband: float = 0.05  # ±2.9° - allow lateral sway
    hip_yaw_deadband: float = 0.03  # ±1.7° - tighter, yaw drift is bad
    hip_pitch_deadband: float = 0.08  # ±4.6° - allow squat variation
    knee_deadband: float = 0.10  # ±5.7° - allow knee bend variation
    wheel_deadband: float = 0.0  # wheels don't have posture target

    # Gating thresholds
    wbc_error_threshold: float = 0.3  # 30% of WBC capacity
    momentum_active_scale: float = 0.5  # 50% authority when momentum active

    # Authority budget
    posture_authority_budget: float = 0.2  # 20% of actuator range


class PostureRegularizer:
    """Posture regularizer for Level 3 stabilization."""

    def __init__(self, config: PostureRegularizerConfig):
        """Initialize posture regularizer.

        Args:
            config: PostureRegularizerConfig with gains and thresholds
        """
        self.config = config
