"""Capture point estimation using height-dependent Linear Inverted Pendulum model."""

import chex
import jax.numpy as jnp
from jax import Array
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


@chex.dataclass
class CapturePointEstimatorConfig:
    """Configuration for capture point estimator."""
    gravity: float = 9.81  # Gravitational acceleration (m/s²)
    min_height: float = 0.35  # Minimum CoM height for stability (m)


class CapturePointEstimator:
    """Computes capture point using height-dependent LIP model.

    The capture point is computed using the Linear Inverted Pendulum (LIP) model
    with height-varying natural frequency:

        ω(h) = √(g / h_com)
        x_cp = x_com + vx_com / ω(h)
        y_cp = y_com + vy_com / ω(h)

    where h_com is the current CoM height above ground.
    """

    def __init__(self, config: CapturePointEstimatorConfig):
        self.config = config

    def update(self, state: CentroidalState) -> CentroidalState:
        """Update capture point and divergence in the centroidal state.

        Args:
            state: CentroidalState with com_pos and com_vel populated

        Returns:
            Updated CentroidalState with capture_point and divergence computed
        """
        # Extract CoM height (z-component)
        h_com = state.com_pos[2]

        # Clamp height to avoid division by zero or instability
        h_com = jnp.maximum(h_com, self.config.min_height)

        # Compute height-dependent natural frequency
        # ω(h) = √(g / h)
        omega = jnp.sqrt(self.config.gravity / h_com)

        # Compute capture point in x-y plane
        # x_cp = x_com + vx_com / ω(h)
        # y_cp = y_com + vy_com / ω(h)
        x_cp = state.com_pos[0] + state.com_vel[0] / omega
        y_cp = state.com_pos[1] + state.com_vel[1] / omega

        capture_point = jnp.array([x_cp, y_cp])

        # Compute divergence (assuming support polygon center at origin)
        # For wheeled biped, support center is midpoint between wheels
        # Simplified: assume support at (0, 0) for now
        support_center = jnp.array([0.0, 0.0])

        # Divergence = (CoM - support) + velocity / ω
        # This is equivalent to: divergence = capture_point - support_center
        divergence = capture_point - support_center

        # Update state with computed values
        state = state.replace(
            capture_point=capture_point,
            divergence=divergence
        )

        return state
