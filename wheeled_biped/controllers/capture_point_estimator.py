"""Capture point estimation using height-dependent Linear Inverted Pendulum model."""

import chex
from jax import Array
from wheeled_biped.controllers.centroidal_state_estimator import CentroidalState


@chex.dataclass
class CapturePointEstimatorConfig:
    """Configuration for capture point estimator."""
    gravity: float = 9.81  # Gravitational acceleration (m/s²)


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
        # Placeholder implementation (will be completed in Task 5)
        return state
