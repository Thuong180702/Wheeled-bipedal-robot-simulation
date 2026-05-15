"""Centroidal balance controller with integrated CoM and capture point tracking."""

import chex
import jax.numpy as jnp
from jax import Array


@chex.dataclass
class CentroidalBalanceConfig:
    """Configuration for centroidal balance controller."""
    # Roll stabilization (from Step 5.25)
    k_roll: float = 20.0
    k_roll_rate: float = 4.0

    # CoM regulation
    k_com_lateral: float = 15.0
    k_com_lateral_damping: float = 3.0
    k_com_sagittal: float = 10.0
    k_com_sagittal_damping: float = 2.0

    # Deadbands
    com_deadband_lateral: float = 0.02  # meters
    com_deadband_sagittal: float = 0.03  # meters

    # Authority budget
    wbc_authority_budget: float = 0.6  # 60% of actuator range


class CentroidalBalanceController:
    """Centroidal WBC with CoM regulation and capture point tracking."""

    def __init__(self, config: CentroidalBalanceConfig):
        self.config = config
