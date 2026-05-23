"""Static balance controller wrapper.

Cancels WBC static equilibrium bias by computing static reference torques
once at initialization, then removing equilibrium bias at runtime.
"""

import mujoco
import numpy as np
from numpy.typing import NDArray

# Import existing calibration helper from simulate_hierarchical_controller
# Do not duplicate - reuse the tested implementation
from scripts.simulate_hierarchical_controller import calibrate_root_z_for_wheel_floor_contact


class StaticBalanceController:
    """Wrapper that cancels WBC static equilibrium bias."""

    def __init__(
        self,
        mj_model,
        mj_data,
        wbc_pipeline,
        calibration_config: dict | None = None,
    ):
        """Initialize with calibrated equilibrium references.

        Args:
            mj_model: MuJoCo model
            mj_data: MuJoCo data (will be copied, not mutated)
            wbc_pipeline: Existing WBC pipeline to wrap
            calibration_config: Config for calibrated initialization
        """
        self.mj_model = mj_model
        self.wbc_pipeline = wbc_pipeline
        self.calibration_config = calibration_config or {}

        # Will be computed in initialization
        self.tau_static_ref = None
        self.tau_wbc_equilibrium = None
        self.equilibrium_state = None
        self.qfrc_inverse_ref = None
        self.qfrc_bias_ref = None
        self.qfrc_constraint_ref = None

        # Compute references using copied data
        self._compute_equilibrium_references(mj_data)

    def _compute_equilibrium_references(self, mj_data):
        """Compute static reference torques at calibrated equilibrium."""
        # TODO: Implement in next step
        pass

    def wrap(
        self,
        tau_wbc_current: NDArray,
        current_state: dict,
    ) -> tuple[NDArray, dict]:
        """Wrap WBC torque to remove equilibrium bias.

        Args:
            tau_wbc_current: Current WBC output (10,)
            current_state: Current robot state for error metrics

        Returns:
            tau_wbc_wrapped: Bias-corrected WBC torque (10,)
            telemetry: Dict with all diagnostic values
        """
        # TODO: Implement in later step
        pass
