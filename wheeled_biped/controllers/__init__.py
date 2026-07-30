"""
Classical baseline controllers for the wheeled bipedal robot.

Available controllers
---------------------
LQRBalanceController
    LQR sagittal balance (TWIP model) + FK-scan height IK + PD lateral balance.
    Intended as a fair, non-RL comparison baseline for Stages 1–3 of the
    balance curriculum (narrow-band, widened, and variable-height standing).
    See ``wheeled_biped/controllers/lqr_balance.py`` for full design rationale.

balance_core_types
    Type definitions and constants for the balance-core controller architecture.

Balance-core components
-----------------------
BalanceCoreTorqueComposer
    Composes torques from multiple balance-core controllers.

ContactSupervisor
    Monitors contact state and provides degradation signals.

LateralRollBalanceController
    Lateral roll balance via hip roll torques.

SagittalWheelBalanceController
    Sagittal balance via wheel torques.

ShapePostureController
    Shape posture control via support-shape joint torques.

SupportFeedforwardController
    Support feedforward torques for hip pitch and knee joints.

TorqueOwnershipValidator
    Validates torque ownership across balance-core controllers.
"""

from wheeled_biped.controllers.lqr_balance import LQRBalanceController
from wheeled_biped.controllers.lqr_anti_windup import LQRIntegralAWController
from wheeled_biped.controllers.fair_lqr_torque import FairLQRTorqueController
from wheeled_biped.controllers.coupled_lqr_3d import CoupledLQR3DBalanceController
from wheeled_biped.controllers.full_lqr import FullStateLQRController
from wheeled_biped.controllers.coupled_lqr_3d_torque import CoupledLQR3DTorqueController
from wheeled_biped.controllers.pi_aw_baseline import PiAwController
from wheeled_biped.controllers import balance_core_types
from wheeled_biped.controllers.balance_core_torque_composer import BalanceCoreTorqueComposer
from wheeled_biped.controllers.contact_supervisor import ContactSupervisor
from wheeled_biped.controllers.lateral_roll_balance_controller import LateralRollBalanceController
from wheeled_biped.controllers.sagittal_wheel_balance_controller import SagittalWheelBalanceController
from wheeled_biped.controllers.shape_posture_controller import ShapePostureController
from wheeled_biped.controllers.support_feedforward_controller import SupportFeedforwardController
from wheeled_biped.controllers.torque_ownership_validator import TorqueOwnershipValidator
from .hip_yaw_ownership import HIP_YAW_MODE_OWNERS, OwnershipError, validate_ownership, hip_yaw_common_owner, hip_yaw_divergence_owner, hip_yaw_mode_ownership_violation

__all__ = [
    "LQRBalanceController",
    "LQRIntegralAWController",
    "FairLQRTorqueController",
    "CoupledLQR3DBalanceController",
    "FullStateLQRController",
    "CoupledLQR3DTorqueController",
    "PiAwController",
    "balance_core_types",
    "BalanceCoreTorqueComposer",
    "ContactSupervisor",
    "LateralRollBalanceController",
    "SagittalWheelBalanceController",
    "ShapePostureController",
    "SupportFeedforwardController",
    "TorqueOwnershipValidator",
    "HIP_YAW_MODE_OWNERS",
    "OwnershipError",
    "validate_ownership",
    "hip_yaw_common_owner",
    "hip_yaw_divergence_owner",
    "hip_yaw_mode_ownership_violation",
]
