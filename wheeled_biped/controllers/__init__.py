"""
Classical baseline controllers for the wheeled bipedal robot.

Available controllers
---------------------
LQRBalanceController
    LQR sagittal balance (TWIP model) + FK-scan height IK + PD lateral balance.
    Intended as a fair, non-RL comparison baseline for Stages 1–3 of the
    balance curriculum (narrow-band, widened, and variable-height standing).
    See ``wheeled_biped/controllers/lqr_balance.py`` for full design rationale.
"""

from wheeled_biped.controllers.lqr_balance import LQRBalanceController

__all__ = ["LQRBalanceController"]
