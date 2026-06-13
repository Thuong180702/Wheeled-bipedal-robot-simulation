"""Gather dynamic telemetry during boundary height hold failure.

Phase 1B: Dynamic diagnostics to understand drift mechanism.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Any

import mujoco
import jax
import jax.numpy as jnp

from wheeled_biped.controllers.balance_core_controller import BalanceCoreController
from wheeled_biped.sim.mujoco_interface import MuJoCoInterface


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
DT = 0.01
STEPS = 1000  # 10 seconds


def load_boundary_setup(setup_path: str) -> dict[str, Any]:
    """Load boundary setup JSON."""
    with open(setup_path, 'r') as f:
        return json.load(f)


def run_boundary_dynamic_test(
    setup_path: str,
    variant_name: str,
    output_dir: Path,
) -> dict:
    """Run short dynamic simulation with rich telemetry."""
    print(f"\n{'='*80}")
    print(f"DYNAMIC TELEMETRY: {variant_name}")
    print(f"{'='*80}\n")

    # Load setup
    setup = load_boundary_setup(setup_path)
    target_height = setup["target_com_z_m"]
    print(f"Target height: {target_height:.4f} m")

    # Load MuJoCo
    mj_model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    mj_data = mujoco.MjData(model)

    # Apply boundary setup
    joint_pos = setup["equilibrium_joint_pos"]
    mj_data.qpos[7:17] = joint_pos
    mj_data.qpos[0:3] = [0.0, 0.0, setup["calibrated_root_z_m"]]
    mj_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    print(f"Applied boundary setup")
    print(f"  Root Z: {mj_data.qpos[2]:.6f} m")
    print(f"  CoM Z: {float(mj_data.subtree_com[1][2]):.6f} m\n")

    # Initialize controller
    # __CONTINUE_HERE__
