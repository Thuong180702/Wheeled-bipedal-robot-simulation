"""Phase B.9 balanced root solver using constrained optimization.

Finds physically balanced initial root poses (root_x, root_roll, root_z) for each B9 height
so that both wheels touch ground with equal contact forces and near-zero CoM lateral offset.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_posture_geometry_inspection import (  # noqa: E402
    VIEWS,
    body_com,
    contact_forces_by_wheel,
    render_pose,
    wheel_bottom_heights,
)
from wheeled_biped.controllers.dual_rate_balance_controller import (  # noqa: E402
    DualRateConfig,
)
from wheeled_biped.controllers.height_ik import HeightIK  # noqa: E402
from wheeled_biped.utils.config import get_model_path  # noqa: E402

VALID_HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_balanced_root_solver"
POSTURE_DIR = PROJECT_ROOT / "outputs" / "phase_b9_posture_balanced_root"
CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "b9_balanced_root_init_table.yaml"


@dataclass
class BalancedRootResult:
    height: float
    root_x: float
    root_z: float
    root_roll: float
    root_pitch: float
    hip_pitch: float
    knee: float
    left_clearance: float
    right_clearance: float
    left_force: float
    right_force: float
    force_diff: float
    com_x: float
    com_lateral_offset: float
    clearance_diff: float
    both_wheels_loaded: bool
    optimization_success: bool
    optimization_message: str


def quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion to roll, pitch, yaw in radians."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = math.atan2(r[2, 1], r[2, 2])
    pitch = math.atan2(-r[2, 0], math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = math.atan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll, pitch, yaw to quaternion."""
    quat = np.zeros(4)
    euler = np.array([roll, pitch, yaw])
    mujoco.mju_euler2Quat(quat, euler, b"xyz")
    return quat


def settle_physics(model: mujoco.MjModel, data: mujoco.MjData, steps: int = 5) -> None:
    """Run a few physics steps to settle contact forces, keeping velocities zero."""
    for _ in range(steps):
        data.qvel[:] = 0.0
        mujoco.mj_step(model, data)


def evaluate_balance_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    settle_steps: int = 5,
) -> dict[str, float]:
    """Evaluate balance metrics after short settling period."""
    settle_physics(model, data, settle_steps)

    left_bottom, right_bottom = wheel_bottom_heights(model, data)
    left_force, right_force = contact_forces_by_wheel(model, data)
    com = body_com(model, data)

    left_clearance = float(left_bottom)
    right_clearance = float(right_bottom)
    clearance_diff = abs(left_clearance - right_clearance)

    wheel_contact_x = 0.5 * (
        data.geom_xpos[model.geom("l_wheel_collision").id, 0]
        + data.geom_xpos[model.geom("r_wheel_collision").id, 0]
    )
    com_lateral_offset = float(com[0] - wheel_contact_x)

    roll, pitch, _ = quat_to_rpy(data.qpos[3:7].copy())

    return {
        "left_clearance": left_clearance,
        "right_clearance": right_clearance,
        "clearance_diff": clearance_diff,
        "left_force": left_force,
        "right_force": right_force,
        "force_diff": abs(left_force - right_force),
        "com_x": float(com[0]),
        "com_lateral_offset": com_lateral_offset,
        "root_roll": roll,
        "root_pitch": pitch,
    }


def objective_function(
    x: np.ndarray,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: np.ndarray,
    weights: dict[str, float],
) -> float:
    """Objective function for balanced root optimization.

    x = [root_x, root_roll, root_z]
    """
    root_x, root_roll, root_z = x

    data.qpos[0] = root_x
    data.qpos[2] = root_z
    data.qpos[3:7] = rpy_to_quat(root_roll, 0.0, 0.0)
    data.qpos[7:17] = joint_targets
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)

    metrics = evaluate_balance_metrics(model, data, settle_steps=5)

    force_diff = metrics["force_diff"]
    clearance_diff = metrics["clearance_diff"]
    left_clearance = metrics["left_clearance"]
    right_clearance = metrics["right_clearance"]
    com_lateral = metrics["com_lateral_offset"]
    roll = metrics["root_roll"]
    pitch = metrics["root_pitch"]

    unload_penalty = 0.0
    if metrics["left_force"] < 0.1 or metrics["right_force"] < 0.1:
        unload_penalty = 1000.0

    if left_clearance > 1e-3 or right_clearance > 1e-3:
        unload_penalty += 500.0 * (max(left_clearance, right_clearance))

    cost = (
        weights["force"] * force_diff ** 2
        + weights["clearance"] * (left_clearance ** 2 + right_clearance ** 2)
        + weights["clearance_diff"] * clearance_diff ** 2
        + weights["com"] * com_lateral ** 2
        + weights["roll"] * roll ** 2
        + weights["pitch"] * pitch ** 2
        + weights["unload"] * unload_penalty
    )

    return cost


def solve_balanced_root(
    height: float,
    model: mujoco.MjModel,
    config: DualRateConfig,
) -> BalancedRootResult:
    """Solve for balanced root pose at given height using constrained optimization."""

    height_ik = HeightIK(
        mj_model=model,
        scan_points=config.ik_scan_points,
        polynomial_degree=config.ik_polynomial_degree,
        symmetric_fold=config.ik_symmetric_fold,
    )
    targets = height_ik.compute_ik_targets(height)
    hip_pitch = float(targets["hip_pitch"])
    knee = float(targets["knee"])

    joint_targets = np.array([
        0.0, 0.0, hip_pitch, knee, 0.0,
        0.0, 0.0, hip_pitch, knee, 0.0,
    ])

    data = mujoco.MjData(model)

    data.qpos[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:17] = joint_targets
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    left_bottom, right_bottom = wheel_bottom_heights(model, data)
    initial_root_z = 1.0 - max(left_bottom, right_bottom) - 1e-3

    x0 = np.array([0.0, 0.0, initial_root_z])

    bounds = [
        (-0.05, 0.05),
        (-0.03, 0.03),
        (initial_root_z - 0.02, initial_root_z + 0.02),
    ]

    weights = {
        "force": 1.0,
        "clearance": 100.0,
        "clearance_diff": 50.0,
        "com": 10.0,
        "roll": 20.0,
        "pitch": 20.0,
        "unload": 1.0,
    }

    result = minimize(
        objective_function,
        x0,
        args=(model, data, joint_targets, weights),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-9},
    )

    root_x_opt, root_roll_opt, root_z_opt = result.x

    data.qpos[0] = root_x_opt
    data.qpos[2] = root_z_opt
    data.qpos[3:7] = rpy_to_quat(root_roll_opt, 0.0, 0.0)
    data.qpos[7:17] = joint_targets
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    final_metrics = evaluate_balance_metrics(model, data, settle_steps=5)

    both_loaded = (
        final_metrics["left_force"] > 0.1
        and final_metrics["right_force"] > 0.1
        and final_metrics["left_clearance"] < 1e-3
        and final_metrics["right_clearance"] < 1e-3
    )

    return BalancedRootResult(
        height=height,
        root_x=root_x_opt,
        root_z=root_z_opt,
        root_roll=root_roll_opt,
        root_pitch=final_metrics["root_pitch"],
        hip_pitch=hip_pitch,
        knee=knee,
        left_clearance=final_metrics["left_clearance"],
        right_clearance=final_metrics["right_clearance"],
        left_force=final_metrics["left_force"],
        right_force=final_metrics["right_force"],
        force_diff=final_metrics["force_diff"],
        com_x=final_metrics["com_x"],
        com_lateral_offset=final_metrics["com_lateral_offset"],
        clearance_diff=final_metrics["clearance_diff"],
        both_wheels_loaded=both_loaded,
        optimization_success=result.success,
        optimization_message=result.message,
    )


def render_balanced_posture(
    height: float,
    result: BalancedRootResult,
    model: mujoco.MjModel,
) -> None:
    """Render balanced posture from all views."""
    joint_targets = np.array([
        0.0, 0.0, result.hip_pitch, result.knee, 0.0,
        0.0, 0.0, result.hip_pitch, result.knee, 0.0,
    ])

    data = mujoco.MjData(model)
    data.qpos[0] = result.root_x
    data.qpos[2] = result.root_z
    data.qpos[3:7] = rpy_to_quat(result.root_roll, 0.0, 0.0)
    data.qpos[7:17] = joint_targets
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    for view in VIEWS.values():
        output_path = POSTURE_DIR / f"balanced_{view.name}_h_{height:.2f}.png"
        render_pose(model, data, output_path, view)


def save_balanced_init_table(results: list[BalancedRootResult]) -> None:
    """Save balanced initialization table as YAML config."""
    config_data = {
        "balanced_root_initialization": {
            "description": "Optimized root poses for B9 postures with balanced wheel contact forces",
            "heights": {},
        }
    }

    for r in results:
        config_data["balanced_root_initialization"]["heights"][f"{r.height:.2f}"] = {
            "root_x": float(r.root_x),
            "root_z": float(r.root_z),
            "root_roll": float(r.root_roll),
            "root_pitch": float(r.root_pitch),
            "hip_pitch": float(r.hip_pitch),
            "knee": float(r.knee),
            "expected_left_clearance": float(r.left_clearance),
            "expected_right_clearance": float(r.right_clearance),
            "expected_left_force": float(r.left_force),
            "expected_right_force": float(r.right_force),
            "expected_com_lateral_offset": float(r.com_lateral_offset),
        }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    POSTURE_DIR.mkdir(parents=True, exist_ok=True)

    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    config = DualRateConfig.from_yaml(
        PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
    )

    results: list[BalancedRootResult] = []
    summary_rows: list[dict[str, Any]] = []

    print("Solving balanced root poses for B9 heights...")
    for height in VALID_HEIGHTS:
        print(f"\nHeight {height:.2f} m:")
        result = solve_balanced_root(height, model, config)
        results.append(result)

        print(f"  Optimization: {'SUCCESS' if result.optimization_success else 'FAILED'}")
        print(f"  Root x: {result.root_x:.6f} m")
        print(f"  Root z: {result.root_z:.6f} m")
        print(f"  Root roll: {math.degrees(result.root_roll):.4f} deg")
        print(f"  Left clearance: {result.left_clearance:.6f} m")
        print(f"  Right clearance: {result.right_clearance:.6f} m")
        print(f"  Clearance diff: {result.clearance_diff:.6f} m")
        print(f"  Left force: {result.left_force:.3f} N")
        print(f"  Right force: {result.right_force:.3f} N")
        print(f"  Force diff: {result.force_diff:.3f} N")
        print(f"  CoM lateral offset: {result.com_lateral_offset:.6f} m")
        print(f"  Both wheels loaded: {result.both_wheels_loaded}")

        summary_rows.append({
            "height": height,
            "root_x": result.root_x,
            "root_z": result.root_z,
            "root_roll_rad": result.root_roll,
            "root_roll_deg": math.degrees(result.root_roll),
            "root_pitch_rad": result.root_pitch,
            "root_pitch_deg": math.degrees(result.root_pitch),
            "hip_pitch": result.hip_pitch,
            "knee": result.knee,
            "left_clearance": result.left_clearance,
            "right_clearance": result.right_clearance,
            "clearance_diff": result.clearance_diff,
            "left_force": result.left_force,
            "right_force": result.right_force,
            "force_diff": result.force_diff,
            "com_x": result.com_x,
            "com_lateral_offset": result.com_lateral_offset,
            "both_wheels_loaded": result.both_wheels_loaded,
            "optimization_success": result.optimization_success,
            "optimization_message": result.optimization_message,
        })

        render_balanced_posture(height, result, model)

    summary_path = OUTPUT_DIR / "balanced_root_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    save_balanced_init_table(results)

    debug_data = {
        "model_path": str(model_path),
        "valid_heights": VALID_HEIGHTS,
        "optimization_method": "L-BFGS-B",
        "settle_steps": 5,
        "results": [
            {
                "height": r.height,
                "success": r.optimization_success,
                "message": r.optimization_message,
                "force_diff": r.force_diff,
                "clearance_diff": r.clearance_diff,
            }
            for r in results
        ],
    }

    with open(OUTPUT_DIR / "optimization_debug.json", "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2)

    print(f"\n[OK] Saved balanced root summary to {summary_path}")
    print(f"[OK] Saved balanced init table to {CONFIG_PATH}")
    print(f"[OK] Saved balanced posture renders to {POSTURE_DIR}")
    print(f"[OK] Saved optimization debug to {OUTPUT_DIR / 'optimization_debug.json'}")


if __name__ == "__main__":
    main()
