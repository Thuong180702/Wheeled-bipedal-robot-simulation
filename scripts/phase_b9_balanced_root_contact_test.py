"""Phase B.9 balanced root contact diagnostic.

Runs 3-mode contact test for each B9 height to verify balanced root initialization:
- Mode A: t=0 initialized state
- Mode B: passive/contact-only settling for 50 steps
- Mode C: PID-hold settling for 50 steps, no wheel LQR

Logs contact forces, clearances, CoM offset, root orientation, and stability indicators.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase_b9_posture_geometry_inspection import (  # noqa: E402
    body_com,
    contact_forces_by_wheel,
    wheel_bottom_heights,
)
from wheeled_biped.controllers.dual_rate_balance_controller import (  # noqa: E402
    DualRateConfig,
)
from wheeled_biped.utils.config import get_model_path  # noqa: E402

VALID_HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase_b9_balanced_root_contact_test"
CONFIG_PATH = PROJECT_ROOT / "configs" / "controllers" / "b9_balanced_root_init_table.yaml"


@dataclass
class ContactTestResult:
    height: float
    mode: str
    step: int
    left_force: float
    right_force: float
    force_diff: float
    left_clearance: float
    right_clearance: float
    clearance_diff: float
    com_x: float
    com_lateral_offset: float
    root_roll: float
    root_pitch: float
    left_wheel_unloaded: bool
    right_wheel_unloaded: bool
    roll_drift_rate: float


def quat_to_rpy(quat: np.ndarray) -> tuple[float, float, float]:
    """Convert quaternion to roll, pitch, yaw in radians."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    roll = np.arctan2(r[2, 1], r[2, 2])
    pitch = np.arctan2(-r[2, 0], np.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2))
    yaw = np.arctan2(r[1, 0], r[0, 0])
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll, pitch, yaw to quaternion."""
    quat = np.zeros(4)
    euler = np.array([roll, pitch, yaw])
    mujoco.mju_euler2Quat(quat, euler, b"xyz")
    return quat


def load_balanced_init_table() -> dict:
    """Load balanced root initialization table."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["balanced_root_initialization"]["heights"]


def initialize_from_table(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: float,
    init_table: dict,
) -> None:
    """Initialize robot from balanced root table."""
    height_key = f"{height:.2f}"
    if height_key not in init_table:
        raise ValueError(f"Height {height} not in balanced init table")

    init = init_table[height_key]

    # Set root pose
    data.qpos[0] = init["root_x"]
    data.qpos[2] = init["root_z"]
    data.qpos[3:7] = rpy_to_quat(init["root_roll"], init["root_pitch"], 0.0)

    # Set joint positions
    hip_pitch = init["hip_pitch"]
    knee = init["knee"]
    joint_targets = np.array([
        0.0, 0.0, hip_pitch, knee, 0.0,
        0.0, 0.0, hip_pitch, knee, 0.0,
    ])
    data.qpos[7:17] = joint_targets

    # Zero velocities
    data.qvel[:] = 0.0

    mujoco.mj_forward(model, data)


def evaluate_contact_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, float]:
    """Evaluate contact state metrics."""
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
        "left_force": left_force,
        "right_force": right_force,
        "force_diff": abs(left_force - right_force),
        "left_clearance": left_clearance,
        "right_clearance": right_clearance,
        "clearance_diff": clearance_diff,
        "com_x": float(com[0]),
        "com_lateral_offset": com_lateral_offset,
        "root_roll": roll,
        "root_pitch": pitch,
        "left_wheel_unloaded": left_force < 0.1,
        "right_wheel_unloaded": right_force < 0.1,
    }


def run_mode_a(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: float,
    init_table: dict,
) -> ContactTestResult:
    """Mode A: t=0 initialized state."""
    initialize_from_table(model, data, height, init_table)
    metrics = evaluate_contact_state(model, data)

    return ContactTestResult(
        height=height,
        mode="A_t0",
        step=0,
        left_force=metrics["left_force"],
        right_force=metrics["right_force"],
        force_diff=metrics["force_diff"],
        left_clearance=metrics["left_clearance"],
        right_clearance=metrics["right_clearance"],
        clearance_diff=metrics["clearance_diff"],
        com_x=metrics["com_x"],
        com_lateral_offset=metrics["com_lateral_offset"],
        root_roll=metrics["root_roll"],
        root_pitch=metrics["root_pitch"],
        left_wheel_unloaded=metrics["left_wheel_unloaded"],
        right_wheel_unloaded=metrics["right_wheel_unloaded"],
        roll_drift_rate=0.0,
    )


def run_mode_b(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: float,
    init_table: dict,
    steps: int = 50,
) -> list[ContactTestResult]:
    """Mode B: passive/contact-only settling."""
    initialize_from_table(model, data, height, init_table)

    results = []
    initial_roll = quat_to_rpy(data.qpos[3:7].copy())[0]

    for step in range(steps + 1):
        metrics = evaluate_contact_state(model, data)
        current_roll = metrics["root_roll"]
        roll_drift_rate = (current_roll - initial_roll) / (step * model.opt.timestep) if step > 0 else 0.0

        results.append(ContactTestResult(
            height=height,
            mode="B_passive",
            step=step,
            left_force=metrics["left_force"],
            right_force=metrics["right_force"],
            force_diff=metrics["force_diff"],
            left_clearance=metrics["left_clearance"],
            right_clearance=metrics["right_clearance"],
            clearance_diff=metrics["clearance_diff"],
            com_x=metrics["com_x"],
            com_lateral_offset=metrics["com_lateral_offset"],
            root_roll=current_roll,
            root_pitch=metrics["root_pitch"],
            left_wheel_unloaded=metrics["left_wheel_unloaded"],
            right_wheel_unloaded=metrics["right_wheel_unloaded"],
            roll_drift_rate=roll_drift_rate,
        ))

        if step < steps:
            # Passive step: zero velocities, let contacts settle
            data.qvel[:] = 0.0
            mujoco.mj_step(model, data)

    return results


def run_mode_c(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    height: float,
    init_table: dict,
    config: DualRateConfig,
    steps: int = 50,
) -> list[ContactTestResult]:
    """Mode C: PID-hold settling, no wheel LQR.

    Applies target joint positions directly to actuators and lets
    MuJoCo's built-in control handle the PD control.
    """
    initialize_from_table(model, data, height, init_table)

    # Get target joint positions from init table
    height_key = f"{height:.2f}"
    init = init_table[height_key]
    hip_pitch = init["hip_pitch"]
    knee = init["knee"]

    # Target positions for leg joints (no wheel commands)
    target_positions = np.array([
        0.0, 0.0, hip_pitch, knee, 0.0,
        0.0, 0.0, hip_pitch, knee, 0.0,
    ])

    results = []
    initial_roll = quat_to_rpy(data.qpos[3:7].copy())[0]

    for step in range(steps + 1):
        metrics = evaluate_contact_state(model, data)
        current_roll = metrics["root_roll"]
        roll_drift_rate = (current_roll - initial_roll) / (step * model.opt.timestep) if step > 0 else 0.0

        results.append(ContactTestResult(
            height=height,
            mode="C_pid_hold",
            step=step,
            left_force=metrics["left_force"],
            right_force=metrics["right_force"],
            force_diff=metrics["force_diff"],
            left_clearance=metrics["left_clearance"],
            right_clearance=metrics["right_clearance"],
            clearance_diff=metrics["clearance_diff"],
            com_x=metrics["com_x"],
            com_lateral_offset=metrics["com_lateral_offset"],
            root_roll=current_roll,
            root_pitch=metrics["root_pitch"],
            left_wheel_unloaded=metrics["left_wheel_unloaded"],
            right_wheel_unloaded=metrics["right_wheel_unloaded"],
            roll_drift_rate=roll_drift_rate,
        ))

        if step < steps:
            # Apply target positions directly to actuators
            # MuJoCo's built-in actuators will handle PD control
            data.ctrl[:] = target_positions
            mujoco.mj_step(model, data)

    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Phase B.9 Balanced Root Contact Diagnostic]\n")

    # Load model and config
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    config = DualRateConfig.from_yaml(
        PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml"
    )

    # Load balanced init table
    print(f"Loading balanced init table: {CONFIG_PATH}")
    init_table = load_balanced_init_table()

    all_results: list[ContactTestResult] = []

    for height in VALID_HEIGHTS:
        print(f"\nHeight {height:.2f} m:")

        # Mode A: t=0
        print("  Mode A: t=0 initialized state")
        data_a = mujoco.MjData(model)
        result_a = run_mode_a(model, data_a, height, init_table)
        all_results.append(result_a)
        print(f"    Force diff: {result_a.force_diff:.3f} N")
        print(f"    Clearance diff: {result_a.clearance_diff:.6f} m")
        print(f"    CoM lateral offset: {result_a.com_lateral_offset:.6f} m")

        # Mode B: passive settling
        print("  Mode B: passive/contact-only settling (50 steps)")
        data_b = mujoco.MjData(model)
        results_b = run_mode_b(model, data_b, height, init_table, steps=50)
        all_results.extend(results_b)
        final_b = results_b[-1]
        print(f"    Final force diff: {final_b.force_diff:.3f} N")
        print(f"    Final roll drift rate: {final_b.roll_drift_rate:.6f} rad/s")
        print(f"    Any wheel unloaded: {final_b.left_wheel_unloaded or final_b.right_wheel_unloaded}")

        # Mode C: PID-hold settling
        print("  Mode C: PID-hold settling (50 steps, no wheel LQR)")
        data_c = mujoco.MjData(model)
        results_c = run_mode_c(model, data_c, height, init_table, config, steps=50)
        all_results.extend(results_c)
        final_c = results_c[-1]
        print(f"    Final force diff: {final_c.force_diff:.3f} N")
        print(f"    Final roll drift rate: {final_c.roll_drift_rate:.6f} rad/s")
        print(f"    Any wheel unloaded: {final_c.left_wheel_unloaded or final_c.right_wheel_unloaded}")

    # Save per-height detailed results
    per_height_path = OUTPUT_DIR / "contact_test_per_height.csv"
    with open(per_height_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "height", "mode", "step",
            "left_force", "right_force", "force_diff",
            "left_clearance", "right_clearance", "clearance_diff",
            "com_x", "com_lateral_offset",
            "root_roll", "root_pitch",
            "left_wheel_unloaded", "right_wheel_unloaded",
            "roll_drift_rate",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                "height": r.height,
                "mode": r.mode,
                "step": r.step,
                "left_force": r.left_force,
                "right_force": r.right_force,
                "force_diff": r.force_diff,
                "left_clearance": r.left_clearance,
                "right_clearance": r.right_clearance,
                "clearance_diff": r.clearance_diff,
                "com_x": r.com_x,
                "com_lateral_offset": r.com_lateral_offset,
                "root_roll": r.root_roll,
                "root_pitch": r.root_pitch,
                "left_wheel_unloaded": r.left_wheel_unloaded,
                "right_wheel_unloaded": r.right_wheel_unloaded,
                "roll_drift_rate": r.roll_drift_rate,
            })

    # Save summary (final state of each mode for each height)
    summary_rows = []
    for height in VALID_HEIGHTS:
        mode_a = [r for r in all_results if r.height == height and r.mode == "A_t0"][0]
        mode_b_final = [r for r in all_results if r.height == height and r.mode == "B_passive"][-1]
        mode_c_final = [r for r in all_results if r.height == height and r.mode == "C_pid_hold"][-1]

        summary_rows.append({
            "height": height,
            "mode_a_force_diff": mode_a.force_diff,
            "mode_a_clearance_diff": mode_a.clearance_diff,
            "mode_a_com_offset": mode_a.com_lateral_offset,
            "mode_b_final_force_diff": mode_b_final.force_diff,
            "mode_b_final_roll_drift": mode_b_final.roll_drift_rate,
            "mode_b_any_unloaded": mode_b_final.left_wheel_unloaded or mode_b_final.right_wheel_unloaded,
            "mode_c_final_force_diff": mode_c_final.force_diff,
            "mode_c_final_roll_drift": mode_c_final.roll_drift_rate,
            "mode_c_any_unloaded": mode_c_final.left_wheel_unloaded or mode_c_final.right_wheel_unloaded,
        })

    summary_path = OUTPUT_DIR / "contact_test_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[OK] Saved per-height results: {per_height_path}")
    print(f"[OK] Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
