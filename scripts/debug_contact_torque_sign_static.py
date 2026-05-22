"""Static-contact diagnostic for force-to-torque sign in wheeled-biped support.

Evaluates short-horizon dynamics under four torque cases at keyframe 0:
A: zero torque
B: +J^T f_up
C: -J^T f_up
D: leg position controller torque only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.leg_position_controller import LegPositionController


MODEL_PATH = "assets/robot/wheeled_biped_real.xml"


@dataclass
class CaseResult:
    root_z: float
    root_z_velocity: float
    root_z_accel: float
    hip_pitch_qacc_l: float
    knee_qacc_l: float
    hip_pitch_qacc_r: float
    knee_qacc_r: float
    total_contact_fz: float


def reset_to_static_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def measure_total_contact_fz(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    total_contact_force_z = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, contact_force)
        contact_frame = np.array(contact.frame).reshape(3, 3)
        force_world = contact_frame.T @ contact_force[:3]
        total_contact_force_z += force_world[2]
    return float(total_contact_force_z)


def evaluate_case(
    model: mujoco.MjModel,
    case_name: str,
    tau_cmd: np.ndarray,
    step_counts: list[int],
) -> dict[int, CaseResult]:
    results: dict[int, CaseResult] = {}

    for n_steps in step_counts:
        data = mujoco.MjData(model)
        reset_to_static_keyframe(model, data)

        for _ in range(n_steps):
            data.ctrl[:] = tau_cmd
            mujoco.mj_step(model, data)

        total_contact_fz = measure_total_contact_fz(model, data)

        # qacc indices: [base(0:6), joints(6:16)]
        hip_pitch_qacc_l = float(data.qacc[6 + 2])
        knee_qacc_l = float(data.qacc[6 + 3])
        hip_pitch_qacc_r = float(data.qacc[6 + 7])
        knee_qacc_r = float(data.qacc[6 + 8])

        results[n_steps] = CaseResult(
            root_z=float(data.qpos[2]),
            root_z_velocity=float(data.qvel[2]),
            root_z_accel=float(data.qacc[2]),
            hip_pitch_qacc_l=hip_pitch_qacc_l,
            knee_qacc_l=knee_qacc_l,
            hip_pitch_qacc_r=hip_pitch_qacc_r,
            knee_qacc_r=knee_qacc_r,
            total_contact_fz=total_contact_fz,
        )

    return results


def print_case_results(case_name: str, results: dict[int, CaseResult]) -> None:
    print(f"\n=== Case {case_name} ===")
    for n_steps, r in results.items():
        print(
            f"steps={n_steps:>2d} | "
            f"root_z={r.root_z:.6f} | vz={r.root_z_velocity:+.6f} | az={r.root_z_accel:+.6f} | "
            f"qacc_hp_l={r.hip_pitch_qacc_l:+.4f} qacc_kn_l={r.knee_qacc_l:+.4f} | "
            f"qacc_hp_r={r.hip_pitch_qacc_r:+.4f} qacc_kn_r={r.knee_qacc_r:+.4f} | "
            f"contact_fz_total={r.total_contact_fz:.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug static support sign for contact torque mapping")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    reset_to_static_keyframe(model, data)

    robot_mass = float(np.sum(model.body_mass))
    gravity = float(abs(model.opt.gravity[2]))
    weight = robot_mass * gravity

    contact_jacobian = ContactJacobian(model)
    j_left, j_right = contact_jacobian.compute_wheel_jacobians(data)

    f_up_left = jnp.array([0.0, 0.0, weight / 2.0])
    f_up_right = jnp.array([0.0, 0.0, weight / 2.0])

    tau_jtf = np.array(j_left.T @ f_up_left + j_right.T @ f_up_right)
    tau_plus = tau_jtf
    tau_minus = -tau_jtf

    leg_controller = LegPositionController()
    joint_pos = jnp.array(data.qpos[7:17])
    joint_vel = jnp.array(data.qvel[6:16])
    target_joint_pos = joint_pos
    tau_leg = np.array(leg_controller.compute_leg_torques(joint_pos, joint_vel, target_joint_pos))

    print("=" * 100)
    print("Static support sign diagnostic")
    print(f"Model: {MODEL_PATH}")
    print(f"Weight: {weight:.3f} N")
    print("Torque norms:")
    print(f"  ||+J^T f_up|| = {np.linalg.norm(tau_plus):.4f}")
    print(f"  ||-J^T f_up|| = {np.linalg.norm(tau_minus):.4f}")
    print(f"  ||tau_leg||   = {np.linalg.norm(tau_leg):.4f}")
    print("=" * 100)

    case_a = evaluate_case(model, "A (tau=0)", np.zeros(model.nu), args.steps)
    case_b = evaluate_case(model, "B (tau=+J^T f_up)", tau_plus, args.steps)
    case_c = evaluate_case(model, "C (tau=-J^T f_up)", tau_minus, args.steps)
    case_d = evaluate_case(model, "D (tau=leg_position_only)", tau_leg, args.steps)

    print_case_results("A (tau=0)", case_a)
    print_case_results("B (tau=+J^T f_up)", case_b)
    print_case_results("C (tau=-J^T f_up)", case_c)
    print_case_results("D (tau=leg_position_only)", case_d)


if __name__ == "__main__":
    main()
