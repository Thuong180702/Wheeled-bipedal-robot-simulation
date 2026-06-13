"""Diagnose contact-Jacobian force-to-torque sign convention.

Compares tau = J^T f versus tau = -J^T f against a zero-control rollout of
the same duration to avoid treating mj_forward contact impulses as baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import mujoco
import numpy as np

from wheeled_biped.controllers.contact_jacobian import ContactJacobian
from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
ACTUATED_QPOS = slice(7, 17)
ACTUATED_QVEL = slice(6, 16)
ROLLOUT_STEPS = 10


@dataclass
class Measurement:
    roll_deg: float
    pitch_deg: float
    left_fz: float
    right_fz: float
    total_fz: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray


def reset_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)
    return model, data


def measure(model: mujoco.MjModel, data: mujoco.MjData) -> Measurement:
    left_fz = 0.0
    right_fz = 0.0

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        contact_force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, contact_force)
        frame = np.array(contact.frame).reshape(3, 3)
        force_world = frame.T @ contact_force[:3]

        if "l_wheel" in geom1_name or "l_wheel" in geom2_name:
            left_fz += float(force_world[2])
        if "r_wheel" in geom1_name or "r_wheel" in geom2_name:
            right_fz += float(force_world[2])

    roll, pitch, _ = compute_orientation_from_quaternion(np.array(data.qpos[3:7]))
    return Measurement(
        roll_deg=np.degrees(roll),
        pitch_deg=np.degrees(pitch),
        left_fz=left_fz,
        right_fz=right_fz,
        total_fz=left_fz + right_fz,
        joint_pos=np.array(data.qpos[ACTUATED_QPOS]),
        joint_vel=np.array(data.qvel[ACTUATED_QVEL]),
    )


def compute_mapped_tau(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    contact_jacobian = ContactJacobian(model)
    half_weight = 0.5 * float(np.sum(model.body_mass)) * abs(float(model.opt.gravity[2]))
    f_left = jnp.array([0.0, 0.0, half_weight])
    f_right = jnp.array([0.0, 0.0, half_weight])
    return np.array(contact_jacobian.map_contact_forces_to_torques(data, f_left, f_right))


def rollout(tau: np.ndarray, steps: int = ROLLOUT_STEPS) -> Measurement:
    model, data = reset_model()
    data.ctrl[:] = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return measure(model, data)


def run_case(sign: float) -> tuple[Measurement, Measurement, np.ndarray]:
    model, data = reset_model()
    tau = sign * compute_mapped_tau(model, data)
    zero_after = rollout(np.zeros(model.nu))
    sign_after = rollout(tau)
    return zero_after, sign_after, tau


def print_case(name: str, zero_after: Measurement, sign_after: Measurement, tau: np.ndarray) -> None:
    print(f"\ncase={name}")
    print(f"  tau_max_abs={np.max(np.abs(tau)):.4f} Nm")
    print(f"  tau={np.array2string(tau, precision=4, suppress_small=True)}")
    print(f"  response_delta_total_fz={sign_after.total_fz - zero_after.total_fz:+.4f} N")
    print(f"  response_delta_left_fz={sign_after.left_fz - zero_after.left_fz:+.4f} N")
    print(f"  response_delta_right_fz={sign_after.right_fz - zero_after.right_fz:+.4f} N")
    print(f"  response_delta_roll={sign_after.roll_deg - zero_after.roll_deg:+.4f} deg")
    print(f"  response_delta_pitch={sign_after.pitch_deg - zero_after.pitch_deg:+.4f} deg")
    print(f"  response_joint_delta={np.array2string(sign_after.joint_pos - zero_after.joint_pos, precision=5, suppress_small=True)}")
    print(f"  response_joint_vel_delta={np.array2string(sign_after.joint_vel - zero_after.joint_vel, precision=5, suppress_small=True)}")


def main() -> None:
    print("WBC torque-sign diagnostic")
    print(f"model={MODEL_PATH}")
    print(f"rollout_steps={ROLLOUT_STEPS}")

    for name, sign in [("positive_jtf", 1.0), ("negative_jtf", -1.0)]:
        zero_after, sign_after, tau = run_case(sign)
        print_case(name, zero_after, sign_after, tau)


if __name__ == "__main__":
    main()
