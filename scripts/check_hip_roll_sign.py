"""Diagnose left/right hip-roll torque sign convention.

Compares each hip-roll pattern against a zero-control rollout of the same
duration to avoid treating mj_forward contact impulses as baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
ACTUATED_QPOS = slice(7, 17)
ACTUATED_QVEL = slice(6, 16)
L_HIP_ROLL = 0
R_HIP_ROLL = 5
ROLLOUT_STEPS = 10
CONTROL_DT = 0.01


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


def rollout(tau: np.ndarray, steps: int = ROLLOUT_STEPS) -> Measurement:
    model, data = reset_model()
    data.ctrl[:] = tau
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return measure(model, data)


def run_case(left_tau: float, right_tau: float) -> tuple[Measurement, Measurement, np.ndarray]:
    model, _ = reset_model()
    tau = np.zeros(model.nu)
    tau[L_HIP_ROLL] = left_tau
    tau[R_HIP_ROLL] = right_tau
    zero_after = rollout(np.zeros(model.nu))
    sign_after = rollout(tau)
    return zero_after, sign_after, tau


def print_case(name: str, zero_after: Measurement, sign_after: Measurement, tau: np.ndarray) -> None:
    delta_roll = sign_after.roll_deg - zero_after.roll_deg
    print(f"\ncase={name}")
    print(f"  tau_l_hip_roll={tau[L_HIP_ROLL]:+.2f} Nm")
    print(f"  tau_r_hip_roll={tau[R_HIP_ROLL]:+.2f} Nm")
    print(f"  response_delta_roll={delta_roll:+.4f} deg")
    print(f"  response_delta_pitch={sign_after.pitch_deg - zero_after.pitch_deg:+.4f} deg")
    print(f"  response_roll_rate_est={delta_roll / (ROLLOUT_STEPS * CONTROL_DT):+.4f} deg/s")
    print(f"  response_delta_left_fz={sign_after.left_fz - zero_after.left_fz:+.4f} N")
    print(f"  response_delta_right_fz={sign_after.right_fz - zero_after.right_fz:+.4f} N")
    print(f"  response_delta_total_fz={sign_after.total_fz - zero_after.total_fz:+.4f} N")
    print(f"  response_delta_l_hip_roll={sign_after.joint_pos[L_HIP_ROLL] - zero_after.joint_pos[L_HIP_ROLL]:+.6f} rad")
    print(f"  response_delta_r_hip_roll={sign_after.joint_pos[R_HIP_ROLL] - zero_after.joint_pos[R_HIP_ROLL]:+.6f} rad")
    print(f"  response_joint_vel_delta={np.array2string(sign_after.joint_vel - zero_after.joint_vel, precision=5, suppress_small=True)}")


def main() -> None:
    print("Hip-roll sign diagnostic")
    print(f"model={MODEL_PATH}")
    print(f"rollout_steps={ROLLOUT_STEPS}")

    torque = 5.0
    cases = [
        ("same_positive", torque, torque),
        ("same_negative", -torque, -torque),
        ("opposite_left_positive", torque, -torque),
        ("opposite_right_positive", -torque, torque),
    ]
    for name, left_tau, right_tau in cases:
        zero_after, sign_after, tau = run_case(left_tau, right_tau)
        print_case(name, zero_after, sign_after, tau)


if __name__ == "__main__":
    main()
