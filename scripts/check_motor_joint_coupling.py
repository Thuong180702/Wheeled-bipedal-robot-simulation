"""Diagnose actuator-to-joint response and left/right motor coupling."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from wheeled_biped.controllers.orientation_utils import compute_orientation_from_quaternion

MODEL_PATH = "assets/robot/wheeled_biped_real.xml"
ACTUATED_QPOS = slice(7, 17)
ACTUATED_QVEL = slice(6, 16)
ROLLOUT_STEPS = 10
TORQUE_NM = 5.0

JOINT_LABELS = [
    "l_hip_roll",
    "l_hip_yaw",
    "l_hip_pitch",
    "l_knee",
    "l_wheel",
    "r_hip_roll",
    "r_hip_yaw",
    "r_hip_pitch",
    "r_knee",
    "r_wheel",
]
PAIR_INDICES = [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)]


@dataclass
class Measurement:
    roll_deg: float
    pitch_deg: float
    left_fz: float
    right_fz: float
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
        roll_deg=float(np.degrees(roll)),
        pitch_deg=float(np.degrees(pitch)),
        left_fz=left_fz,
        right_fz=right_fz,
        joint_pos=np.array(data.qpos[ACTUATED_QPOS]),
        joint_vel=np.array(data.qvel[ACTUATED_QVEL]),
    )


def rollout(ctrl: np.ndarray, steps: int = ROLLOUT_STEPS) -> Measurement:
    model, data = reset_model()
    data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return measure(model, data)


def actuator_joint_names(model: mujoco.MjModel) -> list[tuple[str, str]]:
    names = []
    for i in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or f"actuator_{i}"
        joint_id = int(model.actuator_trnid[i, 0])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        names.append((actuator_name, joint_name))
    return names


def print_single_motor_cases() -> None:
    model, _ = reset_model()
    zero = rollout(np.zeros(model.nu))
    names = actuator_joint_names(model)

    print("\nSingle-actuator pulse response")
    print(f"torque_nm={TORQUE_NM}")
    for idx, (actuator_name, joint_name) in enumerate(names):
        ctrl = np.zeros(model.nu)
        ctrl[idx] = TORQUE_NM
        pos = rollout(ctrl)
        ctrl[idx] = -TORQUE_NM
        neg = rollout(ctrl)

        pos_joint_delta = pos.joint_pos - zero.joint_pos
        neg_joint_delta = neg.joint_pos - zero.joint_pos
        pos_vel_delta = pos.joint_vel - zero.joint_vel
        neg_vel_delta = neg.joint_vel - zero.joint_vel
        dominant_idx = int(np.argmax(np.abs(pos_joint_delta)))

        print(f"\nidx={idx} actuator={actuator_name} joint={joint_name}")
        print(f"  +tau_primary_joint_delta={pos_joint_delta[idx]:+.6f} rad")
        print(f"  -tau_primary_joint_delta={neg_joint_delta[idx]:+.6f} rad")
        print(f"  +tau_primary_vel_delta={pos_vel_delta[idx]:+.6f} rad/s")
        print(f"  -tau_primary_vel_delta={neg_vel_delta[idx]:+.6f} rad/s")
        print(f"  +tau_dominant_joint={JOINT_LABELS[dominant_idx]} delta={pos_joint_delta[dominant_idx]:+.6f} rad")
        print(f"  +tau_roll_delta={pos.roll_deg - zero.roll_deg:+.5f} deg")
        print(f"  +tau_pitch_delta={pos.pitch_deg - zero.pitch_deg:+.5f} deg")
        print(f"  +tau_fz_delta_l={pos.left_fz - zero.left_fz:+.4f} N")
        print(f"  +tau_fz_delta_r={pos.right_fz - zero.right_fz:+.4f} N")
        print(f"  +tau_joint_delta_vec={np.array2string(pos_joint_delta, precision=5, suppress_small=True)}")
        print(f"  -tau_joint_delta_vec={np.array2string(neg_joint_delta, precision=5, suppress_small=True)}")


def print_pair_cases() -> None:
    model, _ = reset_model()
    zero = rollout(np.zeros(model.nu))

    print("\nLeft/right pair response")
    for left_idx, right_idx in PAIR_INDICES:
        for pattern_name, left_sign, right_sign in [
            ("same_positive", 1.0, 1.0),
            ("opposite_left_positive", 1.0, -1.0),
        ]:
            ctrl = np.zeros(model.nu)
            ctrl[left_idx] = left_sign * TORQUE_NM
            ctrl[right_idx] = right_sign * TORQUE_NM
            result = rollout(ctrl)
            joint_delta = result.joint_pos - zero.joint_pos
            print(f"\npair={JOINT_LABELS[left_idx]}/{JOINT_LABELS[right_idx]} pattern={pattern_name}")
            print(f"  roll_delta={result.roll_deg - zero.roll_deg:+.5f} deg")
            print(f"  pitch_delta={result.pitch_deg - zero.pitch_deg:+.5f} deg")
            print(f"  left_joint_delta={joint_delta[left_idx]:+.6f} rad")
            print(f"  right_joint_delta={joint_delta[right_idx]:+.6f} rad")
            print(f"  left_fz_delta={result.left_fz - zero.left_fz:+.4f} N")
            print(f"  right_fz_delta={result.right_fz - zero.right_fz:+.4f} N")


def main() -> None:
    print("Motor/joint coupling diagnostic")
    print(f"model={MODEL_PATH}")
    print(f"rollout_steps={ROLLOUT_STEPS}")
    print_single_motor_cases()
    print_pair_cases()


if __name__ == "__main__":
    main()
