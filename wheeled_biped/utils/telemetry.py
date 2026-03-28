"""
Telemetry: Thu thập và hiển thị dữ liệu robot (vị trí khớp, lực, vận tốc, thân).

Dùng cho visualize.py — ghi CSV log + vẽ biểu đồ matplotlib.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Tên 10 khớp theo thứ tự actuator
JOINT_NAMES = [
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

# Header đầy đủ cho CSV
CSV_HEADER = (
    ["step", "time_s"]
    # Thân: vị trí XYZ, quaternion, euler RPY, vận tốc tuyến tính, vận tốc góc
    + ["torso_x", "torso_y", "torso_z"]
    + ["quat_w", "quat_x", "quat_y", "quat_z"]
    + ["roll_rad", "pitch_rad", "yaw_rad"]
    + ["body_vx", "body_vy", "body_vz"]
    + ["body_wx", "body_wy", "body_wz"]
    # Vị trí khớp (10)
    + [f"{j}_pos" for j in JOINT_NAMES]
    # Vận tốc khớp (10)
    + [f"{j}_vel" for j in JOINT_NAMES]
    # Lực/torque motor (10)
    + [f"{j}_torque" for j in JOINT_NAMES]
    # Ctrl (10)
    + [f"{j}_ctrl" for j in JOINT_NAMES]
)


def quat_to_euler_np(quat: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] → euler [roll, pitch, yaw] (rad)."""
    w, x, y, z = quat
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return np.array([roll, pitch, yaw])


class TelemetryRecorder:
    """Thu thập telemetry từ mj_data mỗi step."""

    def __init__(self, control_dt: float = 0.02):
        self.control_dt = control_dt
        self.data: list[dict] = []
        self._step = 0

    def record(self, mj_data) -> dict:
        """Ghi 1 snapshot từ mj_data. Trả về dict row."""
        qpos = np.array(mj_data.qpos, copy=True)
        qvel = np.array(mj_data.qvel, copy=True)
        ctrl = np.array(mj_data.ctrl, copy=True)
        # actuator_force = actual force applied
        act_force = np.array(mj_data.actuator_force, copy=True)

        torso_pos = qpos[:3]
        torso_quat = qpos[3:7]
        euler = quat_to_euler_np(torso_quat)

        # Body-frame velocities (world frame for simplicity in log)
        lin_vel = qvel[:3]
        ang_vel = qvel[3:6]

        joint_pos = qpos[7:17]
        joint_vel = qvel[6:16]

        row = {
            "step": self._step,
            "time_s": round(self._step * self.control_dt, 4),
            # Torso
            "torso_x": torso_pos[0],
            "torso_y": torso_pos[1],
            "torso_z": torso_pos[2],
            "quat_w": torso_quat[0],
            "quat_x": torso_quat[1],
            "quat_y": torso_quat[2],
            "quat_z": torso_quat[3],
            "roll_rad": euler[0],
            "pitch_rad": euler[1],
            "yaw_rad": euler[2],
            "body_vx": lin_vel[0],
            "body_vy": lin_vel[1],
            "body_vz": lin_vel[2],
            "body_wx": ang_vel[0],
            "body_wy": ang_vel[1],
            "body_wz": ang_vel[2],
        }
        # Joints
        for i, name in enumerate(JOINT_NAMES):
            row[f"{name}_pos"] = joint_pos[i]
            row[f"{name}_vel"] = joint_vel[i]
            row[f"{name}_torque"] = act_force[i] if i < len(act_force) else 0.0
            row[f"{name}_ctrl"] = ctrl[i] if i < len(ctrl) else 0.0

        self.data.append(row)
        self._step += 1
        return row

    def save_csv(self, path: str | Path) -> Path:
        """Lưu toàn bộ data ra CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        return path

    def to_numpy(self) -> dict[str, np.ndarray]:
        """Chuyển data thành dict of numpy arrays (tiện cho plot)."""
        if not self.data:
            return {}
        result = {}
        for key in self.data[0]:
            result[key] = np.array([row[key] for row in self.data])
        return result


def plot_telemetry(
    data: dict[str, np.ndarray],
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Vẽ biểu đồ telemetry đầy đủ.

    6 subplots:
      1. Torso position (X, Y, Z)
      2. Torso orientation (Roll, Pitch, Yaw)
      3. Torso velocity (linear + angular)
      4. Joint positions (hip_pitch, knee — cả 2 bên)
      5. Joint velocities (tất cả 10)
      6. Motor torques (tất cả 10)
    """
    import matplotlib.pyplot as plt

    t = data["time_s"]
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    fig.suptitle("Robot Telemetry", fontsize=14, fontweight="bold")

    # --- 1. Torso Position ---
    ax = axes[0, 0]
    ax.plot(t, data["torso_x"], label="X", alpha=0.8)
    ax.plot(t, data["torso_y"], label="Y", alpha=0.8)
    ax.plot(t, data["torso_z"], label="Z (height)", linewidth=2)
    ax.set_ylabel("Position (m)")
    ax.set_title("Torso Position")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 2. Torso Orientation ---
    ax = axes[0, 1]
    ax.plot(t, np.degrees(data["roll_rad"]), label="Roll", alpha=0.8)
    ax.plot(t, np.degrees(data["pitch_rad"]), label="Pitch", alpha=0.8)
    ax.plot(t, np.degrees(data["yaw_rad"]), label="Yaw", alpha=0.8)
    ax.set_ylabel("Angle (deg)")
    ax.set_title("Torso Orientation (RPY)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- 3. Torso Velocity ---
    ax = axes[1, 0]
    ax.plot(t, data["body_vx"], label="vx", alpha=0.7)
    ax.plot(t, data["body_vy"], label="vy", alpha=0.7)
    ax.plot(t, data["body_vz"], label="vz", alpha=0.7)
    ax.set_ylabel("Lin Vel (m/s)")
    ax.set_title("Torso Linear Velocity")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes[1, 1]
    ax2.plot(t, data["body_wx"], label="wx", alpha=0.7)
    ax2.plot(t, data["body_wy"], label="wy", alpha=0.7)
    ax2.plot(t, data["body_wz"], label="wz", alpha=0.7)
    ax2.set_ylabel("Ang Vel (rad/s)")
    ax2.set_title("Torso Angular Velocity")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- 4. Joint Positions (focus: hip_pitch + knee) ---
    ax = axes[2, 0]
    focus_joints = [
        ("l_hip_pitch_pos", "L hip_pitch", "C0"),
        ("r_hip_pitch_pos", "R hip_pitch", "C1"),
        ("l_knee_pos", "L knee", "C2"),
        ("r_knee_pos", "R knee", "C3"),
        ("l_hip_roll_pos", "L hip_roll", "C4"),
        ("r_hip_roll_pos", "R hip_roll", "C5"),
    ]
    for key, label, color in focus_joints:
        if key in data:
            ax.plot(t, data[key], label=label, alpha=0.8, color=color)
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Positions")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- 5. Motor Torques ---
    ax = axes[2, 1]
    torque_joints = [
        ("l_hip_pitch_torque", "L hip_pitch"),
        ("r_hip_pitch_torque", "R hip_pitch"),
        ("l_knee_torque", "L knee"),
        ("r_knee_torque", "R knee"),
        ("l_wheel_torque", "L wheel"),
        ("r_wheel_torque", "R wheel"),
    ]
    for key, label in torque_joints:
        if key in data:
            ax.plot(t, data[key], label=label, alpha=0.7)
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Motor Torques")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"  Biểu đồ đã lưu: {path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
