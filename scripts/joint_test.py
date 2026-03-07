"""
Joint Test Viewer — kiểm tra hướng và giới hạn từng khớp.

KHÔNG dùng vật lý — robot treo lơ lửng, slider điều khiển
góc khớp trực tiếp (rad), không phải torque.

Chạy:
    python scripts/joint_test.py

Panel 'Controls' bên phải: kéo slider = đặt góc khớp (rad)
Panel 'Joints'   bên trái: đọc giá trị góc hiện tại (rad)
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import mujoco
import mujoco.viewer
import numpy as np

# ──────────────────────────────────────────────────────────────
# Cấu hình khớp: tên actuator → (tên joint, giới hạn rad, mô tả)
# ──────────────────────────────────────────────────────────────
JOINT_CFG = [
    # act_name              jnt_name        lo     hi    mô tả
    ("l_hip_roll_motor", "l_hip_roll", -0.7, 0.7, "Hông T   + ra ngoài / - vào trong"),
    ("l_hip_yaw_motor", "l_hip_yaw", -0.4, 0.4, "Yaw T    + xoay phải / - xoay trái"),
    ("l_hip_pitch_motor", "l_hip_pitch", -0.5, 1.8, "Đùi  T   + trước ✅  / - sau"),
    ("l_knee_motor", "l_knee", -0.5, 2.7, "Gối  T   + trước ✅  / - sau"),
    ("l_wheel_motor", "l_wheel", -3.14, 3.14, "Bánh T   + tiến     / - lùi"),
    ("r_hip_roll_motor", "r_hip_roll", -0.7, 0.7, "Hông P   + ra ngoài / - vào trong"),
    ("r_hip_yaw_motor", "r_hip_yaw", -0.4, 0.4, "Yaw P    + xoay phải / - xoay trái"),
    ("r_hip_pitch_motor", "r_hip_pitch", -0.5, 1.8, "Đùi  P   + trước ✅  / - sau"),
    ("r_knee_motor", "r_knee", -0.5, 2.7, "Gối  P   + trước ✅  / - sau"),
    ("r_wheel_motor", "r_wheel", -3.14, 3.14, "Bánh P   + tiến     / - lùi"),
]


def main():
    xml_path = os.path.join(ROOT, "assets", "robot", "wheeled_biped_real.xml")
    if not os.path.exists(xml_path):
        print(f"[ERROR] Không tìm thấy model: {xml_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # ── Tắt trọng lực (robot lơ lửng) ──────────────────────────
    model.opt.gravity[:] = 0.0

    # ── Load keyframe đứng, nâng lên cao 1.5 m để nhìn rõ ──────
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 1.5  # Z = 1.5 m
    data.qpos[3:7] = [1, 0, 0, 0]  # quaternion thẳng đứng
    mujoco.mj_forward(model, data)

    # Lưu root pose cố định (sẽ reset mỗi bước)
    root_qpos = data.qpos[:7].copy()

    # ── Build map: act_id → qpos_addr, override ctrlrange ───────
    act_map = {}  # act_id → qposadr
    for act_name, jnt_name, lo, hi, _ in JOINT_CFG:
        act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
        if act_id < 0 or jnt_id < 0:
            continue
        model.actuator_ctrlrange[act_id] = [lo, hi]
        qadr = model.jnt_qposadr[jnt_id]
        act_map[act_id] = qadr
        # Slider khởi tạo bằng góc keyframe hiện tại
        data.ctrl[act_id] = float(data.qpos[qadr])

    # ── In thông tin ra console ──────────────────────────────────
    print("=" * 68)
    print("  JOINT KINEMATIC VIEWER  —  slider = góc khớp (rad), vật lý TẮT")
    print("=" * 68)
    print(f"  {'Slider (Actuator)':<22} {'Khớp':<14} {'Range (rad)':^16}  Ý nghĩa")
    print("  " + "-" * 66)
    for act_name, jnt_name, lo, hi, desc in JOINT_CFG:
        print(f"  {act_name:<22} {jnt_name:<14} [{lo:+.2f}, {hi:+.2f}]  {desc}")
    print("=" * 68)
    print("  Mở panel 'Controls' (bên phải viewer) → kéo slider")
    print("  Panel 'Joints'   (bên trái)  → đọc góc hiện tại (rad)")
    print("  Nhấn ESC hoặc đóng cửa sổ để thoát")
    print("=" * 68)

    # ── Viewer loop ──────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 140
        viewer.cam.lookat[:] = [0.0, 0.0, 1.5]

        while viewer.is_running():
            # Cố định root (không để rơi/xoay)
            data.qpos[:7] = root_qpos
            data.qvel[:] = 0.0

            # Slider → góc khớp trực tiếp
            for act_id, qadr in act_map.items():
                val = float(data.ctrl[act_id])
                lo = model.actuator_ctrlrange[act_id, 0]
                hi = model.actuator_ctrlrange[act_id, 1]
                data.qpos[qadr] = np.clip(val, lo, hi)

            # Forward kinematics only — không tính lực / động lực học
            mujoco.mj_kinematics(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
