"""Chẩn đoán hướng gập khớp chân trái vs chân phải.
Tìm tổ hợp góc đúng để cả 2 chân cùng gập về phía trước.
"""

import os

import mujoco

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "robot", "wheeled_biped_real.xml"
)
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# Indices
jnt_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]

print("Joints:", jnt_names)
print()


def get_body_world_pos(name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()


def get_joint_qpos_adr(name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return model.jnt_qposadr[jid]


def test_config(l_hp, l_kn, r_hp, r_kn, label=""):
    """Test a joint configuration and report world positions."""
    mujoco.mj_resetData(model, data)
    # Root at 0.71m, upright
    data.qpos[0:3] = [0, 0, 0.71]
    data.qpos[3:7] = [1, 0, 0, 0]

    # Set joints
    data.qpos[get_joint_qpos_adr("l_hip_roll")] = 0
    data.qpos[get_joint_qpos_adr("l_hip_yaw")] = 0
    data.qpos[get_joint_qpos_adr("l_hip_pitch")] = l_hp
    data.qpos[get_joint_qpos_adr("l_knee")] = l_kn
    data.qpos[get_joint_qpos_adr("l_wheel")] = 0
    data.qpos[get_joint_qpos_adr("r_hip_roll")] = 0
    data.qpos[get_joint_qpos_adr("r_hip_yaw")] = 0
    data.qpos[get_joint_qpos_adr("r_hip_pitch")] = r_hp
    data.qpos[get_joint_qpos_adr("r_knee")] = r_kn
    data.qpos[get_joint_qpos_adr("r_wheel")] = 0

    mujoco.mj_forward(model, data)

    torso = get_body_world_pos("torso")
    l_thigh = get_body_world_pos("l_thigh")
    l_knee = get_body_world_pos("l_knee_link")
    l_wheel = get_body_world_pos("l_wheel_link")
    r_thigh = get_body_world_pos("r_thigh")
    r_knee = get_body_world_pos("r_knee_link")
    r_wheel = get_body_world_pos("r_wheel_link")

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(
        f"  l_hip_pitch={l_hp:+.2f}  l_knee={l_kn:+.2f}"
        f"  |  r_hip_pitch={r_hp:+.2f}  r_knee={r_kn:+.2f}"
    )
    print(f"{'=' * 70}")
    print(f"  {'Body':20s}  {'X (left+)':>10s}  {'Y (back+)':>10s}  {'Z (up+)':>10s}")
    print(f"  {'─' * 55}")
    print(f"  {'torso':20s}  {torso[0]:+10.4f}  {torso[1]:+10.4f}  {torso[2]:+10.4f}")
    print(f"  {'l_thigh':20s}  {l_thigh[0]:+10.4f}  {l_thigh[1]:+10.4f}  {l_thigh[2]:+10.4f}")
    print(f"  {'l_knee_link':20s}  {l_knee[0]:+10.4f}  {l_knee[1]:+10.4f}  {l_knee[2]:+10.4f}")
    print(f"  {'l_wheel_link':20s}  {l_wheel[0]:+10.4f}  {l_wheel[1]:+10.4f}  {l_wheel[2]:+10.4f}")
    print(f"  {'r_thigh':20s}  {r_thigh[0]:+10.4f}  {r_thigh[1]:+10.4f}  {r_thigh[2]:+10.4f}")
    print(f"  {'r_knee_link':20s}  {r_knee[0]:+10.4f}  {r_knee[1]:+10.4f}  {r_knee[2]:+10.4f}")
    print(f"  {'r_wheel_link':20s}  {r_wheel[0]:+10.4f}  {r_wheel[1]:+10.4f}  {r_wheel[2]:+10.4f}")

    # Check: knee Y relative to hip Y (positive = forward)
    l_knee_fwd = l_knee[1] - l_thigh[1]
    r_knee_fwd = r_knee[1] - r_thigh[1]
    l_wheel_z = l_wheel[2]
    r_wheel_z = r_wheel[2]

    print("\n  Hướng gối (Y so với đùi):")
    l_dir = "(SAU)" if l_knee_fwd > 0.01 else "(TRƯỚC)" if l_knee_fwd < -0.01 else "(GIỮA)"
    r_dir = "(SAU)" if r_knee_fwd > 0.01 else "(TRƯỚC)" if r_knee_fwd < -0.01 else "(GIỮA)"
    print(f"    L knee offset Y = {l_knee_fwd:+.4f} {l_dir}")
    print(f"    R knee offset Y = {r_knee_fwd:+.4f} {r_dir}")
    print(f"  Bánh xe Z: L={l_wheel_z:.4f}  R={r_wheel_z:.4f}")

    return l_knee_fwd, r_knee_fwd, l_wheel_z, r_wheel_z


print("=" * 70)
print("  CHẨN ĐOÁN HƯỚNG GẬP KHỚP: TÌM CẤU HÌNH ĐÚNG")
print("=" * 70)

# Test 1: Current keyframe (l=+0.3,+0.5 r=+0.3,+0.5) - cả 2 chân cùng dấu dương
test_config(0.3, 0.5, 0.3, 0.5, "HIỆN TẠI: l_hp=+0.3, l_kn=+0.5 | r_hp=+0.3, r_kn=+0.5")

# Test 2: Same signs both legs
test_config(
    0.3,
    0.5,
    0.3,
    0.5,
    "THỬ 1: Cùng dấu dương: l_hp=+0.3, l_kn=+0.5 | r_hp=+0.3, r_kn=+0.5",
)

# Test 3: Inverted
test_config(
    -0.3,
    -0.5,
    -0.3,
    -0.5,
    "THỬ 2: Cùng dấu âm: l_hp=-0.3, l_kn=-0.5 | r_hp=-0.3, r_kn=-0.5",
)

# Test 4: Mixed
test_config(0.3, 0.5, -0.3, -0.5, "THỬ 3: Hoán đổi: l_hp=+0.3, l_kn=+0.5 | r_hp=-0.3, r_kn=-0.5")

# Test with different magnitudes
test_config(0.3, 0.8, 0.3, 0.8, "THỬ 4: Gập nhiều: l_hp=+0.3, l_kn=+0.8 | r_hp=+0.3, r_kn=+0.8")
test_config(
    -0.3,
    -0.8,
    0.3,
    0.8,
    "THỬ 5: Hiện tại gập nhiều: l_hp=-0.3, l_kn=-0.8 | r_hp=+0.3, r_kn=+0.8",
)

# Test 6: all zeros
test_config(0, 0, 0, 0, "THẲNG: Tất cả = 0")

# Additional test with small angle to detect direction
print("\n\n" + "=" * 70)
print("  PHÂN TÍCH HƯỚNG TRỤC: Góc nhỏ +0.1 cho từng khớp riêng lẻ")
print("=" * 70)

for jname in ["l_hip_pitch", "r_hip_pitch", "l_knee", "r_knee"]:
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.71]
    data.qpos[3:7] = [1, 0, 0, 0]

    adr = get_joint_qpos_adr(jname)
    # Test +0.1
    data.qpos[adr] = 0.1
    mujoco.mj_forward(model, data)

    side = "l" if jname.startswith("l") else "r"
    knee_pos = get_body_world_pos(f"{side}_knee_link")
    wheel_pos = get_body_world_pos(f"{side}_wheel_link")

    # Reset and test -0.1
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0, 0, 0.71]
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[adr] = -0.1
    mujoco.mj_forward(model, data)

    knee_neg = get_body_world_pos(f"{side}_knee_link")
    wheel_neg = get_body_world_pos(f"{side}_wheel_link")

    dy_knee = knee_pos[1] - knee_neg[1]
    dz_wheel = wheel_pos[2] - wheel_neg[2]

    print(f"\n  {jname:18s}: +0.1 vs -0.1")
    print(f"    Gối ΔY = {dy_knee:+.4f} (dương → +0.1 đẩy gối ra trước)")
    print(f"    Xe  ΔZ = {dz_wheel:+.4f} (âm → +0.1 hạ bánh xe)")
