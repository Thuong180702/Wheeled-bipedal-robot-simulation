"""Kiểm tra toàn diện MJCF model so với URDF gốc.
Bao gồm: masses, inertia, joint limits, forces, collision, material properties.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "assets", "robot", "wheeled_biped_real.xml"
)

# ====== DỮ LIỆU URDF GỐC ======
URDF_MASSES = {
    "torso": 2.5,
    "l_hip_roll_link": 0.5,
    "r_hip_roll_link": 0.5,
    "l_hip_yaw_link": 0.5,
    "r_hip_yaw_link": 0.5,
    "l_thigh": 0.8,
    "r_thigh": 0.8,
    "l_knee_link": 0.6,
    "r_knee_link": 0.6,
    "l_wheel_link": 0.1,
    "r_wheel_link": 0.1,
}

URDF_JOINT_LIMITS = {
    "l_hip_roll": (-0.7, 0.7),
    "r_hip_roll": (-0.7, 0.7),
    "l_hip_yaw": (-0.4, 0.4),
    "r_hip_yaw": (-0.4, 0.4),
    "l_hip_pitch": (-0.5, 1.8),
    "r_hip_pitch": (-0.5, 1.8),
    "l_knee": (-0.5, 2.7),
    "r_knee": (-0.5, 2.7),
    "l_wheel": None,
    "r_wheel": None,  # continuous
}

URDF_EFFORTS = {
    "l_hip_roll": 22,
    "r_hip_roll": 22,
    "l_hip_yaw": 22,
    "r_hip_yaw": 22,
    "l_hip_pitch": 22,
    "r_hip_pitch": 22,
    "l_knee": 44,
    "r_knee": 44,
    "l_wheel": 22,
    "r_wheel": 22,
}

URDF_VELOCITIES = {
    "l_hip_roll": 11.52,
    "r_hip_roll": 11.52,
    "l_hip_yaw": 11.52,
    "r_hip_yaw": 11.52,
    "l_hip_pitch": 11.52,
    "r_hip_pitch": 11.52,
    "l_knee": 5.75,
    "r_knee": 5.75,
    "l_wheel": 11.52,
    "r_wheel": 11.52,
}

URDF_INERTIA_DIAG = {
    "torso": (0.005598, 0.015241, 0.017751),
    "l_hip_roll_link": (0.00036543, 0.00031624, 0.00032404),
    "r_hip_roll_link": (0.00036543, 0.00031624, 0.00032404),
    "l_hip_yaw_link": (0.00035308, 0.000209, 0.00022857),
    "r_hip_yaw_link": (0.00022857, 0.00035308, 0.000209),
    "l_thigh": (0.0041434, 0.0044915, 0.00041762),
    "r_thigh": (0.0041434, 0.0044915, 0.00041762),
    "l_knee_link": (0.0018923, 0.0020924, 0.00021843),
    "r_knee_link": (0.0018923, 0.0020924, 0.00021843),
    "l_wheel_link": (0.00018832, 0.00012247, 0.00012247),
    "r_wheel_link": (0.00018832, 0.00012247, 0.00012247),
}


def check_separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_basic_info(model):
    check_separator("THÔNG TIN CƠ BẢN")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} nbody={model.nbody}")
    print(f"  njnt={model.njnt} ngeom={model.ngeom} nsite={model.nsite}")
    print(f"  timestep={model.opt.timestep}")
    print(f"  gravity={model.opt.gravity}")


def check_masses(model):
    check_separator("KIỂM TRA KHỐI LƯỢNG (URDF vs MJCF)")
    total_mass = 0
    all_ok = True
    for body_name, urdf_mass in URDF_MASSES.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            print(f"  ✗ Body '{body_name}' KHÔNG TÌM THẤY!")
            all_ok = False
            continue
        mjcf_mass = model.body_mass[bid]
        total_mass += mjcf_mass
        diff = abs(mjcf_mass - urdf_mass)
        status = "✓" if diff < 0.01 else "✗"
        if diff >= 0.01:
            all_ok = False
        print(
            f"  {status} {body_name:20s}: URDF={urdf_mass:.3f} kg  MJCF={mjcf_mass:.3f} kg  Δ={diff:.4f}"
        )
    print(f"\n  Tổng khối lượng MJCF: {total_mass:.3f} kg")
    urdf_total = sum(URDF_MASSES.values())
    print(f"  Tổng khối lượng URDF: {urdf_total:.3f} kg")
    return all_ok


def check_inertia(model):
    check_separator("KIỂM TRA QUÁN TÍNH (URDF vs MJCF)")
    all_ok = True
    for body_name, urdf_diag in URDF_INERTIA_DIAG.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        mjcf_inertia = model.body_inertia[bid]  # Principal moments (sorted desc)
        urdf_sorted = sorted(urdf_diag, reverse=True)
        mjcf_sorted = sorted(mjcf_inertia, reverse=True)

        max_diff = max(abs(a - b) for a, b in zip(urdf_sorted, mjcf_sorted))
        rel_diff = max_diff / max(urdf_sorted) * 100 if max(urdf_sorted) > 0 else 0
        status = "✓" if rel_diff < 5 else "~" if rel_diff < 15 else "✗"
        if rel_diff >= 15:
            all_ok = False
        print(
            f"  {status} {body_name:20s}: URDF={[f'{v:.6f}' for v in urdf_sorted]}  MJCF={[f'{v:.6f}' for v in mjcf_sorted]}  Δ={rel_diff:.1f}%"
        )
    return all_ok


def check_joint_limits(model):
    check_separator("KIỂM TRA GIỚI HẠN KHỚP (URDF vs MJCF)")
    all_ok = True
    for jname, urdf_lim in URDF_JOINT_LIMITS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            print(f"  ✗ Joint '{jname}' KHÔNG TÌM THẤY!")
            all_ok = False
            continue

        is_limited = model.jnt_limited[jid]
        if urdf_lim is None:
            status = "✓" if not is_limited else "✗"
            if is_limited:
                all_ok = False
            print(
                f"  {status} {jname:18s}: Continuous — MJCF limited={bool(is_limited)}"
            )
        else:
            if not is_limited:
                print(f"  ✗ {jname:18s}: URDF có limit nhưng MJCF không limited!")
                all_ok = False
                continue
            mjcf_range = model.jnt_range[jid]
            low_ok = abs(mjcf_range[0] - urdf_lim[0]) < 0.01
            high_ok = abs(mjcf_range[1] - urdf_lim[1]) < 0.01
            status = "✓" if (low_ok and high_ok) else "✗"
            if not (low_ok and high_ok):
                all_ok = False
            print(
                f"  {status} {jname:18s}: URDF=[{urdf_lim[0]:.2f}, {urdf_lim[1]:.2f}]  "
                f"MJCF=[{mjcf_range[0]:.2f}, {mjcf_range[1]:.2f}]"
            )
    return all_ok


def check_actuator_forces(model):
    check_separator("KIỂM TRA LỰC ACTUATOR (URDF effort vs MJCF forcerange)")
    all_ok = True
    for jname, urdf_effort in URDF_EFFORTS.items():
        motor_name = jname + "_motor"
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, motor_name)
        if aid < 0:
            print(f"  ✗ Actuator '{motor_name}' KHÔNG TÌM THẤY!")
            all_ok = False
            continue

        force_limited = model.actuator_forcelimited[aid]
        force_range = model.actuator_forcerange[aid]
        ctrl_range = model.actuator_ctrlrange[aid]

        force_ok = abs(force_range[1] - urdf_effort) < 0.1
        status = "✓" if (force_limited and force_ok) else "✗"
        if not (force_limited and force_ok):
            all_ok = False
        print(
            f"  {status} {motor_name:24s}: URDF effort={urdf_effort:5.1f} Nm  "
            f"MJCF force=[{force_range[0]:.0f},{force_range[1]:.0f}]  "
            f"ctrl=[{ctrl_range[0]:.0f},{ctrl_range[1]:.0f}]"
        )
    return all_ok


def check_collision_geoms(model):
    check_separator("KIỂM TRA COLLISION GEOMETRY")
    all_ok = True
    bodies_need_collision = [
        "torso",
        "l_hip_roll_link",
        "r_hip_roll_link",
        "l_hip_yaw_link",
        "r_hip_yaw_link",
        "l_thigh",
        "r_thigh",
        "l_knee_link",
        "r_knee_link",
        "l_wheel_link",
        "r_wheel_link",
    ]

    for body_name in bodies_need_collision:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            print(f"  ✗ Body '{body_name}' không tìm thấy")
            all_ok = False
            continue

        has_collision = False
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == bid:
                contype = model.geom_contype[gid]
                conaffinity = model.geom_conaffinity[gid]
                if contype > 0 and conaffinity > 0:
                    has_collision = True
                    gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
                    gtype_names = [
                        "plane",
                        "hfield",
                        "sphere",
                        "capsule",
                        "ellipsoid",
                        "cylinder",
                        "box",
                        "mesh",
                    ]
                    gtype = (
                        gtype_names[model.geom_type[gid]]
                        if model.geom_type[gid] < len(gtype_names)
                        else "?"
                    )
                    friction = model.geom_friction[gid]
                    solref = model.geom_solref[gid]
                    condim = model.geom_condim[gid]
                    print(
                        f"  ✓ {body_name:20s} → {gname:28s} type={gtype:8s} condim={condim} "
                        f"friction=[{friction[0]:.2f},{friction[1]:.3f},{friction[2]:.4f}] "
                        f"solref=[{solref[0]:.4f},{solref[1]:.2f}]"
                    )
                    break

        if not has_collision:
            print(f"  ✗ {body_name:20s} → KHÔNG CÓ COLLISION GEOM!")
            all_ok = False

    return all_ok


def check_material_properties(model):
    check_separator("KIỂM TRA VẬT LIỆU (Nhôm vs Cao su)")

    print("\n  --- Nhôm (thân, chân): ---")
    print("  Thuộc tính đúng: friction~0.4-0.6, solref stiff (timeconst≤0.005)")

    aluminum_geoms = [
        "torso_collision",
        "l_hip_roll_collision",
        "r_hip_roll_collision",
        "l_thigh_collision",
        "r_thigh_collision",
        "l_shin_collision",
        "r_shin_collision",
    ]
    for gname in aluminum_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            print(f"  ✗ Geom '{gname}' không tìm thấy")
            continue
        f = model.geom_friction[gid]
        sr = model.geom_solref[gid]
        si = model.geom_solimp[gid]
        ok = 0.3 <= f[0] <= 0.7 and sr[0] <= 0.005
        status = "✓" if ok else "~"
        print(
            f"  {status} {gname:28s}: μ={f[0]:.2f}  solref=[{sr[0]:.4f},{sr[1]:.1f}]  "
            f"solimp=[{si[0]:.2f},{si[1]:.2f},{si[2]:.3f}]"
        )

    print("\n  --- Cao su (lốp bánh xe): ---")
    print(
        "  Thuộc tính đúng: friction~1.0-1.5, solref mềm hơn (timeconst≥0.003), condim=6"
    )

    rubber_geoms = ["l_wheel_collision", "r_wheel_collision"]
    for gname in rubber_geoms:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            print(f"  ✗ Geom '{gname}' không tìm thấy")
            continue
        f = model.geom_friction[gid]
        sr = model.geom_solref[gid]
        si = model.geom_solimp[gid]
        cd = model.geom_condim[gid]
        ok = 0.8 <= f[0] <= 1.5 and sr[0] >= 0.003 and cd == 6
        status = "✓" if ok else "✗"
        print(
            f"  {status} {gname:28s}: μ={f[0]:.2f}  solref=[{sr[0]:.4f},{sr[1]:.1f}]  "
            f"solimp=[{si[0]:.2f},{si[1]:.2f},{si[2]:.3f}]  condim={cd}"
        )


def check_standing_stability(model):
    check_separator("KIỂM TRA ỔN ĐỊNH ĐỨNG (1000 bước = 2 giây)")
    data = mujoco.MjData(model)

    # Load keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "standing")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)

    mujoco.mj_forward(model, data)
    torso_z_start = data.xpos[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    ][2]

    # Simulate 1000 steps
    for _ in range(1000):
        mujoco.mj_step(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_z_end = data.xpos[torso_id][2]
    torso_quat = data.xquat[torso_id]

    print(
        f"  Torso Z: {torso_z_start:.4f}m → {torso_z_end:.4f}m (Δ={torso_z_end-torso_z_start:.4f}m)"
    )
    print(
        f"  Torso quat: [{torso_quat[0]:.4f}, {torso_quat[1]:.4f}, {torso_quat[2]:.4f}, {torso_quat[3]:.4f}]"
    )

    # Check all body positions above ground
    print("\n  Vị trí Z mỗi body sau 2 giây:")
    all_above = True
    for i in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        bz = data.xpos[i][2]
        status = "✓" if bz > -0.01 else "✗"
        if bz <= -0.01:
            all_above = False
        print(f"    {status} {bname:20s}: z = {bz:.4f}m")

    stable = abs(torso_z_end - torso_z_start) < 0.05 and all_above
    print(f"\n  {'✓' if stable else '✗'} Ổn định: {'CÓ' if stable else 'KHÔNG'}")
    return stable


def check_fallen_collision(model):
    check_separator("KIỂM TRA COLLISION KHI NGÃ (thả rơi từ 0.5m)")
    data = mujoco.MjData(model)

    # Đặt robot nằm nghiêng ở z=0.5m
    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0  # x
    data.qpos[1] = 0  # y
    data.qpos[2] = 0.3  # z (thấp)
    data.qpos[3] = 0.707  # quat w
    data.qpos[4] = 0.707  # quat x (nghiêng 90° quanh X = nằm nghiêng)
    data.qpos[5] = 0  # quat y
    data.qpos[6] = 0  # quat z

    mujoco.mj_forward(model, data)

    # Simulate 2000 steps (4 giây)
    for _ in range(2000):
        mujoco.mj_step(model, data)

    print("  Vị trí Z sau 4 giây (robot nằm nghiêng):")
    any_below = False
    for i in range(1, model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        bz = data.xpos[i][2]
        status = "✓" if bz > -0.02 else "✗ CHÌM!"
        if bz <= -0.02:
            any_below = True
        print(f"    {status} {bname:20s}: z = {bz:.4f}m")

    if not any_below:
        print("\n  ✓ KHÔNG có body nào chìm dưới mặt đất!")
    else:
        print("\n  ✗ CÓ body chìm dưới mặt đất!")
    return not any_below


def main():
    print("=" * 60)
    print("  KIỂM TRA TOÀN DIỆN MÔ HÌNH WHEELED BIPED REAL")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)

    results = {}
    results["basic"] = True
    check_basic_info(model)
    results["mass"] = check_masses(model)
    results["inertia"] = check_inertia(model)
    results["joints"] = check_joint_limits(model)
    results["forces"] = check_actuator_forces(model)
    results["collision"] = check_collision_geoms(model)
    check_material_properties(model)
    results["standing"] = check_standing_stability(model)
    results["fallen"] = check_fallen_collision(model)

    check_separator("KẾT QUẢ TỔNG HỢP")
    labels = {
        "mass": "Khối lượng",
        "inertia": "Quán tính",
        "joints": "Giới hạn khớp",
        "forces": "Lực actuator",
        "collision": "Collision geom",
        "standing": "Ổn định đứng",
        "fallen": "Collision khi ngã",
    }
    all_pass = True
    for key, label in labels.items():
        ok = results.get(key, False)
        if not ok:
            all_pass = False
        print(f"  {'✓' if ok else '✗'} {label}")

    print(f"\n  {'✓ TẤT CẢ PASS!' if all_pass else '✗ CÓ LỖI CẦN SỬA!'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
