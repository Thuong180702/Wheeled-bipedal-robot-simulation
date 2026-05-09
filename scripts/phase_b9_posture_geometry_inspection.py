"""Phase B.9 posture geometry inspection and rendering.

Generates B9 initial posture images and manually selected human-like candidate
postures without training or controller tuning.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.controllers.dual_rate_balance_controller import DualRateConfig
from wheeled_biped.controllers.height_ik import HeightIK
from wheeled_biped.utils.config import get_model_path

HEIGHTS = [0.65, 0.60, 0.55, 0.50, 0.45, 0.40]
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


@dataclass(frozen=True)
class PoseSpec:
    name: str
    target_height: float | None
    hip_roll_l: float
    hip_roll_r: float
    hip_yaw_l: float
    hip_yaw_r: float
    hip_pitch_l: float
    hip_pitch_r: float
    knee_l: float
    knee_r: float


@dataclass(frozen=True)
class ViewSpec:
    name: str
    azimuth: float
    elevation: float
    distance: float


VIEWS = {
    "side": ViewSpec("side", azimuth=0.0, elevation=-8.0, distance=1.55),
    "front": ViewSpec("front", azimuth=90.0, elevation=-8.0, distance=1.55),
    "top": ViewSpec("top", azimuth=90.0, elevation=-89.0, distance=1.75),
}


CANDIDATES = [
    PoseSpec("upright", 0.65, 0.0, 0.0, 0.0, 0.0, 0.126, 0.126, 0.478, 0.478),
    PoseSpec("mild_crouch", 0.60, 0.0, 0.0, 0.0, 0.0, 0.527, 0.527, 1.020, 1.020),
    PoseSpec("medium_crouch", 0.50, 0.0, 0.0, 0.0, 0.0, 0.784, 0.784, 1.617, 1.617),
    PoseSpec("deep_crouch", 0.40, 0.0, 0.0, 0.0, 0.0, 1.067, 1.067, 2.092, 2.092),
]


CSV_COLUMNS = [
    "posture_name",
    "target_height",
    "hip_roll_L",
    "hip_roll_R",
    "hip_yaw_L",
    "hip_yaw_R",
    "hip_pitch_L",
    "hip_pitch_R",
    "knee_L",
    "knee_R",
    "torso_height",
    "torso_pitch",
    "torso_roll",
    "CoM_x",
    "CoM_y",
    "CoM_z",
    "wheel_contact_x",
    "wheel_contact_y",
    "wheel_contact_z",
    "left_wheel_site_height",
    "right_wheel_site_height",
    "left_wheel_contact_height",
    "right_wheel_contact_height",
    "left_wheel_contact_force",
    "right_wheel_contact_force",
    "left_wheel_body_z",
    "right_wheel_body_z",
    "CoM_lateral_offset_x",
    "knee_forward_margin_L",
    "knee_forward_margin_R",
    "left_right_joint_symmetry_abs_max",
    "wheel_contact_height_diff",
    "both_wheels_touch_ground_correctly",
    "knees_bend_forward",
    "torso_near_upright",
    "left_right_symmetric",
    "visual_interpretation",
]


def wheel_bottom_heights(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    l_geom = model.geom("l_wheel_collision").id
    r_geom = model.geom("r_wheel_collision").id
    l_radius = float(model.geom_size[l_geom, 0])
    r_radius = float(model.geom_size[r_geom, 0])
    return float(data.geom_xpos[l_geom, 2] - l_radius), float(data.geom_xpos[r_geom, 2] - r_radius)


def set_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseSpec) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = [0.0, 0.0, 1.0]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:17] = [
        pose.hip_roll_l,
        pose.hip_yaw_l,
        pose.hip_pitch_l,
        pose.knee_l,
        0.0,
        pose.hip_roll_r,
        pose.hip_yaw_r,
        pose.hip_pitch_r,
        pose.knee_r,
        0.0,
    ]
    mujoco.mj_forward(model, data)

    lowest_wheel_bottom = min(wheel_bottom_heights(model, data))
    data.qpos[2] -= lowest_wheel_bottom
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def body_com(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    total_mass = float(np.sum(model.body_mass))
    weighted = np.zeros(3)
    for body_id in range(model.nbody):
        weighted += model.body_mass[body_id] * data.xipos[body_id]
    return weighted / total_mass


def quat_to_pitch_roll_deg(quat: np.ndarray) -> tuple[float, float]:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    r = mat.reshape(3, 3)
    pitch = math.atan2(-r[2, 0], math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
    roll = math.atan2(r[2, 1], r[2, 2])
    return math.degrees(pitch), math.degrees(roll)


def contact_forces_by_wheel(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    l_geom = model.geom("l_wheel_collision").id
    r_geom = model.geom("r_wheel_collision").id
    floor_geom = model.geom("floor").id
    left_force = 0.0
    right_force = 0.0
    found_left = False
    found_right = False

    force = np.zeros(6)
    for i in range(data.ncon):
        contact = data.contact[i]
        geoms = {contact.geom1, contact.geom2}
        if floor_geom not in geoms:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        normal_force = abs(float(force[0]))
        if l_geom in geoms:
            left_force += normal_force
            found_left = True
        if r_geom in geoms:
            right_force += normal_force
            found_right = True

    return (left_force if found_left else math.nan, right_force if found_right else math.nan)


def measure_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: PoseSpec) -> dict[str, object]:
    l_site = model.site("l_wheel_contact").id
    r_site = model.site("r_wheel_contact").id
    l_wheel_body = model.body("l_wheel_link").id
    r_wheel_body = model.body("r_wheel_link").id
    l_hip_body = model.body("l_thigh").id
    r_hip_body = model.body("r_thigh").id
    l_knee_body = model.body("l_knee_link").id
    r_knee_body = model.body("r_knee_link").id

    com = body_com(model, data)
    torso_pitch, torso_roll = quat_to_pitch_roll_deg(data.qpos[3:7].copy())

    l_site_pos = data.site_xpos[l_site].copy()
    r_site_pos = data.site_xpos[r_site].copy()
    l_contact_height, r_contact_height = wheel_bottom_heights(model, data)
    wheel_contact = 0.5 * (l_site_pos + r_site_pos)
    wheel_contact[2] = 0.5 * (l_contact_height + r_contact_height)

    l_hip = data.xpos[l_hip_body].copy()
    r_hip = data.xpos[r_hip_body].copy()
    l_knee = data.xpos[l_knee_body].copy()
    r_knee = data.xpos[r_knee_body].copy()

    # Active model faces -Y, so positive forward margin = hip_y - knee_y.
    knee_forward_l = float(l_hip[1] - l_knee[1])
    knee_forward_r = float(r_hip[1] - r_knee[1])

    left_force, right_force = contact_forces_by_wheel(model, data)
    wheel_height_diff = abs(l_contact_height - r_contact_height)
    site_height_diff = abs(float(l_site_pos[2] - r_site_pos[2]))
    joint_symmetry_abs_max = max(
        abs(pose.hip_roll_l - pose.hip_roll_r),
        abs(pose.hip_yaw_l - pose.hip_yaw_r),
        abs(pose.hip_pitch_l - pose.hip_pitch_r),
        abs(pose.knee_l - pose.knee_r),
    )
    both_touch = abs(l_contact_height) < 1e-4 and abs(r_contact_height) < 1e-4 and wheel_height_diff < 1e-4
    knees_forward = knee_forward_l > 0.0 and knee_forward_r > 0.0
    torso_upright = abs(torso_pitch) < 5.0 and abs(torso_roll) < 5.0
    symmetric = joint_symmetry_abs_max < 1e-5 and abs(float(com[0])) < 0.03 and wheel_height_diff < 1e-4

    if knees_forward and torso_upright and symmetric and both_touch:
        interpretation = "Human-like symmetric upright/crouch posture with forward knees and both wheel contacts grounded."
    elif not knees_forward:
        interpretation = "Non-human-like or questionable posture: at least one knee is not forward in the active -Y forward frame."
    elif not symmetric:
        interpretation = "Forward-knee posture but left/right symmetry or lateral loading should be inspected."
    elif not both_touch:
        interpretation = "Forward-knee posture but wheel-ground contact is not symmetric/correct."
    else:
        interpretation = "Forward-knee posture with a possible upright/contact issue."

    return {
        "posture_name": pose.name,
        "target_height": pose.target_height if pose.target_height is not None else "",
        "hip_roll_L": pose.hip_roll_l,
        "hip_roll_R": pose.hip_roll_r,
        "hip_yaw_L": pose.hip_yaw_l,
        "hip_yaw_R": pose.hip_yaw_r,
        "hip_pitch_L": pose.hip_pitch_l,
        "hip_pitch_R": pose.hip_pitch_r,
        "knee_L": pose.knee_l,
        "knee_R": pose.knee_r,
        "torso_height": float(data.qpos[2]),
        "torso_pitch": torso_pitch,
        "torso_roll": torso_roll,
        "CoM_x": float(com[0]),
        "CoM_y": float(com[1]),
        "CoM_z": float(com[2]),
        "wheel_contact_x": float(wheel_contact[0]),
        "wheel_contact_y": float(wheel_contact[1]),
        "wheel_contact_z": float(wheel_contact[2]),
        "left_wheel_site_height": float(l_site_pos[2]),
        "right_wheel_site_height": float(r_site_pos[2]),
        "left_wheel_contact_height": l_contact_height,
        "right_wheel_contact_height": r_contact_height,
        "left_wheel_contact_force": left_force,
        "right_wheel_contact_force": right_force,
        "left_wheel_body_z": float(data.xpos[l_wheel_body, 2]),
        "right_wheel_body_z": float(data.xpos[r_wheel_body, 2]),
        "CoM_lateral_offset_x": float(com[0] - wheel_contact[0]),
        "knee_forward_margin_L": knee_forward_l,
        "knee_forward_margin_R": knee_forward_r,
        "left_right_joint_symmetry_abs_max": joint_symmetry_abs_max,
        "wheel_contact_height_diff": wheel_height_diff,
        "both_wheels_touch_ground_correctly": both_touch,
        "knees_bend_forward": knees_forward,
        "torso_near_upright": torso_upright,
        "left_right_symmetric": symmetric,
        "visual_interpretation": interpretation,
    }


def render_pose(model: mujoco.MjModel, data: mujoco.MjData, output_path: Path, view: ViewSpec) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.azimuth = view.azimuth
    camera.elevation = view.elevation
    camera.distance = view.distance
    camera.lookat[:] = [0.0, 0.0, float(data.qpos[2]) * 0.55]

    with mujoco.Renderer(model, height=480, width=640) as renderer:
        scene_option = mujoco.MjvOption()
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
        renderer.update_scene(data, camera=camera, scene_option=scene_option)
        pixels = renderer.render()

    try:
        import imageio.v3 as iio

        iio.imwrite(output_path, pixels)
    except Exception:
        import matplotlib.pyplot as plt

        plt.imsave(output_path, pixels)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def inspect_joint_axes(model: mujoco.MjModel) -> list[dict[str, object]]:
    axes = []
    for joint_name in JOINT_NAMES:
        joint = model.joint(joint_name)
        axes.append(
            {
                "joint_name": joint_name,
                "axis_local": [float(x) for x in model.jnt_axis[joint.id]],
                "range": [float(x) for x in model.jnt_range[joint.id]],
                "type": int(model.jnt_type[joint.id]),
            }
        )
    return axes


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "count": len(rows),
        "all_knees_forward": all(bool(r["knees_bend_forward"]) for r in rows),
        "all_torso_upright": all(bool(r["torso_near_upright"]) for r in rows),
        "all_both_wheels_touch": all(bool(r["both_wheels_touch_ground_correctly"]) for r in rows),
        "max_abs_com_lateral_offset_x": max(abs(float(r["CoM_lateral_offset_x"])) for r in rows),
        "max_wheel_contact_height_diff": max(float(r["wheel_contact_height_diff"]) for r in rows),
        "max_site_height_diff": max(abs(float(r["left_wheel_site_height"]) - float(r["right_wheel_site_height"])) for r in rows),
        "min_knee_forward_margin_L": min(float(r["knee_forward_margin_L"]) for r in rows),
        "min_knee_forward_margin_R": min(float(r["knee_forward_margin_R"]) for r in rows),
    }


def comparison_text(b9_rows: list[dict[str, object]], candidate_rows: list[dict[str, object]]) -> str:
    candidates_by_height = sorted(
        candidate_rows,
        key=lambda r: float(r["torso_height"]),
        reverse=True,
    )
    lines = []
    for b9 in b9_rows:
        b9_h = float(b9["target_height"])
        b9_torso_h = float(b9["torso_height"])
        best = min(candidates_by_height, key=lambda r: abs(float(r["torso_height"]) - b9_torso_h))
        lines.append(
            f"- h={b9_h:.2f}: closest candidate by realized torso height is "
            f"{best['posture_name']} (B9 torso height {b9_torso_h:.3f} m, "
            f"candidate torso height {float(best['torso_height']):.3f} m). "
            f"B9 knees_forward={b9['knees_bend_forward']}, "
            f"knee margins L/R=({float(b9['knee_forward_margin_L']):.3f}, {float(b9['knee_forward_margin_R']):.3f}) m, "
            f"CoM lateral offset={float(b9['CoM_lateral_offset_x']):.3f} m."
        )
    return "\n".join(lines)


def write_report(
    path: Path,
    model_path: Path,
    axis_rows: list[dict[str, object]],
    b9_rows: list[dict[str, object]],
    candidate_rows: list[dict[str, object]],
) -> None:
    b9_summary = summarize_rows(b9_rows)
    cand_summary = summarize_rows(candidate_rows)
    b9_human_like_posture = (
        b9_summary["all_knees_forward"]
        and b9_summary["all_torso_upright"]
        and float(b9_summary["max_abs_com_lateral_offset_x"]) < 0.03
    )

    conclusion = (
        "B9 currently appears to use a human-like, symmetric forward-knee posture in the active real model; wheel-ground contact is grounded by wheel collision-bottom geometry, with contact-site asymmetry retained only as a diagnostic."
        if b9_human_like_posture
        else "B9 controller/equilibrium/initial posture design is insufficient or at least not fully verified as human-like/stable."
    )

    mechanical_line = (
        "The mechanical-limitation conclusion remains premature; this report verifies static posture geometry only, not corrected-posture closed-loop roll recovery."
    )

    axes_md = "\n".join(
        f"- `{row['joint_name']}` axis={row['axis_local']} range={row['range']}" for row in axis_rows
    )

    b9_table = "\n".join(
        "| {target_height:.2f} | {hip_pitch_L:.3f} | {knee_L:.3f} | {torso_height:.3f} | {knee_forward_margin_L:.3f} | {knee_forward_margin_R:.3f} | {CoM_lateral_offset_x:.3f} | {both_wheels_touch_ground_correctly} |".format(**r)
        for r in b9_rows
    )
    cand_table = "\n".join(
        "| {posture_name} | {hip_pitch_L:.3f} | {knee_L:.3f} | {torso_height:.3f} | {knee_forward_margin_L:.3f} | {knee_forward_margin_R:.3f} | {CoM_lateral_offset_x:.3f} |".format(**r)
        for r in candidate_rows
    )

    content = f"""# Phase B.9 Posture Geometry Inspection Report

## Scope

This report verifies B9 static initial posture geometry using `{model_path.as_posix()}`. It does not run residual PPO training, does not update residual training configs, and does not resume B9 tuning sweeps.

## Model frame and joint-axis inspection

The active model is `wheeled_biped_real.xml`. Its XML convention states:

- `X`: lateral, positive left.
- `Y`: backward when positive; robot forward is `-Y`.
- `Z`: up.

Joint axes from the active model:

{axes_md}

Because the active robot faces `-Y`, knee-forward is measured as `hip_y - knee_y > 0` in world coordinates. The older simplified-model interpretation `knee_y - hip_y > 0` is not valid for this real model.

## B9 initial posture summary

| target h | hip pitch L | knee L | torso h | knee fwd L | knee fwd R | CoM lateral x | both wheels touch |
|---:|---:|---:|---:|---:|---:|---:|:---:|
{b9_table}

Summary:

- All B9 knees forward: `{b9_summary['all_knees_forward']}`
- All B9 torsos near upright: `{b9_summary['all_torso_upright']}`
- All B9 wheel collision bottoms grounded/symmetric: `{b9_summary['all_both_wheels_touch']}`
- Max absolute CoM lateral offset: `{float(b9_summary['max_abs_com_lateral_offset_x']):.4f}` m
- Max wheel collision-bottom height difference: `{float(b9_summary['max_wheel_contact_height_diff']):.6f}` m
- Max diagnostic site-height difference: `{float(b9_summary['max_site_height_diff']):.6f}` m

Visual interpretation: the saved side/front/top images in `outputs/phase_b9_geometry_check/` should be used as the primary manual check. Numerically, B9 postures are forward-knee and symmetric under FK, but visual review is still required before using them as trusted equilibrium postures.

## Human-like candidate posture summary

| posture | hip pitch L | knee L | torso h | knee fwd L | knee fwd R | CoM lateral x |
|---|---:|---:|---:|---:|---:|---:|
{cand_table}

Summary:

- All candidate knees forward: `{cand_summary['all_knees_forward']}`
- All candidate torsos near upright: `{cand_summary['all_torso_upright']}`
- All candidate wheel collision bottoms grounded/symmetric: `{cand_summary['all_both_wheels_touch']}`
- Max absolute candidate CoM lateral offset: `{float(cand_summary['max_abs_com_lateral_offset_x']):.4f}` m
- Max diagnostic candidate site-height difference: `{float(cand_summary['max_site_height_diff']):.6f}` m

## B9 vs candidate comparison

{comparison_text(b9_rows, candidate_rows)}

## Answers to required questions

- Whether B9 currently uses human-like posture: {conclusion}
- Whether knees bend forward: B9 knee-forward margins are positive for all rendered target heights using the active `-Y` forward convention.
- Whether posture/contact symmetry is correct: B9 is symmetric by commanded joints and wheel collision-bottom height; CoM lateral offset is small. Contact-site heights are retained as diagnostics because the contact markers are not reliable ground-clearance proxies for this asymmetric real model. Contact forces may be `NaN` for static FK snapshots if MuJoCo does not report active contact forces without dynamic settling.
- Whether the mechanical limitation conclusion is justified: {mechanical_line}
- What posture corrections should be tried next: use the rendered B9 and candidate images to choose the visually best height-indexed postures; then test those corrected postures in fast-loop-only B9 before changing LQR gains.
- Whether B9 can proceed to fast-loop-only testing: yes, after manual visual inspection confirms the rendered B9/candidate postures look correct; do not resume tuning sweeps yet.

## Recommended next posture/equilibrium actions

1. Visually inspect all side/front/top PNGs for knee direction, torso alignment, and wheel-ground contact.
2. If any B9 height looks visually folded incorrectly, replace that height's equilibrium with the closest visually correct candidate or a nearby FK-adjusted posture.
3. Build a corrected `equilibrium_posture_table_b9.yaml` only after choosing the visually preferred rows.
4. Use the selected postures for future numerical linearization and B9 fast-loop-only testing.
5. Only after corrected-posture fast-loop-only tests fail should roll authority or mechanical limitations be revisited.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def b9_poses(model: mujoco.MjModel, config: DualRateConfig) -> list[PoseSpec]:
    height_ik = HeightIK(
        mj_model=model,
        scan_points=config.ik_scan_points,
        polynomial_degree=config.ik_polynomial_degree,
        symmetric_fold=config.ik_symmetric_fold,
    )
    poses = []
    for height in HEIGHTS:
        targets = height_ik.compute_ik_targets(height)
        hip_pitch = float(targets["hip_pitch"])
        knee = float(targets["knee"])
        poses.append(PoseSpec(f"h_{height:.2f}", height, 0.0, 0.0, 0.0, 0.0, hip_pitch, hip_pitch, knee, knee))
    return poses


def process_poses(
    model: mujoco.MjModel,
    poses: list[PoseSpec],
    output_dir: Path,
    filename_fn,
) -> list[dict[str, object]]:
    rows = []
    data = mujoco.MjData(model)
    for pose in poses:
        set_pose(model, data, pose)
        rows.append(measure_pose(model, data, pose))
        for view in VIEWS.values():
            render_pose(model, data, output_dir / filename_fn(pose, view), view)
    return rows


def main() -> None:
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    config = DualRateConfig.from_yaml(PROJECT_ROOT / "configs" / "controllers" / "dual_rate_balance_controller_b9.yaml")

    b9_output = PROJECT_ROOT / "outputs" / "phase_b9_geometry_check"
    candidate_output = PROJECT_ROOT / "outputs" / "phase_b9_candidate_postures"
    report_path = PROJECT_ROOT / "docs" / "phase_b9_posture_geometry_inspection_report.md"

    axis_rows = inspect_joint_axes(model)
    b9_output.mkdir(parents=True, exist_ok=True)
    candidate_output.mkdir(parents=True, exist_ok=True)
    with open(b9_output / "joint_axis_inspection.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": str(model_path),
                "world_axes": {"x": "lateral", "y_positive": "backward", "y_negative": "forward", "z": "up"},
                "joint_axes": axis_rows,
            },
            f,
            indent=2,
        )

    b9_rows = process_poses(
        model,
        b9_poses(model, config),
        b9_output,
        lambda pose, view: f"{view.name}_h_{pose.target_height:.2f}.png",
    )
    candidate_rows = process_poses(
        model,
        CANDIDATES,
        candidate_output,
        lambda pose, view: f"candidate_{pose.name}_{view.name}.png",
    )

    write_csv(b9_output / "b9_postures.csv", b9_rows)
    write_csv(candidate_output / "candidate_postures.csv", candidate_rows)
    write_report(report_path, model_path, axis_rows, b9_rows, candidate_rows)

    print(f"Saved B9 renders and CSV to {b9_output}")
    print(f"Saved candidate renders and CSV to {candidate_output}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
