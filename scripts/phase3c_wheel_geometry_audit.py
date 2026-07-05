#!/usr/bin/env python
"""Phase 3C — Wheel Geometry Audit for Rolling Constraints.

Inspects the MuJoCo model to extract wheel rolling geometry constants:
  - Wheel body IDs
  - Wheel joint IDs and qvel indices
  - Wheel joint axes (local and world frame)
  - Wheel radii from collision geom metadata
  - Wheel collision geom names and classification
  - Contact frame normal/tangent convention

All data comes from the model XML / MuJoCo API. No hardcoded constants.

Usage:
  python scripts/phase3c_wheel_geometry_audit.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Phase 3C modules must be imported first to register in sys.modules
# before other wheeled_biped imports interact with the editable finder.
import wheeled_biped.wbc.offline_rolling_constraints  # noqa: F401

import json
import mujoco
import numpy as np

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.wbc.offline_rolling_constraints import build_wheel_rolling_constants


def main():
    model_path = get_model_path()
    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))

    print(f"\nModel summary:")
    print(f"  nq = {model.nq}")
    print(f"  nv = {model.nv}")
    print(f"  nu = {model.nu}")
    print(f"  njnt = {model.njnt}")
    print(f"  ngeom = {model.ngeom}")
    print(f"  nbody = {model.nbody}")

    # ── Build rolling constants ────────────────────────────────────────
    rolling = build_wheel_rolling_constants(model)

    print(f"\n{'='*60}")
    print(f"WHEEL ROLLING GEOMETRY")
    print(f"{'='*60}")

    print(f"\nWheel radii (from collision geom metadata):")
    print(f"  left  radius = {rolling['l_wheel_radius']:.6f} m")
    print(f"  right radius = {rolling['r_wheel_radius']:.6f} m")

    print(f"\nWheel body IDs:")
    print(f"  l_wheel_link = {rolling['l_wheel_body_id']}")
    print(f"  r_wheel_link = {rolling['r_wheel_body_id']}")

    print(f"\nWheel joint IDs and qvel indices:")
    print(f"  l_wheel: joint_id={rolling['l_wheel_joint_id']}, qvel_index={rolling['l_wheel_qvel_index']}")
    print(f"  r_wheel: joint_id={rolling['r_wheel_joint_id']}, qvel_index={rolling['r_wheel_qvel_index']}")

    print(f"\nWheel joint axes (joint-local frame):")
    print(f"  l_wheel axis_local = {rolling['l_wheel_axis_local']}")
    print(f"  r_wheel axis_local = {rolling['r_wheel_axis_local']}")

    print(f"\nWheel collision geom IDs:")
    print(f"  l_wheel_collision = {rolling['l_wheel_geom_id']}")
    print(f"  r_wheel_collision = {rolling['r_wheel_geom_id']}")

    # ── Verify geom metadata ──────────────────────────────────────────
    print(f"\nGeom metadata verification:")
    for side, gid in [("left", rolling["l_wheel_geom_id"]),
                       ("right", rolling["r_wheel_geom_id"])]:
        gtype = model.geom_type[gid]
        gsize = model.geom_size[gid]
        gbody = model.geom_bodyid[gid]
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        type_names = {0: "NONE", 1: "PLANE", 2: "HSPHERE", 3: "SPHERE",
                      4: "CAPSULE", 5: "CYLINDER", 6: "BOX", 7: "MESH"}
        print(f"  {side}: {gname} — type={type_names.get(gtype, str(gtype))}, "
              f"size={gsize}, body_id={gbody}")

    # ── Joint axis validation in world frame ───────────────────────────
    print(f"\nJoint axis in nominal configuration (world frame):")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for side, joint_id, axis_local in [
        ("left", rolling["l_wheel_joint_id"], rolling["l_wheel_axis_local"]),
        ("right", rolling["r_wheel_joint_id"], rolling["r_wheel_axis_local"]),
    ]:
        # Get joint's body world orientation
        joint_body_id = model.jnt_bodyid[joint_id]
        body_xmat = data.xmat[joint_body_id].reshape(3, 3).copy()
        axis_world = body_xmat @ axis_local
        print(f"  {side}: axis_local={axis_local}, axis_world={axis_world}")

    # ── Check validity ──────────────────────────────────────────────────
    checks = []
    checks.append(("radius_left_positive", rolling["l_wheel_radius"] > 0))
    checks.append(("radius_right_positive", rolling["r_wheel_radius"] > 0))
    checks.append(("radius_finite", np.isfinite(rolling["l_wheel_radius"]) and
                   np.isfinite(rolling["r_wheel_radius"])))
    checks.append(("qvel_left_valid", 0 <= rolling["l_wheel_qvel_index"] < model.nv))
    checks.append(("qvel_right_valid", 0 <= rolling["r_wheel_qvel_index"] < model.nv))
    checks.append(("axis_left_nonzero", np.linalg.norm(rolling["l_wheel_axis_local"]) > 0))
    checks.append(("axis_right_nonzero", np.linalg.norm(rolling["r_wheel_axis_local"]) > 0))
    checks.append(("body_ids_positive", rolling["l_wheel_body_id"] >= 0 and
                   rolling["r_wheel_body_id"] >= 0))
    checks.append(("geom_ids_positive", rolling["l_wheel_geom_id"] >= 0 and
                   rolling["r_wheel_geom_id"] >= 0))

    print(f"\n{'='*60}")
    print(f"VALIDATION CHECKS")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    if all_pass:
        print(f"\nVERDICT: PASS — Wheel geometry successfully extracted.")
    else:
        print(f"\nVERDICT: FAIL — Cannot determine wheel geometry robustly.")
        return 1

    # ── Output JSON ──────────────────────────────────────────────────────
    output = {
        "phase": "3C",
        "component": "wheel_geometry_audit",
        "constants_version": rolling["constants_version"],
        "wheel_radius_left": rolling["l_wheel_radius"],
        "wheel_radius_right": rolling["r_wheel_radius"],
        "wheel_axis_left_local": rolling["l_wheel_axis_local"].tolist(),
        "wheel_axis_right_local": rolling["r_wheel_axis_local"].tolist(),
        "wheel_qvel_index_left": rolling["l_wheel_qvel_index"],
        "wheel_qvel_index_right": rolling["r_wheel_qvel_index"],
        "wheel_body_id_left": rolling["l_wheel_body_id"],
        "wheel_body_id_right": rolling["r_wheel_body_id"],
        "wheel_geom_id_left": rolling["l_wheel_geom_id"],
        "wheel_geom_id_right": rolling["r_wheel_geom_id"],
        "nv": model.nv,
        "all_checks_passed": all_pass,
    }
    print(f"\nJSON summary:\n{json.dumps(output, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
