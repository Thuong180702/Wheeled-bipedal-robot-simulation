"""Contact state inspection utilities.

Read-only inspection of MuJoCo contact data: contact pairs, positions,
normals, and estimated contact forces.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from mujoco import mjtObj


def inspect_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, Any]:
    """Return current contact list with detailed per-contact information.

    Inspects all active contacts, resolves geom/body names, and
    computes contact forces via mj_contactForce.

    Args:
        model: MuJoCo MjModel.
        data: MuJoCo MjData (mj_forward already called, physics stepped).

    Returns:
        dict with keys:
            ncon, contacts (list), wheel_contacts, floor_contacts,
            left_wheel_in_contact, right_wheel_in_contact,
            total_contact_force_world.
    """
    ncon = data.ncon
    contacts: list[dict[str, Any]] = []

    # Resolve known contact-relevant geom IDs
    floor_id = _safe_name2id(model, mjtObj.mjOBJ_GEOM, "floor")
    l_wheel_geom_id = _safe_name2id(model, mjtObj.mjOBJ_GEOM, "l_wheel_collision")
    r_wheel_geom_id = _safe_name2id(model, mjtObj.mjOBJ_GEOM, "r_wheel_collision")

    wheel_contact_indices: list[int] = []
    floor_contact_indices: list[int] = []
    l_wheel_in_contact = False
    r_wheel_in_contact = False

    total_force = np.zeros(3)

    for i in range(ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        # Resolve geom/body names
        g1_name = _safe_id2name(model, mjtObj.mjOBJ_GEOM, g1)
        g2_name = _safe_id2name(model, mjtObj.mjOBJ_GEOM, g2)
        body1_id = int(model.geom_bodyid[g1]) if g1 < model.ngeom else -1
        body2_id = int(model.geom_bodyid[g2]) if g2 < model.ngeom else -1
        b1_name = _safe_id2name(model, mjtObj.mjOBJ_BODY, body1_id) if body1_id >= 0 else "<none>"
        b2_name = _safe_id2name(model, mjtObj.mjOBJ_BODY, body2_id) if body2_id >= 0 else "<none>"

        # Contact frame: first 3 columns, last column is contact position
        frame = np.array(c.frame).reshape(3, 3)
        normal = frame[:, 0].copy()  # contact normal in world frame
        pos = np.array(c.pos)

        # Contact force
        force_raw = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force_raw)
        force_contact = force_raw[:3]  # in contact frame
        # Transform to world frame
        force_world = frame.T @ force_contact

        total_force += force_world

        # Classify
        involves_floor = (g1 == floor_id) or (g2 == floor_id)
        involves_l_wheel = (g1 == l_wheel_geom_id) or (g2 == l_wheel_geom_id)
        involves_r_wheel = (g1 == r_wheel_geom_id) or (g2 == r_wheel_geom_id)
        is_wheel_contact = involves_l_wheel or involves_r_wheel

        if is_wheel_contact:
            wheel_contact_indices.append(i)
        if involves_floor:
            floor_contact_indices.append(i)
        if involves_l_wheel:
            l_wheel_in_contact = True
        if involves_r_wheel:
            r_wheel_in_contact = True

        contacts.append({
            "index": i,
            "geom1": g1_name,
            "geom2": g2_name,
            "body1": b1_name,
            "body2": b2_name,
            "position": pos.tolist(),
            "normal": normal.tolist(),
            "force_contact_frame": force_contact.tolist(),
            "force_world": force_world.tolist(),
            "distance": float(c.dist),
            "involves_floor": involves_floor,
            "involves_l_wheel": involves_l_wheel,
            "involves_r_wheel": involves_r_wheel,
        })

    return {
        "ncon": ncon,
        "contacts": contacts,
        "wheel_contact_indices": wheel_contact_indices,
        "floor_contact_indices": floor_contact_indices,
        "left_wheel_in_contact": l_wheel_in_contact,
        "right_wheel_in_contact": r_wheel_in_contact,
        "total_contact_force_world": total_force.tolist(),
    }


def _safe_name2id(model: mujoco.MjModel, obj_type: int, name: str) -> int:
    """Safe name-to-ID lookup — returns -1 if not found."""
    try:
        return mujoco.mj_name2id(model, obj_type, name)
    except Exception:
        return -1


def _safe_id2name(model: mujoco.MjModel, obj_type: int, obj_id: int) -> str:
    """Safe ID-to-name lookup — returns '<unnamed>' if not found."""
    if obj_id < 0:
        return "<none>"
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else f"<unnamed_{obj_id}>"
