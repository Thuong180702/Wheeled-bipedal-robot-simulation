"""Model index and state snapshot utilities.

Read-only inspection of MuJoCo model structure: joint/actuator/body/geom/site
name-to-index mappings, dimensions, limits, and state snapshots.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from mujoco import mjtObj


def build_model_index_report(model: mujoco.MjModel) -> dict[str, Any]:
    """Return name/index maps for joints, actuators, bodies, geoms, and sites.

    Also reports qpos/qvel/actuator dimensions and actuator force/control limits.

    Args:
        model: MuJoCo MjModel instance.

    Returns:
        dict with keys:
            nq, nv, nu, nbody, njnt, ngeom, nsite, nkey, nsensor,
            joints, actuators, bodies, geoms, sites,
            actuator_ctrlrange, actuator_forcerange,
            qpos0, keyframe_names.
    """
    # ── dimensions ──────────────────────────────────────────────
    report: dict[str, Any] = {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "njnt": model.njnt,
        "ngeom": model.ngeom,
        "nsite": model.nsite,
        "nkey": model.nkey,
        "nsensor": model.nsensor,
    }

    # ── joint name → index ──────────────────────────────────────
    joints: dict[str, dict[str, Any]] = {}
    for jid in range(model.njnt):
        name = mujoco.mj_id2name(model, mjtObj.mjOBJ_JOINT, jid) or f"<unnamed_joint_{jid}>"
        jtype = model.jnt_type[jid]
        qpos_adr = model.jnt_qposadr[jid]
        dof_adr = model.jnt_dofadr[jid]
        joints[name] = {
            "id": jid,
            "type": int(jtype),
            "type_name": _joint_type_name(jtype),
            "qpos_adr": int(qpos_adr),
            "dof_adr": int(dof_adr),
            "limited": bool(model.jnt_limited[jid]),
            "range": model.jnt_range[jid].tolist() if model.jnt_limited[jid] else None,
        }
    report["joints"] = joints

    # ── actuator name → index ───────────────────────────────────
    actuators: dict[str, dict[str, Any]] = {}
    for aid in range(model.nu):
        name = mujoco.mj_id2name(model, mjtObj.mjOBJ_ACTUATOR, aid) or f"<unnamed_actuator_{aid}>"
        trnid = model.actuator_trnid[aid]
        # trnid[0] is the transmission joint id
        joint_name = (
            mujoco.mj_id2name(model, mjtObj.mjOBJ_JOINT, trnid[0])
            if trnid[0] >= 0
            else "<none>"
        )
        actuators[name] = {
            "id": aid,
            "joint_id": int(trnid[0]),
            "joint_name": joint_name,
            "ctrlrange": model.actuator_ctrlrange[aid].tolist(),
            "forcerange": model.actuator_forcerange[aid].tolist(),
            "gear": model.actuator_gear[aid].tolist() if hasattr(model, "actuator_gear") else None,
            "trntype": int(model.actuator_trntype[aid]),
        }
    report["actuators"] = actuators
    report["actuator_ctrlrange"] = model.actuator_ctrlrange.tolist()
    report["actuator_forcerange"] = model.actuator_forcerange.tolist()

    # ── body name → index ───────────────────────────────────────
    bodies: dict[str, dict[str, Any]] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mjtObj.mjOBJ_BODY, bid) or f"<unnamed_body_{bid}>"
        parent_id = int(model.body_parentid[bid])
        parent_name = (
            mujoco.mj_id2name(model, mjtObj.mjOBJ_BODY, parent_id)
            if parent_id >= 0
            else "<world>"
        )
        bodies[name] = {
            "id": bid,
            "parent_id": parent_id,
            "parent_name": parent_name,
            "mass": float(model.body_mass[bid]) if hasattr(model, "body_mass") else None,
        }
    report["bodies"] = bodies

    # ── geom name → index ───────────────────────────────────────
    geoms: dict[str, dict[str, Any]] = {}
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mjtObj.mjOBJ_GEOM, gid) or f"<unnamed_geom_{gid}>"
        body_id = int(model.geom_bodyid[gid])
        body_name = (
            mujoco.mj_id2name(model, mjtObj.mjOBJ_BODY, body_id)
            if body_id >= 0
            else "<world>"
        )
        geoms[name] = {
            "id": gid,
            "body_id": body_id,
            "body_name": body_name,
            "type": int(model.geom_type[gid]),
            "size": model.geom_size[gid].tolist(),
        }
    report["geoms"] = geoms

    # ── site name → index ───────────────────────────────────────
    sites: dict[str, dict[str, Any]] = {}
    for sid in range(model.nsite):
        name = mujoco.mj_id2name(model, mjtObj.mjOBJ_SITE, sid) or f"<unnamed_site_{sid}>"
        body_id = int(model.site_bodyid[sid])
        body_name = (
            mujoco.mj_id2name(model, mjtObj.mjOBJ_BODY, body_id)
            if body_id >= 0
            else "<world>"
        )
        sites[name] = {
            "id": sid,
            "body_id": body_id,
            "body_name": body_name,
            "pos": model.site_pos[sid].tolist(),
        }
    report["sites"] = sites

    # ── default pose ────────────────────────────────────────────
    report["qpos0"] = model.qpos0.tolist()
    report["keyframe_names"] = [
        mujoco.mj_id2name(model, mjtObj.mjOBJ_KEY, k) or f"keyframe_{k}"
        for k in range(model.nkey)
    ]

    return report


def extract_state_snapshot(
    model: mujoco.MjModel, data: mujoco.MjData
) -> dict[str, Any]:
    """Return a serializable snapshot of the current simulation state.

    Extracts base pose, joint positions/velocities, actuator ctrl values,
    body poses, and COM position.

    Args:
        model: MuJoCo MjModel instance.
        data: MuJoCo MjData instance (must have mj_forward already called).

    Returns:
        dict with keys:
            qpos, qvel, qacc, ctrl, act, time,
            base_position, base_quaternion,
            base_linear_velocity, base_angular_velocity,
            joint_positions, joint_velocities,
            body_positions, com_position, com_velocity.
    """
    snapshot: dict[str, Any] = {
        "qpos": data.qpos.copy().tolist(),
        "qvel": data.qvel.copy().tolist(),
        "qacc": data.qacc.copy().tolist() if data.qacc is not None else None,
        "ctrl": data.ctrl.copy().tolist(),
        "act": data.act.copy().tolist() if data.act is not None else None,
        "time": float(data.time),
    }

    # Floating base (free joint: qpos[0:7] = [x,y,z, qw,qx,qy,qz])
    snapshot["base_position"] = data.qpos[0:3].copy().tolist()
    snapshot["base_quaternion"] = data.qpos[3:7].copy().tolist()

    # Base linear/angular velocity (qvel[0:6])
    snapshot["base_linear_velocity"] = data.qvel[0:3].copy().tolist()
    snapshot["base_angular_velocity"] = data.qvel[3:6].copy().tolist()

    # Joint positions (actuated joints: qpos[7:17])
    snapshot["joint_positions"] = data.qpos[7:17].copy().tolist()
    snapshot["joint_velocities"] = data.qvel[6:16].copy().tolist()

    # ── body positions ──────────────────────────────────────────
    body_positions: dict[str, list[float]] = {}
    for bid in range(model.nbody):
        name = mujoco.mj_id2name(model, mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"
        body_positions[name] = data.xpos[bid].copy().tolist()
    snapshot["body_positions"] = body_positions

    # ── COM ─────────────────────────────────────────────────────
    # subtotal COM: data.subtree_com[body_id] is the subtree COM in world frame
    # torso is body 1 (world is 0). Use torso subtree_com as approximate robot COM.
    torso_id = mujoco.mj_name2id(model, mjtObj.mjOBJ_BODY, "torso")
    if torso_id >= 0:
        snapshot["com_position"] = data.subtree_com[torso_id].copy().tolist()
    else:
        snapshot["com_position"] = None

    # COM velocity: data.cvel is (nbody, 6) — per-body subtree COM velocity.
    # torso subtree (body 1) approximates full robot COM velocity.
    if hasattr(data, "cvel") and data.cvel is not None:
        # data.cvel[torso_id] gives torso subtree COM linear + angular velocity
        snapshot["com_velocity"] = data.cvel[torso_id, 0:3].copy().tolist()
    else:
        # Fallback: body-origin velocity via mj_objectVelocity
        com_vel = np.zeros(6)
        mujoco.mj_objectVelocity(
            model,
            data,
            mjtObj.mjOBJ_BODY,
            torso_id,
            com_vel,
            0,  # flg_local = False → world frame
        )
        snapshot["com_velocity"] = com_vel[0:3].tolist()

    # ── NaN/Inf check ───────────────────────────────────────────
    snapshot["qpos_finite"] = bool(np.all(np.isfinite(data.qpos)))
    snapshot["qvel_finite"] = bool(np.all(np.isfinite(data.qvel)))

    return snapshot


def _joint_type_name(jtype: int) -> str:
    """Convert MuJoCo joint type integer to human-readable name."""
    type_names = {
        mujoco.mjtJoint.mjJNT_FREE: "free",
        mujoco.mjtJoint.mjJNT_BALL: "ball",
        mujoco.mjtJoint.mjJNT_SLIDE: "slide",
        mujoco.mjtJoint.mjJNT_HINGE: "hinge",
    }
    return type_names.get(jtype, f"unknown({jtype})")
