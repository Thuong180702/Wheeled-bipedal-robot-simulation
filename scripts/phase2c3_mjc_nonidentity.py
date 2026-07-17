#!/usr/bin/env python
"""Test: is MuJoCo free-base velocity-dependent bias zero at non-identity orientations?"""
from __future__ import annotations
import sys
from pathlib import Path
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from wheeled_biped.utils.config import get_model_path

model = mujoco.MjModel.from_xml_path(str(get_model_path()))
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

from scipy.spatial.transform import Rotation
nv = model.nv

def cpu_vel_bias(qpos, qvel):
    d = mujoco.MjData(model); d.qpos[:] = qpos; d.qvel[:] = qvel; d.qacc[:] = 0
    mujoco.mj_forward(model, d); mujoco.mj_inverse(model, d)
    full = np.array(d.qfrc_inverse, dtype=np.float64)
    d0 = mujoco.MjData(model); d0.qpos[:] = qpos; d0.qvel[:] = 0; d0.qacc[:] = 0
    mujoco.mj_forward(model, d0); mujoco.mj_inverse(model, d0)
    return full - np.array(d0.qfrc_inverse, dtype=np.float64)

def make_orientation(roll_deg, pitch_deg, yaw_deg):
    R = Rotation.from_euler('xyz', np.deg2rad([roll_deg, pitch_deg, yaw_deg])).as_matrix()
    quat = Rotation.from_matrix(R).as_quat()  # [x,y,z,w]
    qpos = data.qpos.copy()
    qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
    return qpos

# Test with non-identity orientations
print(f"{'Orientation':<22} {'wz=1':>12} {'vx=1':>12} {'wz+vx':>12} {'cross ||fb||':>14}")
print("-" * 76)

for label, r, p, y in [
    ("identity", 0, 0, 0),
    ("roll 10", 10, 0, 0),
    ("pitch 10", 0, 10, 0),
    ("yaw 15", 0, 0, 15),
    ("r5 p8 y12", 5, 8, 12),
    ("roll 30", 30, 0, 0),
    ("pitch 45", 0, 45, 0),
    ("yaw 90", 0, 0, 90),
]:
    qp = make_orientation(r, p, y)
    q0 = np.zeros(nv)
    qwz = np.zeros(nv); qwz[5] = 1.0
    qvx = np.zeros(nv); qvx[0] = 1.0
    qboth = np.zeros(nv); qboth[0] = 1.0; qboth[5] = 1.0

    vb_wz = cpu_vel_bias(qp, qwz)
    vb_vx = cpu_vel_bias(qp, qvx)
    vb_both = cpu_vel_bias(qp, qboth)
    vb_zero = cpu_vel_bias(qp, q0)
    cross = vb_both - vb_wz - vb_vx + vb_zero

    n_wz = np.linalg.norm(vb_wz[0:6])
    n_vx = np.linalg.norm(vb_vx[0:6])
    n_both = np.linalg.norm(vb_both[0:6])
    n_cross = np.linalg.norm(cross[0:6])
    print(f"{label:<22} {n_wz:>12.2e} {n_vx:>12.2e} {n_both:>12.2e} {n_cross:>14.2e}")

# Also show the actual components for a couple cases
print("\n--- Detailed non-identity cross-terms ---")
for label, r, p, y in [("roll 30", 30, 0, 0), ("pitch 45", 0, 45, 0), ("yaw 90", 0, 0, 90)]:
    qp = make_orientation(r, p, y)
    qwz = np.zeros(nv); qwz[5] = 1.0
    qvx = np.zeros(nv); qvx[0] = 1.0
    qboth = np.zeros(nv); qboth[0] = 1.0; qboth[5] = 1.0
    q0 = np.zeros(nv)

    vb_wz = cpu_vel_bias(qp, qwz)
    vb_vx = cpu_vel_bias(qp, qvx)
    vb_both = cpu_vel_bias(qp, qboth)
    vb_zero = cpu_vel_bias(qp, q0)
    cross = vb_both - vb_wz - vb_vx + vb_zero

    print(f"\n  {label}:")
    print(f"    Pure wz vb[0:6]: {vb_wz[0:6]}")
    print(f"    Pure vx vb[0:6]: {vb_vx[0:6]}")
    print(f"    Cross fb[0:3]: {cross[0:3]}")
    print(f"    Cross fb[3:6]: {cross[3:6]}")
    print(f"    ||cross fb||: {np.linalg.norm(cross[0:6]):.2e}")

# Also check with randomized base velocity
print("\n--- Random base velocity at non-identity orientations ---")
rng = np.random.default_rng(42)
for label, r, p, y in [("identity", 0, 0, 0), ("roll 10", 10, 0, 0), ("yaw 30", 0, 0, 30)]:
    qp = make_orientation(r, p, y)
    qv = np.zeros(nv)
    qv[0:6] = rng.uniform(-1, 1, 6)
    vb = cpu_vel_bias(qp, qv)
    print(f"  {label}: rand_qvel, ||vb[0:6]|| = {np.linalg.norm(vb[0:6]):.2e}")
