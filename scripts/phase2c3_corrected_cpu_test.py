#!/usr/bin/env python
"""Correct test: use data.qfrc_bias directly from mj_forward (NOT mj_inverse)."""
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
nv = model.nv

def cpu_qfrc_bias(qpos, qvel):
    """Get qfrc_bias directly from mj_forward."""
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(model, d)
    # qfrc_bias is computed during mj_forward
    return np.array(d.qfrc_bias, dtype=np.float64)

# Base pose at identity
qpos_base = data.qpos.copy()
qpos_base[3:7] = [1.0, 0.0, 0.0, 0.0]

# Test: pure base velocities and cross-terms
print("CORRECT CPU TEST (data.qfrc_bias directly)")
print("=" * 60)
print()

# Gravity at identity
grav = cpu_qfrc_bias(qpos_base, np.zeros(nv))
print(f"Gravity fb[0:6]: {grav[0:6]}")

# Pure velocities
print("\nPure free-base velocities:")
for name, qv_raw in [
    ("wz=1", [0,0,0, 0,0,1]),
    ("vx=1", [1,0,0, 0,0,0]),
    ("wx=1", [0,0,0, 1,0,0]),
    ("wy=1", [0,0,0, 0,1,0]),
    ("vy=1", [0,1,0, 0,0,0]),
    ("vz=1", [0,0,1, 0,0,0]),
]:
    qv = np.zeros(nv); qv[0:6] = qv_raw
    bias = cpu_qfrc_bias(qpos_base, qv)
    vb = bias - grav
    print(f"  {name}: vb[0:3]={vb[0:3]} vb[3:6]={vb[3:6]} ||vb||={np.linalg.norm(vb[0:6]):.2e}")

# Cross-terms
print("\nCross-terms (wz + vx):")
qwz = np.zeros(nv); qwz[5] = 1.0
qvx = np.zeros(nv); qvx[0] = 1.0
qboth = np.zeros(nv); qboth[0] = 1.0; qboth[5] = 1.0

b_wz = cpu_qfrc_bias(qpos_base, qwz)
b_vx = cpu_qfrc_bias(qpos_base, qvx)
b_both = cpu_qfrc_bias(qpos_base, qboth)
b_zero = cpu_qfrc_bias(qpos_base, np.zeros(nv))

cross = b_both - b_wz - b_vx + b_zero
print(f"  Cross fb[0:3] (force):  {cross[0:3]}")
print(f"  Cross fb[3:6] (torque): {cross[3:6]}")
print(f"  ||cross fb||: {np.linalg.norm(cross[0:6]):.2e}")

# Also check actuated part
print(f"  Cross actuated[6:16]: {cross[6:16]}")
print(f"  ||cross act||: {np.linalg.norm(cross[6:16]):.2e}")

# All 9 w x v pairs
print("\nAll 9 angular x linear cross-terms:")
for ai, an in [(3,"wx"),(4,"wy"),(5,"wz")]:
    for li, ln in [(0,"vx"),(1,"vy"),(2,"vz")]:
        q_ang = np.zeros(nv); q_ang[ai] = 1.0
        q_lin = np.zeros(nv); q_lin[li] = 1.0
        q_sum = np.zeros(nv); q_sum[ai] = 1.0; q_sum[li] = 1.0
        b_ang = cpu_qfrc_bias(qpos_base, q_ang)
        b_lin = cpu_qfrc_bias(qpos_base, q_lin)
        b_sum = cpu_qfrc_bias(qpos_base, q_sum)
        cross9 = b_sum - b_ang - b_lin + b_zero
        n9 = np.linalg.norm(cross9[0:6])
        print(f"  {an}+{ln}: ||cross fb||={n9:.2e} force={cross9[0:3]} torque={cross9[3:6]}")

# test with all 6 base DOFs
print("\nAll 6 base DOFs at 0.5:")
qv6 = np.zeros(nv); qv6[0:6] = [0.5, 0.3, 0.2, 0.4, 0.6, 0.8]
b6 = cpu_qfrc_bias(qpos_base, qv6)
vb6 = b6 - grav
print(f"  vb[0:6] = {vb6[0:6]}")
print(f"  ||vb[0:6]|| = {np.linalg.norm(vb6[0:6]):.6f}")
