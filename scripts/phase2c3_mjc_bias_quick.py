#!/usr/bin/env python
"""Quick verification: is MuJoCo free-base velocity-dependent bias ALWAYS zero?"""
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

qpos_base = data.qpos.copy()
qpos_base[3:7] = [1.0, 0.0, 0.0, 0.0]  # identity quat

nv = model.nv

def cpu_vel_bias(qvel):
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_base
    d.qvel[:] = qvel
    d.qacc[:] = 0.0
    mujoco.mj_forward(model, d)
    mujoco.mj_inverse(model, d)
    full = np.array(d.qfrc_inverse, dtype=np.float64)

    d0 = mujoco.MjData(model)
    d0.qpos[:] = qpos_base
    d0.qvel[:] = 0.0
    d0.qacc[:] = 0.0
    mujoco.mj_forward(model, d0)
    mujoco.mj_inverse(model, d0)
    grav = np.array(d0.qfrc_inverse, dtype=np.float64)
    return full - grav

test_vels = [
    ("wx=1", [0,0,0, 1,0,0]),
    ("wy=1", [0,0,0, 0,1,0]),
    ("wz=1", [0,0,0, 0,0,1]),
    ("vx=1", [1,0,0, 0,0,0]),
    ("vy=1", [0,1,0, 0,0,0]),
    ("vz=1", [0,0,1, 0,0,0]),
    ("wx=1,vx=1", [1,0,0, 1,0,0]),
    ("wy=1,vy=1", [0,1,0, 0,1,0]),
    ("wz=1,vx=1", [1,0,0, 0,0,1]),
    ("wx=1,vy=1", [0,1,0, 1,0,0]),
    ("wx=2", [0,0,0, 2,0,0]),
    ("wz=2", [0,0,0, 0,0,2]),
    ("wx=1,wz=1", [0,0,0, 1,0,1]),
    ("all_base", [0.5, 0.3, 0.2, 1.0, 0.8, 0.6]),
]

print(f"{'Case':<18} {'vb[0]':>10} {'vb[1]':>10} {'vb[2]':>10} {'vb[3]':>10} {'vb[4]':>10} {'vb[5]':>10} {'||vb[0:6]||':>12}")
print("-" * 82)
for name, vb_raw in test_vels:
    qv = np.zeros(nv)
    qv[0:6] = vb_raw
    vb = cpu_vel_bias(qv)
    norm = np.linalg.norm(vb[0:6])
    print(f"{name:<18} {vb[0]:>10.6f} {vb[1]:>10.6f} {vb[2]:>10.6f} {vb[3]:>10.6f} {vb[4]:>10.6f} {vb[5]:>10.6f} {norm:>12.2e}")
