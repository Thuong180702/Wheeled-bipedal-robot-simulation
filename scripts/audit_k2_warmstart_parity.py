#!/usr/bin/env python3
"""
K2 Physics Initialization and Warm-Start Parity Audit
======================================================
Compare post-init state between Python source-style init (4 mj_forward) and
JAX dedicated-style init (1 mj_forward). Also test 2-forward source-equivalent init.

Usage:
  python scripts/audit_k2_warmstart_parity.py --height low_0p380
  python scripts/audit_k2_warmstart_parity.py --heights low_0p320,low_0p360,low_0p380,high_0p450
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent.parent
SETUP_DIR = ROOT / "outputs" / "physical_target_height_setups"


def load_setup(height_label: str) -> dict:
    path = SETUP_DIR / f"{height_label}_setup.json"
    with open(path) as f:
        return json.load(f)


def apply_qpos(mj_data, height_setup: dict):
    """Apply height setup joint positions to mj_data."""
    mj_data.qpos[7:17] = [
        height_setup.get("hip_roll_left", 0.0),
        height_setup.get("hip_yaw_left", 0.0),
        height_setup.get("hip_pitch_ref", 0.0),
        height_setup.get("knee_ref", 0.0), 0.0,
        height_setup.get("hip_roll_right", 0.0),
        height_setup.get("hip_yaw_right", 0.0),
        height_setup.get("hip_pitch_ref", 0.0),
        height_setup.get("knee_ref", 0.0), 0.0,
    ]


def audit_init_parity(height_label: str):
    """Compare post-init state between different init modes."""
    print(f"\n{'='*70}")
    print(f"Warm-Start Parity Audit — {height_label}")
    print(f"{'='*70}")

    height_setup = load_setup(height_label)
    model_path = str(ROOT / "assets" / "robot" / "wheeled_biped_real.xml")

    # ── Mode 1: Dedicated-style (1 mj_forward) ──────────────────────────
    mj_model_1 = mujoco.MjModel.from_xml_path(model_path)
    mj_data_1 = mujoco.MjData(mj_model_1)
    apply_qpos(mj_data_1, height_setup)
    if "calibrated_root_z_m" in height_setup:
        mj_data_1.qpos[2] = height_setup["calibrated_root_z_m"]
    mujoco.mj_forward(mj_model_1, mj_data_1)
    state_1 = {
        "qpos": np.array(mj_data_1.qpos, copy=True),
        "qvel": np.array(mj_data_1.qvel, copy=True),
        "qacc_warmstart": np.array(mj_data_1.qacc_warmstart, copy=True),
        "qfrc_constraint": np.array(mj_data_1.qfrc_constraint, copy=True),
        "actuator_force": np.array(mj_data_1.actuator_force, copy=True),
        "ctrl": np.array(mj_data_1.ctrl, copy=True),
    }

    # ── Mode 2: 2-forward source-equivalent ─────────────────────────────
    mj_model_2 = mujoco.MjModel.from_xml_path(model_path)
    mj_data_2 = mujoco.MjData(mj_model_2)
    apply_qpos(mj_data_2, height_setup)
    # First mj_forward (before root_z calibration)
    mujoco.mj_forward(mj_model_2, mj_data_2)
    # Apply calibrated root_z
    if "calibrated_root_z_m" in height_setup:
        mj_data_2.qpos[2] = height_setup["calibrated_root_z_m"]
    # Second mj_forward (after root_z calibration)
    mujoco.mj_forward(mj_model_2, mj_data_2)
    state_2 = {
        "qpos": np.array(mj_data_2.qpos, copy=True),
        "qvel": np.array(mj_data_2.qvel, copy=True),
        "qacc_warmstart": np.array(mj_data_2.qacc_warmstart, copy=True),
        "qfrc_constraint": np.array(mj_data_2.qfrc_constraint, copy=True),
        "actuator_force": np.array(mj_data_2.actuator_force, copy=True),
        "ctrl": np.array(mj_data_2.ctrl, copy=True),
    }

    # ── Mode 3: Full source-style (4 mj_forward + equilibrium capture) ──
    mj_model_3 = mujoco.MjModel.from_xml_path(model_path)
    mj_data_3 = mujoco.MjData(mj_model_3)
    apply_qpos(mj_data_3, height_setup)
    # (1) After keyframe reset
    mujoco.mj_forward(mj_model_3, mj_data_3)
    # (2) After root_z calibration
    if "calibrated_root_z_m" in height_setup:
        mj_data_3.qpos[2] = height_setup["calibrated_root_z_m"]
    mujoco.mj_forward(mj_model_3, mj_data_3)
    # (3) For equilibrium capture
    mujoco.mj_forward(mj_model_3, mj_data_3)
    # (4) For support center capture
    mujoco.mj_forward(mj_model_3, mj_data_3)
    eq_joint = np.array(mj_data_3.qpos[7:17], copy=True)
    state_3 = {
        "qpos": np.array(mj_data_3.qpos, copy=True),
        "qvel": np.array(mj_data_3.qvel, copy=True),
        "qacc_warmstart": np.array(mj_data_3.qacc_warmstart, copy=True),
        "qfrc_constraint": np.array(mj_data_3.qfrc_constraint, copy=True),
        "actuator_force": np.array(mj_data_3.actuator_force, copy=True),
        "ctrl": np.array(mj_data_3.ctrl, copy=True),
    }

    # ── Mode 4: Dedicated with source-warmstart (2-forward, then mj_step) ──
    mj_model_4 = mujoco.MjModel.from_xml_path(model_path)
    mj_data_4 = mujoco.MjData(mj_model_4)
    apply_qpos(mj_data_4, height_setup)
    mujoco.mj_forward(mj_model_4, mj_data_4)
    if "calibrated_root_z_m" in height_setup:
        mj_data_4.qpos[2] = height_setup["calibrated_root_z_m"]
    mujoco.mj_forward(mj_model_4, mj_data_4)
    # Also do mj_step (like step 0 diagnostic step in Python path)
    n_substeps = max(1, int(round(0.01 / mj_model_4.opt.timestep)))
    for _ in range(n_substeps):
        mujoco.mj_step(mj_model_4, mj_data_4)
    state_4 = {
        "qpos": np.array(mj_data_4.qpos, copy=True),
        "qvel": np.array(mj_data_4.qvel, copy=True),
        "qacc_warmstart": np.array(mj_data_4.qacc_warmstart, copy=True),
        "qfrc_constraint": np.array(mj_data_4.qfrc_constraint, copy=True),
        "actuator_force": np.array(mj_data_4.actuator_force, copy=True),
        "ctrl": np.array(mj_data_4.ctrl, copy=True),
    }

    # ── Compare ──────────────────────────────────────────────────────────
    modes = {
        "1-forward (dedicated)": state_1,
        "2-forward (source-equiv)": state_2,
        "4-forward (full source)": state_3,
        "2-forward + mj_step": state_4,
    }

    # Compare all modes against each other
    mode_names = list(modes.keys())
    print(f"\n{'Field':<25s} | {'1->2 delta':>14s} | {'1->3 delta':>14s} | {'2->3 delta':>14s} | {'1->4 delta':>14s}")
    print("-" * 100)

    key_fields = [
        ("qpos[2] (root_z)", lambda s: s["qpos"][2]),
        ("qpos[7] (l_hip_roll)", lambda s: s["qpos"][7]),
        ("qpos[11] (l_knee)", lambda s: s["qpos"][11]),
        ("max|qvel|", lambda s: np.max(np.abs(s["qvel"]))),
        ("max|qfrc_constraint|", lambda s: np.max(np.abs(s["qfrc_constraint"]))),
        ("max|actuator_force|", lambda s: np.max(np.abs(s["actuator_force"]))),
        ("max|qacc_warmstart|", lambda s: np.max(np.abs(s["qacc_warmstart"]))),
        ("max|ctrl|", lambda s: np.max(np.abs(s["ctrl"]))),
        ("body pitch (qpos[4])", lambda s: s["qpos"][4]),
        ("body roll (qpos[5])", lambda s: s["qpos"][5]),
        ("body yaw (qpos[6])", lambda s: s["qpos"][6]),
    ]

    all_deltas = {}
    for name, field_fn in key_fields:
        v1 = field_fn(state_1)
        v2 = field_fn(state_2)
        v3 = field_fn(state_3)
        v4 = field_fn(state_4)
        d12 = abs(v1 - v2)
        d13 = abs(v1 - v3)
        d23 = abs(v2 - v3)
        d14 = abs(v1 - v4)
        all_deltas[name] = (d12, d13, d23, d14)
        print(f"{name:<25s} | {d12:14.6e} | {d13:14.6e} | {d23:14.6e} | {d14:14.6e}")

    # ── Joint-level qfrc_constraint comparison ──────────────────────────
    print(f"\n{'Joint qfrc_constraint':<25s} | {'1-forward':>14s} | {'2-forward':>14s} | {'4-forward':>14s}")
    print("-" * 75)
    joint_names = ["l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
                   "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel"]
    for i, name in enumerate(joint_names):
        idx = 6 + i  # qfrc_constraint indexing
        v1 = state_1["qfrc_constraint"][idx] if idx < len(state_1["qfrc_constraint"]) else 0
        v2 = state_2["qfrc_constraint"][idx] if idx < len(state_2["qfrc_constraint"]) else 0
        v3 = state_3["qfrc_constraint"][idx] if idx < len(state_3["qfrc_constraint"]) else 0
        print(f"{name:<25s} | {v1:14.8f} | {v2:14.8f} | {v3:14.8f}")

    # ── Pitch comparison ────────────────────────────────────────────────
    print(f"\n{'Body orientation':<25s} | {'1-forward':>14s} | {'2-forward':>14s} | {'4-forward':>14s} | {'2fw+step':>14s}")
    print("-" * 90)
    import math
    for idx, label in [(4, "pitch_x (qpos[4])"), (5, "roll_y (qpos[5])"), (6, "yaw_z (qpos[6])")]:
        values = [state_1["qpos"][idx], state_2["qpos"][idx], state_3["qpos"][idx], state_4["qpos"][idx]]
        print(f"{label:<25s} | {values[0]:14.10f} | {values[1]:14.10f} | {values[2]:14.10f} | {values[3]:14.10f}")

    # ── Contact force comparison ────────────────────────────────────────
    print(f"\n{'Contact forces':<25s} | {'1-forward':>14s} | {'2-forward':>14s} | {'4-forward':>14s}")
    print("-" * 75)
    for j, (name, idx) in enumerate([
        ("l_wheel_contact_x", 0), ("l_wheel_contact_z", 2),
        ("r_wheel_contact_x", 3), ("r_wheel_contact_z", 5),
    ]):
        # Contact forces are in efc_force or computed from constraint solver
        # For now, check contact array from qfrc_constraint
        pass

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n--- Verdict ---")
    max_diff_12 = max(v[0] for v in all_deltas.values())
    max_diff_13 = max(v[1] for v in all_deltas.values())
    max_diff_23 = max(v[2] for v in all_deltas.values())

    if max_diff_12 < 1e-12:
        print("1-forward vs 2-forward: IDENTICAL (warm-start has no effect with same root_z)")
    elif max_diff_12 < 1e-6:
        print(f"1-forward vs 2-forward: NEGLIGIBLE DIFFERENCE (max={max_diff_12:.2e})")
    else:
        print(f"1-forward vs 2-forward: SIGNIFICANT DIFFERENCE (max={max_diff_12:.2e})")

    if max_diff_13 < 1e-12:
        print("1-forward vs 4-forward: IDENTICAL")
    elif max_diff_13 < 1e-6:
        print(f"1-forward vs 4-forward: NEGLIGIBLE DIFFERENCE (max={max_diff_13:.2e})")
    else:
        print(f"1-forward vs 4-forward: SIGNIFICANT DIFFERENCE (max={max_diff_13:.2e})")

    return all_deltas


def main():
    parser = argparse.ArgumentParser(description="K2 Warm-Start Parity Audit")
    parser.add_argument("--height", default="low_0p380", help="Height label")
    parser.add_argument("--heights", default=None,
                       help="Comma-separated heights to audit in batch")
    args = parser.parse_args()

    if args.heights:
        heights = [h.strip() for h in args.heights.split(",")]
    else:
        heights = [args.height]

    for h in heights:
        audit_init_parity(h)

    print(f"\n{'='*70}")
    print("CONCLUSION: If all deltas are near zero, warm-start is NOT the root cause.")
    print("If significant deltas found, implement source-equivalent 2-forward init.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
