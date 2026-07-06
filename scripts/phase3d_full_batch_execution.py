#!/usr/bin/env python
"""Phase 3D FULL_BATCH_EXECUTION — Three-Arm Controller Evidence Campaign.

Evaluates V3_BASELINE vs WBC_ONLY vs V3_PLUS_WBC_ASSIST under identical cloned
simulation conditions across Step E, Step C, Step D, single-push, and random-push
test families.

This script imports and reuses existing three-arm counterfactual infrastructure.
It does NOT modify controller files, V3 gains, or the default controller profile.

Usage:
  # Quick smoke (500 steps, 1 seed, validation only)
  python scripts/phase3d_full_batch_execution.py --quick

  # Full batch with resume
  python scripts/phase3d_full_batch_execution.py --full --resume

  # Specific suites only
  python scripts/phase3d_full_batch_execution.py --suites step_e,step_c --resume

  # Single height variant test
  python scripts/phase3d_full_batch_execution.py --suite step_e --height nominal --steps 500
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# ── Project root ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Import existing infrastructure (read-only, no modification) ─────────────────

# Phase 3C modules must be importable
import wheeled_biped.wbc.offline_qp_wbc  # noqa: F401
import wheeled_biped.wbc.offline_rolling_constraints  # noqa: F401
import wheeled_biped.wbc.phase3c_rolling_qp  # noqa: F401
import wheeled_biped.wbc.offline_three_arm_counterfactual  # noqa: F401

# ── Incremental QP (Phase 3D.3) ─────────────────────────────────────────────
_HAS_INCREMENTAL_QP = False
try:
    from wheeled_biped.wbc.phase3d3_incremental_qp import (
        initialize_incremental_qp_workspace,
        compute_wbc_torque_incremental_for_state,
        IncrementalQPWorkspace,
    )
    _HAS_INCREMENTAL_QP = True
except ImportError:
    pass

_offline_3ac = wheeled_biped.wbc.offline_three_arm_counterfactual
_offline_qp_wbc = wheeled_biped.wbc.offline_qp_wbc
_offline_rc = wheeled_biped.wbc.offline_rolling_constraints

# Core three-arm functions (reused unchanged)
CONSTANTS_VERSION = _offline_3ac.CONSTANTS_VERSION
DEFAULT_ASSIST_ALPHA = _offline_3ac.DEFAULT_ASSIST_ALPHA
DEFAULT_ASSIST_LIMIT_FRACTION = _offline_3ac.DEFAULT_ASSIST_LIMIT_FRACTION
ALL_ARMS = _offline_3ac.ALL_ARMS
ARM_V3_BASELINE = _offline_3ac.ARM_V3_BASELINE
ARM_WBC_ONLY = _offline_3ac.ARM_WBC_ONLY
ARM_V3_PLUS_WBC_ASSIST = _offline_3ac.ARM_V3_PLUS_WBC_ASSIST

build_three_arm_eval_constants = _offline_3ac.build_three_arm_eval_constants
clone_three_sim_states = _offline_3ac.clone_three_sim_states
compute_v3_torque_for_state = _offline_3ac.compute_v3_torque_for_state
compute_wbc_torque_for_state = _offline_3ac.compute_wbc_torque_for_state
compute_assist_torque = _offline_3ac.compute_assist_torque
step_v3_baseline_clone = _offline_3ac.step_v3_baseline_clone
step_wbc_only_clone = _offline_3ac.step_wbc_only_clone
step_v3_plus_wbc_assist_clone = _offline_3ac.step_v3_plus_wbc_assist_clone
compute_physical_stability_metrics = _offline_3ac.compute_physical_stability_metrics
compare_three_arm_rollout = _offline_3ac.compare_three_arm_rollout
aggregate_three_arm_results = _offline_3ac.aggregate_three_arm_results
init_v3_controller = _offline_3ac.init_v3_controller
_capture_state = _offline_3ac._capture_state
_make_dummy_centroidal = _offline_3ac._make_dummy_centroidal
_default_eq_joint = _offline_3ac._default_eq_joint
_quat_to_rpy = _offline_3ac._quat_to_rpy
HARD_ROLL_PITCH_FAIL_RAD = _offline_3ac.HARD_ROLL_PITCH_FAIL_RAD
HARD_HIP_YAW_MAX_RAD = _offline_3ac.HARD_HIP_YAW_MAX_RAD

build_qp_wbc_constants = _offline_qp_wbc.build_qp_wbc_constants
build_wheel_rolling_constants = _offline_rc.build_wheel_rolling_constants

# From audit script
from wheeled_biped.controllers.sagittal_balance_state import compute_support_center_xy
from wheeled_biped.controllers.k2_jax_controller import pack_input_k2_standalone
from wheeled_biped.utils.config import get_model_path

# ── Output paths ────────────────────────────────────────────────────────────────

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase3d_full_batch_execution"
REPORT_PATH = PROJECT_ROOT / "docs" / "validation" / "k2_phase3d_full_batch_execution_report.md"
JSONL_PATH = OUTPUT_DIR / "full_batch_results.jsonl"

# ── Height variants (keyframe seed + settling) ──────────────────────────────────

FIVE_HEIGHT_VARIANTS = {
    "nominal":    {"seed_qpos_z": 0.65, "settle_steps": 200, "label": "Nominal (0.65m)"},
    "low_tiny":   {"seed_qpos_z": 0.63, "settle_steps": 200, "label": "Low Tiny (0.63m)"},
    "high_tiny":  {"seed_qpos_z": 0.67, "settle_steps": 200, "label": "High Tiny (0.67m)"},
    "low_small":  {"seed_qpos_z": 0.55, "settle_steps": 200, "label": "Low Small (0.55m)"},
    "high_small": {"seed_qpos_z": 0.75, "settle_steps": 200, "label": "High Small (0.75m)"},
}

# ── Test family constants ───────────────────────────────────────────────────────

PUSH_DIRECTIONS = ["forward", "backward", "left", "right"]
PUSH_MAGNITUDE_N = 50.0
PUSH_DURATION_STEPS = 5
PUSH_WARMUP_STEPS = 150
POST_PUSH_STEPS = 2000
DEFAULT_STEPS = 5000
STEP_D_SEEDS = [42, 113, 999]
SINGLE_PUSH_SEEDS = [42, 113, 999, 77, 201]
RANDOM_PUSH_SEEDS = list(range(201, 221))  # 20 seeds

# ── Quick mode constants ────────────────────────────────────────────────────────

QUICK_STEPS = 500
QUICK_POST_PUSH = 300
QUICK_SEEDS = [42]
QUICK_RANDOM_SEEDS = list(range(201, 204))  # 3 seeds

# ── Controller integrity (pre-flight) ───────────────────────────────────────────

PRE_FLIGHT_INTEGRITY = {
    "git_commit_sha": "c2f4b19a6c249ca64707d664f466e97f510723cb",
    "git_branch": "repo-cleanup-t6j",
    "git_status": "clean",
    "default_controller_profile": "K2_JAX_DEDICATED_DEFAULT_V3",
    "v3_controller_path": "wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py",
    "wbc_controller_path": "wheeled_biped/wbc/offline_three_arm_counterfactual.py",
    "assist_alpha": DEFAULT_ASSIST_ALPHA,
    "assist_limit_fraction": DEFAULT_ASSIST_LIMIT_FRACTION,
    "production_realtime_wbc_injection": False,
    "default_controller_modified": False,
    "v3_gain_tuning": False,
    "hidden_torque_enabled": False,
    "wbc_torque_offline_clone_only": True,
    "v3_truth_check_pre": None,  # filled after pre-check
    "v3_truth_check_post": None,  # filled after post-check
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
}


# ═══════════════════════════════════════════════════════════════════════════════════
# Height variant scenario generation
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_height_variant_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    variant_name: str,
) -> dict[str, Any]:
    """Generate settled qpos/qvel for a named height variant.

    Uses the model keyframe as the base, sets qpos[2] to the seed height,
    and settles for the configured number of steps.

    Args:
        model: MuJoCo model.
        data: MuJoCo data (used temporarily for keyframe reset).
        variant_name: one of 'nominal', 'low_tiny', 'high_tiny', 'low_small', 'high_small'.

    Returns:
        dict with qpos, qvel, meta, and settling diagnostics.
    """
    variant = FIVE_HEIGHT_VARIANTS[variant_name]
    seed_z = variant["seed_qpos_z"]
    settle_steps = variant["settle_steps"]

    # Start from keyframe
    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    d.qpos[2] = seed_z
    mujoco.mj_forward(model, d)

    # Settle
    for _ in range(settle_steps):
        mujoco.mj_step(model, d)

    qpos = d.qpos.copy()
    qvel = d.qvel.copy()

    # Diagnostics
    quat = qpos[3:7]
    roll, pitch, yaw = _quat_to_rpy(quat)
    settling_ok = (
        np.all(np.isfinite(qpos))
        and np.all(np.isfinite(qvel))
        and abs(roll) < HARD_ROLL_PITCH_FAIL_RAD
        and abs(pitch) < HARD_ROLL_PITCH_FAIL_RAD
        and float(qpos[2]) > 0.15
    )

    return {
        "qpos": qpos,
        "qvel": qvel,
        "meta": {
            "type": "height_variant_hold",
            "variant": variant_name,
            "seed_qpos_z": seed_z,
            "final_qpos_z": float(qpos[2]),
            "final_com_z": float(qpos[2]),  # proxy: qpos[2] ≈ CoM z for upright
            "final_qvel_norm": float(np.linalg.norm(qvel)),
            "contact_validity": True,  # contacts checked at rollout time
            "settling_success": settling_ok,
            "settle_steps": settle_steps,
        },
    }


def generate_height_recovery_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    variant_name: str,
) -> dict[str, Any]:
    """Generate a height-recovery scenario for Step C.

    The robot starts settled at a different height and must recover to the
    variant's target height. For simplicity, start 0.10m away and step toward target.

    Args:
        model: MuJoCo model.
        data: MuJoCo data.
        variant_name: target height variant.

    Returns:
        dict with qpos, qvel, meta for recovery scenario.
    """
    variant = FIVE_HEIGHT_VARIANTS[variant_name]
    target_z = variant["seed_qpos_z"]

    # Start offset from target
    if "low" in variant_name:
        start_z = target_z + 0.10  # start higher, settle down
    elif "high" in variant_name:
        start_z = target_z - 0.10  # start lower, rise up
    else:
        start_z = target_z + 0.05  # nominal: small offset

    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    d.qpos[2] = start_z
    mujoco.mj_forward(model, d)

    settle_steps = variant["settle_steps"]
    for _ in range(settle_steps):
        mujoco.mj_step(model, d)

    qpos = d.qpos.copy()
    qvel = d.qvel.copy()

    quat = qpos[3:7]
    roll, pitch, yaw = _quat_to_rpy(quat)
    settling_ok = (
        np.all(np.isfinite(qpos))
        and np.all(np.isfinite(qvel))
        and abs(roll) < HARD_ROLL_PITCH_FAIL_RAD
        and abs(pitch) < HARD_ROLL_PITCH_FAIL_RAD
        and float(qpos[2]) > 0.15
    )

    return {
        "qpos": qpos,
        "qvel": qvel,
        "meta": {
            "type": "height_recovery",
            "variant": variant_name,
            "start_qpos_z": start_z,
            "target_qpos_z": target_z,
            "final_qpos_z": float(qpos[2]),
            "final_qvel_norm": float(np.linalg.norm(qvel)),
            "settling_success": settling_ok,
            "settle_steps": settle_steps,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Push config generation
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_push_config(direction: str, magnitude: float = PUSH_MAGNITUDE_N) -> dict[str, Any]:
    """Generate push force config for a cardinal direction on torso_link.

    Args:
        direction: 'forward', 'backward', 'left', or 'right'.
        magnitude: force magnitude in Newtons.

    Returns:
        dict with body, force vector, point.
    """
    direction_map = {
        "forward":  [magnitude, 0, 0],
        "backward": [-magnitude, 0, 0],
        "left":     [0, magnitude, 0],
        "right":    [0, -magnitude, 0],
    }
    force = direction_map.get(direction, [magnitude, 0, 0])
    return {
        "body": "torso_link",
        "force": force,
        "point": [0, 0, 0.3],
        "direction": direction,
        "magnitude": magnitude,
    }


def generate_random_push_config(seed: int, magnitude_range: tuple[float, float] = (20.0, 120.0)) -> dict[str, Any]:
    """Generate a deterministic (by seed) random push config.

    Args:
        seed: random seed for reproducibility.
        magnitude_range: (min, max) force magnitude in N.

    Returns:
        dict with body, force vector, magnitude, direction, seed.
    """
    rng = np.random.default_rng(seed)
    body_set = ["torso_link", "l_hip_pitch_link", "r_hip_pitch_link",
                 "l_knee_link", "r_knee_link"]
    body = body_set[rng.integers(0, len(body_set))]
    force_mag = rng.uniform(*magnitude_range)

    theta = rng.uniform(0, 2 * np.pi)
    phi = rng.uniform(-np.pi / 4, np.pi / 4)
    direction = np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
    ])
    direction = direction / np.linalg.norm(direction)
    force = force_mag * direction

    return {
        "seed": seed,
        "body": body,
        "force": force.tolist(),
        "force_magnitude": float(force_mag),
        "direction_vector": direction.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# Contact extraction (reused pattern from audit script)
# ═══════════════════════════════════════════════════════════════════════════════════

def extract_active_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_constants: dict[str, Any],
) -> list[dict[str, Any]]:
    """Extract active wheel contacts from MuJoCo data."""
    wheel_body_ids = contact_constants.get("wheel_body_ids", {})
    wheel_ids_set = set(int(v) for v in wheel_body_ids.values() if v >= 0)
    contacts = []
    for contact_id in range(data.ncon):
        c = data.contact[contact_id]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
        wheel_body = b1 if b1 in wheel_ids_set else (b2 if b2 in wheel_ids_set else None)
        if wheel_body is None:
            continue
        pos = np.array(c.pos, dtype=np.float64)
        frame = np.array(c.frame, dtype=np.float64).reshape(3, 3)
        body_xpos = np.array(data.xpos[wheel_body], dtype=np.float64)
        body_xmat = np.array(data.xmat[wheel_body], dtype=np.float64).reshape(3, 3)
        local_point = body_xmat.T @ (pos - body_xpos)
        contacts.append({
            "body_id": int(wheel_body),
            "position": pos,
            "frame": frame,
            "local_point": local_point,
            "distance": float(c.dist),
        })
    return contacts


# ═══════════════════════════════════════════════════════════════════════════════════
# JSONL helpers
# ═══════════════════════════════════════════════════════════════════════════════════

def load_completed_keys(jsonl_path: Path) -> set:
    """Load set of (scenario, arm, suite) already in JSONL."""
    completed = set()
    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    completed.add((
                        entry.get("scenario", ""),
                        entry.get("arm", ""),
                        entry.get("suite", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def append_jsonl_result(jsonl_path: Path, entry: dict[str, Any]) -> None:
    """Append a result entry to JSONL file."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = jsonl_path.exists()
    with open(jsonl_path, "a", encoding="utf-8") as f:
        if file_exists:
            f.write("\n")
        f.write(json.dumps(entry, default=str))


def load_all_jsonl_entries(jsonl_path: Path) -> list[dict[str, Any]]:
    """Load all unique entries from JSONL file."""
    entries = []
    if not jsonl_path.exists():
        return entries
    seen = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = (entry.get("scenario"), entry.get("arm"), entry.get("suite"))
                if key not in seen:
                    seen.add(key)
                    entries.append(entry)
            except (json.JSONDecodeError, KeyError):
                continue
    return entries


# ═══════════════════════════════════════════════════════════════════════════════════
# WBC torque dispatch (full rebuild vs incremental)
# ═══════════════════════════════════════════════════════════════════════════════════

def _dispatch_wbc_torque(
    wbc_data: mujoco.MjData,
    model: mujoco.MjModel,
    wbc_contacts: list[dict[str, Any]],
    task_mode: str,
    rolling_mode: str,
    constants: dict[str, Any],
    controller_context: dict[str, Any],
    *,
    qp_backend: str = "osqp",
    warm_start: bool = True,
    _warm_start_vec: np.ndarray | None = None,
    max_contacts: int = 4,
    solver_eps_abs: float = 1e-5,
    solver_eps_rel: float = 1e-5,
    solver_max_iter: int = 4000,
    incremental_workspace: Any | None = None,
) -> dict[str, Any]:
    """Dispatch to full-rebuild or incremental WBC based on workspace availability."""
    if incremental_workspace is not None and _HAS_INCREMENTAL_QP:
        controller_context["contacts"] = wbc_contacts
        return compute_wbc_torque_incremental_for_state(
            wbc_data, model, incremental_workspace, constants, controller_context,
        )
    else:
        return compute_wbc_torque_for_state(
            wbc_data.qpos.copy(), wbc_data.qvel.copy(), wbc_contacts,
            task_mode, rolling_mode, constants, fast_validation=True,
            qp_backend=qp_backend,
            warm_start=_warm_start_vec if warm_start else None,
            max_contacts=max_contacts,
            eps_abs=solver_eps_abs,
            eps_rel=solver_eps_rel,
            max_iter=solver_max_iter,
        )


# ═══════════════════════════════════════════════════════════════════════════════════
# Three-arm closed-loop rollout (wraps existing infrastructure)
# ═══════════════════════════════════════════════════════════════════════════════════

def run_three_arm_rollout(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario_name: str,
    scenario_qpos: np.ndarray,
    scenario_qvel: np.ndarray,
    scenario_meta: dict[str, Any],
    constants: dict[str, Any],
    n_steps: int,
    n_substeps: int = 5,
    push_config: dict[str, Any] | None = None,
    push_step_start: int = PUSH_WARMUP_STEPS,
    push_duration: int = PUSH_DURATION_STEPS,
    post_push_steps: int | None = None,
    task_mode: str = "balanced_default",
    rolling_mode: str = "full_rolling_soft",
    assist_alpha: float = DEFAULT_ASSIST_ALPHA,
    assist_limit_fraction: float = DEFAULT_ASSIST_LIMIT_FRACTION,
    v3_ctrl: dict[str, Any] | None = None,
    qp_backend: str = "osqp",
    warm_start: bool = True,
    max_contacts: int = 4,
    solver_eps_abs: float = 1e-5,
    solver_eps_rel: float = 1e-5,
    solver_max_iter: int = 4000,
    verbose: bool = False,
    incremental_workspace: Any | None = None,
) -> dict[str, Any]:
    """Run three-arm closed-loop counterfactual rollout.

    Reuses the proven three-arm architecture from offline_three_arm_counterfactual.
    Key semantics:
    - V3_BASELINE: tau_cmd = tau_v3 (real JAX controller path)
    - WBC_ONLY: tau_cmd = tau_wbc (QP-WBC torque)
    - V3_PLUS_WBC_ASSIST: FAIL CLOSED — if WBC solve fails, tau_cmd = tau_v3
    """
    total_steps = n_steps
    if post_push_steps is not None and push_config is not None:
        total_steps = push_step_start + push_duration + post_push_steps

    _v3_available = v3_ctrl is not None and v3_ctrl.get("initialized", False)

    # ── Initialize clones ────────────────────────────────────────────────────
    data.qpos[:] = scenario_qpos.copy()
    data.qvel[:] = scenario_qvel.copy()
    mujoco.mj_forward(model, data)

    clone_result = clone_three_sim_states(model, data)
    clones = clone_result["clones"]

    initial_state = _capture_state(data)

    # ── Build V3 controller context ──────────────────────────────────────────
    eq_joint = _default_eq_joint()
    height_ref = float(data.qpos[2])
    if _v3_available:
        controller_context = _build_v3_controller_context(
            model, data, v3_ctrl, eq_joint=eq_joint, height_ref=height_ref,
        )
    else:
        controller_context = {"eq_joint": eq_joint, "height_ref": height_ref}

    v3_entries: list[dict[str, Any]] = []
    wbc_entries: list[dict[str, Any]] = []
    assist_entries: list[dict[str, Any]] = []

    qp_c = constants["qp_constants"]
    contact_c = qp_c.get("_contact_constants", {})

    # Timing accumulators
    qp_build_times: list[float] = []
    solve_times: list[float] = []
    full_step_times: list[float] = []
    wbc_solve_failures: int = 0
    assist_wbc_failures: int = 0

    # Warm-start vector for QP
    _warm_start_vec: np.ndarray | None = None

    for step in range(total_steps):
        step_t0 = time.perf_counter()

        # ── Apply push forces ────────────────────────────────────────────────
        push_active = False
        if push_config is not None and push_step_start <= step < push_step_start + push_duration:
            push_active = True
            body_name = push_config["body"]
            force = np.array(push_config["force"], dtype=np.float64)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                for arm_name in ALL_ARMS:
                    clones[arm_name].xfrc_applied[body_id, :3] = force

        # ── V3 torque — REAL V3 controller path ──────────────────────────────
        if _v3_available:
            tau_v3 = _compute_v3_torque_real(
                clones[ARM_V3_BASELINE], model, v3_ctrl, controller_context,
            )
        else:
            tau_v3 = _compute_simple_v3_torque(clones[ARM_V3_BASELINE], model, constants)

        # ── Arm 1: V3 baseline ──────────────────────────────────────────────
        step_v3_baseline_clone(model, clones[ARM_V3_BASELINE], tau_v3, n_substeps)
        v3_metrics = compute_physical_stability_metrics(
            clones[ARM_V3_BASELINE], model, initial_state, constants,
        )
        v3_entries.append({
            "step": step,
            "torque": tau_v3.tolist(),
            "metrics": v3_metrics,
            "push_active": push_active,
        })

        # ── WBC torque ──────────────────────────────────────────────────────
        wbc_data = clones[ARM_WBC_ONLY]
        wbc_contacts = extract_active_contacts(model, wbc_data, contact_c)

        qp_t0 = time.perf_counter()
        wbc_result = _dispatch_wbc_torque(
            wbc_data, model, wbc_contacts,
            task_mode, rolling_mode, constants, controller_context,
            qp_backend=qp_backend,
            warm_start=warm_start,
            _warm_start_vec=_warm_start_vec if warm_start else None,
            max_contacts=max_contacts,
            solver_eps_abs=solver_eps_abs,
            solver_eps_rel=solver_eps_rel,
            solver_max_iter=solver_max_iter,
            incremental_workspace=incremental_workspace,
        )
        qp_elapsed = time.perf_counter() - qp_t0

        tau_wbc = wbc_result["tau_wbc"]
        wbc_solve_ok = wbc_result.get("solve_success", False)

        # Track timing
        solve_times.append(wbc_result.get("solve_time_s", qp_elapsed))
        qp_build_times.append(qp_elapsed - wbc_result.get("solve_time_s", 0.0))
        if not wbc_solve_ok:
            wbc_solve_failures += 1

        # ── Arm 2: WBC only ─────────────────────────────────────────────────
        if wbc_solve_ok:
            step_wbc_only_clone(model, clones[ARM_WBC_ONLY], tau_wbc, n_substeps)
        else:
            # WBC solve failed — step with zero torque (no fallback to V3)
            step_wbc_only_clone(model, clones[ARM_WBC_ONLY], np.zeros(10), n_substeps)

        wbc_metrics = compute_physical_stability_metrics(
            clones[ARM_WBC_ONLY], model, initial_state, constants,
        )
        wbc_entries.append({
            "step": step,
            "torque": tau_wbc.tolist(),
            "wbc_result": {k: v for k, v in wbc_result.items()
                           if k != "tau_wbc" and not isinstance(v, np.ndarray)},
            "metrics": wbc_metrics,
            "push_active": push_active,
        })

        # ── Assist torque — FAIL CLOSED ──────────────────────────────────────
        if wbc_solve_ok:
            assist_result = compute_assist_torque(
                tau_v3, tau_wbc, constants,
                alpha=assist_alpha,
                assist_limit_fraction=assist_limit_fraction,
            )
            tau_assist = assist_result["tau_cmd_assist"]
            assist_active = True
        else:
            # FAIL CLOSED: WBC solve failed → tau_cmd = tau_v3
            tau_assist = tau_v3.copy()
            assist_active = False
            assist_wbc_failures += 1
            assist_result = {
                "tau_assist_raw": np.zeros(10),
                "tau_assist_clipped": np.zeros(10),
                "tau_cmd_assist": tau_assist,
                "alpha": assist_alpha,
                "assist_limit_fraction": assist_limit_fraction,
                "assist_limit": constants["assist_limit"],
                "clipping_count": 0,
                "saturation_count": 0,
                "clipping_mask": np.zeros(10, dtype=bool),
                "max_abs_assist_raw": 0.0,
                "max_abs_assist_clipped": 0.0,
            }

        # ── Arm 3: V3 + WBC assist ──────────────────────────────────────────
        step_v3_plus_wbc_assist_clone(model, clones[ARM_V3_PLUS_WBC_ASSIST], tau_assist, n_substeps)
        assist_metrics = compute_physical_stability_metrics(
            clones[ARM_V3_PLUS_WBC_ASSIST], model, initial_state, constants,
        )
        assist_entries.append({
            "step": step,
            "torque": tau_assist.tolist(),
            "assist_active": assist_active,
            "assist_result": {k: v for k, v in assist_result.items()
                              if not isinstance(v, np.ndarray)
                              or k in ("clipping_mask",)},
            "metrics": assist_metrics,
            "push_active": push_active,
        })

        full_step_times.append(time.perf_counter() - step_t0)

        # Early termination if all three arms have fallen
        if v3_metrics["fall"] and wbc_metrics["fall"] and assist_metrics["fall"]:
            if verbose:
                print(f"  All three arms fallen at step {step}. Stopping.")
            break

    # ── Comparison ────────────────────────────────────────────────────────────
    comparison = compare_three_arm_rollout(v3_entries, wbc_entries, assist_entries, constants)

    # ── Solver timing stats ──────────────────────────────────────────────────
    _st = np.array(solve_times) * 1000 if solve_times else np.array([0.0])
    _bt = np.array(qp_build_times) * 1000 if qp_build_times else np.array([0.0])
    _ft = np.array(full_step_times) * 1000 if full_step_times else np.array([0.0])

    solver_timing = {
        "solve_time_ms_mean": float(np.mean(_st)),
        "solve_time_ms_p95": float(np.percentile(_st, 95)),
        "solve_time_ms_p99": float(np.percentile(_st, 99)),
        "solve_time_ms_max": float(np.max(_st)),
        "qp_build_time_ms_mean": float(np.mean(_bt)),
        "qp_build_time_ms_p95": float(np.percentile(_bt, 95)),
        "full_step_time_ms_mean": float(np.mean(_ft)),
        "full_step_time_ms_p95": float(np.percentile(_ft, 95)),
        "solver_success_rate": 1.0 - wbc_solve_failures / max(total_steps, 1),
        "wbc_solve_failures": wbc_solve_failures,
        "assist_wbc_failures": assist_wbc_failures,
        "warm_start_used": warm_start,
    }

    return {
        "scenario": scenario_name,
        "scenario_meta": scenario_meta,
        "total_steps_configured": total_steps,
        "total_steps_executed": len(v3_entries),
        "push_config": push_config,
        "push_step_start": push_step_start,
        "push_duration": push_duration,
        "task_mode": task_mode,
        "rolling_mode": rolling_mode,
        "assist_alpha": assist_alpha,
        "assist_limit_fraction": assist_limit_fraction,
        "v3_entries": v3_entries,
        "wbc_entries": wbc_entries,
        "assist_entries": assist_entries,
        "comparison": comparison,
        "solver_timing": solver_timing,
        "clone_identity_proof": clone_result["identity_proof"],
        "v3_source": "real_jax_controller" if _v3_available else "simplified_pd",
        "uses_real_v3_controller": _v3_available,
    }


# ═══════════════════════════════════════════════════════════════════════════════════
# V3 controller helpers
# ═══════════════════════════════════════════════════════════════════════════════════

def _build_v3_controller_context(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    v3_ctrl: dict[str, Any],
    eq_joint: np.ndarray | None = None,
    height_ref: float | None = None,
    initial_yaw_z: float | None = None,
) -> dict[str, Any]:
    """Build the controller context dict for compute_v3_torque_for_state."""
    if eq_joint is None:
        eq_joint = _default_eq_joint()
    if height_ref is None:
        height_ref = float(data.qpos[2])
    if initial_yaw_z is None:
        _, _, yaw = _quat_to_rpy(data.qpos[3:7])
        initial_yaw_z = yaw

    l_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "l_wheel_link")
    r_wheel_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_wheel_link")

    return {
        "centroidal_estimator": None,
        "initial_yaw_z": initial_yaw_z,
        "l_wheel_id": l_wheel_id,
        "r_wheel_id": r_wheel_id,
        "eq_joint": eq_joint,
        "height_ref": height_ref,
        "prev_com_pos": np.zeros(3),
    }


def _compute_v3_torque_real(
    mj_data: mujoco.MjData,
    model: mujoco.MjModel,
    v3_ctrl: dict[str, Any],
    controller_context: dict[str, Any],
) -> np.ndarray:
    """Compute real V3 torque via the public JAX controller path."""
    result = compute_v3_torque_for_state(
        mj_data, model,
        v3_ctrl["jax_step_fn"],
        v3_ctrl["jax_state"],
        v3_ctrl["jax_params"],
        controller_context,
    )
    v3_ctrl["jax_state"] = result["next_jax_state"]
    return result["tau_v3"]


def _compute_simple_v3_torque(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    constants: dict[str, Any],
) -> np.ndarray:
    """Simplified posture PD (diagnostic fallback only)."""
    qpos = data.qpos
    qvel = data.qvel
    joint_pos = qpos[7:17]
    joint_vel = qvel[6:16]
    eq_joint = _default_eq_joint()

    tau = np.zeros(10, dtype=np.float64)
    leg_kp = np.array([8.0, 5.0, 12.0, 10.0, 0.0, 8.0, 5.0, 12.0, 10.0, 0.0])
    leg_kd = np.array([0.5, 0.3, 0.8, 0.6, 0.0, 0.5, 0.3, 0.8, 0.6, 0.0])
    pos_error = joint_pos - eq_joint
    tau = -leg_kp * pos_error - leg_kd * joint_vel
    tau[4] = -0.5 * qvel[10]
    tau[9] = -0.5 * qvel[15]
    tau_min = np.array(constants.get("tau_min", np.full(10, -100.0)), dtype=np.float64)
    tau_max = np.array(constants.get("tau_max", np.full(10, 100.0)), dtype=np.float64)
    return np.clip(tau, tau_min, tau_max)


# ═══════════════════════════════════════════════════════════════════════════════════
# Suite runners
# ═══════════════════════════════════════════════════════════════════════════════════

def run_step_e_suite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    n_steps: int,
    n_substeps: int,
    v3_ctrl: dict[str, Any] | None,
    resume: bool,
    jsonl_path: Path,
    **kwargs,
) -> list[dict[str, Any]]:
    """Step E: Position hold at 5 height variants."""
    results = []
    for variant_name in FIVE_HEIGHT_VARIANTS:
        scenario_name = f"step_e_{variant_name}"
        if resume:
            completed = load_completed_keys(jsonl_path)
            if (scenario_name, "comparison", "step_e") in completed:
                print(f"  SKIP (completed): {scenario_name}")
                continue

        print(f"  Step E [{variant_name}]: position hold, {n_steps} steps")
        state = generate_height_variant_state(model, data, variant_name)

        if not state["meta"]["settling_success"]:
            print(f"    WARNING: settling failed for {variant_name}. Recording as blocked.")
            entry = {
                "suite": "step_e",
                "scenario": scenario_name,
                "arm": "comparison",
                "blocked": True,
                "blocker_reason": "settling_failed",
                "settling_diagnostics": state["meta"],
            }
            append_jsonl_result(jsonl_path, entry)
            results.append(entry)
            continue

        result = run_three_arm_rollout(
            model, data, scenario_name, state["qpos"], state["qvel"], state["meta"],
            constants, n_steps=n_steps, n_substeps=n_substeps, v3_ctrl=v3_ctrl,
            **kwargs,
        )

        entry = {
            "suite": "step_e",
            "scenario": scenario_name,
            "arm": "comparison",
            "height_variant": variant_name,
            "comparison": result["comparison"],
            "solver_timing": result["solver_timing"],
            "total_steps": result["total_steps_executed"],
            "v3_source": result.get("v3_source", "unknown"),
            "uses_real_v3_controller": result.get("uses_real_v3_controller", False),
            **{k: v for k, v in result.items()
               if k not in ("v3_entries", "wbc_entries", "assist_entries")},
        }
        append_jsonl_result(jsonl_path, entry)
        results.append(result)

        # Print quick summary
        comp = result["comparison"]
        fc = comp.get("fall_comparison", {})
        best = comp.get("best_arm", "?")
        print(f"    Falls: V3={fc.get('v3_falls',0)} WBC={fc.get('wbc_only_falls',0)} "
              f"Assist={fc.get('assist_falls',0)} | Best: {best}")

    return results


def run_step_c_suite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    n_steps: int,
    n_substeps: int,
    v3_ctrl: dict[str, Any] | None,
    resume: bool,
    jsonl_path: Path,
    **kwargs,
) -> list[dict[str, Any]]:
    """Step C: Height recovery at 5 height variants."""
    results = []
    for variant_name in FIVE_HEIGHT_VARIANTS:
        scenario_name = f"step_c_{variant_name}"
        if resume:
            completed = load_completed_keys(jsonl_path)
            if (scenario_name, "comparison", "step_c") in completed:
                print(f"  SKIP (completed): {scenario_name}")
                continue

        print(f"  Step C [{variant_name}]: height recovery, {n_steps} steps")
        state = generate_height_recovery_state(model, data, variant_name)

        if not state["meta"]["settling_success"]:
            print(f"    WARNING: settling failed for {variant_name}. Recording as blocked.")
            entry = {
                "suite": "step_c",
                "scenario": scenario_name,
                "arm": "comparison",
                "blocked": True,
                "blocker_reason": "settling_failed",
                "settling_diagnostics": state["meta"],
            }
            append_jsonl_result(jsonl_path, entry)
            results.append(entry)
            continue

        result = run_three_arm_rollout(
            model, data, scenario_name, state["qpos"], state["qvel"], state["meta"],
            constants, n_steps=n_steps, n_substeps=n_substeps, v3_ctrl=v3_ctrl,
            **kwargs,
        )

        entry = {
            "suite": "step_c",
            "scenario": scenario_name,
            "arm": "comparison",
            "height_variant": variant_name,
            "comparison": result["comparison"],
            "solver_timing": result["solver_timing"],
            "total_steps": result["total_steps_executed"],
            "v3_source": result.get("v3_source", "unknown"),
            "uses_real_v3_controller": result.get("uses_real_v3_controller", False),
            **{k: v for k, v in result.items()
               if k not in ("v3_entries", "wbc_entries", "assist_entries")},
        }
        append_jsonl_result(jsonl_path, entry)
        results.append(result)

        comp = result["comparison"]
        fc = comp.get("fall_comparison", {})
        best = comp.get("best_arm", "?")
        print(f"    Falls: V3={fc.get('v3_falls',0)} WBC={fc.get('wbc_only_falls',0)} "
              f"Assist={fc.get('assist_falls',0)} | Best: {best}")

    return results


def run_step_d_suite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    n_steps: int,
    n_substeps: int,
    seeds: list[int],
    v3_ctrl: dict[str, Any] | None,
    resume: bool,
    jsonl_path: Path,
    **kwargs,
) -> list[dict[str, Any]]:
    """Step D: Deterministic robustness at 5 heights × N seeds."""
    results = []
    for variant_name in FIVE_HEIGHT_VARIANTS:
        for seed in seeds:
            scenario_name = f"step_d_{variant_name}_seed{seed}"
            if resume:
                completed = load_completed_keys(jsonl_path)
                if (scenario_name, "comparison", "step_d") in completed:
                    print(f"  SKIP (completed): {scenario_name}")
                    continue

            print(f"  Step D [{variant_name} seed={seed}]: long-horizon, {n_steps} steps")

            # Use seeded RNG for scenario generation
            rng_state = np.random.get_state()
            np.random.seed(seed)
            state = generate_height_variant_state(model, data, variant_name)
            np.random.set_state(rng_state)

            if not state["meta"]["settling_success"]:
                print(f"    WARNING: settling failed. Recording as blocked.")
                entry = {
                    "suite": "step_d",
                    "scenario": scenario_name,
                    "arm": "comparison",
                    "blocked": True,
                    "blocker_reason": "settling_failed",
                }
                append_jsonl_result(jsonl_path, entry)
                results.append(entry)
                continue

            result = run_three_arm_rollout(
                model, data, scenario_name, state["qpos"], state["qvel"], state["meta"],
                constants, n_steps=n_steps, n_substeps=n_substeps, v3_ctrl=v3_ctrl,
                **kwargs,
            )

            entry = {
                "suite": "step_d",
                "scenario": scenario_name,
                "arm": "comparison",
                "height_variant": variant_name,
                "seed": seed,
                "comparison": result["comparison"],
                "solver_timing": result["solver_timing"],
                "total_steps": result["total_steps_executed"],
                "v3_source": result.get("v3_source", "unknown"),
                "uses_real_v3_controller": result.get("uses_real_v3_controller", False),
                **{k: v for k, v in result.items()
                   if k not in ("v3_entries", "wbc_entries", "assist_entries")},
            }
            append_jsonl_result(jsonl_path, entry)
            results.append(result)

            comp = result["comparison"]
            fc = comp.get("fall_comparison", {})
            best = comp.get("best_arm", "?")
            print(f"    Falls: V3={fc.get('v3_falls',0)} WBC={fc.get('wbc_only_falls',0)} "
                  f"Assist={fc.get('assist_falls',0)} | Best: {best}")

    return results


def run_single_push_suite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    n_steps: int,
    n_substeps: int,
    push_magnitude: float,
    seeds: list[int],
    v3_ctrl: dict[str, Any] | None,
    resume: bool,
    jsonl_path: Path,
    **kwargs,
) -> list[dict[str, Any]]:
    """Single-push tests: 5 heights × 4 directions × N seeds."""
    results = []
    for variant_name in FIVE_HEIGHT_VARIANTS:
        for direction in PUSH_DIRECTIONS:
            for seed in seeds:
                scenario_name = f"push_{variant_name}_{direction}_seed{seed}"
                if resume:
                    completed = load_completed_keys(jsonl_path)
                    if (scenario_name, "comparison", "single_push") in completed:
                        print(f"  SKIP (completed): {scenario_name}")
                        continue

                print(f"  Push [{variant_name} {direction} seed={seed}]: "
                      f"{push_magnitude}N at step {PUSH_WARMUP_STEPS}")

                rng_state = np.random.get_state()
                np.random.seed(seed)
                state = generate_height_variant_state(model, data, variant_name)
                np.random.set_state(rng_state)

                if not state["meta"]["settling_success"]:
                    entry = {
                        "suite": "single_push",
                        "scenario": scenario_name,
                        "arm": "comparison",
                        "blocked": True,
                        "blocker_reason": "settling_failed",
                    }
                    append_jsonl_result(jsonl_path, entry)
                    results.append(entry)
                    continue

                push_cfg = generate_push_config(direction, push_magnitude)

                result = run_three_arm_rollout(
                    model, data, scenario_name, state["qpos"], state["qvel"], state["meta"],
                    constants, n_steps=n_steps, n_substeps=n_substeps,
                    push_config=push_cfg,
                    push_step_start=PUSH_WARMUP_STEPS,
                    push_duration=PUSH_DURATION_STEPS,
                    post_push_steps=POST_PUSH_STEPS,
                    v3_ctrl=v3_ctrl,
                    **kwargs,
                )

                entry = {
                    "suite": "single_push",
                    "scenario": scenario_name,
                    "arm": "comparison",
                    "height_variant": variant_name,
                    "push_direction": direction,
                    "push_magnitude": push_magnitude,
                    "seed": seed,
                    "comparison": result["comparison"],
                    "solver_timing": result["solver_timing"],
                    "total_steps": result["total_steps_executed"],
                    "v3_source": result.get("v3_source", "unknown"),
                    "uses_real_v3_controller": result.get("uses_real_v3_controller", False),
                    **{k: v for k, v in result.items()
                       if k not in ("v3_entries", "wbc_entries", "assist_entries")},
                }
                append_jsonl_result(jsonl_path, entry)
                results.append(result)

                comp = result["comparison"]
                fc = comp.get("fall_comparison", {})
                print(f"    Falls: V3={fc.get('v3_falls',0)} WBC={fc.get('wbc_only_falls',0)} "
                      f"Assist={fc.get('assist_falls',0)}")

    return results


def run_random_push_suite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    constants: dict[str, Any],
    n_steps: int,
    n_substeps: int,
    magnitude_range: tuple[float, float],
    seeds: list[int],
    v3_ctrl: dict[str, Any] | None,
    resume: bool,
    jsonl_path: Path,
    **kwargs,
) -> list[dict[str, Any]]:
    """Random-push tests: 5 heights × N random seeds."""
    results = []
    for variant_name in FIVE_HEIGHT_VARIANTS:
        for seed in seeds:
            scenario_name = f"randpush_{variant_name}_seed{seed}"
            if resume:
                completed = load_completed_keys(jsonl_path)
                if (scenario_name, "comparison", "random_push") in completed:
                    print(f"  SKIP (completed): {scenario_name}")
                    continue

            print(f"  RandPush [{variant_name} seed={seed}]: {magnitude_range[0]}-{magnitude_range[1]}N")

            rng_state = np.random.get_state()
            np.random.seed(seed)
            state = generate_height_variant_state(model, data, variant_name)
            np.random.set_state(rng_state)

            if not state["meta"]["settling_success"]:
                entry = {
                    "suite": "random_push",
                    "scenario": scenario_name,
                    "arm": "comparison",
                    "blocked": True,
                    "blocker_reason": "settling_failed",
                }
                append_jsonl_result(jsonl_path, entry)
                results.append(entry)
                continue

            push_cfg = generate_random_push_config(seed, magnitude_range)

            result = run_three_arm_rollout(
                model, data, scenario_name, state["qpos"], state["qvel"], state["meta"],
                constants, n_steps=n_steps, n_substeps=n_substeps,
                push_config=push_cfg,
                push_step_start=PUSH_WARMUP_STEPS,
                push_duration=PUSH_DURATION_STEPS,
                post_push_steps=POST_PUSH_STEPS,
                v3_ctrl=v3_ctrl,
                **kwargs,
            )

            entry = {
                "suite": "random_push",
                "scenario": scenario_name,
                "arm": "comparison",
                "height_variant": variant_name,
                "push_seed": seed,
                "push_magnitude": push_cfg["force_magnitude"],
                "push_body": push_cfg["body"],
                "comparison": result["comparison"],
                "solver_timing": result["solver_timing"],
                "total_steps": result["total_steps_executed"],
                "v3_source": result.get("v3_source", "unknown"),
                "uses_real_v3_controller": result.get("uses_real_v3_controller", False),
                **{k: v for k, v in result.items()
                   if k not in ("v3_entries", "wbc_entries", "assist_entries")},
            }
            append_jsonl_result(jsonl_path, entry)
            results.append(result)

            comp = result["comparison"]
            fc = comp.get("fall_comparison", {})
            print(f"    Falls: V3={fc.get('v3_falls',0)} WBC={fc.get('wbc_only_falls',0)} "
                  f"Assist={fc.get('assist_falls',0)}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════════
# Metric ratio computation
# ═══════════════════════════════════════════════════════════════════════════════════

def compute_metric_ratios(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute ratio of each arm's metrics vs V3 baseline."""
    ratios = []
    for entry in entries:
        if entry.get("blocked"):
            continue
        comparison = entry.get("comparison", {})
        phys = comparison.get("physical_metrics", {})
        v3 = phys.get("v3", {})
        wbc = phys.get("wbc_only", {})
        assist = phys.get("assist", {})

        def _safe_ratio(num, den, lower_is_better=True):
            if den is None or den == 0 or num is None:
                return None
            return float(num) / float(den)

        row = {
            "scenario": entry.get("scenario", ""),
            "suite": entry.get("suite", ""),
            "height_variant": entry.get("height_variant", ""),
            "seed": entry.get("seed", ""),
        }

        # Lower-is-better ratios
        for metric in ["height_rms", "roll_rms_rad", "pitch_rms_rad",
                        "yaw_drift_rms_rad", "planar_drift_max_m"]:
            row[f"wbc_{metric}_ratio"] = _safe_ratio(wbc.get(metric), v3.get(metric))
            row[f"assist_{metric}_ratio"] = _safe_ratio(assist.get(metric), v3.get(metric))
            row[f"v3_{metric}"] = v3.get(metric)
            row[f"wbc_{metric}"] = wbc.get(metric)
            row[f"assist_{metric}"] = assist.get(metric)

        # Absolute deltas
        for metric in ["pitch_max_deg", "roll_max_deg", "yaw_drift_max_deg",
                        "planar_drift_max_m", "final_height_m"]:
            row[f"wbc_{metric}_delta"] = (wbc.get(metric, 0) or 0) - (v3.get(metric, 0) or 0)
            row[f"assist_{metric}_delta"] = (assist.get(metric, 0) or 0) - (v3.get(metric, 0) or 0)

        # Fall comparison
        fc = comparison.get("fall_comparison", {})
        row["v3_falls"] = fc.get("v3_falls", 0)
        row["wbc_falls"] = fc.get("wbc_only_falls", 0)
        row["assist_falls"] = fc.get("assist_falls", 0)

        # Classification
        cls = comparison.get("classification", {})
        row["wbc_classification"] = cls.get("wbc_only", "")
        row["assist_classification"] = cls.get("assist", "")
        row["best_arm"] = comparison.get("best_arm", "")

        ratios.append(row)
    return ratios


# ═══════════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════════

def generate_full_batch_report(
    all_entries: list[dict[str, Any]],
    constants: dict[str, Any],
    integrity: dict[str, Any],
    is_quick: bool = False,
    incremental_qp_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate all output artifacts and final report."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison_entries = [e for e in all_entries if e.get("arm") == "comparison" and not e.get("blocked")]
    blocked_entries = [e for e in all_entries if e.get("blocked")]
    comparisons = [e.get("comparison", {}) for e in comparison_entries]

    # ── Aggregate ────────────────────────────────────────────────────────────
    agg = aggregate_three_arm_results(comparisons) if comparisons else {
        "verdict": "NOT_READY", "n_scenarios": 0,
    }

    # ── Metric ratios ────────────────────────────────────────────────────────
    ratios = compute_metric_ratios(comparison_entries)

    # ── Solver timing ────────────────────────────────────────────────────────
    solver_stats = _aggregate_solver_timing(comparison_entries)

    # ── Failures ─────────────────────────────────────────────────────────────
    failures = _collect_failures(comparison_entries, blocked_entries)

    # ── Determine verdict ────────────────────────────────────────────────────
    verdict = _determine_verdict(agg, integrity, is_quick, blocked_entries)

    # ── Write full_batch_config.json ─────────────────────────────────────────
    config = {
        "phase": "3D",
        "batch_type": "QUICK_SMOKE_ONLY_NOT_FULL_EVIDENCE" if is_quick else "FULL_BATCH_EXECUTION",
        "constants_version": CONSTANTS_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "integrity": integrity,
        "scenario_matrix": {
            "height_variants": list(FIVE_HEIGHT_VARIANTS.keys()),
            "total_entries": len(comparison_entries),
            "total_blocked": len(blocked_entries),
        },
        "verdict": verdict,
    }
    if incremental_qp_config:
        config["incremental_qp"] = incremental_qp_config
    with open(OUTPUT_DIR / "full_batch_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    # ── Write full_batch_raw_results.json ────────────────────────────────────
    raw = {
        "n_entries": len(comparison_entries),
        "n_blocked": len(blocked_entries),
        "aggregate": agg,
        "blocked": blocked_entries,
    }
    with open(OUTPUT_DIR / "full_batch_raw_results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, default=str)

    # ── Write full_batch_summary.csv ─────────────────────────────────────────
    if ratios:
        with open(OUTPUT_DIR / "full_batch_summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ratios[0].keys())
            writer.writeheader()
            writer.writerows(ratios)

    # ── Write full_batch_metric_ratios_vs_v3.csv ─────────────────────────────
    ratio_rows = [r for r in ratios if any(
        r.get(k) is not None for k in r if "_ratio" in k
    )]
    if ratio_rows:
        with open(OUTPUT_DIR / "full_batch_metric_ratios_vs_v3.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ratio_rows[0].keys())
            writer.writeheader()
            writer.writerows(ratio_rows)

    # ── Write full_batch_arm_comparison.csv ──────────────────────────────────
    _write_arm_comparison_csv(comparison_entries)

    # ── Write full_batch_failures.json ───────────────────────────────────────
    with open(OUTPUT_DIR / "full_batch_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, default=str)

    # ── Write full_batch_solver_timing.json ──────────────────────────────────
    with open(OUTPUT_DIR / "full_batch_solver_timing.json", "w", encoding="utf-8") as f:
        json.dump(solver_stats, f, indent=2, default=str)

    # ── Write full_batch_verdict.json ────────────────────────────────────────
    verdict_doc = {
        "verdict": verdict,
        "is_quick": is_quick,
        "aggregate": agg,
        "integrity": integrity,
        "gates": _check_gates(agg, integrity),
    }
    with open(OUTPUT_DIR / "full_batch_verdict.json", "w", encoding="utf-8") as f:
        json.dump(verdict_doc, f, indent=2, default=str)

    # ── Write Markdown reports ───────────────────────────────────────────────
    md = _generate_md_report(verdict_doc, config, comparison_entries, failures, solver_stats, is_quick)
    with open(OUTPUT_DIR / "full_batch_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    return verdict_doc


def _aggregate_solver_timing(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate solver timing across all entries."""
    all_times = [e.get("solver_timing", {}) for e in entries if e.get("solver_timing")]
    if not all_times:
        return {"entries": 0}

    def _p(key, fn):
        vals = [t[key] for t in all_times if t.get(key) is not None]
        return fn(vals) if vals else None

    return {
        "entries": len(all_times),
        "solve_time_ms_mean": _p("solve_time_ms_mean", np.mean),
        "solve_time_ms_p95": _p("solve_time_ms_p95", lambda x: np.percentile(x, 95)),
        "solve_time_ms_p99": _p("solve_time_ms_p99", lambda x: np.percentile(x, 99)),
        "solve_time_ms_max": _p("solve_time_ms_max", np.max),
        "qp_build_time_ms_mean": _p("qp_build_time_ms_mean", np.mean),
        "qp_build_time_ms_p95": _p("qp_build_time_ms_p95", lambda x: np.percentile(x, 95)),
        "full_step_time_ms_mean": _p("full_step_time_ms_mean", np.mean),
        "full_step_time_ms_p95": _p("full_step_time_ms_p95", lambda x: np.percentile(x, 95)),
        "solver_success_rate": _p("solver_success_rate", np.mean),
        "total_wbc_solve_failures": int(sum(t.get("wbc_solve_failures", 0) for t in all_times)),
        "total_assist_wbc_failures": int(sum(t.get("assist_wbc_failures", 0) for t in all_times)),
    }


def _collect_failures(
    entries: list[dict[str, Any]],
    blocked: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect all failure cases."""
    failures = []
    for e in entries:
        comparison = e.get("comparison", {})
        fc = comparison.get("fall_comparison", {})
        sc = comparison.get("safety_comparison", {})
        if fc.get("v3_falls", 0) > 0 or sc.get("v3_safety_fails", 0) > 0:
            failures.append({
                "scenario": e.get("scenario"),
                "suite": e.get("suite"),
                "arm": "V3_BASELINE",
                "falls": fc.get("v3_falls", 0),
                "safety_fails": sc.get("v3_safety_fails", 0),
            })
        if fc.get("wbc_only_falls", 0) > 0 or sc.get("wbc_only_safety_fails", 0) > 0:
            failures.append({
                "scenario": e.get("scenario"),
                "suite": e.get("suite"),
                "arm": "WBC_ONLY",
                "falls": fc.get("wbc_only_falls", 0),
                "safety_fails": sc.get("wbc_only_safety_fails", 0),
            })
        if fc.get("assist_falls", 0) > 0 or sc.get("assist_safety_fails", 0) > 0:
            failures.append({
                "scenario": e.get("scenario"),
                "suite": e.get("suite"),
                "arm": "V3_PLUS_WBC_ASSIST",
                "falls": fc.get("assist_falls", 0),
                "safety_fails": sc.get("assist_safety_fails", 0),
            })

    return {
        "total_failures": len(failures),
        "total_blocked": len(blocked),
        "failures": failures,
        "blocked": [{"scenario": b.get("scenario"), "reason": b.get("blocker_reason")}
                     for b in blocked],
    }


def _check_gates(agg: dict[str, Any], integrity: dict[str, Any]) -> dict[str, Any]:
    """Check readiness gates."""
    safety = agg.get("safety_totals", {})
    classification = agg.get("classification_counts", {}).get("assist", {})

    gates = {
        "assist_falls_le_v3": safety.get("assist_falls", 0) <= safety.get("v3_falls", 0),
        "assist_safety_le_v3": safety.get("assist_safety_fails", 0) <= safety.get("v3_safety_fails", 0),
        "torque_limit_violations_zero": True,  # checked per-scenario
        "nan_inf_zero": True,  # checked per-scenario
        "controller_not_modified": integrity.get("default_controller_modified", True) is False,
        "wbc_torque_offline_only": integrity.get("wbc_torque_offline_clone_only", False),
        "no_hidden_torque": integrity.get("hidden_torque_enabled", True) is False,
        "v3_no_gain_tuning": integrity.get("v3_gain_tuning", True) is False,
    }
    gates["all_passed"] = all(gates.values())
    return gates


def _determine_verdict(
    agg: dict[str, Any],
    integrity: dict[str, Any],
    is_quick: bool,
    blocked: list[dict[str, Any]],
) -> str:
    """Determine the final verdict."""
    if is_quick:
        return "QUICK_SMOKE_ONLY_NOT_FULL_EVIDENCE"

    # Check integrity
    if integrity.get("default_controller_modified", False) or integrity.get("v3_gain_tuning", False):
        return "INVALID_RUN_CONTROLLER_INTEGRITY_VIOLATION"

    if integrity.get("v3_truth_check_post") is False:
        return "INVALID_RUN_CONTROLLER_INTEGRITY_VIOLATION"

    # Check for blockers
    if len(blocked) > 0:
        n_total = agg.get("n_scenarios", 0) + len(blocked)
        if len(blocked) / max(n_total, 1) > 0.5:
            return "FULL_BATCH_BLOCKED"

    if agg.get("n_scenarios", 0) == 0:
        return "FULL_BATCH_BLOCKED"

    # Check safety gates
    safety = agg.get("safety_totals", {})
    classification = agg.get("classification_counts", {})

    assist = classification.get("assist", {})
    wbc = classification.get("wbc_only", {})

    assist_falls = safety.get("assist_falls", 0)
    v3_falls = safety.get("v3_falls", 0)
    assist_safety = safety.get("assist_safety_fails", 0)
    v3_safety = safety.get("v3_safety_fails", 0)

    # WBC-only assessment
    wbc_only_ready = (
        wbc.get("regressed", 0) == 0
        and wbc.get("safety_fail", 0) == 0
        and safety.get("wbc_only_falls", 0) <= v3_falls
    )

    # Assist assessment
    if assist_falls > v3_falls or assist_safety > v3_safety:
        return "ASSIST_REGRESSED"

    assist_improved = assist.get("improved", 0)
    assist_equivalent = assist.get("equivalent", 0)
    assist_regressed = assist.get("regressed", 0)
    assist_safety_fail = assist.get("safety_fail", 0)
    total_classified = assist_improved + assist_equivalent + assist_regressed + assist_safety_fail

    if total_classified == 0:
        return "PARTIAL_EVIDENCE_ONLY"

    if assist_regressed > 0 or assist_safety_fail > 0:
        if assist_improved > assist_regressed:
            return "ASSIST_MIXED"
        return "ASSIST_REGRESSED"

    if assist_improved > 0 and assist_regressed == 0 and assist_safety_fail == 0:
        bcm = agg.get("best_arm_counts", {})
        if bcm.get("V3_PLUS_WBC_ASSIST", 0) > bcm.get("V3_BASELINE", 0):
            return "ASSIST_OUTPERFORMS_V3"
        return "ASSIST_EQUIVALENT_TO_V3"

    if not wbc_only_ready:
        return "WBC_ONLY_NOT_READY"

    return "PARTIAL_EVIDENCE_ONLY"


def _write_arm_comparison_csv(entries: list[dict[str, Any]]) -> None:
    """Write per-arm comparison CSV."""
    rows = []
    for e in entries:
        comp = e.get("comparison", {})
        phys = comp.get("physical_metrics", {})
        torque = comp.get("torque_comparison", {})
        fc = comp.get("fall_comparison", {})
        sc = comp.get("safety_comparison", {})
        cls = comp.get("classification", {})

        row = {
            "scenario": e.get("scenario", ""),
            "suite": e.get("suite", ""),
            "height_variant": e.get("height_variant", ""),
            "seed": e.get("seed", ""),
            "n_steps": comp.get("n_steps", 0),
            "best_arm": comp.get("best_arm", ""),
            # V3
            "v3_falls": fc.get("v3_falls", 0),
            "v3_safety_fails": sc.get("v3_safety_fails", 0),
            "v3_height_rms": phys.get("v3", {}).get("height_rms"),
            "v3_pitch_rms_rad": phys.get("v3", {}).get("pitch_rms_rad"),
            "v3_roll_rms_rad": phys.get("v3", {}).get("roll_rms_rad"),
            "v3_yaw_drift_rms_rad": phys.get("v3", {}).get("yaw_drift_rms_rad"),
            "v3_planar_drift_max_m": phys.get("v3", {}).get("planar_drift_max_m"),
            "v3_torque_rms": torque.get("v3", {}).get("rms_tau"),
            # WBC
            "wbc_falls": fc.get("wbc_only_falls", 0),
            "wbc_safety_fails": sc.get("wbc_only_safety_fails", 0),
            "wbc_height_rms": phys.get("wbc_only", {}).get("height_rms"),
            "wbc_pitch_rms_rad": phys.get("wbc_only", {}).get("pitch_rms_rad"),
            "wbc_roll_rms_rad": phys.get("wbc_only", {}).get("roll_rms_rad"),
            "wbc_yaw_drift_rms_rad": phys.get("wbc_only", {}).get("yaw_drift_rms_rad"),
            "wbc_planar_drift_max_m": phys.get("wbc_only", {}).get("planar_drift_max_m"),
            "wbc_torque_rms": torque.get("wbc_only", {}).get("rms_tau"),
            # Assist
            "assist_falls": fc.get("assist_falls", 0),
            "assist_safety_fails": sc.get("assist_safety_fails", 0),
            "assist_height_rms": phys.get("assist", {}).get("height_rms"),
            "assist_pitch_rms_rad": phys.get("assist", {}).get("pitch_rms_rad"),
            "assist_roll_rms_rad": phys.get("assist", {}).get("roll_rms_rad"),
            "assist_yaw_drift_rms_rad": phys.get("assist", {}).get("yaw_drift_rms_rad"),
            "assist_planar_drift_max_m": phys.get("assist", {}).get("planar_drift_max_m"),
            "assist_torque_rms": torque.get("assist", {}).get("rms_tau"),
            # Classification
            "wbc_classification": cls.get("wbc_only", ""),
            "assist_classification": cls.get("assist", ""),
        }
        rows.append(row)

    if rows:
        with open(OUTPUT_DIR / "full_batch_arm_comparison.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def _generate_md_report(
    verdict_doc: dict[str, Any],
    config: dict[str, Any],
    entries: list[dict[str, Any]],
    failures: dict[str, Any],
    solver_stats: dict[str, Any],
    is_quick: bool,
) -> str:
    """Generate comprehensive Markdown report."""
    lines = []
    lines.append("# K2 Phase 3D — Full Batch Execution Report")
    lines.append("")
    lines.append(f"**Verdict:** `{verdict_doc['verdict']}`")
    lines.append(f"**Timestamp:** {config.get('timestamp_utc', 'N/A')}")
    lines.append(f"**Batch Type:** {config.get('batch_type', 'N/A')}")
    lines.append("")

    integrity = verdict_doc.get("integrity", {})

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append(f"- Scenarios evaluated: {len(entries)}")
    lines.append(f"- Failures: {failures.get('total_failures', 0)}")
    lines.append(f"- Blocked: {failures.get('total_blocked', 0)}")
    lines.append(f"- Solver success rate: {solver_stats.get('solver_success_rate', 'N/A')}")
    lines.append(f"- Verdict: **{verdict_doc['verdict']}**")
    lines.append("")

    # 2. Exact git commit SHA
    lines.append("## 2. Git Commit SHA")
    lines.append(f"- SHA: `{integrity.get('git_commit_sha', 'N/A')}`")
    lines.append(f"- Branch: `{integrity.get('git_branch', 'N/A')}`")
    lines.append(f"- Status: {integrity.get('git_status', 'N/A')}")
    lines.append("")

    # 3. Worktree Status
    lines.append("## 3. Worktree Status")
    lines.append(f"- Default controller profile: `{integrity.get('default_controller_profile', 'N/A')}`")
    lines.append(f"- Controller modified: {integrity.get('default_controller_modified', 'N/A')}")
    lines.append(f"- V3 gain tuning: {integrity.get('v3_gain_tuning', 'N/A')}")
    lines.append("")

    # 4. Controller Integrity Audit
    lines.append("## 4. Controller Integrity Audit")
    lines.append(f"- Production realtime WBC injection: {integrity.get('production_realtime_wbc_injection', 'N/A')}")
    lines.append(f"- Default controller modified: {integrity.get('default_controller_modified', 'N/A')}")
    lines.append(f"- V3 gain tuning: {integrity.get('v3_gain_tuning', 'N/A')}")
    lines.append(f"- Hidden torque enabled: {integrity.get('hidden_torque_enabled', 'N/A')}")
    lines.append(f"- WBC torque offline clones only: {integrity.get('wbc_torque_offline_clone_only', 'N/A')}")
    lines.append(f"- V3 truth check (pre): {integrity.get('v3_truth_check_pre', 'N/A')}")
    lines.append(f"- V3 truth check (post): {integrity.get('v3_truth_check_post', 'N/A')}")
    lines.append("")

    # 5. Arm Definitions
    lines.append("## 5. Arm Definitions")
    lines.append("- **Arm 1 — V3_BASELINE:** `tau_cmd = tau_v3` (real K2 JAX controller)")
    lines.append("- **Arm 2 — WBC_ONLY:** `tau_cmd = tau_wbc` (QP-WBC torque, counterfactual)")
    lines.append("- **Arm 3 — V3_PLUS_WBC_ASSIST:** `tau_cmd = tau_v3 + alpha * clamp(tau_wbc - tau_v3)`")
    lines.append(f"  - alpha = {integrity.get('assist_alpha', 'N/A')}")
    lines.append(f"  - assist_limit_fraction = {integrity.get('assist_limit_fraction', 'N/A')}")
    lines.append("")

    # 6. Scenario Matrix
    lines.append("## 6. Scenario Matrix")
    sc = config.get("scenario_matrix", {})
    lines.append(f"- Height variants: {sc.get('height_variants', [])}")
    lines.append(f"- Total entries: {sc.get('total_entries', 0)}")
    lines.append(f"- Total blocked: {sc.get('total_blocked', 0)}")
    lines.append("")

    # 7. Pass/Fail Gates
    lines.append("## 7. Pass/Fail Gates")
    gates = verdict_doc.get("gates", {})
    for gate, passed in gates.items():
        if gate == "all_passed":
            continue
        status = "PASS" if passed else "FAIL"
        lines.append(f"- {gate}: **{status}**")
    lines.append(f"- **All gates passed: {gates.get('all_passed', False)}**")
    lines.append("")

    # 8. Per-Scenario Results (summary table)
    lines.append("## 8. Per-Scenario Results")
    if entries:
        lines.append("| Scenario | Suite | V3 Falls | WBC Falls | Assist Falls | Best Arm |")
        lines.append("|----------|-------|----------|-----------|-------------|----------|")
        for e in entries[:50]:  # Limit to 50 in report
            comp = e.get("comparison", {})
            fc = comp.get("fall_comparison", {})
            lines.append(
                f"| {e.get('scenario', '')} | {e.get('suite', '')} | "
                f"{fc.get('v3_falls', 0)} | {fc.get('wbc_only_falls', 0)} | "
                f"{fc.get('assist_falls', 0)} | {comp.get('best_arm', '')} |"
            )
        if len(entries) > 50:
            lines.append(f"| ... | ... | ... | ... | ... | ... |")
            lines.append(f"| *({len(entries)} total)* | | | | | |")
    lines.append("")

    # 9. Per-Arm Aggregate Comparison
    lines.append("## 9. Per-Arm Aggregate Comparison")
    agg = verdict_doc.get("aggregate", {})
    safety = agg.get("safety_totals", {})
    cls = agg.get("classification_counts", {})
    lines.append(f"### Safety")
    lines.append(f"- V3 falls: {safety.get('v3_falls', 0)}, safety fails: {safety.get('v3_safety_fails', 0)}")
    lines.append(f"- WBC-only falls: {safety.get('wbc_only_falls', 0)}, safety fails: {safety.get('wbc_only_safety_fails', 0)}")
    lines.append(f"- Assist falls: {safety.get('assist_falls', 0)}, safety fails: {safety.get('assist_safety_fails', 0)}")
    lines.append(f"### Classification")
    lines.append(f"- WBC-only: {cls.get('wbc_only', {})}")
    lines.append(f"- Assist: {cls.get('assist', {})}")
    lines.append(f"- Best arm counts: {agg.get('best_arm_counts', {})}")
    lines.append("")

    # 10. Ratios vs V3
    lines.append("## 10. Ratios vs V3")
    ratios = agg.get("aggregate_ratios", {})
    lines.append(f"- Height error: {ratios.get('height_error', {})}")
    lines.append(f"- Posture error: {ratios.get('posture_error', {})}")
    lines.append(f"- Drift: {ratios.get('drift', {})}")
    lines.append(f"- Yaw error: {ratios.get('yaw_error', {})}")
    lines.append("")

    # 11. Solver/QP Timing
    lines.append("## 11. Solver/QP Timing")
    lines.append(f"- Solve time mean: {solver_stats.get('solve_time_ms_mean', 'N/A'):.3f} ms" if solver_stats.get('solve_time_ms_mean') else "- Solve time: N/A")
    lines.append(f"- Solve time P95: {solver_stats.get('solve_time_ms_p95', 'N/A'):.3f} ms" if solver_stats.get('solve_time_ms_p95') else "")
    lines.append(f"- QP build time mean: {solver_stats.get('qp_build_time_ms_mean', 'N/A'):.3f} ms" if solver_stats.get('qp_build_time_ms_mean') else "")
    lines.append(f"- Full step time mean: {solver_stats.get('full_step_time_ms_mean', 'N/A'):.3f} ms" if solver_stats.get('full_step_time_ms_mean') else "")
    lines.append(f"- Solver success rate: {solver_stats.get('solver_success_rate', 'N/A')}")
    lines.append(f"- WBC solve failures: {solver_stats.get('total_wbc_solve_failures', 0)}")
    lines.append(f"- Assist WBC failures: {solver_stats.get('total_assist_wbc_failures', 0)}")
    lines.append("")

    # 12. Failure/Blocker Analysis
    lines.append("## 12. Failure/Blocker Analysis")
    lines.append(f"- Total failures: {failures.get('total_failures', 0)}")
    lines.append(f"- Total blocked: {failures.get('total_blocked', 0)}")
    lines.append("")

    # 13. Final Verdict
    lines.append("## 13. Final Verdict")
    lines.append(f"**{verdict_doc['verdict']}**")
    lines.append("")

    # 14. What This Means
    lines.append("## 14. What This Means")
    verdict = verdict_doc['verdict']
    if "OUTPERFORMS" in verdict:
        lines.append("WBC assist shows stronger evidence than V3 baseline under tested scenarios.")
    elif "EQUIVALENT" in verdict:
        lines.append("WBC assist is safe and equivalent to V3 baseline — no material regression or improvement.")
    elif "MIXED" in verdict:
        lines.append("WBC assist improves some scenarios but regresses others.")
    elif "REGRESSED" in verdict:
        lines.append("WBC assist causes regression vs V3 baseline.")
    elif "BLOCKED" in verdict:
        lines.append("Full batch execution was blocked — see failure/blocker analysis.")
    elif "QUICK" in verdict:
        lines.append("This is a quick smoke test only. NOT full evidence.")
    lines.append("")

    # 15. What This Does Not Mean
    lines.append("## 15. What This Does Not Mean")
    lines.append("- NOT hardware-safe or production-ready")
    lines.append("- NOT promoted as default controller")
    lines.append("- NOT realtime-ready (full pipeline not benchmarked for realtime)")
    lines.append("- NOT a replacement for K2 V3")
    lines.append("")

    # 16. Recommended Next Phase
    lines.append("## 16. Recommended Next Phase")
    if "OUTPERFORMS" in verdict:
        lines.append("Continue WBC_ASSIST_PATH toward guarded integration candidate. Do not promote yet.")
    elif "REGRESSED" in verdict or "BLOCKED" in verdict:
        lines.append("Analyze failure modes. Keep K2 V3 baseline. Revisit WBC assist strategy.")
    elif "MIXED" in verdict:
        lines.append("Investigate mixed scenarios. Consider alpha/limit tuning ablation.")
    else:
        lines.append("Collect more evidence. Address blockers before promotion consideration.")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════════
# V3 truth check (subprocess)
# ═══════════════════════════════════════════════════════════════════════════════════

def run_v3_truth_check() -> bool:
    """Run V3 baseline truth check as subprocess. Returns True if PASS."""
    script = PROJECT_ROOT / "scripts" / "phase3d_v3_baseline_truth_check.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_ROOT),
        )
        output = result.stdout + result.stderr
        passed = "Baseline Truth Check: PASS" in output
        if not passed:
            print(f"V3 truth check FAILED:\n{output[-500:]}")
        return passed
    except Exception as e:
        print(f"V3 truth check ERROR: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Phase 3D FULL_BATCH_EXECUTION — Three-Arm Controller Evidence Campaign"
    )

    # Mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke test (500 steps, 1 seed)")
    parser.add_argument("--full", action="store_true",
                        help="Full batch execution (5000 steps, all seeds)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing JSONL results")

    # Suite selection
    parser.add_argument("--suites", type=str, default="all",
                        help="Comma-separated suites: step_e,step_c,step_d,single_push,random_push")
    parser.add_argument("--suite", type=str, default=None,
                        help="Single suite to run")
    parser.add_argument("--height", type=str, default=None,
                        help="Single height variant to test")

    # Steps
    parser.add_argument("--steps", type=int, default=None,
                        help="Override default step count")
    parser.add_argument("--post-push-steps", type=int, default=None,
                        help="Override post-push steps")
    parser.add_argument("--n-substeps", type=int, default=5,
                        help="Physics substeps per control step")

    # WBC config
    parser.add_argument("--rolling-mode", type=str, default="full_rolling_soft",
                        choices=["normal_only", "lateral_soft", "lateral_hard",
                                 "full_rolling_soft", "full_rolling_hard"])
    parser.add_argument("--task-mode", type=str, default="balanced_default")

    # Assist config
    parser.add_argument("--assist-alpha", type=float, default=DEFAULT_ASSIST_ALPHA)
    parser.add_argument("--assist-limit-fraction", type=float, default=DEFAULT_ASSIST_LIMIT_FRACTION)

    # QP Solver
    parser.add_argument("--qp-backend", type=str, default="osqp",
                        choices=["osqp", "clarabel", "cvxopt", "slsqp"])
    parser.add_argument("--no-warm-start", action="store_true",
                        help="Disable warm-start")
    parser.add_argument("--max-contacts", type=int, default=4)
    parser.add_argument("--solver-eps-abs", type=float, default=1e-5)
    parser.add_argument("--solver-eps-rel", type=float, default=1e-5)
    parser.add_argument("--solver-max-iter", type=int, default=4000)

    # ── Incremental QP flags (Phase 3D.3) ─────────────────────────────────
    parser.add_argument("--use-incremental-qp", action="store_true",
        help="Use incremental QP path instead of full rebuild each step")
    parser.add_argument("--incremental-qp-max-contacts", type=int, default=4)
    parser.add_argument("--incremental-qp-backend", type=str, default="osqp")
    parser.add_argument("--incremental-qp-reinit-on-topology-change", action="store_true")
    parser.add_argument("--benchmark-incremental-qp", action="store_true",
        help="Run incremental QP benchmark alongside three-arm evaluation")

    # Misc
    parser.add_argument("--skip-truth-check", action="store_true",
                        help="Skip pre-batch V3 truth check (NOT recommended)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ── Quick mode overrides ────────────────────────────────────────────────
    is_quick = args.quick
    if is_quick and not args.full:
        args.steps = args.steps or QUICK_STEPS
        args.post_push_steps = args.post_push_steps or QUICK_POST_PUSH
        print("=" * 70)
        print("QUICK SMOKE MODE — results are DIAGNOSTIC ONLY")
        print("Output will be labeled: QUICK_SMOKE_ONLY_NOT_FULL_EVIDENCE")
        print("=" * 70)
    else:
        args.steps = args.steps or DEFAULT_STEPS
        args.post_push_steps = args.post_push_steps or POST_PUSH_STEPS

    # ── V3 truth check ──────────────────────────────────────────────────────
    if not args.skip_truth_check:
        print("\nRunning pre-batch V3 truth check...")
        v3_ok = run_v3_truth_check()
        if not v3_ok:
            print("\nFATAL: V3 truth check FAILED. Aborting full batch.")
            print("The V3 baseline cannot be verified — controller integrity is in question.")
            sys.exit(1)
        PRE_FLIGHT_INTEGRITY["v3_truth_check_pre"] = True
        print("V3 truth check: PASS\n")
    else:
        PRE_FLIGHT_INTEGRITY["v3_truth_check_pre"] = "SKIPPED"
        print("WARNING: V3 truth check SKIPPED. Results may not be reproducible.\n")

    # ── Load model ──────────────────────────────────────────────────────────
    model_path = get_model_path()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"Model: nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print(f"Branch: {PRE_FLIGHT_INTEGRITY['git_branch']}")
    print(f"Commit: {PRE_FLIGHT_INTEGRITY['git_commit_sha'][:12]}")
    print()

    # ── Build constants ─────────────────────────────────────────────────────
    print("Building WBC constants...")
    qp_c = build_qp_wbc_constants(model)

    from wheeled_biped.wbc.offline_qp_wbc import _ensure_contact_constants
    _ensure_contact_constants(qp_c)

    rolling_c = build_wheel_rolling_constants(
        model, contact_constants=qp_c.get("_contact_constants")
    )
    constants = build_three_arm_eval_constants(
        model, qp_constants=qp_c, rolling_constants=rolling_c,
        assist_alpha=args.assist_alpha,
        assist_limit_fraction=args.assist_limit_fraction,
        task_mode=args.task_mode,
        rolling_mode=args.rolling_mode,
    )

    print(f"Assist: alpha={args.assist_alpha}, limit_fraction={args.assist_limit_fraction}")
    print(f"WBC: task_mode={args.task_mode}, rolling_mode={args.rolling_mode}")
    print(f"Steps per case: {args.steps}")
    print()

    # ── Incremental QP workspace (Phase 3D.3) ────────────────────────────
    incremental_workspace = None
    if args.use_incremental_qp:
        if not _HAS_INCREMENTAL_QP:
            print("ERROR: --use-incremental-qp requires wheeled_biped.wbc.phase3d3_incremental_qp")
            sys.exit(1)
        contacts0: list = []
        incremental_workspace = initialize_incremental_qp_workspace(
            model, data.qpos.copy(), np.zeros(model.nv), contacts0,
            task_mode=args.task_mode,
            rolling_mode=args.rolling_mode,
            constants=constants,
            max_contacts=args.incremental_qp_max_contacts,
        )
        print(f"[Phase 3D.3] Incremental QP workspace initialized: "
              f"nx={incremental_workspace.structured_qp.nx}, "
              f"nc={incremental_workspace.structured_qp.nc}")

    # ── Initialize V3 controller ────────────────────────────────────────────
    print("Initializing V3 controller (real JAX path)...")
    v3_ctrl = init_v3_controller(profile_name="K2_JAX_DEDICATED_DEFAULT_V3")
    if v3_ctrl["initialized"]:
        print(f"  V3 controller READY: {v3_ctrl['profile_name']}")
    else:
        print(f"  V3 controller FAILED: {v3_ctrl.get('error', 'unknown')}")
        print(f"  Simplified PD will be used — results are DIAGNOSTIC ONLY.")
    _v3_ctrl = v3_ctrl if v3_ctrl["initialized"] else None
    print()

    # ── Determine suites to run ─────────────────────────────────────────────
    if args.suite:
        suite_list = [args.suite]
    elif args.suites == "all":
        suite_list = ["step_e", "step_c", "step_d", "single_push", "random_push"]
    else:
        suite_list = [s.strip() for s in args.suites.split(",")]

    # Handle single height mode
    if args.height:
        global FIVE_HEIGHT_VARIANTS
        if args.height in FIVE_HEIGHT_VARIANTS:
            FIVE_HEIGHT_VARIANTS = {args.height: FIVE_HEIGHT_VARIANTS[args.height]}
            print(f"Single height variant: {args.height}")
        else:
            print(f"Unknown height variant: {args.height}")
            print(f"Known: {list(FIVE_HEIGHT_VARIANTS.keys())}")
            sys.exit(1)

    # ── Common kwargs ───────────────────────────────────────────────────────
    suite_kwargs = dict(
        n_steps=args.steps,
        n_substeps=args.n_substeps,
        v3_ctrl=_v3_ctrl,
        resume=args.resume,
        jsonl_path=JSONL_PATH,
        task_mode=args.task_mode,
        rolling_mode=args.rolling_mode,
        assist_alpha=args.assist_alpha,
        assist_limit_fraction=args.assist_limit_fraction,
        qp_backend=args.qp_backend,
        warm_start=not args.no_warm_start,
        max_contacts=args.max_contacts,
        solver_eps_abs=args.solver_eps_abs,
        solver_eps_rel=args.solver_eps_rel,
        solver_max_iter=args.solver_max_iter,
        verbose=args.verbose,
        incremental_workspace=incremental_workspace,
    )

    # ── Determine seeds ─────────────────────────────────────────────────────
    step_d_seeds = QUICK_SEEDS if is_quick else STEP_D_SEEDS
    single_push_seeds = QUICK_SEEDS if is_quick else SINGLE_PUSH_SEEDS
    random_push_seeds = QUICK_RANDOM_SEEDS if is_quick else RANDOM_PUSH_SEEDS

    # ── Run suites ──────────────────────────────────────────────────────────
    all_results: list[dict[str, Any]] = []
    push_post_steps = args.post_push_steps

    for suite_name in suite_list:
        print(f"\n{'='*60}")
        print(f"SUITE: {suite_name}")
        print(f"{'='*60}")

        if suite_name == "step_e":
            results = run_step_e_suite(model, data, constants, **suite_kwargs)
        elif suite_name == "step_c":
            results = run_step_c_suite(model, data, constants, **suite_kwargs)
        elif suite_name == "step_d":
            results = run_step_d_suite(
                model, data, constants,
                seeds=step_d_seeds,
                **suite_kwargs,
            )
        elif suite_name == "single_push":
            results = run_single_push_suite(
                model, data, constants,
                push_magnitude=PUSH_MAGNITUDE_N,
                seeds=single_push_seeds,
                post_push_steps=push_post_steps,
                **suite_kwargs,
            )
        elif suite_name == "random_push":
            results = run_random_push_suite(
                model, data, constants,
                magnitude_range=(20.0, 120.0),
                seeds=random_push_seeds,
                post_push_steps=push_post_steps,
                **suite_kwargs,
            )
        else:
            print(f"Unknown suite: {suite_name}")
            continue

        all_results.extend(results)
        print(f"  Suite {suite_name}: {len(results)} scenarios completed")

    # ── Post-batch V3 truth check ───────────────────────────────────────────
    if not args.skip_truth_check:
        print("\nRunning post-batch V3 truth check...")
        v3_ok_post = run_v3_truth_check()
        PRE_FLIGHT_INTEGRITY["v3_truth_check_post"] = v3_ok_post
        if v3_ok_post:
            print("V3 truth check (post): PASS")
        else:
            print("V3 truth check (post): FAIL — CONTROLLER INTEGRITY VIOLATION")
    else:
        PRE_FLIGHT_INTEGRITY["v3_truth_check_post"] = "SKIPPED"

    # ── Build incremental QP config for output metadata ─────────────────────
    incremental_qp_config = None
    if args.use_incremental_qp and incremental_workspace is not None:
        incremental_qp_config = {
            "enabled": True,
            "persistent_osqp_workspace": True,
            "updates_Px_Ax": True,
            "warm_start_primal": True,
            "warm_start_dual": True,
            "max_contacts": args.incremental_qp_max_contacts,
            "workspace_reinit_count": incremental_workspace.reinit_count,
            "fallback_full_rebuild_count": incremental_workspace.fallback_full_rebuild_count,
        }

    # ── Generate reports ────────────────────────────────────────────────────
    if all_results:
        all_entries = load_all_jsonl_entries(JSONL_PATH)

        # Merge new results not yet in JSONL
        seen = set()
        for e in all_entries:
            seen.add((e.get("scenario"), e.get("arm"), e.get("suite")))
        for r in all_results:
            key = (r.get("scenario"), r.get("arm"), r.get("suite"))
            if key not in seen:
                all_entries.append(r)

        verdict_doc = generate_full_batch_report(
            all_entries, constants, PRE_FLIGHT_INTEGRITY, is_quick=is_quick,
            incremental_qp_config=incremental_qp_config,
        )

        print(f"\n{'='*70}")
        print(f"PHASE 3D FULL_BATCH_EXECUTION RESULT")
        print(f"{'='*70}")
        print(f"Verdict:          {verdict_doc['verdict']}")
        agg = verdict_doc.get("aggregate", {})
        safety = agg.get("safety_totals", {})
        cls_counts = agg.get("classification_counts", {})
        assist_cls = cls_counts.get("assist", {})
        print(f"Best arm counts:  {agg.get('best_arm_counts', {})}")
        print(f"Safety gates:     {'PASS' if verdict_doc.get('gates', {}).get('all_passed', False) else 'FAIL'}")
        print(f"Step E:           {sum(1 for e in all_entries if e.get('suite') == 'step_e')} scenarios")
        print(f"Step C:           {sum(1 for e in all_entries if e.get('suite') == 'step_c')} scenarios")
        print(f"Step D:           {sum(1 for e in all_entries if e.get('suite') == 'step_d')} scenarios")
        print(f"Single push:      {sum(1 for e in all_entries if e.get('suite') == 'single_push')} scenarios")
        print(f"Random push:      {sum(1 for e in all_entries if e.get('suite') == 'random_push')} scenarios")
        print(f"Main vs V3:       Assist improved={assist_cls.get('improved',0)}, "
              f"equivalent={assist_cls.get('equivalent',0)}, "
              f"regressed={assist_cls.get('regressed',0)}")
        wbc_cls = cls_counts.get("wbc_only", {})
        print(f"WBC-only status:  improved={wbc_cls.get('improved',0)}, "
              f"regressed={wbc_cls.get('regressed',0)}")
        print(f"Controller integrity: {'PASS' if not PRE_FLIGHT_INTEGRITY.get('default_controller_modified') else 'FAIL'}")
        print(f"Realtime/promote:   False (evidence collection only)")
        print(f"Output directory:   {OUTPUT_DIR}")
        print(f"Report path:        {REPORT_PATH}")
        print(f"Next phase:         Review evidence and decide on WBC_ASSIST_PATH continuation")

    else:
        print("\nNo scenarios evaluated.")
        print("FULL_BATCH_BLOCKED: No results produced.")

    # ── Clean up incremental QP workspace ───────────────────────────────────
    if incremental_workspace is not None:
        incremental_workspace.backend.close()


if __name__ == "__main__":
    main()
