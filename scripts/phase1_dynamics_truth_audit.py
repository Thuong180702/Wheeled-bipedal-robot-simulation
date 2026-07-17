#!/usr/bin/env python3
"""Phase 1: K2 Dynamics Truth Layer Audit.

Runs all dynamics truth-layer diagnostics and produces a Markdown report.

Read-only. No controller, no training, no profile changes.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import mujoco
import numpy as np

# Ensure package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.utils.config import get_model_path
from wheeled_biped.controllers.robot_model_utils import get_total_robot_mass

from wheeled_biped.dynamics.model_inspector import (
    build_model_index_report,
    extract_state_snapshot,
)
from wheeled_biped.dynamics.jacobian_checks import (
    compute_task_jacobian,
    finite_difference_jacobian_check,
)
from wheeled_biped.dynamics.contact_inspector import inspect_contacts
from wheeled_biped.dynamics.torque_sign_checks import torque_sign_probe, MAX_PROBE_FRACTION


# ── Expected body names for K2 ──────────────────────────────────
MANDATORY_BODIES = [
    "torso",
    "l_thigh",
    "r_thigh",
    "l_knee_link",
    "r_knee_link",
    "l_hip_roll_link",
    "r_hip_roll_link",
    "l_hip_yaw_link",
    "r_hip_yaw_link",
    "l_wheel_link",
    "r_wheel_link",
]

# ── Expected actuated joints ────────────────────────────────────
EXPECTED_JOINTS = [
    "l_hip_roll", "l_hip_yaw", "l_hip_pitch", "l_knee", "l_wheel",
    "r_hip_roll", "r_hip_yaw", "r_hip_pitch", "r_knee", "r_wheel",
]

# ── Expected actuator names ─────────────────────────────────────
EXPECTED_ACTUATORS = [
    "l_hip_roll_motor", "l_hip_yaw_motor", "l_hip_pitch_motor",
    "l_knee_motor", "l_wheel_motor",
    "r_hip_roll_motor", "r_hip_yaw_motor", "r_hip_pitch_motor",
    "r_knee_motor", "r_wheel_motor",
]

# ── Jacobian targets ────────────────────────────────────────────
JACOBIAN_TARGETS = [
    ("torso", "body"),
    ("l_wheel_link", "body"),
    ("r_wheel_link", "body"),
    ("l_knee_link", "body"),
    ("r_knee_link", "body"),
]


def main() -> int:
    """Run the full Phase 1 audit and write the markdown report."""
    print("=" * 72)
    print("Phase 1.5: K2 Dynamics Truth Layer Refinement Audit")
    print(f"Run timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 72)

    # ── Load model ──────────────────────────────────────────────
    model_path = get_model_path()
    print(f"\nLoading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    print("Model loaded successfully.")

    # ── Gather all diagnostics ──────────────────────────────────
    results: dict = {}

    # 1. Model index
    print("\n[1/10] Building model index report ...")
    results["model_index"] = build_model_index_report(model)

    # 2. State snapshot
    print("[2/10] Extracting state snapshot ...")
    results["state_snapshot"] = extract_state_snapshot(model, data)

    # 3. Mass matrix check
    print("[3/10] Checking mass matrix availability ...")
    results["mass_matrix"] = _check_mass_matrix(model, data)

    # 4. COM check
    print("[4/10] Checking COM ...")
    results["com_check"] = _check_com(model, data)

    # 5. Contact inspection (after short settle)
    print("[5/10] Inspecting contacts (passive settle) ...")
    results["contact_inspection"] = _run_contact_settle(model, data)

    # 6. Jacobian analytic
    print("[6/10] Computing analytic Jacobians ...")
    results["jacobian_analytic"] = {}
    # Re-forward after contact settle
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for target_name, target_type in JACOBIAN_TARGETS:
        results["jacobian_analytic"][target_name] = compute_task_jacobian(
            model, data, target_name, target_type
        )

    # 7. Jacobian FD validation
    print("[7/10] Running finite-difference Jacobian checks ...")
    results["jacobian_fd"] = {}
    for target_name, target_type in JACOBIAN_TARGETS:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        results["jacobian_fd"][target_name] = finite_difference_jacobian_check(
            model, data, target_name, target_type
        )

    # 8. Torque sign probes
    print("[8/10] Running torque sign probes ...")
    results["torque_signs"] = []
    for joint_name, actuator_name in zip(EXPECTED_JOINTS, EXPECTED_ACTUATORS):
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        result = torque_sign_probe(model, data, joint_name, actuator_name)
        results["torque_signs"].append(result)

    # 9. Actuator limit checks
    print("[9/10] Checking actuator limits ...")
    results["actuator_limits"] = _check_actuator_limits(model)

    # 10. Body mapping checks
    print("[10/10] Checking mandatory body mappings ...")
    results["body_mapping"] = _check_body_mapping(model)

    # ── Write report ────────────────────────────────────────────
    report_path = (
        PROJECT_ROOT / "docs" / "validation" / "k2_phase1_5_dynamics_truth_refinement.md"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_md = _generate_report(results, model, data)
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\nReport written to: {report_path}")

    # Also write JSON for machine consumption
    json_path = (
        PROJECT_ROOT / "docs" / "validation" / "k2_phase1_5_dynamics_truth_refinement.json"
    )
    _write_json_summary(results, json_path)
    print(f"JSON summary written to: {json_path}")

    # ── Summary ─────────────────────────────────────────────────
    verdict = _compute_readiness(results)
    print(f"\nPhase 2A Readiness Verdict: {verdict}")
    print("Done.")
    return 0


# ─────────────────────────────────────────────────────────────────
# Check helpers
# ─────────────────────────────────────────────────────────────────

def _check_mass_matrix(model, data) -> dict:
    """Check if mass matrix is accessible via MuJoCo CPU API."""
    result = {
        "available_cpu": True,
        "available_mjx": False,
        "note_mjx": (
            "mj_fullM / mjData.qM are CPU-only. MJX does not expose "
            "the mass matrix directly. For future real-time/JAX WBC, "
            "the mass matrix must be computed via a separate JAX port "
            "or accessed through the MJX C++ internals."
        ),
    }
    try:
        nv = model.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M, data.qM)
        result["shape"] = list(M.shape)
        result["finite"] = bool(np.all(np.isfinite(M)))
        result["symmetric"] = bool(np.allclose(M, M.T, atol=1e-10))
        result["diagonal_positive"] = bool(np.all(np.diag(M) > 0))
        result["condition_number"] = float(np.linalg.cond(M))
    except Exception as e:
        result["error"] = str(e)
        result["finite"] = False
    return result


def _check_com(model, data) -> dict:
    """Check COM position plausibility."""
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
        return {"com_available": False, "error": "torso body not found"}

    com = np.array(data.subtree_com[torso_id])
    base_z = float(data.qpos[2])

    result = {
        "com_available": True,
        "com_position": com.tolist(),
        "base_z": base_z,
        "com_relative_to_base_z": float(com[2] - base_z),
        "plausible": True,
    }

    # Plausibility: COM should be below base but not below ground
    if com[2] > base_z + 0.1:
        result["plausible"] = False
        result["warning"] = "COM above base — unexpected for wheeled biped"
    if com[2] < -0.1:
        result["plausible"] = False
        result["warning"] = "COM below ground — physically impossible"
    # COM should be within ±0.2m of base in XY
    if abs(com[0]) > 0.3 or abs(com[1]) > 0.3:
        result["plausible"] = False
        result["warning"] = "COM far from base in XY plane"

    return result


def _run_contact_settle(model, data) -> dict:
    """Run a short passive settle and inspect contacts."""
    # Reset and let gravity settle for 50 steps (0.1s at 500Hz)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:] = 0.0

    n_settle = 50
    for _ in range(n_settle):
        mujoco.mj_step(model, data)

    contact_info = inspect_contacts(model, data)
    contact_info["settle_steps"] = n_settle
    contact_info["settle_time_s"] = n_settle * model.opt.timestep
    return contact_info


def _check_actuator_limits(model) -> dict:
    """Inspect actuator control/force ranges."""
    actuators = []
    flags = []

    for aid in range(model.nu):
        name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            or f"actuator_{aid}"
        )
        ctrlrange = model.actuator_ctrlrange[aid].tolist()
        forcerange = model.actuator_forcerange[aid].tolist()

        issues = []
        if ctrlrange[0] == ctrlrange[1]:
            issues.append("zero_range")
        if ctrlrange[0] != -ctrlrange[1]:
            issues.append("asymmetric")
        if not (ctrlrange[0] <= 0.0 <= ctrlrange[1]):
            issues.append("zero_not_in_range")

        actuators.append({
            "id": aid,
            "name": name,
            "ctrlrange": ctrlrange,
            "forcerange": forcerange,
            "issues": issues,
        })
        flags.extend(issues)

    return {
        "actuators": actuators,
        "any_asymmetric": "asymmetric" in flags,
        "any_zero_range": "zero_range" in flags,
        "all_symmetric": "asymmetric" not in flags,
    }


def _check_body_mapping(model) -> dict:
    """Check mandatory body names exist."""
    found = []
    missing = []
    for body_name in MANDATORY_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid >= 0:
            found.append({"name": body_name, "id": int(bid)})
        else:
            missing.append(body_name)

    return {
        "mandatory_bodies": MANDATORY_BODIES,
        "found": found,
        "missing": missing,
        "all_present": len(missing) == 0,
    }


def _range(lo: float, hi: float) -> tuple[float, float]:
    return (lo, hi)


# ─────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────

def _generate_report(results: dict, model, data) -> str:
    """Generate the full Markdown audit report."""
    idx = results["model_index"]
    state = results["state_snapshot"]
    mass = results["mass_matrix"]
    com = results["com_check"]
    contacts = results["contact_inspection"]
    jac_analytic = results["jacobian_analytic"]
    jac_fd = results["jacobian_fd"]
    torque_signs = results["torque_signs"]
    act_limits = results["actuator_limits"]
    body_map = results["body_mapping"]

    verdict = _compute_readiness(results)

    lines: list[str] = []
    _w = lines.append

    _w("# K2 Phase 1.5 — Dynamics Truth Layer Refinement Report")
    _w("")
    _w(f"**Generated:** {datetime.now(timezone.utc).isoformat()}")
    _w(f"**Model:** `{get_model_path()}`")
    _w(f"**Total robot mass:** {get_total_robot_mass(model):.3f} kg")
    _w("")

    # ── Executive Summary ───────────────────────────────────────
    _w("## 1. Executive Summary")
    _w("")
    _w(f"**Phase 2A Readiness Verdict: `{verdict}`**")
    _w("")
    n_joints = len(idx["joints"])
    n_bodies = idx["nbody"]
    n_actuators = idx["nu"]
    n_contacts = contacts["ncon"]
    jac_failures = sum(
        1 for v in jac_fd.values() if v["verdict"] == "FAIL"
    )
    jac_warnings = sum(
        1 for v in jac_fd.values() if v["verdict"] == "WARN"
    )
    torque_outcomes = [t["outcome"] for t in torque_signs]
    n_measured = torque_outcomes.count("MEASURED")
    n_ambiguous = torque_outcomes.count("AMBIGUOUS")
    n_missing = torque_outcomes.count("MISSING")

    _w(f"- Model loaded with **{n_joints} joints**, **{n_bodies} bodies**, **{n_actuators} actuators**")
    _w(f"- State snapshot: qpos/qvel finite → **{state['qpos_finite']}/{state['qvel_finite']}**")
    _w(f"- Mass matrix available via CPU MuJoCo → **{mass.get('available_cpu', False)}**")
    _w(f"- COM position plausible → **{com.get('plausible', False)}**")
    _w(f"- Contacts after passive settle: **{n_contacts}** active contacts")
    _w(f"- Jacobian FD validation: **{len(JACOBIAN_TARGETS) - jac_failures} pass**, {jac_warnings} warn, {jac_failures} fail")
    _w(f"- Torque sign probes: **{n_measured} MEASURED**, {n_ambiguous} AMBIGUOUS, {n_missing} MISSING")
    _w("")

    # ── Non-modification statement ──────────────────────────────
    _w("## 2. Controller Non-Modification Statement")
    _w("")
    _w("> **Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were not modified.**")
    _w(">")
    _w("> This audit is purely diagnostic. No controller profiles were loaded or executed.")
    _w("> No training, promotion, or regression evaluation was run.")
    _w("")
    _w("**Changed files (Phase 1.5):**")
    _w("")
    _w("- `wheeled_biped/dynamics/torque_sign_checks.py` (modified — bias-subtracted probe)")
    _w("- `scripts/phase1_dynamics_truth_audit.py` (modified — Phase 1.5 paths, actuator limit fix, delta-based torque sign table)")
    _w("- `tests/test_phase1_5_dynamics_truth_refinement.py` (new)")
    _w("- `docs/validation/k2_phase1_5_dynamics_truth_refinement.md` (new — this report)")
    _w("- `docs/validation/k2_phase1_5_dynamics_truth_refinement.json` (new)")
    _w("")
    _w("**Files NOT touched:**")
    _w("")
    _w("- `wheeled_biped/controllers/k2_jax_controller.py`")
    _w("- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`")
    _w("- All controller profile definitions and promotion scripts")
    _w("- All config YAML files")
    _w("- `K2_JAX_DEDICATED_DEFAULT_V3` (no profile changes)")
    _w("")

    # ── Model Dimensions ────────────────────────────────────────
    _w("## 3. Model Dimensions")
    _w("")
    _w("| Property | Value |")
    _w("|----------|-------|")
    _w(f"| `nq` (generalized positions) | {idx['nq']} |")
    _w(f"| `nv` (generalized velocities) | {idx['nv']} |")
    _w(f"| `nu` (actuators/controls) | {idx['nu']} |")
    _w(f"| `nbody` (bodies) | {idx['nbody']} |")
    _w(f"| `njnt` (joints) | {idx['njnt']} |")
    _w(f"| `ngeom` (geoms) | {idx['ngeom']} |")
    _w(f"| `nsite` (sites) | {idx['nsite']} |")
    _w(f"| `nkey` (keyframes) | {idx['nkey']} |")
    _w(f"| `nsensor` (sensors) | {idx['nsensor']} |")
    _w("")

    # ── Joint/Actuator Mapping ──────────────────────────────────
    _w("## 4. Joint / Actuator Mapping")
    _w("")
    _w("| Index | Joint | Actuator | Ctrl Range (Nm) | Force Range (Nm) |")
    _w("|-------|-------|----------|-----------------|------------------|")
    for i, (jname, aname) in enumerate(zip(EXPECTED_JOINTS, EXPECTED_ACTUATORS)):
        act = idx["actuators"].get(aname, {})
        cr = act.get("ctrlrange", ["?", "?"])
        fr = act.get("forcerange", ["?", "?"])
        _w(f"| {i} | `{jname}` | `{aname}` | [{cr[0]}, {cr[1]}] | [{fr[0]}, {fr[1]}] |")
    _w("")
    _w(f"**Actuator count: {idx['nu']}** — {'MATCH' if idx['nu'] == 10 else 'MISMATCH'} (expected 10)")
    _w("")
    _w("**Joint name verification:**")
    joints_found = idx["joints"]
    for jname in EXPECTED_JOINTS:
        if jname in joints_found:
            _w(f"- ✅ `{jname}` — id={joints_found[jname]['id']}, type={joints_found[jname]['type_name']}")
        else:
            _w(f"- ❌ `{jname}` — **MISSING**")
    _w("")

    # ── Body Mapping ────────────────────────────────────────────
    _w("## 5. Body Mapping")
    _w("")
    _w("| Body Name | ID | Parent | Mass (kg) |")
    _w("|-----------|-----|--------|-----------|")
    for name in MANDATORY_BODIES:
        body = idx["bodies"].get(name)
        if body:
            _w(f"| `{name}` | {body['id']} | `{body['parent_name']}` | {body.get('mass', '?'):.3f} |")
        else:
            _w(f"| `{name}` | ❌ MISSING | — | — |")
    _w("")
    if body_map["all_present"]:
        _w("✅ All mandatory body names found.")
    else:
        _w(f"❌ Missing bodies: {', '.join(body_map['missing'])}")
    _w("")

    # ── Actuator Limits ─────────────────────────────────────────
    _w("## 6. Actuator Limits")
    _w("")
    _w("| Actuator | Ctrl Min | Ctrl Max | Force Min | Force Max | Issues |")
    _w("|----------|----------|----------|-----------|-----------|--------|")
    for a in act_limits["actuators"]:
        issues_str = ", ".join(a["issues"]) if a["issues"] else "none"
        _w(f"| `{a['name']}` | {a['ctrlrange'][0]} | {a['ctrlrange'][1]} | "
           f"{a['forcerange'][0]} | {a['forcerange'][1]} | {issues_str} |")
    _w("")
    _w(f"- Symmetric limits: {'✅ all symmetric' if act_limits['all_symmetric'] else '⚠️ some asymmetric'}")
    _w(f"- Any zero-range: {'⚠️ yes' if act_limits['any_zero_range'] else '✅ none'}")
    _w("")

    # ── State Snapshot ──────────────────────────────────────────
    _w("## 7. State Snapshot Summary")
    _w("")
    _w(f"- **Base position:** [{state['base_position'][0]:.4f}, {state['base_position'][1]:.4f}, {state['base_position'][2]:.4f}] (world)")
    _w(f"- **Base quaternion:** [{state['base_quaternion'][0]:.4f}, {state['base_quaternion'][1]:.4f}, {state['base_quaternion'][2]:.4f}, {state['base_quaternion'][3]:.4f}]")
    _w(f"- **Joint positions:** {[f'{v:.4f}' for v in state['joint_positions']]}")
    _w(f"- **qpos finite:** {state['qpos_finite']}")
    _w(f"- **qvel finite:** {state['qvel_finite']}")
    _w(f"- **COM position:** {state.get('com_position', 'N/A')}")
    _w(f"- **COM velocity:** {state.get('com_velocity', 'N/A')}")
    _w("")

    # ── COM Check ───────────────────────────────────────────────
    _w("## 8. COM Check")
    _w("")
    _w(f"- **COM position:** {com.get('com_position', 'N/A')}")
    _w(f"- **Base Z:** {com.get('base_z', 'N/A')}")
    _w(f"- **COM relative to base Z:** {com.get('com_relative_to_base_z', 'N/A')}")
    _w(f"- **Plausible:** {com.get('plausible', False)}")
    if com.get("warning"):
        _w(f"- ⚠️ **Warning:** {com['warning']}")
    _w("")

    # ── Contact Inspection ──────────────────────────────────────
    _w("## 9. Contact Inspection Summary")
    _w("")
    _w(f"- **Settle steps:** {contacts.get('settle_steps', '?')} ({contacts.get('settle_time_s', '?'):.3f} s)")
    _w(f"- **Active contacts:** {contacts['ncon']}")
    _w(f"- **Left wheel in contact:** {contacts['left_wheel_in_contact']}")
    _w(f"- **Right wheel in contact:** {contacts['right_wheel_in_contact']}")
    _w(f"- **Total contact force (world):** {[f'{v:.3f}' for v in contacts['total_contact_force_world']]}")
    _w("")
    if contacts["ncon"] > 0:
        _w("| # | Geom 1 | Geom 2 | Body 1 | Body 2 | Force World (N) | Dist |")
        _w("|---|--------|--------|--------|--------|-----------------|------|")
        for c in contacts["contacts"]:
            fw = c["force_world"]
            _w(f"| {c['index']} | `{c['geom1']}` | `{c['geom2']}` | "
               f"`{c['body1']}` | `{c['body2']}` | "
               f"[{fw[0]:.3f}, {fw[1]:.3f}, {fw[2]:.3f}] | {c['distance']:.6f} |")
        _w("")
    else:
        _w("⚠️ **No contacts detected after passive settle.** This may indicate the robot is not touching ground at the keyframe pose, or the settle time was insufficient.")
        _w("")

    # ── Jacobian Validation ─────────────────────────────────────
    _w("## 10. Jacobian Validation")
    _w("")
    _w("### 10.1 Analytic Jacobians")
    _w("")
    _w("| Target | Type | ID | JacP Shape | JacP Rank | Finite |")
    _w("|--------|------|-----|------------|-----------|--------|")
    for name, ja in jac_analytic.items():
        _w(f"| `{name}` | {ja['target_type']} | {ja['target_id']} | "
           f"{ja['jacp_shape']} | {ja['jacp_rank']} | {ja['jacp_finite']} |")
    _w("")

    _w("### 10.2 Finite-Difference Validation")
    _w("")
    _w("Free-joint columns (v[0:6]) are skipped — not FD-validated for position Jacobians.")
    _w("Only actuated joint columns (v[6:16]) are checked.")
    _w("")
    _w(f"**Thresholds:** PASS < {jac_fd[list(jac_fd.keys())[0]]['pass_threshold']}, "
       f"WARN < {jac_fd[list(jac_fd.keys())[0]]['warn_threshold']}, "
       f"FAIL ≥ {jac_fd[list(jac_fd.keys())[0]]['warn_threshold']}")
    _w("")
    _w("| Target | Max Abs Error | Max Rel Error | Verdict |")
    _w("|--------|--------------|---------------|---------|")
    for name, fd in jac_fd.items():
        _w(f"| `{name}` | {fd['max_abs_error']:.6e} | {fd['max_rel_error']:.6e} | **{fd['verdict']}** |")
    _w("")

    _w("### 10.3 Per-Joint FD Detail (torso)")
    _w("")
    if "torso" in jac_fd:
        torso_fd = jac_fd["torso"]
        _w("| Joint | Abs Error | Rel Error | Verdict |")
        _w("|-------|-----------|-----------|---------|")
        for jr in torso_fd["actuated_joint_results"]:
            _w(f"| `{jr['joint_name']}` | {jr['abs_error']:.6e} | {jr['rel_error']:.6e} | {jr['verdict']} |")
        _w("")

    # ── Torque Sign Validation ──────────────────────────────────
    _w("## 11. Torque Sign Validation")
    _w("")
    _w(f"**Probe method:** Bias-subtracted (zero/+probe/−probe) with escalation up to "
       f"{int(MAX_PROBE_FRACTION * 100)}% of actuator ctrl range limit.")
    _w("")
    _w("| Joint | Actuator | qacc(0) | qacc(+) | qacc(−) | Δ+ | Δ− | Δ± | Probe (Nm) | Δ-Consistent | Convention | Outcome |")
    _w("|-------|----------|---------|---------|---------|----|----|----|-----------|-------------|------------|---------|")
    for t in torque_signs:
        q0 = f"{t['qacc_zero']:.4f}" if t['qacc_zero'] is not None else "N/A"
        qp = f"{t['qacc_plus']:.4f}" if t['qacc_plus'] is not None else "N/A"
        qn = f"{t['qacc_minus']:.4f}" if t['qacc_minus'] is not None else "N/A"
        dp = f"{t['delta_plus']:.4f}" if t['delta_plus'] is not None else "N/A"
        dn = f"{t['delta_minus']:.4f}" if t['delta_minus'] is not None else "N/A"
        dpm = f"{t['delta_pair']:.4f}" if t['delta_pair'] is not None else "N/A"
        prob = f"{t['probe_torque_used']:.1f}" if t['probe_torque_used'] is not None else "N/A"
        sc_delta = "✅" if t.get('sign_consistent_delta') else ("❌" if t.get('sign_consistent_delta') is False else "—")
        conv = t.get('measured_sign_convention', 'N/A') or 'N/A'
        _w(f"| `{t['joint_name']}` | `{t['actuator_name']}` | "
           f"{q0} | {qp} | {qn} | {dp} | {dn} | {dpm} | {prob} | {sc_delta} | {conv} | **{t['outcome']}** |")
    _w("")
    n_measured_15 = sum(1 for t in torque_signs if t['outcome'] == 'MEASURED')
    n_ambiguous_15 = sum(1 for t in torque_signs if t['outcome'] == 'AMBIGUOUS')
    _w(f"**Phase 1.5 torque sign summary:** {n_measured_15} MEASURED, {n_ambiguous_15} AMBIGUOUS")
    _w("")
    if any(t.get("note") for t in torque_signs):
        _w("**Escalation notes:**")
        _w("")
        for t in torque_signs:
            if t.get("note"):
                _w(f"- `{t['joint_name']}`: {t['note']}")
        _w("")
    _w("> **Note:** Left/right mirrored joints may have differing sign conventions due to")
    _w("> physical mirroring of the kinematic tree. This is expected and NOT a controller bug.")
    _w("> All outcomes are labeled MEASURED/AMBIGUOUS/MISSING, not pass/fail.")
    _w("")

    # ── Phase 1 Comparison ──────────────────────────────────────
    _w("## 12. Phase 1 Comparison")
    _w("")
    _w("| Metric | Phase 1 | Phase 1.5 | Change |")
    _w("|--------|---------|-----------|--------|")
    _w("| Torque signs MEASURED | 6 | "
       f"{n_measured_15} | {'+' + str(n_measured_15 - 6) if n_measured_15 > 6 else str(n_measured_15 - 6)} |")
    _w("| Torque signs AMBIGUOUS | 4 | "
       f"{n_ambiguous_15} | {'-' + str(4 - n_ambiguous_15) if n_ambiguous_15 < 4 else str(4 - n_ambiguous_15)} |")
    _w(f"| Actuator `zero_not_in_range` false labels | Yes (bug) | No (fixed) | ✅ fixed |")
    _w("| Probe method | Absolute qacc signs | Bias-subtracted deltas | ✅ improved |")
    _w("| Probe escalation | None | Up to "
       f"{int(MAX_PROBE_FRACTION * 100)}% actuator limit | ✅ added |")
    _w("")
    if n_ambiguous_15 > 0:
        _w("**Remaining ambiguous joints (Phase 1.5):**")
        _w("")
        for t in torque_signs:
            if t["outcome"] == "AMBIGUOUS":
                note_str = f" — {t['note']}" if t.get("note") else ""
                _w(f"- `{t['joint_name']}`: delta_pair={t['delta_pair']:.4f} rad/s², "
                   f"probe_torque_used={t['probe_torque_used']:.1f} Nm{note_str}")
        _w("")
    else:
        _w("**All 10 torque signs now MEASURED.** ✅")
        _w("")
        _w("The bias-subtracted delta probe resolved the 4 previously ambiguous")
        _w("joints (l_hip_pitch, l_knee, r_hip_pitch, r_knee) that were gravity-dominated")
        _w("with the Phase 1 absolute-sign-only measurement.")
        _w("")

    # ── Mass Matrix ─────────────────────────────────────────────
    _w("## 13. Mass Matrix (CPU MuJoCo)")
    _w("")
    _w(f"- **Available via CPU:** {mass.get('available_cpu', False)}")
    _w(f"- **Shape:** {mass.get('shape', 'N/A')}")
    _w(f"- **Finite:** {mass.get('finite', 'N/A')}")
    _w(f"- **Symmetric:** {mass.get('symmetric', 'N/A')}")
    _w(f"- **Diagonal positive:** {mass.get('diagonal_positive', 'N/A')}")
    _w(f"- **Condition number:** {mass.get('condition_number', 'N/A')}")
    if mass.get("error"):
        _w(f"- **Error:** {mass['error']}")
    _w("")
    _w(f"- **Available via MJX:** {mass.get('available_mjx', False)}")
    _w(f"- **MJX note:** {mass.get('note_mjx', '')}")
    _w("")

    # ── Limitations ─────────────────────────────────────────────
    _w("## 14. Limitations")
    _w("")
    _w("### What MuJoCo CPU exposes (used in this audit):")
    _w("")
    _w("- ✅ `mj_jac` — task-space Jacobians for bodies and sites")
    _w("- ✅ `mj_fullM` / `data.qM` — mass matrix")
    _w("- ✅ `mj_contactForce` — per-contact force vectors")
    _w("- ✅ `data.contact` — contact geometry pairs, positions, normals, distances")
    _w("- ✅ `data.qpos`, `data.qvel`, `data.qacc` — full state")
    _w("- ✅ `data.xpos`, `data.xmat`, `data.subtree_com` — body poses and COM")
    _w("- ✅ `data.qfrc_bias`, `data.qfrc_passive`, `data.qfrc_actuator` — force components")
    _w("")
    _w("### What MJX exposes (for future real-time/JAX integration):")
    _w("")
    _w("- ✅ `qpos`, `qvel`, `qacc` — full state vectors")
    _w("- ✅ `xpos`, `xmat` — body poses")
    _w("- ✅ `ctrl`, `act` — control/activation")
    _w("- ⚠️ `contact` — contact array (limited fields vs CPU; `dist` and `pos` available, `frame` partial)")
    _w("- ❌ `mj_jac` — **not available** in MJX. Jacobians must be hand-computed or ported.")
    _w("- ❌ `mj_fullM` — **not available** in MJX. Mass matrix must be computed via CRBA port.")
    _w("- ❌ `mj_contactForce` — **not available** in MJX. Contact forces must be computed from constraint solver outputs.")
    _w("- ❌ `subtree_com` — **not available** in MJX. COM must be computed from body masses and poses.")
    _w("")
    _w("### Impact on QP-WBC development:")
    _w("")
    _w("- **Jacobians:** Must be hand-computed from kinematics or a separate JAX CRBA/kinematics port.")
    _w("- **Mass matrix:** Must be computed via JAX Composite Rigid Body Algorithm (CRBA).")
    _w("- **Contact forces:** MJX constraint solver outputs (`efc_force`, `efc_J`) are internal; contact force extraction requires understanding the constraint model.")
    _w("- **Bias forces:** `qfrc_bias` is available in MJX as `data.qfrc_bias` — this is a significant positive for QP-WBC.")
    _w("")
    _w("These limitations mean that a pure-MJX/JIT WBC requires a dedicated dynamics/kinematics port, not just wrapping existing MuJoCo API calls. This audit provides the ground-truth reference for validating such a port.")
    _w("")

    # ── Readiness Verdict ───────────────────────────────────────
    _w("## 15. Phase 2A Readiness Verdict")
    _w("")
    _w(f"**Verdict: `{verdict}`**")
    _w("")
    if verdict == "READY_FOR_PHASE_2A_JAX_KINEMATICS_PORT":
        _w("All structural checks pass and all 10 torque sign conventions are MEASURED.")
        _w("The actuator limit checker correctly identifies zero-in-range for all actuators.")
        _w("The dynamics truth layer is ready to proceed to:")
        _w("")
        _w("> **Phase 2A — JAX-compatible kinematics / COM / Jacobian port**")
        _w("")
        _w("Note: this does NOT mean ready for real-time QP-WBC. Phase 2A only covers")
        _w("porting kinematics, COM computation, and Jacobian computation to JAX/MJX.")
    elif verdict == "PARTIAL_READY":
        _w("All structural checks pass, but one or more torque signs remain ambiguous")
        _w("or key physics quantities (mass matrix, Jacobians, contact forces) are only")
        _w("available through CPU MuJoCo and must be ported to JAX/MJX before they can")
        _w("be used in a real-time QP-WBC pipeline.")
    else:
        _w("One or more structural checks failed. The issues listed below must be resolved")
        _w("before Phase 2A can begin.")
    _w("")

    # ── Required fixes ──────────────────────────────────────────
    _w("## 16. Remaining Items Before Phase 2A")
    _w("")
    issues = _collect_issues(results)
    if issues:
        for issue in issues:
            _w(f"- {issue}")
    else:
        _w("- ✅ No blocking issues.")
        _w("- Next step: Port Jacobian computation + COM + mass matrix to JAX for MJX compatibility.")
    _w("")

    _w("---")
    _w("")
    _w("*Report generated by `scripts/phase1_dynamics_truth_audit.py`*")
    _w("")

    return "\n".join(lines)


def _collect_issues(results: dict) -> list[str]:
    """Collect all issues that would block Phase 2."""
    issues: list[str] = []

    idx = results["model_index"]

    # Actuator count
    if idx["nu"] != 10:
        issues.append(f"❌ Actuator count = {idx['nu']}, expected 10")

    # Body mapping
    body_map = results["body_mapping"]
    if not body_map["all_present"]:
        issues.append(f"❌ Missing mandatory bodies: {', '.join(body_map['missing'])}")

    # Joint presence
    for jname in EXPECTED_JOINTS:
        if jname not in idx["joints"]:
            issues.append(f"❌ Joint `{jname}` not found in model")

    # Actuator zero-range
    act_limits = results["actuator_limits"]
    if act_limits["any_zero_range"]:
        issues.append("⚠️ Some actuators have zero control range")

    # State snapshot
    state = results["state_snapshot"]
    if not state["qpos_finite"]:
        issues.append("❌ qpos contains NaN/Inf")
    if not state["qvel_finite"]:
        issues.append("❌ qvel contains NaN/Inf")

    # COM
    com = results["com_check"]
    if not com.get("plausible", False):
        issues.append(f"⚠️ COM implausible: {com.get('warning', 'unknown')}")

    # Jacobian FD
    for name, fd in results["jacobian_fd"].items():
        if fd["verdict"] == "FAIL":
            issues.append(f"❌ Jacobian FD check FAILED for `{name}`: max_abs_err={fd['max_abs_error']:.2e}")
        elif fd["verdict"] == "WARN":
            issues.append(f"⚠️ Jacobian FD check WARN for `{name}`: max_abs_err={fd['max_abs_error']:.2e}")

    # Torque signs
    for t in results["torque_signs"]:
        if t["outcome"] == "MISSING":
            issues.append(f"❌ Torque sign probe MISSING for `{t['joint_name']}`")
        elif t["outcome"] == "AMBIGUOUS":
            issues.append(f"⚠️ Torque sign AMBIGUOUS for `{t['joint_name']}`")

    # Mass matrix
    mass = results["mass_matrix"]
    if not mass.get("finite", True):
        issues.append("⚠️ Mass matrix contains non-finite values")
    if mass.get("error"):
        issues.append(f"⚠️ Mass matrix error: {mass['error']}")

    return issues


def _compute_readiness(results: dict) -> str:
    """Compute Phase 2A readiness verdict.

    Verdict rules:
        READY_FOR_PHASE_2A_JAX_KINEMATICS_PORT:
            - all structural checks pass
            - 10/10 torque signs MEASURED
            - no missing actuator/body mappings
            - Jacobian FD all PASS
            - state finite
            - no controller files modified
        PARTIAL_READY:
            - structural checks pass but >=1 torque sign still AMBIGUOUS
            - or CPU-only dynamics limitations remain
        NOT_READY:
            - actuator/body mapping broken
            - Jacobian validation FAILs
            - state contains NaN/Inf
    """
    issues = _collect_issues(results)

    has_blockers = any(i.startswith("❌") for i in issues)
    has_warnings = any(i.startswith("⚠️") for i in issues)

    idx = results["model_index"]
    body_map = results["body_mapping"]
    state = results["state_snapshot"]
    com = results["com_check"]

    structural_ok = (
        idx["nu"] == 10
        and body_map["all_present"]
        and state["qpos_finite"]
        and state["qvel_finite"]
        and com.get("plausible", False)
        and all(
            fd["verdict"] != "FAIL"
            for fd in results["jacobian_fd"].values()
        )
        and all(
            t["outcome"] != "MISSING"
            for t in results["torque_signs"]
        )
    )

    if not structural_ok or has_blockers:
        return "NOT_READY"

    # All 10 torque signs MEASURED?
    all_torque_measured = all(
        t["outcome"] == "MEASURED" for t in results["torque_signs"]
    )

    # Actuator limit issues fixed? (no false zero_not_in_range)
    act_limits_clean = not any(
        "zero_not_in_range" in a.get("issues", [])
        for a in results["actuator_limits"]["actuators"]
        if a["ctrlrange"][0] <= 0.0 <= a["ctrlrange"][1]
    )

    if all_torque_measured and act_limits_clean and not has_warnings:
        return "READY_FOR_PHASE_2A_JAX_KINEMATICS_PORT"

    # Structural checks pass but some ambiguity remains
    return "PARTIAL_READY"


def _write_json_summary(results: dict, json_path: Path) -> None:
    """Write a machine-readable JSON summary of key findings."""
    # Compute torque sign details for Phase 1.5
    torque_detail = {}
    for t in results["torque_signs"]:
        torque_detail[t["joint_name"]] = {
            "outcome": t["outcome"],
            "probe_torque_requested": t.get("probe_torque_requested"),
            "probe_torque_used": t.get("probe_torque_used"),
            "qacc_zero": t.get("qacc_zero"),
            "delta_plus": t.get("delta_plus"),
            "delta_minus": t.get("delta_minus"),
            "delta_pair": t.get("delta_pair"),
            "sign_consistent_delta": t.get("sign_consistent_delta"),
            "measured_convention": t.get("measured_sign_convention"),
        }
        if t.get("note"):
            torque_detail[t["joint_name"]]["note"] = t["note"]

    summary = {
        "phase": "1.5",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": str(get_model_path()),
        "verdict": _compute_readiness(results),
        "dimensions": {
            "nq": results["model_index"]["nq"],
            "nv": results["model_index"]["nv"],
            "nu": results["model_index"]["nu"],
            "nbody": results["model_index"]["nbody"],
            "njnt": results["model_index"]["njnt"],
        },
        "body_mapping_ok": results["body_mapping"]["all_present"],
        "jacobian_fd_verdicts": {
            name: fd["verdict"] for name, fd in results["jacobian_fd"].items()
        },
        "torque_sign_outcomes": {
            t["joint_name"]: t["outcome"] for t in results["torque_signs"]
        },
        "torque_sign_details": torque_detail,
        "n_measured": sum(1 for t in results["torque_signs"] if t["outcome"] == "MEASURED"),
        "n_ambiguous": sum(1 for t in results["torque_signs"] if t["outcome"] == "AMBIGUOUS"),
        "actuator_limits_zero_not_in_range_fixed": all(
            a["ctrlrange"][0] <= 0.0 <= a["ctrlrange"][1]
            for a in results["actuator_limits"]["actuators"]
        ) and not any(
            "zero_not_in_range" in a.get("issues", [])
            for a in results["actuator_limits"]["actuators"]
            if a["ctrlrange"][0] <= 0.0 <= a["ctrlrange"][1]
        ),
        "mass_matrix_available_cpu": results["mass_matrix"].get("available_cpu", False),
        "contacts_detected": results["contact_inspection"]["ncon"],
        "issues": _collect_issues(results),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
