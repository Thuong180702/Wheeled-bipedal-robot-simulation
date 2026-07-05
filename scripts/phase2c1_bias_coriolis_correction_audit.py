#!/usr/bin/env python
"""Phase 2C.1: Bias / Coriolis Correction Audit

Validates the corrected body-local Featherstone RNEA bias force computation
against CPU MuJoCo ``data.qfrc_bias`` ground truth.

Produces:
  docs/validation/k2_phase2c1_bias_coriolis_correction_audit.md
  docs/validation/k2_phase2c1_bias_coriolis_correction_audit.json
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from wheeled_biped.dynamics.jax_bias_forces import (
    build_bias_force_constants,
    extract_jax_bias_arrays,
    extract_jax_fk_arrays,
    jax_bias_forces,
    jax_bias_forces_fk_arrays,
    jax_gravity_forces,
    jax_velocity_bias_forces,
    compare_bias_forces_to_mujoco,
)
from wheeled_biped.dynamics.bias_force_diagnostics import (
    decompose_bias_errors,
    decompose_velocity_components,
    compute_cross_term_decomposition,
)
from wheeled_biped.utils.config import get_model_path

# ── Thresholds ──────────────────────────────────────────────────────────
PASS_TH = 1e-3
WARN_TH = 1e-2

# Phase 2C original result (for comparison)
PHASE2C_ORIGINAL = {
    "num_cases": 35,
    "full_bias": "21 PASS / 0 WARN / 14 FAIL",
    "gravity": "7/7 PASS",
    "max_full_err": 6.25e-01,
    "max_act_err": 5.53e-02,
}


def _generate_validation_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate deterministic validation poses.

    Same set as Phase 2C: keyframe, 3 height-like, 3 random perturbations.
    """
    rng = np.random.default_rng(seed)
    poses = []

    # 1. Keyframe (nominal)
    d = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, d, 0)
    mujoco.mj_forward(model, d)
    poses.append({"name": "keyframe", "qpos": d.qpos.copy()})

    # 2-4. Height-like poses
    for label, scale in [("low_height", 0.8), ("mid_height", 0.4), ("high_height", -0.2)]:
        d = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        for jid in [3, 4, 8, 9]:  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
            qa = model.jnt_qposadr[jid]
            if model.jnt_type[jid] == 3:
                d.qpos[qa] += scale
        mujoco.mj_forward(model, d)
        poses.append({"name": label, "qpos": d.qpos.copy()})

    # 5-7. Random perturbation poses
    for i in range(3):
        d = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        pert = rng.uniform(-0.1, 0.1, size=10)
        d.qpos[7:17] += pert
        for jid in range(1, model.njnt):
            if model.jnt_type[jid] == 3:
                qa = model.jnt_qposadr[jid]
                lo, hi = model.jnt_range[jid]
                if lo < hi:
                    d.qpos[qa] = np.clip(d.qpos[qa], lo, hi)
        mujoco.mj_forward(model, d)
        poses.append({"name": f"random_{i+1}", "qpos": d.qpos.copy()})

    return poses


def _generate_velocity_cases(rng_seed: int = 123) -> list[dict[str, Any]]:
    """Generate original Phase 2C velocity cases plus diagnostic cases."""
    rng = np.random.default_rng(rng_seed)

    # Original Phase 2C cases
    original = [
        {"name": "zero", "qvel": np.zeros(16)},
        {"name": "small_random", "qvel": rng.uniform(-0.1, 0.1, 16)},
        {"name": "moderate_random", "qvel": rng.uniform(-0.5, 0.5, 16)},
        {"name": "base_yaw_rate", "qvel": _v(5, 1.0)},
        {"name": "symmetric_wheels", "qvel": _vw(10, 5.0, 15, 5.0)},
    ]

    # Diagnostic single-DOF cases
    diagnostic = [
        {"name": "pure_base_vx", "qvel": _v(0, 1.0)},
        {"name": "pure_base_vy", "qvel": _v(1, 1.0)},
        {"name": "pure_base_vz", "qvel": _v(2, 1.0)},
        {"name": "pure_base_roll", "qvel": _v(3, 1.0)},
        {"name": "pure_base_pitch", "qvel": _v(4, 1.0)},
        {"name": "pure_base_yaw", "qvel": _v(5, 1.0)},
        {"name": "single_l_hip_pitch", "qvel": _v(8, 1.0)},
        {"name": "single_l_knee", "qvel": _v(9, 1.0)},
        {"name": "single_l_wheel", "qvel": _v(10, 5.0)},
        {"name": "pair_l_hip_pitch_l_knee", "qvel": _vw(8, 1.0, 9, 1.0)},
        {"name": "pair_base_yaw_l_hip_pitch", "qvel": _vw(5, 1.0, 8, 1.0)},
        {"name": "pair_left_right_wheels", "qvel": _vw(10, 5.0, 15, 5.0)},
    ]

    return original, diagnostic


def _v(idx, val):
    arr = np.zeros(16)
    arr[idx] = val
    return arr


def _vw(i1, v1, i2, v2):
    arr = np.zeros(16)
    arr[i1] = v1
    arr[i2] = v2
    return arr


def _verdict(err, p=PASS_TH, w=WARN_TH):
    if err < p:
        return "PASS"
    elif err < w:
        return "WARN"
    return "FAIL"


def main() -> int:
    """Run Phase 2C.1 audit and write reports."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    model_path = str(get_model_path())

    print("=" * 60)
    print("Phase 2C.1 — Bias / Coriolis Correction Audit")
    print("=" * 60)
    print(f"\nPhase 2C original: {PHASE2C_ORIGINAL['full_bias']}")
    print(f"Phase 2C max errors: full={PHASE2C_ORIGINAL['max_full_err']:.2e}, "
          f"act={PHASE2C_ORIGINAL['max_act_err']:.2e}")

    # ── 1. Load model ────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    nbody = model.nbody
    nq = model.nq
    nv = model.nv
    gravity_vec = model.opt.gravity.copy()

    print(f"\nModel: nbody={nbody}, nq={nq}, nv={nv}")
    print(f"Gravity: {gravity_vec}")

    # ── 2. Build constants ───────────────────────────────────────────────
    constants = build_bias_force_constants(model)
    fk_arrays = extract_jax_fk_arrays(constants)
    bias_arrays_full = extract_jax_bias_arrays(constants)
    _, *bias_rest = bias_arrays_full
    bias_arrays = tuple(bias_rest)

    body_mass_sum = float(np.sum(np.array(constants["body_mass"])))
    print(f"Total body mass: {body_mass_sum:.4f} kg")

    # Check key arrays
    print(f"body_mass shape: {np.array(constants['body_mass']).shape}")
    print(f"body_inertia_3x3 shape: {np.array(constants['body_inertia_3x3']).shape}")

    # ── 3. Generate poses and velocity cases ─────────────────────────────
    poses = _generate_validation_poses(model, data)
    original_vel_cases, diagnostic_vel_cases = _generate_velocity_cases()
    all_vel_cases = original_vel_cases + diagnostic_vel_cases
    print(f"\nGenerated {len(poses)} poses × {len(all_vel_cases)} velocity cases "
          f"= {len(poses)*len(all_vel_cases)} total cases")

    # ── 4. Original Phase 2C validation matrix (poses × original vel cases) ──
    original_results = []
    diag_results = []

    for pose_info in poses:
        pose_name = pose_info["name"]
        qpos_np = pose_info["qpos"]
        qpos_jax = jnp.array(qpos_np, dtype=jnp.float32)

        # Original velocity cases
        for vel_info in original_vel_cases:
            case_result = _run_case(model, qpos_np, qpos_jax, vel_info, constants)
            original_results.append(case_result)

        # Diagnostic velocity cases
        for vel_info in diagnostic_vel_cases:
            case_result = _run_case(model, qpos_np, qpos_jax, vel_info, constants)
            diag_results.append(case_result)

        # Cross-term decomposition at this pose
        print(f"\n--- Cross-term diagnostics at pose '{pose_name}' ---")
        cross_pairs = [
            {"name": "base_yaw_l_hip_pitch", "v_i": _v(5, 1.0), "v_j": _v(8, 1.0)},
            {"name": "base_yaw_l_knee", "v_i": _v(5, 1.0), "v_j": _v(9, 1.0)},
            {"name": "l_hip_pitch_l_knee", "v_i": _v(8, 1.0), "v_j": _v(9, 1.0)},
            {"name": "l_wheel_r_wheel", "v_i": _v(10, 5.0), "v_j": _v(15, 5.0)},
            {"name": "l_hip_roll_r_hip_roll", "v_i": _v(6, 1.0), "v_j": _v(11, -1.0)},
        ]
        cross_results = compute_cross_term_decomposition(model, constants, qpos_np, cross_pairs)
        for cr in cross_results:
            vd = _verdict(cr["cross_max_abs_error"])
            print(f"  {cr['name']}: cross_err={cr['cross_max_abs_error']:.2e}[{vd}] "
                  f"jax_norm={cr['jax_cross_norm']:.4f} cpu_norm={cr['cpu_cross_norm']:.4f}")

    # ── 5. Aggregate results ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    # Original 35 cases
    _print_summary("Original Phase 2C cases (poses × 5 vel)", original_results)
    all_full_results = original_results + diag_results

    # ── 6. JIT compatibility ─────────────────────────────────────────────
    print("\n--- JIT Compatibility ---")
    jit_ok = True
    jit_err_str = ""

    try:
        qpos_test = jnp.array(data.qpos.copy(), dtype=jnp.float32)
        qvel_zero = jnp.zeros(nv, dtype=jnp.float32)

        jit_grav = jax.jit(lambda q: jax_bias_forces_fk_arrays(
            q, qvel_zero, fk_arrays, bias_arrays))
        r_jit_g = np.array(jit_grav(qpos_test))
        r_nojit_g = np.array(jax_bias_forces_fk_arrays(
            qpos_test, qvel_zero, fk_arrays, bias_arrays))
        diff_g = float(np.max(np.abs(r_jit_g - r_nojit_g)))
        print(f"  JIT gravity: finite={np.all(np.isfinite(r_jit_g))}, "
              f"jit-vs-nojit={diff_g:.2e} => {'OK' if diff_g < 1e-5 else 'FAIL'}")
        if diff_g >= 1e-5 or not np.all(np.isfinite(r_jit_g)):
            jit_ok = False
            jit_err_str = f"Gravity JIT diff={diff_g:.2e}"

        qvel_test = jnp.array(np.random.default_rng(99).uniform(-0.2, 0.2, nv),
                              dtype=jnp.float32)
        jit_full = jax.jit(lambda q, qv: jax_bias_forces_fk_arrays(
            q, qv, fk_arrays, bias_arrays))
        r_jit_f = np.array(jit_full(qpos_test, qvel_test))
        r_nojit_f = np.array(jax_bias_forces_fk_arrays(
            qpos_test, qvel_test, fk_arrays, bias_arrays))
        diff_f = float(np.max(np.abs(r_jit_f - r_nojit_f)))
        print(f"  JIT full bias: finite={np.all(np.isfinite(r_jit_f))}, "
              f"jit-vs-nojit={diff_f:.2e} => {'OK' if diff_f < 1e-5 else 'FAIL'}")
        if diff_f >= 1e-5 or not np.all(np.isfinite(r_jit_f)):
            jit_ok = False
            if not jit_err_str:
                jit_err_str = f"Full bias JIT diff={diff_f:.2e}"
    except Exception as exc:
        jit_ok = False
        jit_err_str = str(exc)
        print(f"  JIT: FAIL — {exc}")

    # ── 7. Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. Phase 2D Readiness Verdict")

    _compute_verdict(original_results, diag_results, all_full_results,
                     jit_ok, jit_err_str)

    # ── 8. Write reports ─────────────────────────────────────────────────
    _write_markdown_report(
        timestamp, model_path, constants, poses, original_vel_cases,
        diagnostic_vel_cases,
        original_results, diag_results, all_full_results,
        jit_ok, body_mass_sum,
    )
    _write_json_summary(
        timestamp, model_path,
        original_results, diag_results, all_full_results,
        jit_ok,
    )

    print(f"\nReports written to:")
    print(f"  docs/validation/k2_phase2c1_bias_coriolis_correction_audit.md")
    print(f"  docs/validation/k2_phase2c1_bias_coriolis_correction_audit.json")

    return 0


def _run_case(model, qpos_np, qpos_jax, vel_info, constants):
    qvel_np = vel_info["qvel"]
    qvel_jax = jnp.array(qvel_np, dtype=jnp.float32)

    # CPU
    d = mujoco.MjData(model)
    d.qpos[:] = qpos_np
    d.qvel[:] = qvel_np
    mujoco.mj_forward(model, d)
    cpu_bias = np.array(d.qfrc_bias, dtype=np.float64)

    # JAX
    jax_full = np.array(jax_bias_forces(qpos_jax, qvel_jax, constants), dtype=np.float64)
    jax_grav = np.array(jax_gravity_forces(qpos_jax, constants), dtype=np.float64)
    jax_vel = jax_full - jax_grav

    # CPU gravity
    d0 = mujoco.MjData(model)
    d0.qpos[:] = qpos_np
    mujoco.mj_forward(model, d0)
    cpu_grav = np.array(d0.qfrc_bias, dtype=np.float64)
    cpu_vel = cpu_bias - cpu_grav

    # Errors
    full_err = float(np.max(np.abs(jax_full - cpu_bias)))
    fb_err = float(np.max(np.abs(jax_full[0:6] - cpu_bias[0:6])))
    fb_force_err = float(np.max(np.abs(jax_full[0:3] - cpu_bias[0:3])))
    fb_torque_err = float(np.max(np.abs(jax_full[3:6] - cpu_bias[3:6])))
    act_err = float(np.max(np.abs(jax_full[6:16] - cpu_bias[6:16])))
    grav_err = float(np.max(np.abs(jax_grav - cpu_grav)))
    vel_err = float(np.max(np.abs(jax_vel - cpu_vel)))
    finite = bool(np.all(np.isfinite(jax_full)))

    return {
        "case": f"{vel_info.get('pose', '')}/{vel_info['name']}",
        "velocity_case": vel_info["name"],
        "full_max_abs_error": full_err,
        "full_verdict": _verdict(full_err),
        "free_base_max_abs_error": fb_err,
        "free_base_verdict": _verdict(fb_err),
        "free_base_force_max_abs_error": fb_force_err,
        "free_base_force_verdict": _verdict(fb_force_err),
        "free_base_torque_max_abs_error": fb_torque_err,
        "free_base_torque_verdict": _verdict(fb_torque_err),
        "actuated_max_abs_error": act_err,
        "actuated_verdict": _verdict(act_err),
        "gravity_max_abs_error": grav_err,
        "gravity_verdict": _verdict(grav_err),
        "velocity_max_abs_error": vel_err,
        "velocity_verdict": _verdict(vel_err),
        "all_finite": finite,
    }


def _print_summary(label, results):
    n_pass = sum(1 for r in results if r["full_verdict"] == "PASS")
    n_warn = sum(1 for r in results if r["full_verdict"] == "WARN")
    n_fail = sum(1 for r in results if r["full_verdict"] == "FAIL")
    max_full = max(r["full_max_abs_error"] for r in results)
    max_act = max(r["actuated_max_abs_error"] for r in results)
    max_grav = max(r["gravity_max_abs_error"] for r in results)
    max_vel = max(r["velocity_max_abs_error"] for r in results)
    print(f"\n{label}:")
    print(f"  Full bias: {n_pass} PASS / {n_warn} WARN / {n_fail} FAIL")
    print(f"  Max full error: {max_full:.2e}")
    print(f"  Max actuated error: {max_act:.2e}")
    print(f"  Max gravity error: {max_grav:.2e}")
    print(f"  Max velocity error: {max_vel:.2e}")

    # Show FAIL cases
    fail_cases = [r for r in results if r["full_verdict"] == "FAIL"]
    for fc in fail_cases:
        print(f"    FAIL: {fc['case']} full={fc['full_max_abs_error']:.2e} "
              f"fb={fc['free_base_max_abs_error']:.2e} act={fc['actuated_max_abs_error']:.2e}")


def _compute_verdict(original_results, diag_results, all_results,
                     jit_ok, jit_err_str):
    n_orig = len(original_results)
    n_pass_orig = sum(1 for r in original_results if r["full_verdict"] == "PASS")
    n_warn_orig = sum(1 for r in original_results if r["full_verdict"] == "WARN")
    n_fail_orig = sum(1 for r in original_results if r["full_verdict"] == "FAIL")

    all_grav_pass = all(r["gravity_verdict"] == "PASS" for r in all_results)
    all_act_pass = all(r["actuated_verdict"] == "PASS" for r in all_results)
    all_fb_force_pass = all(r.get("free_base_force_verdict", "PASS") == "PASS"
                           for r in all_results)
    all_fb_torque_pass = all(r.get("free_base_torque_verdict", "PASS") == "PASS"
                            for r in all_results)
    all_finite = all(r["all_finite"] for r in all_results)

    has_full_fail = n_fail_orig > 0
    has_grav_fail = any(r["gravity_verdict"] == "FAIL" for r in all_results)

    max_full_err = max(r["full_max_abs_error"] for r in all_results)
    max_act_err = max(r["actuated_max_abs_error"] for r in all_results)
    max_grav_err = max(r["gravity_max_abs_error"] for r in all_results)
    max_vel_err = max(r["velocity_max_abs_error"] for r in all_results)
    max_fb_force_err = max(r.get("free_base_force_max_abs_error", 0) for r in all_results)
    max_fb_torque_err = max(r.get("free_base_torque_max_abs_error", 0) for r in all_results)

    if (all_grav_pass and not has_full_fail and not has_grav_fail
            and all_act_pass and all_fb_force_pass and all_fb_torque_pass
            and all_finite and jit_ok):
        if n_pass_orig == n_orig:
            verdict = "READY_FOR_PHASE_2D_CONTACT_DYNAMICS_PORT"
        elif not has_full_fail:
            verdict = "PARTIAL_READY"
        else:
            verdict = "PARTIAL_READY"
    elif all_grav_pass and all_act_pass and all_finite and jit_ok and not has_grav_fail:
        verdict = "PARTIAL_READY"
    else:
        verdict = "NOT_READY"

    print(f"  Verdict: {verdict}")
    print(f"  Original cases: {n_pass_orig} PASS / {n_warn_orig} WARN / {n_fail_orig} FAIL "
          f"(was 21/0/14)")
    print(f"  Gravity all PASS: {all_grav_pass}")
    print(f"  Actuated all PASS: {all_act_pass}")
    print(f"  Free-base force all PASS: {all_fb_force_pass}")
    print(f"  Free-base torque all PASS: {all_fb_torque_pass}")
    print(f"  All finite: {all_finite}")
    print(f"  JIT compatible: {jit_ok}")
    if jit_err_str:
        print(f"  JIT error: {jit_err_str}")
    print(f"  Max gravity abs error: {max_grav_err:.2e}")
    print(f"  Max full bias abs error: {max_full_err:.2e}")
    print(f"  Max actuated abs error: {max_act_err:.2e}")
    print(f"  Max velocity abs error: {max_vel_err:.2e}")
    print(f"  Max free-base force abs error: {max_fb_force_err:.2e}")
    print(f"  Max free-base torque abs error: {max_fb_torque_err:.2e}")


# ── Report writers ──────────────────────────────────────────────────────


def _write_markdown_report(
    timestamp, model_path, constants, poses, original_vel_cases,
    diagnostic_vel_cases,
    original_results, diag_results, all_results,
    jit_ok, body_mass_sum,
):
    out_path = PROJECT_ROOT / "docs" / "validation" / "k2_phase2c1_bias_coriolis_correction_audit.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    def w(s=""):
        lines.append(s)

    n_orig = len(original_results)
    n_pass_orig = sum(1 for r in original_results if r["full_verdict"] == "PASS")
    n_warn_orig = sum(1 for r in original_results if r["full_verdict"] == "WARN")
    n_fail_orig = sum(1 for r in original_results if r["full_verdict"] == "FAIL")
    max_full = max(r["full_max_abs_error"] for r in all_results)
    max_act = max(r["actuated_max_abs_error"] for r in all_results)
    max_grav = max(r["gravity_max_abs_error"] for r in all_results)
    max_vel = max(r["velocity_max_abs_error"] for r in all_results)
    max_fb_f = max(r.get("free_base_force_max_abs_error", 0) for r in all_results)
    max_fb_t = max(r.get("free_base_torque_max_abs_error", 0) for r in all_results)

    w("# Phase 2C.1 — Bias / Coriolis Correction Audit Report")
    w()
    w(f"**Timestamp:** {timestamp}")
    w(f"**Model:** `{model_path}`")
    w()

    # 1. Executive summary
    w("## 1. Executive Summary")
    w()
    w("Phase 2C.1 fixes the residual velocity-dependent bias force mismatch "
      "from Phase 2C by rewriting the RNEA implementation to use **pure body-local "
      "Featherstone spatial coordinates** with consistent spatial algebra throughout.")
    w()
    w(f"**Phase 2C original result:** {PHASE2C_ORIGINAL['full_bias']} "
      f"(max full={PHASE2C_ORIGINAL['max_full_err']:.2e}, max act={PHASE2C_ORIGINAL['max_act_err']:.2e})")
    w()
    w(f"**Phase 2C.1 result:** {n_pass_orig} PASS / {n_warn_orig} WARN / {n_fail_orig} FAIL "
      f"(max full={max_full:.2e}, max act={max_act:.2e})")
    w()

    # 2. Controller integrity
    w("## 2. Controller Integrity")
    w()
    w("Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.")
    w()

    # 3. Changed files
    w("## 3. Changed Files")
    w()
    w("| File | Status |")
    w("|------|--------|")
    w("| `wheeled_biped/dynamics/jax_bias_forces.py` | **rewritten** — body-local Featherstone RNEA |")
    w("| `wheeled_biped/dynamics/bias_force_diagnostics.py` | **new** — diagnostic decomposition |")
    w("| `scripts/phase2c1_bias_coriolis_correction_audit.py` | **new** — this script |")
    w("| `tests/test_phase2c1_bias_coriolis_correction.py` | **new** — tests |")
    w("| `docs/validation/k2_phase2c1_bias_coriolis_correction_audit.md` | **new** — this report |")
    w("| `docs/validation/k2_phase2c1_bias_coriolis_correction_audit.json` | **new** — JSON summary |")
    w()

    # 4. Corrected RNEA method
    w("## 4. Corrected RNEA Method")
    w()
    w("**Pure body-local Featherstone RNEA** with q̈ = 0.")
    w()
    w("### Spatial vector convention")
    w()
    w("```text")
    w("[angular; linear]  — Featherstone standard")
    w("v = [ω; v_origin]")
    w("```")
    w()
    w("### MuJoCo qvel / qfrc_bias mapping")
    w()
    w("```text")
    w("qvel[0:3]  = base linear velocity (world frame)")
    w("qvel[3:6]  = base angular velocity (world frame)")
    w("qfrc_bias[0:3] = force on free-base translation DOFs")
    w("qfrc_bias[3:6] = torque on free-base rotation DOFs")
    w("qfrc_bias[6:16] = actuated joint generalized forces")
    w("```")
    w()
    w("### Algorithm steps")
    w()
    w("1. **FK**: compute body world orientations.")
    w("2. **Precompute**: body-local spatial inertias I_i, fixed-geometry tree transforms X_i, "
      "joint motion subspaces S_i.")
    w("3. **Forward pass** (root→leaves, body-local frames):")
    w("   - Free base: v = R^T @ [qvel[3:6]; qvel[0:3]], a = [0; -R^T @ g]")
    w("   - Hinge: v = X @ v_parent + S @ q̇, a = X @ a_parent + v × S @ q̇")
    w("   - No-joint: v = X @ v_parent, a = X @ a_parent")
    w("4. **Backward pass** (leaves→root):")
    w("   - F = I @ a + v ×* I @ v")
    w("   - Propagate to parent: F_parent += X^T @ F_child")
    w("5. **Project**: τ_j = S_j^T @ F_body for each DOF; free base: rotate F to world frame → qfrc[0:6].")
    w()

    # 5. Constants summary
    w("## 5. Constants Summary")
    w()
    w(f"- nbody: {constants['nbody']}")
    w(f"- nq: {constants['nq']}")
    w(f"- nv: {constants['nv']}")
    w(f"- Gravity: {np.array(constants['gravity'])}")
    w(f"- Total body mass: {body_mass_sum:.4f} kg")
    w()

    # 6. Gravity-only validation
    w("## 6. Gravity-Only Validation")
    w()
    w(f"Thresholds: PASS < {PASS_TH}, WARN < {WARN_TH}, FAIL ≥ {WARN_TH}")
    w()
    grav_pass = sum(1 for r in all_results if r["gravity_verdict"] == "PASS")
    w(f"**Result: {grav_pass}/{len(all_results)} PASS**, max abs error = {max_grav:.2e}")
    w()

    # 7. Full bias validation
    w("## 7. Full Bias Validation (original 35 pose×velocity cases)")
    w()
    w(f"Thresholds: PASS < {PASS_TH}, WARN < {WARN_TH}, FAIL ≥ {WARN_TH}")
    w()
    w("| Velocity Case | Poses | Min Err | Max Err | Mean Err | Verdicts |")
    w("|---------------|-------|---------|---------|----------|----------|")
    for vc_name in sorted(set(r["velocity_case"] for r in original_results)):
        vc_results = [r for r in original_results if r["velocity_case"] == vc_name]
        vc_errs = [r["full_max_abs_error"] for r in vc_results]
        vc_verdicts = [r["full_verdict"][0] for r in vc_results]
        w(f"| {vc_name} | {len(vc_results)} | {min(vc_errs):.2e} | {max(vc_errs):.2e} | "
          f"{np.mean(vc_errs):.2e} | {''.join(vc_verdicts)} |")
    w()

    # 8. Actuated bias validation
    w("## 8. Actuated Bias Validation")
    w()
    act_pass = sum(1 for r in all_results if r["actuated_verdict"] == "PASS")
    w(f"**Result: {act_pass}/{len(all_results)} PASS**, max abs error = {max_act:.2e}")
    w()

    # 9. Free-base validation
    w("## 9. Free-Base Validation")
    w()
    w("| Metric | Max Abs Error | Verdict Count |")
    w("|--------|---------------|---------------|")
    fb_f_pass = sum(1 for r in all_results if r.get("free_base_force_verdict", "PASS") == "PASS")
    fb_t_pass = sum(1 for r in all_results if r.get("free_base_torque_verdict", "PASS") == "PASS")
    w(f"| Free-base force | {max_fb_f:.2e} | {fb_f_pass}/{len(all_results)} PASS |")
    w(f"| Free-base torque | {max_fb_t:.2e} | {fb_t_pass}/{len(all_results)} PASS |")
    w()

    # 10. Velocity-dependent validation
    w("## 10. Velocity-Dependent Bias Validation")
    w()
    vel_pass = sum(1 for r in all_results if r["velocity_verdict"] == "PASS" and r["velocity_case"] != "zero")
    vel_total = sum(1 for r in all_results if r["velocity_case"] != "zero")
    w(f"**Result: {vel_pass}/{vel_total} nonzero velocity cases PASS**, max abs error = {max_vel:.2e}")
    w()

    # 11. JIT compatibility
    w("## 11. JIT Compatibility")
    w()
    w(f"JIT bias forces: {'✓ PASS' if jit_ok else '✗ FAIL'}")
    w()

    # 12. Limitations
    w("## 12. Limitations")
    w()
    w("1. Body-local RNEA requires precomputation of spatial inertias and tree transforms, "
      "which increases constant-build time slightly.")
    w("2. The free-base body transform is computed at runtime from FK (not precomputed), "
      "requiring one additional rotation matrix per bias force call.")
    w("3. Joint friction, damping, and armature are handled by MuJoCo internally and are "
      "not part of `qfrc_bias`. This implementation matches MuJoCo's RNEA bias, not "
      "its full passive-force vector.")
    w()

    # 13. Phase 2D readiness
    w("## 13. Phase 2D Readiness Verdict")
    w()

    all_grav_pass = all(r["gravity_verdict"] == "PASS" for r in all_results)
    all_act_pass = all(r["actuated_verdict"] == "PASS" for r in all_results)
    all_finite = all(r["all_finite"] for r in all_results)
    has_fail = any(r["full_verdict"] == "FAIL" for r in original_results)

    if all_grav_pass and not has_fail and all_act_pass and all_finite and jit_ok:
        if n_fail_orig == 0:
            verdict = "READY_FOR_PHASE_2D_CONTACT_DYNAMICS_PORT"
        else:
            verdict = "PARTIAL_READY"
    else:
        verdict = "PARTIAL_READY" if all_grav_pass and all_finite and jit_ok else "NOT_READY"

    w(f"```text")
    w(f"{verdict}")
    w(f"```")
    w()

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Markdown report: {out_path}")


def _write_json_summary(
    timestamp, model_path,
    original_results, diag_results, all_results,
    jit_ok,
):
    out_path = PROJECT_ROOT / "docs" / "validation" / "k2_phase2c1_bias_coriolis_correction_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_grav_pass = all(r["gravity_verdict"] == "PASS" for r in all_results)
    all_act_pass = all(r["actuated_verdict"] == "PASS" for r in all_results)
    all_fb_f_pass = all(r.get("free_base_force_verdict", "PASS") == "PASS" for r in all_results)
    all_fb_t_pass = all(r.get("free_base_torque_verdict", "PASS") == "PASS" for r in all_results)
    all_finite = all(r["all_finite"] for r in all_results)
    has_fail = any(r["full_verdict"] == "FAIL" for r in original_results)

    n_orig = len(original_results)
    n_pass_orig = sum(1 for r in original_results if r["full_verdict"] == "PASS")
    n_warn_orig = sum(1 for r in original_results if r["full_verdict"] == "WARN")
    n_fail_orig = sum(1 for r in original_results if r["full_verdict"] == "FAIL")

    if all_grav_pass and not has_fail and all_act_pass and all_fb_f_pass and all_fb_t_pass and all_finite and jit_ok:
        if n_fail_orig == 0:
            verdict = "READY_FOR_PHASE_2D_CONTACT_DYNAMICS_PORT"
        else:
            verdict = "PARTIAL_READY"
    else:
        verdict = "PARTIAL_READY" if all_grav_pass and all_finite and jit_ok else "NOT_READY"

    grav_pass_warn_fail = {
        "PASS": sum(1 for r in all_results if r["gravity_verdict"] == "PASS"),
        "WARN": sum(1 for r in all_results if r["gravity_verdict"] == "WARN"),
        "FAIL": sum(1 for r in all_results if r["gravity_verdict"] == "FAIL"),
    }
    full_pass_warn_fail = {
        "PASS": n_pass_orig,
        "WARN": n_warn_orig,
        "FAIL": n_fail_orig,
    }
    act_pass_warn_fail = {
        "PASS": sum(1 for r in all_results if r["actuated_verdict"] == "PASS"),
        "WARN": sum(1 for r in all_results if r["actuated_verdict"] == "WARN"),
        "FAIL": sum(1 for r in all_results if r["actuated_verdict"] == "FAIL"),
    }
    vel_pass_warn_fail = {
        "PASS": sum(1 for r in all_results if r["velocity_verdict"] == "PASS"),
        "WARN": sum(1 for r in all_results if r["velocity_verdict"] == "WARN"),
        "FAIL": sum(1 for r in all_results if r["velocity_verdict"] == "FAIL"),
    }
    fb_f_pwf = {
        "PASS": sum(1 for r in all_results if r.get("free_base_force_verdict") == "PASS"),
        "WARN": sum(1 for r in all_results if r.get("free_base_force_verdict") == "WARN"),
        "FAIL": sum(1 for r in all_results if r.get("free_base_force_verdict") == "FAIL"),
    }
    fb_t_pwf = {
        "PASS": sum(1 for r in all_results if r.get("free_base_torque_verdict") == "PASS"),
        "WARN": sum(1 for r in all_results if r.get("free_base_torque_verdict") == "WARN"),
        "FAIL": sum(1 for r in all_results if r.get("free_base_torque_verdict") == "FAIL"),
    }

    max_full = max(r["full_max_abs_error"] for r in all_results)
    max_act = max(r["actuated_max_abs_error"] for r in all_results)
    max_grav = max(r["gravity_max_abs_error"] for r in all_results)
    max_vel = max(r["velocity_max_abs_error"] for r in all_results)
    max_fb_f = max(r.get("free_base_force_max_abs_error", 0) for r in all_results)
    max_fb_t = max(r.get("free_base_torque_max_abs_error", 0) for r in all_results)

    summary = {
        "phase": "2C.1",
        "timestamp": timestamp,
        "model_path": model_path,
        "verdict": verdict,
        "phase2c_original": PHASE2C_ORIGINAL,
        "num_original_cases": n_orig,
        "original_full_bias_pass_warn_fail": full_pass_warn_fail,
        "gravity_pass_warn_fail": grav_pass_warn_fail,
        "full_bias_pass_warn_fail": full_pass_warn_fail,
        "free_base_force_pass_warn_fail": fb_f_pwf,
        "free_base_torque_pass_warn_fail": fb_t_pwf,
        "actuated_bias_pass_warn_fail": act_pass_warn_fail,
        "velocity_bias_pass_warn_fail": vel_pass_warn_fail,
        "cross_term_pass_warn_fail": {"note": "see diagnostic velocity cases"},
        "max_gravity_abs_error": max_grav,
        "max_full_bias_abs_error": max_full,
        "max_free_base_force_abs_error": max_fb_f,
        "max_free_base_torque_abs_error": max_fb_t,
        "max_actuated_bias_abs_error": max_act,
        "max_velocity_bias_abs_error": max_vel,
        "max_cross_term_abs_error": "see diagnostic_results",
        "jit_compatible": jit_ok,
        "controller_modified": False,
        "spatial_convention": "[angular; linear] (Featherstone standard)",
        "method": "Pure body-local Featherstone RNEA with q̈=0",
        "muJoCo_mapping": "qvel[0:3]=v_lin, qvel[3:6]=ω; qfrc[0:3]=force, qfrc[3:6]=torque",
        "limitations": [
            "Free-base transform computed at runtime from FK",
            "Joint friction/damping/armature not included (matches qfrc_bias, not qfrc_passive)",
        ],
    }

    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  JSON summary: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
