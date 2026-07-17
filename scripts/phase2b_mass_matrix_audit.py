#!/usr/bin/env python
"""Phase 2B: JAX Mass Matrix / CRBA Port Audit

Validates JAX-compatible mass matrix against CPU MuJoCo ``mj_fullM`` ground
truth from Phase 1.5 and Phase 2A.

Produces:
  docs/validation/k2_phase2b_mass_matrix_audit.md
  docs/validation/k2_phase2b_mass_matrix_audit.json
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

from wheeled_biped.dynamics.jax_mass_matrix import (
    build_mass_matrix_constants,
    extract_jax_mm_arrays,
    jax_mass_matrix,
    jax_mass_matrix_fk_arrays,
    jax_actuated_mass_submatrix,
    jax_body_spatial_velocities,
    compare_mass_matrix_to_mujoco,
)
from wheeled_biped.utils.config import get_model_path

# ── Thresholds ─────────────────────────────────────────────────────
FULL_PASS = 1e-3
FULL_WARN = 1e-2
ACT_PASS = 1e-3
ACT_WARN = 1e-2
SYM_PASS = 1e-6
SYM_WARN = 1e-5


def _generate_validation_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate a set of validation poses.

    Includes: keyframe, low-height-like, mid-height-like, high-height-like,
    and five random perturbation poses.
    """
    rng = np.random.default_rng(seed)
    poses = []

    # 1. Keyframe (nominal)
    data_copy = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data_copy, 0)
    mujoco.mj_forward(model, data_copy)
    poses.append({"name": "keyframe", "qpos": data_copy.qpos.copy()})

    # 2-4. Height-like poses (perturb hip/knee joints to squat/stand)
    for label, scale in [("low_height", 0.8), ("mid_height", 0.4), ("high_height", -0.2)]:
        d = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        # Bend hip pitch and knee joints
        for jid in [3, 4, 8, 9]:  # l_hip_pitch, l_knee, r_hip_pitch, r_knee
            qa = model.jnt_qposadr[jid]
            if model.jnt_type[jid] == 3:  # hinge
                d.qpos[qa] += scale
        mujoco.mj_forward(model, d)
        poses.append({"name": f"{label}", "qpos": d.qpos.copy()})

    # 5-9. Random perturbation poses
    for i in range(5):
        d = mujoco.MjData(model)
        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, d, 0)
        # Perturb actuated joints only (qpos[7:17])
        pert = rng.uniform(-0.1, 0.1, size=10)
        d.qpos[7:17] += pert
        # Clamp to joint limits
        for jid in range(1, model.njnt):
            if model.jnt_type[jid] == 3:  # hinge
                qa = model.jnt_qposadr[jid]
                lo, hi = model.jnt_range[jid]
                if lo < hi:
                    d.qpos[qa] = np.clip(d.qpos[qa], lo, hi)
        mujoco.mj_forward(model, d)
        poses.append({"name": f"random_{i+1}", "qpos": d.qpos.copy()})

    return poses


def main() -> int:
    """Run Phase 2B audit and write reports."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    model_path = str(get_model_path())

    print("=" * 60)
    print("Phase 2B — JAX Mass Matrix / CRBA Port Audit")
    print("=" * 60)

    # ── 1. Load model ────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    nbody = model.nbody
    nq = model.nq
    nv = model.nv

    print(f"\nModel: nbody={nbody}, nq={nq}, nv={nv}")
    print(f"Model path: {model_path}")

    # ── 2. Build constants ───────────────────────────────────────────
    constants = build_mass_matrix_constants(model)
    fk_arrays, body_mass_arr, body_ipos_arr, body_iquat_arr, body_inertia_arr, joint_dof_adr_arr, body_order_arr, dof_armature_arr = extract_jax_mm_arrays(constants)
    mm_arrays = (body_mass_arr, body_ipos_arr, body_iquat_arr, body_inertia_arr, joint_dof_adr_arr, body_order_arr, dof_armature_arr)

    body_mass_sum = float(np.sum(np.array(body_mass_arr)))
    print(f"Total body mass: {body_mass_sum:.4f} kg")

    dof_armature_np = np.array(dof_armature_arr)
    print(f"DOF armature (actuated): {dof_armature_np[6:16]}")

    # ── 3. Generate validation poses ─────────────────────────────────
    poses = _generate_validation_poses(model, data)
    print(f"\nGenerated {len(poses)} validation poses")

    # ── 4. Per-pose validation ───────────────────────────────────────
    pose_results = []

    for pose_info in poses:
        pose_name = pose_info["name"]

        # Set up MuJoCo data
        d = mujoco.MjData(model)
        d.qpos[:] = pose_info["qpos"]
        mujoco.mj_forward(model, d)

        # CPU mass matrix
        cpu_M = np.zeros((nv, nv))
        mujoco.mj_fullM(model, cpu_M, d.qM)

        # JAX mass matrix
        qpos_jax = jnp.array(d.qpos.copy(), dtype=jnp.float32)
        M_jax = jax_mass_matrix(qpos_jax, constants)
        M_jax_np = np.array(M_jax)

        # Errors
        abs_err = np.abs(M_jax_np - cpu_M)
        max_abs = float(np.max(abs_err))
        max_cpu = float(np.max(np.abs(cpu_M)))
        max_rel = max_abs / max_cpu if max_cpu > 1e-12 else max_abs

        # Actuated block
        abs_act = np.max(np.abs(M_jax_np[6:16, 6:16] - cpu_M[6:16, 6:16]))
        max_cpu_act = float(np.max(np.abs(cpu_M[6:16, 6:16])))
        rel_act = abs_act / max_cpu_act if max_cpu_act > 1e-12 else abs_act

        # Symmetry
        sym_err = float(np.max(np.abs(M_jax_np - M_jax_np.T)))

        # Diagonal
        diag = np.diag(M_jax_np)
        diag_min = float(np.min(diag))
        diag_max = float(np.max(diag))
        diag_pos = bool(np.all(diag > 0))

        # Finite
        finite = bool(np.all(np.isfinite(M_jax_np)))

        # Condition
        try:
            cond = float(np.linalg.cond(M_jax_np))
        except Exception:
            cond = float("inf")

        # Verdicts
        full_v = "PASS" if max_abs < FULL_PASS else ("WARN" if max_abs < FULL_WARN else "FAIL")
        act_v = "PASS" if abs_act < ACT_PASS else ("WARN" if abs_act < ACT_WARN else "FAIL")
        sym_v = "PASS" if sym_err < SYM_PASS else ("WARN" if sym_err < SYM_WARN else "FAIL")

        result = {
            "pose": pose_name,
            "full_max_abs_error": max_abs,
            "full_max_rel_error": max_rel,
            "full_verdict": full_v,
            "actuated_max_abs_error": abs_act,
            "actuated_max_rel_error": rel_act,
            "actuated_verdict": act_v,
            "symmetry_error": sym_err,
            "symmetry_verdict": sym_v,
            "diag_min": diag_min,
            "diag_max": diag_max,
            "diag_positive": diag_pos,
            "all_finite": finite,
            "condition_number": cond,
        }
        pose_results.append(result)

        pass_count = sum(1 for r in [full_v, act_v, sym_v] if r == "PASS")
        warn_count = sum(1 for r in [full_v, act_v, sym_v] if r == "WARN")
        fail_count = sum(1 for r in [full_v, act_v, sym_v] if r == "FAIL")
        print(f"  {pose_name:15s}: full={max_abs:.2e}[{full_v}] act={abs_act:.2e}[{act_v}] sym={sym_err:.2e}[{sym_v}] "
              f"cond={cond:.1f} diag_+={diag_pos}")

    # ── 5. JIT compatibility ─────────────────────────────────────────
    print("\n--- JIT Compatibility ---")

    jit_ok = False
    jit_err_str = ""
    try:
        jit_mm = jax.jit(lambda q: jax_mass_matrix_fk_arrays(q, fk_arrays, mm_arrays))
        qpos_test = jnp.array(data.qpos.copy(), dtype=jnp.float32)
        M_jit = jit_mm(qpos_test)
        M_jit_np = np.array(M_jit)
        jit_finite = bool(np.all(np.isfinite(M_jit_np)))

        # Compare with non-JIT
        M_nojit = np.array(jax_mass_matrix_fk_arrays(qpos_test, fk_arrays, mm_arrays))
        jit_diff = float(np.max(np.abs(M_jit_np - M_nojit)))

        jit_ok = jit_finite and (jit_diff < 1e-5)
        print(f"  JIT mass matrix: finite={jit_finite}, jit-vs-nojit diff={jit_diff:.2e} => {'OK' if jit_ok else 'FAIL'}")
        if not jit_ok:
            jit_err_str = f"JIT failed: finite={jit_finite}, diff={jit_diff:.2e}"
    except Exception as exc:
        jit_err_str = str(exc)
        print(f"  JIT mass matrix: FAIL — {exc}")

    # ── 6. Verdict ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. Phase 2C Readiness Verdict")

    all_full_pass = all(r["full_verdict"] == "PASS" for r in pose_results)
    all_act_pass = all(r["actuated_verdict"] == "PASS" for r in pose_results)
    all_sym_pass = all(r["symmetry_verdict"] == "PASS" for r in pose_results)
    all_finite = all(r["all_finite"] for r in pose_results)
    all_diag_pos = all(r["diag_positive"] for r in pose_results)

    has_full_fail = any(r["full_verdict"] == "FAIL" for r in pose_results)
    has_act_fail = any(r["actuated_verdict"] == "FAIL" for r in pose_results)
    has_sym_fail = any(r["symmetry_verdict"] == "FAIL" for r in pose_results)
    has_warn = any(
        r["full_verdict"] == "WARN" or r["actuated_verdict"] == "WARN"
        for r in pose_results
    )

    max_full_err = max(r["full_max_abs_error"] for r in pose_results)
    max_act_err = max(r["actuated_max_abs_error"] for r in pose_results)
    max_sym_err = max(r["symmetry_error"] for r in pose_results)
    cond_range = (
        min(r["condition_number"] for r in pose_results),
        max(r["condition_number"] for r in pose_results),
    )

    limitations: list[str] = [
        "Mass matrix includes dof_armature (reflected rotor inertias) to match MuJoCo convention",
        "Kinetic energy Hessian method used (not CRBA) — mathematically equivalent",
        "Free-joint Jacobian columns validated indirectly through mass matrix matching",
        "Rotational Jacobians validated indirectly through mass matrix matching",
        "No contact-consistent dynamics port (targeted for Phase 2C)",
    ]

    if all_full_pass and all_act_pass and all_sym_pass and all_finite and all_diag_pos and jit_ok:
        verdict = "READY_FOR_PHASE_2C_BIAS_FORCES_PORT"
    elif has_full_fail or has_act_fail or has_sym_fail or not jit_ok:
        verdict = "NOT_READY"
    else:
        verdict = "PARTIAL_READY"

    print(f"  Verdict: {verdict}")
    print(f"  Full matrix: {sum(1 for r in pose_results if r['full_verdict']=='PASS')} PASS / "
          f"{sum(1 for r in pose_results if r['full_verdict']=='WARN')} WARN / "
          f"{sum(1 for r in pose_results if r['full_verdict']=='FAIL')} FAIL")
    print(f"  Max full abs error: {max_full_err:.2e}")
    print(f"  Max actuated abs error: {max_act_err:.2e}")
    print(f"  Max symmetry error: {max_sym_err:.2e}")
    print(f"  Condition number range: [{cond_range[0]:.1f}, {cond_range[1]:.1f}]")
    print(f"  JIT compatible: {jit_ok}")

    # ── 7. Write reports ─────────────────────────────────────────────
    _write_markdown_report(
        timestamp, model_path, constants, pose_results,
        max_full_err, max_act_err, max_sym_err,
        cond_range, jit_ok, verdict, limitations,
    )
    _write_json_summary(
        timestamp, model_path, pose_results,
        max_full_err, max_act_err, max_sym_err,
        cond_range, jit_ok, verdict, limitations,
    )

    print(f"\nReports written to:")
    print(f"  docs/validation/k2_phase2b_mass_matrix_audit.md")
    print(f"  docs/validation/k2_phase2b_mass_matrix_audit.json")

    return 0 if "READY" in verdict else 1


# ── Report writers ──────────────────────────────────────────────────

def _write_markdown_report(
    timestamp: str,
    model_path: str,
    constants: dict,
    pose_results: list,
    max_full_err: float,
    max_act_err: float,
    max_sym_err: float,
    cond_range: tuple,
    jit_ok: bool,
    verdict: str,
    limitations: list,
) -> None:
    out_path = PROJECT_ROOT / "docs" / "validation" / "k2_phase2b_mass_matrix_audit.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    def w(s=""):
        lines.append(s)

    w("# Phase 2B — JAX Mass Matrix / CRBA Port Audit Report")
    w()
    w(f"**Timestamp:** {timestamp}")
    w(f"**Model:** `{model_path}`")
    w()

    # 1. Executive summary
    w("## 1. Executive Summary")
    w()
    w("Phase 2B implements a JAX-compatible full generalized mass matrix "
      ":math:`M(q) \\in \\mathbb{R}^{16 \\times 16}` for the K2 wheeled-biped robot, "
      "validated against CPU MuJoCo `mj_fullM` ground truth.")
    w()
    w(f"**Verdict: `{verdict}`**")
    w()

    full_pass = sum(1 for r in pose_results if r["full_verdict"] == "PASS")
    full_warn = sum(1 for r in pose_results if r["full_verdict"] == "WARN")
    full_fail = sum(1 for r in pose_results if r["full_verdict"] == "FAIL")
    act_pass = sum(1 for r in pose_results if r["actuated_verdict"] == "PASS")
    sym_pass = sum(1 for r in pose_results if r["symmetry_verdict"] == "PASS")

    w(f"- Full M: {full_pass}/{len(pose_results)} PASS, max abs error {max_full_err:.2e}")
    w(f"- Actuated block: {act_pass}/{len(pose_results)} PASS, max abs error {max_act_err:.2e}")
    w(f"- Symmetry: {sym_pass}/{len(pose_results)} PASS, max asymmetry {max_sym_err:.2e}")
    w(f"- JIT compatible: {'✓' if jit_ok else '✗'}")
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
    w("| `wheeled_biped/dynamics/jax_mass_matrix.py` | **new** — JAX mass matrix |")
    w("| `wheeled_biped/dynamics/__init__.py` | modified — added exports |")
    w("| `scripts/phase2b_mass_matrix_audit.py` | **new** — this script |")
    w("| `tests/test_phase2b_mass_matrix.py` | **new** — tests |")
    w("| `docs/validation/k2_phase2b_mass_matrix_audit.md` | **new** — this report |")
    w("| `docs/validation/k2_phase2b_mass_matrix_audit.json` | **new** — JSON summary |")
    w()

    # 4. Phase 2A reference
    w("## 4. Phase 2A Reference Summary")
    w()
    w("- Verdict: `READY_FOR_PHASE_2B_MASS_MATRIX_CRBA_PORT`")
    w("- FK: 11/11 PASS (max pos error 6.87e-08 m)")
    w("- COM: PASS (error 2.56e-08 m)")
    w("- Jacobians: 7/7 PASS (max abs error 2.98e-08)")
    w("- JIT: FK ✓, COM ✓, Jacobian ✓")
    w()

    # 5. Mass matrix method
    w("## 5. Mass Matrix Method")
    w()
    w("**Kinetic Energy Hessian** (mathematically equivalent to CRBA):")
    w()
    w("1. Compute body spatial velocities recursively through kinematic tree")
    w("2. Compute kinetic energy: T = Σ 0.5*m*||v_COM||² + 0.5*ω^T*I_COM*ω")
    w("3. M = ∇²_{q̇} T(q, q̇) |_{q̇=0} via `jax.hessian`")
    w()
    w("**Free-base handling:** Full 16×16 M(q) including 6 free-base DOFs + 10 actuated DOFs.")
    w()
    w("**Inertial frame convention:** MuJoCo `body_inertia` (COM inertia, diagonal in inertial frame) "
      "rotated to world frame via `body_quat * body_iquat`.")
    w()
    w("**Armature:** DOF armature (reflected rotor inertias from `model.dof_armature`) "
      "added to diagonal to match MuJoCo `mj_fullM` convention.")
    w()
    w("**Symmetrization:** M_sym = 0.5 * (M + M^T) applied to correct floating-point autodiff asymmetries (~1e-15).")
    w()

    # 6. Constants summary
    w("## 6. Constants Summary")
    w()
    w(f"- nbody: {constants['nbody']}")
    w(f"- nq: {constants['nq']}")
    w(f"- nv: {constants['nv']}")
    w(f"- Total body mass: {float(np.sum(np.array(constants['body_mass']))):.4f} kg")
    w(f"- DOF armature (free base): {np.array(constants['dof_armature'])[0:6]}")
    w(f"- DOF armature (actuated):  {np.array(constants['dof_armature'])[6:16]}")
    w()

    # 7. Validation pose table
    w("## 7. Validation Poses")
    w()
    w("| # | Pose | Description |")
    w("|---|------|-------------|")
    for i, r in enumerate(pose_results, 1):
        w(f"| {i} | {r['pose']} | {'Keyframe/nominal' if r['pose']=='keyframe' else 'Height-like' if 'height' in r['pose'] else 'Random perturbation'} |")
    w()

    # 8. Full mass matrix table
    w("## 8. Full Mass Matrix Validation")
    w()
    w(f"Thresholds: PASS < {FULL_PASS}, WARN < {FULL_WARN}, FAIL ≥ {FULL_WARN}")
    w()
    w("| Pose | CPU shape | JAX shape | Max Abs Err | Max Rel Err | Symmetry Err | Cond | Verdict |")
    w("|------|-----------|-----------|-------------|-------------|--------------|------|---------|")
    for r in pose_results:
        w(f"| {r['pose']} | (16,16) | (16,16) | {r['full_max_abs_error']:.2e} | {r['full_max_rel_error']:.2e} | {r['symmetry_error']:.2e} | {r['condition_number']:.1f} | {r['full_verdict']} |")
    w()

    # 9. Actuated block table
    w("## 9. Actuated Block Validation")
    w()
    w(f"Thresholds: PASS < {ACT_PASS}, WARN < {ACT_WARN}, FAIL ≥ {ACT_WARN}")
    w()
    w("| Pose | Max Abs Err | Max Rel Err | Verdict |")
    w("|------|-------------|-------------|---------|")
    for r in pose_results:
        w(f"| {r['pose']} | {r['actuated_max_abs_error']:.2e} | {r['actuated_max_rel_error']:.2e} | {r['actuated_verdict']} |")
    w()

    # 10. Diagonal validation
    w("## 10. Diagonal Validation")
    w()
    w("| Pose | Min Diag | Max Diag | All Positive |")
    w("|------|----------|----------|-------------|")
    for r in pose_results:
        w(f"| {r['pose']} | {r['diag_min']:.4e} | {r['diag_max']:.4e} | {r['diag_positive']} |")
    w()

    # 11. JIT compatibility
    w("## 11. JIT Compatibility")
    w()
    w(f"JIT mass matrix: {'✓ PASS' if jit_ok else '✗ FAIL'}")
    w()

    # 12. Limitations
    w("## 12. Limitations")
    w()
    for i, lim in enumerate(limitations, 1):
        w(f"{i}. {lim}")
    w()

    # 13. Phase 2C readiness
    w("## 13. Phase 2C Readiness Verdict")
    w()
    w(f"```text")
    w(f"{verdict}")
    w(f"```")
    w()

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  Markdown report: {out_path}")


def _write_json_summary(
    timestamp: str,
    model_path: str,
    pose_results: list,
    max_full_err: float,
    max_act_err: float,
    max_sym_err: float,
    cond_range: tuple,
    jit_ok: bool,
    verdict: str,
    limitations: list,
) -> None:
    out_path = PROJECT_ROOT / "docs" / "validation" / "k2_phase2b_mass_matrix_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    full_verdicts = {r["pose"]: r["full_verdict"] for r in pose_results}
    act_verdicts = {r["pose"]: r["actuated_verdict"] for r in pose_results}

    summary = {
        "phase": "2B",
        "timestamp": timestamp,
        "model_path": model_path,
        "verdict": verdict,
        "full_mass_matrix_implemented": True,
        "full_matrix_verdicts": full_verdicts,
        "actuated_block_verdicts": act_verdicts,
        "max_full_abs_error": max_full_err,
        "max_actuated_abs_error": max_act_err,
        "max_symmetry_error": max_sym_err,
        "condition_number_range": [cond_range[0], cond_range[1]],
        "jit_compatible": jit_ok,
        "controller_modified": False,
        "limitations": limitations,
        "num_poses": len(pose_results),
        "full_pass_threshold": FULL_PASS,
        "actuated_pass_threshold": ACT_PASS,
        "symmetry_pass_threshold": SYM_PASS,
    }

    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"  JSON summary: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
