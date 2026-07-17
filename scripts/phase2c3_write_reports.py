#!/usr/bin/env python
"""Write Phase 2C.3 audit reports from already-collected validation data."""
from __future__ import annotations
import json, sys
from pathlib import Path
from datetime import datetime, timezone

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TIMESTAMP = datetime.now(timezone.utc).isoformat()

RESULTS = {
    "phase": "2C.3",
    "verdict": "PARTIAL_READY",
    "constants_version": "phase2c3_free_base_projection",
    "num_original_cases": 35,
    "gravity_pass_warn_fail": {"PASS": 133, "WARN": 0, "FAIL": 0},
    "full_bias_pass_warn_fail": {"PASS": 21, "WARN": 7, "FAIL": 7},
    "free_base_force_pass_warn_fail": {"PASS": 133, "WARN": 0, "FAIL": 0},
    "free_base_torque_pass_warn_fail": {"PASS": 119, "WARN": 7, "FAIL": 7},
    "actuated_bias_pass_warn_fail": {"PASS": 21, "WARN": 7, "FAIL": 7},
    "velocity_bias_pass_warn_fail": {"PASS": 112, "WARN": 7, "FAIL": 7},
    "cross_term_pass_warn_fail": {"note": "free-base w x v: PASS (fixed in 2C.3), other pairs: PASS"},
    "base_orientation_pass_warn_fail": {"note": "PASS at identity, WARN/FAIL at non-identity (pre-existing RNEA issue)"},
    "max_gravity_abs_error": 6.16e-06,
    "max_full_bias_abs_error": 0.062,
    "max_free_base_force_abs_error": 9.40e-06,
    "max_free_base_torque_abs_error": 0.062,
    "max_actuated_bias_abs_error": 0.058,
    "max_velocity_bias_abs_error": 0.062,
    "max_cross_term_abs_error": 0.0,
    "max_base_orientation_abs_error": 0.324,
    "jit_compatible": True,
    "controller_modified": False,
    "free_joint_convention": {
        "qvel_frame": "world",
        "qvel_order": "[v_lin; omega]",
        "qfrc_frame": "world",
        "qfrc_order": "[force; torque]",
        "root_force_origin": "body_origin",
        "projection": "S_free^T @ F_spatial_root with gyroscopic correction",
    },
    "root_force_projection": {
        "method": "body_local_rnea_with_free_base_correction",
        "force_correction": "m_total * omega x v_lin",
        "torque_correction": "-M_cross^T @ (v_lin x omega)",
        "M_cross_source": "precomputed_at_identity",
    },
    "phase2c_reference": {
        "full_bias": "21 PASS / 0 WARN / 14 FAIL",
        "max_full_err": 6.25e-01, "max_act_err": 5.53e-02,
    },
    "phase2c1_reference": {
        "full_bias": "21 PASS / 0 WARN / 14 FAIL",
        "max_full_err": 1.92, "max_act_err": 0.078,
    },
    "phase2c2_reference": {
        "full_bias": "21 PASS / 0 WARN / 14 FAIL",
        "max_full_err": 1.38, "max_act_err": 0.0629,
    },
    "remaining_issues": [
        "FB torque error ~0.06 at non-identity poses (M_cross approximation with joint position changes)",
        "Actuated bias residual ~0.06 from mixed base+actuated velocity cases",
        "Body-local RNEA centrifugal force error at non-identity base orientations (pre-existing)",
        "7 FAIL, 7 WARN in original 35-case matrix (improved from 14 FAIL in 2C.2)",
    ],
    "limitations": [
        "M_cross precomputed at identity; degrades with joint position changes",
        "Actuated joint bias has residual errors from mixed velocity coupling",
        "Non-identity base orientation: pre-existing centrifugal force error in body-local RNEA",
        "Joint friction/damping/armature not included",
    ],
}

# Write JSON
json_path = PROJECT_ROOT / "docs" / "validation" / "k2_phase2c3_free_base_projection_audit.json"
json_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(RESULTS, indent=2, default=str))
print(f"JSON: {json_path}")

# Write Markdown
md_path = PROJECT_ROOT / "docs" / "validation" / "k2_phase2c3_free_base_projection_audit.md"
lines = []
def w(s=""): lines.append(s)

w("# Phase 2C.3 — Floating-Base Force Projection Audit Report")
w()
w(f"**Timestamp:** {TIMESTAMP}")
w()

w("## 1. Executive Summary")
w()
w("Phase 2C.3 adds a free-base gyroscopic correction to the body-local "
  "Featherstone RNEA.  The correction removes the spurious ω×v cross-term "
  "from qfrc_bias[0:6] that MuJoCo's free-joint generalised-force projection "
  "excludes.")
w()
w("**Phase 2C:** 21 PASS / 0 WARN / 14 FAIL (max full=6.25e-01, max act=5.53e-02)")
w("**Phase 2C.1:** 21 PASS / 0 WARN / 14 FAIL (max full=1.92, max act=0.078)")
w("**Phase 2C.2:** 21 PASS / 0 WARN / 14 FAIL (max full=1.38, max act=0.063)")
w("**Phase 2C.3:** 21 PASS / 7 WARN / 7 FAIL (max full=0.062, max FB force=9.4e-06, max FB torque=0.062, max act=0.058)")
w()
w("**Verdict: `PARTIAL_READY`**")
w()

w("### Key improvements over Phase 2C.2:")
w("- Free-base force error: 1.38 → 9.4e-06 (147,000× reduction, PASS)")
w("- Free-base torque error: 0.43 → 0.062 (7× reduction)")
w("- Max full bias error: 1.38 → 0.062 (22× reduction)")
w("- Small random cases: FAIL → WARN (all 7 cases)")
w("- ω×v cross-term: FAIL → PASS (free-base force)")
w()

w("## 2. Controller Integrity")
w()
w("Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.")
w()

w("## 3. Changed Files")
w()
w("| File | Status |")
w("|------|--------|")
w("| `wheeled_biped/dynamics/jax_bias_forces.py` | **modified** — free-base projection correction (Phase 2C.3) |")
w("| `scripts/phase2c3_free_base_projection_audit.py` | **new** — audit script |")
w("| `scripts/phase2c3_diagnostic_probe.py` | **new** — diagnostic probe |")
w("| `scripts/phase2c3_root_cause_isolation.py` | **new** — root cause isolation |")
w("| `scripts/phase2c3_mdot_analysis.py` | **new** — M-dot analysis |")
w("| `scripts/phase2c3_corrected_cpu_test.py` | **new** — CPU convention test |")
w("| `scripts/phase2c3_mjc_nonidentity.py` | **new** — non-identity CPU test |")
w("| `tests/test_phase2c3_free_base_projection.py` | **new** — 32 tests |")
w("| `docs/validation/k2_phase2c3_free_base_projection_audit.md` | **new** — this report |")
w("| `docs/validation/k2_phase2c3_free_base_projection_audit.json` | **new** — JSON summary |")
w()

w("## 4. MuJoCo Free-Joint Convention Findings")
w()
w("### Diagnostic probes confirmed:")
w()
w("```text")
w("qvel[0:3]  = base linear velocity of body origin (world frame)")
w("qvel[3:6]  = base angular velocity (world frame)")
w("qfrc_bias[0:3] = force on free-base translation DOFs (world frame)")
w("qfrc_bias[3:6] = torque on free-base rotation DOFs (world frame)")
w("qfrc_bias[6:16] = actuated joint generalised forces")
w("```")
w()
w("### Key empirical finding:")
w()
w("MuJoCo's free-joint velocity-dependent generalised force is **additive** "
  "across base velocity DOFs: the ω×v cross-term is structurally zero at "
  "machine precision for ALL tested orientations and ALL 9 angular × linear "
  "velocity pairs.")
w()
w("The body-local Featherstone RNEA computes the full spatial Coriolis "
  "wrench at the torso body origin, which includes the gyroscopic force "
  "ω × (m v) that MuJoCo excludes from the free-joint generalised force.  "
  "This gyroscopic term is physically absorbed into the mass-matrix "
  "coupling M[0:3, 3:6] rather than appearing in qfrc_bias.")
w()

w("## 5. Free-Base Correction Method")
w()
w("### Force correction")
w("```text")
w("f_corr = m_total * omega_world x v_lin_world")
w("qfrc_bias[0:3] -= f_corr")
w("```")
w()

w("### Torque correction")
w("```text")
w("tau_corr = -M_cross^T @ (v_lin_world x omega_world)")
w("qfrc_bias[3:6] -= tau_corr")
w("```")
w()
w("where M_cross = M[0:3, 3:6] (3×3 mass-matrix coupling block) is precomputed "
  "at identity orientation and rotated to the current torso orientation at runtime.")
w()
w("The correction is applied AFTER the RNEA backward pass, at the projection "
  "step where F_torso (body-local spatial force) is mapped to MuJoCo qfrc[0:6].")
w()

w("## 6. Constants Summary")
w()
w("- Constants version: `phase2c3_free_base_projection`")
w("- Total system mass: 8.1000 kg")
w("- Total COM (body-local): [-0.0033, -0.0508, 0.0818]")
w("- M_cross_world_identity: precomputed (3×3)")
w("- Gravity: [0, 0, -9.81]")
w()

w("## 7. Gravity-Only Validation")
w()
w("**Result: 133/133 PASS**, max abs error = 6.16e-06")
w()
w("Gravity passes at all 7 poses (keyframe, low_height, mid_height, "
  "high_height, random_1/2/3) with identity base orientation.")
w()

w("## 8. Full Bias Validation (original 35 cases)")
w()
w("Thresholds: PASS < 1e-3, WARN < 1e-2, FAIL ≥ 1e-2")
w()
w("| Velocity Case | Cases | Max Err | FB Force | FB Torque | Act Err | Verdicts |")
w("|---------------|-------|---------|----------|-----------|---------|----------|")
w("| base_yaw_rate | 7 | 6.16e-06 | 6.16e-06 | 4.88e-07 | 3.80e-07 | PPPPPPP |")
w("| moderate_random | 7 | 6.24e-02 | 9.40e-06 | 6.24e-02 | 5.75e-02 | FFFFFFF |")
w("| small_random | 7 | 3.83e-03 | 9.93e-06 | 3.83e-03 | 3.46e-03 | WWWWWWW |")
w("| symmetric_wheels | 7 | 6.16e-06 | 6.16e-06 | 3.93e-07 | 3.98e-07 | PPPPPPP |")
w("| zero | 7 | 6.16e-06 | 6.16e-06 | 3.93e-07 | 3.98e-07 | PPPPPPP |")
w()
w("**Comparison with Phase 2C.2:**")
w("- Small random: FAIL → WARN (improved!)")
w("- Moderate random max error: 1.38 → 0.062 (22× reduction)")
w("- FB force error eliminated (PASS for all cases)")
w()

w("## 9. Free-Base Force Validation")
w()
w("**Result: 133/133 PASS**, max abs error = 9.40e-06 N")
w()
w("The gyroscopic force correction m_total · ω × v completely eliminates "
  "the free-base force cross-term that dominated Phase 2C.2 errors.")
w()

w("## 10. Free-Base Torque Validation")
w()
w("Max abs error = 0.062 Nm (FAIL at strict 1e-3 threshold)")
w()
w("The torque error is dominated by non-identity poses (low_height, "
  "mid_height, high_height with moderate_random velocities).  At identity "
  "orientation (keyframe), torque error is < 4e-7 (PASS).")
w()
w("The residual torque error comes from two sources:")
w("1. M_cross changing with joint positions (knee bend etc.) — the "
  "precomputed identity M_cross becomes approximate")
w("2. Pre-existing centrifugal force error in the body-local RNEA at "
  "non-identity base orientations (affects pure angular velocity cases)")
w()

w("## 11. Actuated Bias Validation")
w()
w("Max abs error = 0.058 Nm (FAIL at strict 1e-3 threshold)")
w()
w("Actuated bias error is unchanged from Phase 2C.2 (~0.063).  It arises "
  "from mixed free-base + actuated velocity coupling not addressed by the "
  "free-base correction.")
w()

w("## 12. Cross-Term Validation")
w()
w("Free-base angular × linear cross-term: **PASS** (was FAIL in Phase 2C.2)")
w("Base angular × actuated cross-term: **PASS** (unchanged)")
w("Base linear × actuated cross-term: **PASS** (unchanged)")
w("Actuated × actuated cross-term: **PASS** (unchanged)")
w()

w("## 13. JIT Compatibility")
w()
w("JIT bias forces: PASS")
w("JIT gravity: PASS")
w("JIT full bias matches no-JIT: PASS")
w()

w("## 14. Tests")
w()
w("```text")
w("32 passed in tests/test_phase2c3_free_base_projection.py")
w("```")
w()
w("Test coverage:")
w("- Import tests: 2")
w("- Constants/version tests: 5")
w("- Gravity tests: 3")
w("- Bias force tests (zero vel, base yaw, wheels, actuated, pure v, pure w): 10")
w("- Free-base correction tests: 3")
w("- Motion subspace tests: 2")
w("- JIT tests: 3")
w("- Controller integrity: 1")
w("- Non-identity orientation: 3")
w()

w("## 15. Limitations")
w()
w("1. **M_cross depends on joint positions.**  The torque correction uses "
  "M_cross precomputed at identity orientation.  When joint positions "
  "change (e.g., knee bent), the mass-matrix coupling M[0:3, 3:6] shifts.  "
  "Computing M_cross at runtime from the full mass matrix would eliminate "
  "this residual error.")
w()
w("2. **Non-identity base orientation centrifugal error.**  The body-local "
  "RNEA has a pre-existing error in the centrifugal force computation when "
  "the base is tilted (roll/pitch ≠ 0).  This affects pure angular velocity "
  "cases at non-identity orientations and is not caused by Phase 2C.3.")
w()
w("3. **Actuated bias residual.**  The actuated joint bias has residual errors "
  "(max ~0.06 Nm) from mixed free-base and actuated velocity cases.  These "
  "arise from the same root coupling mechanism that the free-base correction "
  "addresses for the free-base DOFs.")
w()
w("4. **Joint friction/damping/armature** are handled by MuJoCo internally "
  "and are not part of qfrc_bias.")
w()

w("## 16. Phase 2D Readiness Verdict")
w()
w("```text")
w("PARTIAL_READY")
w("```")
w()
w("### Criteria met:")
w("- [x] Full 16-vector bias(q,qvel) implemented")
w("- [x] Gravity-only PASS at all poses (with identity base orientation)")
w("- [x] Free-base force PASS for all original 35 cases")
w("- [x] Free-base torque PASS at identity orientation")
w("- [x] Free-base ω×v cross-term PASS")
w("- [x] All entries finite")
w("- [x] JIT compatibility confirmed")
w("- [x] Controller files unchanged")
w("- [x] No hidden CPU MuJoCo calls")
w()
w("### Criteria NOT met:")
w("- [ ] Full bias PASS for all 35 cases (7 FAIL at moderate_random)")
w("- [ ] Free-base torque PASS for all cases (FAIL at non-identity poses)")
w("- [ ] Actuated bias PASS for all cases (FAIL at moderate_random)")
w("- [ ] Max full bias error < 1e-3 (currently 0.062)")
w("- [ ] Max actuated bias error < 1e-3 (currently 0.058)")
w()
w("**Recommendation:** Do NOT proceed to Phase 2D.  Address the following:")
w("1. Compute M_cross at runtime from the FK chain to fix torque error at "
  "non-identity joint positions")
w("2. Investigate and fix the pre-existing centrifugal force error at "
  "non-identity base orientations in the body-local RNEA")
w("3. Address the actuated bias residual from mixed velocity coupling")

md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Markdown: {md_path}")
print("Done.")
