# Phase 2C.3 — Floating-Base Force Projection Audit Report

**Timestamp:** 2026-07-02T12:24:50.546493+00:00

## 1. Executive Summary

Phase 2C.3 adds a free-base gyroscopic correction to the body-local Featherstone RNEA.  The correction removes the spurious ω×v cross-term from qfrc_bias[0:6] that MuJoCo's free-joint generalised-force projection excludes.

**Phase 2C:** 21 PASS / 0 WARN / 14 FAIL (max full=6.25e-01, max act=5.53e-02)
**Phase 2C.1:** 21 PASS / 0 WARN / 14 FAIL (max full=1.92, max act=0.078)
**Phase 2C.2:** 21 PASS / 0 WARN / 14 FAIL (max full=1.38, max act=0.063)
**Phase 2C.3:** 21 PASS / 7 WARN / 7 FAIL (max full=0.062, max FB force=9.4e-06, max FB torque=0.062, max act=0.058)

**Verdict: `PARTIAL_READY`**

### Key improvements over Phase 2C.2:
- Free-base force error: 1.38 → 9.4e-06 (147,000× reduction, PASS)
- Free-base torque error: 0.43 → 0.062 (7× reduction)
- Max full bias error: 1.38 → 0.062 (22× reduction)
- Small random cases: FAIL → WARN (all 7 cases)
- ω×v cross-term: FAIL → PASS (free-base force)

## 2. Controller Integrity

Controller code and `K2_JAX_DEDICATED_DEFAULT_V3` parameters were **not** modified.

## 3. Changed Files

| File | Status |
|------|--------|
| `wheeled_biped/dynamics/jax_bias_forces.py` | **modified** — free-base projection correction (Phase 2C.3) |
| `scripts/phase2c3_free_base_projection_audit.py` | **new** — audit script |
| `scripts/phase2c3_diagnostic_probe.py` | **new** — diagnostic probe |
| `scripts/phase2c3_root_cause_isolation.py` | **new** — root cause isolation |
| `scripts/phase2c3_mdot_analysis.py` | **new** — M-dot analysis |
| `scripts/phase2c3_corrected_cpu_test.py` | **new** — CPU convention test |
| `scripts/phase2c3_mjc_nonidentity.py` | **new** — non-identity CPU test |
| `tests/test_phase2c3_free_base_projection.py` | **new** — 32 tests |
| `docs/validation/k2_phase2c3_free_base_projection_audit.md` | **new** — this report |
| `docs/validation/k2_phase2c3_free_base_projection_audit.json` | **new** — JSON summary |

## 4. MuJoCo Free-Joint Convention Findings

### Diagnostic probes confirmed:

```text
qvel[0:3]  = base linear velocity of body origin (world frame)
qvel[3:6]  = base angular velocity (world frame)
qfrc_bias[0:3] = force on free-base translation DOFs (world frame)
qfrc_bias[3:6] = torque on free-base rotation DOFs (world frame)
qfrc_bias[6:16] = actuated joint generalised forces
```

### Key empirical finding:

MuJoCo's free-joint velocity-dependent generalised force is **additive** across base velocity DOFs: the ω×v cross-term is structurally zero at machine precision for ALL tested orientations and ALL 9 angular × linear velocity pairs.

The body-local Featherstone RNEA computes the full spatial Coriolis wrench at the torso body origin, which includes the gyroscopic force ω × (m v) that MuJoCo excludes from the free-joint generalised force.  This gyroscopic term is physically absorbed into the mass-matrix coupling M[0:3, 3:6] rather than appearing in qfrc_bias.

## 5. Free-Base Correction Method

### Force correction
```text
f_corr = m_total * omega_world x v_lin_world
qfrc_bias[0:3] -= f_corr
```

### Torque correction
```text
tau_corr = -M_cross^T @ (v_lin_world x omega_world)
qfrc_bias[3:6] -= tau_corr
```

where M_cross = M[0:3, 3:6] (3×3 mass-matrix coupling block) is precomputed at identity orientation and rotated to the current torso orientation at runtime.

The correction is applied AFTER the RNEA backward pass, at the projection step where F_torso (body-local spatial force) is mapped to MuJoCo qfrc[0:6].

## 6. Constants Summary

- Constants version: `phase2c3_free_base_projection`
- Total system mass: 8.1000 kg
- Total COM (body-local): [-0.0033, -0.0508, 0.0818]
- M_cross_world_identity: precomputed (3×3)
- Gravity: [0, 0, -9.81]

## 7. Gravity-Only Validation

**Result: 133/133 PASS**, max abs error = 6.16e-06

Gravity passes at all 7 poses (keyframe, low_height, mid_height, high_height, random_1/2/3) with identity base orientation.

## 8. Full Bias Validation (original 35 cases)

Thresholds: PASS < 1e-3, WARN < 1e-2, FAIL ≥ 1e-2

| Velocity Case | Cases | Max Err | FB Force | FB Torque | Act Err | Verdicts |
|---------------|-------|---------|----------|-----------|---------|----------|
| base_yaw_rate | 7 | 6.16e-06 | 6.16e-06 | 4.88e-07 | 3.80e-07 | PPPPPPP |
| moderate_random | 7 | 6.24e-02 | 9.40e-06 | 6.24e-02 | 5.75e-02 | FFFFFFF |
| small_random | 7 | 3.83e-03 | 9.93e-06 | 3.83e-03 | 3.46e-03 | WWWWWWW |
| symmetric_wheels | 7 | 6.16e-06 | 6.16e-06 | 3.93e-07 | 3.98e-07 | PPPPPPP |
| zero | 7 | 6.16e-06 | 6.16e-06 | 3.93e-07 | 3.98e-07 | PPPPPPP |

**Comparison with Phase 2C.2:**
- Small random: FAIL → WARN (improved!)
- Moderate random max error: 1.38 → 0.062 (22× reduction)
- FB force error eliminated (PASS for all cases)

## 9. Free-Base Force Validation

**Result: 133/133 PASS**, max abs error = 9.40e-06 N

The gyroscopic force correction m_total · ω × v completely eliminates the free-base force cross-term that dominated Phase 2C.2 errors.

## 10. Free-Base Torque Validation

Max abs error = 0.062 Nm (FAIL at strict 1e-3 threshold)

The torque error is dominated by non-identity poses (low_height, mid_height, high_height with moderate_random velocities).  At identity orientation (keyframe), torque error is < 4e-7 (PASS).

The residual torque error comes from two sources:
1. M_cross changing with joint positions (knee bend etc.) — the precomputed identity M_cross becomes approximate
2. Pre-existing centrifugal force error in the body-local RNEA at non-identity base orientations (affects pure angular velocity cases)

## 11. Actuated Bias Validation

Max abs error = 0.058 Nm (FAIL at strict 1e-3 threshold)

Actuated bias error is unchanged from Phase 2C.2 (~0.063).  It arises from mixed free-base + actuated velocity coupling not addressed by the free-base correction.

## 12. Cross-Term Validation

Free-base angular × linear cross-term: **PASS** (was FAIL in Phase 2C.2)
Base angular × actuated cross-term: **PASS** (unchanged)
Base linear × actuated cross-term: **PASS** (unchanged)
Actuated × actuated cross-term: **PASS** (unchanged)

## 13. JIT Compatibility

JIT bias forces: PASS
JIT gravity: PASS
JIT full bias matches no-JIT: PASS

## 14. Tests

```text
32 passed in tests/test_phase2c3_free_base_projection.py
```

Test coverage:
- Import tests: 2
- Constants/version tests: 5
- Gravity tests: 3
- Bias force tests (zero vel, base yaw, wheels, actuated, pure v, pure w): 10
- Free-base correction tests: 3
- Motion subspace tests: 2
- JIT tests: 3
- Controller integrity: 1
- Non-identity orientation: 3

## 15. Limitations

1. **M_cross depends on joint positions.**  The torque correction uses M_cross precomputed at identity orientation.  When joint positions change (e.g., knee bent), the mass-matrix coupling M[0:3, 3:6] shifts.  Computing M_cross at runtime from the full mass matrix would eliminate this residual error.

2. **Non-identity base orientation centrifugal error.**  The body-local RNEA has a pre-existing error in the centrifugal force computation when the base is tilted (roll/pitch ≠ 0).  This affects pure angular velocity cases at non-identity orientations and is not caused by Phase 2C.3.

3. **Actuated bias residual.**  The actuated joint bias has residual errors (max ~0.06 Nm) from mixed free-base and actuated velocity cases.  These arise from the same root coupling mechanism that the free-base correction addresses for the free-base DOFs.

4. **Joint friction/damping/armature** are handled by MuJoCo internally and are not part of qfrc_bias.

## 16. Phase 2D Readiness Verdict

```text
PARTIAL_READY
```

### Criteria met:
- [x] Full 16-vector bias(q,qvel) implemented
- [x] Gravity-only PASS at all poses (with identity base orientation)
- [x] Free-base force PASS for all original 35 cases
- [x] Free-base torque PASS at identity orientation
- [x] Free-base ω×v cross-term PASS
- [x] All entries finite
- [x] JIT compatibility confirmed
- [x] Controller files unchanged
- [x] No hidden CPU MuJoCo calls

### Criteria NOT met:
- [ ] Full bias PASS for all 35 cases (7 FAIL at moderate_random)
- [ ] Free-base torque PASS for all cases (FAIL at non-identity poses)
- [ ] Actuated bias PASS for all cases (FAIL at moderate_random)
- [ ] Max full bias error < 1e-3 (currently 0.062)
- [ ] Max actuated bias error < 1e-3 (currently 0.058)

**Recommendation:** Do NOT proceed to Phase 2D.  Address the following:
1. Compute M_cross at runtime from the FK chain to fix torque error at non-identity joint positions
2. Investigate and fix the pre-existing centrifugal force error at non-identity base orientations in the body-local RNEA
3. Address the actuated bias residual from mixed velocity coupling
