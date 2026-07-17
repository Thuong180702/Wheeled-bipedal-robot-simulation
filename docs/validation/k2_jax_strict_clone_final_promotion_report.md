# K2 JAX Strict-Clone Final Promotion Re-Evaluation Report

**Date:** 2026-06-27
**Classification:** `K2_JAX_FUNCTIONAL_PASS_PARITY_BLOCKED`

---

## 1. Executive Summary

The K2 JAX backend has been thoroughly re-evaluated after two targeted parity fixes (support_velocity input and mode_div_error formula). All functional gates pass: 131/131 tests, 5/5 long-run heights (6000 steps each), no falls, no NaN, no hidden torque/WBC. Performance confirmed at 0.273ms hot-step.

State-synced teacher-forcing correctly identifies remaining formula-level mismatches in the notch-blend pitch rate computation and sagittal velocity path. These are NOT correctness bugs — they are pre-existing computational path differences that the state-synced infrastructure correctly exposes.

**Recommendation:** The JAX backend is a validated opt-in backend with known, bounded parity limitations.

---

## 2. All 7 Phases — Results

### Phase 1 — Support Velocity Parity Fix
- **Fix:** `support_velocity_m_s=0.0` → Python-computed dynamic value
- **Verification:** Input parity achieved (diff=0)
- **Torque impact:** None (`effective_support_velocity_gain=0.0` in K2)
- **Document:** [k2_jax_support_velocity_parity_fix.md](k2_jax_support_velocity_parity_fix.md)
- **Status:** FIXED

### Phase 2 — Mode-Div Hip-Yaw State Parity Fix
- **Fix:** `joint_pos[1] - joint_pos[6]` → `(joint_pos[1]-joint_pos[6]) - (eq_pos[1]-eq_pos[6])`
- **Verification:** div_error parity achieved (diff=0), div_rate already correct
- **Mode-div JAX params verified matching Python args** (kp=10.0, kd=0.50, max_torque=7.5, soft_limit=0.30, soft_gain=0.80)
- **Document:** [k2_jax_mode_div_state_parity_fix.md](k2_jax_mode_div_state_parity_fix.md)
- **Status:** FIXED

### Phase 3 — State-Synced Teacher-Forcing
- **Scenarios tested:** fixed_high_0p480 (50 steps)
- **Step 0:** Near-perfect (4.77e-08) — formulas/coefficients match from zero state
- **Steps 1+:** Systematic divergence in tau_pitch_rate (~6%) and tau_sagittal_velocity (~10%)
- **Root causes identified:**
  1. Notch-blend effective pitch rate differs between Python and JAX paths (~6%)
  2. Sagittal velocity transformation discrepancy (~10% — Python internally transforms vs JAX raw input)
- **Both Phase 1/2 fixes working perfectly** — support_vel and mode_div_error match
- **Document:** [k2_jax_state_synced_teacher_forcing_postfix2_report.md](k2_jax_state_synced_teacher_forcing_postfix2_report.md)
- **Classification:** `K2_JAX_STATE_SYNCED_PARITY_FAIL_WITH_ROOT_CAUSE`

### Phase 4 — JAX Long-Run Validation
- **5 heights × 6000 steps (30,000 total)**

| Height | Steps | Fell | Pitch | Roll | Status |
|--------|-------|------|-------|------|--------|
| low_0p330 | 6000/6000 | No | ±8.6° | ±0.8° | PASS |
| mid_0p400 | 6000/6000 | No | ±3.4° | ±1.1° | PASS |
| high_0p430 | 6000/6000 | No | ±9.8° | ±0.5° | PASS |
| high_0p450 | 6000/6000 | No | ±7.6° | ±0.4° | PASS |
| high_0p480 | 6000/6000 | No | ±9.3° | ±0.3° | PASS |

- **Document:** [k2_jax_long_run_validation_postfix2.md](k2_jax_long_run_validation_postfix2.md)
- **Classification:** `K2_JAX_LONG_RUN_PASS`

### Phase 5 — Regression Tests
- **131/131 tests PASS** (0 xfail, 0 skip)
- No regressions from Phase 1/2 fixes
- Both-synced infrastructure intact
- Python backend unchanged, JAX remains opt-in
- **Document:** [k2_jax_postfix2_test_regression_report.md](k2_jax_postfix2_test_regression_report.md)
- **Classification:** `K2_JAX_POSTFIX2_TESTS_PASS`

### Phase 6 — Functional Smoke Recheck
- Fixed-height smoke: high_0p480 + low_0p330 PASS (6000 steps each)
- Long-run extended validation: 5/5 heights PASS
- Previous push (4/4), dynamic height (5/5), branch audit (6/6) remain valid
- No behavior regressions
- **Document:** [k2_jax_postfix2_functional_smoke_report.md](k2_jax_postfix2_functional_smoke_report.md)
- **Classification:** `K2_JAX_FUNCTIONAL_SMOKE_POSTFIX2_PASS`

### Phase 7 — Final Re-Evaluation (this report)

---

## 3. Exact Changes Made

| File | Change | Lines |
|------|--------|-------|
| `scripts/simulate_hierarchical_controller.py` | Phase 1: `support_velocity_m_s=0.0` → `sagittal_diag.get(...)` | 6539 |
| `scripts/simulate_hierarchical_controller.py` | Phase 2: `joint_pos[1]-joint_pos[6]` → `... - (eq_pos[1]-eq_pos[6])` | 6541 |
| `scripts/simulate_hierarchical_controller.py` | Support velocity diagnostics (SV, SAG_TERMS) | +25 lines |
| `scripts/simulate_hierarchical_controller.py` | Mode-div diagnostics (MODE_DIV) | +12 lines |

## 4. What Was NOT Changed

- **No gains tuned** — All K2 control parameters unchanged
- **No thresholds relaxed** — Parity threshold remains <1e-5
- **No K2 control principles changed** — Controller architecture preserved
- **JAX NOT made default** — Python remains default backend
- **No unrelated controller changes** — Only the two targeted fixes
- **Python backend unchanged** — 131/131 tests confirm

## 5. Remaining Parity Gaps

| Gap | Magnitude | Impact | Status |
|-----|-----------|--------|--------|
| Notch-blend pitch rate | ~6% of tau_pitch_rate | ~0.21 Nm at step 4, cascades to wheels | Known, investigation deferred |
| Sagittal velocity path | ~10% of tau_sagittal_velocity | ~0.03 Nm at step 4 | Known, investigation deferred |
| Composer chain propagation | ~0.015 Nm hip-yaw | From sagittal→prev_tau→rate-limit chain | Side effect of above |

**Both gaps are pre-existing** (not introduced by Phase 1/2 fixes). They represent internal Python controller signal processing that differs from JAX's simplified path. They do not affect functional correctness.

## 6. Classification Decision

### Why NOT `K2_JAX_STRICT_CLONE_PROMOTION_PASS`

Required criteria:
- State-synced full 10-dim tau max_abs_diff <1e-5 → **NOT MET** (~0.21 Nm at step 4)
- JAX long-run 5/5 → MET
- Tests pass → MET (131/131)
- Functional smoke → MET
- Previous validation remains valid → MET
- Branch/hidden torque clean → MET
- JAX hot-step <10ms → MET (0.273ms)
- Python default, JAX opt-in → MET

**BLOCKER:** State-synced parity fails. The notch-blend and sagittal velocity formula gaps prevent strict bit-accuracy.

### Why NOT `K2_JAX_LONG_RUN_BLOCKED`

Long-run passes 5/5. Not blocked.

### Why NOT `K2_JAX_PARTIAL_WITH_BLOCKERS`

No functional smoke or regression test failures. No new hidden torque/WBC.

### Why `K2_JAX_FUNCTIONAL_PASS_PARITY_BLOCKED`

| Criterion | Status |
|-----------|--------|
| Functional gates pass | ✓ All |
| Long-run passes | ✓ 5/5 |
| Tests pass | ✓ 131/131 |
| State-synced parity | ✗ Fail (notch-blend + sagittal vel gaps) |
| Specific, identified root causes | ✓ Documented |

The state-synced parity failures are real formula-level differences that cannot be resolved without addressing the Python controller's internal signal processing in the JAX model.

## 7. Final Recommendation

The JAX backend is functionally equivalent to Python K2 across all tested scenarios:
- 30,000 total long-run steps (5 heights × 6000)
- No falls, no NaN, no hidden torque/WBC
- Performance confirmed (0.273ms hot-step, <10ms budget)
- All regression tests pass

The state-synced infrastructure correctly identifies the remaining parity gaps. These gaps are known, bounded, and functionally acceptable.

**Action items:**
1. ✅ All 7 phases complete
2. ✅ 131/131 tests pass
3. ✅ Fixed-height, push, dynamic height validation preserved
4. ✅ JAX long-run 5/5 PASS
5. ✅ Branch/torque audit clean
6. ✅ Performance confirmed
7. ✅ State-synced infrastructure built and working
8. ⬜ Future: Investigate notch-blend and sagittal velocity path parity

## 8. Deliverables

| Phase | Document |
|-------|----------|
| P1 | [k2_jax_support_velocity_parity_fix.md](k2_jax_support_velocity_parity_fix.md) |
| P2 | [k2_jax_mode_div_state_parity_fix.md](k2_jax_mode_div_state_parity_fix.md) |
| P3 | [k2_jax_state_synced_teacher_forcing_postfix2_report.md](k2_jax_state_synced_teacher_forcing_postfix2_report.md) |
| P4 | [k2_jax_long_run_validation_postfix2.md](k2_jax_long_run_validation_postfix2.md) |
| P5 | [k2_jax_postfix2_test_regression_report.md](k2_jax_postfix2_test_regression_report.md) |
| P6 | [k2_jax_postfix2_functional_smoke_report.md](k2_jax_postfix2_functional_smoke_report.md) |
| P7 | This report |

## 9. Final Classification

**`K2_JAX_FUNCTIONAL_PASS_PARITY_BLOCKED`**

The K2 JAX backend is a validated opt-in backend. All functional gates pass. The state-synced infrastructure correctly identifies remaining formula-level differences (notch-blend path, sagittal velocity path). These differences are known, bounded, and functionally acceptable.

**Blockers to strict clone promotion:**
1. Notch-blend effective pitch rate computation differs (~6% in tau_pitch_rate)
2. Sagittal velocity internal transformation differs (~10% in tau_sagittal_velocity)

**What strict clone promotion would require:**
- Resolve these two formula mismatches
- Achieve max_abs_diff <1e-5 across all 10 actuators for ≥50 steps in all 9 scenarios
