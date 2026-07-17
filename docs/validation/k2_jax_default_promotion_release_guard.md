# K2 JAX Default Promotion — Release Guard

**Date:** 2026-06-29
**Phase:** 5 — Full Release Guard After Default Switch

---

## 1. Full 9-Scenario Both-Synced Parity

Source: `docs/validation/k2_jax_full_semantic_port_release_hardening_final_report.md`

| # | Scenario | Max 10-dim Diff | Status |
|---|----------|-----------------|--------|
| 1 | fixed_high_0p480 | 9.54e-08 | PASS |
| 2 | fixed_low_0p330 | 9.54e-08 | PASS |
| 3 | ramp_up | 9.54e-08 | PASS |
| 4 | ramp_down | 9.54e-08 | PASS |
| 5 | up_down_cycle | 9.54e-08 | PASS |
| 6 | gate_dwell | 9.54e-08 | PASS |
| 7 | gate_chatter | 9.54e-08 | PASS |
| 8 | push_fwd_90N | 9.54e-08 | PASS |
| 9 | push_bwd_90N | 4.70e-07 | PASS |

**9/9 PASS** — All below 1e-5 strict parity threshold.

---

## 2. Implicit-Default Functional Validation

Source: Phase 4 regression validation (this promotion).

| Category | Scenarios | Passed |
|----------|-----------|--------|
| Fixed height | fixed_high_0p480, fixed_low_0p330 | 2/2 |
| Dynamic height | ramp_up | 1/1 |
| Push recovery | push_fwd_90N, push_bwd_90N | 2/2 |
| **Total** | | **5/5** |

All scenarios use implicit JAX default (no `--controller-backend` flag).
No NaN, no hidden torque/WBC, no fall.

---

## 3. Explicit Python Fallback Validation

| Scenario | Status |
|----------|--------|
| fixed_high_0p480 | PASS |
| fixed_low_0p330 | PASS |
| push_fwd_90N | PASS |
| push_bwd_90N | PASS |

**4/4 PASS** — Python fallback fully functional.

---

## 4. Explicit Both-Synced Validation

| Scenario | Max Diff | Status |
|----------|----------|--------|
| fixed_high_0p480 | 9.54e-08 | PASS |
| push_fwd_90N | 9.54e-08 | PASS |
| push_bwd_90N | 4.20e-07 | PASS |

**3/3 PASS** — Both-synced parity maintained.

---

## 5. Test Suite

| Suite | Tests | Result |
|-------|-------|--------|
| test_k2_jax_backend_cli.py | 20 | 20 PASS |
| test_k2_jax_component_parity.py | 85 | 85 PASS |
| test_k2_jax_step_parity.py | 55 | 55 PASS |
| test_stage1_behavior_unchanged.py | 4 | 4 PASS |
| **Total** | **164** | **164 PASS** |

---

## 6. Performance Sanity

From prior validation:
- Hot-step timing: ~0.185 ms (well below 10 ms target)
- JIT compilation: ~1.4 s (one-time cost)
- No repeated recompilation detected

---

## 7. Non-Negotiable Gates

| Gate | Status |
|------|--------|
| 9/9 both-synced parity | PASS |
| Implicit-default functional validation | PASS |
| Explicit Python fallback | PASS |
| Tests pass | PASS |
| Performance < 10 ms | PASS |
| No hidden torque/WBC | PASS |
| No NaN | PASS |
| No unexpected safety violation | PASS |
| Python backend preserved | YES |
| Unvalidated profiles NOT promoted | YES |

---

## 8. Verdict

**All release guard gates pass.** Ready for final promotion classification.
