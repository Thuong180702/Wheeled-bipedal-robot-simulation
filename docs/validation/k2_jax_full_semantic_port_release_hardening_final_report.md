# K2 JAX Full Semantic Port — Release Hardening Final Report

**Date:** 2026-06-29
**Classification:** **K2_JAX_FULL_SEMANTIC_PORT_RELEASE_HARDENING_PASS**

---

## 1. Final 9-Scenario Both-Synced Parity

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

**9/9 PASS** — All scenarios below 1e-5 threshold.

---

## 2. Root Cause: 6 Semantic Mismatches

### 2.1 Missing contact_valid in APCR1ND safety gate
Python: safety_pass = contact_valid AND com_z_safe AND roll_safe AND pitch_safe
JAX (before): safety_pass = com_z_safe AND roll_safe AND pitch_safe

### 2.2 Converging steps updated outside safety gate
Python: Only updates when safety_pass=True
JAX (before): Unconditional update

### 2.3 recenter_held not reset on safety fail
Python: Resets to False when safety fails
JAX (before): Preserved previous value

### 2.4 Position cap boost unconditional
Python: Only applies inside APCR1N gate
JAX (before): Second clip applied unconditionally

### 2.5 Missing contact_valid in position cap safety
Same pattern as 2.1, separate instance

### 2.6 JAX contact_valid input definition mismatch (CRITICAL)
- Python: left_wheel_contact AND right_wheel_contact AND contact_force_valid
- JAX (before): contact_force_valid only
- Impact: During push at step 270, wheel lift causes contact_valid=False in Python but True in JAX → APCR1ND gate diverges → 0.341 Nm torque mismatch

---

## 3. Test Results
- 147/147 tests PASS (125 original + 22 new APCR1ND tests)
- No regressions, no xfail, no skip

---

## 4. Non-Negotiable Checks
- 9/9 both-synced parity PASS
- push_fwd_90N: 9.54e-08 PASS
- push_bwd_90N: 4.70e-07 PASS
- No hidden torque/WBC/NaN
- No gain tuning, no threshold relaxation
- Python unchanged, Python default, JAX opt-in

---

## 5. Verdict
**Classification: K2_JAX_FULL_SEMANTIC_PORT_RELEASE_HARDENING_PASS**
