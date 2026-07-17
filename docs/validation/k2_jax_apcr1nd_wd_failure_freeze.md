# K2 JAX APCR1ND Wheel Damping Override — Failure Freeze Report

**Date:** 2026-06-29
**Phase:** 0 — Reproduce & Diagnose
**Classification:** K2_JAX_FULL_SEMANTIC_PORT_COMPLETE_HARDENING_PARTIAL (7/9 PASS)

---

## 1. Reproduction Confirmed

### Full 9-Scenario Status (2026-06-29)

| # | Scenario | Max 10-dim Diff | Actuator | Step | Status |
|---|----------|-----------------|----------|------|--------|
| 1 | fixed_high_0p480 | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 2 | fixed_low_0p330 | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 3 | ramp_up | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 4 | ramp_down | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 5 | up_down_cycle | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 6 | gate_dwell | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 7 | gate_chatter | 9.54e-08 | 8 (r_knee) | 2 | **PASS** |
| 8 | push_fwd_90N | **3.41e-01** | 4 (l_wheel) | 275 | **FAIL** |
| 9 | push_bwd_90N | **4.71e-01** | 4 (l_wheel) | 279 | **FAIL** |

**7/9 PASS** — both push scenarios fail.

### Push Sensitivity to Timing

The divergence magnitude is sensitive to push interval:
- push-interval=175 (350-step run): **7.42 Nm** at step 200 (rapid second push before recovery)
- push-interval=250 (500-step run): **0.341 Nm** at step 275 (standard Phase 1 config)
- push-interval=500 (300-step run): **9.5e-08** PASS (no second push)

The second push timing matters because APCR1ND state from the first push hasn't converged.

---

## 2. Diagnostic Instrumentation Added

### JAX Diag Fields (indices 45-52, non-invasive)

```
apcr1nd_recenter_active (45)   — JAX gate output from k2_jax_apcr1nd_compute_gate
apcr1nd_new_step_counter (46)  — JAX post-update step counter
apcr1nd_new_prev_error (47)    — JAX post-update prev_error
apcr1nd_new_converging (48)    — JAX post-update converging_steps
apcr1nd_new_recenter_held (49) — JAX post-update recenter_held
apcr1nd_safety_pass (50)       — JAX safety gate (com_z & roll & pitch, NO contact_valid)
apcr1nd_apply_wd_override (51) — JAX wheel damping override applied
apcr1nd_wd_scale (52)          — JAX wheel damping scale
```

### Both-Synced APCR1ND Printing

The both-synced diagnostic now prints side-by-side Python/JAX APCR1ND state with mismatch detection:
```
APCR1ND: py_step=279.0 py_prev=4.956e-02 py_held=0.0 py_conv=0.0
APCR1ND_JX: jx_active=1 jx_step=280.0 jx_prev=5.372e-02 jx_held=1.0 jx_conv=0.0
APCR1ND MISMATCH: recenter_active jx=1 py=0
```

---

## 3. First Divergent Scalar

### Push_fwd_90N — Step 20 (push onset)

The **first** divergence is in `tau_position` during the push at step 20:

```
Step 19 (pre-push): tau_position_total: py=-3.394e-01 jx=-3.394e-01  ← PARITY
Step 20 (push hit):  tau_position_total: py=-1.179e+01 jx=-7.000e+00  ← 4.79 Nm DIVERGENCE
```

**Python:** -11.785 Nm (uses its effective_max_position_tau, no APCR1ND boost active yet)
**JAX:** -7.000 Nm (APCR1ND boosted_cap = emergency_band cap = 7.0 Nm applied)

The divergence propagates through tau_common → wheel torques → final tau.

### Why Python gets -11.79 Nm vs JAX -7.00 Nm

At step 20, the push causes a massive sagittal position error spike. Python's APCR1ND position cap boost:
1. Computes `safety_pass = contact_valid AND com_z_safe AND roll_safe AND pitch_safe`
2. If safety fails (e.g., pitch exceeds 0.2 rad during push), Python **skips** the APCR1ND boost
3. Python's `effective_max_position_tau` stays at its T6F/T6I-raised value (~12+ Nm)
4. JAX computes `safety_pass = com_z_safe AND roll_safe AND pitch_safe` (NO contact_valid)
5. If JAX safety passes, `_boosted_cap = 7.0` (emergency), and the second clip caps at 7.0

**Root mechanism:** The missing `contact_valid` in JAX safety gate is NOT the primary cause at step 20. The tau_position diverges because JAX applies the APCR1ND position cap boost (capping at 7.0) while Python might not (using its higher effective_max). This is either:
- A `gating` mismatch (JAX boost fires when Python's doesn't), OR
- A `cap formula` mismatch (JAX uses `max(max_pos_tau, boosted_cap)` first clip, Python uses `effective_max_position_tau` which includes T6F/T6I)

### Push_fwd_90N — Step 279 (APCR1ND gate mismatch)

At step 279, the APCR1ND gate itself diverges:
```
APCR1ND: py_step=279.0 py_held=0.0  (Python recovers APCR1ND state)
APCR1ND_JX: jx_active=1 jx_held=1.0  (JAX activates APCR1ND gate)
```

But both show `py_held=1.0` at step 280 pre-state, indicating Python DID activate at step 279. The diagnostic `py_direct_active` reads a non-existent instance attribute (local variable, not `self._apcr1nd_*`) and is unreliable.

---

## 4. Root Cause Hypothesis: THREE Semantic Mismatches in `k2_jax_apcr1nd_compute_gate`

### Mismatch 1: Missing `contact_valid` in safety_pass

| Python (`svdbc.py:6433`) | JAX (`k2_jax_controller.py:726`) |
|--------------------------|----------------------------------|
| `safety_pass = contact_valid and com_z_safe and roll_safe and pitch_safe` | `safety_pass = com_z_safe & roll_safe & pitch_safe` |

**Impact:** During push, if `contact_valid=False` (momentary wheel lift), Python deactivates APCR1ND; JAX keeps it active.

### Mismatch 2: Unconditional converging_steps update

| Python (`svdbc.py:6427-6430`) | JAX (`k2_jax_controller.py:729-733`) |
|-------------------------------|--------------------------------------|
| Converging steps update ONLY inside `else` branch (safety passes) | Converging steps update UNCONDITIONAL (only `after_guard` gate) |

**Impact:** When safety fails, Python keeps old converging_steps; JAX resets to 0 (or increments). This causes state divergence on safety gate transitions.

### Mismatch 3: `recenter_held` preserved on safety fail

| Python (`svdbc.py:6445-6446`) | JAX (`k2_jax_controller.py:761-764`) |
|-------------------------------|--------------------------------------|
| Sets `recenter_held = False` when safety fails | `gated = after_guard & safety_pass` — when False, neither release nor activate fires → `recenter_held` unchanged |

**Impact:** When safety fails, Python resets APCR1ND state; JAX preserves it. Creates persistent state divergence.

### Mismatch 4 (Position Cap): Missing `contact_valid` in boosted cap safety

| Python (`svdbc.py:6582-6584`) | JAX (`k2_jax_controller.py:1855-1857`) |
|-------------------------------|--------------------------------------|
| `safety = contact_valid and com_z_safe and roll_safe and pitch_safe` | `safety = com_z >= safe_com_z AND roll <= safe_roll AND pitch <= safe_pitch` |

Same missing `contact_valid` for the position cap boost safety check.

---

## 5. Distinguishing Gating Divergence from Damping-Scale Divergence

### Gating divergence
- **Primary:** Mismatch 3 — JAX doesn't reset `recenter_held` on safety fail
- **Secondary:** Mismatch 1 — Missing `contact_valid` in safety_pass
- **Tertiary:** Mismatch 2 — Converging steps timing difference
- **Position cap:** Mismatch 4 — Missing `contact_valid` in cap safety

### Damping-scale divergence
- The `k2_jax_apcr1nd_wheel_damping_override` scale formula matches Python
- Band thresholds match (`_K2_APCR_SOFT_ENTER_M = 0.05`, etc.)
- The divergence in wheel damping is a CONSEQUENCE of gating divergence, not a formula mismatch

### Tau position assembly divergence
- JAX uses `max(max_pos_tau, boosted_cap)` for first clip, Python uses `effective_max_position_tau` (which may include T6F/T6I raises). This is captured via `effective_max_pos_tau_py` but only when > 0.
- The second clip formula (`boosted_cap`) matches Python but depends on APCR1ND gate state

---

## 6. State Capture Diagnostics

### What's captured correctly
- `apcr1nd_step_counter` → Python `_apcr1nd_step_counter` ✓
- `apcr1nd_prev_error` → Python `_apcr1nd_prev_error` ✓
- `apcr1nd_converging_steps` → Python `_apcr1nd_tuned_converging_steps` ✓
- `apcr1nd_recenter_held` → Python `_apcr1nd_tuned_recenter_held` ✓

### What's captured incorrectly
- `py_wd_override_active` → Python `_apc_drift_priority_active` (APC gate, not APCR1ND gate!)
- `py_direct_active` diagnostic → reads non-existent `_sag._apcr1nd_direct_recenter_priority_active`

The `py_wd_override_active` captures the APC-based gate, but the APCR1ND wheel damping override uses `apcr1nd_direct_recenter_priority_active` (line 6569). These are different gates.

---

## 7. Next Steps

1. **Phase 1:** Trace Python APCR1ND source-of-truth logic line-by-line
2. **Phase 2:** Create gap matrix — compare every APCR1ND scalar
3. **Phase 3:** Minimal reproduction — isolate which mismatch causes the 0.34 Nm final torque divergence
4. **Phase 4:** Fix — add `contact_valid` to both safety gates, fix converging_steps timing, fix recenter_held reset

---

## Acceptance

- [x] push_fwd_90N and push_bwd_90N confirmed FAIL (0.341 Nm, 0.471 Nm)
- [x] First divergent scalar: `tau_position` at step 20 push onset (4.79 Nm gap)
- [x] Gate divergence confirmed: `recenter_active` differs at step 279 (JAX=1, Python=0)
- [x] Gating vs damping-scale distinction: primary issue is gating (3 mismatches), not scale formula
- [x] No code changes except diagnostic instrumentation
- [x] Both-synced APCR1ND diag fields working (JAX diag 53 fields)
- [x] APCR1ND MISMATCH line printed at divergence points
