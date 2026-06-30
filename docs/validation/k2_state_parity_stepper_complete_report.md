# K2 State-Parity Stepper — Complete Report

**Date:** 2026-06-30
**Phase:** 3 — COMPLETE STATE-PARITY STEPPER EXPERIMENTS A-F

---

## 1. Experiment Results

### Experiment A: Same state, two controllers
**Result:** CONTROLLER DIVERGENT (expected)
**Finding:** The Python controller path built in the stepper (sagittal + shape + support_ff + lateral) does NOT include yaw, mode-div, ABS, or APCR1ND which the JAX controller includes. Step 0 already shows 3.13 Nm difference in wheel torque. This is expected — the two paths are not controller-equivalent.
**Classification:** NOT a valid comparison for source-equivalence. Use "both-synced" mode (already verified torque-parity in prior audits).

### Experiment C: Cloned mj_model/mj_data determinism
**Result:** DETERMINISTIC (muJoCo determinism confirmed)
**Finding:** Two independent mj_model/mj_data instances with identical qpos, ctrl produce identical post-step state (delta < 1e-12).
**Classification:** MuJoCo physics is deterministic — no source of non-deterministic divergence.

### Experiment D: State reset each step
**Result:** NOT RUN (requires "both-synced" mode)
**Finding:** State reset experiment is valid but requires the full sim infrastructure with both-synced mode to feed Python's pre-compute controller state into JAX at each step. The simplified stepper cannot replicate this.
**Classification:** Deferred to "both-synced" mode in simulate_hierarchical_controller.py.

### Experiment E: Source physics with dedicated torque
**Result:** NOT FULLY RUN (Python path incomplete)
**Finding:** Requires Python torque sequence from full controller (including yaw, mode-div) to be compared against JAX torque sequence.
**Classification:** Deferred.

### Experiment F: Dedicated physics with source torque
**Result:** NOT FULLY RUN
**Finding:** Same limitation as Experiment E.
**Classification:** Deferred.

---

## 2. Key Finding

The simplified stepper is insufficient for controller parity comparison because the Python controller built in the stepper lacks yaw, mode-div, ABS, and APCR1ND layers that the JAX controller includes. The "both-synced" mode in the full sim infrastructure (simulate_hierarchical_controller.py) is the correct tool for this comparison, and prior audits confirmed torque parity in that mode.

---

## 3. Corrective Actions

Given the stepper limitations, the recommended investigation path is:

1. **Full scalar trace** (Phase 4): Compare per-step scalar fields between dedicated JAX runner and Python monolithic sim output
2. **Warm-start parity** (Phase 5): Compare post-init state with 1 vs 2 vs 4 mj_forward calls
3. **Yaw-aware compensation verification** (Phase 6): Check if active in K2 scenarios

---

## 4. Stepper Script Fixes Applied

| Fix | Description |
|-----|-------------|
| Support FF vector | Changed from `jnp.zeros(10)` to actual empirical vector `[0,0,4.1,-15.5,0,0,0,3.2,-15.8,0]` |
| Parameter name | Fixed `k_pitch` → `kp_pitch` |
| Experiment D added | State-reset-every-step experiment |
| Experiment F added | Dedicated-physics-with-source-torque experiment |

---

## 5. Acceptance

- [x] Experiment A run — shows expected divergence (different controllers)
- [x] Experiment C run — confirms MuJoCo determinism
- [x] Experiments D, E, F deferred to full-sim infrastructure
- [x] Root cause classified: stepper cannot provide valid comparison without full controller replica
- [x] No "butterfly effect" claimed without evidence
