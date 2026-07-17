# K2 JAX Post-Fix Dynamic Height Validation

**Date:** 2026-06-27
**Classification:** `K2_JAX_DYNAMIC_HEIGHT_VALIDATION_PASS`

---

## 1. Summary

Post-fix JAX backend dynamic height gate-crossing validation: **5/5 PASS** (pre-fix: 0/5).

The D1 (notch coefficient) and D12 (calibrated outer loop v1→v2) fixes directly resolved the dynamic gate-crossing failures. The notch height gate (0.42-0.48m smoothstep) now transitions correctly between active/inactive states, and the calibrated outer loop uses correct v2 gains during height transitions.

---

## 2. Results

All scenarios run via `validate_k2_dynamic_height_gate_crossing.py --profile k2 --controller-backend jax`.

| Scenario | Steps | fell | hip_yaw_abs_max (rad) | pitch RMS (deg) | Duration |
|----------|-------|------|----------------------|-----------------|----------|
| ramp_up (0.33→0.48m) | 5000 | False | 0.0534 | 3.15 | 1654s |
| ramp_down (0.48→0.33m) | 5000 | False | 0.0977 | 5.84 | 999s |
| up_down_cycle (0.33→0.48→0.33m) | 7000 | False | 0.0534 | 3.32 | 1103s |
| gate_dwell (0.42/0.45/0.48m) | 6000 | False | 0.0534 | 3.05 | 916s |
| gate_chatter (0.40-0.47m) | 5000 | False | 0.0629 | 2.98 | 618s |

---

## 3. Comparison: Pre-Fix vs Post-Fix

| Scenario | Pre-Fix | Post-Fix | Improvement |
|----------|---------|----------|-------------|
| ramp_up | FAIL (height_too_low) | PASS | ✓ |
| ramp_down | FAIL (height_too_low) | PASS | ✓ |
| up_down_cycle | FAIL (height_too_low) | PASS | ✓ |
| gate_dwell | FAIL (height_too_low) | PASS | ✓ |
| gate_chatter | FAIL (height_too_low) | PASS | ✓ |

**Root cause of pre-fix failures:** D1 (notch coefficient mismatch at 0.42-0.48m gate) caused incorrect pitch_rate damping during gate transitions, and D12 (v1 outer loop gains at 0.48m: Kp=1.575 vs correct v2 Kp=1.050) caused excessive outer-loop response near the upper gate boundary.

---

## 4. Safety Verification

| Metric | Value | Limit | Status |
|--------|-------|-------|--------|
| hip_yaw_abs_max | 0.0977 rad (max across all) | 0.30 rad | ✓ |
| Pitch RMS max | 5.84 deg | — | Acceptable |
| Roll (all scenarios) | Within limits | 5° | ✓ |
| NaN count | 0 | 0 | ✓ |
| Actuator limit violations | 0 | 0 | ✓ |
| Hidden torque | 0 | 0 | ✓ (Phase 7) |
| WBC leakage | 0 | 0 | ✓ (Phase 7) |

---

## 5. Classification

**`K2_JAX_DYNAMIC_HEIGHT_VALIDATION_PASS`**

All 5 dynamic height gate-crossing scenarios pass with JAX backend. No falls, no NaN, no safety violations. The D1/D12 fixes directly resolved the pre-fix dynamic failures. JAX backend is functionally equivalent to Python K2 for dynamic height transitions.
