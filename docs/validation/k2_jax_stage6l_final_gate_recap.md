# Stage 6L: Final Gate Recap — JAX K2 Backend Validation

**Date:** 2026-06-27
**Classification:** `STAGE6L_PASS_READY_FOR_STAGE7`

## Root Cause Analysis

The JAX backend failure at 500+ steps was caused by three independent issues:

### 1. Adaptive Bias Trim (ABS): EMA vs Sliding Window Mean
- **Root cause:** JAX ABS used exponential moving average (EMA, α=2/301) while Python uses true sliding window arithmetic mean over 300 samples. EMA converges more slowly to sign changes, causing ABS trim to lag.
- **Fix:** Replaced EMA with a 300-element ring buffer maintaining running sum. True sliding window mean now matches Python exactly.

### 2. Torque Limit Mismatch  
- **Root cause:** JAX params used uniform torque limit of 57 Nm for all 10 joints instead of per-joint limits.
- **Fix:** Pass per-joint torque_limit array to JAX params.

### 3. Hardcoded K2 Gain Mismatches
- **Root cause:** JAX hardcoded `k_position=0.0`, `k_velocity=0.0`, fixed `max_position_tau=3.0` — Python uses 40.0, 15.0, and scheduled 4.0→6.0 respectively.
- **Fix:** Aligned all gains to K2 profile defaults, implemented continuous max_position_tau scheduling.

## Validation Results

### Fixed Height
| Step | Count | Result |
|------|-------|--------|
| Step C | 7 heights | **7/7 PASS** ✅ |
| Step E | 10 heights | **10/10 PASS** ✅ |
| **Total** | **17** | **17/17 PASS** |

### Push Matrix
| Step | Count | Result |
|------|-------|--------|
| Step D | 3 heights × 2 dirs | **6/6 PASS** ✅ |
| Single push | 2 directions | **2/2 PASS** ✅ |
| **Total** | **8** | **8/8 PASS** |

### Dynamic Height
| Scenario | Steps | fell | pitch RMS | hip_yaw | Result |
|----------|-------|------|-----------|---------|--------|
| ramp_up 0.33→0.48 | 5000 | False | 3.39° | 0.407 | **PASS** ✅ |
| ramp_down 0.48→0.33 | 5000 | False | 7.52° | 0.406 | **PASS** ✅ |
| up_down_cycle | 7000 | False | 4.28° | 0.409 | **PASS** ✅ |
| gate_dwell | 6000 | False | 2.89° | 0.418 | **PASS** ✅ |
| gate_chatter | 5000 | False | 6.69° | 0.429 | **PASS** ✅ |
| **Total** | **5** | | | | **5/5 PASS** |

### Tests
| Suite | Count | Result |
|-------|-------|--------|
| Component parity | 94 | 94/94 PASS ✅ |
| Step parity + audit + CLI | 34 | 34/34 PASS ✅ |
| **Total** | **128** | **128/128 PASS** |

### Gate Summary
| Gate | Count | Result |
|------|-------|--------|
| Step C (fixed 7) | 7 | ✅ |
| Step E (fixed 10) | 10 | ✅ |
| Step D (push matrix) | 6 | ✅ |
| Single push | 2 | ✅ |
| Dynamic height | 5 | ✅ |
| Tests | 128 | ✅ |
| **TOTAL** | **158/158** | **ALL PASS** |

## Files Changed

1. `wheeled_biped/controllers/k2_jax_controller.py` — Ring buffer ABS, correct K2 gains, scheduled max_position_tau
2. `scripts/simulate_hierarchical_controller.py` — Per-joint torque limits, `both` debug mode
3. `tests/test_k2_jax_step_parity.py` — Updated state field sources
