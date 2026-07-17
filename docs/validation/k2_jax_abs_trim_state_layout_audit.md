# K2 JAX ABS Trim State Layout Audit — Phase 1

**Date:** 2026-06-28
**Branch:** `repo-cleanup-t6j`

## Audit Results

### 1. Field Count Match
- `len(K2_JAX_STATE_FIELDS)` = 834 ✓
- `K2_JAX_STATE_SIZE` = 834 ✓
- **PASS** — exact match

### 2. Index Constant → Field Name Verification

All 25 index constants correspond to correct field names in `K2_JAX_STATE_FIELDS`:

| Constant | Index | Field Name | Status |
|----------|-------|-----------|--------|
| `_S_NOTCH_X1` | 0 | `notch_x1` | PASS |
| `_S_NOTCH_X2` | 1 | `notch_x2` | PASS |
| `_S_NOTCH_Y1` | 2 | `notch_y1` | PASS |
| `_S_NOTCH_Y2` | 3 | `notch_y2` | PASS |
| `_S_PREV_TAU_START` | 4 | `prev_tau_0` | PASS |
| `_S_FILTERED_COM_Z` | 14 | `filtered_com_z` | PASS |
| `_S_PREV_SUPPORT_ERROR` | 15 | `prev_support_error` | PASS |
| `_S_OL_PITCH_REF_SMOOTHED` | 16 | `outer_loop_pitch_ref_smoothed_deg` | PASS (naming convention) |
| `_S_OL_PREV_SUPPORT_ERROR` | 17 | `outer_loop_prev_support_error_m` | PASS (naming convention) |
| `_S_OL_SUPPORT_ERROR_RATE` | 18 | `outer_loop_support_error_rate_smoothed` | PASS (naming convention) |
| `_ABS_SLOW_SUM` | 19 | `abs_slow_sum` | PASS |
| `_ABS_FAST_SUM` | 20 | `abs_fast_sum` | PASS |
| `_ABS_TRIM_TAU` | 21 | `abs_trim_tau` | PASS |
| `_ABS_HOLD_STEPS` | 22 | `abs_hold_steps` | PASS |
| `_ABS_PREV_ERR_SIGN` | 23 | `abs_prev_err_sign` | PASS |
| `_ABS_ZC_COUNT` | 24 | `abs_zc_count` | PASS |
| `_ABS_SLOW_COUNT` | 25 | `abs_slow_count` | PASS |
| `_ABS_SLOW_PTR` | 26 | `abs_slow_ptr` | PASS |
| `_ABS_GUARD_TRIGGER` | 27 | `abs_guard_trigger` | PASS |
| `_ABS_ZC_BUF_COUNT` | 328 | `abs_zc_buf_count` | PASS |
| `_ABS_ZC_BUF_PTR` | 329 | `abs_zc_buf_ptr` | PASS |
| `_S_APCR1ND_STEP_COUNTER` | 830 | `apcr1nd_step_counter` | PASS |
| `_S_APCR1ND_PREV_ERROR` | 831 | `apcr1nd_prev_error` | PASS |
| `_S_APCR1ND_CONVERGING_STEPS` | 832 | `apcr1nd_tuned_converging_steps` | PASS |
| `_S_APCR1ND_RECENTER_HELD` | 833 | `apcr1nd_tuned_recenter_held` | PASS |

**PASS** — all index constants verified. Three outer-loop constants have slightly different names than their constant identifiers, but the semantic mapping is correct (verified by both-synced trace showing matching values).

### 3. Range Non-Overlap Check

| Subsystem | Index Range | Size |
|-----------|------------|------|
| Notch filter | [0, 4) | 4 |
| Previous torque | [4, 14) | 10 |
| Filtered CoM Z | [14, 15) | 1 |
| Previous support error | [15, 16) | 1 |
| Outer loop | [16, 19) | 3 |
| ABS core fields | [19, 28) | 9 |
| ABS slow ring buffer | [28, 328) | 300 |
| ABS ZC header | [328, 330) | 2 |
| ABS ZC ring buffer | [330, 830) | 500 |
| APCR1ND gating | [830, 834) | 4 |

**PASS** — No overlapping ranges. Total: 4+10+1+1+3+9+300+2+500+4 = 834 ✓

### 4. Full Coverage
All 834 indices assigned, no gaps, no extra indices.

### 5. Write-Back Order — No Overwrites

The `k2_jax_controller_step` function writes to:
1. Notch state (indices 0-3) — via notch update
2. ABS state (indices 19-27, 28-830) — via `_k2_jax_adaptive_bias_trim`
3. Outer loop state (indices 16-18) — via outer loop update
4. APCR1ND state (indices 830-833) — via APCR1ND update

The `_ABS_TRIM_TAU` (index 21) is written ONLY inside `_k2_jax_adaptive_bias_trim`, and no subsequent writes to the new_state overwrite it:
- APCR1ND writes to [830, 834) — does NOT overlap with 21
- Outer loop writes to [16, 18] — does NOT overlap with 21
- Notch writes to [0, 3] — does NOT overlap with 21

**PASS** — no overwrite of `_ABS_TRIM_TAU`.

### 6. Phase 0 Diagnostic Bug Fix

The `simulate_hierarchical_controller.py` both-synced diagnostic was reading `_jax_state[21]` (the zero-initialized incremental state) instead of `_jax_state_synced[21]` (the correctly packed state). This caused the false diagnosis that JAX ABS trim diverges from Python. Fixed in Phase 0.

### 7. Diag Field Audit

`K2_JAX_DIAG_FIELDS` expanded from 32 to 44 fields in Phase 0 to include ABS intermediates. Index constants added:
- `_D_ABS_SLOW_MEAN` = 32 through `_D_ABS_HOLD_STEPS` = 43

**PASS** — diag size and fields consistent.

## Acceptance

| Check | Status |
|-------|--------|
| `K2_JAX_STATE_FIELDS` length equals `K2_JAX_STATE_SIZE` | ✓ PASS |
| Every index constant points to intended field name | ✓ PASS |
| `_ABS_TRIM_TAU` index still points to trim value after ZC/APCR1ND shifts | ✓ PASS (index 21, field "abs_trim_tau") |
| `pack_state_k2()` writes `_ABS_TRIM_TAU` at same index that step reads | ✓ PASS (both use constant `_ABS_TRIM_TAU=21`) |
| `pack_state_from_python_k2()` writes at same index | ✓ PASS (verified by `_jax_state_synced[21]` trace) |
| JAX `new_state` writes updated trim back to same index | ✓ PASS (verified by `_jax_new_state[21]` trace) |
| No later `.at[...]` write overwrites `_ABS_TRIM_TAU` with zero | ✓ PASS (APCR1ND writes [830,834)) |
| APCR1ND shifted fields do not overlap ABS fields | ✓ PASS |
| ZC buffer fields do not overlap ABS trim/slow buffer fields | ✓ PASS |
| Diag fields not confused with state fields | ✓ PASS |

## Classification

**K2_JAX_ABS_TRIM_STATE_LAYOUT_CORRECT** — No overlapping indices, no stale index constants, no wrong field ordering. State layout is verified correct for all 834 fields.
