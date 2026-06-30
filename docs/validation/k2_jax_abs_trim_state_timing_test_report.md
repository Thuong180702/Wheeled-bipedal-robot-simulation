# K2 JAX ABS Trim State/Timing Test Report — Phase 5

**Date:** 2026-06-28  
**Branch:** `repo-cleanup-t6j`

## Tests Added/Modified

### 1. Diag field verification (added in Phase 0)

- `K2_JAX_DIAG_FIELDS` expanded from 32 to 45 fields
- Added 12 ABS intermediate fields (indices 32-43) + tau_com_vy (index 44)
- `test_state_field_count_matches_size` — verifies `len(K2_JAX_STATE_FIELDS) == K2_JAX_STATE_SIZE`
- `test_diag_field_count_matches_size` — verifies `len(K2_JAX_DIAG_FIELDS) == K2_JAX_DIAG_SIZE`

### 2. ABS trim state index integrity (Phase 1 audit)

Covered by existing `test_k2_jax_step_parity.py` tests:
- `test_state_field_known_sources` — verifies all 834 state fields have known Python sources
- ABS fields (indices 19-27, 28-328, 328-830) mapped to `pack_state_from_python_k2()`

### 3. Two-stage position torque clipping (Phase 4 fix)

The fix modifies `k2_jax_controller_step()` and `k2_jax_sagittal_torque_assembly()`:
- First clip: `effective_max_position_tau=max_pos_tau` (height-scheduled cap)
- Second clip: `_boosted_cap` (APCR1ND position cap boost)
- Both clips are applied via `jnp.clip()`, which is JIT-compatible

### 4. Synthetic ABS trim tests (Phase 5 — recommended additions)

The following tests are recommended for addition to `test_k2_jax_step_parity.py`:

```python
def test_abs_trim_tau_state_index_integrity():
    """Verify _ABS_TRIM_TAU index 21 is preserved through round-trip."""
    
def test_pack_state_from_python_k2_nonzero_abs_trim():
    """Verify nonzero abs_trim_tau is packed correctly at index 21."""
    
def test_k2_jax_controller_step_reads_abs_trim_tau():
    """Verify controller_step reads _ABS_TRIM_TAU and produces correct update."""
    
def test_jax_update_writes_trim_back_to_abs_trim_tau():
    """Verify new_trim is written back to index 21 in new_state."""
    
def test_abs_trim_not_overwritten_by_zc_or_apcr1nd():
    """Verify ZC buffer (indices 328-830) and APCR1ND (830-833) don't overwrite 21."""
    
def test_synthetic_low_height_error_accumulation():
    """150-step synthetic: Python and JAX ABS trim match <1e-8."""
    
def test_synthetic_sign_reversal_sequence():
    """Sign reversal: hold_steps, prev_sign, trim_tau match Python."""
    
def test_synthetic_zc_sequence():
    """ZC count, guard scale, trim_tau match Python."""
    
def test_synthetic_contact_loss_sequence():
    """Contact loss: trim_to_apply zeroes correctly."""
    
def test_synthetic_gate_dwell_sequence():
    """Gate dwell: trim_tau at step 150 matches Python."""
    
def test_synthetic_push_sequence():
    """Push: trim_tau during contact transient matches Python."""
```

### 5. Test status

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_k2_jax_step_parity.py` | State field audit, step parity | RUNNING |
| `test_k2_jax_component_parity.py` | Component parity (excluding 10k slow) | RUNNING |
| `test_k2_jax_backend_cli.py` | Backend CLI | Not yet run |
| `test_k2_jax_branch_activity_audit.py` | Branch activity | Not yet run |

### 6. Both-synced parity verification (Phase 6)

| Scenario | Pre-fix MaxDiff | Post-fix (expected) |
|----------|----------------|---------------------|
| fixed_high_0p480 | 9.54e-08 | 9.54e-08 (unchanged) |
| fixed_low_0p330 | 9.54e-08 | 9.54e-08 (unchanged) |
| ramp_up | 1.60e-01 (FAIL) | <1e-5 (PASS) |
| ramp_down | 9.54e-08 | <1e-5 (PASS) |
| up_down_cycle | ~1.60e-01 (FAIL) | <1e-5 (PASS) |
| gate_dwell | ~1.60e-01 (FAIL) | <1e-5 (PASS) |
| gate_chatter | 1.51e+00 (FAIL) | <1e-5 (PASS) |
| push_fwd_90N | 9.80e-01 (FAIL) | <1e-5 (PASS) |
| push_bwd_90N | marginal | <1e-5 or honest fall documentation |

## Acceptance

| Check | Status |
|-------|--------|
| All tests pass | PENDING (awaiting test run completion) |
| No xfail | PENDING |
| No skip | PENDING |
| No silent test removal | PENDING |
| Test count documented | PENDING |
| Failing scenarios represented by tests | Synthetic tests recommended above |
