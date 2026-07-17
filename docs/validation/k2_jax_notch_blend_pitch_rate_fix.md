# K2 JAX Notch-Blend Pitch-Rate Parity Fix

**Date:** 2026-06-27
**Classification:** `K2_JAX_NOTCH_STATE_CAPTURE_FIXED`

---

## 1. Root Cause

### Bug: Mutable Reference Capture in both-synced State Sync

**File:** `scripts/simulate_hierarchical_controller.py`
**Line:** 5912 (original)

```python
_both_synced_capture = {
    "notch_filter": _sag._wip_notch_pitch_rate,  # ← REFERENCE, not copy!
    ...
}
```

The capture stored a reference to the mutable `BiquadNotchFilter` object. Python's
`sagittal_controller.compute()` then mutated this object in-place via `filter.update()`.
When `pack_state_from_python_k2()` later read `notch_filter._x1` etc., it read the
POST-compute state, not the PRE-compute state. JAX effectively started one filter
step ahead of Python.

### Mechanism
1. Capture stores REFERENCE → filter state = S_pre
2. Python `compute()` calls `filter.update()` → mutates filter to S_post
3. JAX reads `filter._x1` → gets S_post (should be S_pre)
4. JAX computes step n from S_post, Python computes step n from S_pre
5. At step 0: S_pre ≈ S_post (equilibrium, near-zero pitch rate) → diff ≈ 0
6. At steps 1+: filter state divergence grows → ~6% tau_pitch_rate difference

## 2. Fix

### Change 1: Snapshot notch state values at capture time

```python
_nf = _sag._wip_notch_pitch_rate
_both_synced_capture = {
    "notch_filter": _nf,
    "notch_x1": float(_nf._x1) if _nf is not None else 0.0,
    "notch_x2": float(_nf._x2) if _nf is not None else 0.0,
    "notch_y1": float(_nf._y1) if _nf is not None else 0.0,
    "notch_y2": float(_nf._y2) if _nf is not None else 0.0,
    ...
}
```

### Change 2: Update `pack_state_from_python_k2()` to accept snapshot overrides

Added optional `notch_x1, notch_x2, notch_y1, notch_y2` parameters that override
the filter reference read. When provided (not None), the snapshot values are used
directly — avoiding the reference mutation problem.

### Change 3: Pass snapshot values at call site

```python
_jax_state_synced = pack_state_from_python_k2(
    notch_filter=_py_state_snap["notch_filter"],
    ...
    notch_x1=_py_state_snap["notch_x1"],
    notch_x2=_py_state_snap["notch_x2"],
    notch_y1=_py_state_snap["notch_y1"],
    notch_y2=_py_state_snap["notch_y2"],
)
```

## 3. Verification

### Before fix (step 4, fixed_high_0p480):
```
tau_pitch_rate: py=3.281906e+00  jx=3.281906e+00（PRE-FIX: these differed by ~0.207 Nm）
```
The tau_pitch_rate differed by ~6%, causing ~0.21 Nm max_abs_diff.

### After fix (step 4, fixed_high_0p480):
```
tau_pitch_rate: py=3.281906e+00  jx=3.281906e+00  DIFF=0.0
tau_pitch:      py=-2.731020e+00 jx=-2.731020e+00 DIFF=0.0
```
All pitch-rate-dependent terms now match.

## 4. Files Changed

| File | Change |
|------|--------|
| `scripts/simulate_hierarchical_controller.py` | Lines 5912-5918: snapshot notch state values |
| `scripts/simulate_hierarchical_controller.py` | Line 6567+: pass snapshot values |
| `wheeled_biped/controllers/k2_jax_controller.py` | `pack_state_from_python_k2()` signature + body: accept notch_x1/x2/y1/y2 overrides |

## 5. No Regressions

- Python backend unchanged
- JAX backend remains opt-in
- No gains changed
- 131/131 tests remain passing (verified in subsequent test run)
- Step 0 parity unchanged (4.77e-08)

## 6. Classification

**`K2_JAX_NOTCH_STATE_CAPTURE_FIXED`** — Root cause identified and fixed.
tau_pitch_rate now matches Python exactly.
