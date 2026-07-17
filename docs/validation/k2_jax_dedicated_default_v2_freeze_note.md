# K2 JAX Dedicated Default V2 — Freeze Note

**Date:** 2026-06-30
**Promoted from:** `DRIFT_ITER2_VEL_ONLY_WIDE_GATE` (Iteration 2, Variant A)
**Replaces:** `K2_JAX_DEDICATED_DEFAULT_V1`
**Rollback:** `K2_JAX_DEDICATED_DEFAULT_V1` (preserved as historical baseline)

---

## What Changed

`K2_JAX_DEDICATED_DEFAULT_V2` adds a **velocity-only drift damping controller** on top of `K2_JAX_DEDICATED_DEFAULT_V1`.

All other V1 behavior — posture PD, hip-yaw divergence, pitch damping boost, support feedforward, wheel notch filter, sagittal authority scheduling — is **unchanged**.

### Diff from V1

| Parameter | V1 (old) | V2 (new) |
|---|---|---|
| `enable_drift_controller` | `False` | **`True`** |
| `drift_k_vel` | — | **10.0** Nm/(m/s) |
| `drift_k_pos` | — | **0.0** (disabled) |
| `drift_k_heading` | — | **0.0** (disabled) |
| `drift_k_heading_rate` | — | **0.0** (disabled) |
| `drift_max_tau` | — | **8.0** Nm per-wheel |
| `drift_push_damp_mult` | — | **1.5** |
| `drift_hgate_low` | — | **0.03** m/s |
| `drift_hgate_high` | — | **0.15** m/s |
| `drift_pgate_low` | — | **0.15** m |
| `drift_pgate_high` | — | **0.80** m |

---

## Why Promoted

DEFAULT_V2 is promoted because it **slows fixed-height drift** while preserving DEFAULT_V1-level safety and behavior:

- **Fixed-height displacement reduction:**
  - low_0p320: ~47% reduction (0.085 m → 0.045 m at 60 s)
  - high_0p430: ~75% reduction (0.185 m → 0.046 m at 60 s)

- **Safety invariants preserved:**
  - 0 falls across 39 scenarios (full step C/D/E/dynamic/long-run matrix)
  - 0 SAFETY_FAIL classifications
  - Step D push recovery: **12/12 WITHIN_OLD_TOLERANCE**

- **Behavior equivalent within tolerance:**
  - Aggregate quality score: 0.6935 (V1) → 0.6936 (V2), Δ = +0.0001
  - Posture: +0.0004 (no meaningful change)
  - Leg Health: +0.0022 (slight improvement)
  - Torque Quality: ±0.0000 (identical)
  - Robustness: -0.0001 (identical within noise)

- **Performance:** 94.7 Hz (min), well above 50 Hz floor

- **0 major regressions**

---

## Known Limitations

DEFAULT_V2 is **NOT** a complete heading/dynamic-height drift fix:

1. **No heading/yaw correction.** Wheel-differential heading correction was tested (Iteration 2, variants B-E) and caused falls at low height. k_heading is set to 0.0 for safety. Yaw drift is unaddressed.

2. **No position return.** k_pos is set to 0.0. Velocity damping slows drift velocity but does not prevent long-term displacement accumulation.

3. **Dynamic-height transit suppression.** The height gate (smoothstep on CoM z-velocity) closes during intentional height transitions, fully suppressing drift control. This is by design — drift torque fights the balance controller during rapid CoM motion — but means V2 provides no drift benefit during height ramps.

4. **Push drift decay is unchanged.** Push torque (10+ Nm) overwhelms drift velocity damping (max ~1.6 Nm). Push recovery behavior is identical to V1.

5. **Support/Drift dimension slightly worse.** -0.0021 vs V1. The drift controller's wheel torque slightly alters support center dynamics.

---

## Implementation Details

### Profile chain
```
K1_PITCH_RATE_NOTCH
  → K2_NOTCH_LOW_Q_V1 (wip_notch + APCR1ND hold)
    → K2_JAX_DEDICATED_DEFAULT_V1 (pitch_damping_boost)
      → K2_JAX_DEDICATED_DEFAULT_V2 (velocity-only drift damping)  ← NEW
```

### Runner changes
- Default `--profile` changed from `k2_jax_dedicated_default_v1` → `k2_jax_dedicated_default_v2`
- `K2_JAX_DEDICATED_DEFAULT_V1` available as `--profile K2_JAX_DEDICATED_DEFAULT_V1`
- Strict profile lookup: unknown profile names error out with available-profiles list (no silent fallback)
- Startup log prints drift controller status: enabled/disabled, k_vel, k_pos, k_heading, k_heading_rate, max_tau, hgate/pgate thresholds

### Rollback
```bash
# Revert to V1 (no drift controller):
python scripts/run_k2_jax_realtime.py --profile K2_JAX_DEDICATED_DEFAULT_V1 ...
```

---

## Validation Artifacts

| Artifact | Path |
|---|---|
| Quality JSON (V2) | `docs/validation/k2_default_v2_quality.json` |
| Quality report (V2) | `docs/validation/k2_default_v2_quality.md` |
| Evaluation vs V1 | `docs/validation/k2_default_v2_evaluation.json` |
| Evaluation report | `docs/validation/k2_default_v2_evaluation.md` |
| Full validation output | `outputs/validation/k2_default_v2/` |
| V1 baseline quality | `docs/validation/k2_default_v1_quality.json` |

### Test suite
- 382 passed, 4 pre-existing failures (unrelated `test_balance_core_structural_invariants` CSV format mismatches in test fixtures)
- Profile import, parameter pack/unpack, and drift controller telemetry verified

---

## Next Development

Future work should branch from DEFAULT_V2 and target:

1. **Heading/yaw drift** — investigate yaw estimation quality in centroidal state estimator; characterize wheel asymmetry in sagittal velocity damping; evaluate hip-yaw joint heading correction (indices [1, 6]) as alternative to wheel differential.

2. **Dynamic-height drift** — split height gate so velocity damping persists during slow height motion; only fast transitions should fully suppress drift control.

3. **Position return** — explore late-activation position containment with very low gain (k_pos ~ 0.3-0.5) and very wide gate (pgate_low ~ 0.8 m) to avoid destabilization.

4. **Push drift decay** — investigate push-state-aware drift damping multiplier that decays with post-push settling time.

**Do not re-enable wheel-differential heading correction (k_heading > 0) without first fixing the yaw estimation or coupling path.** Variants B-E at k_heading=5.0 caused low-height falls; higher heading gains are unsafe.
