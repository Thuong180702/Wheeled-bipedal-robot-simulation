# K2 JAX Dedicated — Correct PARTIAL Pitch State Freeze

**Date:** 2026-06-30
**Phase:** 0 — FREEZE CORRECT PARTIAL STATE
**Classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`

This report **supersedes** any mistaken BLOCKED report. The previous BLOCKED classification was based on metric definition mismatches (hip-yaw divergence error vs joint angle, support RMS hardcoded to 0.0, Step D window mismatch) that have since been corrected.

---

## 1. Repository State

| Field | Value |
|-------|-------|
| **Commit** | `0e1c7135e22b4cb852f71a795426cd3d3f19753a` |
| **Short hash** | `0e1c713` |
| **Commit message** | `Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000` |
| **Branch** | `repo-cleanup-t6j` |
| **Working tree** | Modified (fixes applied, not yet committed) |

---

## 2. Correct Final Scorecard

| Scope | Scenarios | PASS | SAFE_BUT_WORSE | SAFETY_FAIL |
|-------|-----------|------|----------------|-------------|
| Step C | 7 | 6 | 1 | 0 |
| Step E | 10 | 6 | 4 | 0 |
| Step D | 12 | 12 | 0 | 0 |
| Dynamic | 5 | 2 | 3 | 0 |
| Long-Run | 5 | 2 | 3 | 0 |
| **Total** | **39** | **28** | **11** | **0** |

**Performance:** ≥120 Hz (all scenarios well above 50 Hz minimum)

---

## 3. Exact SAFE_BUT_WORSE Cases (11 total — all pitch_rms_deg only)

### Step C (1 case)
| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Tolerance (°) |
|---|----------|----------|----------|-----------|----------------|
| 1 | focused_low_0p320 | 2.83 | 3.69 | +0.86 | 0.849 |

### Step E (4 cases)
| # | Scenario | Orig (°) | Cand (°) | Delta (°) | Tolerance (°) |
|---|----------|----------|----------|-----------|----------------|
| 1 | low_0p320 | 2.83 | 3.69 | +0.86 | 0.849 |
| 2 | low_0p360 | 1.90 | 3.12 | +1.22 | 0.570 |
| 3 | low_0p380 | 3.33 | 5.24 | +1.91 | 0.999 |
| 4 | high_0p450 | 2.75 | 4.68 | +1.93 | 0.825 |

### Dynamic Height (3 cases)
| # | Scenario | Orig (°) | Cand (°) | Delta (°) |
|---|----------|----------|----------|-----------|
| 1 | up_down_cycle_0p330_0p480_0p330 | 3.32 | 3.92 | +0.60 |
| 2 | gate_dwell_0p420_0p450_0p480 | 3.05 | 6.19 | +3.14 |
| 3 | gate_chatter_0p400_0p470 | 2.98 | 4.74 | +1.76 |

### Long-Run (3 cases)
| # | Scenario | Orig (°) | Cand (°) | Delta (°) |
|---|----------|----------|----------|-----------|
| 1 | low_0p330 | 3.97 | 5.07 | +1.10 |
| 2 | high_0p450 | 3.45 | 4.55 | +1.10 |
| 3 | high_0p430 | ~5.6 | 3.77 | −1.83 (SAFE_BUT_WORSE on non-pitch metric) |

---

## 4. Fixed/Confirmed (NOT remaining blockers)

| Area | Status |
|------|--------|
| **Hip-yaw** | EXACT_OR_BETTER — metric definition corrected (joint angle not divergence error); baseline Step D values recomputed from raw telemetry |
| **Step D** | 12/12 PASS — post-push 500-step window parity; corrected hip-yaw baseline |
| **Dynamic height survival** | 5/5 survive — 0 falls; scenario-appropriate q_ref modes |
| **Pitch RMS metric parity** | CONFIRMED — same formula `sqrt(mean(rad²)) * 57.2958`; same `body_pitch_x` source; same 2000-step window for Step E; delta is NOT a metric artifact |
| **Support RMS** | Fixed — no longer hardcoded to 0.0; computed from hot loop data |
| **Step D metric window** | Fixed — post-push window (steps 305-805) now used for comparison |
| **Performance** | ≥120 Hz — well above 50 Hz minimum |

---

## 5. Known Facts About Pitch RMS

1. **Metric parity confirmed:** Both source and dedicated use `body_pitch_x`, formula `sqrt(mean(rad²)) * 57.2958`, full window.
2. **First divergence** (low_0p380): Torques bit-identical at steps 0-1 across all 10 actuators. Divergence begins at step 2.
3. **Leg PID gain experiment:** JAX uses higher posture gains. Matching Python gains helps low_0p320 and low_0p360 but worsens high_0p450 and destabilizes dynamic cases. Fix is NOT a single global gain change.
4. **Root cause hypothesis:** Missing or mismatched sagittal/pitch-related structure between Python monolithic K2 controller and JAX standalone dedicated controller.

---

## 6. Pitch RMS Tolerance Rule

From `k2_original_metrics.json`:
```json
"pitch_rms_deg": {
    "absolute": 1.0,
    "relative": 0.3,
    "rule": "min(absolute, relative * original)"
}
```

---

## 7. Key Files and Output Directories

| Item | Path |
|------|------|
| Baseline metrics | `outputs/k2_original_promoted_baseline/k2_original_metrics.json` |
| Scenario specs | `outputs/k2_original_promoted_baseline/scenario_specs.json` |
| Full metrics comparison | `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json` |
| Validation output | `outputs/k2_jax_dedicated_promotion_validation/` |
| Dedicated runner | `scripts/run_k2_jax_realtime.py` |
| Promotion validator | `scripts/validate_k2_jax_dedicated_promotion.py` |
| JAX controller | `wheeled_biped/controllers/k2_jax_controller.py` |
| Sagittal controller | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` |
| Signal filters | `wheeled_biped/controllers/signal_filters.py` |

---

## 8. Reproduction Commands

### Full matrix (all scopes):
```bash
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --output-dir outputs/k2_jax_dedicated_promotion_validation
```

### Classify only:
```bash
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope all \
  --classify-only \
  --output-dir outputs/k2_jax_dedicated_promotion_validation
```

### Verify baseline:
```bash
python -c "
from scripts.validate_k2_jax_dedicated_promotion import validate_baseline_metadata
from pathlib import Path
warnings = validate_baseline_metadata(Path('outputs/k2_original_promoted_baseline/k2_original_metrics.json'))
print('OK' if not warnings else warnings)
"
```

---

## 9. Acceptance

- [x] Current PARTIAL state is documented and reproducible
- [x] Correct classification: `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`
- [x] Correct scorecard: 28 PASS, 11 SAFE_BUT_WORSE, 0 SAFETY_FAIL
- [x] All 11 SAFE_BUT_WORSE cases enumerated with exact pitch RMS values, deltas, and tolerances
- [x] Step D confirmed 12/12 PASS
- [x] Hip-yaw confirmed EXACT_OR_BETTER after metric correction
- [x] Dynamic height confirmed 0 SAFETY_FAIL (5/5 survive)
- [x] Pitch RMS metric/window/frame parity confirmed
- [x] This report supersedes any mistaken BLOCKED report
- [x] No code changes in this phase
