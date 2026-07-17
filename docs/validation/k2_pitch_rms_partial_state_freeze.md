# K2 JAX Dedicated — Pitch RMS Partial State Freeze

**Date:** 2026-06-30
**Purpose:** Freeze current PARTIAL state before pitch RMS investigation begins

---

## 1. Current State

| Field | Value |
|-------|-------|
| **Commit** | `0e1c7135e22b4cb852f71a795426cd3d3f19753a` |
| **Branch** | `repo-cleanup-t6j` |
| **Commit message** | Stage 6K: Dynamic runner extended, JAX ramp_up terminates at step 556/5000 |
| **Current classification** | `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL` |
| **Final report** | `docs/validation/k2_jax_dedicated_full_original_k2_promotion_pass_report.md` |

---

## 2. Four Step E SAFE_BUT_WORSE Pitch Cases

All from validation run in `outputs/k2_jax_dedicated_promotion_validation/`.

Pitch RMS tolerance rule: `min(1.0°, 0.3 × original)` — from `k2_original_metrics.json` tolerances section.

| # | Scenario | Original Pitch RMS (°) | Dedicated Pitch RMS (°) | Delta (°) | Tolerance (°) | Status |
|---|----------|----------------------|------------------------|-----------|----------------|--------|
| 1 | `low_0p320` | 2.83 | 3.69 | +0.86 | 0.849 | SAFE_BUT_WORSE |
| 2 | `low_0p360` | 1.90 | 3.12 | +1.22 | 0.570 | SAFE_BUT_WORSE |
| 3 | `low_0p380` | 3.33 | 5.24 | +1.91 | 0.999 | SAFE_BUT_WORSE |
| 4 | `high_0p450` | 2.75 | 4.68 | +1.93 | 0.825 | SAFE_BUT_WORSE |

All 4 cases show 0.86–1.93° higher pitch RMS in the dedicated JAX runner vs original Python K2.

---

## 3. Passing Step E Cases (Controls)

| # | Scenario | Original (°) | Dedicated (°) | Delta (°) | Tolerance (°) | Status |
|---|----------|-------------|---------------|-----------|----------------|--------|
| 1 | `low_0p300` | 2.68 | 2.91 | +0.23 | 0.804 | WITHIN_OLD_TOLERANCE |
| 2 | `low_0p330` | 3.63 | 3.96 | +0.33 | 1.000 | WITHIN_OLD_TOLERANCE |
| 3 | `low_0p340` | 2.97 | 1.86 | −1.11 | 0.891 | EXACT_OR_BETTER |
| 4 | `high_0p430` | 4.98 | 3.13 | −1.85 | 1.000 | EXACT_OR_BETTER |
| 5 | `high_0p465` | 3.55 | 3.62 | +0.07 | 1.000 | WITHIN_OLD_TOLERANCE |
| 6 | `high_0p480` | 3.96 | 4.28 | +0.32 | 1.000 | WITHIN_OLD_TOLERANCE |

Pattern: 4 failures are at low_0p320, low_0p360, low_0p380, high_0p450. The adjacent heights (low_0p330, low_0p340, high_0p430, high_0p465, high_0p480) all pass.

---

## 4. Tolerance Analysis

The `pitch_rms_deg` tolerance from `k2_original_metrics.json`:

```json
"pitch_rms_deg": {
    "absolute": 1.0,
    "relative": 0.3,
    "rule": "min(absolute, relative * original)"
}
```

For each failing case, the delta exceeds the tolerance by:

| Scenario | Tolerance | Delta | Excess |
|----------|-----------|-------|--------|
| low_0p320 | 0.849° | 0.864° | 0.015° (marginal — 1.7% over) |
| low_0p360 | 0.570° | 1.218° | 0.648° (significant — 114% over) |
| low_0p380 | 0.999° | 1.915° | 0.916° (significant — 92% over) |
| high_0p450 | 0.825° | 1.934° | 1.109° (significant — 134% over) |

`low_0p320` is borderline (0.015° excess). The other three have substantial deltas.

---

## 5. Constraints (Non-Negotiable)

- Do NOT relax pitch RMS tolerance (absolute 1.0° or relative 0.3×).
- Do NOT tune gains blindly.
- Do NOT change K2 profile parameters unless original K2 used exact same values.
- Do NOT change physics timestep, model, or termination logic.
- Every patch must map to a first divergent scalar or a verified metric extraction bug.

---

## 6. Output Directories

| Scope | Output Dir |
|-------|-----------|
| Step E | `outputs/k2_jax_dedicated_promotion_validation/step_e/` |
| Step C | `outputs/k2_jax_dedicated_promotion_validation/step_c/` |
| Step D | `outputs/k2_jax_dedicated_promotion_validation/step_d/` |
| Dynamic | `outputs/k2_jax_dedicated_promotion_validation/dynamic_height/` |
| Long-run | `outputs/k2_jax_dedicated_promotion_validation/long_run/` |
| Baseline | `outputs/k2_original_promoted_baseline/k2_original_metrics.json` |
| All metrics | `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json` |

---

## 7. Commands to Reproduce

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

### Step E only:
```bash
python scripts/validate_k2_jax_dedicated_promotion.py \
  --scope step_e \
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

## 8. Acceptance

- [x] Current pitch-only PARTIAL state is documented.
- [x] 4 SAFE_BUT_WORSE cases identified with exact values, deltas, and tolerances.
- [x] Output directories and reproduction commands documented.
- [x] No code changes made in this phase.
- [x] Tolerance analysis shows low_0p320 is borderline (0.015° excess); other three are substantial.
