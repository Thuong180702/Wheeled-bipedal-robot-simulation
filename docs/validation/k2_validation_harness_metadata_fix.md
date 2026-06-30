# K2 Validation Harness Metadata Fix Report

**Date:** 2026-06-29
**Phase:** 8 — UPDATE VALIDATION HARNESS AND BASELINE METADATA
**Status:** ✅ COMPLETE

---

## 1. Changes Applied

### 1.1 Baseline JSON metadata

**File:** `outputs/k2_original_promoted_baseline/k2_original_metrics.json`

Added:

| Metadata Field | Scope | Content |
|---|---|---|
| `step_d.metric_window` | Step D | Post-push window definition (steps 305-805, 500 steps) |
| `step_d.source_backend` | Step D | `"python"` |
| `step_e.source_backend` | Step E | `"python"` |
| `step_c.source_backend` | Step C | `"python"` |
| `dynamic_height.source_backend` | Dynamic | `"jax_monolithic"` |
| `dynamic_height.q_ref_semantics` | Dynamic | Static q_ref + scenario-appropriate usage note |
| `meta.hip_yaw_metric_definition` | Global | Canonical definition: `max(|l_hip_yaw_pos|, |r_hip_yaw_pos|)` |
| `meta.corrections_applied` | Global | Log of all Phase 1-7 fixes |

### 1.2 Scenario specs file

**File:** `outputs/k2_original_promoted_baseline/scenario_specs.json` (NEW)

Documents:
- Source backend per scope
- Metric window definitions per scope
- Hip-yaw metric canonical definition
- Push timing for Step D
- Global metric definitions with dedicated runner field mappings

### 1.3 Baseline metadata validator

**File:** `scripts/validate_k2_jax_dedicated_promotion.py`

Added `validate_baseline_metadata()` function that checks:
- Step D `_hip_yaw_correction` metadata present
- Step D `metric_window` metadata present
- Source backend per scope
- Hip-yaw metric definition in meta

Runs automatically before any validation scenarios.

---

## 2. Verification

```
python -c "from scripts.validate_k2_jax_dedicated_promotion import validate_baseline_metadata; ..."
All baseline metadata checks passed
```

---

## 3. Acceptance Criteria

| Criterion | Status |
|---|---|
| Classifier has access to metric window metadata | ✅ Via baseline JSON `step_d.metric_window` |
| Source backend documented per scope | ✅ In baseline JSON and scenario_specs.json |
| Step D hip-yaw baseline verified or marked corrected | ✅ `_hip_yaw_correction` metadata on all 12 scenarios |
| Scenario specs document source-of-truth | ✅ `scenario_specs.json` created |
| Baseline metadata validation runs before classification | ✅ Auto-check in `_run()` |
| Dynamic q_ref semantics documented | ✅ In baseline and scenario_specs |
