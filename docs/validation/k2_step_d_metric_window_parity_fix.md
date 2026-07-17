# K2 Step D Metric Window Parity Fix Report

**Date:** 2026-06-29
**Phase:** 1 — FIX STEP D METRIC WINDOW PARITY
**Status:** ✅ COMPLETE

---

## 1. Problem Statement

The dedicated runner's Step D classification was comparing full-episode candidate metrics against post-push 500-step original metrics, resulting in invalid comparisons.

**Original canonical window:**
- Push starts at step 300
- Push duration: 5 steps
- Post-push window: steps 305-805 (500 steps)
- Metrics: `post_pitch_rms_500_deg`, `post_support_rms_500_m`

**Candidate (before fix):**
- `extract_metrics_from_summary()` read full-episode `pitch_rms_deg` from summary
- Passed it to classifier as `post_pitch_rms_500_deg`
- `support_rms_m` was hardcoded to 0.0

---

## 2. Changes Applied

### 2.1 `scripts/run_k2_jax_realtime.py`

**Post-push window detection:**
```python
push_end_step = 0
if push_schedule:
    push_end_step = max(s1 for _, s1, _, _ in push_schedule)
POST_PUSH_WINDOW = 500
```

**Post-push metric tracking (in hot loop):**
```python
if push_end_step > 0 and step >= push_end_step:
    in_post_push = step < push_end_step + POST_PUSH_WINDOW
    if in_post_push and not sm["post_push_active"]:
        sm["post_push_active"] = True
    if sm["post_push_active"] and in_post_push:
        sm["post_pitch_count"] += 1
        sm["post_pitch_sum"] += pitch_x
        sm["post_pitch_sum_sq"] += pitch_x * pitch_x
        sm["post_support_count"] += 1
        support_err = float(support_xy[1]) - float(support_center_eq[1])
        sm["post_support_sum"] += support_err
        sm["post_support_sum_sq"] += support_err * support_err
```

**Post-push RMS computation:**
```python
post_pitch_rms_500_deg = 0.0
post_support_rms_500_m = 0.0
if sm["post_push_active"] and sm["post_pitch_count"] > 0:
    post_pitch_rms_500_deg = (sm["post_pitch_sum_sq"] / sm["post_pitch_count"]) ** 0.5 * 57.2958
if sm["post_push_active"] and sm["post_support_count"] > 0:
    post_support_rms_500_m = (sm["post_support_sum_sq"] / sm["post_support_count"]) ** 0.5
```

**Summary JSON output:**
```python
"post_push_window": {
    "push_end_step": push_end_step,
    "window_start_step": push_end_step,
    "window_end_step": push_end_step + POST_PUSH_WINDOW,
    "window_steps": POST_PUSH_WINDOW,
    "active": sm["post_push_active"],
    "post_pitch_rms_500_deg": round(post_pitch_rms_500_deg, 6),
    "post_support_rms_500_m": round(post_support_rms_500_m, 6),
}
```

### 2.2 `scripts/validate_k2_jax_dedicated_promotion.py`

**Updated `extract_metrics_from_summary()`:**
```python
post_push = s.get("post_push_window", {})
post_pitch_rms_500 = post_push.get("post_pitch_rms_500_deg", 0.0)
post_support_rms_500 = post_push.get("post_support_rms_500_m", 0.0)
return {
    ...
    "post_pitch_rms_500_deg": post_pitch_rms_500,
    "post_support_rms_500_m": post_support_rms_500,
    ...
}
```

**Updated `classify_scope()` Step D section:**
```python
elif scope == "step_d":
    sc = classifier.classify_step_d_condition(sid, {
        "fell": fell, "hip_yaw_max_rad": hy_max,
        "post_pitch_rms_500_deg": r.get("post_pitch_rms_500_deg", 0.0),
        "post_support_rms_500_m": r.get("post_support_rms_500_m", 0.0),
    })
```

---

## 3. Verification

### 3.1 Smoke test: push scenario

```
python scripts/run_k2_jax_realtime.py \
  --height-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --push-seq outputs/k2_step_d_push_matrix_validation/push_sequences/push_sagittal_forward_60N_step300.json \
  --steps 2000 --telemetry summary \
  --output-dir outputs/k2_jax_dedicated_phase1_test/high_0p480_fwd_60N
```

**Result:**
```
Post-push window:
  push_end_step: 305
  window_start_step: 305
  window_end_step: 805
  window_steps: 500
  active: true
  post_pitch_rms_500_deg: 5.805475
  post_support_rms_500_m: 0.104834

Full-episode pitch_rms: 4.8953 deg
Post-push 500-step pitch_rms: 5.8055 deg
```

✅ Post-push metrics are correctly computed from the 500-step window
✅ Post-push pitch RMS differs from full-episode RMS (as expected — push elevates pitch)
✅ Non-push scenarios still work fine (post_push_window.active=false)

### 3.2 Compilation check
```
python -m py_compile scripts/run_k2_jax_realtime.py  → OK
python -m py_compile scripts/validate_k2_jax_dedicated_promotion.py → OK
```

---

## 4. Acceptance Criteria

| Criterion | Status |
|---|---|
| No full-episode candidate RMS compared to post-push original RMS | ✅ The classifier now receives window-specific metrics |
| Step D table includes post-push 500-step candidate values | ✅ `post_push_window` section in summary.json |
| Metric window metadata included | ✅ window_start_step, window_end_step, window_steps |
| Classifier refuses incompatible windows | ⚠️ Not yet — classifier does not check window metadata (Phase 8) |
| Any previous SAFE_BUT_WORSE caused by metric-window mismatch removed | ⚠️ Cannot verify yet — need full rerun (Phase 10) |

---

## 5. Comparison: Original vs Dedicated (example)

**high_0p480 sagittal_forward 60N:**

| Metric | Original (Python K2) | Dedicated (JAX K2) |
|---|---|---|
| post_pitch_rms_500_deg | 0.1376 | 5.8055 |
| post_support_rms_500_m | 0.1125 | 0.1048 |
| hip_yaw_max_rad | 0.0* | 0.0396 |

*Note: Original hip_yaw=0.0 is a baseline artifact (column name mismatch in original summary script). See Phase 2.

The dedicated runner shows significantly higher post-push pitch RMS than the original Python K2 path. This is a real behavioral difference (not a metric window artifact) that will be addressed in Phases 3-5.

---

## 6. Known Limitations

1. **Support RMS:** Currently uses support center Y deviation from equilibrium as a proxy for `support_position_error_m`. The original uses a different definition (possibly from the sagittal controller). Phase 6 will address this.
2. **Classifier window awareness:** The classifier does not yet verify metric window compatibility — this is deferred to Phase 8.
3. **Support error definition:** The dedicated runner's support error may differ in definition from the original's `support_position_error_m` column. This needs scalar trace verification (Phase 3).

---

## 7. Next Phase

Phase 2: Verify and fix Step D hip-yaw baseline (`0.0` values).
