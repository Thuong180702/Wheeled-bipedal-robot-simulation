# K2 Standalone Mode Semantic Patch Report

**Date:** 2026-06-29
**Phase:** 5 — PATCH METRIC DEFINITION MISMATCH
**Status:** ✅ COMPLETE

---

## 1. Root Cause

The dedicated runner's `hip_yaw_div.max_rad` (divergence error metric) was being compared against the original baseline's `hip_yaw_max_rad` (joint angle metric). These are fundamentally different quantities:

| Quantity | Formula | Typical value (low_0p300) |
|---|---|---|
| Hip-yaw joint angle max | max(|l_hip_yaw_pos|, |r_hip_yaw_pos|) | ~0.04 rad |
| Hip-yaw divergence error | (l_pos - r_pos) - (l_ref - r_ref) | ~0.07 rad |

The divergence error is systematically larger because it measures left-right asymmetry rather than absolute joint excursion.

**The scalar trace comparison confirmed:** actual hip-yaw joint positions between source and dedicated match within 0.001 rad. No behavioral patch was needed — only a metric definition fix.

---

## 2. Changes Applied

### 2.1 `scripts/run_k2_jax_realtime.py`

**Added hip-yaw joint position tracking:**

```python
# Summary dict init:
"max_hip_yaw_pos": 0.0,

# Hot loop tracking:
sm["max_hip_yaw_pos"] = max(sm["max_hip_yaw_pos"], abs(float(joint_pos[1])), abs(float(joint_pos[6])))

# Summary JSON output:
"hip_yaw_joint_max_rad": sm["max_hip_yaw_pos"],
```

### 2.2 `scripts/validate_k2_jax_dedicated_promotion.py`

**Updated `extract_metrics_from_summary()`:**

```python
# Phase 5 metric fix: use hip_yaw_joint_max_rad (joint angle max) for
# canonical hip-yaw comparison. The original baseline measures
# max(|l_hip_yaw_pos|, |r_hip_yaw_pos|), NOT divergence error.
hy_joint_max = s.get("hip_yaw_joint_max_rad", 0.0)
if hy_joint_max == 0.0:
    # Fallback for old summary format
    hy_joint_max = s.get("hip_yaw_div", {}).get("max_rad", 0.0)
```

### 2.3 `outputs/k2_original_promoted_baseline/k2_original_metrics.json`

Already corrected in Phase 2: Step D hip-yaw values corrected from 0.0 to actual joint-angle values.

---

## 3. Verification

### Before fix (divergence error metric)
```
hip_yaw_max_rad: SAFE_BUT_WORSE (orig=0.131, cand=0.070, divergence error)
```

### After fix (joint angle metric)
```
hip_yaw_max_rad: EXACT_OR_BETTER (orig=0.131, cand=0.043, joint angle)
```

The dedicated runner's hip-yaw joint angle (0.043 rad) is **BETTER** than the original (0.131 rad).

---

## 4. Classification Impact

After Phase 5 metric fix, the hip-yaw regression is RESOLVED for all scenarios that use `hip_yaw_max_rad`:

- Step C: hip-yaw EXACT_OR_BETTER
- Step E: hip-yaw EXACT_OR_BETTER (was SAFE_BUT_WORSE)
- Step D: hip-yaw EXACT_OR_BETTER (was SAFE_BUT_WORSE)
- Long-run: hip-yaw EXACT_OR_BETTER

The remaining SAFE_BUT_WORSE classifications are now limited to:
- **pitch_rms_deg** — real behavioral difference (~1° higher RMS)
- **support_rms_m** — hardcoded to 0.0 (Phase 6)

---

## 5. Non-Patch: standalone_mode

The `standalone_mode=True` hypothesis was tested via scalar trace and found to be **FALSE**. Both source and dedicated compute identical sagittal/support intermediates. The only difference is:
1. 1-step telemetry offset (pre vs post mj_step recording)
2. Telemetry metric naming conventions

No changes to `standalone_mode` semantics were needed.
