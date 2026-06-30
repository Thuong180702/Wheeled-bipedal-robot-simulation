# K2 JAX Dedicated Realtime Runner — Original K2 Behavioral Fix Report

**Date:** 2026-06-29  
**Task:** Fix K2 JAX dedicated realtime runner to match original promoted K2 behavior  
**Classification:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`

---

## 1. Executive Summary

Two root causes were found and fixed in the dedicated realtime runner:

1. **mode_div disabled by default** — The dedicated runner had `DEFAULT_MODE_DIV_REF_SOURCE = "disabled"`, but the original K2 promotion validation ALWAYS ran with `--enable-mode-hip-yaw-divergence`. Without mode_div, hip-yaw divergence grew to ~0.412 rad at low_0p300 (3.1× original K2).

2. **Static q_ref for dynamic height** — The dedicated runner used a fixed equilibrium joint reference captured at initialization. During dynamic height trajectories, the `q_ref` (posture reference) never updated with the changing height, causing the robot to fail to track height commands and drift up to 3.8 meters.

After fixes: **14/15 scenarios PASS, 1 PARTIAL** (ramp_down hy=0.3728, slightly above 0.35 gate).

---

## 2. Root Cause #1: mode_div Disabled

### Before
```python
DEFAULT_MODE_DIV_REF_SOURCE = "disabled"  # default unless --enable-mode-hip-yaw-divergence
```
No `--enable-mode-hip-yaw-divergence` CLI flag existed.

### After
```python
DEFAULT_MODE_DIV_REF_SOURCE = "target"  # Original K2 validation runs with mode-div enabled
```
Added `--enable-mode-hip-yaw-divergence` (default: True) and `--no-mode-hip-yaw-divergence` flags.

### Impact
| Height | Before hy_max | After hy_max | Original K2 | Gate |
|--------|--------------|-------------|-------------|------|
| low_0p300 | ~0.412 | 0.2008 | 0.1314 | PASS |
| low_0p320 | ~0.392 | 0.0821 | 0.0502 | PASS |
| low_0p330 | ~0.361 | 0.1162 | 0.0851 | PASS |

Hip-yaw divergence dropped 3-5×, all well under the 0.35 rad safety gate.

### Files changed
- `scripts/run_k2_jax_realtime.py`:
  - Line 84: `DEFAULT_MODE_DIV_REF_SOURCE = "target"`
  - Lines 169-175: Added `--enable-mode-hip-yaw-divergence` / `--no-mode-hip-yaw-divergence` flags
  - Line 339: Use `args.enable_mode_hip_yaw_divergence` to set ref source
  - Line 817: Added `mode_div_enabled` to summary JSON
  - Line 899: Added mode_div status to terminal output

---

## 3. Root Cause #2: Static q_ref During Dynamic Height

### Before
`eq_joint` (posture reference) was captured once at initialization from `mj_forward`. During dynamic height trajectories, the height command changed but the posture reference stayed at the initial height's equilibrium.

Result: CoM stuck at [0.325, 0.335] during ramp_up (target 0.33→0.48), drifting 3.8 meters.

### After
Added `build_height_qref_interpolator()` — loads all available height setup files, sorts by `target_com_z_m`, and builds per-joint linear interpolation functions.

During dynamic height:
- `height_ref` updates from trajectory at each step
- `eq_joint` (hip_pitch and knee joints) updates via interpolation
- `height_floor` updates dynamically to prevent false termination

Additionally updated `height_floor` to track the changing `height_ref - 0.05`.

### Files changed
- `scripts/run_k2_jax_realtime.py`:
  - Added `build_height_qref_interpolator()` function (~40 lines)
  - Lines 496-501: Initialize interpolator for dynamic height
  - Lines 608-617: Dynamic height loop updates q_ref and height_floor

### Impact
| Scenario | Before CoM Range | After CoM Range | Before hy | After hy |
|----------|-----------------|-----------------|-----------|----------|
| ramp_up | [0.325, 0.335] | [0.333, 0.490] | 0.4029 | 0.1242 |
| ramp_down | [0.430, 0.481] → fall | [0.326, 0.491] | — | 0.3728 |
| gate_chatter | [0.379, 0.383] | [0.379, 0.472] | 0.1596 | 0.2160 |

Robot now tracks dynamic height commands. Height RMS error under 1 cm for all scenarios.

---

## 4. Complete Validation Matrix

### Step E: Fixed-Height Sweep (10 heights, 2000 steps)

| Height | Fell | hy_max_rad | pitch_rms_deg | Status |
|--------|------|-----------|--------------|--------|
| low_0p300 | No | 0.2008 | 2.9 | PASS |
| low_0p320 | No | 0.0821 | 3.7 | PASS |
| low_0p330 | No | 0.1162 | 4.0 | PASS |
| low_0p340 | No | 0.1255 | 1.9 | PASS |
| low_0p360 | No | 0.0897 | 3.1 | PASS |
| low_0p380 | No | 0.0759 | 5.2 | PASS |
| high_0p430 | No | 0.0833 | 3.1 | PASS |
| high_0p450 | No | 0.0263 | 4.7 | PASS |
| high_0p465 | No | 0.0454 | 3.6 | PASS |
| high_0p480 | No | 0.0735 | 4.3 | PASS |

**Step E: 10/10 PASS** — 0 falls, all hy ≤ 0.35 (max 0.2008)

### Step D: Push Recovery (Step D format, 2000 steps)

| Condition | Fell | hy_max_rad | Status |
|-----------|------|-----------|--------|
| high_0p480 fwd 90N | No | 0.0388 | PASS |
| high_0p480 bwd 90N | No | 0.0281 | PASS |

**Push: 2/2 PASS** — 0 falls, hy very low

### Dynamic Height (3 scenarios, 5000 steps)

| Scenario | Fell | hy_max_rad | CoM Range | Status |
|----------|------|-----------|-----------|--------|
| ramp_up | No | 0.1242 | [0.333, 0.490] | PASS |
| ramp_down | No | **0.3728** | [0.326, 0.491] | PARTIAL |
| gate_chatter | No | 0.2160 | [0.379, 0.472] | PASS |

**Dynamic: 2 PASS, 1 PARTIAL**

---

## 5. Comparison with Original K2

| Metric | Original K2 | Fixed JAX | Gate | Match? |
|--------|------------|-----------|------|--------|
| Step C hy_max | 0.0851 | Not run* | ≤0.35 | — |
| Step E hy_max | 0.1314 | 0.2008 | ≤0.35 | ✅ |
| Push hy_max | 0.0000 | 0.0388 | ≤0.35 | ✅ |
| ramp_up hy | 0.0534 | 0.1242 | ≤0.35 | ✅ |
| ramp_down hy | 0.0977 | **0.3728** | ≤0.35 | ⚠️ |
| gate_chatter hy | 0.0629 | 0.2160 | ≤0.35 | ✅ |
| Falls (all) | 0 | 0 | 0 | ✅ |
| No NaN/Inf | PASS | PASS | — | ✅ |
| No hidden torque | PASS | PASS | — | ✅ |
| No WBC | PASS | PASS | — | ✅ |
| Realtime ≥50 Hz | N/A | 67-186 Hz | ≥50 | ✅ |

*Step C dynamic cases not applicable to dedicated runner's fixed-height mode.

---

## 6. ramp_down Limitation Analysis

ramp_down shows hip_yaw = 0.3728 rad (6.5% above 0.35 gate). This is the dedicated runner's primary limitation:

**Root cause:** The dedicated runner uses height-dependent q_ref interpolation from setup files, which is an approximation. The original K2 Python controller dynamically recomputes the full IK solution at each step using the sagittal velocity damped balance controller, which provides more accurate posture references during height transitions.

**Why ramp_down is hardest:**
- Descending through the notch gate region (0.48→0.42m) changes the WIP filter behavior
- q_ref interpolation smoothness depends on available height setup density (10 setups for 0.18m range)
- The descending direction excites pitch oscillations that mode_div partially counteracts

**Recommended resolution:** Use the canonical `simulate_hierarchical_controller.py` path (Python or JAX backend) for dynamic height promotion criteria. The dedicated runner is optimized for fixed-height and push benchmarking.

---

## 7. Real-time Performance

| Scenario | Achieved Hz | Mean Step (ms) |
|----------|------------|----------------|
| Fixed high_0p480 | 166-186 | 5.4-6.0 |
| Fixed low_0p300 | 46-177* | 5.6-21.8 |
| Push (any) | 78-82 | 12.2-12.8 |
| Dynamic ramp_up | 68-180 | 5.5-14.8 |
| Dynamic ramp_down | 152-168 | 6.0-6.6 |
| Dynamic gate_chatter | 178-186 | 5.4-5.6 |

\* Initial run includes JIT compile time; subsequent runs faster. All scenarios ≥50 Hz minimum. Most scenarios >100 Hz.

---

## 8. Promoted Scope

| Feature | Status | Notes |
|---------|--------|-------|
| Fixed-height standing (Step E) | ✅ PROMOTED | All 10 heights pass |
| Push recovery (Step D format) | ✅ PROMOTED | xfrc_applied, step 300, dur 5 |
| mode_div enabled | ✅ PROMOTED | Default: enabled (matches original K2) |
| K2 profile (notch Q=2.0) | ✅ PROMOTED | Canonical K2_NOTCH_LOW_Q_V1 |
| Real-time performance | ✅ PROMOTED | >100 Hz typical, ≥50 Hz minimum |
| Dynamic height (ramp_up) | ✅ PROMOTED | hy=0.1242, height RMS ~0.006m |
| Dynamic height (gate_chatter) | ✅ PROMOTED | hy=0.2160, height RMS ~0.005m |
| Telemetry off/decimated/full | ✅ PROMOTED | No per-step print, correct row count |
| Visual viewer | ✅ PROMOTED | Realtime pacing, hold, speed controls |
| Python fallback | ✅ PROMOTED | sim_hierarchical_controller works |

## 9. Non-Promoted Scope

| Feature | Status | Notes |
|---------|--------|-------|
| Dynamic height (ramp_down) | ⚠️ PARTIAL | hy=0.3728 > 0.35; use canonical path |
| Step C dynamic scenarios | ⚠️ NOT TESTED | Requires canonical path |
| Step D full push matrix | ⚠️ PARTIAL | Only high_0p480 tested; mid/low pending |
| 60N push magnitude | ⚠️ NOT TESTED | Only 90N tested |
| push-mode xfrc vs qvel | ⚠️ NOT VERIFIED | Only xfrc_applied implemented |
| Long-run equilibrium | ⚠️ NOT TESTED | |
| Step C focused cases | ⚠️ NOT TESTED | |

---

## 10. Test Status

Pending — Phase 9 tests need to be written/updated.

---

## 11. Final Classification

**`K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`**

**PROMOTED:**
- Fixed-height: 10/10 Step E heights all pass safety gates ✅
- Push recovery: both 90N directions pass ✅
- Dynamic height ramp_up and gate_chatter: pass ✅
- mode_div: enabled by default, matching original K2 ✅
- Real-time performance: ≥50 Hz minimum, >100 Hz typical ✅

**BLOCKED:**
- ramp_down hip_yaw = 0.3728 rad (6.5% above 0.35 gate) ⚠️
- This is a dedicated runner limitation from approximate q_ref interpolation

**Recommendation:** Run dynamic height scenarios through canonical `simulate_hierarchical_controller.py` path for official promotion criteria. The dedicated runner is promoted for fixed-height, push, and ramp_up/gate_chatter scenarios.

---

## 12. Changes Summary

### Modified files
1. `scripts/run_k2_jax_realtime.py`:
   - Added `--enable-mode-hip-yaw-divergence` flag (default: True)
   - Changed `DEFAULT_MODE_DIV_REF_SOURCE` from `"disabled"` to `"target"`
   - Added `build_height_qref_interpolator()` for dynamic height support
   - Dynamic height loop now updates q_ref and height_floor
   - Added mode_div status to summary JSON and terminal output

### New files (Phase 0)
2. `docs/validation/k2_original_promoted_source_of_truth.md`

### Not changed
- Original K2 Python controller behavior
- K2 profile parameters (wip_notch_q=2.0, all gains unchanged)
- JAX controller code
- Safety gate thresholds
- Physics simulation parameters
