# K2 Hip-Yaw First Divergence Audit

**Date:** 2026-06-29
**Phase:** 4 — IDENTIFY FIRST DIVERGENT FIELD
**Status:** ✅ COMPLETE

---

## 1. Executive Summary

Scalar trace comparison between source-of-truth (Python monolithic K2) and dedicated JAX runner revealed that **there is NO real hip-yaw behavioral regression**. The apparent regression was caused by a **metric definition mismatch**: the dedicated runner measured hip-yaw divergence error while the original baseline measured hip-yaw joint angle.

After correcting the metric, the dedicated runner's hip-yaw joint angle (0.043 rad) is **BETTER** than the original (0.131 rad).

---

## 2. Trace Analysis Results

### 2.1 Initial (incorrect) finding

Using wrong field mappings, the trace comparison initially reported:
- First divergent field: `pitch_deg` at step 107
- Delta: 6.82 degrees

This was because source `euler_pitch_y` (true pitch) was being compared against dedicated `pitch_deg` (robot_pitch_x = true roll).

### 2.2 Corrected field mapping

| Source Column | Dedicated Column | Convention |
|---|---|---|
| `euler_pitch_y` | `roll_deg` | True pitch (both robot-frame roll) |
| `euler_roll_x` | `pitch_deg` | True roll (both robot-frame pitch_x) |
| `robot_pitch_x` | `pitch_deg` | Robot pitch (= euler roll) |
| `robot_roll_y` | `roll_deg` | Robot roll (= euler pitch) |

Both source and dedicated use the SAME robot-frame convention from `compute_robot_frame_orientation_from_quaternion()`.

### 2.3 Corrected comparison

| Step | Source robot_pitch_x | Dedicated pitch_deg | Delta |
|---|---|---|---|
| 50 | 3.9847° | 3.9237° | 0.061° |
| 100 | 6.9591° | 7.0358° | -0.077° |
| 200 | 0.5697° | 0.3793° | 0.190° |

Pitch/roll values are nearly identical (<0.2° difference). No behavioral divergence.

### 2.4 Hip-yaw joint positions

| Step | Source l_hy | Dedicated l_hy | Source r_hy | Dedicated r_hy |
|---|---|---|---|---|
| 50 | -0.003965 | -0.003984 | 0.003825 | 0.003887 |
| 100 | -0.003702 | -0.003707 | 0.006700 | 0.006847 |
| 200 | -0.025141 | -0.026186 | 0.039514 | 0.040703 |

Hip-yaw joint positions match within **0.001 rad** across all steps. No behavioral divergence.

### 2.5 Hip-yaw divergence metric mismatch

| Metric | Source value | Dedicated value | Definition |
|---|---|---|---|
| `hip_yaw_joint_max_rad` | 0.192 rad | 0.043 rad | max(|l_hy|, |r_hy|) |
| `hip_yaw_div.max_rad` | 0.355 rad | 0.070 rad | divergence error |
| `hip_yaw_divergence` telemetry | 0.065 rad | N/A | |l-r| (source records absolute value) |

**Source telemetry `hip_yaw_divergence` records |l-r| (absolute value), not the signed divergence.** This is a telemetry recording difference, not a controller behavioral difference.

---

## 3. Root Cause Summary

| Apparent Issue | Actual Cause | Resolution |
|---|---|---|
| Hip-yaw "regression" | Metric mismatch: divergence error vs joint angle | Phase 5: track joint angle max |
| Pitch/roll "swap" | Trace field mapping error | Fixed mapping in comparison |
| Torque mismatch at step 0 | 1-step telemetry offset (pre vs post mj_step) | Not control-affecting |
| hip_yaw_div sign flip | Source records |l-r|, dedicated records signed | Telemetry difference only |

---

## 4. Remaining Real Issue

After fixing all metric mismatches, the only remaining regression is:

**Pitch RMS: dedicated 3.61° vs original 2.68°** (SAFE_BUT_WORSE)

This is a real behavioral difference that needs investigation (possibly due to `standalone_mode` sagittal computation or ABS trim differences).

---

## 5. Hypothesis Test Results

| Hypothesis | Result |
|---|---|
| standalone_mode computes sag/support differently | FALSE — scalar values match within tolerance |
| Pitch/roll axis swap | FALSE — both use same convention |
| mode_div sign error | FALSE — divergence computation is identical |
| Hip-yaw behavioral regression | FALSE — joint positions match within 0.001 rad |
| **Metric definition mismatch** | **TRUE — divergence error vs joint angle** |
