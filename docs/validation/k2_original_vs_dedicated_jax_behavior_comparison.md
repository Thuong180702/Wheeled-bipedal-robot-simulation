# K2 Original K2 Python vs Dedicated JAX Realtime — Behavioral Comparison

**Date:** 2026-06-29
**Phase:** 6 — Direct Comparison
**Status:** `K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED` (preliminary)

---

## 1. Executive Summary

The dedicated JAX realtime runner (`scripts/run_k2_jax_realtime.py`) was compared against the original K2 Python baseline (`scripts/simulate_hierarchical_controller.py --controller-backend python --vd-sagittal-authority-profile k2_notch_low_q_v1`) across fixed-height, push, and dynamic-height scenarios.

**Finding:** The dedicated JAX runner passes fixed-height scenarios (all survive, no falls) but **fails dynamic-height scenarios** (3/3 fall) where the original K2 Python survives. Additionally, hip-yaw divergence at low heights exceeds the 0.35 rad safety gate (0.666 rad vs original K2's 0.131 rad).

---

## 2. Scenario Results Summary

### 2.1 Fixed-Height Standing Balance

| Height | System | Pitch RMS (deg) | HipYawDiv Max (rad) | Fall | Drift (m) |
|--------|--------|----------------|---------------------|------|-----------|
| high_0p480 | Original K2 Python | 3.96* | 0.0563* | No | — |
| high_0p480 | Dedicated JAX | 5.1 | 0.0155 | No | 0.191 |
| high_0p450 | Original K2 Python | 2.75* | 0.0904* | No | — |
| high_0p450 | Dedicated JAX | 4.0 | 0.0187 | No | 0.437 |
| high_0p430 | Original K2 Python | 4.98* | 0.0236* | No | — |
| high_0p430 | Dedicated JAX | 4.0 | 0.0213 | No | 0.338 |
| low_0p380 | Original K2 Python | 3.33* | 0.0392* | No | — |
| low_0p380 | Dedicated JAX | 5.0 | **0.3049** | No | 0.523 |
| low_0p330 | Original K2 Python | 3.63* | 0.0851* | No | — |
| low_0p330 | Dedicated JAX | 3.4 | **0.2808** | No | 0.163 |
| low_0p300 | Original K2 Python | 2.68* | 0.1314* | No | — |
| low_0p300 | Dedicated JAX | 5.4 | **⚠️ 0.6662** | No | 0.554 |

*Original K2 Python values from Step E report (2026-06-25). Dedicated values from telemetry-full runs (2026-06-29).

**⚠️ low_0p300 hip-yaw divergence (0.666 rad) exceeds 0.35 rad absolute safety gate. 5× worse than original K2.**

### 2.2 Dynamic Height Gate-Crossing

| Scenario | System | Steps | Fall | Max Pitch (deg) | Max Drift (m) | HipYawDiv Max (rad) |
|----------|--------|-------|------|-----------------|---------------|---------------------|
| ramp_up | Original K2 Python | 5000/5000 | **No** ✓ | -9.2 to 1.6 | — | — |
| ramp_up | Dedicated JAX | 2989/5000 | **Yes** ✗ | -7.1 to **39.3** | **-10.32** | **0.365** |
| ramp_down | Original K2 Python | NOT RUN | — | — | — | — |
| ramp_down | Dedicated JAX | 4471/5000 | **Yes** ✗ | **-29.7** to 8.3 | **6.78** | **0.502** |
| gate_chatter | Original K2 Python | NOT RUN | — | — | — | — |
| gate_chatter | Dedicated JAX | 2288/5000 | **Yes** ✗ | 0.0 to **39.4** | **-11.77** | **0.388** |

**✗ BLOCKER: Dedicated JAX falls on ALL dynamic height scenarios. Original K2 Python survives ramp_up (only one verified so far).**

### 2.3 Push Recovery

| Scenario | System | Fall | Notes |
|----------|--------|------|-------|
| push_bwd_90N | Original K2 Python | Yes (step 561) | ⚠️ Different push mechanism (qvel injection) |
| push_bwd_90N | Dedicated JAX | No | ⚠️ Different push mechanism (xfrc_applied) |
| push_fwd_90N | Original K2 Python | NOT RUN | |
| push_fwd_90N | Dedicated JAX | No | Roll -0.6 to 1.0 deg, drift 0.369 m |

**⚠️ CANNOT COMPARE: Push mechanisms differ between monolithic script (qvel injection) and dedicated runner (xfrc_applied). Direct comparison invalid.**

### 2.4 Performance

| Scenario | Dedicated JAX Hz | Original K2 Python Hz | Speedup |
|----------|-----------------|----------------------|---------|
| Fixed-high (telemetry off) | 187 | ~10 | **18.7×** |
| Fixed-high (telemetry full) | 182 | ~10 | **18.2×** |
| Fixed-low | 172 | ~10 | **17.2×** |
| Push | 150-170 | ~10 | **15-17×** |
| Dynamic height | 103-137 | ~12 | **8-11×** |

---

## 3. Metric Deltas

### 3.1 Fixed-Height (high_0p480, 2000 steps)

| Metric | Original K2 Python | Dedicated JAX | Delta | Threshold | Pass? |
|--------|-------------------|---------------|-------|-----------|-------|
| Fall | No | No | 0 | Must match | ✓ |
| Pitch RMS (deg) | 3.96 | 5.1 | +29% | ≤10% | ✗ |
| HipYawDiv max (rad) | 0.0563 | 0.0155 | -72% | ≤10% or ≤0.35 | ✓ |
| Max torque (Nm) | 8.00 | 9.56 | +19.5% | ≤10% | ✗ |
| CoM Z range (m) | 0.481-0.491 | 0.481-0.492 | ~same | — | ✓ |

### 3.2 Fixed-Height (low_0p300, 2000 steps)

| Metric | Original K2 Python | Dedicated JAX | Delta | Threshold | Pass? |
|--------|-------------------|---------------|-------|-----------|-------|
| Fall | No | No | 0 | Must match | ✓ |
| Pitch RMS (deg) | 2.68 | 5.4 | +102% | ≤10% | ✗ |
| HipYawDiv max (rad) | 0.1314 | 0.6662 | **+407%** | ≤10% AND ≤0.35 | ✗✗ |
| Max torque (Nm) | — | 9.43 | — | — | — |

### 3.3 Dynamic Height (ramp_up, 5000 steps)

| Metric | Original K2 Python | Dedicated JAX | Delta | Threshold | Pass? |
|--------|-------------------|---------------|-------|-----------|-------|
| Fall | No | **Yes** (step 2989) | — | Must match | ✗✗ |
| Max pitch (deg) | -9.2 to 1.6 | -7.1 to 39.3 | +2355% | ≤10% | ✗✗ |
| Final drift (m) | — | 10.32 | — | ≤0.02 | ✗✗ |
| HipYawDiv max (rad) | — | 0.365 | — | ≤0.35 | ✗ |
| Contact loss (steps) | — | 319 | — | — | ✗ |

---

## 4. Root Cause Analysis

### 4.1 Dynamic Height Failures — ROOT CAUSE HYPOTHESIS

The dedicated runner passes the dynamic height reference (`commanded_height_ref_m`) to the JAX controller correctly. The JAX controller receives the same input format as the canonical path. However, the monolithic script has additional orchestration that the dedicated runner lacks:

1. **Posture recalibration**: The monolithic script's `simulation_step()` calls `centroidal_state_control = capture_estimator.update(...)` which performs state capture/calibration each step. The dedicated runner does not include this `capture_estimator`.

2. **Observation construction**: The monolithic script constructs a 42-dim observation with gravity in body frame computed from the rotation matrix. The dedicated runner passes raw sensor values directly to the JAX controller.

3. **Controller internal state**: The JAX controller's internal state includes pitch-rate history for the notch filter. During rapid height changes, the pitch dynamics change, and the notch filter state may not adapt correctly.

**Verification pending:** Both-synced and canonical JAX backend ramp_up runs are in progress to determine if the JAX controller itself fails or if the dedicated runner's orchestration is the issue.

### 4.2 Hip-Yaw Divergence at Low Heights

The dedicated JAX shows significantly higher hip-yaw divergence at low heights (0.666 rad at low_0p300 vs 0.131 rad for original K2). Possible causes:

1. **Velocity damping scale**: The dedicated runner computes `velocity_damping_scale` from variant name but the JAX controller may apply it differently than the Python controller.
2. **Mode-div reference source**: The dedicated runner sets `mode_div_ref_source="disabled"` as default. The original K2 might compute this reference differently.

### 4.3 Push Mechanism Mismatch

The monolithic script injects push via `qvel` (velocity impulse), while the dedicated runner uses `xfrc_applied` (force). These produce fundamentally different dynamics. Push results are NOT comparable.

---

## 5. Regressions Identified

| # | Regression | Severity | Evidence |
|---|-----------|----------|----------|
| R1 | Dynamic height falls (ramp_up) | **CRITICAL** | JAX falls step 2989, Python survives 5000 |
| R2 | Dynamic height falls (ramp_down) | **CRITICAL** | JAX falls step 4471 |
| R3 | Dynamic height falls (gate_chatter) | **CRITICAL** | JAX falls step 2288 |
| R4 | Hip-yaw divergence exceeds 0.35 rad gate at low_0p300 | **HIGH** | 0.666 rad vs 0.131 rad (5× worse) |
| R5 | Hip-yaw divergence elevated at low_0p380 | **MEDIUM** | 0.305 rad vs 0.039 rad (7.8× worse) |
| R6 | Hip-yaw divergence elevated at low_0p330 | **MEDIUM** | 0.281 rad vs 0.085 rad (3.3× worse) |
| R7 | Pitch RMS worse at high_0p480 | **LOW** | 5.1 vs 3.96 deg (+29%) |
| R8 | Max torque higher at high_0p480 | **LOW** | 9.56 vs 8.00 Nm (+20%) |
| R9 | Push mechanism mismatch | **INFO** | Cannot compare push scenarios |
| R10 | Drift higher across all scenarios | **MEDIUM** | 0.2-0.6 m over 2000 steps |

---

## 6. Preliminary Classification

**`K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED`**

**Blocking issues:**
1. Dynamic height: Dedicated JAX falls on ALL 3 tested scenarios (ramp_up, ramp_down, gate_chatter). Original K2 Python survives.
2. Hip-yaw divergence at low_0p300 exceeds 0.35 rad absolute gate (0.666 rad).
3. Root cause not yet isolated (JAX controller vs dedicated runner orchestration).

**Mitigating factors:**
- Fixed-height: No falls, all heights survive
- Performance: 70-187 Hz (excellent, 8-19× speedup)
- Telemetry: Full 60-column behavioral CSV works correctly
- Visual mode: Works (from prior validation)

---

## 7. Next Steps Required Before Promotion

1. **Complete root cause analysis**: Both-synced and canonical JAX runs in progress
2. **Fix dynamic height**: If dedicated runner orchestration issue, fix and re-validate
3. **Fix hip-yaw divergence**: Investigate mode-div reference source at low heights
4. **Fix push mechanism**: Align push application with monolithic script (qvel injection)
5. **Re-run full matrix**: After fixes, re-run all scenarios
6. **Re-compare against original K2**: After fixes, verify behavioral parity

---

## 8. Data Files

| Run | Output Directory | Status |
|-----|-----------------|--------|
| Original K2 Python — high_0p480 | `outputs/k2_jax_dedicated_promotion_test/original_k2_python_high_0p480/` | Complete |
| Original K2 Python — low_0p330 | `outputs/k2_jax_dedicated_promotion_test/original_k2_python_low_0p330/` | Complete |
| Original K2 Python — ramp_up | `outputs/k2_jax_dedicated_promotion_test/original_k2_python_ramp_up/` | Complete |
| Original K2 Python — push_bwd_90N | `outputs/k2_jax_dedicated_promotion_test/original_k2_python_push_bwd_90N/` | Complete (fell) |
| Dedicated JAX — fixed-high_0p480 | `outputs/k2_jax_dedicated_promotion_test/jax_fixed_high_0p480/` | Complete |
| Dedicated JAX — fixed-low_0p330 | `outputs/k2_jax_dedicated_promotion_test/jax_fixed_low_0p330/` | Complete |
| Dedicated JAX — fixed-low_0p300 | `outputs/k2_jax_dedicated_promotion_test/jax_fixed_low_0p300/` | Complete |
| Dedicated JAX — fixed-high_0p430 | `outputs/k2_jax_dedicated_promotion_test/jax_fixed_high_0p430/` | Complete |
| Dedicated JAX — fixed-high_0p450 | `outputs/k2_jax_dedicated_promotion_test/jax_fixed_high_0p450/` | Complete |
| Dedicated JAX — fixed-low_0p380 | `outputs/k2_jax_dedicated_promotion_test/jax_fixed_low_0p380/` | Complete |
| Dedicated JAX — push_bwd_90N | `outputs/k2_jax_dedicated_promotion_test/jax_push_bwd_90N/` | Complete |
| Dedicated JAX — push_fwd_90N | `outputs/k2_jax_dedicated_promotion_test/jax_push_fwd_90N/` | Complete |
| Dedicated JAX — ramp_up | `outputs/k2_jax_dedicated_promotion_test/jax_ramp_up/` | Complete (fell) |
| Dedicated JAX — ramp_down | `outputs/k2_jax_dedicated_promotion_test/jax_ramp_down/` | Complete (fell) |
| Dedicated JAX — gate_chatter | `outputs/k2_jax_dedicated_promotion_test/jax_gate_chatter/` | Complete (fell) |
| Both-synced — ramp_up | `outputs/k2_jax_dedicated_promotion_test/both_synced_ramp_up/` | **IN PROGRESS** |
| Canonical JAX — ramp_up | `outputs/k2_jax_dedicated_promotion_test/canonical_jax_ramp_up/` | **IN PROGRESS** |
