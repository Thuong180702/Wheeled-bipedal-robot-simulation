# K2 Drift Iteration 2 — Activity Diagnosis

**Date:** 2026-06-30
**Profile:** `K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED_CANDIDATE`
**Status:** Drift controller was NOT active in previous validation due to profile map key mismatch. Fixed. This diagnosis uses actual controller output.

---

## Critical Bug Fix

The profile map in `run_k2_jax_realtime.py` had key `"k2_jax_dedicated_default_v1_drift_fixed"` but the CLI passed `K2_JAX_DEDICATED_DEFAULT_V1_DRIFT_FIXED_CANDIDATE`. The fallback to `_K2_AUTH_SCHED` (DEFAULT_V1, `enable_drift_controller=False`) meant **zero drift torque in all previous validation runs**. Added the CLI key to the profile map.

---

## Question 1: Is drift torque nonzero for most drifting intervals?

**NO — torque is zero in 93-100% of steps.**

| Case | tau_drift_bounded nonzero steps | Pct |
|---|---|---|
| low_0p320 | 381/6000 | 6.4% |
| high_0p430 | 96/6000 | 1.6% |
| ramp_down | 0/6000 | 0% |
| push_mid_back_60 | 0/6000 | 0% |

## Question 2: Are gates suppressing drift torque to near zero?

**YES — `height_gate` is the primary bottleneck, killing 91-100% of all drift control.**

| Case | stability_gate (mean) | height_gate (mean) | height_gate nonzero % |
|---|---|---|---|
| low_0p320 | 0.361 | **0.053** | 9.0% |
| high_0p430 | 0.534 | **0.020** | 3.3% |
| ramp_down | 0.531 | **0.000** | 0% |
| push_mid_back_60 | 0.782 | **0.000** | 0.03% |

**Root cause:** The height_gate formula is:
```
height_gate = 1 - smoothstep((com_z_vel_abs - 0.005) / (0.03 - 0.005))
```
This starts suppressing at 0.005 m/s CoM velocity and fully kills drift at 0.03 m/s. Normal balance oscillation exceeds 0.005 m/s constantly, so the gate is almost always partially or fully closed.

Other gates:
- `heading_gate`: mean=0.005, nonzero=287/6000 (4.8%) — depends on height_gate, so inherited suppression
- `position_gate`: mean=0.000, nonzero=96/6000 (1.6%) — depends on height_gate, essentially dead

## Question 3: Is tanh bounding active?

Yes where torques exist. `max_tau=5.0` means tanh is transparent for the tiny raw torques (max 0.63 Nm). Raw ≈ bounded.

## Question 4: Is velocity damping sign correct?

**YES.** Body drift velocity is positive (forward), mean bounded torque is [+0.0035, -0.0029] Nm. The left wheel slightly accelerates, right slightly brakes → produces a rightward turning moment, consistent with the yaw error correction. The sagittal component (symmetric) is effectively zero due to height_gate suppression.

## Question 5: Is heading torque sign correct?

Unknown — heading gate is active in only 287/6000 steps (4.8%). When active, gating values are so small that heading torque is negligible. Yaw error grows from 8.7 deg (baseline) to 19.3 deg (drift v2) — **heading drift is WORSE with the drift controller**, possibly due to asymmetric torque from weak velocity damping.

## Question 6: Is position return activating too early or too late?

Position gate is **essentially never active** (96/6000 steps max). The robot drifts 0.17m max and the position gate requires `smoothstep((drift_distance - 0.02) / (0.20 - 0.02))` → only starts at 0.02m drift. Even when distance threshold is met, height_gate kills it.

## Question 7: Does drift torque fight pitch damping or sagittal balance?

**No.** Drift torques (max 0.63 Nm) are negligible compared to total wheel torques (2-10 Nm). They cannot meaningfully fight anything. The problem is the opposite — they're too small to have any effect.

---

## Before/After Comparison (low_0p320, DEFAULT_V1 baseline vs Drift v2)

| Metric | Baseline | Drift v2 | Delta |
|---|---|---|---|
| final_displacement_m | 0.048 | 0.032 | **-33%** ✅ |
| max_displacement_m | 0.155 | 0.172 | +11% ❌ |
| yaw_drift_deg | 8.7 | 19.3 | **+122%** ❌❌ |
| pitch_rms_deg | 4.14 | 4.11 | -1% ✅ |
| falls | 0 | 0 | ✅ |
| drift_torque_active_pct | 0% | 6.4% | — |

---

## Conclusions

1. **height_gate is the showstopper.** It suppresses 91-100% of drift control. The transition region (0.005-0.030 m/s) is far too narrow.
2. **Velocity damping sign is correct** but magnitude is negligible due to gating.
3. **Heading drift got WORSE** — likely from tiny asymmetric torques causing slow rotation without effective heading correction.
4. **Position return is entirely dead** due to combined height_gate + position_gate suppression.
5. **During dynamic height and push recovery, drift controller is completely disabled.**
6. **Wheel torque budget is NOT a constraint** — drift torques are <1% of total wheel torques.

---

## Phase 2 Implications

The existing gate architecture is fundamentally correct but thresholds are wrong. Required changes:

1. **Widen height_gate transition** from (0.005, 0.03) to (0.03, 0.15) m/s — allow drift damping during normal CoM oscillation
2. **Split height_gate**: velocity damping should have a more permissive height gate than position return
3. **During dynamic height**: velocity damping should be reduced but NOT killed; heading hold should yield; position return should be disabled
4. **Heading gate needs independent height_gate** — currently inherits full suppression
5. **Push damping needs to survive height_gate** — push recovery is where drift velocity is highest
