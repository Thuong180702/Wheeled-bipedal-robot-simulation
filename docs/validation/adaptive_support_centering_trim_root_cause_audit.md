# adaptive_support_centering_trim — Root-Cause Audit

**Date:** 2026-06-14
**Base profile audited:** `support_centering_bias_trim`
**Setup:** high_0p480
**Steps:** 5000 (4999 telemetry rows)
**Drift column used:** `active_pitch_crossing_signed_error_m` (PRIMARY, priority 1 of 4)
**Telemetry source:** `outputs/hierarchical_controller_sim/telemetry_1781415963.csv` (full-rate, decimation=1)

> **Metric policy reminder:** Final error is reported but is NOT a pass/fail criterion.
> Pitch RMS / wheel velocity are diagnostic only, not rejection criteria.

---

## A. Drift boundedness

| Metric | Value |
|--------|-------|
| max abs error | 0.1828 m |
| P2P | 0.2078 m |
| error RMS | 0.0965 m |
| error IQR | 0.1010 m |
| MAE | 0.0797 m |
| outside ±0.05 | 61.2% |
| outside ±0.08 | 47.2% |
| outside ±0.10 | 39.0% |
| outside ±0.15 | 14.1% |
| worst 500-step max abs | 0.1828 m |
| worst 500-step outside ±0.10 | 48.0% |
| final error (reported, not pass/fail) | +0.1178 m |

Drift is bounded — no divergence, max abs stays at 0.18 m well below the 0.24 m safety limit.

## B. Drift centering / symmetry

| Metric | Value |
|--------|-------|
| mean signed error | +0.0787 m |
| median signed error | +0.0736 m |
| positive % | 94.8% |
| negative % | 5.1% |
| zero crossings (total) | 18 |
| positive area | 395.90 |
| negative area | 2.30 |
| bias ratio `abs(pos-neg)/total` | 0.988 |
| time inside ±0.03 | 26.3% |
| time inside ±0.05 | 38.8% |
| time inside ±0.08 | 52.8% |

**This is the core problem.** Drift is almost entirely one-sided positive (bias ratio 0.988, positive area 172× negative area). The robot sits at a persistent +0.07–0.08 m offset rather than oscillating around zero.

## C. Drift accumulation

| Metric | Value |
|--------|-------|
| first 1000 MAE | 0.0790 m |
| last 1000 MAE | 0.0771 m |
| accumulation ratio | 0.976 |

No accumulation — the bias is steady-state, not growing. Ratio < 1.0 means it is marginally improving over the run. This rules out runaway drift.

## D. Posture stability

| Metric | Value |
|--------|-------|
| pitch deg (min/mean/max/RMS) | -0.91 / 3.65 / 8.28 / 4.45 |
| roll deg (min/mean/max/RMS) | -0.22 / 0.02 / 0.27 / 0.10 |
| CoM Z m (min/mean/max) | 0.4585 / 0.4829 / 0.4915 |
| contact | double-contact dominant, no fall |
| termination | completed (survived 5000) |

Posture is stable. Pitch is modest (max 8.3 deg), roll is negligible, height holds near 0.48 m target.

## E. Hip-yaw stability

| Metric | Value |
|--------|-------|
| hip_yaw_abs_max (min/mean/max/RMS) | 0.0000 / 0.1006 / 0.3025 / 0.1320 |
| hip_yaw_error_rms (min/mean/max/RMS) | 0.0000 / 0.0982 / 0.3025 / 0.1310 |

**Telemetry gap: NONE.** `hip_yaw_abs_max` and `hip_yaw_error_rms` are populated. Hip-yaw is bounded but reaches 0.30 rad at peak, occasionally above the proposed 0.25 rad gate. This is a minor secondary limiter at high height, not the dominant cause of the positive bias.

## F. Smoothness / perceived oscillation (diagnostic only)

| Metric | Value |
|--------|-------|
| wheel velocity RMS | 3.19 rad/s |
| wheel velocity max abs | 6.63 rad/s |
| spikes > 5 rad/s | 413 |
| spikes > 6 rad/s | 43 |
| spikes > 7 rad/s | 0 |
| wheel acceleration RMS | 0.158 |

No spikes above 7 rad/s. Wheel motion is consistent with active balancing, not oscillatory instability.

## G. Bias trim behavior (the key finding)

| Metric | Value |
|--------|-------|
| trim active % | 96.3% |
| safety gate pass % | 98.5% |
| direction correct % | 100.0% |
| tau range | [-0.35, 0.0] Nm |
| **saturation %** | **93.2%** |
| mean error when active | +0.0781 m |
| block reasons | positive_bias_correcting 4814, inside_exit 85, upright_gate_fail 76, hold 23, contact 1 |

**The trim is saturated 93.2% of the time at its -0.35 Nm cap, yet mean signed error remains +0.0787 m.** The trim is doing everything it can and is still authority-limited. It is also bang-bang: a threshold rule that snaps to full ±0.35 Nm rather than scaling with error magnitude.

---

## Answers to required questions

1. **Is the current support-centering trim too weak?** YES. Saturated 93.2% at the -0.35 Nm cap and the bias persists at +0.0787 m. Authority-limited.
2. **Is it too bang-bang / saturated?** YES. Threshold rule jumps straight to full ±0.35 Nm; cannot modulate proportionally.
3. **Does it reduce mean drift but not enough?** YES. +0.0953 → +0.0787 m vs T6I, still far from zero, positive % still 94.8%.
4. **Does it improve centering without increasing posture instability?** YES. Direction correct 100%, posture stable, accumulation ratio 0.976.
5. **Is hip-yaw stable or drifting?** Mostly stable — bounded at mean 0.10 / max 0.30 rad, with occasional excursions above 0.25 rad.
6. **Is there a missing hip-yaw telemetry gap?** NO. `hip_yaw_abs_max` and `hip_yaw_error_rms` are available.
7. **Is the standing posture stable despite the remaining drift?** YES. No fall, pitch RMS 4.45 deg, roll RMS 0.10 deg, height held.

---

## Classification

**`SUPPORT_TRIM_TOO_WEAK_BUT_POSTURE_STABLE`**

The bias trim is directionally correct and posture-safe, but it is (a) authority-limited — saturated 93% at -0.35 Nm — and (b) bang-bang rather than proportional. The fix is an **adaptive proportional trim** with a higher height-aware ceiling at high heights (where positive bias is largest), proportional gain so it modulates with error magnitude, near-zero relief, sign-reversal and oscillation guards, plus a hip-yaw safety gate to handle the secondary 0.30 rad excursions.

JSON: `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/adaptive_support_centering_trim_root_cause_audit.json`
