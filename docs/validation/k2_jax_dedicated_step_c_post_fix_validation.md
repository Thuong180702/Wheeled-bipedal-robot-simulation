# K2 JAX Dedicated Runner -- Step C Post-Fix Validation Report

**Date:** 2026-06-29
**Comparison:** `k2_notch_low_q_v1` (Python original) vs JAX dedicated runner (candidate)
**Data sources:**
- `outputs/k2_jax_dedicated_promotion_validation/all_metrics_comparison.json`
- `outputs/k2_jax_dedicated_promotion_validation/step_c/*/summary.json`
- `outputs/k2_original_promoted_baseline/k2_original_metrics.json`

---

## Phase 5: Step C Strict Verification

### Background

Step C originally used dynamic height ladder/random patterns with the original K2 Python controller. The original Step C scenarios executed dynamic height commands (slow ladder up/down, random dwells at 500 and 200 steps, abrupt stress pattern, and a long random pattern) at the low height baseline of 0.33 m. The mode_div mechanism was enabled with the K2 notch-low-q profile (wip_notch_q=2.0).

For the dedicated JAX runner, the architecture runs at fixed height per simulation. Therefore:

- **C1-C5** are run as fixed-height at 0.33 m, matching the original baseline height_m: 0.33. Since the controller is pinned at fixed height, the different height command patterns (ladder, random, abrupt) do not influence the simulation -- C1-C4 produce identical candidate metrics as they are the same sim run.
- **C5** is a longer 3000-step run (vs 2000 for C1-C4), producing slightly different metrics due to longer accumulation.
- **focused_low_0p320** runs as fixed-height at 0.32 m.
- **focused_high_0p480** runs as fixed-height at 0.48 m.

### Original baseline metrics

From `k2_original_metrics.json`, the Step C original baseline (Python backend, mode_div enabled, dynamic height patterns at low_0p330):

| Scenario | height_m | fell | pitch_rms_deg | hip_yaw_max_rad | support_rms_m |
|----------|----------|------|---------------|-----------------|----------------|
| C1-C5    | 0.33     | false | 3.63          | 0.0851          | 0.0386         |
| focused_low_0p320 | 0.32 | false | 2.83       | 0.0502          | 0.0525         |
| focused_high_0p480| 0.48 | false | 3.96       | 0.0563          | 0.0471         |

Note: C1-C5 in the original baseline all shared identical metrics as they were recorded at the same common height.

### All 7 Step C Cases -- Candidate vs Original

#### C1 -- Slow Ladder Up/Down (fixed 0.33 m, 2000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 3.963 | 3.63 | +0.333 | 1.000 | PASS |
| hip_yaw_max_rad | 0.1162 | 0.0851 | +0.0311 | 0.0500 | PASS |
| support_rms_m | 0.0894 | 0.0386 | +0.0508 | 0.0193 | FAIL |

**Class: SAFE_BUT_WORSE** (no fall; support_rms exceeds tolerance)

#### C2 -- Random 500-Dwell (fixed 0.33 m, 2000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 3.963 | 3.63 | +0.333 | 1.000 | PASS |
| hip_yaw_max_rad | 0.1162 | 0.0851 | +0.0311 | 0.0500 | PASS |
| support_rms_m | 0.0894 | 0.0386 | +0.0508 | 0.0193 | FAIL |

**Class: SAFE_BUT_WORSE** (identical sim run to C1)

#### C3 -- Random 200-Dwell (fixed 0.33 m, 2000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 3.963 | 3.63 | +0.333 | 1.000 | PASS |
| hip_yaw_max_rad | 0.1162 | 0.0851 | +0.0311 | 0.0500 | PASS |
| support_rms_m | 0.0894 | 0.0386 | +0.0508 | 0.0193 | FAIL |

**Class: SAFE_BUT_WORSE** (identical sim run to C1)

#### C4 -- Abrupt Stress (fixed 0.33 m, 2000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 3.963 | 3.63 | +0.333 | 1.000 | PASS |
| hip_yaw_max_rad | 0.1162 | 0.0851 | +0.0311 | 0.0500 | PASS |
| support_rms_m | 0.0894 | 0.0386 | +0.0508 | 0.0193 | FAIL |

**Class: SAFE_BUT_WORSE** (identical sim run to C1)

#### C5 -- Long Random (fixed 0.33 m, 3000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 4.512 | 3.63 | +0.882 | 1.000 | PASS |
| hip_yaw_max_rad | 0.1823 | 0.0851 | +0.0972 | 0.0500 | FAIL |
| support_rms_m | 0.0894 | 0.0386 | +0.0508 | 0.0193 | FAIL |

**Class: SAFE_BUT_WORSE** (longer run accumulates more hip-yaw divergence; still within safety gate of 0.35 rad)

#### focused_low_0p320 (fixed 0.32 m, 2000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 3.694 | 2.83 | +0.864 | 0.849 | FAIL |
| hip_yaw_max_rad | 0.0821 | 0.0502 | +0.0319 | 0.0500 | PASS |
| support_rms_m | 0.1161 | 0.0525 | +0.0636 | 0.0200 | FAIL |

**Class: SAFE_BUT_WORSE** (pitch_rms barely exceeds relative tolerance by +0.015 deg; support significantly worse)

#### focused_high_0p480 (fixed 0.48 m, 2000 steps)

| Metric | Candidate (JAX) | Original (Python) | Delta | Tolerance | Verdict |
|--------|-----------------|-------------------|-------|-----------|---------|
| fell | false | false | -- | fall=0 req | PASS |
| pitch_rms_deg | 4.280 | 3.96 | +0.320 | 1.000 | PASS |
| hip_yaw_max_rad | 0.0735 | 0.0563 | +0.0172 | 0.0500 | PASS |
| support_rms_m | 0.1150 | 0.0471 | +0.0679 | 0.0200 | FAIL |

**Class: SAFE_BUT_WORSE** (all primary gates pass except support_rms)

---

### Tolerance computation reference

Tolerances from `k2_original_metrics.json`:

```
pitch_rms_deg:    min(1.0,  0.30 * original)
hip_yaw_max_rad:  min(0.05, 2.00 * original)
support_rms_m:    min(0.02, 0.50 * original)
```

Per-case tolerances applied above.

---

### Summary

#### Completion and safety

- **7/7 cases completed**, 0 falls, 0 NaN/Inf, 0 terminations.
- All cases pass the absolute safety gates:
  - `falls_max: 0` -- no falls in any run
  - `hip_yaw_max_rad: 0.35` -- all values well below (worst is C5 at 0.1823 rad)
  - `nan_inf_allowed: false` -- none detected

#### Class breakdown

| Class | Count | Cases |
|-------|-------|-------|
| SAFE_BUT_WORSE | 7 | C1, C2, C3, C4, C5, focused_low_0p320, focused_high_0p480 |

#### Metric deltas summary

```
            pitch_rms_deg  hip_yaw_max_rad  support_rms_m
            cand  orig  delta  cand  orig  delta  cand   orig  delta
C1-C4       3.96  3.63  +0.33  0.116 0.085 +0.031 0.089  0.039 +0.051
C5          4.51  3.63  +0.88  0.182 0.085 +0.097 0.089  0.039 +0.051
foc_low     3.69  2.83  +0.86  0.082 0.050 +0.032 0.116  0.053 +0.064
foc_high    4.28  3.96  +0.32  0.074 0.056 +0.017 0.115  0.047 +0.068
```

#### Key observations

1. **Support RMS is consistently worse** across all 7 cases. The JAX dedicated runner shows support polygon wander approximately 2-3x the original Python controller. This is the single metric preventing any case from reaching WITHIN_TOLERANCE.

2. **C5 shows the largest hip-yaw drift** (0.1823 rad vs 0.0851). This is expected for a 3000-step run (1.5x longer than C1-C4). The value remains well below the absolute safety gate of 0.35 rad.

3. **focused_low_0p320 pitch RMS** barely exceeds the relative tolerance (+0.864 vs 0.849), missing by only 0.015 deg. This is a marginal failure.

4. **focused_high_0p480 is the best case** -- it passes pitch and hip-yaw tolerances and only fails on support_rms. The hip_yaw delta (+0.017 rad) is the smallest across all cases.

5. **C1-C4 are identical sim runs** -- the different height command patterns (ladder, random_500, random_200, abrupt) have no effect on the simulation when running at fixed height with the dedicated runner architecture.

#### Overall Step C class: SAFE_BUT_WORSE

All 7 cases complete without falling and without violating any absolute safety gate. However, every case exceeds the support_rms tolerance, and three individual tolerance breaches occur across pitch (focused_low) and hip_yaw (C5). The JAX dedicated runner exhibits systematically larger support polygon wander than the original Python controller, but the robot remains stable and does not fall in any scenario.

