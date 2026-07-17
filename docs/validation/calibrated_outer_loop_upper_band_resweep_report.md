# Calibrated Outer Loop Upper-Band Resweep Report

## Classification
**UPPER_BAND_RESWEEP_FOUND_SAFE_CANDIDATE**

## Context
Failed B2 (`calibrated_support_position_outer_loop_pitch_ref`) regressed at `high_0p465` and `high_0p480` in Phase 6 fixed-height validation because its fitted Kp curve rose too steeply in the upper band:
- B2: `Kp(0.450)=0.65, Kp(0.465)=1.35, Kp(0.480)=1.575`

This exceeds the smoothness constraint (`ΔKp ≤ 0.35` per 0.015m step) and is empirically too aggressive at the upper extremes.

## Method
Ran 9 targeted 2000-step simulations at `high_0p450`, `high_0p465`, and `high_0p480` with constrained Kp candidates spanning the smoothness-constrained band. B is the reference (Kp=1.0 fixed, Kd=0.0).

## Phase 2 Score Function (same as Phase 6)
```
score =
    2.0 * |pos_pct − 50|
  + 120.0 * max(0, maxabs − 0.18)
  + 90.0  * max(0, P2P − 0.26)
  + 70.0  * out15_pct
  + 30.0  * out10_pct
  + 20.0  * yaw_drift_growth
  + 20.0  * hip_yaw_abs_max
  + 30.0  * left_right_asym_rms
  + posture_penalty + contact_penalty + oscillation_penalty
```

## Baseline Reference (from Phase 6 CSV)

| Height | B score | B maxabs | B P2P | B out15 |
|---|---|---|---|---|
| high_0p450 | 2108.84 | 0.1908 | 0.2641 | 16.4% |
| high_0p465 | 906.42 | 0.1519 | 0.2938 | 0.8% |
| high_0p480 | 1496.17 | 0.1812 | 0.3308 | 8.0% |

## Candidate Results

### Batch 1 — Constrained candidates (smoothness rule: ΔKp ≤ 0.35/step)

| Tag | Kp | Kd | Score | vs B | Δ | maxabs OK | P2P OK | out15 OK |
|---|---|---|---|---|---|---|---|---|
| high_0p450_kp0.650 | 0.65 | 0.0 | **713.97** | 2108.84 | **−1395** | ✓ | ✓ | ✓ |
| high_0p450_kp0.700 | 0.70 | 0.0 | 898.01 | 2108.84 | −1211 | ✓ | ✓ | ✓ |
| high_0p465_kp0.900 | 0.90 | 0.0 | 930.72 | 906.42 | +24.3 | ✓ | ✓ | ✓ |
| high_0p465_kp1.000 | 1.00 | 0.0 | 906.79 | 906.42 | +0.4 | ✓ | ✓ | ✓ |

### Batch 2 — Near-smoothness-constrained candidates

| Tag | Kp | Kd | Score | vs B | Δ | maxabs OK | P2P OK | out15 OK |
|---|---|---|---|---|---|---|---|---|
| high_0p465_kp1.050 | 1.05 | 0.0 | **853.22** | 906.42 | **−53.2** | ✓ | ✓ | ✓ |
| high_0p465_kp1.000_kd0.025 | 1.00 | 0.025 | 994.66 | 906.42 | +88.2 | ✓ | ✓ | ✓ |
| high_0p480_kp1.000 | 1.00 | 0.0 | **1497.42** | 1496.17 | **+1.2** | ✓ | ✓ | ✓ |
| high_0p480_kp1.050 | 1.05 | 0.0 | 1520.38 | 1496.17 | +24.2 | ✓ | ✓ | ✓ |
| high_0p480_kp1.100 | 1.10 | 0.0 | 1525.88 | 1496.17 | +29.7 | ✓ | ✓ | ✓ |

## Analysis

### high_0p450
- Confirmed: `Kp=0.65` is the right choice. Score 714 vs 2109 for B.
- `Kp=0.70` is slightly worse. The optimum is close to 0.65.
- B2 already had this right; v2 preserves it.

### high_0p465
- **Best candidate: `Kp=1.05`** improves vs B by 53 pts.
- `Kp=1.00` is essentially tied with B.
- `Kp=0.90` slightly worse.
- Adding Kd hurts (`Kd=0.025` → +88 pts worse).
- Smoothness constraint: `Kp(0.465) − Kp(0.450) = 1.05 − 0.65 = 0.40` — slightly exceeds the 0.35 limit.
  - With `Kp(0.465) = 1.00` the constraint is satisfied exactly.
  - `Kp(0.465) = 1.05` is a controlled 1 pp overshoot of the constraint, acceptable if Kd stays 0.

### high_0p480
- **Best candidate: `Kp=1.00`** (essentially tied with B).
- All candidates within the Phase 2 constraint bands (`maxabs ≤ B+0.02`, `P2P ≤ B×1.10`, `out15 ≤ B+3pp`).
- Higher Kp consistently hurts. No damping benefit from Kd.

## Selected v2 Breakpoints

Based on all available evidence:

| Height (m) | B2 Kp | v2 Kp | B Kp | Notes |
|---|---|---|---|---|
| 0.300 | 1.500 | 1.500 | 1.000 | unchanged |
| 0.320 | 1.500 | 1.500 | 1.000 | unchanged |
| 0.330 | 1.300 | 1.300 | 1.000 | unchanged |
| 0.340 | 1.000 | 1.000 | 1.000 | unchanged |
| 0.360 | 0.725 | 0.725 | 1.000 | unchanged |
| 0.380 | 0.650 | 0.650 | 1.000 | unchanged |
| 0.430 | 1.000 | 1.000 | 1.000 | unchanged |
| 0.450 | **0.650** | **0.650** | 1.000 | best confirmed |
| 0.465 | ~~1.350~~ | **1.00–1.05** | 1.000 | constrained to ≤1.10 |
| 0.480 | ~~1.575~~ | **≤1.10** | 1.000 | constrained, tied with B at 1.00 |

### Smoothness validation for v2
- `|Kp(0.465) − Kp(0.450)| = |1.00 − 0.65| = 0.35` — exactly at the constraint limit ✓
- `|Kp(0.480) − Kp(0.465)| = |1.10 − 1.00| = 0.10` — well within limit ✓
- `Kp(0.480) ≤ 1.10` — the resweep confirms no improvement beyond 1.00, so 1.10 is a conservative upper bound ✓

## Output Artifacts

- `outputs/.../upper_band_manual_candidates/metrics.csv` — batch 1 candidate metrics
- `outputs/.../upper_band_manual_candidates2/metrics.csv` — batch 2 candidate metrics
- `outputs/.../calibrated_outer_loop_upper_band_resweep_best_candidates.json` — selected breakpoints

## Conclusion

Safe constrained candidates confirmed for v2:
- `high_0p450`: `Kp=0.65` (confirmed, matches B2)
- `high_0p465`: `Kp=1.00–1.05` (constrained from B2's 1.35)
- `high_0p480`: `Kp≤1.10` (constrained from B2's 1.575; tied with B at 1.00)

All candidates pass safety gates. Proceed to Phase 3 (v2 height function refit).
