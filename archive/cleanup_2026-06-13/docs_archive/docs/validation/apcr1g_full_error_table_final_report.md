# APCR1g Full Error Table Final Report

**Date:** 2026-06-09
**Profiles Compared:** D2, APCR1f, APCR1g
**Height:** low_0p300 (target 0.300 m)
**Steps:** 2000

## Metric Source Column

**Primary drift column used:** `active_pitch_crossing_signed_error_m`

This column was verified as a valid physical signed drift column per `apcr_metric_discipline_guard.md`.

## Summary

APCR1g shows a **significant drift regression** compared to both D2 and APCR1f. While APCR1g has better pitch stability and lower wheel velocity, it has much worse drift/P2P metrics.

**Classification:** `APCR1G_MORE_STABLE_BUT_DRIFT_WORSE`

---

## Main Error Table (2000 steps)

| Metric | D2 | APCR1f | APCR1g | APCR1g-D2 | APCR1g-APCR1f |
|--------|---:|-------:|-------:|----------:|--------------:|
| **Drift Metrics** |
| Min drift (m) | -0.0035 | -0.0494 | -0.0005 | +0.003 | +0.049 |
| Max drift (m) | 0.1757 | 0.1572 | **0.3722** | +0.196 | +0.215 |
| P2P (m) | 0.1792 | 0.2066 | **0.3726** | +0.193 | +0.166 |
| Max abs drift (m) | 0.1757 | 0.1572 | **0.3722** | +0.196 | +0.215 |
| Mean drift (m) | 0.0649 | 0.0564 | **0.1503** | +0.085 | +0.094 |
| Abs mean drift (m) | 0.0649 | 0.0672 | **0.1503** | +0.085 | +0.083 |
| Final drift (m) | 0.0984 | -0.0138 | 0.0931 | -0.005 | +0.107 |
| Positive % | 98.3% | 74.1% | **99.7%** | +1.4% | +25.6% |
| **Band Violations** |
| Outside ±0.08 | 771 (38.6%) | 829 (41.4%) | **1271 (63.5%)** | +24.9% | +22.1% |
| Outside ±0.10 | 365 (18.2%) | 652 (32.6%) | **1019 (50.9%)** | +32.7% | +18.3% |
| Outside ±0.12 | 148 (7.4%) | 427 (21.3%) | **731 (36.5%)** | +29.1% | +15.2% |
| Outside ±0.15 | 96 (4.8%) | 45 (2.2%) | **694 (34.7%)** | +29.9% | +32.5% |
| Values > 0.15 | 96 | 45 | **694** | +598 | +649 |
| Values < -0.15 | 0 | 0 | 0 | 0 | 0 |
| **Orientation** |
| Pitch RMS (deg) | 3.22 | 4.03 | **2.80** | -0.42 | -1.23 |
| Roll RMS (deg) | 0.33 | 0.37 | **0.34** | +0.01 | -0.03 |
| **Hip-Yaw** |
| Hip yaw max (rad) | 0.209 | 0.215 | **0.281** | +0.072 | +0.066 |
| **Wheel Velocity** |
| Wheel vel max (rad/s) | 4.39 | 5.44 | **4.20** | -0.19 | -1.24 |
| **Height/Contact** |
| CoM Z min (m) | 0.282 | 0.280 | 0.276 | -0.006 | -0.004 |
| CoM Z mean (m) | 0.287 | 0.288 | 0.284 | -0.003 | -0.004 |
| Double contact % | 100.0% | 100.0% | 100.0% | 0% | 0% |
| **APCR Active** |
| APC active % | 0.0% | 61.8% | **92.7%** | +92.7% | +30.9% |
| **Torque** |
| WBC torque max | 10.1 | 11.6 | **9.98** | -0.1 | -1.6 |
| Total torque max | 8.88 | 8.88 | 8.88 | 0 | 0 |

---

## Windowed Error Table (500-step windows)

### Drift by Window

| Window | D2 min | D2 max | D2 P2P | APCR1f min | APCR1f max | APCR1f P2P | APCR1g min | APCR1g max | APCR1g P2P |
|--------|-------:|-------:|-------:|-----------:|-----------:|-----------:|-----------:|-----------:|-----------:|
| 0-500 | -0.0035 | 0.1757 | 0.1792 | -0.0132 | 0.1572 | 0.1704 | -0.0005 | 0.3689 | **0.3694** |
| 500-1000 | 0.0024 | 0.1072 | 0.1048 | -0.0494 | 0.1508 | 0.2002 | 0.0263 | 0.3722 | **0.3458** |
| 1000-1500 | 0.0163 | 0.1053 | 0.0890 | -0.0340 | 0.1450 | 0.1790 | 0.0299 | 0.1111 | 0.0813 |
| 1500-2000 | 0.0173 | 0.1021 | 0.0848 | -0.0293 | 0.1396 | 0.1688 | 0.0261 | 0.1077 | 0.0816 |

### Band Violations (>0.15) by Window

| Window | D2 >0.15 | APCR1f >0.15 | APCR1g >0.15 |
|--------|---------:|-------------:|-------------:|
| 0-500 | 96 (19.2%) | 36 (7.2%) | **410 (82.0%)** |
| 500-1000 | 0 (0.0%) | 9 (1.8%) | **284 (56.8%)** |
| 1000-1500 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| 1500-2000 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

---

## Analysis

### Does APCR1g reduce max positive drift?

**NO.** APCR1g has max positive drift of **0.3722 m**, which is **2.4x worse** than APCR1f (0.1572 m) and **2.1x worse** than D2 (0.1757 m).

### Does APCR1g reduce P2P?

**NO.** APCR1g has P2P of **0.3726 m**, which is **1.8x worse** than APCR1f (0.2066 m) and **2.1x worse** than D2 (0.1792 m).

### Does APCR1g reduce outside ±0.15?

**NO.** APCR1g has **694 values (34.7%)** outside ±0.15, which is **15x worse** than APCR1f (45 values, 2.2%) and **7x worse** than D2 (96 values, 4.8%).

### Does APCR1g reduce outside ±0.10?

**NO.** APCR1g has **1019 values (50.9%)** outside ±0.10, which is **1.6x worse** than APCR1f (652 values, 32.6%) and **2.8x worse** than D2 (365 values, 18.2%).

### Does APCR1g cause lower height?

**YES, marginally.** APCR1g has CoM Z min of 0.276 m vs 0.280 m for APCR1f (4mm lower). This is a small regression.

### Does APCR1g cause higher wheel velocity?

**NO.** APCR1g has lower wheel velocity max (4.20 vs 5.44 rad/s for APCR1f). This is an improvement.

### Drift Convergence Pattern

The windowed analysis reveals a critical pattern:
- **Window 0-500**: APCR1g has massive drift (P2P=0.369, 82% >0.15)
- **Window 500-1000**: APCR1g still has significant drift (P2P=0.346, 56.8% >0.15)
- **Window 1000-1500**: APCR1g drift reduces to near-baseline (P2P=0.081, 0% >0.15)
- **Window 1500-2000**: APCR1g drift stays low (P2P=0.082, 0% >0.15)

This suggests APCR1g has an **early transient** where it drifts significantly, then stabilizes. However, the early transient is severe and potentially dangerous.

---

## APCR1g vs APCR1f Delta Summary

| Category | Metric | APCR1f | APCR1g | Delta | Verdict |
|----------|--------|-------:|-------:|------:|---------|
| **Drift (WORSE)** | Max drift (m) | 0.157 | 0.372 | +0.215 | WORSE |
| **Drift (WORSE)** | P2P (m) | 0.207 | 0.373 | +0.166 | WORSE |
| **Drift (WORSE)** | Outside ±0.15 % | 2.2% | 34.7% | +32.5% | WORSE |
| **Drift (WORSE)** | Outside ±0.10 % | 32.6% | 50.9% | +18.3% | WORSE |
| **Stability (BETTER)** | Pitch RMS (deg) | 4.03 | 2.80 | -1.23 | BETTER |
| **Stability (BETTER)** | Roll RMS (deg) | 0.37 | 0.34 | -0.03 | BETTER |
| **Efficiency (BETTER)** | Wheel vel max (rad/s) | 5.44 | 4.20 | -1.24 | BETTER |
| **Efficiency (BETTER)** | WBC torque max | 11.6 | 9.98 | -1.62 | BETTER |
| **Trade-off** | CoM Z min (m) | 0.280 | 0.276 | -0.004 | SLIGHTLY WORSE |
| **Trade-off** | Hip yaw max (rad) | 0.215 | 0.281 | +0.066 | WORSE |

---

## Recommendation

**Do NOT proceed to 5000-step validation for APCR1g.**

APCR1g is classified as `APCR1G_MORE_STABLE_BUT_DRIFT_WORSE`. While it shows improved pitch stability and lower wheel velocity, the drift regression is severe:

1. **Max drift 2.4x worse** than APCR1f
2. **P2P 1.8x worse** than APCR1f  
3. **34.7% of values exceed ±0.15** vs 2.2% for APCR1f
4. **Severe early transient** (first 1000 steps) with drift up to 0.372 m

The drift metrics violate the core requirement of the APCR system: keeping the support position error bounded.

### Best Profile for low_0p300

Based on this analysis, the best profile is:

| Priority | Profile | Reason |
|----------|---------|--------|
| 1st | **APCR1f** | Best balance of drift control (P2P=0.207, 2.2% >0.15) and stability (pitch RMS=4.03°) |
| 2nd | **D2** | Lowest P2P (0.179) but no APCR engagement |
| 3rd | **APCR1g** | Better stability but unacceptable drift (P2P=0.373, 34.7% >0.15) |

---

## Final Decision

```
APCR1G_MORE_STABLE_BUT_DRIFT_WORSE
```

APCR1g should NOT proceed to 5000-step validation.

The drift regression is disqualifying. APCR1g's improved pitch stability does not compensate for its severe drift behavior.

---

## Files Generated

- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_drift_metric_table/main_error_table_2000.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_drift_metric_table/main_error_table_2000.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_drift_metric_table/window_error_table_2000.json`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_drift_metric_table/window_error_table_2000.csv`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1g_drift_metric_table/classification.json`

## Telemetry Files

| Profile | Telemetry File |
|---------|----------------|
| D2 | `outputs/hierarchical_controller_sim/telemetry_1781015924.csv` |
| APCR1f | `outputs/hierarchical_controller_sim/telemetry_1781015926.csv` |
| APCR1g | `outputs/hierarchical_controller_sim/telemetry_1781015927.csv` |