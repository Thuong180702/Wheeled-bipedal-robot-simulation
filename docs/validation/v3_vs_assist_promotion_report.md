# V3 vs V3+WBC Assist — Promotion Comparison Report

**Date:** 2026-07-21 07:09 UTC
**Total scenarios:** 48
**Excluded (invalid seed state):** 2
**Elapsed time:** 42.2 min

## 0. Excluded Scenarios — V3 Failed to Reach a Valid Standing Seed State

These scenarios were dropped before comparison because V3's own settle phase (from the model keyframe) did not produce a valid standing state — i.e. V3 itself fell over (roll/pitch beyond the hard limit or height below the floor threshold) before the V3-vs-Assist comparison could even begin. Including them would compare both arms starting from an already-fallen pose, which is not a meaningful test of either controller.

| Scenario | Suite | Final height (m) | Final |qvel| |
|----------|-------|:---:|:---:|
| step_c_high_tiny | step_c | 0.111 | 156.877 |
| step_c_high_small | step_c | 0.115 | 95.670 |

This is evidence that V3's standalone height-holding is not reliable across the tested height range — see the roll/pitch instability finding for root-cause discussion.

## 1. Executive Summary

| Metric | V3 Baseline | V3+WBC Assist | Verdict |
|--------|:----------:|:------------:|---------|
| **Total Falls** | 0 | 0 | ✅ SAFE |
| **Scenarios** | 48 | 48 | — |
| **Equivalent** | — | 48 | ✅ |
| **Improved** | — | 0 | ⬆️ |
| **Mixed** | — | 0 | ⚠️ |
| **Regressed** | — | 0 | ❌ |

## 2. Aggregate Metric Comparison

### 2.1 Tilt

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Pitch Rms Deg | 1.3631 | 1.3629 | 0.999820 | ✅ |
| Pitch Max Deg | 2.0745 | 2.0777 | 1.001555 | ✅ |
| Roll Rms Deg | 4.2319 | 4.2669 | 1.008282 | ✅ |
| Roll Max Deg | 7.1204 | 7.1721 | 1.007260 | ✅ |

### 2.2 Drift

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Planar Drift Final M | 0.0316 | 0.0319 | 1.009229 | ✅ |
| Planar Drift Max M | 0.0720 | 0.0726 | 1.008003 | ✅ |
| Yaw Drift Rms Deg | 1.1590 | 1.1523 | 0.994271 | ✅ |
| Yaw Drift Max Deg | 2.4536 | 2.4351 | 0.992439 | ✅ |

### 2.3 Posture

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Height Rms M | 0.5429 | 0.5424 | 0.999212 | ✅ |
| Height Error Rms M | 0.0037 | 0.0033 | 0.902632 | ⚠️ |
| Height Min M | 0.5383 | 0.5382 | 0.999744 | ✅ |
| Height Max M | 0.5449 | 0.5445 | 0.999255 | ✅ |

### 2.4 Vibration

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Torque Oscillation Rms | 0.6110 | 0.6075 | 0.994281 | ✅ |
| Jvel Oscillation Rms | 0.4720 | 0.4641 | 0.983397 | ✅ |
| Pitch Oscillation Rms | 0.0003 | 0.0003 | 1.002130 | ✅ |
| Roll Oscillation Rms | 0.0011 | 0.0011 | 1.004269 | ✅ |

### 2.5 Stability

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Survival Steps | 416.1458 | 416.1458 | 1.000000 | ✅ |
| Com Vel Rms | 0.0732 | 0.0735 | 1.003254 | ✅ |
| Ang Vel Rms | 0.1283 | 0.1287 | 1.003421 | ✅ |

### 2.6 Effort

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Torque Rms | 3.1832 | 3.1844 | 1.000366 | ✅ |
| Torque Max | 8.6782 | 8.6799 | 1.000196 | ✅ |
| Wheel Power Proxy | 3.3024 | 3.3123 | 1.003021 | ✅ |

### 2.7 LF Sway

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Pitch Lf Power Deg | 1.3633 | 1.3630 | 0.999794 | ✅ |

## 3. Results by Test Suite

### 3.1 Step E — Fixed-Height Balance (5 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.3596 | 1.3602 | 1.000398 |
| Roll Rms Deg | 3.7272 | 3.7655 | 1.010260 |
| Planar Drift Max M | 0.0400 | 0.0407 | 1.017479 |
| Yaw Drift Rms Deg | 0.7218 | 0.7240 | 1.003026 |
| Height Error Rms M | 0.0037 | 0.0033 | 0.882114 |
| Torque Oscillation Rms | 0.1271 | 0.1270 | 0.999171 |
| Pitch Oscillation Rms | 0.0001 | 0.0001 | 1.002898 |
| **Falls** | 0 | 0 | 1.00x |

### 3.2 Step C — Height Transitions (3 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.3609 | 1.3618 | 1.000718 |
| Roll Rms Deg | 3.7686 | 3.8111 | 1.011275 |
| Planar Drift Max M | 0.0414 | 0.0422 | 1.018174 |
| Yaw Drift Rms Deg | 0.7240 | 0.7262 | 1.003040 |
| Height Error Rms M | 0.0037 | 0.0032 | 0.867919 |
| Torque Oscillation Rms | 0.1648 | 0.1660 | 1.007439 |
| Pitch Oscillation Rms | 0.0001 | 0.0001 | 1.002699 |
| **Falls** | 0 | 0 | 1.00x |

### 3.3 Step D — Random Height Commands (5 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.3596 | 1.3602 | 1.000398 |
| Roll Rms Deg | 3.7272 | 3.7655 | 1.010260 |
| Planar Drift Max M | 0.0400 | 0.0407 | 1.017479 |
| Yaw Drift Rms Deg | 0.7218 | 0.7240 | 1.003026 |
| Height Error Rms M | 0.0037 | 0.0033 | 0.882114 |
| Torque Oscillation Rms | 0.1271 | 0.1270 | 0.999171 |
| Pitch Oscillation Rms | 0.0001 | 0.0001 | 1.002898 |
| **Falls** | 0 | 0 | 1.00x |

### 3.4 Single Push (50N) (20 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.4682 | 1.4674 | 0.999494 |
| Roll Rms Deg | 4.8948 | 4.9252 | 1.006227 |
| Planar Drift Max M | 0.1165 | 0.1169 | 1.003394 |
| Yaw Drift Rms Deg | 1.9743 | 1.9562 | 0.990819 |
| Height Error Rms M | 0.0034 | 0.0032 | 0.929539 |
| Torque Oscillation Rms | 1.2696 | 1.2611 | 0.993362 |
| Pitch Oscillation Rms | 0.0005 | 0.0005 | 1.002047 |
| **Falls** | 0 | 0 | 1.00x |

### 3.5 Random Push (20-120N) (15 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.2258 | 1.2254 | 0.999715 |
| Roll Rms Deg | 3.7770 | 3.8146 | 1.009935 |
| Planar Drift Max M | 0.0400 | 0.0407 | 1.017479 |
| Yaw Drift Rms Deg | 0.4503 | 0.4513 | 1.002278 |
| Height Error Rms M | 0.0039 | 0.0035 | 0.890540 |
| Torque Oscillation Rms | 0.1448 | 0.1447 | 0.999170 |
| Pitch Oscillation Rms | 0.0001 | 0.0001 | 1.002474 |
| **Falls** | 0 | 0 | 1.00x |

## 4. Per-Scenario Detailed Comparison

| Scenario | Suite | V3 Falls | Assist Falls | Pitch Ratio | Roll Ratio | Drift Ratio | Yaw Ratio | Height Ratio | Torque Osc Ratio | Class |
|----------|-------|:--------:|:------------:|:-----------:|:----------:|:-----------:|:---------:|:------------:|:----------------:|:-----:|
| randpush_high_small_backward_54N_seed203 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_backward_74N_seed201 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_64N_seed202 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_tiny_backward_54N_seed203 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_high_tiny_backward_74N_seed201 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_high_tiny_forward_64N_seed202 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_low_small_backward_54N_seed203 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_backward_74N_seed201 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_64N_seed202 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_tiny_backward_54N_seed203 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_low_tiny_backward_74N_seed201 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_low_tiny_forward_64N_seed202 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_nominal_backward_54N_seed203 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_nominal_backward_74N_seed201 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| randpush_nominal_forward_64N_seed202 | random_push | 0 | 0 | 0.999524 | 1.016559 | 1.029131 | 1.003797 | 0.817566 | 0.998616 | ✅ |
| push_high_small_backward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_forward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_left_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_right_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_tiny_backward_seed42 | single_push | 0 | 0 | 1.002992 | 1.014827 | 1.011623 | 0.998907 | 0.799597 | 1.025543 | ✅ |
| push_high_tiny_forward_seed42 | single_push | 0 | 0 | 0.996612 | 1.010268 | 1.033123 | 0.979925 | 1.001055 | 0.973894 | ✅ |
| push_high_tiny_left_seed42 | single_push | 0 | 0 | 0.997221 | 1.012499 | 0.979214 | 0.939259 | 0.858634 | 0.898551 | ✅ |
| push_high_tiny_right_seed42 | single_push | 0 | 0 | 1.000216 | 1.003687 | 1.007829 | 1.006702 | 0.878032 | 1.067516 | ✅ |
| push_low_small_backward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_forward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_left_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_right_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_tiny_backward_seed42 | single_push | 0 | 0 | 1.002992 | 1.014827 | 1.011623 | 0.998907 | 0.799597 | 1.025543 | ✅ |
| push_low_tiny_forward_seed42 | single_push | 0 | 0 | 0.996600 | 1.010341 | 1.033330 | 0.978121 | 1.001035 | 0.974915 | ✅ |
| push_low_tiny_left_seed42 | single_push | 0 | 0 | 0.997033 | 1.012466 | 0.979107 | 0.937844 | 0.858545 | 0.898368 | ✅ |
| push_low_tiny_right_seed42 | single_push | 0 | 0 | 1.000204 | 1.003716 | 1.007899 | 1.006692 | 0.878055 | 1.067473 | ✅ |
| push_nominal_backward_seed42 | single_push | 0 | 0 | 1.002992 | 1.014827 | 1.011623 | 0.998907 | 0.799597 | 1.025543 | ✅ |
| push_nominal_forward_seed42 | single_push | 0 | 0 | 0.996613 | 1.010267 | 1.033123 | 0.979760 | 1.001020 | 0.973883 | ✅ |
| push_nominal_left_seed42 | single_push | 0 | 0 | 0.997403 | 1.012487 | 0.979222 | 0.939811 | 0.858620 | 0.898618 | ✅ |
| push_nominal_right_seed42 | single_push | 0 | 0 | 1.000216 | 1.003687 | 1.007829 | 1.006702 | 0.878032 | 1.067516 | ✅ |
| step_c_low_small | step_c | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_c_low_tiny | step_c | 0 | 0 | 1.001207 | 1.016740 | 1.024880 | 1.004893 | 0.800527 | 1.015559 | ✅ |
| step_c_nominal | step_c | 0 | 0 | 1.000904 | 1.017122 | 1.029266 | 1.004152 | 0.798467 | 0.998032 | ✅ |
| step_d_high_small_seed42 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_high_tiny_seed42 | step_d | 0 | 0 | 1.000663 | 1.017100 | 1.029131 | 1.005044 | 0.803523 | 0.998618 | ✅ |
| step_d_low_small_seed42 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_low_tiny_seed42 | step_d | 0 | 0 | 1.000663 | 1.017100 | 1.029131 | 1.005044 | 0.803523 | 0.998618 | ✅ |
| step_d_nominal_seed42 | step_d | 0 | 0 | 1.000663 | 1.017100 | 1.029131 | 1.005044 | 0.803523 | 0.998618 | ✅ |
| step_e_high_small | step_e | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_e_high_tiny | step_e | 0 | 0 | 1.000663 | 1.017100 | 1.029131 | 1.005044 | 0.803523 | 0.998618 | ✅ |
| step_e_low_small | step_e | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_e_low_tiny | step_e | 0 | 0 | 1.000663 | 1.017100 | 1.029131 | 1.005044 | 0.803523 | 0.998618 | ✅ |
| step_e_nominal | step_e | 0 | 0 | 1.000663 | 1.017100 | 1.029131 | 1.005044 | 0.803523 | 0.998618 | ✅ |

## 5. Safety Gates

| Gate | Result |
|------|:------:|
| Assist falls ≤ V3 falls (0 ≤ 0) | ✅ PASS |
| Zero regressions | ✅ PASS |
| Zero safety failures | ✅ PASS |

## 6. Promotion Verdict

**Verdict:** **PROMOTE_READY** — V3+WBC Assist is safe to promote as equivalent to V3

### Key Findings

1. **Fall comparison:** Assist 0 vs V3 0 falls
2. **Regression count:** 0 scenarios
3. **Improvement count:** 0 scenarios
4. **Equivalent count:** 48 scenarios

---
*Generated by scripts/promote_v3_vs_assist.py*