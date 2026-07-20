# V3 vs V3+WBC Assist — Promotion Comparison Report

**Date:** 2026-07-20 07:46 UTC
**Total scenarios:** 223
**Excluded (invalid seed state):** 2
**Elapsed time:** 154.7 min

## 0. Excluded Scenarios — V3 Failed to Reach a Valid Standing Seed State

These scenarios were dropped before comparison because V3's own settle phase (from the model keyframe) did not produce a valid standing state — i.e. V3 itself fell over (roll/pitch beyond the hard limit or height below the floor threshold) before the V3-vs-Assist comparison could even begin. Including them would compare both arms starting from an already-fallen pose, which is not a meaningful test of either controller.

| Scenario | Suite | Final height (m) | Final |qvel| |
|----------|-------|:---:|:---:|
| step_c_high_tiny | step_c | 0.104 | 273.194 |
| step_c_high_small | step_c | 0.090 | 244.147 |

This is evidence that V3's standalone height-holding is not reliable across the tested height range — see the roll/pitch instability finding for root-cause discussion.

## 1. Executive Summary

| Metric | V3 Baseline | V3+WBC Assist | Verdict |
|--------|:----------:|:------------:|---------|
| **Total Falls** | 0 | 0 | ✅ SAFE |
| **Scenarios** | 223 | 223 | — |
| **Equivalent** | — | 223 | ✅ |
| **Improved** | — | 0 | ⬆️ |
| **Mixed** | — | 0 | ⚠️ |
| **Regressed** | — | 0 | ❌ |

## 2. Aggregate Metric Comparison

### 2.1 Tilt

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Pitch Rms Deg | 1.0880 | 1.0880 | 1.000035 | ✅ |
| Pitch Max Deg | 1.9982 | 1.9998 | 1.000765 | ✅ |
| Roll Rms Deg | 5.0351 | 5.0505 | 1.003061 | ✅ |
| Roll Max Deg | 9.6468 | 9.6703 | 1.002432 | ✅ |

### 2.2 Drift

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Planar Drift Final M | 0.1246 | 0.1250 | 1.003734 | ✅ |
| Planar Drift Max M | 0.1915 | 0.1924 | 1.004787 | ✅ |
| Yaw Drift Rms Deg | 1.7523 | 1.7431 | 0.994729 | ✅ |
| Yaw Drift Max Deg | 3.2652 | 3.2668 | 1.000495 | ✅ |

### 2.3 Posture

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Height Rms M | 0.5422 | 0.5418 | 0.999257 | ✅ |
| Height Error Rms M | 0.0048 | 0.0044 | 0.918703 | ⚠️ |
| Height Min M | 0.5376 | 0.5375 | 0.999759 | ✅ |
| Height Max M | 0.5450 | 0.5445 | 0.999180 | ✅ |

### 2.4 Vibration

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Torque Oscillation Rms | 0.4320 | 0.4331 | 1.002514 | ✅ |
| Jvel Oscillation Rms | 0.4855 | 0.4860 | 1.001128 | ✅ |
| Pitch Oscillation Rms | 0.0003 | 0.0003 | 1.000225 | ✅ |
| Roll Oscillation Rms | 0.0020 | 0.0020 | 0.997958 | ✅ |

### 2.5 Stability

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Survival Steps | 352.0179 | 352.0179 | 1.000000 | ✅ |
| Com Vel Rms | 0.1647 | 0.1646 | 0.999395 | ✅ |
| Ang Vel Rms | 0.2228 | 0.2222 | 0.997222 | ✅ |

### 2.6 Effort

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Torque Rms | 3.1994 | 3.2006 | 1.000369 | ✅ |
| Torque Max | 8.7620 | 8.7643 | 1.000263 | ✅ |
| Wheel Power Proxy | 7.4891 | 7.4675 | 0.997125 | ✅ |

### 2.7 LF Sway

| Metric | V3 (mean) | Assist (mean) | Ratio (A/V3) | Status |
|--------|:---------:|:-------------:|:------------:|:------:|
| Pitch Lf Power Deg | 1.0885 | 1.0884 | 0.999847 | ✅ |

## 3. Results by Test Suite

### 3.1 Step E — Fixed-Height Balance (5 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.1610 | 1.1626 | 1.001391 |
| Roll Rms Deg | 4.7986 | 4.8187 | 1.004179 |
| Planar Drift Max M | 0.1741 | 0.1752 | 1.006508 |
| Yaw Drift Rms Deg | 1.9411 | 1.9486 | 1.003878 |
| Height Error Rms M | 0.0048 | 0.0044 | 0.915052 |
| Torque Oscillation Rms | 0.3374 | 0.3407 | 1.009773 |
| Pitch Oscillation Rms | 0.0002 | 0.0002 | 0.996557 |
| **Falls** | 0 | 0 | 1.00x |

### 3.2 Step C — Height Transitions (3 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.2385 | 1.2392 | 1.000632 |
| Roll Rms Deg | 4.7162 | 4.7439 | 1.005866 |
| Planar Drift Max M | 0.1150 | 0.1150 | 0.999415 |
| Yaw Drift Rms Deg | 1.1956 | 1.1965 | 1.000687 |
| Height Error Rms M | 0.0036 | 0.0032 | 0.880113 |
| Torque Oscillation Rms | 0.2892 | 0.2803 | 0.969345 |
| Pitch Oscillation Rms | 0.0001 | 0.0001 | 0.997880 |
| **Falls** | 0 | 0 | 1.00x |

### 3.3 Step D — Random Height Commands (15 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.1610 | 1.1626 | 1.001391 |
| Roll Rms Deg | 4.7986 | 4.8187 | 1.004179 |
| Planar Drift Max M | 0.1741 | 0.1752 | 1.006508 |
| Yaw Drift Rms Deg | 1.9411 | 1.9486 | 1.003878 |
| Height Error Rms M | 0.0048 | 0.0044 | 0.915052 |
| Torque Oscillation Rms | 0.3374 | 0.3407 | 1.009773 |
| Pitch Oscillation Rms | 0.0002 | 0.0002 | 0.996557 |
| **Falls** | 0 | 0 | 1.00x |

### 3.4 Single Push (50N) (100 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 1.2327 | 1.2318 | 0.999306 |
| Roll Rms Deg | 5.1708 | 5.1850 | 1.002746 |
| Planar Drift Max M | 0.2146 | 0.2153 | 1.003198 |
| Yaw Drift Rms Deg | 2.2482 | 2.2223 | 0.988505 |
| Height Error Rms M | 0.0046 | 0.0042 | 0.918107 |
| Torque Oscillation Rms | 0.5369 | 0.5387 | 1.003307 |
| Pitch Oscillation Rms | 0.0006 | 0.0006 | 1.000954 |
| **Falls** | 0 | 0 | 1.00x |

### 3.5 Random Push (20-120N) (100 scenarios)

| Metric | V3 | Assist | Ratio |
|--------|:--:|:------:|:-----:|
| Pitch Rms Deg | 0.9241 | 0.9247 | 1.000642 |
| Roll Rms Deg | 4.9563 | 4.9716 | 1.003092 |
| Planar Drift Max M | 0.1741 | 0.1752 | 1.006508 |
| Yaw Drift Rms Deg | 1.2355 | 1.2392 | 1.003007 |
| Height Error Rms M | 0.0051 | 0.0047 | 0.920762 |
| Torque Oscillation Rms | 0.3502 | 0.3505 | 1.000722 |
| Pitch Oscillation Rms | 0.0002 | 0.0002 | 0.998376 |
| **Falls** | 0 | 0 | 1.00x |

## 4. Per-Scenario Detailed Comparison

| Scenario | Suite | V3 Falls | Assist Falls | Pitch Ratio | Roll Ratio | Drift Ratio | Yaw Ratio | Height Ratio | Torque Osc Ratio | Class |
|----------|-------|:--------:|:------------:|:-----------:|:----------:|:-----------:|:---------:|:------------:|:----------------:|:-----:|
| randpush_high_small_backward_54N_seed203 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_backward_74N_seed201 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_backward_88N_seed210 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_49N_seed212 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_49N_seed219 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_58N_seed211 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_62N_seed214 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_64N_seed202 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_69N_seed207 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_forward_91N_seed213 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_left_101N_seed204 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_left_43N_seed205 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_left_59N_seed206 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_left_78N_seed209 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_left_80N_seed218 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_left_93N_seed217 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_right_25N_seed215 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_right_38N_seed220 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_right_88N_seed208 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_small_right_93N_seed216 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_high_tiny_backward_54N_seed203 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_backward_74N_seed201 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_backward_88N_seed210 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_49N_seed212 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_49N_seed219 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_58N_seed211 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_62N_seed214 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_64N_seed202 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_69N_seed207 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_forward_91N_seed213 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_left_101N_seed204 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_left_43N_seed205 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_left_59N_seed206 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_left_78N_seed209 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_left_80N_seed218 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_left_93N_seed217 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_right_25N_seed215 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_right_38N_seed220 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_right_88N_seed208 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_high_tiny_right_93N_seed216 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_small_backward_54N_seed203 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_backward_74N_seed201 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_backward_88N_seed210 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_49N_seed212 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_49N_seed219 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_58N_seed211 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_62N_seed214 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_64N_seed202 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_69N_seed207 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_forward_91N_seed213 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_left_101N_seed204 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_left_43N_seed205 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_left_59N_seed206 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_left_78N_seed209 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_left_80N_seed218 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_left_93N_seed217 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_right_25N_seed215 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_right_38N_seed220 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_right_88N_seed208 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_small_right_93N_seed216 | random_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| randpush_low_tiny_backward_54N_seed203 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_backward_74N_seed201 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_backward_88N_seed210 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_49N_seed212 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_49N_seed219 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_58N_seed211 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_62N_seed214 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_64N_seed202 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_69N_seed207 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_forward_91N_seed213 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_left_101N_seed204 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_left_43N_seed205 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_left_59N_seed206 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_left_78N_seed209 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_left_80N_seed218 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_left_93N_seed217 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_right_25N_seed215 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_right_38N_seed220 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_right_88N_seed208 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_low_tiny_right_93N_seed216 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_backward_54N_seed203 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_backward_74N_seed201 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_backward_88N_seed210 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_49N_seed212 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_49N_seed219 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_58N_seed211 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_62N_seed214 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_64N_seed202 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_69N_seed207 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_forward_91N_seed213 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_left_101N_seed204 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_left_43N_seed205 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_left_59N_seed206 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_left_78N_seed209 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_left_80N_seed218 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_left_93N_seed217 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_right_25N_seed215 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_right_38N_seed220 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_right_88N_seed208 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| randpush_nominal_right_93N_seed216 | random_push | 0 | 0 | 1.001069 | 1.005153 | 1.010846 | 1.005012 | 0.867936 | 1.001203 | ✅ |
| push_high_small_backward_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_backward_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_backward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_backward_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_backward_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_forward_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_forward_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_forward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_forward_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_forward_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_left_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_left_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_left_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_left_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_left_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_right_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_right_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_right_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_right_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_small_right_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_high_tiny_backward_seed113 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_high_tiny_backward_seed201 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_high_tiny_backward_seed42 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_high_tiny_backward_seed77 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_high_tiny_backward_seed999 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_high_tiny_forward_seed113 | single_push | 0 | 0 | 1.000207 | 0.995845 | 0.995348 | 0.930199 | 0.864112 | 1.018367 | ✅ |
| push_high_tiny_forward_seed201 | single_push | 0 | 0 | 1.000207 | 0.995845 | 0.995348 | 0.930199 | 0.864112 | 1.018367 | ✅ |
| push_high_tiny_forward_seed42 | single_push | 0 | 0 | 1.000207 | 0.995845 | 0.995348 | 0.930199 | 0.864112 | 1.018367 | ✅ |
| push_high_tiny_forward_seed77 | single_push | 0 | 0 | 1.000207 | 0.995845 | 0.995348 | 0.930199 | 0.864112 | 1.018367 | ✅ |
| push_high_tiny_forward_seed999 | single_push | 0 | 0 | 1.000207 | 0.995845 | 0.995348 | 0.930199 | 0.864112 | 1.018367 | ✅ |
| push_high_tiny_left_seed113 | single_push | 0 | 0 | 1.000196 | 1.008803 | 1.007167 | 1.005096 | 0.870446 | 0.996054 | ✅ |
| push_high_tiny_left_seed201 | single_push | 0 | 0 | 1.000196 | 1.008803 | 1.007167 | 1.005096 | 0.870446 | 0.996054 | ✅ |
| push_high_tiny_left_seed42 | single_push | 0 | 0 | 1.000196 | 1.008803 | 1.007167 | 1.005096 | 0.870446 | 0.996054 | ✅ |
| push_high_tiny_left_seed77 | single_push | 0 | 0 | 1.000196 | 1.008803 | 1.007167 | 1.005096 | 0.870446 | 0.996054 | ✅ |
| push_high_tiny_left_seed999 | single_push | 0 | 0 | 1.000196 | 1.008803 | 1.007167 | 1.005096 | 0.870446 | 0.996054 | ✅ |
| push_high_tiny_right_seed113 | single_push | 0 | 0 | 0.999608 | 1.006338 | 1.003882 | 1.000966 | 0.872122 | 1.004654 | ✅ |
| push_high_tiny_right_seed201 | single_push | 0 | 0 | 0.999608 | 1.006338 | 1.003882 | 1.000966 | 0.872122 | 1.004654 | ✅ |
| push_high_tiny_right_seed42 | single_push | 0 | 0 | 0.999608 | 1.006338 | 1.003882 | 1.000966 | 0.872122 | 1.004654 | ✅ |
| push_high_tiny_right_seed77 | single_push | 0 | 0 | 0.999608 | 1.006338 | 1.003882 | 1.000966 | 0.872122 | 1.004654 | ✅ |
| push_high_tiny_right_seed999 | single_push | 0 | 0 | 0.999608 | 1.006338 | 1.003882 | 1.000966 | 0.872122 | 1.004654 | ✅ |
| push_low_small_backward_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_backward_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_backward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_backward_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_backward_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_forward_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_forward_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_forward_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_forward_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_forward_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_left_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_left_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_left_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_left_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_left_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_right_seed113 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_right_seed201 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_right_seed42 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_right_seed77 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_small_right_seed999 | single_push | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| push_low_tiny_backward_seed113 | single_push | 0 | 0 | 0.996873 | 1.006672 | 1.012224 | 0.996990 | 0.842475 | 1.005320 | ✅ |
| push_low_tiny_backward_seed201 | single_push | 0 | 0 | 0.996873 | 1.006672 | 1.012224 | 0.996990 | 0.842475 | 1.005320 | ✅ |
| push_low_tiny_backward_seed42 | single_push | 0 | 0 | 0.996873 | 1.006672 | 1.012224 | 0.996990 | 0.842475 | 1.005320 | ✅ |
| push_low_tiny_backward_seed77 | single_push | 0 | 0 | 0.996873 | 1.006672 | 1.012224 | 0.996990 | 0.842475 | 1.005320 | ✅ |
| push_low_tiny_backward_seed999 | single_push | 0 | 0 | 0.996873 | 1.006672 | 1.012224 | 0.996990 | 0.842475 | 1.005320 | ✅ |
| push_low_tiny_forward_seed113 | single_push | 0 | 0 | 1.000236 | 0.995879 | 0.995345 | 0.930056 | 0.864122 | 1.018267 | ✅ |
| push_low_tiny_forward_seed201 | single_push | 0 | 0 | 1.000236 | 0.995879 | 0.995345 | 0.930056 | 0.864122 | 1.018267 | ✅ |
| push_low_tiny_forward_seed42 | single_push | 0 | 0 | 1.000236 | 0.995879 | 0.995345 | 0.930056 | 0.864122 | 1.018267 | ✅ |
| push_low_tiny_forward_seed77 | single_push | 0 | 0 | 1.000236 | 0.995879 | 0.995345 | 0.930056 | 0.864122 | 1.018267 | ✅ |
| push_low_tiny_forward_seed999 | single_push | 0 | 0 | 1.000236 | 0.995879 | 0.995345 | 0.930056 | 0.864122 | 1.018267 | ✅ |
| push_low_tiny_left_seed113 | single_push | 0 | 0 | 1.000229 | 1.008454 | 1.007088 | 1.003366 | 0.869984 | 0.994833 | ✅ |
| push_low_tiny_left_seed201 | single_push | 0 | 0 | 1.000229 | 1.008454 | 1.007088 | 1.003366 | 0.869984 | 0.994833 | ✅ |
| push_low_tiny_left_seed42 | single_push | 0 | 0 | 1.000229 | 1.008454 | 1.007088 | 1.003366 | 0.869984 | 0.994833 | ✅ |
| push_low_tiny_left_seed77 | single_push | 0 | 0 | 1.000229 | 1.008454 | 1.007088 | 1.003366 | 0.869984 | 0.994833 | ✅ |
| push_low_tiny_left_seed999 | single_push | 0 | 0 | 1.000229 | 1.008454 | 1.007088 | 1.003366 | 0.869984 | 0.994833 | ✅ |
| push_low_tiny_right_seed113 | single_push | 0 | 0 | 0.999644 | 1.006299 | 1.003950 | 1.000768 | 0.872302 | 1.002228 | ✅ |
| push_low_tiny_right_seed201 | single_push | 0 | 0 | 0.999644 | 1.006299 | 1.003950 | 1.000768 | 0.872302 | 1.002228 | ✅ |
| push_low_tiny_right_seed42 | single_push | 0 | 0 | 0.999644 | 1.006299 | 1.003950 | 1.000768 | 0.872302 | 1.002228 | ✅ |
| push_low_tiny_right_seed77 | single_push | 0 | 0 | 0.999644 | 1.006299 | 1.003950 | 1.000768 | 0.872302 | 1.002228 | ✅ |
| push_low_tiny_right_seed999 | single_push | 0 | 0 | 0.999644 | 1.006299 | 1.003950 | 1.000768 | 0.872302 | 1.002228 | ✅ |
| push_nominal_backward_seed113 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_nominal_backward_seed201 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_nominal_backward_seed42 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_nominal_backward_seed77 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_nominal_backward_seed999 | single_push | 0 | 0 | 0.996866 | 1.006679 | 1.012233 | 0.996966 | 0.842469 | 1.005333 | ✅ |
| push_nominal_forward_seed113 | single_push | 0 | 0 | 1.000248 | 0.995815 | 0.995319 | 0.929969 | 0.864100 | 1.018430 | ✅ |
| push_nominal_forward_seed201 | single_push | 0 | 0 | 1.000248 | 0.995815 | 0.995319 | 0.929969 | 0.864100 | 1.018430 | ✅ |
| push_nominal_forward_seed42 | single_push | 0 | 0 | 1.000248 | 0.995815 | 0.995319 | 0.929969 | 0.864100 | 1.018430 | ✅ |
| push_nominal_forward_seed77 | single_push | 0 | 0 | 1.000248 | 0.995815 | 0.995319 | 0.929969 | 0.864100 | 1.018430 | ✅ |
| push_nominal_forward_seed999 | single_push | 0 | 0 | 1.000248 | 0.995815 | 0.995319 | 0.929969 | 0.864100 | 1.018430 | ✅ |
| push_nominal_left_seed113 | single_push | 0 | 0 | 0.999263 | 1.008912 | 1.007203 | 1.005415 | 0.871006 | 0.998029 | ✅ |
| push_nominal_left_seed201 | single_push | 0 | 0 | 0.999263 | 1.008912 | 1.007203 | 1.005415 | 0.871006 | 0.998029 | ✅ |
| push_nominal_left_seed42 | single_push | 0 | 0 | 0.999263 | 1.008912 | 1.007203 | 1.005415 | 0.871006 | 0.998029 | ✅ |
| push_nominal_left_seed77 | single_push | 0 | 0 | 0.999263 | 1.008912 | 1.007203 | 1.005415 | 0.871006 | 0.998029 | ✅ |
| push_nominal_left_seed999 | single_push | 0 | 0 | 0.999263 | 1.008912 | 1.007203 | 1.005415 | 0.871006 | 0.998029 | ✅ |
| push_nominal_right_seed113 | single_push | 0 | 0 | 0.997922 | 1.007036 | 1.005876 | 0.978357 | 0.872643 | 1.005787 | ✅ |
| push_nominal_right_seed201 | single_push | 0 | 0 | 0.997922 | 1.007036 | 1.005876 | 0.978357 | 0.872643 | 1.005787 | ✅ |
| push_nominal_right_seed42 | single_push | 0 | 0 | 0.997922 | 1.007036 | 1.005876 | 0.978357 | 0.872643 | 1.005787 | ✅ |
| push_nominal_right_seed77 | single_push | 0 | 0 | 0.997922 | 1.007036 | 1.005876 | 0.978357 | 0.872643 | 1.005787 | ✅ |
| push_nominal_right_seed999 | single_push | 0 | 0 | 0.997922 | 1.007036 | 1.005876 | 0.978357 | 0.872643 | 1.005787 | ✅ |
| step_c_low_small | step_c | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_c_low_tiny | step_c | 0 | 0 | 1.000874 | 1.009894 | 0.955379 | 0.969596 | 0.793213 | 0.869470 | ✅ |
| step_c_nominal | step_c | 0 | 0 | 1.001013 | 1.008000 | 1.015933 | 1.010336 | 0.808674 | 1.004216 | ✅ |
| step_d_high_small_seed113 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_high_small_seed42 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_high_small_seed999 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_high_tiny_seed113 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_high_tiny_seed42 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_high_tiny_seed999 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_low_small_seed113 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_low_small_seed42 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_low_small_seed999 | step_d | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_d_low_tiny_seed113 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_low_tiny_seed42 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_low_tiny_seed999 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_nominal_seed113 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_nominal_seed42 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_d_nominal_seed999 | step_d | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_e_high_small | step_e | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_e_high_tiny | step_e | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_e_low_small | step_e | 0 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | ✅ |
| step_e_low_tiny | step_e | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |
| step_e_nominal | step_e | 0 | 0 | 1.002319 | 1.006966 | 1.010846 | 1.006463 | 0.858420 | 1.016288 | ✅ |

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
4. **Equivalent count:** 223 scenarios

---
*Generated by scripts/promote_v3_vs_assist.py*