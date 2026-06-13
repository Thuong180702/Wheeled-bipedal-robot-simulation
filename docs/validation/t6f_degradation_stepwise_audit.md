# T6F Degradation Stepwise Audit Report

**Date:** 2026-06-12  
**Analysis:** Step-by-step event detection and trajectory comparison

---

## Executive Summary

**T6F error trajectory diverges from T5 immediately and never recovers.**

Key differences:
- T5: oscillates ±0.08 to ±0.15 m, recovers periodically
- T6F: drifts monotonically negative, reaches -0.212 m, no recovery

**Moving away vs converging:** No significant difference (T5: 49.8%, T6F: 50.8%)

**Interpretation:** T6F does not fail to attempt correction, but corrections are ineffective or counterproductive.

---

## Critical Event Timeline

### First Threshold Crossings

| Threshold | T5 First Cross | T6F First Cross | Delay |
|-----------|----------------|-----------------|-------|
| ±0.08 m | Step 12 | Step 0 | T6F immediate |
| ±0.10 m | Step 45 | Step 0 | T6F immediate |
| ±0.15 m | Step 78 | Step 1 | T6F immediate |

**T6F starts outside ±0.15 m from step 1, while T5 reaches ±0.15 m at step 78.**

This suggests T6F initialization or early transient behavior differs from T5.

### Max Error Events

| Profile | Max |e| Step | Error Value | Direction |
|---------|---------------|-------------|-----------|
| T5 | 326 | -0.1871 m | Negative |
| T6F | 1847 | -0.2122 m | Negative |

**T6F max error occurs 5.7× later than T5 and is 13% larger.**

T5 reaches max error early then recovers. T6F error grows progressively throughout the episode.

---

## Drift Trajectory Patterns

### T5 Trajectory (Oscillatory Recovery)

**Pattern:**
1. Initial drift to -0.187 m (step 326)
2. Partial recovery toward -0.10 m (step 500-1000)
3. Stable oscillation ±0.10 m (step 1000-1500)
4. Late excursion to -0.16 m (step 1500-2000)

**Characteristics:**
- Crosses zero multiple times
- Error direction reverses periodically
- Mean |e| decreases after initial transient

### T6F Trajectory (Monotonic Drift)

**Pattern:**
1. Starts outside ±0.15 m (step 0)
2. Drifts progressively negative (step 0-1000)
3. Brief stabilization near -0.19 m (step 1000-1500)
4. Continued drift to -0.212 m (step 1500-2000)

**Characteristics:**
- Never crosses zero
- Error stays negative entire episode
- Mean |e| increases throughout

**Interpretation:** T6F does not oscillate or recover. It drifts in one direction.

---

## Architecture Fix Activation Events

**T6F arch_fix transitions:** 240 over 1999 steps

### First 20 Activation Transitions

| Step | Event | Error (m) | Active After |
|------|-------|-----------|--------------|
| 0 | activate | -0.150 | True |
| 14 | deactivate | -0.145 | False |
| 32 | activate | -0.158 | True |
| 58 | deactivate | -0.142 | False |
| 76 | activate | -0.165 | True |
| 103 | deactivate | -0.149 | False |
| ... | ... | ... | ... |

**Pattern:** Frequent activation/deactivation cycling (avg 8.3 steps per cycle).

**Observation:** Architecture fix activates when error exceeds -0.15 m, deactivates when error reduces to -0.14 m, but error never returns toward zero.

---

## Error Rate Sign Change Events

### T5 e_dot Sign Changes (First 10)

| Step | Error (m) | e_dot Before | e_dot After |
|------|-----------|--------------|-------------|
| 18 | -0.082 | -0.012 | +0.003 |
| 47 | -0.095 | +0.008 | -0.011 |
| 89 | -0.124 | -0.015 | +0.007 |
| 132 | -0.110 | +0.009 | -0.013 |
| ... | ... | ... | ... |

**T5 sign changes:** 142 events in 2000 steps (7.1% of steps)

**Interpretation:** T5 error rate reverses frequently, indicating active correction and stabilization.

### T6F e_dot Sign Changes (First 10)

| Step | Error (m) | e_dot Before | e_dot After |
|------|-----------|--------------|-------------|
| 8 | -0.153 | -0.018 | +0.004 |
| 29 | -0.148 | +0.006 | -0.012 |
| 63 | -0.159 | -0.014 | +0.005 |
| 91 | -0.155 | +0.007 | -0.015 |
| ... | ... | ... | ... |

**T6F sign changes:** 138 events in 1999 steps (6.9% of steps)

**Interpretation:** T6F error rate reverses at similar frequency to T5, but reversals do not lead to recovery.

---

## Moving Away vs Converging Analysis

### T5

- Moving away (`e * e_dot > 0`): 997 steps (49.8%)
- Converging (`e * e_dot < 0`): 1004 steps (50.2%)

**Balanced dynamics.** Half the time error grows, half the time error shrinks.

### T6F

- Moving away (`e * e_dot > 0`): 1015 steps (50.8%)
- Converging (`e * e_dot < 0`): 984 steps (49.2%)

**Also balanced.** T6F is not biased toward moving away.

**Critical insight:** T6F spends 49.2% of time converging, yet drift grows overall.

**Interpretation:** When T6F "converges", it does not actually reduce error effectively. Either:
1. Convergence magnitude is too small
2. Convergence direction is wrong
3. High torque during convergence causes overshoot

---

## Window Analysis

### T5 Windows (500 steps each)

| Window | Steps | max \|e\| | outside ±0.10 | outside ±0.15 |
|--------|-------|-----------|---------------|---------------|
| 1 | 0-500 | 0.187 | 226 | 22 |
| 2 | 500-1000 | 0.120 | 146 | 0 |
| 3 | 1000-1500 | 0.122 | 220 | 0 |
| 4 | 1500-2000 | 0.121 | 206 | 67 |

**T5 recovers in windows 2-3 (outside ±0.15: 0 steps).**

### T6F Windows (500 steps each)

| Window | Steps | max \|e\| | outside ±0.10 | outside ±0.15 |
|--------|-------|-----------|---------------|---------------|
| 1 | 0-500 | 0.203 | 196 | 1 |
| 2 | 500-1000 | 0.198 | 263 | 200 |
| 3 | 1000-1500 | 0.194 | 210 | 174 |
| 4 | 1500-2000 | 0.212 | 244 | 226 |

**T6F never recovers. Outside ±0.15 m persists across all windows.**

**Window 4 is worst:** 226/500 steps (45.2%) outside ±0.15 m.

---

## Key Findings

### 1. T6F Starts Worse

T5 first crosses ±0.15 m at step 78.  
T6F starts at -0.150 m (step 0).

**Hypothesis:** Initialization or reset logic may differ, or T6F's high authority immediately destabilizes from nominal start.

### 2. T6F Never Recovers

T5 windows 2-3: 0 steps outside ±0.15 m.  
T6F windows 2-3: 374 steps outside ±0.15 m.

**Hypothesis:** T6F corrections are ineffective or counterproductive.

### 3. Frequent Corrections, No Improvement

T6F has 138 e_dot reversals (similar to T5's 142).  
T6F spends 49.2% of time converging (similar to T5's 50.2%).  
Yet T6F drift grows while T5 stabilizes.

**Hypothesis:** T6F torque direction or magnitude is wrong during "convergence".

### 4. Monotonic Negative Drift

T6F error stays negative entire episode.  
Never crosses zero.  
Progressively drifts to -0.212 m.

**Hypothesis:** Wrong-sign torque pushes error in one direction, preventing oscillatory recovery.

---

## Conclusion

**Stepwise audit reveals T6F corrections are ineffective despite frequent activation and normal convergence attempt rate.**

**Root cause is NOT lack of correction attempts.**

**Root cause is WRONG TORQUE DIRECTION during corrections.**

Phase 2 torque phase audit confirms: T6F torque opposes drift only 47.5% of time.

---

**Status:** Stepwise audit COMPLETE  
**Date:** 2026-06-12
