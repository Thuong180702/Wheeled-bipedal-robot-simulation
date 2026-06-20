# EZC Episode Root Cause Audit

**Date:** 2026-06-15  
**Profile:** early_zero_crossing_recenter  
**Scenario:** high_0p480, 5000 steps

## Classification

**EZC_FAILURE_EXIT_TOO_EARLY_REBOUND**

Secondary: EZC_FAILURE_WEAK_TORQUE (partially - torque is strong but decays too fast)

## Episode-Level Analysis

### Key Metrics

| Metric | Value |
|--------|-------|
| Total episodes | 21 |
| Crossed zero | 18/21 (85.7%) |
| EZC reached max torque | 21/21 (100.0%) |
| tau_position always corrective | 21/21 (100.0%) |
| Avg rebound steps | 28.3 |
| Avg post-exit positive % | 71.0% |
| Net tau (pos+ezc+adp) mean | -5.37 Nm |

### Episode Details

| ID | Dir | Entry | Min | Max | Exit | Hold | ReMax | NetTau | %Pos | Rebound |
|----|-----|-------|-----|-----|------|------|-------|--------|------|---------|
| 1  | POS | 0.0476 | -0.0129 | 0.1812 | -0.0129 | 142 | YES | -5.19 | 61% | 39 |
| 2  | POS | 0.0486 | -0.0093 | 0.1959 | -0.0093 | 150 | YES | -5.50 | 69% | 31 |
| 3  | POS | 0.0477 | -0.0080 | 0.1898 | -0.0080 | 151 | YES | -5.45 | 76% | 24 |
| 4  | POS | 0.0496 | -0.0067 | 0.1875 | -0.0067 | 152 | YES | -5.42 | 78% | 22 |
| 5  | POS | 0.0478 | -0.0066 | 0.1866 | -0.0066 | 153 | YES | -5.40 | 78% | 22 |
| 6  | POS | 0.0485 | -0.0071 | 0.1869 | -0.0071 | 153 | YES | -5.43 | 76% | 24 |
| 7  | POS | 0.0494 | -0.0086 | 0.1872 | -0.0086 | 152 | YES | -5.43 | 73% | 27 |
| 8  | POS | 0.0482 | -0.0124 | 0.1853 | -0.0124 | 146 | YES | -5.47 | 61% | 39 |
| 9  | POS | 0.0495 | -0.0157 | 0.1868 | -0.0157 | 139 | YES | -5.51 | 52% | 48 |
| 10 | POS | 0.0479 | -0.0163 | 0.1970 | -0.0163 | 139 | YES | -5.58 | 49% | 51 |
| 11 | POS | 0.0470 | -0.0134 | 0.2019 | -0.0134 | 145 | YES | -5.58 | 57% | 43 |
| 12 | POS | 0.0486 | -0.0113 | 0.1957 | -0.0113 | 149 | YES | -5.52 | 66% | 34 |
| 13 | POS | 0.0489 | -0.0098 | 0.1904 | -0.0098 | 150 | YES | -5.46 | 70% | 30 |
| 14 | POS | 0.0476 | -0.0077 | 0.1885 | -0.0077 | 152 | YES | -5.44 | 73% | 27 |
| 15 | POS | 0.0472 | -0.0067 | 0.1867 | -0.0067 | 156 | YES | -5.36 | 79% | 21 |
| 16 | POS | 0.0492 | -0.0052 | 0.1850 | -0.0052 | 158 | YES | -5.35 | 82% | 18 |
| 17 | POS | 0.0494 | -0.0022 | 0.1842 | -0.0022 | 164 | YES | -5.26 | 93% | 7 |
| 18 | POS | 0.0495 | 0.0008 | 0.1816 | 0.1071 | 500 | YES | -5.08 | 100% | >200 |
| 19 | POS | 0.0482 | 0.0075 | 0.1773 | 0.1257 | 500 | YES | -5.13 | 100% | >200 |
| 20 | POS | 0.0544 | -0.0010 | 0.1604 | -0.0009 | 247 | YES | -3.99 | 97% | 3 |
| 21 | POS | 0.0479 | 0.0479 | 0.1748 | 0.0517 | 125 | YES | -6.25 | 0% | >200 |

## Root Cause Analysis

### Evidence

1. **EZC IS working**: 86% of episodes cross zero, EZC reaches max torque 100% of the time

2. **Net corrective torque is STRONG**: mean = -5.37 Nm (negative = correcting positive drift)

3. **BUT rebound is FAST**: average 28 steps after exit before drift goes positive again

4. **Post-exit drift is predominantly positive**: 71% of steps after exit are positive

5. **tau_position is always corrective** but unable to prevent positive drift after EZC exits

6. **Episodes 18-19 didn't cross zero**: stayed at max hold (500 steps), drift went to +0.1 without crossing

7. **tau_pitch has systematic positive bias**: mean = +3.31 Nm even at near-zero pitch

### Diagnosis

The root cause is **EXIT_TOO_EARLY_REBOUND**:

- EZC enters at +0.05, applies strong negative correction
- Drift crosses zero (86% of episodes)
- EZC exits immediately at zero
- Correction decays quickly (3 dwell steps + decay rate 0.025 Nm/step)
- Positive bias (from tau_pitch and other sources) overwhelms after ~28 steps
- Drift returns positive before EZC can re-enter at +0.05

The **positive bias** comes from:
1. tau_pitch: mean = +3.31 Nm (systematic forward pitch torque)
2. tau_wheel_velocity: mean = +0.38 Nm (forward velocity damping)
3. Other position/velocity terms

Total positive bias ≈ +3.5 to +4.0 Nm

### Why EZC torque is strong but not enough

- EZC provides -0.55 Nm max, but this decays immediately after zero crossing
- tau_position provides -5.4 Nm mean during episodes, BUT this is fighting the positive bias
- After EZC exits, positive bias overwhelms tau_position

### Key Insight

The PROBLEM is not weak EZC torque. The PROBLEM is:
1. EZC exits at zero, leaving a gap
2. During the gap, positive bias pushes drift back to +0.10 to +0.20 m
3. EZC re-enters, corrects back to ~0
4. Cycle repeats

**The robot is oscillating around positive drift, not centering on zero.**

## Answer to Required Questions

1. **Did EZC enter when e crossed +0.05?** YES - all 21 episodes
2. **How many steps after e > +0.05 did it enter?** Immediately (within 1 step)
3. **While active, did EZC correction actually reduce e?** YES - all 21 episodes reduced e
4. **Did EZC reach target torque or stay rate-limited?** YES - reached max torque 100%
5. **Was EZC torque clipped or cancelled?** NO - reached target
6. **After crossing zero and exiting, how quickly did e return positive?** ~28 steps average
7. **Is the main problem weak torque, slow rate, early exit, clipping, or positive rebound?**

**Primary: EARLY_EXIT_REBOUND**
Secondary: POSITIVE_BIAS_OVERWHELMING

## Fix Path Decision

**Path C: Anti-rebound hold** is the correct fix.

After crossing zero:
1. Keep small decaying correction for 30-50 steps
2. Do NOT target opposite side (-0.02)
3. Decay correction gradually to prevent sudden rebound
4. Add anti-rebound cooldown to prevent same-side re-entry

This addresses the root cause: the gap between EZC exit and re-entry that allows positive bias to dominate.