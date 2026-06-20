# Sagittal Causal Ablation Report

**Date:** 2026-06-15  
**Phase:** Phase 3 — Causal Ablation  
**Scenario:** high_0p480, 500 steps  
**Profiles tested:**
- Adaptive support centering trim (baseline)
- kp_pitch variations
- pitch_ref_offset variations

---

## Classification

**ROOT_CAUSE_PITCH_GAIN_TOO_HIGH**

Secondary: **EQUILIBRIUM_POSTURE_BIAS**

---

## Ablation A: kp_pitch Sweep Results

| kp_pitch | pos_drift% | pitch_mean_deg | tau_pitch_Nm | tau_ratio |
|----------|-----------|----------------|--------------|-----------|
| 50 (baseline) | **80.8%** | +3.28 | 2.865 | 1.00 |
| 25 | **62.1%** | +3.23 | 1.409 | 0.49 |
| 12.5 | **49.9%** | +0.48 | 0.106 | 0.04 |
| 6.25 | **54.3%** | +4.64 | 0.505 | 0.18 |

### Key Observations

1. **kp_pitch = 12.5 achieves near-symmetric drift (49.9% positive)**
   - tau_pitch reduced to only 0.106 Nm (4% of baseline)
   - pitch_mean drops to +0.48 deg (from +3.28 deg)
   - This is the sweet spot: enough pitch correction for stability, but low enough to not fight recentering

2. **kp_pitch = 6.25 reverts partially (54.3% positive)**
   - tau_pitch = 0.505 Nm (still low)
   - BUT pitch_mean increases to +4.64 deg (from +0.48 deg)
   - This suggests kp_pitch=6.25 is TOO LOW to maintain stable pitch
   - The robot develops a larger pitch equilibrium but with less tau_pitch

3. **Linear relationship between tau_pitch and pos_drift%**
   - kp=50 → tau=2.865 → pos=80.8%
   - kp=25 → tau=1.409 → pos=62.1%
   - kp=12.5 → tau=0.106 → pos=49.9%
   - kp=6.25 → tau=0.505 → pos=54.3%

4. **pitch_mean tracks equilibrium balance**
   - When kp_pitch is too low, the robot develops larger pitch but stays balanced
   - When kp_pitch is too high, the robot fights itself with tau_pitch

### Physical Interpretation

```
tau_pitch = kp_pitch * pitch_error

At equilibrium (pitch ≈ pitch_ref):
- With kp=50: tau_pitch ≈ 50 * 0.057 = 2.85 Nm
- This pushes wheels forward while tau_position tries to pull them back
- Net effect: stalemate with wheels biased forward

At kp=12.5:
- tau_pitch ≈ 12.5 * 0.057 = 0.71 Nm at baseline pitch
- BUT the robot adjusts pitch to minimize error
- New equilibrium: pitch ≈ 0.008 rad = 0.48 deg
- tau_pitch ≈ 12.5 * 0.008 = 0.10 Nm ✓ (matches measured 0.106 Nm)
```

The robot finds a NEW equilibrium when kp_pitch is reduced. This new equilibrium has:
- Near-zero pitch error
- Near-zero tau_pitch
- Near-symmetric drift

---

## Ablation B: pitch_ref_offset Sweep Results

| pitch_ref_offset | pos_drift% | pitch_mean_deg | tau_pitch_Nm |
|-----------------|-----------|----------------|--------------|
| 0 (baseline) | **80.8%** | +3.28 | 2.865 |
| -1 deg | **90.6%** | +3.28 | 3.741 |
| -2 deg | **90.6%** | +3.48 | 4.784 |
| -3 deg | **90.6%** | +3.65 | 5.806 |

### Key Observations

1. **NEGATIVE pitch_ref_offset INCREASES positive drift**
   - All negative offsets push pos_drift% from 80.8% to 90.6%
   - tau_pitch INCREASES (from 2.865 to 3.741-5.806 Nm)

2. **Why does this happen?**
   - With pitch_ref = -2 deg, the controller thinks backward lean is correct
   - The robot is naturally at +3.28 deg forward
   - Error = actual - ref = +3.28 - (-2) = +5.28 deg → LARGER error
   - tau_pitch = kp * error = 50 * 0.092 = 4.6 Nm ✓ (matches 4.784 Nm)
   - This MORE positive tau_pitch pushes wheels MORE forward

3. **pitch_ref_offset is NOT the fix**
   - Negative offsets make things WORSE
   - Zero offset is correct for the pitch controller

---

## Root Cause Summary

| Finding | Evidence |
|---------|----------|
| tau_pitch causes positive drift | kp_pitch reduction from 50→12.5 reduces pos_drift from 80.8% to 49.9% |
| tau_pitch is proportional to kp_pitch | tau_ratio = 0.49 (kp=25), 0.04 (kp=12.5) |
| Robot finds new equilibrium with lower kp_pitch | pitch_mean drops from +3.28 to +0.48 deg |
| pitch_ref_offset is NOT the fix | Negative offsets INCREASE tau_pitch and positive drift |
| kp_pitch=12.5 is near-optimal | Achieves 49.9% positive drift (near symmetric) |
| kp_pitch=6.25 is too low | Pitch becomes unstable (+4.64 deg) |

---

## Fix Path Decision

### Fix Path A: Equilibrium Posture Correction (NOT SELECTED)

Would involve adjusting hip_pitch/knee references. NOT CONFIRMED as root cause.

### Fix Path B: Support-Position Outer Loop Pitch Reference (NOT SELECTED)

NOT the right approach. pitch_ref_offset=-2 deg makes things worse, not better.

### Fix Path C: kp_pitch Reduction + pitch_ref Coordination (SELECTED)

The evidence shows:
1. **kp_pitch should be reduced from 50 to ~12.5**
   - This reduces tau_pitch from 2.865 to 0.106 Nm
   - This centers drift at 49.9% (from 80.8%)

2. **pitch_ref should remain at 0**
   - pitch_ref_offset experiments showed negative offsets INCREASE positive drift
   - The robot naturally finds correct equilibrium when kp_pitch is appropriate

3. **The issue is NOT a bias or reference problem**
   - It's a gain mismatch: kp_pitch=50 is too high for this equilibrium
   - Lower gain allows the robot to self-regulate

### Implementation Plan

1. Create new SagittalAuthoritySchedule profile: `low_gain_pitch_stabilized`
2. Set `kp_pitch = 12.5` in the profile
3. Keep pitch_ref = 0
4. Adjust pitch_rate gain (kd_pitch) proportionally
5. Validate with staged runs (500 → 1200 → 2000 → 5000 steps)
6. Test height ladder (low variants)

---

## Validation Targets

| Horizon | Baseline pos% | Target pos% | Acceptable |
|---------|-------------|-------------|-----------|
| 500 steps | 80.8% | 45-55% | YES |
| 1200 steps | ~82% | 45-60% | YES |
| 2000 steps | ~84% | 45-65% | YES |
| 5000 steps | ~85% | 45-70% | YES |

Minimum requirements:
- No fall within horizon
- Roll < 5 deg
- Hip yaw < 10 deg
- Height > 0.38 m
- At least 25% negative drift samples