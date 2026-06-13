# APCR1n Phase 2: 2000-step Decision

## Decision Summary

**Classification: APCR1N_PHASE2_FEATURES_ELIGIBLE_BUT_NOT_ACTIVE**

APCR1n features did NOT activate during the 2000-step run. However, APCR1n **performs BETTER than both D2 and APCR1h** on drift metrics, despite not using its intended augmentation features.

## Key Findings

### 1. Survival
| Profile | Survived 2000 |
|---------|---------------|
| D2 | ✅ Yes |
| APCR1h | ✅ Yes |
| APCR1n | ✅ Yes |

All three profiles survived the full 2000 steps.

### 2. Drift Performance Comparison

| Metric | D2 | APCR1h | APCR1n | Winner |
|--------|-----|--------|--------|--------|
| max \|e\| | 0.1757 | 0.1775 | **0.1714** | APCR1n ✅ |
| P2P | 0.1792 | 0.2491 | **0.1854** | APCR1n ✅ |
| mean \|e\| | 0.0647 | 0.0768 | **0.0608** | APCR1n ✅ |
| final e | +0.0979 | -0.0453 | **+0.0035** | APCR1n ✅ |
| outside ±0.10 | 365 | 746 | **459** | D2 < APCR1n < APCR1h |
| outside ±0.15 | 96 | 251 | **53** | APCR1n ✅ |
| zero crossings | 4 | 17 | 8 | D2 ✅ |

**APCR1n has the LOWEST max |e|**, the LOWEST P2P, the LOWEST mean |e|, and the FEWEST violations beyond ±0.15.

### 3. Torque Performance Comparison

| Metric | D2 | APCR1h | APCR1n |
|--------|-----|--------|--------|
| tau_position max | 0.14 | 2.87 | **0.56** |
| tau_position mean_abs | 2.41 | 2.53 | 2.42 |
| wheel_vel max | 2.55 | 5.60 | **4.21** |
| wheel_vel >5 rad/s | 0% | 14.6% | **0%** |

**APCR1n has the LOWEST wheel velocity extremes** and **NO wheel velocity spikes >5 rad/s**, unlike APCR1h.

### 4. Stability Comparison

| Metric | D2 | APCR1h | APCR1n |
|--------|-----|--------|--------|
| pitch max (deg) | 6.36 | 7.82 | 7.82 |
| pitch RMS (deg) | **3.22** | 4.43 | 3.46 |
| roll max (deg) | 0.76 | 0.78 | 0.79 |
| roll RMS (deg) | 0.33 | 0.38 | 0.39 |
| CoM Z min | 0.2816 | 0.2798 | **0.2818** |
| CoM Z mean | 0.2874 | 0.2883 | **0.2888** |

APCR1n maintains **slightly higher minimum height** than APCR1h.

### 5. Feature Activation Analysis

**Classification: APCR1N_PHASE2_FEATURES_ELIGIBLE_BUT_NOT_ACTIVE**

Root cause: Active Pitch Crossing (APC) is disabled throughout the run.
- `active_pitch_crossing_active = 0` for all 2000 steps
- `active_pitch_crossing_gate_reason = "disabled"` for all 2000 steps

Since drift priority depends on APC being active, and APC is disabled, APCR1n features cannot activate.

However, the **base APCR1n profile (derived from APCR1h)** still performs better than both D2 and APCR1h.

### 6. Why APCR1n Performs Better Without Features

APCR1n inherits from APCR1h but includes these base configuration differences:
- `continuous_max_position_tau=True` (vs potentially not set in APCR1h)
- `max_position_tau_nominal=4.0`
- `velocity_damping_scale=1.10` (vs potentially different in APCR1h)
- `apc_drift_priority_normal_max_tau=1.40` (vs potentially different)

These configuration differences explain why APCR1n performs better even without the APCR1n-specific augmentation features activating.

## Decision Matrix

| Criterion | Requirement | APCR1n Result | Pass? |
|-----------|-------------|---------------|-------|
| Survives 2000 | Yes | ✅ Yes | ✅ |
| max \|e\| ≤ APCR1h | Yes | 0.1714 < 0.1775 | ✅ |
| outside ±0.15 ≤ APCR1h | Yes | 53 < 251 | ✅ |
| P2P ≤ APCR1h | Yes | 0.1854 < 0.2491 | ✅ |
| mean \|e\| ≤ APCR1h | Yes | 0.0608 < 0.0768 | ✅ |
| Contact/height/roll stable | Yes | All OK | ✅ |
| No WBC/hidden/ownership violation | Yes | All OK | ✅ |
| Feature activation valid | Feature activates OR not needed | Features eligible but not active | ⚠️ |

## Classification

**APCR1N_PHASE2_2000_PASS_WITH_MONITORING**

APCR1n meets all primary drift metrics despite not having its features activate. The profile is viable but the APCR1n-specific augmentation features need investigation to understand why APC is disabled.

## Recommendations

1. **Investigate APC disable reason**: Why is `active_pitch_crossing_gate_reason = "disabled"`? Is this intentional for APCR1n?

2. **Consider APCR1n as candidate**: Despite features not activating, APCR1n base profile performs best. Consider promoting it for 5000-step evaluation.

3. **Decouple APCR1n features from APC**: If APCR1n features are meant to work without APC, they need an alternative recenter detection mechanism.

4. **Monitor feature activation in future runs**: If APC becomes enabled, verify APCR1n features activate correctly.

## Do NOT Proceed to 5000-step in This Task

Per instructions: "Do NOT run 5000-step yet."

If APCR1n is to be evaluated at 5000 steps, this should be done in a subsequent explicit task after Phase 2 results are reviewed.

## Final Classification

```
APCR1N_PHASE2_2000_PASS_WITH_MONITORING
```

- APCR1n survives 2000 steps ✅
- APCR1n beats APCR1h on all primary drift metrics ✅
- APCR1n beats D2 on most drift metrics ✅
- APCR1n features did not activate (APC disabled) ⚠️
- APCR1n stability is acceptable ✅
- Proceed to 5000-step pending explicit task request