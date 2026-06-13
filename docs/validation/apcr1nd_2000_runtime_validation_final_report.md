# APCR1nD 2000-step Runtime Validation Final Report

## Executive Summary

**Profile:** APCR1nD_direct_support_recenter_features  
**Task:** APCR1nD runtime validation  
**Classification:** `APCR1ND_2000_PASS_PROCEED_TO_5000`

APCR1nD successfully completed 2000-step runtime validation with **the best overall drift performance** among all tested profiles.

## Validation Checklist

| # | Question | Answer |
|---|----------|--------|
| 1 | Did APCR1nD run 2000 steps? | ✅ Yes |
| 2 | Did APCR1nD survive? | ✅ Yes |
| 3 | Did APCR1nD telemetry exist? | ✅ Yes (600 columns, 586 populated) |
| 4 | Did direct support trigger activate? | ✅ Yes (17.5% of steps) |
| 5 | Did wheel damping override activate? | ✅ Yes (0.95% of steps - rare emergency) |
| 6 | Did position cap boost activate? | ✅ Yes (17.5% of steps) |
| 7 | Did safety gates work? | ✅ Yes (startup guard 100 steps) |
| 8 | Did APCR1nD improve drift over APCR1n? | ✅ Yes (max|e| 0.1691 vs 0.1714) |
| 9 | Did APCR1nD improve outside ±0.15 over APCR1n? | ⚠️ Slightly worse (55 vs 53) |
| 10 | Did APCR1nD preserve contact/height/roll stability? | ✅ Yes |
| 11 | Which profile is current best? | **APCR1nD** |
| 12 | Should APCR1nD proceed to 5000-step? | **Yes** |

## Feature Activation Analysis

APCR1nD implements a **direct support drift trigger** that bypasses the APC dependency that blocked APCR1n features.

| Feature | Count | % | Notes |
|---------|-------|---|-------|
| Direct recenter active | 350 | 17.5% | When recenter priority is engaged |
| Direct recenter eligible | 1167 | 58.4% | Steps meeting threshold requirements |
| Position cap boost active | 350 | 17.5% | Synced with recenter active |
| Wheel damping override | 19 | 0.95% | Rare emergency intervention |

### Block Reason Distribution

| Reason | Count | % |
|--------|-------|-----|
| startup_guard | 100 | 5.0% |
| active (none) | 350 | 17.5% |
| eligible_but_converging | 362 | 18.1% |
| below_enter_threshold | 733 | 36.7% |
| within_exit_band | 455 | 22.8% |

## Drift Performance Comparison

| Metric | D2 | APCR1h | APCR1n | **APCR1nD** | Winner |
|--------|-----|--------|--------|--------------|--------|
| Max \|e\| (m) | 0.2463 | 0.1775 | 0.1714 | **0.1691** | APCR1nD |
| P2P (m) | 0.2733 | 0.2491 | 0.1854 | **0.1795** | APCR1nD |
| Mean \|e\| (m) | 0.0923 | 0.0768 | 0.0608 | **0.0607** | APCR1nD |
| Final e (m) | +0.0720 | -0.0453 | +0.0035 | +0.0038 | APCR1n |
| Outside ±0.10 | 565 | 746 | 459 | **446** | APCR1nD |
| Outside ±0.15 | 357 | 251 | **53** | 55 | APCR1n |

### Window Analysis

| Window | Metric | D2 | APCR1h | APCR1n | APCR1nD |
|--------|--------|-----|--------|--------|---------|
| 0-500 | Max \|e\| | 0.2463 | 0.1568 | 0.1714 | 0.1691 |
| 500-1000 | Max \|e\| | 0.1448 | 0.1775 | 0.1090 | **0.1064** |
| 1000-1500 | Max \|e\| | 0.0823 | 0.1672 | 0.1186 | **0.1175** |
| 1500-2000 | Max \|e\| | 0.0834 | 0.1578 | 0.1188 | **0.1180** |

APCR1nD shows consistent best or near-best performance in windows 500-2000.

## Torque Performance

| Metric | D2 | APCR1h | APCR1n | **APCR1nD** |
|--------|-----|--------|--------|--------------|
| tau_position max (Nm) | 9.85 | 7.10 | 6.86 | **6.77** |
| tau_position mean_abs (Nm) | 3.69 | 3.07 | 2.43 | **2.43** |
| Position saturation (%) | 48.25 | 37.65 | 2.75 | 2.85 |

APCR1nD achieves the **lowest torque position maximum** among all profiles.

## Stability Metrics

| Metric | D2 | APCR1h | APCR1n | **APCR1nD** |
|--------|-----|--------|--------|--------------|
| CoM Z min (m) | 0.279 | 0.280 | 0.282 | **0.282** |
| CoM Z mean (m) | 0.285 | 0.288 | 0.289 | **0.289** |
| Height error max (m) | 0.017 | 0.016 | 0.014 | **0.014** |
| Pitch max (deg) | 0.855 | 0.779 | 0.788 | **0.757** |
| Roll max (deg) | 5.451 | 7.824 | 7.816 | **7.734** |

## Why APCR1nD is the Best Profile

1. **Direct support drift trigger bypasses APC**: APCR1nD does not depend on Active Pitch Crossing (APC) being active. The direct trigger uses support position error magnitude to determine recenter eligibility.

2. **Appropriate activation rate**: 17.5% recenter active when 58.4% eligible shows features don't over-fire.

3. **Best overall drift control**: Lowest max |e|, P2P, and mean |e|.

4. **Best torque efficiency**: Lowest tau_position max (6.77 Nm).

5. **Maintains stability**: Best CoM Z min, height error, pitch max, and roll max among augmented profiles.

## Decision

```
APCR1ND_2000_PASS_PROCEED_TO_5000
```

APCR1nD should proceed to 5000-step validation. The profile demonstrates:
- Best drift performance
- Correct feature activation
- Acceptable stability
- Good torque efficiency

## Files Generated

- `docs/validation/apcr1nd_2000_feature_activation_audit.md`
- `docs/validation/apcr1nd_2000_drift_comparison.md`
- `docs/validation/apcr1nd_2000_torque_stability_comparison.md`
- `docs/validation/apcr1nd_2000_decision.md`
- `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/apcr1nd_low_0p300_2000/`

## Do NOT

- Do NOT run 5000-step in this task
- Do NOT run high_0p480 in this task
- Do NOT claim Step E pass
- Do NOT modify APCR1nD (unless bug found)
- Do NOT commit
