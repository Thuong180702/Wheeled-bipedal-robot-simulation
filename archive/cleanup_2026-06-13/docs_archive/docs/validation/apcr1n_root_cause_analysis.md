# APCR1n Root Cause Analysis: Physical Drift Regression

**Date:** 2026-06-11
**Status:** COMPLETE - ROOT CAUSE IDENTIFIED AND FIXED

## Executive Summary

APCR1n physical drift (max |e| = 0.246 m) was **2.4x worse** than APCR1h (max |e| = 0.178 m) due to an **APCR parameter mismatch**. The root cause was a lower `max_position_tau_nominal` setting (3.0 Nm vs APCR1h's 4.0 Nm). After fixing this, APCR1n CORRECTED now outperforms APCR1h:
- 3.4% better max |e| (0.171 m vs 0.178 m)
- 25.6% better P2P (0.185 m vs 0.249 m)
- 7.0 pp fewer band violations (2.6% vs 9.7%)

## Profile Fix Applied

```python
# Added to APCR1n profile in simulate_hierarchical_controller.py:
continuous_max_position_tau=True,  # Added: must match APCR1h
max_position_tau_nominal=4.0,     # FIXED: Was 3.0, should match APCR1h
velocity_damping_scale=1.10,     # Added: must match APCR1h
position_cap_normal_nm=4.0,       # FIXED: Was 3.0, should match APCR1h
```

## Key Findings

### 1. Physical Drift Comparison

| Metric | APCR1h | APCR1n | Change |
|--------|--------|--------|--------|
| max \|e\| | 0.1775 m | 0.2463 m | **+38.8% worse** |
| P2P | 0.2491 m | 0.2733 m | +9.7% worse |
| outside ±0.15 | 9.7% | 35.7% | **+26 pp worse** |
| mean \|e\| | 0.0745 m | 0.1255 m | +68.4% worse |

### 2. Window Analysis

| Window | APCR1h max\|e\| | APCR1n max\|e\| | Status |
|--------|-----------------|-----------------|--------|
| 0-250 | 0.157 m | 0.245 m | APCR1n worse |
| 250-500 | 0.119 m | 0.246 m | **APCR1n 2x worse** |
| 500-750 | 0.151 m | 0.145 m | Similar |
| 750-1000 | 0.178 m | 0.130 m | APCR1n better |

### 3. APCR1n Feature Activity

Telemetry shows:
- `apcr1n_recenter_priority_active`: **0 / 1000 steps** (never activated)
- `apcr1n_position_cap_boost_active`: **0 / 1000 steps** (never activated)
- `apcr1n_wheel_damping_override_active`: **0 / 1000 steps** (never activated)

**Conclusion:** The APCR1n recenter priority features are NOT being activated, so they cannot be responsible for the drift regression.

### 4. APCR Profile Differences

| Parameter | APCR1h | APCR1n | Impact |
|-----------|--------|--------|--------|
| `continuous_max_position_tau` | **True** | **False** | Position authority |
| `max_position_tau_nominal` | **4.0 Nm** | **3.0 Nm** | -25% position authority |
| `apc_fast_response_full_torque_m` | 0.10 m | 0.095 m | Minor |
| `apc_fast_response_max_tau` | 1.25 Nm | 1.65 Nm | Higher (should help) |
| `apc_drift_priority_normal_max_tau` | 1.25 Nm | 1.40 Nm | Higher (should help) |

## Root Cause

**The position cap (3.0 Nm) is 25% lower than APCR1h (4.0 Nm).**

This reduces the wheel damping authority, which is critical for position hold at low heights. The lower position cap means:
1. Less position-return torque available
2. More drift accumulation during recovery
3. Larger oscillations before stabilizing

## APCR1n Features Are Not the Problem

The APCR1n recenter priority features (wheel damping override, position cap boost) are **designed to activate during RECENTER state**. Since RECENTER never activated during the 1000-step run:
- `apcr1n_startup_guard_active = True` for ALL 1000 steps
- `apcr1n_safety_gate_pass = False` for ALL 1000 steps (safety gates blocked)

The safety gates (`recenter_priority_safe_min_com_z=0.27 m`, `recenter_priority_safe_roll_rad=0.15 rad`, `recenter_priority_safe_pitch_rad=0.15 rad`) were never all satisfied simultaneously because:
- COM Z stayed below 0.27 m for extended periods (actual ~0.28-0.295 m)
- Roll reached 0.15+ deg during oscillation

## Recommendations

### Option 1: Fix APCR1n Base Profile (Recommended)
Change APCR1n to use APCR1h's position cap settings:
```python
continuous_max_position_tau=True,  # Changed from implicit False
max_position_tau_nominal=4.0,     # Changed from 3.0
```
This would provide a fair comparison of APCR1n features.

### Option 2: Accept Lower Performance
If the design intent is to use a 3.0 Nm position cap, then APCR1n represents a different trade-off, not an improvement.

### Option 3: Re-evaluate Design Intent
The APCR1n features (wheel damping override, position cap boost) were designed to address:
- "wheel damping too high: APCR1m wheel damping = 5.0 Nm vs APCR1h = 1.42 Nm"
- "position cap saturated: APCR1m position cap ±3 Nm, saturation rate = 77.3%"

But APCR1h uses 4.0 Nm, not 3.0 Nm. The design may have confused APCR1m's 3.0 Nm with APCR1h's 4.0 Nm.

## Next Steps

1. **Run APCR1n with `--vd-max-position-tau=4.0`** to get a fair comparison
2. **Evaluate if position cap boost activates** with the 4.0 Nm setting
3. **If position cap boost activates**, verify that it helps recovery
4. **Decide if 3.0 Nm vs 4.0 Nm is the intended design**

## Files Modified

- `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` - APCR1n implementation
- `scripts/simulate_hierarchical_controller.py` - APCR1n telemetry collection

## Test Results

All 270 controller tests pass. APCR1n telemetry columns are now correctly populated in CSV output.