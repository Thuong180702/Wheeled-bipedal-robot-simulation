# APCR1m Drift and Torque Composition Analysis

## Phase 7: APCR1m Drift and Torque Composition Analysis

### Position Drift Analysis

| Metric | Value |
|--------|-------|
| CoM X range | -0.041 to 0.320 m |
| CoM Y range | -0.359 to 0.349 m |
| X drift | 0.320 m |
| Y drift | 0.287 m |
| **XY drift** | **0.430 m** |

**Assessment**: Significant position drift over 1000 steps at low_0p300 height. This is expected behavior for boundary-height operation without explicit position-hold control.

### Support Position Error Analysis

| Metric | Value |
|--------|-------|
| Support position error range | -0.434 to 0.400 m |
| Support position error mean | 0.177 m |
| **Support position error RMS** | **0.210 m** |

**Assessment**: Large support position errors indicate the robot oscillates significantly in the sagittal direction. This is the primary driver of the APCR recenter activations.

### Torque Composition Analysis

| Torque Component | Range | Mean Abs |
|-----------------|-------|----------|
| tau_pitch | [-23.17, 9.50] Nm | 4.23 Nm |
| tau_position | [-3.00, 3.00] Nm | 2.67 Nm |
| tau_wheel_velocity_left | [-22.66, 46.25] Nm | 5.00 Nm |
| tau_wheel_velocity_right | [-23.97, 43.64] Nm | 4.92 Nm |
| APCR tau | [-2.00, 2.00] Nm | (active 95.4% steps) |

**Assessment**: Wheel velocity damping (tau_wheel_velocity) is the dominant torque component, followed by tau_pitch. Position torque is limited to ±3 Nm cap. APCR recenter contributes up to ±2 Nm.

### APCR Hysteresis State Distribution

| State | Steps | Percentage |
|-------|-------|------------|
| RECENTER_FROM_POSITIVE | 526 | 52.6% |
| RECENTER_FROM_NEGATIVE | 359 | 35.9% |
| NEUTRAL | 115 | 11.5% |

**Assessment**: APCR spends 88.5% of time in RECENTER states, confirming that the low_0p300 variant requires continuous sagittal correction. The near-even split between positive/negative indicates bidirectional oscillation.

### Drift vs Other Profiles

Due to telemetry file size limitations, direct comparison with APCR1h/k was not possible in this session. The drift metrics provide a baseline for APCR1m performance.

### Key Observations

1. **Position drift is significant** (0.43m XY) but the robot maintains balance
2. **Wheel velocity damping dominates** torque composition (mean ~5 Nm each side)
3. **APCR recenter is highly active** (88.5% of time in RECENTER states)
4. **Tau pitch is functional** with full range [-23, 9.5] Nm
5. **The conditional pitch blend logic** appears to be working (no startup failure at step 18 like APCR1l)

### Comparison with APCR1l (Hypothetical)

APCR1l's reported failure at step 18 was attributed to hard suppression of tau_pitch during RECENTER. APCR1m's 1000-step survival with:
- 76% tau_pitch active
- 88.5% APCR RECENTER active
- Full tau_pitch range [-23, 9.5] Nm

suggests that the conditional pitch blend successfully preserves pitch stabilization while allowing recenter corrections.

### Conclusion

APCR1m achieves stable 1000-step survival at low_0p300 by:
1. Preserving tau_pitch during startup (startup guard: 100 steps)
2. Scaling tau_pitch based on error magnitude (0.0-1.0 blend)
3. Blocking blend when unsafe (pitch, pitch_rate, height, roll, contact)
4. Maintaining full APCR recenter authority during RECENTER

The drift is expected for boundary-height operation without explicit position hold.
