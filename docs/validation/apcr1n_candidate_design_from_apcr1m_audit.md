# APCR1n Candidate Design from APCR1m Audit

## Phase 9: Decide Whether to Design APCR1n

### Executive Summary

**Decision: Design APCR1n**

Root cause of APCR1m's poor drift performance is clearly identified:
1. Wheel velocity damping is 3.5x larger than APCR1h (5.0 Nm vs 1.4 Nm)
2. Position torque cap at ±3 Nm is too restrictive (raw would be ±15 Nm)
3. tau_position saturated 77.3% during RECENTER

APCR1n should address these issues while preserving APCR1m's startup stability fix.

---

## Root Causes Identified

### 1. Wheel Velocity Damping Dominance (PRIMARY)
- APCR1m: 5.0 Nm abs_mean wheel velocity damping
- APCR1h: 1.4 Nm abs_mean wheel velocity damping
- **APCR1m has 3.5x larger wheel damping**
- This causes excessive wheel braking even when trying to recenter

### 2. Position Torque Cap Saturation (SECONDARY)
- tau_position capped at ±3 Nm
- Raw tau_position would be ±15 Nm
- **Saturated 77.3% of the time**
- During RECENTER: 87.3% saturated
- Limits recenter effectiveness

### 3. tau_pitch Fighting Drift (TERTIARY)
- tau_pitch has correct sign only 1.5% during RECENTER
- **tau_pitch fights drift 98.5% of the time**
- APCR1m's pitch blend reduces tau_pitch by 32.6% but still insufficient

---

## APCR1n Design Options

### Option A: APCR1n_Wheel_Damping_Override (Recommended)

**Concept**: Reduce wheel velocity damping during RECENTER when it fights drift.

**Changes**:
1. Add wheel velocity damping reduction factor during RECENTER
2. Scale wheel damping by 0.3 when RECENTER active AND final torque fights drift
3. Preserve wheel damping when it opposes drift

**Parameters**:
```python
vd_wheel_damping_recenter_scale = 0.3  # Reduce wheel damping during RECENTER
vd_wheel_damping_min = 0.5  # Minimum damping (leave some damping)
```

**Rationale**: APCR1h has 1.4 Nm wheel damping. APCR1n should target ~1.5-2.0 Nm during RECENTER.

---

### Option B: APCR1n_Position_Cap_Boost

**Concept**: Increase position torque cap during safe RECENTER.

**Changes**:
1. Increase tau_position cap from ±3 Nm to ±5 Nm during safe RECENTER
2. Use same safety gates as APCR1m (pitch, height, contact, roll)
3. Gradual ramp-up to avoid instability

**Parameters**:
```python
position_cap_normal = 3.0  # Nm
position_cap_recenter = 5.0  # Nm (during safe RECENTER)
position_cap_ramp_steps = 50  # Gradual increase
```

**Rationale**: Raw tau_position can reach ±15 Nm. A 5 Nm cap during RECENTER would provide 67% more authority.

---

### Option C: APCR1n_Final_Torque_Guard

**Concept**: Add a final torque direction check to prevent fighting drift.

**Changes**:
1. After composing all torques, check if final torque direction matches drift direction
2. If it fights drift, reduce tau_pitch and wheel damping proportionally
3. Allow APCR to dominate

**Parameters**:
```python
final_tau_direction_check_enabled = True
final_tau_fight_reduction_factor = 0.5  # Reduce components that fight drift
```

**Rationale**: 62.8% of RECENTER steps have final torque fighting drift. This would address the root symptom.

---

### Option D: APCR1n_Mixed_Torque_Blend (Comprehensive)

**Concept**: Clean blend of all torque components based on drift-priority state.

**Changes**:
1. Implement torque blending priority:
   - RECENTER from positive: APCR > position > pitch > wheel
   - RECENTER from negative: APCR > position > pitch > wheel
2. Dynamically scale lower-priority components
3. Use APCR as the primary drift correction signal

**Parameters**:
```python
torque_blend_priority = ["apcr", "position", "pitch", "wheel"]
torque_blend_decay_factor = 0.5  # Reduce lower priority by 50% per step
```

**Rationale**: Comprehensive solution that addresses all issues at once.

---

## Recommended APCR1n Design

**APCR1n = APCR1h base + Option A (wheel damping override) + Option B (position cap boost)**

### Design Principles

1. **Base**: Use APCR1h as the base (lowest drift, 1.4 Nm wheel damping)
2. **Preserve**: Keep APCR1m's startup guard (prevents startup instability)
3. **Fix**: Reduce wheel damping during RECENTER when it fights drift
4. **Fix**: Increase position cap during safe RECENTER

### Target Metrics

| Metric | APCR1h | APCR1m | APCR1n Target |
|--------|--------|--------|---------------|
| max \|e\| | 0.178m | 0.434m | < 0.200m |
| wheel damping | 1.4 Nm | 5.0 Nm | 1.5-2.0 Nm |
| position cap | ±3 Nm | ±3 Nm | ±5 Nm during RECENTER |
| P2P | 0.249m | 0.833m | < 0.300m |

---

## Implementation Plan

### Phase 1: Wheel Damping Override
1. Add `vd_wheel_damping_recenter_scale` parameter
2. Check if RECENTER active AND final torque fights drift
3. Reduce wheel damping by scale factor
4. Preserve minimum damping of 0.5 Nm

### Phase 2: Position Cap Boost
1. Add `position_cap_recenter` parameter
2. Use same safety gates as APCR1m
3. Apply during safe RECENTER only

### Phase 3: Startup Guard
1. Copy APCR1m's startup guard (100 steps)
2. No pitch blending during startup

### Phase 4: Testing
1. 1000-step validation at low_0p300
2. Compare against APCR1h and APCR1m
3. Verify no startup instability

---

## Files Generated

- `apcr1n_candidate_design_from_apcr1m_audit.md`
- `apcr1n_candidate_design_from_apcr1m_audit.json`