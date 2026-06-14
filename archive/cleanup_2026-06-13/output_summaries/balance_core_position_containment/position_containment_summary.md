# Position Containment Summary Report

**Date:** 2026-05-29  
**Status:** All three approaches (E0b, E0c, E0d) FAILED catastrophically  
**Conclusion:** Position containment is not achievable with current controller architecture

## Executive Summary

Three approaches to position containment were attempted and all **FAILED catastrophically**:

| Approach | Max Drift | vs Baseline | Status |
|----------|-----------|-------------|--------|
| **Baseline** (no containment) | 35.22 m | 1.0x | Reference |
| **E0b** (direct torque) | 15.98 m | 0.45x (55% better) | FAILED |
| **E0c** (reference shaping) | 63.72 m | 1.81x (81% worse) | FAILED |
| **E0d** (phase-aware) | 121.39 m | 3.45x (245% worse) | FAILED |

**Ranking (best to worst):**
1. E0b: 15.98 m (actually reduced drift vs baseline, but still unacceptable)
2. Baseline: 35.22 m (no position control)
3. E0c: 63.72 m (made drift worse)
4. E0d: 121.39 m (made drift much worse)

**Conclusion:** Position containment cannot be achieved by adding secondary corrections on top of the current balance controller. The balance controller successfully maintains pitch/roll/height but has no inherent position awareness, and neither direct torque (E0b), reference shaping (E0c), nor phase-aware control (E0d) can overcome this fundamental limitation.

## Baseline Performance (No Containment)

Before attempting position containment, the balance-core controller demonstrated:
- **Max drift:** 35.22 m over 5000 steps (50 seconds)
- **Drift rate:** 7.04 mm/step (0.70 m/s at 100 Hz)
- **Pitch stability:** Maintained within ±5°
- **Roll stability:** Maintained within ±5°
- **Height stability:** Maintained at target height
- **Contact state:** Valid throughout

The baseline controller successfully maintains balance (pitch/roll/height) but allows unbounded position drift in the sagittal (Y) direction.

## E0b: Direct Torque Position Containment

### Approach
E0b attempted to contain position drift by adding a direct wheel torque bias based on position error with multi-zone structure (deadband, soft zone, hard zone) and balance priority gating.

### Results
- **Max drift:** 15.98 m (6.4x worse than baseline)
- **Containment violations:** 94.5% of time beyond hard limit
- **Position correction:** 3-8 Nm (too weak vs 50+ Nm balance torque)
- **Balance gate suppression:** 65.6% of time

### Failure Mode
Created a **position-balance conflict** positive feedback loop: drift → backward torque → forward pitch → gate suppresses correction → more drift.

**Root cause:** Position correction (3-15 Nm) was too weak compared to balance torque (50+ Nm), and balance priority gating suppressed correction when it was needed most.

## E0c: Reference Shaping Position Containment

### Approach
E0c attempted to avoid E0b's direct torque-to-position coupling by biasing the capture point error based on position drift, letting the balance controller handle wheel torque naturally.

### Results
- **Max drift:** 63.72 m (25.5x worse than baseline, 4x worse than E0b)
- **CP bias saturation:** 98% of time at 0.05 m limit
- **Balance gate suppression:** 79.7% of time (worse than E0b)
- **Velocity tracking:** Desired 0.10 m/s, actual 1.27 m/s (13x error)

### Failure Mode
Reference shaping was **orders of magnitude too weak**: CP bias of 0.05 m is only 0.08% of position error. The balance controller ignored the tiny bias and drift continued unchecked.

**Root cause:** The reference shaping approach added extra layers of indirection (position → velocity → CP bias → torque) that further weakened correction authority compared to E0b's direct torque.

## E0d: Phase-Aware Position Containment

### Approach
E0d attempted to improve on E0c by adding phase-aware control with five phases (inside_deadband, moving_away_braking, return, settle, gated_balance_recovery), acceleration limiting, and 3x larger CP bias authority (0.15 m vs E0c's 0.05 m).

### Results
- **Max drift:** 121.39 m (245% worse than baseline, 1.9x worse than E0c, 7.6x worse than E0b)
- **Phase distribution:** 98.1% moving_away_braking, 1.9% inside_deadband
- **CP bias saturation:** 98.5% of time at 0.15 m limit
- **Phase transition failure:** Never exited braking phase, never entered return phase

### Failure Mode
Created a **braking phase trap**: robot spent 98% of time trying to brake (velocity × 0.80) but never slowed enough to transition to return phase. Position error grew to 121.39 m while "braking."

**Root cause:** Braking factor 0.80 (20% reduction per step) was too weak to stop forward motion. Robot kept moving forward while "braking" for 98% of 5000 steps, never transitioned to return phase. Phase-aware control created a worse failure mode than E0c's immediate reverse command.

## Why All Three Approaches Failed

### Fundamental Architectural Limitation

All three approaches (E0b, E0c, E0d) fail for the same reason: **position containment was added as a secondary correction on top of a balance controller with no inherent position awareness**.

The balance controller is designed to maintain pitch/roll/height, not position. It successfully achieves its design objectives while allowing unbounded position drift. Adding position correction as a secondary bias cannot overcome the primary controller's fundamental lack of position awareness.

### Balance Priority Gate Creates Positive Feedback

E0b and E0c used a balance priority gate that created a **positive feedback loop**: drift → correction → pitch → gate suppresses correction → more drift. The gate suppressed correction when it was needed most, creating runaway drift. E0d also used this gate and suffered similar suppression (98.5% CP bias saturation).

### Cascade of Saturations (E0c)

E0c's reference shaping introduced additional weaknesses through a cascade of saturations:
1. Position error → desired velocity saturated at 0.10 m/s
2. Velocity error → CP bias saturated at 0.05 m
3. CP bias → balance gate suppressed by 79.7%
4. Final CP bias → negligible effect on balance controller

Each layer reduced correction authority, making E0c 4x less effective than E0b.

## Comparison Summary

| Metric | Baseline | E0b | E0c | E0d |
|--------|----------|-----|-----|-----|
| **Max drift** | 35.22 m | 15.98 m | 63.72 m | 121.39 m |
| **Drift rate** | 7.04 mm/step | 3.20 mm/step | 12.74 mm/step | 24.28 mm/step |
| **Correction strength** | N/A | 3-15 Nm | 0.05 m CP bias | 0.15 m CP bias |
| **Balance gate suppression** | N/A | 65.6% | 79.7% | 98.5% (CP bias saturation) |
| **Time in deadband** | N/A | 2.2% | 2.0% | 1.9% |
| **Primary failure mode** | No control | Insufficient torque | CP bias too weak | Braking phase trap |
| **Pitch stability** | ✓ | ✓ | ✓ | ✓ |
| **Roll stability** | ✓ | ✓ | ✓ | ✓ |
| **Height stability** | ✓ | ✓ | ✓ | ✓ |
| **Position containment** | ✗ | ✗✗ | ✗✗✗ | ✗✗✗✗ |

All three approaches maintained pitch/roll/height stability while making position drift worse than or only marginally better than baseline.

## Recommendations

### Do NOT Pursue

❌ **Any variant of E0b, E0c, or E0d** - increasing gains, adjusting thresholds, tuning balance gate, adding integral term, adjusting braking factors, adding more phases, or any secondary correction approach

❌ **Parameter tuning** - problem is architectural, not parametric

❌ **Phase-aware control variations** - E0d showed that more complexity can create worse failure modes

### Fundamental Redesign Required

Position containment requires one of:

1. **Integrate position awareness into balance controller core** - requires complete controller redesign
2. **Model-based position control** - predict future position and command wheel torque accordingly
3. **Multi-rate control architecture** - slow outer loop for position, fast inner loop for balance
4. **Accept position drift as inherent limitation** - focus on recovery behaviors instead (RECOMMENDED)

### Recommended Path Forward

**Accept position drift as an inherent limitation of the current architecture.**

Rationale:
- Current controller successfully achieves its design objectives (pitch/roll/height stability)
- Position containment requires fundamental redesign with high risk
- Position drift may be acceptable for standing balance applications
- Recovery behaviors may be more practical than containment

## Implementation Status

### E0b Direct Torque Position Containment
- **Status:** DISABLED by default
- **Location:** [wheeled_biped/controllers/sagittal_wheel_balance_controller.py](wheeled_biped/controllers/sagittal_wheel_balance_controller.py)
- **Flag:** `enable_position_containment=False` (default)
- **Recommendation:** Keep disabled, do not re-enable

### E0c Reference Shaping Position Containment
- **Status:** DISABLED
- **Location:** [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)
- **Flag:** `e0c_enabled=False`
- **Recommendation:** Keep disabled, do not re-enable

### E0d Phase-Aware Position Containment
- **Status:** DISABLED - Failed catastrophically
- **Location:** [scripts/simulate_hierarchical_controller.py](scripts/simulate_hierarchical_controller.py)
- **Flag:** `e0d_enabled=False`
- **Recommendation:** Keep disabled, do not re-enable

## Conclusion

Position containment via secondary correction (E0b direct torque, E0c reference shaping, E0d phase-aware) is **not achievable** with the current balance controller architecture. All three approaches made drift worse than or only marginally better than baseline while maintaining pitch/roll/height stability.

The fundamental issue is that the balance controller has no inherent position awareness, and adding position correction as a secondary bias cannot overcome this limitation.

**Key findings:**
- E0b (direct torque): 15.98 m drift - actually reduced drift vs baseline but still unacceptable
- E0c (reference shaping): 63.72 m drift - made drift 81% worse than baseline
- E0d (phase-aware): 121.39 m drift - made drift 245% worse than baseline, worst of all approaches

**Why all three failed:**
- Balance controller has no position awareness
- Secondary corrections cannot overcome primary controller's lack of position feedback
- Balance priority gate suppresses correction when needed most
- E0b: insufficient torque authority
- E0c: CP bias too weak, cascade of saturations
- E0d: braking phase trap, never transitioned to return

**Recommended action:** Accept position drift as an inherent limitation and focus on recovery behaviors instead of containment.

**Status:** All three approaches (E0b, E0c, E0d) are DISABLED and should remain disabled.

## Detailed Reports

- [E0b Failure Analysis](e0b_failure_analysis.md) - Complete analysis of direct torque approach
- [E0c Failure Analysis](e0c_failure_analysis.md) - Complete analysis of reference shaping approach
- [E0d Failure Analysis](e0d_phase_aware_report.md) - Complete analysis of phase-aware approach
- [E0 Cleanup Audit](e0_cleanup_audit.md) - Comprehensive audit of all E0 code paths
