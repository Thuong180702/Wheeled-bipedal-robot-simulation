# Phase B.9 Step 5.19: Controller Authority Reallocation

## Executive Summary

**Implementation**: PID output clamping to reserve actuator headroom for WBC corrections.

**Status**: Implementation complete, tests passing, evaluation pending.

**Root Cause (from Step 5.18c)**: PID outputs ~30 Nm and saturates actuators before WBC residuals (~1 Nm) can contribute. Authority ratio: 1:30 (WBC:PID).

**Approach**: Clamp PID output to fraction of actuator range (e.g., 70% = ±21 Nm), reserving headroom (30% = ±9 Nm) for WBC residuals.

**Critical Architectural Concern**: Authority reallocation may not solve the fundamental problem. See analysis below.

---

## Implementation

### 1. Low-Level Control Modification

**File**: [wheeled_biped/sim/low_level_control.py:170-203](wheeled_biped/sim/low_level_control.py#L170-L203)

Added `pid_authority_fraction` parameter to `hybrid_pid_plus_torque_control`:

```python
def hybrid_pid_plus_torque_control(
    pid_ctrl: jnp.ndarray,
    normalized_torque_residual: jnp.ndarray,
    ctrl_min: jnp.ndarray,
    ctrl_max: jnp.ndarray,
    max_ctrl_fraction: float = 1.0,
    allow_mask: jnp.ndarray | None = None,
    pid_authority_fraction: float = 1.0,  # NEW
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # Clamp PID output to reserved fraction
    pid_fraction = jnp.clip(pid_authority_fraction, 0.0, 1.0)
    pid_limit_min = ctrl_min * pid_fraction
    pid_limit_max = ctrl_max * pid_fraction
    pid_clamped = jnp.clip(pid_ctrl, pid_limit_min, pid_limit_max)
    
    # Compute WBC residual
    residual = normalized_motor_torque_control(...)
    
    # Blend and clip to full range
    final = jnp.clip(pid_clamped + residual, ctrl_min, ctrl_max)
    return final, residual
```

**Backward compatible**: Default `pid_authority_fraction=1.0` preserves existing behavior.

### 2. Environment Integration

**File**: [wheeled_biped/envs/balance_env.py:138-147](wheeled_biped/envs/balance_env.py#L138-L147)

Added config parameter:

```python
self._pid_authority_fraction = float(torque_cfg.get("pid_authority_fraction", 1.0))
```

**File**: [wheeled_biped/envs/balance_env.py:440-448](wheeled_biped/envs/balance_env.py#L440-L448)

Updated call site:

```python
scaled_action, torque_residual_ctrl = hybrid_pid_plus_torque_control(
    raw_pid_ctrl,
    state.info["torque_residual_action"],
    self._ctrl_min,
    self._ctrl_max,
    self._torque_max_ctrl_fraction,
    self._torque_allow_mask,
    self._pid_authority_fraction,  # NEW
)
```

### 3. Test Configs

Created 6 test configs in `outputs/phase_b9_step5_19_controller_authority_reallocation/`:

| Config | PID Authority | Reserved for WBC | Notes |
|--------|---------------|------------------|-------|
| `pid_authority_1.0.yaml` | 100% (±30 Nm) | 0% | Baseline, no clamping |
| `pid_authority_0.9.yaml` | 90% (±27 Nm) | 10% (±3 Nm) | Conservative |
| `pid_authority_0.8.yaml` | 80% (±24 Nm) | 20% (±6 Nm) | Moderate |
| `pid_authority_0.7.yaml` | 70% (±21 Nm) | 30% (±9 Nm) | Balanced |
| `pid_authority_0.6.yaml` | 60% (±18 Nm) | 40% (±12 Nm) | Aggressive |
| `pid_authority_0.5.yaml` | 50% (±15 Nm) | 50% (±15 Nm) | Extreme |

All configs use Step 5.18c best WBC gains:
- `k_roll=20.0`, `k_roll_rate=2.0`
- `k_pitch=5.0`, `k_pitch_rate=0.5`
- `max_ctrl_fraction=0.5`

### 4. Tests

**File**: [tests/test_phase_b9_step5_19_controller_authority_reallocation.py](tests/test_phase_b9_step5_19_controller_authority_reallocation.py)

All 7 tests passing:
- ✓ Default backward compatibility
- ✓ PID output clamping
- ✓ Headroom reservation for WBC
- ✓ Bounds checking [0, 1]
- ✓ Action dimension unchanged
- ✓ ctrlrange respected
- ✓ Backward compatibility

---

## Architectural Analysis

### The Authority Suppression Problem

From Step 5.18c audit:

```
PID output:        ~30 Nm (saturates at ctrlrange limits)
WBC residual:      ~1 Nm (after max_ctrl_fraction=0.15 scaling)
Authority ratio:   1:30 (WBC:PID)
Saturation rate:   93.75%
Delivery fraction: <50% (WBC corrections clipped)
```

**Control flow**:
```
final_ctrl = clip(PID_ctrl + WBC_residual, ctrl_min, ctrl_max)
```

When PID saturates at ±30 Nm, WBC residuals are clipped away.

### The Authority Reallocation Solution

**Proposed fix**:
```
PID_clamped = clip(PID_ctrl, ±21 Nm)  # 70% of range
WBC_residual = normalized_action * 0.5 * 30 Nm  # ~2-3 Nm
final_ctrl = clip(PID_clamped + WBC_residual, ±30 Nm)
```

**Expected outcome**:
- PID limited to ±21 Nm
- WBC can add ±2-3 Nm without clipping
- Total authority: ±23-24 Nm (within ±30 Nm limit)
- WBC delivery fraction: ~100% (no clipping)

### Critical Architectural Concern

**The fundamental mismatch**:

1. **PID is trying to maintain posture** with large torques (~30 Nm)
   - Hip pitch/knee need ~20-30 Nm to hold the robot upright
   - Wheels need ~15 Nm for forward/backward balance
   - These are not "excessive" outputs—they're necessary for posture

2. **WBC is trying to add stabilization corrections** with small torques (~2 Nm)
   - Roll correction: ~1-2 Nm
   - Pitch correction: ~1-2 Nm
   - These are small perturbative corrections, not primary control

**The problem with clamping PID**:

If we clamp PID to 70% (±21 Nm):
- Hip pitch/knee lose 9 Nm of authority
- This may be insufficient to maintain posture
- Robot may collapse or fall faster
- WBC corrections (~2 Nm) cannot compensate for lost PID authority

**Analogy**:
- PID is the "main engine" keeping the robot upright
- WBC is the "trim tab" for fine corrections
- Clamping the main engine to make room for the trim tab may cause the robot to fall

### Why This Might Not Work

**Scenario 1: PID authority insufficient**
```
PID wants:     25 Nm (to maintain posture)
PID clamped:   21 Nm (70% limit)
WBC adds:      +2 Nm (stabilization)
Final:         23 Nm
Result:        Insufficient to maintain posture → robot falls
```

**Scenario 2: WBC corrections still too small**
```
PID clamped:   21 Nm
WBC adds:      +2 Nm
Final:         23 Nm
vs. needed:    30 Nm (what unclamped PID was trying to do)
Gap:           7 Nm deficit
Result:        Robot still falls, just with WBC "helping" slightly
```

**Scenario 3: Authority reallocation helps**
```
PID clamped:   21 Nm (sufficient for posture)
WBC adds:      +2 Nm (stabilization)
Final:         23 Nm (sufficient total)
Result:        Robot stabilizes with WBC corrections
```

### The Real Root Cause

The authority suppression is a **symptom**, not the root cause.

**Root cause**: The hybrid PID+WBC architecture has a fundamental mismatch:
- PID is a **primary controller** (maintains posture)
- WBC is a **corrective controller** (adds stabilization)
- But PID already saturates trying to maintain posture
- This means the robot is **marginally stable** even with full PID authority

**Implication**: If PID needs 30 Nm to barely keep the robot upright, and we clamp it to 21 Nm, the robot will fall—regardless of WBC corrections.

### Alternative Interpretations

**Optimistic view**: Maybe PID is "over-controlling" and saturating unnecessarily. If we clamp it, PID will learn to be more efficient, and WBC can handle the rest.

**Counter-argument**: PID gains were already tuned in prior steps. If PID could maintain posture with less authority, it would. The fact that it saturates suggests the robot is genuinely difficult to stabilize at h=0.60.

---

## Evaluation Plan

### Phase 1: Quick Evaluation (h=0.60)

Test all 6 candidates at h=0.60 for 5 episodes each.

**Metrics**:
- Survival time
- Fall rate
- Roll RMS
- Pitch RMS

**Baseline (Step 5.18c)**:
- Survival: 0.86s
- Fall rate: 0.80
- Roll RMS: 15.9 deg

**Success criteria**:
- Survival > 0.86s (improvement over baseline)
- Roll RMS ≤ 15.9 deg (no degradation)

### Phase 2: Full Validation (all heights)

Only for best 1-2 candidates from Phase 1.

**Heights**: 0.65, 0.60, 0.55, 0.50, 0.45, 0.40 m

**Baseline (reset-fixed)**:
- Survival: 3.8167s
- Fall rate: 0.8333

**Success criteria**:
- Survival > 3.8167s (beats reset-fixed baseline)
- Step 6 gate passes

---

## Predicted Outcomes

### Likely Outcome: Marginal or No Improvement

**Reasoning**:
1. PID saturation indicates robot is marginally stable
2. Clamping PID weakens primary control authority
3. WBC corrections (~2 Nm) too small to compensate
4. Robot falls faster with weakened PID

**Expected result**:
- `pid_authority_fraction=1.0`: survival ~0.86s (baseline)
- `pid_authority_fraction=0.9`: survival ~0.7-0.8s (slight degradation)
- `pid_authority_fraction=0.7`: survival ~0.5-0.6s (significant degradation)
- `pid_authority_fraction=0.5`: survival ~0.3-0.4s (severe degradation)

### Optimistic Outcome: Small Improvement

**Reasoning**:
1. PID is over-controlling and wasting authority
2. Clamping forces PID to be more efficient
3. WBC corrections fill the gap
4. Robot stabilizes with better authority allocation

**Expected result**:
- `pid_authority_fraction=0.8-0.9`: survival ~1.0-1.2s (+15-40% improvement)
- Best candidate beats h=0.60 baseline but not reset-fixed baseline

### Breakthrough Outcome: Significant Improvement

**Reasoning**:
1. Authority reallocation fundamentally changes control dynamics
2. WBC corrections become effective with reserved headroom
3. Robot achieves stable balance with hybrid control

**Expected result**:
- `pid_authority_fraction=0.7`: survival >3.8s (beats reset-fixed baseline)
- Step 6 gate passes

**Probability**: Low (~10-20%)

---

## Decision Framework

### If Evaluation Shows Improvement

**Marginal improvement** (survival 0.9-1.5s at h=0.60):
- Authority reallocation helps but insufficient for Step 6
- Recommendation: Combine with other approaches (dynamic gain scheduling, hierarchical arbitration)

**Significant improvement** (survival >3.8s across all heights):
- Authority reallocation solves the problem
- Recommendation: Proceed to Step 6

### If Evaluation Shows No Improvement or Degradation

**No improvement** (survival ~0.86s):
- Authority reallocation ineffective
- PID authority is necessary, not excessive
- Recommendation: Question hybrid PID+WBC architecture

**Degradation** (survival <0.7s):
- Clamping PID weakens primary control
- Robot falls faster
- Recommendation: Abandon authority reallocation approach

---

## Alternative Approaches (if authority reallocation fails)

### 1. Dynamic Gain Scheduling
Reduce PID gains when WBC is active:
```python
if abs(roll) > threshold:
    pid_gains *= 0.7  # reduce PID authority
    wbc_gains *= 1.5  # increase WBC authority
```

### 2. Hierarchical Arbitration
Explicit priority management:
```python
if near_fall:
    final = wbc_ctrl  # WBC takes over
else:
    final = pid_ctrl + wbc_residual  # normal hybrid
```

### 3. Frequency Separation
- Low-frequency PID (posture maintenance)
- High-frequency WBC (fast stabilization)

### 4. Architectural Redesign
Question whether hybrid PID+WBC is viable:
- Pure WBC control (no PID)
- Pure PID control (no WBC)
- Different hybrid architecture (e.g., cascaded control)

---

## Files Modified

1. [wheeled_biped/sim/low_level_control.py](wheeled_biped/sim/low_level_control.py) - Added `pid_authority_fraction` parameter
2. [wheeled_biped/envs/balance_env.py](wheeled_biped/envs/balance_env.py) - Wired config and call site
3. [tests/test_phase_b9_step5_19_controller_authority_reallocation.py](tests/test_phase_b9_step5_19_controller_authority_reallocation.py) - 7 tests, all passing

## Files Created

1. `outputs/phase_b9_step5_19_controller_authority_reallocation/pid_authority_*.yaml` - 6 test configs
2. [scripts/phase_b9_step5_19_authority_reallocation_evaluation.py](scripts/phase_b9_step5_19_authority_reallocation_evaluation.py) - Full evaluation script
3. [scripts/phase_b9_step5_19_quick_eval.py](scripts/phase_b9_step5_19_quick_eval.py) - Quick evaluation script

---

## Next Steps

1. **Run evaluation**: Execute `python scripts/phase_b9_step5_19_quick_eval.py`
2. **Analyze results**: Compare against predicted outcomes
3. **Make decision**: Based on decision framework above
4. **Update reports**: Document findings in Phase B.9 reports

---

## Conclusion

Authority reallocation via PID output clamping is **implemented and tested**, but **may not solve the fundamental problem**.

The root cause is not just authority suppression—it's that the robot is **marginally stable** even with full PID authority. Clamping PID may weaken primary control and cause faster falls.

**Evaluation is required** to determine whether authority reallocation helps, hurts, or has no effect.

**Step 6 remains BLOCKED** until a controller beats the reset-fixed baseline (3.8167s survival across all heights).
