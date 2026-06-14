# Phase B.9 Step 5.24 — Torque-First Stabilized WBC Summary

## Executive Summary

**CRITICAL FAILURE**: Low-gain stabilization components overwhelmed WBC authority, recreating the exact authority suppression problem identified in Step 5.21.

**Best Result**: damping_impedance
- Survival: 0.78s (+14.7% vs Step 5.22, -9.3% vs Step 5.18c)
- **WBC authority: 11.3%** (requirement: >70%)
- **WBC dominance: FAILED**

**Root Cause**: Stabilization gains (damping=1.0, impedance_kp=2.0) generated torques much larger than WBC proportional gains (k_roll=20, k_pitch=5), causing authority suppression.

**Conclusion**: Torque-first WBC with stabilization is NOT viable. Position control (Step 5.18c) is the correct canonical architecture.

---

## Evaluation Results

### Stabilization Ablation (9 candidates tested)

| Candidate | Survival (s) | WBC Auth % | Damping Auth % | Impedance Auth % | Saturation % |
|-----------|--------------|------------|----------------|------------------|--------------|
| baseline_pure_wbc | 0.64 | 100.0 | 0.0 | 0.0 | 80.15 |
| damping_light | 0.75 | 25.2 | 74.8 | 0.0 | 84.98 |
| damping_moderate | 0.74 | 13.3 | 86.7 | 0.0 | 87.16 |
| smoothing_light | 0.67 | 100.0 | 0.0 | 0.0 | 60.97 |
| smoothing_moderate | 0.67 | 100.0 | 0.0 | 0.0 | 64.78 |
| impedance_weak | 0.74 | 42.3 | 0.0 | 57.7 | 84.69 |
| damping_smoothing | 0.77 | 19.3 | 80.7 | 0.0 | 59.65 |
| **damping_impedance** | **0.78** | **11.3** | **72.2** | **16.5** | **85.70** |
| full_stabilized | 0.76 | 14.3 | 57.8 | 27.9 | 66.58 |

---

## Authority Suppression Analysis

### The Failure Pattern

**Step 5.21 finding**: PID position control (30 Nm) suppressed WBC torque residuals (~1 Nm) → 97% PID, 3% WBC

**Step 5.24 finding**: Damping + impedance stabilization suppressed WBC torque commands → 11.3% WBC, 88.7% stabilization

**Same root cause**: Adding stabilization components with gains comparable to or larger than the primary controller suppresses the primary controller's authority.

### Why Stabilization Overwhelmed WBC

**WBC torque magnitude** (proportional gains only):
- k_roll = 20.0 → for 2° roll error: torque = 20 * 0.035 rad = 0.7 (normalized)
- k_pitch = 5.0 → for 2° pitch error: torque = 5 * 0.035 rad = 0.175 (normalized)

**Damping torque magnitude** (damping_gain = 1.0):
- For joint velocity = 1.0 rad/s: damping_torque = -1.0 * 1.0 = -1.0 (normalized)
- Damping acts on ALL joints continuously, not just when errors exist

**Impedance torque magnitude** (impedance_kp = 2.0):
- For position error = 0.15 rad (knee bend): impedance_torque = 2.0 * 0.15 = 0.3 (normalized)
- Impedance acts on ALL joints continuously to restore nominal pose

**Result**: Damping and impedance generate continuous torques comparable to or larger than WBC's error-driven torques, overwhelming WBC authority.

---

## Why Stabilization Improved Survival Despite Suppressing WBC

**Paradox**: damping_impedance achieved 0.78s survival (best result) despite only 11.3% WBC authority.

**Explanation**: The stabilization components (damping + impedance) ARE doing the balancing work, not WBC.

**This is position control in disguise**:
- Impedance (kp=2.0) provides position feedback → restoring force toward nominal pose
- Damping (gain=1.0) provides velocity damping → oscillation suppression
- WBC (k_roll=20, k_pitch=5) provides weak orientation correction

**This is NOT torque-first WBC** - it's weak position control with WBC as a minor correction term.

---

## Comparison to Step 5.18c Position Control

### Step 5.18c (Position Control)
- **Architecture**: WBC gains → position targets → PID control → torques
- **Survival**: 0.86s
- **Control type**: Position control with PID stabilization
- **Authority**: WBC sets targets, PID provides tracking and stabilization

### Step 5.24 (Torque-First + Stabilization)
- **Architecture**: WBC torque + damping + impedance → torques
- **Survival**: 0.78s (best)
- **Control type**: Weak position control with WBC correction
- **Authority**: Stabilization dominates (88.7%), WBC is minor (11.3%)

**Conclusion**: Step 5.24's best result is functionally equivalent to position control, just implemented differently and performing worse (0.78s vs 0.86s).

---

## Why Torque-First WBC Cannot Work

### Fundamental Incompatibility

**Torque control requires**:
- Direct torque commands without position feedback
- Proportional gains strong enough to generate corrective torques
- No competing stabilization mechanisms

**Stability requires**:
- Damping to suppress oscillations
- Position feedback to prevent drift
- Integral action to eliminate steady-state error

**The conflict**: Adding stabilization to torque control suppresses the torque controller's authority, converting it back to position control.

### The Authority Dilemma

**Option 1**: Pure torque control (no stabilization)
- Result: 0.64s survival (Step 5.22 baseline)
- Problem: Unstable, oscillates, drifts

**Option 2**: Torque control + weak stabilization
- Result: 0.78s survival (Step 5.24 best)
- Problem: Stabilization suppresses WBC authority (11.3%)

**Option 3**: Torque control + very weak stabilization (10-100x smaller gains)
- Predicted result: Still unstable, minimal improvement
- Problem: Stabilization too weak to help, WBC gains still insufficient

**Option 4**: Torque control + much stronger WBC gains
- Predicted result: Higher saturation, still unstable
- Problem: Strong proportional gains alone don't provide damping or integral action

**No viable path forward for torque-first WBC.**

---

## Answers to Required Questions

### 1. Does low-gain stabilization improve torque-first WBC?

**YES** - survival improved from 0.64s (pure WBC) to 0.78s (damping+impedance).

**BUT** - improvement came at the cost of WBC authority suppression (100% → 11.3%).

### 2. Which stabilization components help most?

**Damping** (damping_gain=1.0): +17% survival (0.64s → 0.75s)
- But suppressed WBC authority to 13.3%

**Impedance** (impedance_kp=2.0): +16% survival (0.64s → 0.74s)
- But suppressed WBC authority to 42.3%

**Damping + Impedance**: +22% survival (0.64s → 0.78s)
- But suppressed WBC authority to 11.3%

**Smoothing**: Minimal impact (+5% survival), preserved WBC authority (100%)

### 3. Is WBC still dominant authority?

**NO** - WBC authority dropped to 11.3% in the best-performing configuration.

**Requirement**: >70% WBC authority
**Actual**: 11.3% WBC authority
**Verdict**: FAILED

### 4. Did saturation remain low?

**NO** - saturation increased with stabilization:
- Baseline pure WBC: 80.15%
- Damping+impedance: 85.70%

Stabilization increased saturation, not decreased it.

### 5. Is behavior now closer to pure RL balancing?

**NO** - behavior is closer to position control (Step 5.18c), not pure RL.

The damping+impedance configuration is functionally equivalent to weak position control with WBC correction.

### 6. Does stabilized torque-first outperform Step 5.22?

**YES** - 0.78s vs 0.68s (+14.7% improvement)

**BUT** - this is not a fair comparison because stabilized torque-first is no longer torque-first (only 11.3% WBC authority).

### 7. Can it beat Step 5.18c?

**NO** - 0.78s vs 0.86s (-9.3% degradation)

Step 5.18c position control remains superior.

### 8. Is this now the canonical humanoid-control architecture?

**NO** - this architecture failed the WBC dominance requirement and underperformed Step 5.18c.

**Canonical architecture**: Position control (Step 5.18c)

---

## Step 6 Implications

**Status**: BLOCKED

**Gate Requirement**: 3.8167s survival (reset-fixed baseline)

**Current Best**:
- Step 5.18c (position control): 0.86s
- Step 5.24 (stabilized torque-first): 0.78s

**Gap**: 2.96s improvement needed (78% improvement required)

**Outlook**: Neither architecture is close to Step 6 gate. PPO residual learning will be required regardless of architecture choice.

---

## Final Architectural Recommendation

### Accept Position Control as Canonical Architecture

**Rationale**:

1. **Performance**: 0.86s (Step 5.18c) vs 0.78s (Step 5.24 best) vs 0.68s (Step 5.22 pure torque)

2. **Stability**: Position control provides implicit stabilization without authority suppression

3. **Simplicity**: Position targets are the natural action space for WBC (inverse kinematics)

4. **Proven**: Step 5.18c already works and is ready for Step 6

5. **Torque-first failed**: Three attempts (Step 5.22, Step 5.23, Step 5.24) all failed to match position control

### Why Torque-First WBC Failed

**Step 5.22**: Pure torque control → 0.68s survival, unstable
**Step 5.23**: Identified control semantics mismatch (position vs torque)
**Step 5.24**: Added stabilization → 0.78s survival, but WBC authority suppressed to 11.3%

**Fundamental issue**: Torque control requires stabilization, but stabilization suppresses torque control authority.

**This is an architectural dead end.**

### Position Control IS Valid WBC

**Clarification**: Position control is a valid WBC architecture:
- WBC computes desired joint angles (inverse kinematics / task-space control)
- PID tracks those angles (low-level joint-space control)
- This is the standard humanoid control architecture

**Step 5.18c is WBC** - just with position targets instead of torque commands.

---

## Recommended Next Steps

### Immediate Action

1. **Accept Step 5.18c (0.86s) as canonical baseline**
2. **Abandon torque-first WBC architecture**
3. **Proceed to Step 6 with position control**

### Step 6 Preparation

**Baseline**: Step 5.18c position control (0.86s survival)

**Goal**: Add PPO residual learning on top of position control

**Action space**: Position target corrections (not torque corrections)

**Expected outcome**: PPO learns to correct WBC position targets to improve survival beyond 0.86s

### Archive Torque-First Work

**Step 5.22**: Pure torque-first WBC (0.68s) - archived
**Step 5.23**: Equivalence audit - documented
**Step 5.24**: Stabilized torque-first (0.78s) - archived

**Lesson learned**: Torque control is not viable for this robot without much stronger WBC gains or fundamentally different stabilization approach.

---

## Conclusion

Phase B.9 Step 5.24 attempted to improve torque-first WBC by adding low-gain stabilization components (damping, smoothing, impedance).

**Result**: Stabilization improved survival (+14.7%) but suppressed WBC authority to 11.3%, failing the >70% dominance requirement.

**Root cause**: Stabilization gains comparable to WBC gains overwhelm WBC authority, recreating the authority suppression problem from Step 5.21.

**Final verdict**: Torque-first WBC is not viable. Position control (Step 5.18c, 0.86s survival) is the correct canonical architecture for Phase B.9.

**Recommendation**: Accept position control as canonical and proceed to Step 6 with PPO residual learning on top of position control baseline.
