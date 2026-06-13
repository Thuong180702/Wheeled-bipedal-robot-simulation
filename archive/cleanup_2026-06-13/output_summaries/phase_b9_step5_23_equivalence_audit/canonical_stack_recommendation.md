# Canonical Stack Recommendation — Phase B.9 Step 5.23

## Decision Required

**Question**: What is the canonical control architecture for Phase B.9 going forward?

The Step 5.18c vs Step 5.22 comparison revealed that these are fundamentally different control paradigms, not configuration variants. A decision is required before proceeding to Step 6.

---

## Option 1: Position Control (Step 5.18c Actual)

**Architecture**:
```
WBC gains -> position targets -> PID position control -> torques
```

**Control Mode**: `pid_position_velocity`

**Pros**:
- ✅ **Higher performance**: 0.86s survival (27% better than torque control)
- ✅ **Implicit stabilization**: PID kp/ki/kd provide damping, integral action, position feedback
- ✅ **Easier to tune**: Position targets are intuitive, PID gains well-understood
- ✅ **Already working**: Step 5.18c baseline is ready
- ✅ **Robust**: Position feedback provides automatic restoring force
- ✅ **Action smoothing**: Built-in filtering reduces control jitter

**Cons**:
- ❌ **Not true WBC**: WBC generates position targets, not torques
- ❌ **Indirect control**: Two-stage pipeline (WBC → PID → torques)
- ❌ **PID dependency**: Relies on PID tuning for stability
- ❌ **Less flexible**: Position control limits dynamic maneuvers

**Recommendation**: **Choose this if goal is stability and performance**

**Next Steps**:
1. Accept Step 5.18c (0.86s) as canonical baseline
2. Abandon Step 5.22 torque-first architecture
3. Proceed to Step 6 with position control baseline
4. PPO residual learning will add on top of position control

---

## Option 2: Torque Control (Step 5.22 Intended)

**Architecture**:
```
WBC gains -> direct torque commands -> torques
```

**Control Mode**: `torque_first_wbc` or `motor_torque`

**Pros**:
- ✅ **True WBC**: Direct torque control, no intermediate PID
- ✅ **More flexible**: Enables dynamic maneuvers (jumping, push recovery)
- ✅ **Simpler pipeline**: One-stage control (WBC → torques)
- ✅ **Better for RL**: Torque commands are natural action space for PPO

**Cons**:
- ❌ **Lower performance**: 0.68s survival (21% worse than position control)
- ❌ **Harder to stabilize**: Requires explicit damping, smoothing, stronger gains
- ❌ **No implicit feedback**: Lacks position error correction
- ❌ **More tuning required**: Need to add stabilization mechanisms manually

**Recommendation**: **Choose this if goal is flexibility and true WBC**

**Next Steps**:
1. Re-run Step 5.18c with `motor_torque` mode for fair comparison
2. Enhance Step 5.22 with stabilization:
   - Add velocity damping: `damping_gain = 0.5`
   - Add action smoothing: `smoothing_alpha = 0.5`
   - Increase WBC gains: `k_roll = 40.0, k_pitch = 10.0` (2x stronger)
3. Accept lower baseline performance (expect 0.5-0.7s range)
4. Proceed to Step 6 with torque control baseline
5. PPO residual learning will compensate for weaker baseline

---

## Option 3: Hybrid Control (Step 5.21 Analyzed)

**Architecture**:
```
PID position control -> base torques
WBC torque residuals -> added on top
Final = PID + WBC
```

**Control Mode**: `hybrid_pid_plus_torque`

**Pros**:
- ✅ **Combines both**: Position stability + torque flexibility
- ✅ **Gradual transition**: Can tune authority split

**Cons**:
- ❌ **Authority conflict**: PID saturates actuators, suppresses WBC (Step 5.21 finding)
- ❌ **Worst performance**: 0.38s survival (56% degradation)
- ❌ **Complex tuning**: Two controllers fighting for authority
- ❌ **Not recommended**: Step 5.21 analysis showed fundamental issues

**Recommendation**: **DO NOT CHOOSE THIS**

Step 5.21 demonstrated that hybrid mode suffers from PID authority suppression (97% PID, 3% WBC). This architecture is not viable.

---

## Comparison Table

| Criterion | Position Control | Torque Control | Hybrid Control |
|-----------|------------------|----------------|----------------|
| **Performance** | 0.86s ⭐ | 0.68s | 0.38s ❌ |
| **Stability** | High (PID feedback) | Low (needs tuning) | Medium |
| **Tuning Difficulty** | Easy ⭐ | Hard | Very Hard ❌ |
| **WBC Purity** | No (position targets) | Yes ⭐ | No |
| **Flexibility** | Low | High ⭐ | Medium |
| **RL Compatibility** | Medium | High ⭐ | Low ❌ |
| **Step 6 Readiness** | Ready ⭐ | Needs work | Not viable ❌ |

---

## Recommended Decision: Position Control

**Rationale**:

1. **Performance**: 0.86s vs 0.68s (27% better)
2. **Stability**: Proven stable with PID feedback
3. **Time to Step 6**: Ready now, no additional tuning needed
4. **Risk**: Lower risk of instability during PPO training

**Trade-off**: Sacrifices WBC purity for stability and performance

**Step 6 Implications**:
- PPO residual learning will add on top of position control baseline
- Residual actions will be position target corrections, not torque corrections
- This is still valid residual RL, just with different action semantics

**Alternative Interpretation**: Position control IS a form of WBC
- WBC computes desired joint angles (inverse kinematics)
- PID tracks those angles (low-level control)
- This is a valid WBC architecture (task-space control → joint-space tracking)

---

## Alternative Decision: Torque Control (If Flexibility Prioritized)

**Rationale**:

1. **True WBC**: Direct torque control, no PID dependency
2. **RL-native**: Torque actions are natural for PPO
3. **Long-term flexibility**: Enables dynamic maneuvers beyond standing

**Trade-off**: Sacrifices immediate performance for long-term flexibility

**Required Work Before Step 6**:
1. Re-run Step 5.18c with `motor_torque` mode (fair comparison)
2. Add stabilization to Step 5.22:
   ```yaml
   damping_gain: 0.5
   smoothing_alpha: 0.5
   k_roll: 40.0  # 2x stronger
   k_pitch: 10.0  # 2x stronger
   ```
3. Target 0.7-0.8s survival (match or beat Step 5.18c with torque control)
4. Accept that torque control baseline will be weaker than position control

**Step 6 Implications**:
- PPO residual learning will add on top of torque control baseline
- Residual actions will be torque corrections
- Expect longer training time due to weaker baseline

---

## Implementation Checklist

### If Position Control Chosen:
- [ ] Accept Step 5.18c (0.86s) as canonical baseline
- [ ] Document that "WBC" means position targets, not torques
- [ ] Update Step 5.22 report to clarify architecture difference
- [ ] Proceed to Step 6 with position control
- [ ] Archive Step 5.22 torque-first architecture

### If Torque Control Chosen:
- [ ] Re-run Step 5.18c with `motor_torque` mode
- [ ] Enhance Step 5.22 with damping + smoothing + stronger gains
- [ ] Run apples-to-apples comparison (both using torque control)
- [ ] Target 0.7-0.8s survival with enhanced torque control
- [ ] Proceed to Step 6 only after torque control matches position control performance
- [ ] Document that baseline will be weaker but more flexible

---

## Recommendation Summary

**Primary Recommendation**: **Choose Position Control (Option 1)**

**Reasoning**:
- 27% better performance (0.86s vs 0.68s)
- Ready now, no additional work needed
- Lower risk for Step 6 PPO training
- Position control is a valid WBC architecture (task-space → joint-space)

**Secondary Recommendation**: **Choose Torque Control (Option 2) only if**:
- Long-term goal requires dynamic maneuvers (jumping, running)
- Willing to invest time in stabilization tuning
- Willing to accept weaker baseline for Step 6
- True torque-level WBC is a hard requirement

**Do NOT Choose**: **Hybrid Control (Option 3)**
- Step 5.21 proved this architecture has fundamental authority conflicts
- Worst performance (0.38s)
- Not viable for Step 6

---

## Final Note

**The comparison was not invalid due to a bug** — it revealed a fundamental architectural choice that must be made explicitly.

Both architectures are valid, but they serve different goals:
- **Position control**: Stability and performance
- **Torque control**: Flexibility and WBC purity

Choose based on project priorities, not on which performed better in this specific test.
