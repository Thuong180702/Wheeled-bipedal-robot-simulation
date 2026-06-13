# T6 Architecture Fix Candidates Design

**Date:** 2026-06-12  
**Phase:** 5 of 11  
**Purpose:** Design and evaluate architecture-fix candidates to allow emergency recenter torque above 4.0 Nm upstream cap

---

## Executive Summary

**Root cause confirmed:** Upstream `max_position_tau_nominal=4.0` clips position torque BEFORE APCR1nD tuned cap layer at line 2353, preventing T5/T6B emergency authority (7.0/8.0 Nm) from reaching wheels.

**Design goal:** Allow emergency recenter torque > 4.0 Nm only when:
- Height is high (>= 0.45m)
- Drift is severe (hard/emergency band)
- APCR1n recenter is active
- All safety gates pass (contact/height/roll/pitch)

**Four candidates designed:** T6F, T6G, T6H, T6I

**Recommendation:** Implement **T6F_budget_cap_raise** first - safest, most testable, preserves all existing safety.

---

## Candidate A: T6F_budget_cap_raise

### Design Philosophy

Conditionally **raise** the upstream position torque cap during safe high-height emergency recenter, rather than bypassing it entirely.

### Behavior Specification

**Activation conditions (ALL must be true):**
1. `height_cmd >= 0.45` m (high-height regime)
2. `apcr1n_recenter_priority_active == True` (APCR1n engaged)
3. `tuned_band_state in {"hard", "emergency"}` (severe drift)
4. `apcr1n_safety_gate_pass == True` (contact/height/roll/pitch safe)

**Authority modification:**
```python
# Before line 2009 upstream clip
if t6f_arch_fix_enabled and all_activation_conditions_met:
    if tuned_band_state == "hard":
        effective_max_position_tau = max(
            effective_max_position_tau,  # scheduled 4.0 Nm at high heights
            6.5  # raised hard cap
        )
    elif tuned_band_state == "emergency":
        effective_max_position_tau = max(
            effective_max_position_tau,  # scheduled 4.0 Nm at high heights
            7.0  # raised emergency cap (conservative first)
        )

# Then apply line 2009 upstream clip as usual
tau_position = jnp.clip(tau_position_before_clip, -effective_max_position_tau, effective_max_position_tau)
```

**Key safety features:**
- Normal/soft bands: unchanged, still 4.0 Nm cap
- Desired band: unchanged, still 4.0 Nm cap
- Hard band: raised to 6.5 Nm only when safe
- Emergency band: raised to 7.0 Nm only when safe
- Final motor torque cap still respected (line ~2500)
- All existing APCR1n safety gates preserved

**Tuned cap interaction:**
After T6F raises upstream cap, the signal reaching line 2353 tuned cap layer can now be up to 6.5/7.0 Nm. The tuned cap then applies its own limits:
- T5 tuned emergency cap: 7.0 Nm → clips T6F hard (6.5) ✓, emergency (7.0) ✓
- T6B tuned emergency cap: 8.0 Nm → clips T6F hard (6.5) ✓, emergency (7.0) ✓

Both are compatible.

### Configuration Parameters

```yaml
# New T6F profile fields
arch_fix_enabled: true
arch_fix_type: "budget_cap_raise"
arch_fix_height_threshold_m: 0.45
arch_fix_hard_max_position_tau: 6.5
arch_fix_emergency_max_position_tau: 7.0
```

### Advantages

1. **Safest approach:** Still a bounded cap, not unlimited
2. **Preserves nominal safety:** 4.0 Nm during normal operation
3. **Testable:** Can verify cap value changes via telemetry
4. **Graduated:** Hard band gets less authority than emergency
5. **Conservative first step:** 7.0 Nm emergency is T5's tuned cap value (proven safe at line 2353)
6. **Reversible:** Can disable via config without code change
7. **Low implementation risk:** Minimal code change before existing clip

### Disadvantages

1. Requires careful gate logic implementation
2. Adds conditional branching to hot path (minimal JAX cost)
3. Still capped at 7.0 Nm - may not be enough if 8.0 Nm is truly needed

### Risk Assessment

**Low risk** if implemented correctly:
- Safety gates are already tested (APCR1n path)
- Cap values (6.5, 7.0) are within T5 tuned layer bounds
- Normal operation unchanged
- Can A/B test against T5

**Failure modes:**
- Gate logic bug → arch fix activates when unsafe → mitigated by thorough testing
- Raised cap insufficient → fall anyway → no worse than T5, just didn't help

### Implementation Complexity

**Low to Medium:**
- Add 5 config fields to `SagittalAuthoritySchedule`
- Add conditional logic before line 2009 clip
- Add 12 telemetry fields
- Add 15 targeted tests

**Estimated LOC change:** ~150 lines code, ~200 lines tests

---

## Candidate B: T6G_emergency_bypass_budget

### Design Philosophy

Fully **bypass** the upstream 4.0 Nm cap, but only in emergency band with all safety gates passing.

### Behavior Specification

**Activation conditions (ALL must be true):**
1. `height_cmd >= 0.45` m
2. `abs(e) >= emergency_band_m` (emergency band only, not hard)
3. `apcr1n_recenter_priority_active == True`
4. `apcr1n_safety_gate_hard_pass == True` (stricter: pitch HARD limit)

**Authority modification:**
```python
# Before line 2009 upstream clip
if t6g_arch_fix_enabled and all_activation_conditions_met:
    # Skip upstream clip entirely in emergency
    tau_position = tau_position_before_clip  # NO CLIP
else:
    # Normal path
    tau_position = jnp.clip(tau_position_before_clip, -effective_max_position_tau, effective_max_position_tau)
```

**Key differences from T6F:**
- Bypasses upstream clip completely (not just raised cap)
- Emergency band only (not hard band)
- Stricter safety: requires pitch HARD limit pass, not just soft

**Tuned cap still active:**
After bypass, signal goes to line 2353 tuned cap:
- T5 tuned emergency cap: 7.0 Nm
- T6B tuned emergency cap: 8.0 Nm

So maximum transmitted torque is still bounded by tuned cap layer.

### Configuration Parameters

```yaml
arch_fix_enabled: true
arch_fix_type: "emergency_bypass"
arch_fix_height_threshold_m: 0.45
arch_fix_emergency_band_threshold_m: 0.12
arch_fix_require_hard_safety: true
```

### Advantages

1. **Cleaner conceptually:** Either clip or don't, no intermediate values
2. **Full authority:** Tuned cap can use its full range (7.0 or 8.0)
3. **Simpler config:** Fewer parameters than T6F

### Disadvantages

1. **Higher risk:** Full bypass is more aggressive than bounded raise
2. **Binary:** No graduated authority for hard band
3. **Harder to debug:** If something goes wrong, less intermediate state to inspect

### Risk Assessment

**Medium risk:**
- Bypassing a safety cap is always riskier than raising it
- Mitigated by stricter safety gates (hard pitch limit)
- Mitigated by tuned cap layer still active downstream

**Failure modes:**
- Safety gate bug → bypass activates unsafely → higher consequence than T6F
- Tuned cap insufficient → full tau_position_before_clip reaches wheels → could be > 8.0 Nm

### Implementation Complexity

**Low to Medium:**
- Add 4 config fields
- Add bypass conditional before line 2009
- Add 10 telemetry fields
- Add 12 targeted tests

**Estimated LOC change:** ~120 lines code, ~180 lines tests

---

## Candidate C: T6H_pitch_reserve_blend

### Design Philosophy

If the 4.0 Nm cap exists to preserve pitch safety margin, **gradually reduce** that margin during high-drift emergency and **restore** it when drift resolves.

### Behavior Specification

**Blending logic:**
```python
# Compute blend factor based on drift magnitude
if abs(e) >= 0.12:  # emergency
    blend = 1.0  # full reduction of pitch reserve
elif abs(e) >= 0.08:  # hard/desired
    blend = smoothstep(abs(e), 0.08, 0.12)  # gradual
else:
    blend = 0.0  # no reduction, keep pitch reserve

# Blend from conservative 4.0 to aggressive 7.5
if height_cmd >= 0.45 and apcr1n_recenter_active:
    effective_max_position_tau = lerp(
        4.0,   # conservative pitch reserve
        7.5,   # aggressive emergency authority
        blend
    )
```

**Key features:**
- Smooth transition, not binary activation
- Automatically reduces when drift resolves
- Respects the pitch safety rationale

### Advantages

1. **Philosophically aligned:** Preserves pitch reserve when possible
2. **Smooth:** No abrupt authority jumps
3. **Self-regulating:** Automatically restores reserve as drift improves

### Disadvantages

1. **Unclear rationale:** Do we actually need pitch reserve blending?
2. **More complex:** Blending logic harder to reason about
3. **Harder to test:** Continuous blend, not discrete states
4. **Phase 4 didn't identify pitch as the issue:** Intent audit suggests stability concern, not specific pitch limit

### Risk Assessment

**Medium risk:**
- Continuous blending could create unexpected transients
- Harder to validate that all blend values are safe

### Implementation Complexity

**Medium:**
- Add blend computation
- Add 6 config fields (thresholds, targets)
- Add 10 telemetry fields
- Add 15 tests covering blend range

**Estimated LOC change:** ~180 lines code, ~220 lines tests

---

## Candidate D: T6I_height_scheduled_position_authority

### Design Philosophy

**Invert** the current height scheduling for emergency recenter: high heights need MORE position authority during emergency, not less.

### Behavior Specification

**Normal operation (unchanged):**
```python
# Normal/soft/desired bands use existing scheduling
effective_max_position_tau = scheduled_k_position(
    z_ref=schedule_height_ref,
    k_nominal=4.0,   # HIGH heights → less authority
    k_low_max=8.0,   # LOW heights → more authority
    ...
)
```

**Emergency operation (inverted):**
```python
if apcr1n_emergency_active and safety_gates_pass:
    # Emergency uses INVERSE scheduling
    effective_max_position_tau = scheduled_k_position(
        z_ref=schedule_height_ref,
        k_nominal=8.0,   # HIGH heights → MORE authority in emergency
        k_low_max=4.0,   # LOW heights → keep conservative
        ...
    )
```

**Rationale:**
- Low heights are stable → don't need aggressive emergency authority
- High heights are unstable → need MORE authority to prevent fall during emergency drift

### Advantages

1. **Architecturally elegant:** Reuses existing height scheduling infrastructure
2. **Height-aware:** Low heights remain conservative, only high heights get boost
3. **Intermediate heights interpolate smoothly:** 0.40-0.45m get graduated boost

### Disadvantages

1. **Inverts design philosophy:** Phase 6 fix said high heights need LESS authority for safety
2. **May be wrong:** If high heights truly need less authority, this makes it worse
3. **Affects all high heights uniformly:** Doesn't distinguish emergency severity

### Risk Assessment

**Medium to High risk:**
- Directly contradicts Phase 6 intent
- If Phase 6 was correct, this could cause instability at ALL high-height emergencies
- Harder to justify without understanding why Phase 6 chose conservative high-height cap

### Implementation Complexity

**Medium:**
- Add emergency scheduling path
- Add 4 config fields
- Add 8 telemetry fields
- Add 12 tests

**Estimated LOC change:** ~160 lines code, ~200 lines tests

---

## Candidate Comparison

| Criterion | T6F (cap raise) | T6G (bypass) | T6H (blend) | T6I (inverse schedule) |
|-----------|----------------|--------------|-------------|----------------------|
| **Safety** | Highest | Medium-High | Medium | Medium-Low |
| **Testability** | High | Medium-High | Medium | Medium |
| **Implementation risk** | Low | Low-Medium | Medium | Medium |
| **Preserves Phase 6 intent** | Yes | Yes | Partially | No |
| **Graduated authority** | Yes (hard/emergency) | No (emergency only) | Yes (continuous) | Yes (by height) |
| **Reversibility** | High | High | Medium | Medium |
| **Conceptual simplicity** | High | High | Medium | Low |
| **Conservative first step** | Yes | Partially | No | No |
| **Alignment with Phase 4** | Excellent | Excellent | Partial | Poor |

---

## Selection Rationale

### Recommended: T6F_budget_cap_raise

**Why T6F is the safest first candidate:**

1. **Preserves all existing safety:**
   - Normal/soft/desired bands unchanged
   - 4.0 Nm cap during routine operation
   - Only raises cap when ALL gates pass

2. **Conservative emergency authority:**
   - Emergency cap 7.0 Nm matches T5 tuned layer (already proven safe at line 2353)
   - Hard cap 6.5 Nm is intermediate value
   - Still bounded, not unlimited

3. **Testable and debuggable:**
   - Discrete states (normal/hard/emergency) easier to test than continuous blend
   - Telemetry clearly shows when arch fix activates
   - Easy to verify cap value changes

4. **Low implementation risk:**
   - Minimal code change (one conditional before existing clip)
   - Reuses existing APCR1n safety gate logic
   - Can disable via config

5. **Falsifiable:**
   - If 7.0 Nm emergency cap is insufficient → can try T6F_plus with 8.0 Nm
   - If it causes instability → can analyze telemetry to see exactly when/why
   - If it doesn't help → proves problem is not upstream cap

### Alternative: T6G_emergency_bypass_budget

**When to consider T6G:**
- If T6F validation shows 7.0 Nm is still insufficient
- If Phase 7 transmission validation proves tuned cap layer (8.0 Nm) is needed
- As a second iteration after T6F data is analyzed

**Why not first:**
- Full bypass is higher risk than bounded raise
- Emergency-only (no hard band) is less graduated
- Harder to justify bypassing an intentional safety cap without trying bounded raise first

### Not recommended for Phase 6: T6H, T6I

**T6H (blend):**
- More complex than needed for first validation
- Phase 4 didn't identify pitch reserve as the specific issue
- Continuous blending harder to test and debug
- Should only pursue if T6F/T6G show intermediate blend values are needed

**T6I (inverse schedule):**
- Contradicts Phase 6 design intent without evidence that intent was wrong
- Higher risk: affects ALL high-height emergencies uniformly
- Less targeted than conditional approaches
- Should only pursue if we determine Phase 6 philosophy was fundamentally flawed

---

## Implementation Plan

### Phase 6: Implement T6F_budget_cap_raise

**Scope:**
- New `T6F_budget_cap_raise` profile inheriting from T5
- Conditional logic before line 2009 upstream clip
- 12 telemetry fields
- 15 targeted tests
- No changes to T5, T6B, or other existing profiles

**Safety validation:**
1. T6F remains opt-in (not default)
2. All existing profiles unchanged
3. Normal/soft/desired bands behave identically to T5
4. Hard/emergency bands only differ when ALL safety gates pass
5. Final motor torque cap still respected

### Future iterations (conditional on Phase 7-10 results)

**If T6F passes Phase 9 (5000-step validation):**
- Consider T6F ready for Step E candidate
- Archive T6G/T6H/T6I designs for future reference

**If T6F fails Phase 8 (2000-step screening) but shows promise:**
- Analyze telemetry to determine failure mode
- Consider T6F_plus with emergency cap 8.0 Nm
- Consider T6G if bypass (not just raise) is needed

**If T6F shows no improvement over T5:**
- Re-examine hypothesis: maybe 4.0 Nm is not the bottleneck
- Consider T6H if smooth blending is needed
- Consider that high-height emergency recenter may need different approach entirely

**If T6F causes instability:**
- Analyze which safety gate was insufficient
- Strengthen gates before trying T6G
- May vindicate Phase 6 conservative philosophy

---

## Telemetry Requirements (All Candidates)

All architecture fix implementations must log:

**Activation state:**
- `arch_fix_enabled` (bool)
- `arch_fix_type` (string)
- `arch_fix_active` (bool)
- `arch_fix_reason` (string: "height", "band", "safety", "recenter", or "all_pass")

**Gate state:**
- `arch_fix_height_gate_pass` (bool)
- `arch_fix_band_gate_pass` (bool)
- `arch_fix_safety_gate_pass` (bool)
- `arch_fix_recenter_gate_pass` (bool)

**Authority state:**
- `effective_max_position_tau_before_arch_fix` (float)
- `effective_max_position_tau_after_arch_fix` (float)
- `arch_fix_requested_cap` (float)

**Transmission verification:**
- `arch_fix_upstream_clip_active` (bool)
- `arch_fix_tau_position_before_clip` (float)
- `arch_fix_tau_position_after_upstream_clip` (float)
- `arch_fix_torque_transmitted_above_4nm` (bool)

---

## Test Requirements (All Candidates)

All architecture fix implementations must pass:

1. **Opt-in verification:** T6F/T6G/T6H/T6I not default
2. **Baseline preservation:** T5/T6B unchanged
3. **Normal band:** arch fix inactive in normal band at high_0p480
4. **Soft band:** arch fix inactive in soft band at high_0p480
5. **Desired band:** arch fix inactive in desired band at high_0p480
6. **Hard band safe activation:** arch fix active when height/band/safety/recenter all pass
7. **Emergency band safe activation:** arch fix active when all gates pass
8. **Height gate:** arch fix inactive at low_0p300 even in emergency
9. **Contact gate:** arch fix inactive when contact invalid
10. **Height gate:** arch fix inactive when CoM height < threshold
11. **Roll gate:** arch fix inactive when roll > soft limit
12. **Pitch gate:** arch fix inactive when pitch > soft limit
13. **Telemetry completeness:** all required fields logged
14. **CSV writer:** telemetry fields written to CSV
15. **Motor cap:** final torque still respects motor limit

---

## Classification

**DESIGN_PHASE_COMPLETE:** All four candidates designed and evaluated.

**SELECTED_CANDIDATE:** T6F_budget_cap_raise

**SELECTION_RATIONALE:** Safest, most testable, preserves Phase 6 intent, conservative first step, low implementation risk.

**ALTERNATIVE_CANDIDATE:** T6G_emergency_bypass_budget (for future iteration if T6F insufficient)

**ARCHIVED_CANDIDATES:** T6H_pitch_reserve_blend, T6I_height_scheduled_position_authority (for future consideration based on T6F results)

---

## Next Phase

**Phase 6:** Implement T6F_budget_cap_raise as opt-in profile with full telemetry and test coverage.

**Date:** 2026-06-12  
**Status:** Design phase complete, proceed to implementation
