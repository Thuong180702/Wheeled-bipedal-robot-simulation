# Upstream 4.0 Nm Cap Intent Audit

**Date:** 2026-06-12  
**Status:** Phase 4 complete  
**Classification:** UPSTREAM_4NM_CAP_INTENTIONAL_BUT_CONFLICTS_WITH_TUNED_LAYER

---

## Executive Summary

**The 4.0 Nm upstream cap is INTENTIONAL, but conflicts with the tuned cap layer design.**

The `max_position_tau_nominal=4.0` is part of a deliberate "Phase 6 joint fix" height-scheduling strategy:
- **Design philosophy:** High heights are inherently less stable → reduce position authority for safety
- **Scheduling behavior:** LOW heights (0.30m) get 8.0 Nm, HIGH heights (0.48m) get 4.0 Nm
- **Rationale:** Limit aggressive position corrections at extreme heights where robot is less stable

**However, this conflicts with the APCR1n/APCR1nD tuned cap layer:**
- Tuned caps expect emergency authority up to 7-8 Nm for drift recovery
- T5/T6B were designed to handle high_0p480 with graduated authority (4.0 → 5.5 → 6.5 → 7.0/8.0)
- But upstream scheduling prevents tuned caps from ever seeing > 4.0 Nm

**This is an architectural inconsistency, not a bug.**

---

## Design Intent Analysis

### Height Scheduling Philosophy

The `scheduled_k_position()` function implements a height-dependent authority reduction:

```python
def scheduled_k_position(
    z_ref: float,
    k_nominal: float,      # Authority at HIGH heights
    k_low_max: float,      # Authority at LOW heights
    z_low: float,          # Lower boundary (0.30m typically)
    z_high: float,         # Upper boundary (0.40m typically)
) -> float:
    """
    Behavior:
        - z_ref = z_low  (0.30m) -> k_position = k_low_max  (MORE authority)
        - z_ref = z_high (0.40m) -> k_position = k_nominal  (LESS authority)
        - z_ref > z_high (0.48m) -> k_position = k_nominal  (clamped)
    """
```

**Applied to `max_position_tau`:**
- `max_position_tau_nominal = 4.0`  (HIGH heights)
- `max_position_tau_low_max = 8.0`  (LOW heights)
- `z_low = 0.300`, `z_high = 0.393` (from k_position scheduling bounds)

**At high_0p480 (0.48m):**
- Height is above `z_high` (0.393)
- Smoothstep → 0
- `effective_max_position_tau = max_position_tau_nominal = 4.0 Nm`

**At low_0p300 (0.30m):**
- Height is at `z_low` (0.300)
- Smoothstep → 1
- `effective_max_position_tau = max_position_tau_low_max = 8.0 Nm`

### Rationale

**Comment from code:** "Phase 6 joint fix"

**Inferred rationale:**
1. **Stability concern:** High heights (0.48m) are inherently less stable
   - Higher CoM → larger pitch moment arm
   - Smaller support polygon
   - Greater fall risk from aggressive corrections

2. **Safety-first design:** Limit position authority to prevent over-correction
   - 4.0 Nm at high heights is conservative
   - Prevents large position torques from destabilizing the robot
   - Paired with increased wheel damping (k_wheel_velocity scheduling)

3. **Graduated authority:** Low heights get more authority (8.0 Nm)
   - Low heights are more stable
   - Can tolerate stronger position corrections
   - Less risk of over-correction causing falls

### Evidence of Intentional Design

1. **Explicit configuration:** Both T5 and T6B set `max_position_tau_nominal=4.0` deliberately
2. **Consistent across profiles:** All APCR1nD/T6 profiles use the same 4.0 Nm nominal
3. **Paired with wheel damping:** High heights also get increased wheel damping for stability
4. **Comment indicates design phase:** "Phase 6 joint fix" suggests deliberate engineering decision
5. **Smooth scheduling:** Uses smoothstep for smooth transitions, not abrupt switch

---

## Conflict with Tuned Cap Layer

### APCR1n/APCR1nD Tuned Cap Design

The tuned cap layer (line 2353) implements graduated emergency authority:

**T5 (APCR1nD_T5_band_limited_balanced):**
- Normal band: 4.0 Nm
- Soft band: 4.5 Nm
- Desired band: 5.5 Nm
- Hard band: 6.5 Nm
- **Emergency band: 7.0 Nm**

**T6B (T6B_high_stronger_emergency):**
- Normal band: 4.0 Nm
- Soft band: 4.5 Nm
- Desired band: 5.8 Nm
- Hard band: 7.0 Nm
- **Emergency band: 8.0 Nm**

**Design assumption:** These caps expect to receive raw torque demands up to ~7.5 Nm and clip them appropriately based on drift severity.

### The Architectural Conflict

**What the tuned cap layer expects:**
```
tau_position_before_clip = 7.485 Nm (observed in Phase 3)
  ↓
[LINE 2353] Tuned cap clips based on band:
  - Emergency: clip to 7.0/8.0 Nm
  - Hard: clip to 6.5/7.0 Nm
  - Desired: clip to 5.5/5.8 Nm
  ↓
Graduated authority depending on drift severity
```

**What actually happens:**
```
tau_position_before_clip = 7.485 Nm
  ↓
[LINE 2009] Upstream clip at 4.0 Nm ← BOTTLENECK
  ↓
tau_position = 4.0 Nm max
  ↓
[LINE 2353] Tuned cap receives pre-clipped 4.0 Nm
  - Emergency 7.0/8.0 Nm cap? Input already below
  - Hard 6.5/7.0 Nm cap? Input already below
  - Desired 5.5/5.8 Nm cap? Input already below
  ↓
No graduated authority - all bands receive same 4.0 Nm max
```

### Why This Matters

1. **T6B's emergency boost is ineffective:** Raising from 7.0 to 8.0 Nm has zero effect
2. **Band graduation is lost:** Emergency gets same authority as hard/desired bands
3. **Phase 0 audit assumption was wrong:** T5 audit identified "AUTHORITY_TOO_WEAK" and suggested raising caps, but the real issue is upstream clipping
4. **Design inconsistency:** Why configure 7-8 Nm emergency caps if upstream limits to 4.0 Nm?

---

## Is the 4.0 Nm Cap Safety-Critical?

### Arguments FOR keeping 4.0 Nm at high heights:

1. **Stability safety:** High heights are inherently unstable
   - 4.0 Nm limits aggressive corrections that could destabilize
   - Conservative approach prevents over-correction falls

2. **Empirical basis:** If Phase 6 was a "fix," there may have been prior failures with higher authority
   - Unknown if there's data showing 8.0 Nm causes instability at 0.48m
   - May have been tuned based on early testing

3. **Paired design:** Works with increased wheel damping
   - Reducing position authority while increasing wheel damping may be a balanced strategy
   - Changing one without the other could break the balance

4. **Low-height success:** The system works well at low_0p300 with 8.0 Nm
   - But low heights are more stable
   - Different stability regime

### Arguments FOR raising it to 6-8 Nm at high heights:

1. **Empirical demand:** Phase 3 shows 7.485 Nm peak demand is legitimate
   - Not noise or error - actual control requirement for drift recovery
   - Robot trying to correct +0.18m drift

2. **Tuned cap exists:** If 4.0 Nm was sufficient, why design 7-8 Nm tuned caps?
   - Either tuned caps are unnecessary, OR upstream cap is too conservative
   - Design conflict suggests upstream cap wasn't considered when tuned caps were designed

3. **Safety gates already exist:** APCR1n has multiple safety gates:
   - Contact valid check
   - CoM height check (min 0.45m typically)
   - Roll limit check
   - Pitch limit check
   - Only applies in safe states

4. **Emergency is rare:** 7-8 Nm is only used in emergency band (abs_error > 0.12m)
   - Not continuously active
   - Only when drift is severe
   - Failing to recover from severe drift → fall anyway

5. **Graduated approach:** Tuned caps provide gradual escalation
   - 4.0 → 4.5 → 5.5 → 6.5 → 7.0/8.0 Nm
   - Not a sudden jump to 8.0 Nm
   - Robot has chance to recover with lower authority first

### Unknown Factors

**We don't know:**
1. Why Phase 6 was needed - what failed before?
2. Whether 8.0 Nm at 0.48m causes instability
3. Whether there's unpublished data showing 4.0 Nm is necessary
4. Whether the designer of Phase 6 was aware of the tuned cap layer
5. Whether the tuned cap designer was aware of the upstream scheduling

---

## Recommendation

**Classification:** `UPSTREAM_4NM_CAP_INTENTIONAL_BUT_NEEDS_EMERGENCY_OVERRIDE`

### Proposed Solution: Conditional Bypass

**Option: Emergency bypass for APCR1n**

Instead of globally raising `max_position_tau_nominal`, add a conditional bypass:

```python
if self.authority_schedule.continuous_max_position_tau:
    effective_max_position_tau = scheduled_k_position(
        z_ref=schedule_height_ref,
        k_nominal=4.0,  # Keep conservative nominal
        k_low_max=8.0,  # Keep low-height authority
        ...
    )
    
    # NEW: Emergency bypass when APCR1n active with safety gates passing
    if (self.authority_schedule.recenter_priority_enabled and
        apcr1n_recenter_priority_active and
        apcr1n_safety_gate_pass):
        # Allow higher authority during safe emergency recenter
        emergency_max_tau = self.authority_schedule.get("emergency_max_position_tau", 8.0)
        effective_max_position_tau = max(effective_max_position_tau, emergency_max_tau)
```

**Rationale:**
1. **Preserves nominal safety:** 4.0 Nm during normal operation
2. **Enables emergency authority:** Allows 7-8 Nm when:
   - APCR1n priority active (severe drift)
   - Safety gates pass (contact/height/roll/pitch OK)
3. **Graduated escalation:** Only bypasses when truly needed
4. **Testable:** Can validate that bypass doesn't cause instability

### Alternative: Height-Dependent Emergency Cap

Make the upstream cap itself height-aware for emergencies:

```python
if emergency_recenter_active and safety_gates_pass:
    # Higher authority at high heights during emergency
    if z_ref >= 0.45:
        effective_max_position_tau = 8.0
    else:
        effective_max_position_tau = 6.0
else:
    # Normal conservative scheduling
    effective_max_position_tau = scheduled_k_position(...)
```

---

## Next Steps

**Phase 5:** Design architecture-fix candidates based on emergency bypass approach

**Phase 6:** Implement opt-in fix profile (e.g., T6F_emergency_authority_bypass)

**Phase 7:** Validate torque transmission (verify > 4.0 Nm reaches wheels)

**Phases 8-9:** If transmission validates, run 2000/5000-step at high_0p480

---

## Artifacts

**Analysis:**
- Code inspection: `sagittal_velocity_damped_balance_controller.py` lines 1683-1696
- Scheduling function: lines 36-65
- T5 configuration: lines 1202-1230
- T6B configuration: lines 1287-1315

**Reports:**
- `docs/validation/upstream_4nm_cap_intent_audit.md` (this document)

---

**Status:** Phase 4 complete  
**Classification:** UPSTREAM_4NM_CAP_INTENTIONAL_BUT_NEEDS_EMERGENCY_OVERRIDE  
**Date:** 2026-06-12
