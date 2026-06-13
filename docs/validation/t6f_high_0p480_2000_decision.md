# T6F Phase 8 Final Decision

**Date:** 2026-06-12  
**Phase:** 8 of 11  
**Classification:** T6F_2000_NOT_BETTER_THAN_T5

---

## Decision

**REJECT T6F_budget_cap_raise**

**DO NOT proceed to Phase 9 (5000-step validation)**

---

## Rationale

Phase 8 screening revealed that T6F architecture fix successfully transmits torque above 4.0 Nm (as proven in Phase 7) but **degrades drift performance** compared to T5:

- outside ±0.10 m: **-115 steps worse** (798 → 913)
- outside ±0.15 m: **-512 steps worse** (89 → 601, 4.4% → 30.1%)
- max |error|: 0.187 m → 0.212 m (+13% worse)

**The architecture fix transmits more torque but uses it counterproductively.**

---

## What Was Proven

### Phase 7 (Torque Transmission) ✓

T6F successfully:
- Raised upstream cap from 4.0 to 6.5/7.0 Nm
- Transmitted position torque up to 7.0 Nm
- Activated in 46% of high_0p480 steps
- Maintained stability and safety gates

**Verdict:** Architecture fix works as designed for torque transmission.

### Phase 8 (Drift Performance) ✗

T6F demonstrated:
- Raised torque authority degrades drift control
- outside ±0.10 and ±0.15 metrics significantly worse than T5
- Degradation persists across all 500-step windows
- No improvement in early transient or mid-episode recovery

**Verdict:** Raised authority makes drift worse, not better.

---

## Root Cause Hypothesis

The upstream 4.0 Nm clip was not the only bottleneck. Possible causes:

1. **Gain mismatch:** Position/velocity gains tuned for 4.0 Nm create overshoot when raised to 7.0 Nm
2. **Wheel saturation:** Transmitted torque saturates at wheel-level 7.5 Nm cap
3. **Phase dynamics:** Higher authority increases phase lag or destabilizes pitch/position coupling
4. **Band logic:** Emergency band triggers with wrong timing/sign/magnitude

---

## Next Steps

### Do NOT proceed to:
- Phase 9 (5000-step validation)
- Phase 10 (low_0p300 regression)
- Phase 11 (final report)

### Recommended investigation:
1. **Root-cause analysis:** Why does raised authority degrade drift?
2. **Gain scheduling:** Add gain reduction when arch fix raises cap
3. **Torque rate limiting:** Smooth torque transitions during cap changes
4. **Band state audit:** Verify emergency band triggers correctly
5. **Wheel saturation analysis:** Check if final torque saturates despite raised upstream cap

### Future candidates:
- **T6G:** T6F + gain scheduling (reduce k_position when cap > 4.0)
- **T6H:** T6F + torque rate limiting (smooth raised cap transitions)
- **T6I:** T6F + emergency band logic fix (if audit reveals issue)

Do not implement T6G/T6H/T6I until root cause is understood.

---

## Conclusion

**Phase 8 correctly rejected T6F.**

The 11-phase protocol is working as designed:
- Phase 7 validated architectural mechanism (torque transmission) ✓
- Phase 8 validated performance impact (drift control) ✗

T6F transmits more torque but makes the problem worse. Proceed to root-cause investigation, not 5000-step validation.

---

**Classification:** T6F_2000_NOT_BETTER_THAN_T5  
**Proceed to Phase 9?** NO  
**Date:** 2026-06-12
