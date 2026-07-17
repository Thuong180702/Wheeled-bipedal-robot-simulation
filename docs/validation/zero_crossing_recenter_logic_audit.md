# zero_crossing_support_recenter — Logic Audit

**Classification:** `CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO`

**Profile audited:** `adaptive_support_centering_trim`

**Telemetry:** `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/adaptive_5000_high_0p480/telemetry_5000.csv`

**Steps:** 5000 | **Height:** high_0p480

---

## Drift Statistics

| Metric | Value |
|--------|-------|
| min drift | -0.032314 m |
| max drift | 0.191802 m |
| P2P | 0.224117 m |
| max abs | 0.191802 m |
| mean signed | 0.079952 m |
| median signed | 0.072931 m |
| positive % | 92.2% |
| negative % | 7.7% |
| zero crossings | 26 |
| positive area | 403.6029 |
| negative area | 3.9240 |
| symmetry ratio | 102.856 |
| time inside ±0.03 | 29.5% |
| time inside ±0.05 | 40.3% |
| time inside ±0.08 | 52.7% |
| time outside ±0.08 | 47.3% |
| time outside ±0.10 | 40.2% |
| time outside ±0.15 | 19.7% |

---

## Violation Episode Analysis

Threshold: ±0.08 m

### Positive Violations (e > +0.08)

- Episodes: 25
- With negative correction: 5
- Crossed zero: 3
- Crossed target (-0.02): 0
- Released early: 0

### Negative Violations (e < -0.08)

- Episodes: 0
- With positive correction: 0
- Crossed zero: 0
- Crossed target (+0.02): 0
- Released early: 0

---

## Classification

**CURRENT_LOGIC_NO_HOLD_THROUGH_ZERO**

Drift is overwhelmingly positive (92.2%) with symmetry ratio 102.9. Controller does NOT force drift to cross to negative side.

---

## Conclusion

The current `adaptive_support_centering_trim` does **NOT** implement hold-through-zero recentering.

**Evidence:**

**A new zero-crossing recenter controller is needed.**
