# early_zero_crossing_recenter — Logic Audit

**Classification:** `OLD_ZC_OVERSHOOT_TARGET_TOO_DEEP`

**Profile audited:** `zero_crossing_support_recenter`

**Telemetry:** `outputs/step_e_extreme_support_fix_eval/active_pitch_crossing/zc_5000_high_0p480/telemetry_5000.csv`

**Steps:** 5000 | **Height:** high_0p480

---

## Drift Statistics

| Metric | Value |
|--------|-------|
| min drift | -0.041332 m |
| max drift | 0.198161 m |
| P2P | 0.239494 m |
| max abs | 0.198161 m |
| mean signed | 0.082292 m |
| median signed | 0.075718 m |
| positive % | 86.4% |
| negative % | 13.6% |
| zero crossings | 36 |
| positive area | 419.6860 |
| negative area | 8.3067 |
| symmetry ratio | 50.524 |
| time inside ±0.03 | 31.6% |
| time inside ±0.05 | 40.7% |
| time inside ±0.08 | 51.4% |
| time outside ±0.08 | 48.6% |
| time outside ±0.10 | 42.4% |
| time outside ±0.15 | 25.6% |

---

## ZC Episode Analysis

### RECENTER_FROM_POSITIVE Episodes (22 total)

For each episode, key observations:

| Ep | Steps | Enter Error | Max Error | Min Error | Crossed Zero | Reached -0.02 | Reached -0.025 | Final Tau |
|----|-------|-------------|-----------|-----------|--------------|---------------|----------------|-----------|
| 1 | 138 | 0.0798 | 0.1823 | -0.0191 | ✓ | ✗ | ✗ | -0.2599 |
| 2 | 144 | 0.0782 | 0.1857 | -0.0110 | ✓ | ✗ | ✗ | -0.2343 |
| 3 | 146 | 0.0793 | 0.1893 | -0.0076 | ✓ | ✗ | ✗ | -0.2240 |
| 4 | 146 | 0.0775 | 0.1870 | -0.0079 | ✓ | ✗ | ✗ | -0.2251 |
| 5 | 146 | 0.0796 | 0.1881 | -0.0082 | ✓ | ✗ | ✗ | -0.2259 |
| 6 | 146 | 0.0783 | 0.1871 | -0.0117 | ✓ | ✗ | ✗ | -0.2365 |
| 7 | 140 | 0.0782 | 0.1860 | -0.0183 | ✓ | ✗ | ✗ | -0.2582 |
| 8 | 129 | 0.0786 | 0.1907 | -0.0166 | ✓ | ✗ | ✗ | -0.2558 |
| 9 | 140 | 0.0796 | 0.1982 | -0.0185 | ✓ | ✗ | ✗ | -0.2591 |
| 10 | 144 | 0.0796 | 0.1859 | -0.0127 | ✓ | ✗ | ✗ | -0.2397 |

**Critical Finding:** All episodes crossed zero BUT none reached the -0.02 exit target.

### RECENTER_FROM_NEGATIVE Episodes

**Count:** 0

The controller did NOT enter RECENTER_FROM_NEGATIVE once in 5000 steps.

---

## Root Cause Analysis

### Finding 1: Exit Target Never Reached

The old ZC logic requires `e <= -0.02` to exit RECENTER_FROM_POSITIVE. However:

- **Best min error achieved:** -0.0191 m (Episode 1)
- **Never reached -0.02:** 0/22 episodes
- **Never reached -0.025:** 0/22 episodes

The controller keeps pushing PAST the zero-crossing point but can't reach the deep -0.02 target. This means:

1. **Correction holds longer than needed** — drift goes significantly negative
2. **P2P increases** — overshoot on the negative side inflates peak-to-peak
3. **Oscillation is exaggerated** — wider swing than necessary

### Finding 2: No Negative Episodes

With 86.4% positive drift, the controller only needed to recenter from positive drift. Zero RECENTER_FROM_NEGATIVE episodes means:

- The -0.02 target was never approached from the negative side
- Negative side behavior is untested
- The asymmetry is self-reinforcing (positive recenter → slight negative → back to positive)

### Finding 3: Drift Still Mostly Positive

Despite ZC logic:
- 86.4% positive vs 13.6% negative
- Symmetry ratio 50.5 (should be ~1.0 for perfect symmetry)
- Mean drift +0.082 m (far from zero)

The ZC recenter improved from adaptive (102.9 ratio) but is still far from symmetric.

---

## Classification

**`OLD_ZC_OVERSHOOT_TARGET_TOO_DEEP`**

The old ZC logic improved drift symmetry significantly vs adaptive trim, but the exit target of -0.02 m is:

1. **Too deep** — drift can't reach it reliably
2. **Not required** — the goal is zero-crossing, not overshoot to -0.02
3. **Causing increased P2P** — pushes drift further negative than needed

**Secondary classification:** `OLD_ZC_ENTRY_TOO_LATE`

The entry threshold of 0.08 m means the controller waits too long before entering recenter. Drift accumulates significantly before correction begins.

---

## Evidence Summary

1. **0/22 positive recenter episodes reached -0.02 exit target**
2. **0/22 episodes reached -0.025 exit target**
3. **Max negative drift: -0.0413 m (deeper than needed)**
4. **P2P: 0.2395 m (increased from adaptive's 0.2241 m)**
5. **Symmetry ratio still 50.5 (should be closer to 1.0)**
6. **No negative recenter episodes triggered**

---

## Conclusion

The old `zero_crossing_support_recenter` implementation:

**IMPROVED** drift symmetry vs `adaptive_support_centering_trim` (symmetry ratio 50.5 vs 102.9)

**BUT** has two problems:

1. **Exit target too deep** — requires reaching -0.02, which drift never achieves. The correction holds until max hold (600 steps), inflating P2P.

2. **Entry threshold too late** — 0.08 m entry allows significant drift accumulation before correction begins.

**Solution:** Create `early_zero_crossing_recenter` that:
- Exits at zero crossing (e <= 0 for positive recenter), not -0.02
- Uses earlier entry threshold (0.05 m)
- Decays correction immediately after zero crossing
- Does NOT require reaching opposite-side target

---

## Action Required

Implement `early_zero_crossing_recenter` profile:
- Base: `zero_crossing_support_recenter`
- Entry: 0.05 m (earlier)
- Exit: zero crossing (e <= 0 or e >= 0)
- No opposite-side target
- Immediate decay after zero crossing