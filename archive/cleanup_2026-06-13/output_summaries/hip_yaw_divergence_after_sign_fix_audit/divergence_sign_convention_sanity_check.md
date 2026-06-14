# Divergence Sign Convention Sanity Check

**Date:** 2026-06-05
**Decision:** `DIVERGENCE_SIGN_CONVENTION_CONFIRMED`

## Critical Question

> Some report text says "divergence torque accelerates divergence 97–99% of the time."
> But one summary field appears to say "div_torque_opposes_error_pct ≈ 97–99%."
> Is there a naming/sign inconsistency?

## Definitions

### Divergence Error
```
divergence = l_error - r_error
```

- **Positive divergence**: left ahead of right (l_error > 0, r_error < 0)
- **Negative divergence**: left behind right (l_error < 0, r_error > 0)

### Divergence Torque
```
divergence_torque = l_torque - r_torque
```

### Opposition Rule

| Condition | de * dt | Meaning | Effect on Divergence |
|-----------|---------|---------|----------------------|
| de > 0, dt > 0 | positive | Same signs | ACCELERATING (wrong) |
| de > 0, dt < 0 | negative | Opposite signs | CORRECTING (right) |
| de < 0, dt > 0 | negative | Opposite signs | CORRECTING (right) |
| de < 0, dt < 0 | positive | Same signs | ACCELERATING (wrong) |

### Field Naming in Analyzer

- **correcting_pct** = `de * dt < 0` (opposite signs) → torque opposes divergence
- **accelerating_pct** = `de * dt > 0` (same signs) → torque accelerates divergence

## Empirical Evidence

### Per-Joint PD Behavior (Post-Sign-Fix)

| Height | Accelerating % | Correcting % |
|--------|----------------|--------------|
| nominal | 99.68% | 0.32% |
| low_0p300 | 97.76% | 2.24% |
| high_0p480 | 99.34% | 0.66% |

**Interpretation**: Per-joint PD produces ACCELERATING torques 97-99% of the time.

### Detailed Trace (nominal, step 154)

```
divergence = l_error - r_error = 0.0029 - (-0.0072) = 0.0100 (positive = left ahead)
l_error = +0.0029 (left behind reference)
r_error = -0.0072 (right ahead of reference)
l_torque = +0.0494 (positive = increase left position)
r_torque = -0.1358 (negative = decrease right position)

Per-joint PD effect:
- Left gets positive torque → left accelerates (but already ahead)
- Right gets negative torque → right decelerates (but already ahead)
- Result: left goes even more ahead, right falls even more behind
- Divergence INCREASES

This is ACCELERATING behavior, confirming the analyzer field naming is correct.
```

## HY2-DIV Sign Convention Analysis

### Code (shape_posture_controller.py lines 230-240)

```python
divergence = l_error - r_error
divergence_rate = l_vel - r_vel

# Antisymmetric torque: opposes divergence
# If left > right (positive divergence), left needs -tau, right needs +tau
tau_div_left_raw = -(self.k_divergence * divergence + self.k_divergence_rate * divergence_rate) * div_gate
tau_div_right_raw = +(self.k_divergence * divergence + self.k_divergence_rate * divergence_rate) * div_gate
```

### HY2-DIV Behavior

| Scenario | divergence | tau_div_left | tau_div_right | Effect |
|----------|-----------|--------------|---------------|--------|
| left ahead | > 0 | negative | positive | left slows, right speeds → div DECREASES |
| left behind | < 0 | positive | negative | left speeds, right slows → div DECREASES |

**Conclusion**: HY2-DIV sign convention is CORRECT.

## Resolved Ambiguity

**Question**: Did the analyzer "correcting" field actually mean "opposes" or "accelerates"?

**Answer**: The field `correcting_pct` correctly means **opposes** (de * dt < 0).

**Evidence**:
1. Per-joint PD is 99% "accelerating" in the analyzer
2. Per-joint PD empirically AMPLIFIES divergence (confirmed by detailed trace)
3. Therefore, "accelerating" = same signs = AMPLIFYING divergence (correct)

**No inconsistency exists**: The root cause conclusion is VALID.

## Final Decision

**`DIVERGENCE_SIGN_CONVENTION_CONFIRMED`**

### Rationale

1. **Per-joint PD accelerates divergence** (99% same-sign = accelerating)
2. **HY2-DIV opposes divergence** (signs designed to produce opposite-sign)
3. **Divergence analyzer field naming is correct**
4. **Root cause conclusion is valid**

## Action

Proceed to implement HY2-DIV with the confirmed sign convention.

## Expected HY2-DIV Behavior

When enabled:
- HY2-DIV will produce ~99% "correcting" torques on divergence mode
- This will OPPOSE divergence instead of accelerating it
- Target: reduce divergence RMS from ~0.24 rad to < 0.10 rad at nominal
