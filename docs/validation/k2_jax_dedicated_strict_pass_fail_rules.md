# K2 JAX Dedicated Realtime — Strict Pass/Fail Classification Rules

**Date:** 2026-06-29
**Task:** Phase 0 — Redefine PASS/FAIL strictly
**Status:** ACTIVE

---

## 1. Problem Statement

Previous classification marked scenarios PASS when they only satisfied:
- no fall
- hip_yaw <= 0.35 rad

This is insufficient for behavioral equivalence to original promoted K2. A candidate that survives but has 2-4x worse hip-yaw than original K2 should NOT be labeled PASS.

---

## 2. Five-Level Strict Classification

### Level 1: `EXACT_OR_BETTER`
**Definition:** Candidate metric is equal to or better than original K2 metric.

Comparison rules:
- `hip_yaw_max`, `pitch_rms`, `support_rms`, `height_rmse`: candidate <= original → EXACT_OR_BETTER
- `LF_power`, `WIP_power`: candidate <= original → EXACT_OR_BETTER (lower is better)
- `fell`: candidate == False when original == False → EXACT_OR_BETTER
- `post_pitch500`, `post_support500`: candidate <= original → EXACT_OR_BETTER
- `drift`: candidate <= original → EXACT_OR_BETTER
- Real-time Hz: candidate >= 50.0 → EXACT_OR_BETTER (threshold, not relative)

### Level 2: `WITHIN_OLD_TOLERANCE`
**Definition:** Candidate is worse than original K2 but still within explicit tolerance from old K2 promotion rules.

Default tolerances (from original K2 promotion reports):
| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| hip_yaw_max | +0.05 rad absolute or 2x relative, whichever is smaller | Original K2 promotion used "WORSE_BUT_SAFE" when delta < 0.05 |
| pitch_rms | +1.0 deg or 30% relative | K2 vs K1 had pitch variations up to ~1 deg |
| support_rms | +0.02 m or 50% relative | K2 vs K1 support variations |
| height_rmse | +0.02 m or 20% relative | Height tracking tolerance |
| LF_power | 5x relative or +0.005 absolute | Power varies with excitation |
| WIP_power | 5x relative or +0.005 absolute | Power varies with excitation |
| post_pitch500 | +0.05 rad or 30% relative | Push recovery pitch tolerance |
| post_support500 | +0.05 m or 30% relative | Push recovery support tolerance |

### Level 3: `SAFE_BUT_WORSE`
**Definition:** Candidate survives and stays under absolute safety gate, but is worse than original K2 beyond `WITHIN_OLD_TOLERANCE`.

This is NOT a promotion PASS. It indicates a regression that must be investigated.

### Level 4: `SAFETY_FAIL`
**Definition:** Candidate triggers an absolute safety gate:
- Fall: True
- NaN/Inf present
- hip_yaw_max > 0.35 rad
- Hidden torque > 0.5 Nm (magnitude)
- WBC authority active
- Non-real-simulation source

This BLOCKS promotion for the affected scope.

### Level 5: `NOT_TESTED`
**Definition:** Required scenario has no candidate data.
This prevents full promotion unless the scenario is explicitly excluded from scope.

---

## 3. Promotion Rules

### Full PASS (`K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PASS`)
Requires ALL of:
1. All required scenarios are `EXACT_OR_BETTER` or `WITHIN_OLD_TOLERANCE`
2. Zero `SAFETY_FAIL` in required scope
3. Zero `SAFE_BUT_WORSE` in required scope
4. Zero `NOT_TESTED` in required scope (or explicitly documented as out-of-scope)
5. Dynamic q_ref uses original-K2-exact computation (not interpolation)
6. mode_div enabled by default
7. Tests pass
8. Realtime >= 50 Hz

### Partial PASS (`K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_PARTIAL`)
If:
- Some scopes pass strictly, but some required cases are `NOT_TESTED` or `SAFE_BUT_WORSE`
- All `SAFETY_FAIL` cases are documented and scoped out

### BLOCKED (`K2_JAX_DEDICATED_REALTIME_ORIGINAL_K2_PROMOTION_BLOCKED`)
If:
- Any required case has `SAFETY_FAIL`
- hip_yaw > 0.35 remains
- Dynamic q_ref remains approximate for promotion
- Push matrix incomplete but claimed as full pass
- Strict classifier not implemented
- Tests fail

---

## 4. Per-Scenario Classification Rules

For each required scenario, classify EACH metric independently. The scenario's overall class is the WORST class among its metrics.

```
scenario_class = max(class(metric_1), class(metric_2), ..., class(metric_N))
```
Where ordering: EXACT_OR_BETTER(1) < WITHIN_OLD_TOLERANCE(2) < SAFE_BUT_WORSE(3) < SAFETY_FAIL(4) < NOT_TESTED(5)

Exception: If ANY safety gate is triggered, scenario class = SAFETY_FAIL regardless of other metrics.

---

## 5. Required Scenarios and Metrics Per Scenario

### Step E (fixed-height, 10 heights)
Required metrics per height:
- fell (safety gate)
- hip_yaw_max (safety gate + equivalence)
- pitch_rms_deg (equivalence)
- support_rms_m (equivalence)
- LF_power (equivalence)
- WIP_power (equivalence)
- NaN/Inf (safety gate)

### Step C (dynamic ladder, 7 cases)
Required metrics per case:
- fell (safety gate)
- hip_yaw_max (safety gate + equivalence)
- pitch_rms_deg (equivalence)
- support_rms_m (equivalence)
- LF_power (equivalence)
- WIP_power (equivalence)

### Step D (push matrix, 12 conditions)
Required metrics per condition:
- fell (safety gate)
- hip_yaw_max (safety gate + equivalence)
- post_pitch500 (equivalence)
- post_support500 (equivalence)
- LF_power (equivalence)
- WIP_power (equivalence)

### Dynamic Height (5 scenarios)
Required metrics per scenario:
- fell (safety gate)
- hip_yaw_max (safety gate + equivalence)
- pitch_rms_deg (equivalence)
- height_rmse (equivalence)

### Long-Run Equilibrium (5 heights)
Required metrics per height:
- fell (safety gate)
- hip_yaw_max (safety gate + equivalence)
- pitch_rms_deg (equivalence)
- pitch_final_deg (equivalence)

---

## 6. Reclassification of Current Results

Using the strict rules, current dedicated runner results are reclassified:

### Step E (Current Dedicated JAX vs Original K2)

| Height | Metric | Original | Candidate | Delta | Tolerance | Class |
|--------|--------|----------|-----------|-------|-----------|-------|
| low_0p300 | hy_max | 0.1314 | 0.2008 | +0.0694 | min(0.05, 2x=0.2628) = 0.05 | **SAFE_BUT_WORSE** |
| low_0p300 | pitch_rms | 2.68 | 2.9 | +0.22 | min(1.0, 30%=0.80) = 0.80 | WITHIN_OLD_TOLERANCE |
| low_0p320 | hy_max | 0.0502 | 0.0821 | +0.0319 | min(0.05, 2x=0.1004) = 0.05 | WITHIN_OLD_TOLERANCE |
| low_0p330 | hy_max | 0.0851 | 0.1162 | +0.0311 | min(0.05, 2x=0.1702) = 0.05 | WITHIN_OLD_TOLERANCE |
| low_0p340 | hy_max | 0.0445 | 0.1255 | +0.0810 | min(0.05, 2x=0.089) = 0.05 | **SAFE_BUT_WORSE** |
| low_0p360 | hy_max | 0.0959 | 0.0897 | -0.0062 | — | EXACT_OR_BETTER |
| low_0p380 | hy_max | 0.0392 | 0.0759 | +0.0367 | min(0.05, 2x=0.0784) = 0.05 | WITHIN_OLD_TOLERANCE |
| high_0p430 | hy_max | 0.0236 | 0.0833 | +0.0597 | min(0.05, 2x=0.0472) = 0.0472 | **SAFE_BUT_WORSE** |
| high_0p450 | hy_max | 0.0904 | 0.0263 | -0.0641 | — | EXACT_OR_BETTER |
| high_0p465 | hy_max | 0.0296 | 0.0454 | +0.0158 | min(0.05, 2x=0.0592) = 0.05 | WITHIN_OLD_TOLERANCE |
| high_0p480 | hy_max | 0.0563 | 0.0735 | +0.0172 | min(0.05, 2x=0.1126) = 0.05 | WITHIN_OLD_TOLERANCE |

**Step E overall: SAFE_BUT_WORSE** (3 heights are SAFE_BUT_WORSE on hy_max)

### Dynamic Height (Current Dedicated JAX vs Original K2)

| Scenario | Metric | Original | Candidate | Delta | Tolerance | Class |
|----------|--------|----------|-----------|-------|-----------|-------|
| ramp_up | hy_max | 0.0534 | 0.1242 | +0.0708 | min(0.05, 2x=0.1068) = 0.05 | **SAFE_BUT_WORSE** |
| ramp_down | hy_max | 0.0977 | **0.3728** | +0.2751 | abs gate 0.35 | **SAFETY_FAIL** |
| gate_chatter | hy_max | 0.0629 | 0.2160 | +0.1531 | min(0.05, 2x=0.1258) = 0.05 | **SAFE_BUT_WORSE** |

**Dynamic overall: SAFETY_FAIL** (ramp_down)

### Step D Push (Current Dedicated JAX vs Original K2)

| Condition | Metric | Original | Candidate | Delta | Class |
|-----------|--------|----------|-----------|-------|-------|
| high_0p480_fwd_90N | hy_max | 0.0000 | 0.0388 | +0.0388 | WITHIN_OLD_TOLERANCE |
| high_0p480_bwd_90N | hy_max | 0.0000 | 0.0281 | +0.0281 | WITHIN_OLD_TOLERANCE |

But: 10/12 conditions NOT_TESTED.
**Step D overall: NOT_TESTED for full matrix** (only 2/12 conditions)

---

## 7. Summary Table

| Scope | Worst Class | Blocks Promotion? |
|-------|------------|-------------------|
| Step E (10 heights) | SAFE_BUT_WORSE | Yes — PARTIAL |
| Step C (7 cases) | NOT_TESTED | Yes — PARTIAL |
| Step D (12 conditions) | NOT_TESTED (10/12) | Yes — PARTIAL |
| Dynamic Height (5 scenarios) | SAFETY_FAIL (ramp_down) | Yes — BLOCKED |
| Long-Run (5 heights) | NOT_TESTED | Yes — PARTIAL |
| Overall | SAFETY_FAIL + NOT_TESTED | **BLOCKED** |

---

## 8. How to Use

### In code

```python
from wheeled_biped.validation.strict_promotion_classifier import (
    StrictClass,
    classify_metric,
    classify_scenario,
    classify_scope,
    is_promotion_pass,
)
```

### In reports

Every comparison table MUST include columns:
- Original Value
- Candidate Value
- Delta
- Tolerance
- Strict Class

No scenario may be labeled PASS without this comparison.

### In CI/tests

Tests must verify:
- No SAFETY_FAIL in promoted scope
- No SAFE_BUT_WORSE in promoted scope
- No NOT_TESTED for required scenarios
- Classifier correctly identifies each level

---

## 9. Non-Negotiable Rules

1. Do NOT relax gates.
2. Do NOT redefine PASS.
3. Do NOT call a scenario PASS if it is worse than original K2 beyond tolerance.
4. Do NOT call a scenario PASS merely because it survived.
5. Do NOT call a scenario PASS merely because hip-yaw < 0.35 if original K2 was much lower.
6. Do NOT tune gains blindly.
7. Do NOT reduce push force, duration, or step count.
8. Do NOT change physics timestep or model.
9. Do NOT change original K2 Python reference.
10. Do NOT remove Python fallback or both-synced.
11. Do NOT claim WBC support.
12. Do NOT promote unsupported profiles.
13. Do NOT add xfail/skip to hide failures.
14. If full K2-equivalent posture/IK computation costs performance, report honestly.
