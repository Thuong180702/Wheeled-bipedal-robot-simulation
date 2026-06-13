# Height Range Extension Strategy Audit - Final Integrated Report

**Date:** 2026-06-06
**Phase:** FINAL_INTEGRATED_REPORT
**Decision:** `READY_FOR_HEIGHT_RANGE_EXTENSION_PLAN`

---

## Executive Summary

This audit examined:
1. The old validated controller baseline
2. The HY2-DIV / posture-control work
3. The gap between validated and desired height ranges
4. A safe strategy to extend standing/posture control to 0.300m and 0.480m

**Key findings:**
- Old baseline is **INTACT** and **PROTECTED**
- Physical envelope is **10× broader** than validated dynamic envelope
- HY2-DIV is **SAFE but INSUFFICIENT** at current authority levels
- **Intermediate heights are UNTESTED** — no dynamic validation between 0.394m and 0.300m
- Height extension is **feasible** with incremental ladder approach

---

## Phase 1: Old Baseline Audit

**Classification:** `OLD_BASELINE_IS_PROTECTED`

| Aspect | Status |
|--------|--------|
| Step E 5/5 PASS | ✓ VERIFIED |
| Step C 5/5 PASS | ✓ VERIFIED |
| D2 profile default | ✓ VERIFIED |
| HY2-DIV disabled | ✓ VERIFIED |
| WBC off | ✓ VERIFIED |

**Validated envelope:** 0.394m to 0.414m (1.95cm span)

---

## Phase 2: HY2-DIV/Posture Audit

**Classification:** `HY2_DIV_SAFE_BUT_INSUFFICIENT`

| Aspect | Status |
|--------|--------|
| Gate pass-through fixed | ✓ VERIFIED |
| A0 survived 5000 steps | ✓ VERIFIED |
| Structural invariants clean | ✓ VERIFIED |
| Divergence controlled | ✗ FAIL (0.245-0.493 rad) |
| HY2-DIV authority sufficient | ✗ FAIL (88.74% clipping) |

**A0 is safest candidate:** No nominal regression, conservative authority, low-height active.

---

## Phase 3: Height-Range Gap Audit

| Envelope | Low (m) | High (m) | Span (cm) |
|----------|---------|----------|-----------|
| **Validated Dynamic** | 0.394 | 0.414 | 1.95 |
| **Physical (static)** | 0.292 | 0.491 | 19.89 |
| **Target Extreme** | 0.300 | 0.480 | - |

**Gap analysis:**
- Low gap: 9.93 cm (from 0.394m to target 0.300m)
- High gap: 6.72 cm (from 0.414m to target 0.480m)

**Region classification:**
- `VALIDATED_DYNAMIC_ENVELOPE`: 0.394-0.414m (5 heights)
- `CONTACT_HEIGHT_SAFE_BUT_POSTURE_FAILS`: 0.300m, 0.480m
- `UNKNOWN_REQUIRES_TEST`: 0.330-0.380m, 0.420-0.465m

---

## Phase 4: Baseline vs Extreme Failure Analysis

### Failure Mechanisms

| Height | Primary Failure | Evidence |
|--------|---------------|----------|
| **0.300m** | HY2-DIV CLIPPING | 88.74% clip, hip_pitch=1.376 |
| **0.480m** | SUPPORT_DRIFT_COUPLING | r=-0.517, support=0.378m |
| **nominal (posture)** | UNKNOWN MISMATCH | 5× worse vs baseline |

### Baseline Success Factors
- Height in validated range
- Moderate joint angles
- Adequate support width
- Low support drift
- Per-joint PD sufficient

---

## Phase 5: Extension Strategy

### Design Principles
1. **Keep old baseline as default** — D2 profile protected
2. **Extreme-height extension opt-in** — explicit profile names
3. **Extend gradually** — 20-30mm ladder steps
4. **Separate posture gates** — defer pitch/support

### Height Ladder

**Low side:**
```
0.394m [VALIDATED] → 0.380m → 0.360m [exists] → 0.340m → 0.330m [exists] → 0.320m → 0.300m [exists, posture FAIL]
```

**High side:**
```
0.414m [VALIDATED] → 0.430m → 0.450m [exists] → 0.465m → 0.480m [exists, posture FAIL]
```

### Candidate Families

| Family | Purpose | Risk |
|--------|---------|------|
| A: Baseline-only | Find first failure | LOW |
| B: HY2-DIV A0 | Test A0 on ladder | MEDIUM |
| C: Extended gate | High-side HY2 coverage | MEDIUM-HIGH |
| D: Strong low authority | Very low height | MEDIUM |
| E: Support-drift-aware | High-side coupling | HIGH |

---

## Phase 6: Next Experimental Plan

### Experiment 0: Baseline Ladder Mapping

**Purpose:** Establish where old controller first fails.

**Command:**
```bash
# LOW SIDE: 0.360m
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/low_0p360_setup.json \
  --steps 500

# HIGH SIDE: 0.450m
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile candidate_D2_wheel_velocity_damping_light \
  --height-variant-setup outputs/physical_target_height_setups/high_0p450_setup.json \
  --steps 500
```

### Stop Conditions
- Contact loss → STOP
- Height collapse → STOP
- Roll instability → STOP
- Structural invariant violation → STOP

---

## What Must NOT Be Changed

| Element | Prohibition |
|---------|------------|
| `candidate_D2_wheel_velocity_damping_light` | DO NOT MODIFY |
| Default HY2-DIV state | DO NOT ENABLE |
| Default WBC state | DO NOT ENABLE |
| Official Step E/C gates | DO NOT RELAX |

---

## Final Decision

```
READY_FOR_HEIGHT_RANGE_EXTENSION_PLAN
```

### Rationale
1. Old baseline is intact and protected
2. HY2-DIV is safe but insufficient — A0 is the safest candidate
3. Physical envelope is 10× broader — room for extension
4. Intermediate heights are untested — ladder approach is needed
5. Extension strategy is defined — ready for first experiment

### What Is NOT Ready
- HY2-DIV authority tuning (A0 insufficient)
- Support-drift-aware controller
- Official Step E/C at extremes
- Step C/D at extremes

### Next Step
User approval → Execute Experiment 0 (Baseline Ladder Mapping)

---

## Files Created

### Phase 0: Worktree Status
- `outputs/height_range_extension_strategy_audit/worktree_status.txt`
- `outputs/height_range_extension_strategy_audit/current_diff_stat.txt`
- `outputs/height_range_extension_strategy_audit/recent_commits.txt`

### Phase 0: Artifact Inventory
- `outputs/height_range_extension_strategy_audit/artifact_inventory.json`
- `outputs/height_range_extension_strategy_audit/artifact_inventory.md`

### Phase 1: Old Baseline Audit
- `docs/validation/old_baseline_controller_audit.md`
- `outputs/height_range_extension_strategy_audit/old_baseline_controller_audit.json`

### Phase 2: HY2-DIV/Posture Audit
- `docs/validation/hy2_div_posture_fix_audit.md`
- `outputs/height_range_extension_strategy_audit/hy2_div_posture_fix_audit.json`

### Phase 3: Height-Range Gap Audit
- `docs/validation/height_range_gap_audit.md`
- `outputs/height_range_extension_strategy_audit/height_range_gap_audit.json`

### Phase 4: Baseline vs Extreme Failure
- `docs/validation/baseline_vs_extreme_failure_gap_analysis.md`
- `outputs/height_range_extension_strategy_audit/baseline_vs_extreme_failure_gap_analysis.json`

### Phase 5: Extension Strategy
- `docs/validation/height_range_extension_strategy.md`
- `outputs/height_range_extension_strategy_audit/height_range_extension_strategy.json`

### Phase 6: Next Experiment Plan
- `docs/validation/height_range_extension_next_experiment_plan.md`
- `outputs/height_range_extension_strategy_audit/height_range_extension_next_experiment_plan.json`

### Final Integrated Report
- `docs/validation/height_range_extension_integrated_audit_report.md` (this document)
- `outputs/height_range_extension_strategy_audit/height_range_extension_integrated_summary.json`