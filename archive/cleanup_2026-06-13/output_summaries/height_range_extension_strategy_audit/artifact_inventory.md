# Height Range Extension Strategy Audit - Artifact Inventory

**Date:** 2026-06-06
**Audit Type:** Phase 0 - Artifact Inventory

## Worktree Status

- Modified files: 6
- Untracked files: 61
- Baseline artifacts affected: **NO**

## Height Range Summary

| Envelope | Low (m) | High (m) | Span (cm) |
|----------|---------|----------|-----------|
| **Validated Dynamic** | 0.3933 | 0.4128 | 1.95 |
| **Physical (static)** | 0.2919 | 0.4908 | 19.89 |
| **Target Extreme** | 0.300 | 0.480 | - |

**Gap Analysis:**
- Low side: 9.93 cm gap between validated min and physical min
- High side: 6.72 cm gap between validated max and physical max

## Baseline Artifacts (5)

| Artifact | Path | Heights | Status | Evidence |
|----------|------|---------|--------|----------|
| Five-variant baseline report | `docs/validation/five_variant_step_e_step_c_baseline_verification_report.md` | 0.394-0.414m | **VALID** | Step E 5/5 PASS, Step C 5/5 PASS |
| Old known-good config | `docs/validation/old_known_good_step_e_step_c_config.md` | 0.394-0.414m | **VALID** | Exact command and profile |
| Step E done | `docs/validation/step_e_height_variant_robustness_done.md` | 0.394-0.414m | **VALID** | D2 profile selected |
| Step C done | `docs/validation/step_c_height_recovery_done.md` | 0.394-0.414m | **VALID** | Step C unblocked |
| Operational envelope | `docs/validation/operational_height_extreme_validation.md` | 0.393-0.413m | **VALID** | Fresh Step E telemetry at extrema |

## Physical Envelope Artifacts (3)

| Artifact | Path | Heights | Status | Evidence |
|----------|------|---------|--------|----------|
| Physical envelope validation | `docs/validation/physical_standing_height_envelope_validation.md` | 0.292-0.491m | **VALID** | 61 valid candidates |
| Physical envelope definition | `docs/validation/physical_standing_height_envelope_definition.md` | - | **VALID** | Static feasibility definition |
| Physical summary JSON | `outputs/physical_standing_height_envelope_search/physical_height_envelope_summary.json` | - | **VALID** | Extrema and metadata |

## HY2-DIV / Posture Artifacts (6)

| Artifact | Path | Heights | Status | Evidence |
|----------|------|---------|--------|----------|
| HY2-DIV gate fix report | `docs/validation/hy2_div_gate_pass_through_fix_report.md` | nominal/low/high | **VALID** | Gate pass-through fixed |
| HY2-DIV authority fix report | `docs/validation/hip_yaw_divergence_authority_fix_report.md` | nominal/low/high | **VALID** | PARTIAL - no candidate passed 5000-step |
| Posture A0 5000 report | `docs/validation/posture_standing_validation_a0_5000_report.md` | 0.30/0.40/0.48m | **VALID** | Survived but divergence exceeded targets |
| Posture gate definition | `docs/validation/posture_standing_validation_gate_definition.md` | - | **VALID** | Priority order defined |
| Hip-yaw sign fix report | `docs/validation/hip_yaw_sign_convention_fix_report.md` | all | **VALID** | Sign 0%→>93%, divergence increased |
| Divergence after sign audit | `docs/validation/hip_yaw_divergence_after_sign_fix_audit.md` | all | **VALID** | Root cause: per-joint PD accelerates divergence |

## Extreme Height Setup Artifacts (5)

| Artifact | Path | Target (m) | Achieved (m) | Status |
|----------|------|-----------|--------------|--------|
| low_0p300_setup | `outputs/physical_target_height_setups/low_0p300_setup.json` | 0.300 | 0.2955 | **VALID** |
| low_0p330_setup | `outputs/physical_target_height_setups/low_0p330_setup.json` | 0.330 | 0.3350 | **VALID** |
| low_0p360_setup | `outputs/physical_target_height_setups/low_0p360_setup.json` | 0.360 | 0.3633 | **VALID** |
| high_0p450_setup | `outputs/physical_target_height_setups/high_0p450_setup.json` | 0.450 | 0.4512 | **VALID** |
| high_0p480_setup | `outputs/physical_target_height_setups/high_0p480_setup.json` | 0.480 | 0.4810 | **VALID** |

## Key Findings

1. **Baseline is INTACT:** All 5 baseline artifacts are valid and reproducible
2. **Physical envelope is BROADER:** 10× wider than validated dynamic envelope
3. **Intermediate heights are READY:** Setup files exist for 0.330, 0.360, 0.450m
4. **HY2-DIV is PARTIAL:** Gate fixed, A0 survived but insufficient authority
5. **Root cause IDENTIFIED:** Per-joint PD accelerates divergence 97-99%

## Artifact Completeness

All required artifacts for the audit are present and valid. The audit can proceed to Phase 1.