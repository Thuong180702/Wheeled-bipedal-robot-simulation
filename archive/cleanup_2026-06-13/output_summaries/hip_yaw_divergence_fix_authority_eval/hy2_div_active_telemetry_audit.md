# HY2-DIV Active Telemetry Audit

## Issue Found

**Problem:** `hip_yaw_div_active` was ambiguous - it reported whether HY2-DIV was *enabled* (config flag), not whether it was *operationally active* (gate > epsilon).

This caused the 500-step validation table to show 100% active for A0 at nominal/high heights, which appeared to conflict with gate verification results showing gate=0.0.

## Terminology Clarification

| Term | Definition |
|------|------------|
| `hip_yaw_div_enabled` | Config flag - true if HY2-DIV controller is enabled |
| `hip_yaw_div_gate_active` | Operational state - true only if gate_smoothstep > epsilon |
| `hip_yaw_div_height_gate` | Continuous smoothstep gate value [0.0, 1.0] |
| `hip_yaw_div_effective_k` | Gate-scaled gain = k_divergence * gate |
| `hip_yaw_div_effective_kd` | Gate-scaled derivative gain = k_divergence_rate * gate |
| `hip_yaw_div_effective_tau_max` | Torque limit (not gate-scaled; gate scales the command) |

## Fix Applied

**File:** `wheeled_biped/controllers/shape_posture_controller.py`

Replaced ambiguous `hip_yaw_div_active` with:
- `hip_yaw_div_enabled` (was `enable_hip_yaw_divergence_damping`)
- `hip_yaw_div_gate_active` (new - gate > 1e-6)
- `hip_yaw_div_effective_k` (new - gate-scaled)
- `hip_yaw_div_effective_kd` (new - gate-scaled)
- `hip_yaw_div_effective_tau_max` (new - clarifies tau_max is not gate-scaled)

## Expected vs Actual Behavior

| Candidate | Height | Expected Enabled | Expected Gate Active | Expected Gate | Actual Gate | Status |
|-----------|--------|------------------|---------------------|---------------|-------------|--------|
| A0 | nominal | true | **false** | 0.0 | 0.0 | MATCH |
| A0 | low_0p300 | true | true | 1.0 | 1.0 | MATCH |
| A0 | high_0p480 | true | **false** | 0.0 | 0.0 | MATCH |
| B1 | nominal | true | true | 0.5 | 0.5 | MATCH |
| B1 | low_0p300 | true | true | 1.0 | 1.0 | MATCH |
| B1 | high_0p480 | true | true | 0.028 | 0.028 | MATCH |

## Key Insight

A0 at nominal/high heights has `enabled=true` but `gate_active=false`. This means:
- HY2-DIV is configured
- But gate is closed (gate_smoothstep = 0.0)
- So effective_k = 0 and HY2 torque ≈ 0

The 500-step "HY2 Active" metric should now be renamed to "HY2 Gate Active" or use `hip_yaw_div_gate_active` instead of `hip_yaw_div_enabled`.

## Tests Updated

- `test_enabled_vs_gate_active_separate` - Verifies enabled vs gate_active are separate
- `test_b1_gate_active_at_nominal_height` - B1 gate_active at nominal
- `test_effective_k_follows_gate` - effective_k follows gate value
- `test_hy2_div_telemetry_fields_exist` - All new fields exist

All 35 tests pass.

## Conclusion

**VERDICT: PASS**

The terminology is now clear. The evaluator should be updated to use `hip_yaw_div_gate_active` instead of `hip_yaw_div_enabled` for operational reporting.