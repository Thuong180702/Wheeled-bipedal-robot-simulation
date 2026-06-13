# WBC Gate Logic Inventory

**Date:** 2026-06-07
**Scope:** Step E official WBC gate logic

## Summary

The WBC gate logic in `scripts/analyze_step_e_extreme_height_d2_official_check.py` uses `tau_wbc_norm > 0.001` as a proxy for WBC "applied". This is a **false positive** because:

- `tau_wbc_norm` includes QP structural support feedforward output
- This is NOT active WBC control authority
- At extreme heights (0.300m, 0.480m), `per_actuator_wbc_authority_enabled = False`
- Active torque owner is `support_feedforward` only

## Official Pass/Fail Gate (MUST FIX)

| File | Line(s) | Current Logic | Issue |
|------|---------|--------------|-------|
| `scripts/analyze_step_e_extreme_height_d2_official_check.py` | 160-162, 275-280 | `wbc_applied = any(v > 0.001 for v in wbc_norm)` | False positive - includes QP structural output |

## Diagnostic-Only Uses (KEEP AS-IS)

| File | Line(s) | Usage | Note |
|------|---------|-------|------|
| `scripts/simulate_hierarchical_controller.py` | 573, 2127, 2615, 2674-2676, 2729, 3543, 4223, 4241, 4268 | Telemetry logging | Should be clearly labeled as structural QP output |

## Telemetry Field Mapping

| Field | Source | Description | Current Value at Extreme Heights |
|-------|--------|-------------|----------------------------------|
| `tau_wbc_norm` | simulate_hierarchical_controller.py | L2 norm of WBC torques from BalanceCoreTorqueComposer | 13.5 Nm (nonzero) |
| `per_actuator_wbc_authority_enabled` | simulate_hierarchical_controller.py | Flag for per-actuator WBC authority | **False** |
| `active_torque_owner_per_joint` | simulate_hierarchical_controller.py | Per-joint torque owner | `none,support_feedforward,...` |

## Recommended Fix

Replace:
```python
wbc_applied = any(v > 0.001 for v in wbc_norm)
```

With:
```python
# Check per-actuator WBC authority flag (actual WBC control)
wbc_authority_enabled = rows[0].get('per_actuator_wbc_authority_enabled', 'False') == 'True'

# Check ownership-based detection as fallback
active_owners = set()
for r in rows:
    owner_str = r.get('active_torque_owner_per_joint', '')
    if owner_str:
        for owner in owner_str.split(','):
            active_owners.add(owner.strip())

# Actual WBC owners (excluding support_feedforward, shape_posture, etc.)
WBC_ACTUAL_OWNERS = {'wbc', 'wbc_correction', 'full_wbc', 'centroidal_wbc'}
has_actual_wbc_owner = bool(active_owners & WBC_ACTUAL_OWNERS)

wbc_applied = wbc_authority_enabled or has_actual_wbc_owner
```

## Impact

| Category | Files |
|----------|-------|
| **Must Modify** | `scripts/analyze_step_e_extreme_height_d2_official_check.py` |
| **Keep Unchanged** | `scripts/simulate_hierarchical_controller.py` (telemetry only) |
| **Tests to Add** | `tests/test_step_e_validator.py` or update existing validator tests |
