# Step E Extreme Failure Root Cause Audit - Artifact Inventory

**Date:** 2026-06-07
**Audit Name:** step_e_extreme_failure_root_cause_audit

## Source Files

| File | Description |
|------|-------------|
| [step_e_extreme_height_d2_official_check_report.md](docs/validation/step_e_extreme_height_d2_official_check_report.md) | Official Step E check report |
| [step_e_extreme_height_d2_metrics.json](outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_metrics.json) | Detailed metrics from official check |
| [step_e_extreme_height_d2_pass_fail.json](outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_pass_fail.json) | Pass/fail summary |
| [step_e_extreme_height_d2_summary.csv](outputs/step_e_extreme_height_d2_official_check/step_e_extreme_height_d2_summary.csv) | CSV summary |

## Telemetry Sources

| Case | File |
|------|------|
| low_0p300 | [low_0p300_5000_telemetry.csv](outputs/step_e_extreme_height_d2_official_check/low_0p300_5000_telemetry.csv) |
| high_0p480 | [high_0p480_5000_telemetry.csv](outputs/step_e_extreme_height_d2_official_check/high_0p480_5000_telemetry.csv) |

## Official Step E Results

### low_0p300 (0.300m)

| Gate | Value | Threshold | Pass/Fail |
|------|-------|-----------|-----------|
| Survived 5000 steps | True | True | ✓ |
| Support position error max | 0.176m | < 0.15m | ✗ FAIL |
| Wheel velocity max | 4.39 rad/s | < 5.0 rad/s | ✓ |
| Hip yaw max | 0.313 rad | < 0.10 rad | ✗ FAIL |
| Contact valid % | 99.98% | ≥ 99.9% | ✓ |
| WBC applied | True | == false | ✗ FAIL |
| Hidden torque | 0 | == 0 | ✓ |
| Ownership violations | 0 | == 0 | ✓ |

### high_0p480 (0.480m)

| Gate | Value | Threshold | Pass/Fail |
|------|-------|-----------|-----------|
| Survived 5000 steps | True | True | ✓ |
| Support position error max | 0.173m | < 0.15m | ✗ FAIL |
| Wheel velocity max | 5.26 rad/s | < 5.0 rad/s | ✗ FAIL |
| Hip yaw max | 0.275 rad | < 0.10 rad | ✗ FAIL |
| Contact valid % | 99.98% | ≥ 99.9% | ✓ |
| WBC applied | True | == false | ✗ FAIL |
| Hidden torque | 0 | == 0 | ✓ |
| Ownership violations | 0 | == 0 | ✓ |

## Height Monitoring Results

### low_0p300
- Target: 0.300m
- Final CoM: 0.276m
- Final height error: -0.024m
- **Classification: HEIGHT_MONITOR_PASS_RELAXED** (relaxed gate: < 0.03m)

### high_0p480
- Target: 0.480m
- Final CoM: 0.464m
- Final height error: -0.016m
- **Classification: HEIGHT_MONITOR_PASS_STRICT** (strict gate: < 0.02m)

## Controller Configuration

- **Controller Mode:** balance-core
- **Sagittal Controller:** velocity-damped
- **Profile:** candidate_D2_wheel_velocity_damping_light
- **HY2-DIV:** disabled

## Key Questions to Answer in Audit

1. **WBC Artifact:** Is the WBC "applied=true" due to actual WBC or QP support feedforward?
2. **Event Order:** Which failure (support drift, hip yaw, wheel velocity) happens first?
3. **Support Drift Root Cause:** Why does support position error exceed 0.15m?
4. **Hip Yaw Root Cause:** Why does hip yaw diverge to 0.3+ rad?
5. **Wheel Velocity (0.480m only):** Why does it exceed 5.0 rad/s at high height?
6. **Causal Chain:** What drives what?

## Artifacts to Create

- Phase 1: `wbc_artifact_audit.json`, `wbc_artifact_audit.md`
- Phase 2: `event_order/event_order_low_0p300.csv`, `event_order_high_0p480.csv`, `event_order_summary.json`
- Phase 3: `support_drift/support_drift_audit.json`
- Phase 4: `hip_yaw/hip_yaw_audit.json`
- Phase 5: `wheel_velocity/wheel_velocity_audit.json`
- Phase 6: `causal_map.json`, `causal_map.md`
- Phase 7: `fix_strategy_plan.json`, `fix_strategy_plan.md`
- Phase 8: `step_e_extreme_height_root_cause_summary.json`
