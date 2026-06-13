# Sagittal Authority Transmission Audit

**Date:** 2026-06-05
**Phase:** 4
**Status:** COMPLETE

---

## Executive Summary

Analyzed baseline and continuous k_position candidates (E1, E2, E3) to understand why k_position scheduling alone failed to fix support drift at low_0p300.

---

## Candidate Metrics Comparison

| Metric | Baseline | E1 (k=60) | E2 (k=80) | E3 (k=100) |
|--------|----------|-----------|-----------|------------|
| Support Error Max (m) | 0.2430 | 0.2216 | 0.2094 | 0.2031 |
| k_position Mean | 40.0000 | 40.0000 | 40.0000 | 40.0000 |
| τ_position Max (Nm) | 9.7187 | 13.2978 | 16.7530 | 20.3053 |
| Position Saturated (%) | 64.3564 | 75.2475 | 87.1287 | 91.0891 |
| τ_wheel RMS (Nm) | 0.0825 | 0.0989 | 0.1015 | 0.1029 |
| k_velocity Mean | 15.0000 | 15.0000 | 15.0000 | 15.0000 |
| τ_velocity RMS (Nm) | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Pitch Max (rad) | 0.0951 | 0.0940 | 0.0983 | 0.1001 |
| τ_pitch RMS (Nm) | 2.7434 | 2.8336 | 2.9139 | 2.9302 |

---

## Failure Mode Classification

### baseline

- `position_torque_cap_saturation`
- `insufficient_velocity_damping`
- `support_velocity_underdamped`
- `extreme_flexion_wheel_effectiveness_loss`
- `contact_coupling_limits_authority`

### E1_k60

- `position_torque_cap_saturation`
- `insufficient_velocity_damping`
- `support_velocity_underdamped`
- `extreme_flexion_wheel_effectiveness_loss`
- `contact_coupling_limits_authority`

### E2_k80

- `position_torque_cap_saturation`
- `insufficient_velocity_damping`
- `support_velocity_underdamped`
- `contact_coupling_limits_authority`

### E3_k100

- `position_torque_cap_saturation`
- `insufficient_velocity_damping`
- `support_velocity_underdamped`
- `contact_coupling_limits_authority`

---

## Interpretation

Based on the failure mode classification, the primary reasons for continuous k_position failure are:

- `support_velocity_underdamped` (present in 4/4 candidates)
- `insufficient_velocity_damping` (present in 4/4 candidates)
- `position_torque_cap_saturation` (present in 4/4 candidates)

---

## Recommended Next Steps for Phase 5

- **Increase max_position_tau** from 3.0 to 6.0 Nm at low heights
- **Increase k_velocity** from 15.0 to 25-30 at low heights
- **Add support velocity damping term** directly

---

## Artifacts Generated

- `authority_transmission_summary.json`
- `authority_transmission_report.md` (this file)
- `authority_saturation_comparison.csv`
- `event_order_comparison.csv`
- `authority_failure_classification.json`
