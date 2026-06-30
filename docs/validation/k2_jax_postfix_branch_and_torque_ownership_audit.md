# K2 JAX Post-Fix Branch and Torque Ownership Audit

**Date:** 2026-06-27
**Classification:** `K2_JAX_BRANCH_AUDIT_CLEAN`

---

## 1. Summary

All 6 branch activity audit tests pass. No unexpected active branches, no hidden torque, no WBC leakage, all disabled mechanisms confirmed inactive, all enabled mechanisms active where expected.

---

## 2. Test Results

```
tests/test_k2_jax_branch_activity_audit.py     6/6 PASS
```

| Test | Result | Detail |
|------|--------|--------|
| `test_disabled_strategies_inactive` | PASS | All disabled K2 mechanisms confirmed inactive/zero |
| `test_enabled_strategies_active` | PASS | All enabled mechanisms active where expected |
| `test_k2_notch_params_correct` | PASS | Notch filter params (fs=100, fc=2.5, Q=2.0) correct |
| `test_no_wbc_enabled` | PASS | WBC completely disabled — no leakage |
| `test_no_hidden_torque_flags` | PASS | No hidden/undocumented torque sources |
| `test_branch_audit_classification` | PASS | 0 UNEXPECTED_ACTIVE branches |

---

## 3. Branch Activity Status

### ENABLED_ACTIVE (all confirmed active):
- Notch filter (2.5 Hz, Q=2.0, height gate 0.42-0.48m)
- Pitch-rate effective computation
- Sagittal wheel balance (pitch + pitch_rate + velocity + support)
- Posture PD (leg position targets)
- Lateral roll balance (hip roll antisymmetric)
- Yaw controller (hip yaw antisymmetric)
- Mode-based hip-yaw divergence (Kp=10.0, Kd=0.5, max_tau=7.5 Nm)
- Height scheduling (k_position, k_wheel_velocity, kd_pitch via smoothstep)
- Calibrated outer loop v2 (Kp/Kd/theta_max/deadband/rate_limit/lowpass)
- Physics equilibrium feedforward (tau per wheel)
- Low-band support shaping
- Empirical support feedforward
- Torque composer (clip + rate limit + lowpass)
- ABS trim logic

### DISABLED_INACTIVE (all confirmed inactive):
- WBC (Whole Body Control) — disabled, zero output
- Stand-up recovery — not implemented
- Locomotion — not implemented
- Stair climbing — not implemented
- Rough terrain — not implemented
- Force feedback PID — not used in K2 profile

---

## 4. Torque Ownership

All torque sources are accounted for:

| Source | Owner | Status |
|--------|-------|--------|
| tau_pitch | Sagittal controller (physics FF) | Active |
| tau_pitch_rate | Sagittal controller (notched pitch rate damping) | Active |
| tau_velocity | Sagittal controller (sagittal velocity damping) | Active |
| tau_support | Sagittal controller (support position error) | Active |
| tau_wheel_vel | Sagittal controller (wheel velocity feedback) | Active |
| tau_position | Posture PD (joint position PID) | Active |
| tau_lateral | Lateral roll balance (hip roll antisymmetric) | Active |
| tau_yaw | Yaw controller (hip yaw antisymmetric) | Active |
| tau_mode_div | Mode-based hip-yaw divergence controller | Active |
| tau_abs | Adaptive bias trim | Active |
| tau_empirical_ff | Empirical support feedforward | Active |
| tau_composer | Final clip + rate limit + lowpass | Active |
| tau_wbc | Whole Body Control | Disabled (0) |
| tau_hidden | Any undocumented source | None detected |

---

## 5. JAX-Extra Diagnostic Mechanisms

All JAX diagnostic fields are non-control-affecting:
- `k2_jax_diag_*` fields: telemetry only, not fed back to torque
- `_jax_state` fields: internal state tracking, no direct torque output
- No post-composer JAX-only torque additions

---

## 6. Classification

**`K2_JAX_BRANCH_AUDIT_CLEAN`**

- 0 UNEXPECTED_ACTIVE branches
- All DISABLED_INACTIVE confirmed inactive
- All ENABLED_ACTIVE branches active where expected
- No hidden torque/WBC leakage
- All JAX diagnostic mechanisms confirmed non-control-affecting
