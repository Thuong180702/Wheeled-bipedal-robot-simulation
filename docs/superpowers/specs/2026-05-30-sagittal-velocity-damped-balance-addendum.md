# Analytic Sagittal Velocity-Damped Balance Controller — Addendum

> **Status:** `model_identification_failed` — LQR/sysid path stopped. New analytic path begins.

## Context

The original implementation plan (2026-05-29) specified a closed-loop system-identification approach to build a discrete-time sagittal dynamics model, then derive an LQR/state-feedback controller from that model.

Gate 4 of that plan ran on 5000 steps of closed-loop nominal telemetry. The identified model passed the one-step R² gate (1.0000) but failed the 20-step rollout R² gate catastrophically (R² = -1.15×10¹⁰). The root cause was identified:

- A[1,1] = 1.328: sagittal velocity grows without damping
- Dominant eigenvalue λ₀ = 1.9649: unstable velocity mode
- The baseline sagittal controller stabilizes pitch but does not damp sagittal velocity

This is a diagnostic finding, not a code bug. The model is useful for understanding what is missing but cannot be used to design a controller.

## Decision

**The LQR/sysid path is stopped. No `SagittalPositionAwareBalanceController` will be designed from the failed identified model.**

A new controller-by-construction approach begins, building the sagittal controller from explicit physical terms rather than from regression over a system that lacks velocity damping.

## What This Is Not

- Not E0b/E0c/E0d — those were add-on torque patches that fought the inner balance loop
- Not a patch correction on top of the old sagittal controller
- Not derived from the failed identified A/B matrices
- Not a capture-point or phase-aware shaping layer
- Not WBC reintroduction

## Controller Strategy

Build a clean sagittal controller in explicit layers:

### Layer 1: Pitch balance parity
Match or preserve the old controller's pitch stabilization behavior. Do not regress nominal 1000-step balance.

### Layer 2: Sagittal velocity damping
Add explicit damping on sagittal velocity. This is the primary fix suggested by Gate 4's diagnostic finding. The controller must actively oppose forward/backward velocity growth.

### Layer 3: Wheel velocity damping
Dampen wheel velocity runaway. Prevent wheels from spinning continuously in one direction.

### Layer 4: Weak position return
Only after velocity damping is validated as stable. Position return must be weak and slow. It generates a desired velocity or state term, not raw external torque. It must not fight pitch recovery.

## State Vector

```text
x = [
    sagittal_position_error,   # m — displacement along initial-heading axis
    sagittal_velocity,        # m/s — CoM velocity along initial-heading axis
    pitch_x,                  # rad — robot-frame sagittal tilt
    pitch_rate_x,             # rad/s — sagittal angular velocity
    wheel_velocity_mean,       # rad/s — mean of left/right wheel velocities
]
```

Axis convention (project standard):
- X: lateral
- Y: sagittal
- Z: vertical

Displacement and velocity are measured in the **initial-heading frame**, not raw world-frame Y alone. Use existing `sagittal_balance_state.py` helpers.

## Control Output

- Output: scalar wheel torque applied to both wheel joints [4, 9]
- Torque ownership: wheel joints only — no hip/knee/hip-roll outputs
- Wheel joints [4, 9] are velocity-target joints through low-level PID

## Controller Formula

```
tau_total = tau_pitch + tau_pitch_rate + tau_sagittal_velocity + tau_wheel_velocity + tau_position
```

Where each term is constructed explicitly. Signs must be verified by unit tests before simulation.

Initial gain strategy (controller-by-construction):

```
tau_pitch             = k_pitch          * pitch_error
tau_pitch_rate        = k_pitch_rate     * pitch_rate
tau_sagittal_velocity = k_velocity       * (-sagittal_velocity)   # damping
tau_wheel_velocity    = k_wheel_velocity * (-wheel_velocity_mean) # damping
tau_position          = k_position       * (-sagittal_position_error)  # weak return
```

- `k_pitch`: restoring torque proportional to pitch tilt (sign: same as old controller — verify)
- `k_pitch_rate`: damping torque proportional to pitch rate (sign: opposes angular velocity — verify)
- `k_velocity`: velocity damping gain (sign: must oppose velocity — verify)
- `k_wheel_velocity`: wheel velocity damping gain (sign: must oppose wheel spin — verify)
- `k_position`: weak position return gain (starts at 0, enabled only after velocity damping is stable)

## Mutual Exclusion

Baseline and new controller are strictly mutually exclusive:

- `--sagittal-controller baseline` → `SagittalWheelBalanceController` active, new controller zero
- `--sagittal-controller velocity-damped` → new controller active, baseline zero

They must never both contribute torque simultaneously. Enforced by CLI routing and verified by test.

## Validation Gates

### Gate A: Spec/Plan Addendum
This document. No implementation yet.

### Gate B: Clean Implementation
- Create `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`
- Output only wheel joints [4, 9]
- No WBC, no ownership change
- CLI flag `--sagittal-controller velocity-damped`
- Default remains `baseline`

### Gate C: State and Frame Validation
- Run existing `tests/test_sagittal_balance_state.py`
- Add tests for nonzero yaw projection, positive/negative displacement, positive/negative velocity

### Gate D: Unit/Sign Tests Before Simulation
All tests must pass before any simulation run:

1. **Pitch restoring**: positive pitch error → wheel torque in restoring direction
2. **Pitch-rate damping**: positive pitch_rate → damping torque
3. **Sagittal velocity damping**: positive sagittal velocity → torque that reduces forward velocity
4. **Wheel velocity damping**: positive wheel_velocity_mean → opposing torque
5. **Position term**: with k_position=0, position has no effect; with small k_position, weak return tendency
6. **Term decomposition**: diagnostics include tau_pitch, tau_pitch_rate, tau_sagittal_velocity, tau_wheel_velocity, tau_position, tau_total_unclipped, tau_total_clipped
7. **Mutual exclusion**: baseline mode → new controller output zero; velocity-damped mode → baseline output zero

### Gate E: Integration Smoke Test
```
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --sagittal-controller baseline --steps 100
python scripts/simulate_hierarchical_controller.py --controller-mode balance-core --sagittal-controller velocity-damped --steps 100
```
Verify: exactly one sagittal controller active, WBC off, ownership violation = 0, wheel joints [4, 9] receive sagittal torque.

### Gate F: Incremental Validation

**F1 — Baseline parity (pitch terms only, velocity/position gains zero):**
- nominal 1000 steps
- Expected: pass, pitch comparable to baseline, no ownership violation

**F2 — Add sagittal velocity damping (k_velocity > 0, k_position = 0):**
- nominal 1000 and 5000 steps
- Goal: reduce sagittal velocity growth, reduce max drift vs 35.22 m baseline
- No pitch/roll/height regression

**F3 — Add wheel velocity damping (k_wheel_velocity > 0):**
- nominal 1000 and 5000 steps
- Goal: reduce wheel velocity runaway, further reduce drift, no oscillatory instability

**F4 — Add weak position return (k_position small > 0):**
- Only after F2 and F3 are stable
- nominal 1000 and 5000 steps
- Goal: bounded position return, no fighting of pitch recovery

**F5 — Height variant regression:**
- high_5cm 500 steps, low_5cm 500 steps if nominal passes

### Gate G: Drift Acceptance

| Gate | Threshold | Description |
|------|-----------|-------------|
| baseline (no containment) | 35.22 m / 5000 steps | current drift |
| E0b (failed) | 15.98 m | historical context |
| E0c (failed) | 63.72 m | historical context |
| E0d (failed) | 121.39 m | historical context |
| **minimum acceptable** | ≤ 17.6 m | 50% of baseline |
| **target** | ≤ 5.0 m | meaningful improvement |
| **preferred** | ≤ 0.50 m max, ≤ 0.20 m final | near-stationary |

Do not claim full position hold unless preferred target is met.

### Gate H: Failure Handling
If any stage fails, stop and classify:
- `pitch_regression` — pitch behavior worse than baseline
- `velocity_damping_sign_error` — velocity term has wrong sign
- `wheel_velocity_damping_sign_error` — wheel velocity term has wrong sign
- `excessive_velocity_damping` — velocity damping too aggressive, causes pitch instability
- `position_term_fights_balance` — position return destabilizes pitch
- `oscillatory_hunting` — controller causes limit-cycle behavior
- `height_collapse` — height drops below safe operating range
- `contact_invalid` — contact state becomes invalid
- `wheel_velocity_runaway` — wheel velocity not contained

Do not blindly tune gains. Report exact telemetry and term decomposition.

## Required Output

Output directory: `outputs/sagittal_velocity_damped_balance/`

Required artifacts:
- `analytic_controller_addendum.md` (this document)
- `velocity_damped_controller_unit_report.md`
- `validation_summary.json`
- `validation_summary.md`
- Per-run telemetry CSVs
- Failure reports if any

Required final report fields:
- files changed
- tests added/updated
- commands run
- controller formula
- term signs
- gains used and why
- validation gates passed/failed
- max/final drift
- pitch/roll/yaw/com_z ranges
- wheel velocity ranges
- torque saturation
- ownership_violation_count
- hidden_torque_norm
- tau_wbc_norm
- comparison against baseline and failed E0 attempts
- whether minimum/target/preferred gates passed
- confirmation no WBC
- confirmation no E0b/E0c/E0d reintroduced
- confirmation old and new sagittal controllers are mutually exclusive
- confirmation torque ownership unchanged

## Naming Rules

Allowed production names:
- `SagittalVelocityDampedBalanceController`
- `SagittalBalanceState` (existing)
- `SagittalBalanceReference`
- `SagittalVelocityDampingConfig`
- `PositionAwareBalanceValidation`

Forbidden production names:
- `E0Controller`, `E0b`, `E0c`, `E0d`
- `Stage2E`, `temp_position_fix`, `position_patch`, `quick_fix`, `hack_containment`
- `SagittalPositionAwareBalanceController` (reserved for LQR path — not used here)

## Execution Policy

Do not execute all gates as one batch. Each gate is a stop gate. Proceed only if the current gate passes.

Do not:
- tune gains blindly
- run both baseline and new controller simultaneously
- add raw position torque as an external correction
- reintroduce WBC
- change torque ownership
- use fake contact force
- use legacy wheel balance or hip-roll centering
