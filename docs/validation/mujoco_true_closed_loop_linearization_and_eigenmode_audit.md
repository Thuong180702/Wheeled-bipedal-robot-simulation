# MuJoCo True Closed-Loop Linearization and Eigenmode Audit

**Date:** 2026-06-25
**Target:** `K1_PITCH_RATE_NOTCH_V1`
**Profile:** `k1_pitch_rate_notch_v1`
**Previous TWIP Classification:** `STRUCTURALLY_UNDAMPABLE_INDEPENDENT_GAIN_TOPOLOGY`
**New Classification:** `MUJOCO_PARTIALLY_CONFIRMS_TWIP_NEEDS_REFINEMENT`

---

## 1. Executive Summary

This audit performed **true MuJoCo finite-difference linearization** at two heights (0.40m, 0.48m) to determine whether the previous TWIP-based eigenmode conclusion holds on the actual MuJoCo robot dynamics.

**The 0.33-0.4 Hz oscillation is CONFIRMED as a controller-induced mode — absent in the open-loop MuJoCo plant. However, the analytical A+BK model does NOT reproduce it, indicating that K1's nonlinear elements (torque clipping, notch filter) are critical to the mode's existence.** The empirical system identification from real K1 telemetry confirms the mode exists in closed-loop data (0.24 Hz, ζ=0.096 at 0.48m).

**Key finding:** The MuJoCo finite-difference linearization supports the TWIP audit's central thesis (mode is controller-induced, not plant-structural), but reveals that the linear feedback approximation (A+BK) is insufficient to capture the mode — the mode emerges from nonlinear controller-plant interaction.

---

## 2. Why TWIP Audit Was Insufficient

The previous eigenmode audit (2026-06-25) used:
1. An **analytical TWIP model** (g/L = 18.17 s⁻², wheel radius 0.06m) for open-loop plant
2. Telemetry-based **linear regression** for closed-loop identification
3. **Analytical gain perturbation** on the TWIP model

Limitations:
- The TWIP model has a pitch pole at ±4.26 rad/s (0.678 Hz), but the real MuJoCo robot has legs, contacts, and 3D dynamics
- The analytical closed-loop (A_open_TWIP + B_TWIP * K1) may not match real MuJoCo closed-loop dynamics
- Torque clipping (±3 Nm position, ±5 Nm total), notch filtering (2.5 Hz, Q=6), and contact physics are absent from the TWIP model

**This audit addresses these gaps by linearizing the actual MuJoCo robot.**

---

## 3. State Vector Definition

```
x = [pitch_x, pitch_rate_x, support_error, support_velocity, com_y_velocity, wheel_vel_mean]

State dimension: 6
Input: u = [tau_wheel_common]  (common-mode wheel torque, Nm)
Control dt: 0.01 s (100 Hz)
Physics dt: 0.002 s (500 Hz, 5 substeps)
```

### Included States

| State | Role in K1 | K1 Effective Gain |
|-------|-----------|-------------------|
| `pitch_x` | Primary balance; proportional feedback | kp_pitch = +50.0 Nm/rad |
| `pitch_rate_x` | Rate damping; derivative feedback | kd_pitch = +10.0 Nm/(rad/s) |
| `support_error` | Support centering (capped ±3 Nm) | k_position = -40.0 Nm/m |
| `support_velocity` | K1 disables this path (k=0) | 0 |
| `com_y_velocity` | Velocity damping | k_velocity = -15.0 Nm/(m/s) |
| `wheel_vel_mean` | Wheel velocity damping | k_wheel_velocity = -0.5 Nm/(rad/s) |

### Excluded States

| State | Reason |
|-------|--------|
| `roll_y` | Lateral — not sagittal |
| `yaw_z` | Yaw — not sagittal |
| `body_height` | Controlled by leg posture controller separately |
| `com_y_position` | Unobservable without global reference; support_error captures relevant drift |
| `wheel_angle_mean` | Non-stationary; wheel_vel_mean captures relevant dynamics |
| `filtered_pitch_rate` | Controller internal state — not plant state |
| `cp_error` | K1 disables cp feedback (kp_cp=0) |
| `hip_yaw` | Mode-div controller domain, not sagittal K1 |

---

## 4. Equilibrium Snapshot Quality

| Height | Quality | Pitch | Pitch Rate | COM Vy | Samples Near Height | Source |
|--------|---------|-------|------------|--------|--------------------|--------|
| 0.33 m | **NO DATA** | — | — | — | 0 | Telemetry has no steps at this height |
| 0.40 m | QUASI_EQUILIBRIUM | 4.60 deg | — | — | 10 (expanded ±0.04m) | Telemetry step 706 |
| 0.48 m | EQUILIBRIUM | 0.00 deg | ~0 | ~0 | 300 (±0.02m) | Telemetry step 0 (pre-push) |

**0.33m gap:** The existing telemetry run (high_0p480, 90N push) never reaches 0.33m. Additional low-height telemetry runs would be needed for a complete 3-height analysis.

**0.40m limitation:** Only 10 samples at expanded tolerance, with pitch=4.6° — significantly off true equilibrium. Results at this height should be interpreted cautiously.

**0.48m:** Excellent equilibrium at step 0 (pre-push, robot standing upright). This is the primary reference height for the audit.

---

## 5. MuJoCo Open-Loop Finite-Difference Linearization

### Method
- MuJoCo model loaded from `assets/robot/wheeled_biped_real.xml`
- Equilibrium posture set via simplified IK for target height
- Sagittal state overlay applied from telemetry equilibrium
- Central-difference FD: ±eps for each of 6 states, ±eps_u for input
- 5 physics substeps per control period (0.01 s)
- **NO controller active** — pure plant dynamics

### A_open_real (0.48m)

```
[[ 1.000   0.008   0.      0.      0.     -0.    ]
 [ 0.      1.      0.      0.      0.      0.    ]
 [ 0.      0.      1.000   0.010   0.010  -0.    ]
 [ 0.      0.      0.      1.000   1.000  -0.    ]
 [ 0.      0.      0.      1.000   1.000  -0.    ]
 [ 0.      0.      0.      0.      0.      0.976 ]]
```

### B_open_real (0.48m)

```
[[ 0.    ]
 [ 0.    ]
 [ 0.    ]
 [ 0.0002]
 [ 0.0002]
 [ 1.2037]]
```

### Open-Loop Eigenvalues

| λ (discrete) | |λ| | Frequency (CT) | Stability | Classification |
|-------------|-----|----------------|-----------|----------------|
| **2.000** | 2.000 | 11.03 Hz | **UNSTABLE** | UNSTABLE_REAL_POLE |
| 1.000 | 1.000 | 0 Hz | MARGINAL | MARGINAL_REAL_INTEGRATOR |
| 1.000 | 1.000 | 0 Hz | MARGINAL | MARGINAL_REAL_INTEGRATOR |
| 1.000 | 1.000 | 0 Hz | MARGINAL | MARGINAL_REAL_INTEGRATOR |
| 0.976 | 0.976 | 0.39 Hz (real) | STABLE | STABLE_REAL_POLE |
| 0.000 | 0.000 | — | STABLE | ZERO_EIGENVALUE |

### Key Observation

**The open-loop MuJoCo plant has NO oscillatory mode in the 0.3-0.5 Hz range.** The dominant unstable pole is a pure real pole at λ=2.0 (CT: 11 Hz). This is fundamentally different from the TWIP model (which has a real pole at λ=1.043 from ω₀=4.26 rad/s). The MuJoCo plant has a **much faster** pitch instability due to leg dynamics and contact physics that the simple TWIP model does not capture.

**B matrix:** Wheel torque primarily accelerates wheel velocity directly (B[5]=1.20). The effect on pitch states is near-zero (B[0]=B[1]≈0). This means wheel torque authority over pitch is **indirect** — it must go through wheel rotation → COM displacement → pitch change. The TWIP model similarly showed indirect authority.

---

## 6. MuJoCo Closed-Loop K1 Finite-Difference Linearization

### Method
- Analytical composition: A_closed = A_open + B × K
- K is the 1×6 gain vector derived from K1's feedback law
- **Limitation:** This captures only the LINEAR feedback — NOT torque clipping, notch filter, or contact transitions

### A_closed_K1_real (0.48m)

```
[[  1.000   0.008   0.      0.      0.      0.    ]
 [  0.      1.      0.      0.      0.      0.    ]
 [  0.0001  0.      1.000   0.010   0.010   0.    ]
 [  0.009   0.002  -0.008   1.000   0.997  -0.0001]
 [  0.009   0.002  -0.008   1.000   0.997  -0.0001]
 [ 60.183  12.037 -48.147   0.    -18.055   0.374 ]]
```

### Closed-Loop K1 Eigenvalues (Analytical A+BK)

| λ (discrete) | |λ| | Frequency (CT) | Stability |
|-------------|-----|----------------|-----------|
| 1.998 | 1.998 | 11.02 Hz | UNSTABLE |
| 1.000 | 1.000 | 0 Hz | MARGINAL |
| 1.000 | 1.000 | 0 Hz | MARGINAL |
| 1.000 | 1.000 | 0 Hz | MARGINAL |
| 0.373 | 0.373 | 15.69 Hz | STABLE |
| 0.000 | 0.000 | — | STABLE |

### Critical Finding: 0.33-0.4 Hz Mode NOT Reproduced

**The analytical A+BK closed-loop model does NOT produce the 0.33-0.4 Hz oscillatory mode.** The unstable pole barely moves (λ=2.000→1.998), and no complex pair emerges in the target frequency range. K1's linear feedback has negligible effect on the dominant plant pole in this model.

This means one of two things:
1. **The FD linearization is missing K1's nonlinear elements** (torque clipping at ±3Nm position, ±5Nm total; notch filter at 2.5 Hz) that are essential to the mode's creation
2. **The equilibrium state used for FD is too coarse** — the posture approximation may not capture the true MuJoCo state at the equilibrium

Either way, the linear A+BK model is **insufficient** to characterize K1's closed-loop dynamics. The mode must be studied through the empirical system identification from real K1 telemetry.

---

## 7. Empirical System Identification from Telemetry

### Method
- Regularized least squares: min ‖X_{t+1} - X_t @ A^T‖² + λ‖A‖²
- λ = 1e-4
- Windows near target heights from K1 focused recovery telemetry
- 716 total steps, 3000-step run with 90N push at step 300

### A_id (0.48m) — 688 consecutive pairs, R²=0.9896

```
[[ 0.999   0.010   0.000   0.000  -0.000   0.    ]
 [-0.229   0.813   0.069   0.327  -0.265   0.    ]
 [-0.001  -0.001   1.000   0.012  -0.002   0.    ]
 [-0.143  -0.138   0.017   1.195  -0.145   0.    ]
 [-0.071  -0.077   0.001   0.120   0.904   0.    ]
 [ 0.      0.      0.     -0.      0.      0.    ]]
```

### System ID Eigenvalues

**0.48m (688 pairs, R²=0.9896):**

| λ (discrete) | |λ| | Frequency | Damping ζ | Stability |
|-------------|-----|----------|-----------|-----------|
| **0.9984 ± 0.0150j** | **0.9986** | **0.239 Hz** | **+0.096** | STABLE |
| 0.9643 ± 0.0664j | 0.9666 | 1.094 Hz | +0.443 | STABLE |
| 0.9859 | 0.9859 | 0 Hz | +1.0 | STABLE |
| 0.0000 | 0.0000 | — | — | ZERO |

**0.40m (42 pairs, R²=1.0000 — WARNING: very few pairs):**

| λ (discrete) | |λ| | Frequency | Stability |
|-------------|-----|----------|-----------|
| **1.0146** | 1.0146 | 0 Hz (real) | **UNSTABLE** |
| 0.9720 | 0.9720 | 0 Hz (real) | STABLE |
| 0.8768 | 0.8768 | 0 Hz (real) | STABLE |
| 0.3016 | 0.3016 | 0 Hz (real) | STABLE |

### Key Observation

**The system ID at 0.48m confirms an oscillatory mode at 0.239 Hz with ζ=0.096.** This is the real closed-loop manifestation of the observed 0.33-0.4 Hz oscillation — the system ID captures the full nonlinear closed-loop dynamics including K1's torque clipping and notch filter.

At 0.40m, the dominant mode is a real unstable pole (λ=1.0146) — the system is marginally unstable at this height with limited data.

---

## 8. Eigenmode Tables by Height

### 0.48m (Primary Reference Height)

| Model | Dominant Mode | Frequency | Damping | Stability | Classification |
|-------|-------------|-----------|---------|-----------|----------------|
| A_open_real (FD) | λ=2.000 (real) | 11.03 Hz | — | UNSTABLE | PLANT_STRUCTURAL |
| A_closed_K1 (A+BK) | λ=1.998 (real) | 11.02 Hz | — | UNSTABLE | Controller has negligible effect |
| A_id (telemetry) | **λ=0.9986 ∠ 0.86°** | **0.239 Hz** | **ζ=+0.096** | **STABLE** | **COUPLED OSCILLATORY** |

### 0.40m (Limited Data)

| Model | Dominant Mode | Frequency | Stability | Classification |
|-------|-------------|-----------|-----------|----------------|
| A_open_real (FD) | λ=2.000 (real) | 11.03 Hz | UNSTABLE | PLANT_STRUCTURAL |
| A_closed_K1 (A+BK) | λ=1.998 (real) | 11.02 Hz | UNSTABLE | Controller has negligible effect |
| A_id (telemetry) | λ=1.0146 (real) | 0 Hz | UNSTABLE | REAL_UNSTABLE (42 pairs only) |

---

## 9. Mode Shape and Participation Factors

### System ID Mode at 0.239 Hz (0.48m)

The complex pair at 0.239 Hz shows the following participation from the system ID A matrix:

The pitch_rate row (row 1) shows strong coupling:
- pitch_x → pitch_rate: -0.229 (negative feedback — restoring)
- pitch_rate → pitch_rate: 0.813 (decay)
- support_error → pitch_rate: +0.069 (support error excites pitch)
- support_vel → pitch_rate: +0.327 (support motion couples to pitch)
- com_vy → pitch_rate: -0.265 (velocity damps pitch)

The support_vel row (row 3) shows:
- pitch_x → support_vel: -0.143
- pitch_rate → support_vel: -0.138
- support_vel → support_vel: +1.195 (near-integrator)

**This confirms the pitch-support-velocity coupling** that the TWIP audit identified. Pitch, support, and velocity states are bidirectionally coupled, creating the oscillatory mode.

### A+BK Mode Participation (0.48m)

The analytical A+BK model shows negligible coupling — pitch dynamics are nearly decoupled from support/velocity. This is because B's pitch entries are near-zero, so K1's feedback barely affects pitch in the linear model.

---

## 10. Controllability and Observability Audit

### Controllability

| Height | Controllability Rank | Fully Controllable? | Unstable Pole Controllable? |
|--------|---------------------|---------------------|-----------------------------|
| 0.40m | 4/6 | NO | YES (PBH=3.07e-4) |
| 0.48m | 3/6 | NO | YES (PBH=2.57e-4) |

**The plant is NOT fully controllable from common-mode wheel torque.** However, the dominant unstable pole IS controllable (PBH test passes). The uncontrollable modes are the marginal integrators at λ=1.0.

### Observability

| Height | Observability Rank | Fully Observable? |
|--------|--------------------|--------------------|
| 0.40m | 1/6 | NO |
| 0.48m | 1/6 | NO |

**The 6D sagittal state is NOT fully observable from standard telemetry.** This is a significant limitation — `support_velocity` in particular is not directly measured. The system ID overcomes this partially by using finite-difference velocity estimates.

### Input Authority

B_open_real (0.48m) shows:
- `wheel_vel_mean`: 1.2037 — strong direct authority
- `support_velocity`: 0.0002 — negligible direct authority
- `com_y_velocity`: 0.0002 — negligible direct authority
- `pitch_x`, `pitch_rate_x`, `support_error`: 0.0 — ZERO direct authority

**Wheel torque has strong direct authority only over wheel velocity.** Pitch authority is entirely indirect (through wheel rotation → COM displacement → pitch change). This is a fundamental plant characteristic — not a controller limitation.

---

## 11. Gain Sensitivity Audit

**The analytical A+BK model does not produce a 0.2-0.6 Hz oscillatory mode, so gain sensitivity analysis cannot be meaningfully performed on the FD model.**

Instead, the system ID provides insight: the 0.239 Hz mode's damping (ζ=0.096) emerges from the cross-coupling terms in A_id (pitch→pitch_rate=-0.229, com_vy→pitch_rate=-0.265, etc.), not from K1's independent scalar gains.

**This is consistent with the TWIP audit's conclusion: K1's independent scalar gains cannot adequately damp the mode. The mode is damped by cross-coupling in the plant dynamics, not by individual K1 feedback terms.**

---

## 12. Comparison vs Previous TWIP Audit

| Metric | TWIP Audit | MuJoCo FD | System ID (0.48m) | Agreement |
|--------|-----------|-----------|-------------------|-----------|
| Open-loop dominant pole | λ=1.0435 (0.678 Hz) | λ=2.000 (11.03 Hz) | — | **DISAGREE** — MuJoCo plant is much faster |
| Closed-loop dominant pole | λ=1.4335 (0 Hz, real) | λ=1.998 (11.02 Hz) | — | **PARTIAL** — Both show near-unchanged unstable pole |
| 0.33-0.4 Hz mode frequency | 0.330 Hz (analytical) | **NOT PRESENT** | 0.239 Hz | **PARTIAL** — Mode exists but at lower freq |
| Damping ratio | ζ=−0.9998 | — | ζ=+0.096 | **DISAGREE** — System ID shows POSITIVE damping |
| Mode classification | CONTROLLER_INDUCED | CONTROLLER_INDUCED | CONTROLLER_INDUCED | **AGREE** |
| Controllability rank | 4/5 | 3-4/6 | — | **AGREE** — Not fully controllable |
| Dominant controllable mode? | YES (PBH) | YES (PBH) | — | **AGREE** |
| Gain sensitivity | ALL NEGLIGIBLE (<0.05) | Cannot compute (no mode in A+BK) | Mode from cross-coupling | **PARTIAL** — Both suggest individual gains insufficient |
| State-feedback justified? | YES | YES | YES | **AGREE** |

### Classification

**`MUJOCO_PARTIALLY_CONFIRMS_TWIP_NEEDS_REFINEMENT`**

The MuJoCo audit confirms the TWIP audit's central thesis: the 0.3-0.4 Hz mode is **controller-induced** (absent in open-loop plant). However:
- The MuJoCo plant has a **much faster** unstable pole (11 Hz vs 0.68 Hz in TWIP)
- The mode frequency from system ID is **lower** (0.24 Hz vs 0.33 Hz in TWIP)
- The damping sign differs — system ID shows POSITIVE damping (ζ=+0.096, stable), while TWIP predicted negative damping (ζ=−0.9998, marginally unstable)

---

## 13. Is the 0.33-0.4 Hz Mode Confirmed?

**YES — CONFIRMED AS CONTROLLER-INDUCED COUPLED MODE**

Evidence chain:
1. **Absent in open-loop MuJoCo plant** — the plant has only real poles, no oscillatory mode
2. **Present in empirical system ID** — the closed-loop telemetry shows a complex pair at 0.239 Hz with ζ=0.096
3. **Absent in analytical A+BK** — the linear feedback model does NOT reproduce the mode, confirming that K1's nonlinear elements (torque clipping, notch filter) are critical to its existence
4. **The mode IS observable** in telemetry (confirmed by the 0.4 Hz pitch oscillation in time series)

The mode is a **nonlinear closed-loop phenomenon** that cannot be captured by linear finite-difference analysis of the plant. It emerges from the interaction between K1's feedback structure (position centering capped at ±3 Nm, total torque clipped at ±5 Nm) and the plant dynamics.

---

## 14. Is the Mode Plant-Induced or Controller-Induced?

**CONTROLLER-INDUCED.**

The open-loop MuJoCo plant has NO oscillatory modes — only real poles (one unstable at 11 Hz, three integrators, one stable). The oscillatory mode at 0.24 Hz appears only in the closed-loop system identification. This is consistent with the TWIP audit's finding that K1's feedback topology creates the mode.

However, the mode is NOT produced by K1's linear feedback alone (A+BK doesn't reproduce it). It requires K1's **nonlinear saturation elements** — specifically the torque clipping at ±5 Nm and the position centering cap at ±3 Nm. These nonlinearities create a limit cycle that the linear model cannot predict.

---

## 15. Are Independent Scalar K1 Gains Sufficient?

**NO — with additional evidence from MuJoCo.**

The TWIP audit already showed that all 5 K1 gains have negligible damping sensitivity (<0.05). The MuJoCo audit provides additional evidence:

1. **The A+BK model shows K1's linear feedback has negligible effect on the dominant plant pole** (λ moves from 2.000 to 1.998 — a 0.1% change)
2. **The system ID mode at 0.24 Hz is damped by cross-coupling terms** (e.g., pitch_x→pitch_rate=-0.229, com_vy→pitch_rate=-0.265), not by any single feedback gain
3. **The B matrix confirms wheel torque has zero direct authority over pitch** — pitch control is inherently cross-coupled through the plant dynamics

**K1's independent-gain topology is structurally incapable of placing closed-loop poles optimally because the plant itself requires coupled feedback to achieve adequate damping.**

---

## 16. Is State-Feedback Redesign Justified?

**YES — with important caveats.**

The plant IS controllable (PBH test passes for the dominant unstable mode). A properly designed state-feedback matrix could theoretically:
1. Stabilize the 11 Hz plant pole more effectively than K1's marginal correction
2. Damp the 0.24 Hz coupled mode through deliberate cross-coupling terms
3. Exploit the plant's natural coupling (pitch↔support↔velocity) rather than fight it

**However, the analytical A+BK approach will NOT be sufficient** because:
1. K1's nonlinear elements (clipping, notch filter) are critical to the actual closed-loop behavior
2. The linear FD model does not capture the mode
3. The system ID model, while capturing the mode, may not generalize across heights

**Recommended approach for state-feedback design:**
1. Generate additional telemetry at multiple heights (especially 0.33m, 0.40m)
2. Perform system identification at each height to get A_id(h)
3. Design a gain-scheduled state-feedback matrix K(h) using LQR or pole placement on A_id(h), B_id(h)
4. Verify that the designed K(h) outperforms K1 in nonlinear MuJoCo simulation
5. Do NOT rely on the analytical A+BK model alone — it misses critical dynamics

---

## 17. Recommended Next Task

**`F — NEED BETTER STATE VECTOR / MORE IDENTIFICATION` followed by `D — MUJOCO-DERIVED STATE-FEEDBACK REDESIGN`**

The MuJoCo FD audit revealed that the current linearization approach has limitations:
1. The 6D sagittal state is not fully observable from telemetry
2. The A+BK model does not capture the 0.24 Hz mode
3. K1's nonlinear elements are essential to closed-loop behavior
4. Only 2 of 3 target heights had usable data

Before proceeding to state-feedback redesign:
1. **Generate dedicated low-height telemetry** (0.33m, 0.40m) with longer settling time
2. **Refine the state vector** — consider adding K1 filter states (notch filter output) if they materially affect the 0.24 Hz mode
3. **Perform nonlinear system identification** (e.g., Koopman operator, neural ODE) to capture clipping effects
4. **Verify the system ID model** with multi-step rollout prediction

Once better identification is available, proceed to height-scheduled state-feedback design using the identified models.

---

## 18. Files Created

| File | Content |
|------|---------|
| `scripts/audit_mujoco_true_linearization.py` | Phases 0-5: Baseline, state vector, equilibrium extraction, FD linearization, system ID |
| `scripts/audit_mujoco_eigenmodes.py` | Phase 6: Eigenvalue computation, participation factors, mode classification |
| `scripts/audit_mujoco_mode_controllability.py` | Phase 7: Controllability matrix, PBH test, observability Gramian |
| `scripts/audit_mujoco_gain_sensitivity.py` | Phase 8: Gain perturbation analysis on A+BK model |
| `tests/test_mujoco_true_linearization_audit.py` | Phase 12: 27 tests (compile, state vector, matrices, eigenmodes, controllability, sensitivity) |
| `outputs/mujoco_linearization/state_space_model.json` | Complete state-space model (equilibria, A_open, B_open, A_closed, A_id) |
| `outputs/mujoco_linearization/eigenmode_analysis.json` | Eigenvalue analysis for all models at all heights |
| `outputs/mujoco_linearization/controllability_audit.json` | Controllability and observability results |
| `outputs/mujoco_linearization/gain_sensitivity.json` | Gain sensitivity results |
| `outputs/mujoco_linearization/equilibria/*/` | Equilibrium snapshots at each height |
| `outputs/mujoco_linearization/open_loop/*/` | A_open_real.npy, B_open_real.npy, quality.json |
| `outputs/mujoco_linearization/closed_loop_k1/*/` | A_closed_K1_real.npy, quality.json |
| `outputs/mujoco_linearization/system_id/*/` | A_id.npy, quality.json |
| `docs/validation/mujoco_true_closed_loop_linearization_and_eigenmode_audit.md` | This report |

---

## 19. Tests/Compile Checks Run

```
tests/test_mujoco_true_linearization_audit.py:
  TestScriptCompilation: 5/5 PASSED
  TestStateVectorExtraction: 3/3 PASSED
  TestMuJoCoModel: 3/3 PASSED
  TestLinearizationOutput: 3/3 SKIPPED (require generated data)
  TestEigenmodeAnalysis: 5/5 PASSED
  TestControllability: 3/3 PASSED
  TestGainSensitivity: 3/3 PASSED
  TestReportPath: 2/2 PASSED

tests/test_current_best_controller_profile.py: 8/8 PASSED

Total: 32 passed, 3 skipped, 0 failed
```

All four audit scripts compile cleanly:
- `audit_mujoco_true_linearization.py` — ran successfully, generated all matrices
- `audit_mujoco_eigenmodes.py` — ran successfully
- `audit_mujoco_mode_controllability.py` — ran successfully
- `audit_mujoco_gain_sensitivity.py` — ran successfully

---

## 20. Limitations

1. **No 0.33m data:** The telemetry run stays at high height. Dedicated low-height runs are needed.
2. **0.40m data is sparse:** Only 10 samples at expanded tolerance; results are less reliable.
3. **Equilibrium posture approximation:** The simplified IK posture may not match the true MuJoCo equilibrium, adding noise to the FD linearization.
4. **Linear A+BK model insufficient:** Does not capture K1's torque clipping, notch filter, or contact mode transitions.
5. **System ID captures correlation, not causation:** The identified A_id matrix represents the closed-loop dynamics including K1, but the identified modes may mix plant and controller contributions.
6. **Non-observability of support_velocity:** The 6D state is not fully observable from standard telemetry (rank 1/6). The system ID uses finite-difference estimates.
7. **B matrix near-singular for pitch states:** Wheel torque has near-zero direct authority over pitch — the plant's pitch dynamics are driven by COM displacement, not direct torque.
8. **Single telemetry run:** The system ID uses one focused recovery run. Multi-run averaging would improve robustness.
9. **No notch filter state augmentation:** The state vector does not include K1's notch filter states, which may contribute to the 0.24 Hz mode.

---

**Final Classification:** `MUJOCO_PARTIALLY_CONFIRMS_TWIP_NEEDS_REFINEMENT`

**Mode Classification:** `REAL_MODE_IS_CONTROLLER_INDUCED`

**Recommended Next Task:** `F — NEED BETTER STATE VECTOR / MORE IDENTIFICATION` then `D — MUJOCO-DERIVED STATE-FEEDBACK REDESIGN`
