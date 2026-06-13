# Phase B.9 Step 5 — Best Standalone Controller Report

## Scope
This report summarizes Phase B.9 Step 5 (LQR gain strengthening) only, based on staged sweep artifacts in `outputs/phase_b9_lqr_gain_strengthening/` and Step 4 baseline metrics from `outputs/phase_b9_slow_loop_gating/slow_loop_summary.csv`.

## Stage A result
Source: `outputs/phase_b9_lqr_gain_strengthening/stage_a_trials.csv`

- Stage A explores `lqr_gain_scale × pitch_gain_mult × wheel_cmd_limit_mult` at height 0.60 m.
- Best Stage A candidate by survival:
  - `lqr_gain_scale=5.0, pitch_gain_mult=5.0, wheel_cmd_limit_mult=1.0`
  - `mean_survival_s=0.88`
  - `mean_fall_rate=1.00`
  - `mean_roll_rms_deg=17.10`
  - `mean_action_sat_rate=0.0`
- Verdict: Stage A identifies stronger gain region but still fully unstable standalone (100% fall).

## Stage B result
Source: `outputs/phase_b9_lqr_gain_strengthening/stage_b_top5.csv`

- Stage B sweeps around Stage A top-5 parents with additional multipliers and filtering.
- Top Stage B frontier reaches:
  - `mean_survival_s=0.6667`
  - `mean_fall_rate=1.00`
  - `mean_roll_rms_deg≈18.48`
  - `mean_action_sat_rate=0.0`
- Top-2 parent selection for Stage C (script-ranked by survival, then carried forward) is from this top frontier and satisfies the user criteria (highest survival, low roll among tied candidates, no severe saturation).
- Verdict: Stage B improves survival structure but remains 100% fall; still insufficient standalone.

## Stage C result
Source: `outputs/phase_b9_lqr_gain_strengthening/stage_c_final.csv`

- Stage C settings:
  - Heights: `[0.65, 0.60, 0.55, 0.50, 0.45, 0.40]`
  - Episodes per height: `5`
  - Total configs evaluated: `18`
- Best Stage C config (`config_id=1`):
  - `mean_survival_s=3.8926666666666665`
  - `mean_fall_rate=0.8333333333333334`
  - `mean_pitch_rms_deg=1.0109260130054514`
  - `mean_roll_rms_deg=21.168228816065504`
  - `mean_action_sat_rate=0.0`
  - dominant fall reason: `tilt`
- Verdict: Stage C delivers large survival gain and lower fall rate vs Step 4 baseline, but still not robust standalone.

## Best LQR gain setting
Sources:
- `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`
- `outputs/phase_b9_lqr_gain_strengthening/best_lqr_summary.json`

```yaml
lqr_gain_scale: 3.0
pitch_gain_mult: 5.0
wheel_cmd_limit_mult: 3.0
pitch_rate_gain_mult: 2.0
com_gain_mult: 2.0
filter_alpha: 0.0
com_rate_gain_mult: 1.0
filter_max_delta_mult: 1.0
```

## Survival comparison vs Step 4 `slow_loop_disabled`
Step 4 baseline source: `outputs/phase_b9_slow_loop_gating/slow_loop_summary.csv`

- Step 4 `slow_loop_disabled` mean survival across heights 0.65→0.40:
  - `(0.40 + 0.38 + 0.34 + 0.72 + 1.00 + 0.64) / 6 = 0.58 s`
- Step 5 Stage C best mean survival:
  - `3.8927 s`

Comparison:
- Absolute gain: `+3.3127 s`
- Relative gain: `+571%` (about `6.71×` longer survival)

## Roll instability improvement status
Using aggregated means:
- Step 4 `slow_loop_disabled` roll RMS mean across heights: `≈21.61°`
- Step 5 Stage C best roll RMS: `21.17°`

Interpretation:
- Roll RMS improves only marginally (`≈0.44°` better, ~2.0%).
- Dominant fall reason remains `tilt` in Stage C.
- Therefore, roll instability is **not fundamentally solved**; improvement is limited.

## Step 5 verdict
**Step 5 status: SUCCESS (for tuning objective), NOT READY for robust standalone adoption.**

- Success because Stage C completed and produced the required artifacts:
  - `stage_c_final.csv`
  - `best_lqr_config.yaml`
  - `best_lqr_summary.json`
- Quantitative objective improved strongly in survival and moderately in fall rate.
- However, controller still shows high failure rate (`83.3%`) and tilt-dominant termination.

## Step 5.13-5.18c: Mainline Controller Infrastructure

After Step 5 (LQR gain strengthening), Steps 5.13-5.18c added:
- **Step 5.13**: Fixed invalid reset equilibrium (root initialization bug)
- **Step 5.14**: Added lateral balance layer (VMC-based roll stabilization)
- **Step 5.15**: Added VMC support redistribution (whole-body control)
- **Step 5.16**: Added Jacobian-informed WBC/VMC mapping
- **Step 5.17**: Proved diagnostic torque authority exists
- **Step 5.18**: Added deployable motor torque interface
- **Step 5.18b**: Activated hybrid PID+torque rollout path
- **Step 5.18c**: Calibrated torque scaling and saturation

**Step 5.18c key findings**:
- Root cause: `PID_CONTROLLER_SATURATION_NOT_TORQUE_RESIDUAL`
- PID outputs: ~30 Nm (saturates at ctrlrange limits)
- WBC torque residuals: ~1 Nm
- Authority ratio: 1:30 (WBC:PID)
- Saturation rate: 93.75%
- Best candidate (strong_k20): h=0.60 survival = 0.86s (+65% vs baseline)
- Does NOT beat reset-fixed baseline (3.8167s)

## Step 5.19: Controller Authority Reallocation

**Implementation**: PID output clamping to reserve actuator headroom for WBC corrections.

**Status**: Implementation complete, tests passing, evaluation pending.

**Approach**: Clamp PID output to fraction of actuator range (e.g., 70% = ±21 Nm), reserving headroom (30% = ±9 Nm) for WBC residuals.

**Files modified**:
- [wheeled_biped/sim/low_level_control.py](wheeled_biped/sim/low_level_control.py) - Added `pid_authority_fraction` parameter
- [wheeled_biped/envs/balance_env.py](wheeled_biped/envs/balance_env.py) - Wired config and call site
- [tests/test_phase_b9_step5_19_controller_authority_reallocation.py](tests/test_phase_b9_step5_19_controller_authority_reallocation.py) - 7 tests, all passing

**Test configs created**: 6 candidates with `pid_authority_fraction` ∈ {1.0, 0.9, 0.8, 0.7, 0.6, 0.5}

**Critical architectural concern**: Authority reallocation may not solve the fundamental problem. PID saturation indicates the robot is marginally stable even with full PID authority. Clamping PID may weaken primary control and cause faster falls. See [phase_b9_step5_19_controller_authority_reallocation_report.md](docs/phase_b9_step5_19_controller_authority_reallocation_report.md) for detailed analysis.

**Evaluation pending**: Run `python scripts/phase_b9_step5_19_quick_eval.py` to test whether authority reallocation improves, degrades, or has no effect on stability.

## Step 5.20: Low-Stiffness Dynamic Balance Transition

**Implementation**: Soft dynamic balance mode with systematic stiffness reduction.

**Status**: Implementation complete, tests passing (7/7), evaluation pending.

**Core hypothesis**: The current controller is over-stiff and fighting natural balancing dynamics. Pure RL previously balanced successfully without persistent saturation, but current PID saturates at ±30 Nm continuously. This suggests the plant is stabilizable, but the classical control structure may be inefficient.

**Approach**: Systematically reduce LQR gains and add deadband to allow dynamic balancing motion instead of forcing exact posture.

**Files modified**:
- [wheeled_biped/controllers/dual_rate_balance_controller.py](wheeled_biped/controllers/dual_rate_balance_controller.py) - Added soft dynamic balance config and stiffness reduction logic

**Test configs created**: 4 candidates with stiffness reduction ∈ {1.0 (baseline), 0.7 (conservative), 0.5 (moderate), 0.3 (aggressive)}

**Design shift**: From pose-first (rigid posture tracking, high PID dominance) to balance-first (soft posture compliance, allow natural torso lean, PID becomes soft tracking layer).

**Evaluation pending**: Run `python scripts/phase_b9_step5_20_evaluation.py` to test whether reducing posture stiffness improves stability.

**Expected outcomes**:
- Optimistic: Lower saturation, longer survival, lower RMS torque → Controller was over-stiff
- Neutral: Efficiency gain without survival gain → Stiffness reduction helps slightly
- Pessimistic: Faster falls, loss of posture control → Current stiffness is necessary

See [step5_20_summary.md](outputs/phase_b9_step5_20_low_stiffness_dynamic_balance/step5_20_summary.md) for detailed analysis.

## Step 6 gating decision (after Step 5.20)
**Decision: DO NOT move to Step 6 yet.**

Rationale:
- Step 5.19 and 5.20 implementations complete but evaluations pending
- Both steps test architectural hypotheses about controller over-constraining
- Current best controller (Step 5.18c strong_k20): h=0.60 survival = 0.86s
- Reset-fixed baseline: 3.8167s survival across all heights
- Step 6 gate requires beating reset-fixed baseline
- Alternative approaches may be needed if both authority reallocation and soft mode fail

## Phase B.9 Step 5.5 — Roll/Tilt Failure Diagnosis and Fix

### Scope and outputs
Source directory: `outputs/phase_b9_step5_5_roll_tilt_fix/`

Generated artifacts:
- `diagnostics.csv`
- `candidate_results.csv`
- `best_candidate_full_diagnostics.csv`
- `full_validation_summary.json`
- `best_roll_fix_config.yaml`
- `best_roll_fix_summary.json`

### Small evaluation (3 episodes × heights 0.60, 0.50, 0.40)
Candidates evaluated independently:
- `baseline`
- `A_weak_hip_roll_pd`
- `B_strong_hip_roll_pd`
- `C_roll_rate_damping`
- `D_contact_force_balance`
- `E_lateral_com_correction`
- `F_reduced_wheel_limit`

Best candidate by ranking rule (survival desc, then fall rate asc, roll RMS asc, sat rate asc):
- `D_contact_force_balance`
  - `mean_survival_s = 7.1067`
  - `mean_fall_rate = 0.6667`
  - `mean_roll_rms_deg = 24.06`
  - `mean_action_sat_rate = 0.0`

### Full validation (5 episodes × heights 0.65→0.40)
Best-candidate full metrics:
- `mean_survival_s = 3.6840`
- `mean_fall_rate = 0.8667`
- `mean_pitch_rms_deg = 0.9201`
- `mean_roll_rms_deg = 21.3888`
- `mean_action_sat_rate = 0.0`
- dominant fall reason: `tilt` (`26/30`), non-fall: `4/30`

### First-divergence analysis
From `best_candidate_full_diagnostics.csv`:
- first divergence variable: `roll`
- mean first divergence time: `0.144 s`
- divergence count summary:
  - `roll: 5`
  - `com_lateral: 1`

Interpretation:
- The earliest unstable signal remains roll divergence.
- Contact-force balancing improves small-eval ranking but does not remove tilt-dominant full-eval failures.

## Step 6 gating decision (after Step 5.5)
**Decision: DO NOT move to Step 6 yet.**

Rationale:
- Full-validation fall rate remains high (`86.67%`).
- Roll RMS remains high (`21.39°`) and above robust-readiness target.
- Dominant failure mode remains `tilt`, with first divergence most often in `roll`.
- Step 5.5 improves candidate selection and diagnosis quality, but does not close standalone robustness.

## Phase B.9 Step 5.7 — Early Roll Stabilizer Design

- Output dir: `outputs/phase_b9_step5_7_early_roll_stabilizer`
- Best variant: `A_roll_rate_damping_from_t0`
- Full-validation survival: 3.4180 s
- Full-validation fall rate: 0.8667
- Full-validation roll RMS: 21.2271 deg
- Step 6 ready: `False`

## Phase B.9 Step 5.8 — Roll Instability Root-Cause Redesign

- Output dir: `outputs/phase_b9_step5_8_roll_redesign`
- Baseline controller accepted: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`
- Previous fixes rejected: Step 5.5 / 5.6 / 5.7 did not beat Step 5 full validation
- Best Step 5.8 variant: `C_hip_roll_preload_from_balanced_root`
- Kept variants after small eval: `C_hip_roll_preload_from_balanced_root, D_lateral_CoM_feedback_through_hip_roll`
- Full-validation survival: 3.8993 s
- Full-validation fall rate: 0.8333
- Full-validation roll RMS: 21.5296 deg
- Beats Step 5 baseline in full validation: `False`
- Step 6 allowed: `False`
- If no beat: keep Step 5 best as current best and Step 6 blocked

## Phase B.9 Step 5.9 — Roll Authority and Coupling Audit

- Output dir: `outputs/phase_b9_step5_9_roll_authority_audit`
- Dominant mechanism: `F_timing_delay_and_early_transient_mismatch`
- Why Step 5.8 failed: No single actuator saturates, but early transient correction remains too late/weak.
- Classical roll-control feasibility: Marginal with current architecture.
- Recommended next path: Keep Step 5 best as prior; do not advance Step 6 now.
- Gate status: Step 6 blocked; keep Step 5 best_lqr_config.yaml as current best

## Phase B.9 Step 5.10 — Early Transient Timing Fix

- Output dir: `outputs/phase_b9_step5_10_early_transient_fix`
- Baseline controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`
- Small-eval variants tested: `A_preload_hip_roll_target_at_t0, B_roll_rate_damping_from_first_step, C_bypass_filter_rate_limiter_first_0p2s, D_startup_emergency_roll_mode, E_predictive_roll_correction`
- Candidates kept after filter: `none`
- Full-validation executed: `False`
- Best variant: `none`
- Best full-validation survival: nan s
- Best full-validation fall rate: nan
- Best full-validation roll RMS: nan deg
- Best full-validation pitch RMS: nan deg
- Baseline beaten in full validation: `False`
- Final decision: `KEEP_STEP5_BASELINE_AND_BLOCK_STEP6`
- Step 6 status: `BLOCKED`

## Phase B.9 Step 5.11 — Corrective Path Validity Audit

- Output dir: `outputs/phase_b9_step5_11_corrective_path_audit`
- Baseline replay matches original Step 5: `True`
- Latency marker semantics: corrective vs generic separation implemented
- Sign/index audit: hip_roll qpos indices [7, 12] confirmed
- Authority probe mean roll amplification: 5.249
- State leakage detected: `False`
- Final decision: `CORRECTIVE_PATH_VALID_BUT_AUTHORITY_INSUFFICIENT`
- Decision reason: Roll amplifies despite corrective action; classical authority is insufficient
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.16 — Jacobian WBC/VMC Design

- Output dir: `outputs/phase_b9_step5_16_jacobian_wbc_vmc/`
- Baseline used: reset-fixed/post-reset Step 5 only.
  - all-height mean survival: `3.8167 s`
  - all-height fall rate: `0.8333`
  - all-height pitch RMS: `1.1938 deg`
  - all-height roll RMS: `21.1630 deg`
  - h=0.60 survival: `0.52 s`
  - h=0.60 fall rate: `1.0`
- Mainline integration: Step 5.14 `lateral_balance` and Step 5.15 `vmc_whole_body` are restored as first-class controller infrastructure, disabled by default.
- Why Step 5.15 was insufficient: heuristic force redistribution detected some contact-force authority but did not improve survival enough to pass the reset-fixed gate.
- Torque-level WBC feasibility: `False` for the current controller path. The controller emits normalized leg position targets and wheel velocity targets; it does not command generalized torques.
- WBC/Jacobian formulation: desired roll torque, lateral force, vertical support, and left/right force redistribution are mapped to bounded normalized hip-roll, hip-pitch, knee, and optional wheel-differential target offsets.
- Jacobian mapping result: `diagnostic_jacobian_available` for MuJoCo diagnostic paths, but runtime controller does not receive `mj_data` contact/Jacobian state each step.
- Response validation result: diagnostic response artifacts generated; no full torque-level stabilizing contact-force controller is available through the current target-offset interface.
- Candidate result: six Step 5.16 candidate modes were materialized in `candidate_results.csv`; none were kept for full validation.
- Full validation result: not run because the small architecture gate did not produce a keepable candidate.
- Final decision: `TORQUE_LEVEL_CONTROL_REQUIRED`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.17 — Torque-Level / Generalized-Force WBC Prototype

- Output dir: `outputs/phase_b9_step5_17_torque_level_wbc_prototype/`
- Baseline used: reset-fixed/post-reset Step 5 only.
  - all-height mean survival: `3.8167 s`
  - all-height fall rate: `0.8333`
  - all-height pitch RMS: `1.1938 deg`
  - all-height roll RMS: `21.1630 deg`
  - h=0.60 survival: `0.52 s`
  - h=0.60 fall rate: `1.0`
- Torque-level feasibility: MJCF actuators are torque-like `<motor>` actuators, but the deployed controller interface still interprets actions as leg position targets and wheel velocity targets before PID writes motor `ctrl`.
- Diagnostic-force interface: `qfrc_applied` and `xfrc_applied` are available in MuJoCo/MJX data. Step 5.17 uses diagnostic-only `qfrc_applied` helpers and does not change PPO action semantics.
- Torque WBC design: hybrid existing PID posture/wheel path plus bounded diagnostic generalized-force residual on allowed actuated joint dofs. Root dofs and hip-yaw dofs are not written.
- Response validation result: static ±2 deg roll diagnostics show torque authority can generate sign-changing roll correction for roll-enabled candidates.
- Candidate result: no survival evaluation was run because the prototype relies on diagnostic `qfrc_applied` injection rather than a deployable low-level torque-control env path.
- Full validation result: not run; no deployable candidate passed the small survival gate.
- Final decision: `LOW_LEVEL_CONTROL_REDESIGN_REQUIRED`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.18 — Deployable Motor-Torque Control Interface

- Output dir: `outputs/phase_b9_step5_18_deployable_motor_torque_interface/`
- Baseline used: reset-fixed/post-reset Step 5 only.
  - all-height mean survival: `3.8167 s`
  - all-height fall rate: `0.8333`
  - all-height pitch RMS: `1.1938 deg`
  - all-height roll RMS: `21.1630 deg`
  - h=0.60 survival: `0.52 s`
  - h=0.60 fall rate: `1.0`
- Motor torque deployability: MJCF defines ten `<motor>` actuators with identity action-index mapping, `gear=1`, explicit `ctrlrange`, and explicit `forcerange`; actuator `ctrl` can be used as deployable simulation motor torque.
- Low-level modes added: `pid_position_velocity` default, opt-in `motor_torque`, and opt-in `hybrid_pid_plus_torque`.
- Default PID path unchanged: `True`; torque modes require explicit `low_level_control` config and `torque_control.enabled: true`.
- Implementation summary: `low_level_control.py` now supports direct normalized motor ctrl and bounded hybrid torque residual; `BalanceEnv` routes modes without changing action dimension or ordering.
- Response validation result: static deployable interface artifacts generated; survival response validation was not run in this patch.
- Candidate result: six candidate definitions materialized; none passed the survival gate because no h=0.60 survival rollout was executed.
- Full validation result: not run; current best controller remains unchanged.
- Final decision: `HYBRID_PID_TORQUE_REQUIRED`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.18b — Hybrid PID + Motor-Torque Rollout Validation

- Output dir: `outputs/phase_b9_step5_18b_hybrid_pid_torque_rollout_validation/`
- Baseline used: reset-fixed/post-reset Step 5 only.
  - all-height mean survival: `3.8167 s`
  - all-height fall rate: `0.8333`
  - h=0.60 survival: `0.52 s`
  - h=0.60 fall rate: `1.0`
- Activation validation: `hybrid_pid_plus_torque` path activated with `torque_control_enabled = True`, `low_level_mode_code = 2`, and nonzero bounded torque residual.
- Interface validation: deployable MJCF actuator `ctrl` path only; `qfrc_applied_abs_max = 0.0` throughout validation.
- Dynamic response validation: 5 candidates × ±2 deg roll perturbations generated actuator-ctrl torque responses; at least one response was nonzero/stabilizing by sign/activation criteria.
- h=0.60 survival rollout: 5 episodes per candidate, 60-step bounded horizon.
- Best h=0.60 candidate: `hybrid_roll_pitch_damping`.
  - mean survival: `0.32 s`
  - fall rate: `1.0`
  - pitch RMS: `0.0582 deg`
  - roll RMS: `25.3383 deg`
  - actuator saturation rate: `0.9375`
  - mean torque residual abs: `0.4803`
  - dominant fall reason: `tilt`
- Small gate passed: `False`; no candidate beat the h=0.60 reset-fixed baseline of `0.52 s`.
- Full validation result: not run because no h=0.60 candidate passed the small gate.
- Final decision: `MOTOR_TORQUE_CONTROL_NEEDS_GAIN_TUNING`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.18c — Motor-Torque Gain Scaling and Saturation Calibration

- Output dir: `outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/`
- Baseline used: reset-fixed/post-reset Step 5 only.
  - all-height mean survival: `3.8167 s`
  - all-height fall rate: `0.8333`
  - h=0.60 survival: `0.52 s`
  - h=0.60 fall rate: `1.0`
- Saturation root cause (Phase 1): `PID_CONTROLLER_SATURATION_NOT_TORQUE_RESIDUAL`
  - Step 5.18b: PID outputs 30 Nm and saturates at ctrlrange limits
  - Step 5.18b: Torque residuals remain under 1 Nm (max 0.96 Nm)
  - Magnitude ratio: PID is 30× larger than torque residual
- Torque scaling verification (Phase 2): No scaling bug found; issue is gain magnitude
- Response validation (Phase 3): 5 candidates tested with k_roll 10-40, max_ctrl_fraction 0.3-0.7
  - Physical torque range: 1.57-10.5 Nm
  - All candidates non-saturating at response level
- h=0.60 survival evaluation (Phase 4): 5 candidates tested, all beat baseline
  - Best candidate: `strong_k20` (k_roll=20.0, k_pitch=5.0, max_ctrl_fraction=0.5)
  - h=0.60 survival: `0.86 s` (baseline: 0.52 s, +65%)
  - h=0.60 fall rate: `0.80` (baseline: 1.0)
- Full validation (Phase 5): Top-2 candidates tested across heights 0.65-0.40
  - Best candidate: `strong_k20`
  - all-height mean survival: `0.86 s`
  - all-height fall rate: `0.80`
  - pitch RMS: `0.058 deg`
  - roll RMS: `15.9 deg`
  - action saturation rate: `0.80`
- Comparison vs baselines:
  - vs h=0.60 baseline: +0.34 s (+65%), beats baseline
  - vs all-height reset-fixed baseline: -2.96 s (-77%), does not beat baseline
- Final decision: `TORQUE_GAIN_CALIBRATION_IMPROVES_BUT_DOES_NOT_PASS_GATE`
- Interpretation: Torque gain calibration (k_roll 1.5→20.0, max_ctrl_fraction 0.15→0.5) produces meaningful physical torques (5.24 Nm) that improve h=0.60 survival by 65%. However, all-height performance remains 77% worse than Step 5 reset-fixed baseline.
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`
