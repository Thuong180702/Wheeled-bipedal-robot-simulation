# Phase B.9 Audit Gate Report (Steps 2/3/4/5)

## Scope
- Audit Step 2/3/4/5 outputs and cross-step consistency before Step 6.
- Fail-fast rule: any inconsistency blocks Step 6.
- Geometry asymmetry < 0.05 mm ignored.

## Per-step result
- Step 2 pass: True
- Step 3 pass: True
- Step 4 pass: True
- Step 5 pass: False

## Cross-step consistency
- Height-set consistent: True
- PID path consistent: True
- Semantic consistent: True
- Units consistent: True
- Stale output risk mitigated: True

## Key finding
- Step 5.6 reports `step6_ready = False`.
- Targeted fix reduced roll RMS slightly, but survival/fall-rate worsened.

## Decision
- Decision: **BLOCK_STEP6**
- Blocking issues: step5_6_step6_ready_false
- Required action: Fix blocking issues first; do not proceed Step 6.

## Artifacts
- outputs/phase_b9_audit_gate/step2_audit_summary.csv
- outputs/phase_b9_audit_gate/step3_audit_summary.csv
- outputs/phase_b9_audit_gate/step4_audit_summary.csv
- outputs/phase_b9_audit_gate/step5_audit_summary.csv
- outputs/phase_b9_audit_gate/cross_step_consistency.json
- outputs/phase_b9_audit_gate/audit_gate_decision.json

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
- Baseline used: reset-fixed/post-reset Step 5 only; stale pre-Step-5.13 metrics are not used.
- Mainline infrastructure status: Step 5.14 `lateral_balance` and Step 5.15 `vmc_whole_body` are first-class controller code and remain disabled by default unless explicitly enabled in candidate experiments.
- Interface audit result: true torque-level WBC is not supported by the current action path; the feasible implementation is Jacobian-informed target-offset VMC.
- Jacobian mapping result: diagnostic MuJoCo Jacobian mapping is available, but the deployed controller does not receive per-step `mj_data`/contact Jacobian state.
- Response validation result: diagnostic artifacts generated; architecture gate did not identify a keepable target-offset WBC candidate.
- Candidate/full validation: six candidates materialized, zero kept, full validation not run.
- Final decision: `TORQUE_LEVEL_CONTROL_REQUIRED`
- Gate decision: `BLOCK_STEP6`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.17 — Torque-Level / Generalized-Force WBC Prototype

- Output dir: `outputs/phase_b9_step5_17_torque_level_wbc_prototype/`
- Torque-level feasibility: MJCF `<motor>` actuators provide torque-like motor control, but the current deployed action path is still position-PID for legs and velocity-PID for wheels.
- Diagnostic path used: `qfrc_applied` generalized-force injection. `xfrc_applied` is also accessible but was not used as the selected prototype path.
- Diagnostic-only status: `true`; this is not hardware-ready and does not change residual PPO action semantics.
- Response validation result: static ±2 deg roll diagnostics generated sign-changing roll correction for roll-enabled candidates.
- Candidate/full validation: five candidates materialized, zero deployable candidates kept, survival/full validation not run.
- Final decision: `LOW_LEVEL_CONTROL_REDESIGN_REQUIRED`
- Gate decision: `BLOCK_STEP6`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.18 — Deployable Motor-Torque Control Interface

- Output dir: `outputs/phase_b9_step5_18_deployable_motor_torque_interface/`
- Motor torque deployability: MJCF `<motor>` actuators are deployable through actuator `ctrl`; action index maps identically to action index for all ten motors.
- Low-level modes added: `pid_position_velocity`, `motor_torque`, and `hybrid_pid_plus_torque`.
- Default PID path unchanged: `True`; the torque path is opt-in only and disabled unless candidate config sets `low_level_control.mode` plus `torque_control.enabled: true`.
- Response validation result: static interface artifacts generated; deployable survival response validation not run.
- Candidate/full validation: six candidates materialized, zero kept, full validation not run.
- Final decision: `HYBRID_PID_TORQUE_REQUIRED`
- Gate decision: `BLOCK_STEP6`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.18b — Hybrid PID + Motor-Torque Rollout Validation

- Output dir: `outputs/phase_b9_step5_18b_hybrid_pid_torque_rollout_validation/`
- Torque path activation: `hybrid_pid_plus_torque` activated with `torque_control_enabled = True` and nonzero bounded torque residual.
- Deployable path check: actuator `ctrl` path only; `qfrc_applied_abs_max = 0.0`.
- Dynamic response validation: 5 candidates × ±2 deg roll perturbations generated nonzero torque responses through actuator `ctrl`.
- h=0.60 candidate rollout: 5 episodes per candidate; best candidate `hybrid_roll_pitch_damping` reached `0.32 s` mean survival, `1.0` fall rate, `25.3383 deg` roll RMS, and `0.9375` actuator saturation rate.
- Small gate passed: `False`; full validation not run.
- Final decision: `MOTOR_TORQUE_CONTROL_NEEDS_GAIN_TUNING`
- Gate decision: `BLOCK_STEP6`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

## Phase B.9 Step 5.18c — Motor-Torque Gain Scaling and Saturation Calibration

- Output dir: `outputs/phase_b9_step5_18c_torque_gain_saturation_calibration/`
- Saturation root cause: `PID_CONTROLLER_SATURATION_NOT_TORQUE_RESIDUAL` (PID outputs 30 Nm, torque residuals <1 Nm)
- Torque scaling verification: No bug found; issue is gain magnitude
- Response validation: 5 candidates (k_roll 10-40) produced 1.57-10.5 Nm physical torques, all non-saturating
- h=0.60 evaluation: Best candidate `strong_k20` reached 0.86s survival (+65% vs baseline 0.52s), fall rate 0.80
- Full validation: `strong_k20` all-height mean survival 0.86s, fall rate 0.80, roll RMS 15.9 deg
- Comparison: beats h=0.60 baseline (+65%) but does not beat all-height reset-fixed baseline 3.8167s (-77%)
- Final decision: `TORQUE_GAIN_CALIBRATION_IMPROVES_BUT_DOES_NOT_PASS_GATE`
- Gate decision: `BLOCK_STEP6`
- Step 5 passed: `False`
- Step 6 status: `BLOCKED`
- Current best controller: `outputs/phase_b9_lqr_gain_strengthening/best_lqr_config.yaml`

