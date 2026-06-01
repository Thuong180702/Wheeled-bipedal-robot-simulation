# Step E Root Cause Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run stop-gated diagnostic-only Step E root-cause tests for H1-H4 without changing production controller behavior.

**Architecture:** Add one standalone diagnostic script that reuses existing controller classes and MuJoCo model helpers, writes required artifacts under `outputs/step_e_root_cause_diagnostics/`, and executes gates in the user-specified order. Add focused unit tests for diagnostic pure helpers so the script has testable contracts without modifying the main simulator.

**Tech Stack:** Python, MuJoCo, JAX/JAX NumPy, pytest, CSV/JSON/Markdown outputs.

---

## File Structure

- Create `scripts/diagnose_step_e_root_causes.py`: standalone diagnostic runner, pure metric helpers, stop-gated execution, artifact validation, report generation.
- Create `tests/test_step_e_root_cause_diagnostics.py`: unit tests for pure helpers: axis sign, velocity call-site difference, stop-gate decisions, missing artifact validation, posture stable-roll metric.
- Do not modify `scripts/simulate_hierarchical_controller.py` unless the standalone script cannot run.

## Tasks

### Task 1: Add diagnostic helper tests

**Files:**
- Create: `tests/test_step_e_root_cause_diagnostics.py`

- [ ] Write tests importing helper functions from `scripts.diagnose_step_e_root_causes`.
- [ ] Cover: flipped axis is exact negative of current, actual velocity passed is raw `com_vy`, velocity difference is `raw_com_vy - projected`, 5000-step gate only opens when both 1000-step runs survive, stable-roll hip-roll error percentage counts only rows with `abs(roll_y_rad) < 0.05`, missing artifact validation returns absent paths.
- [ ] Run `pytest tests/test_step_e_root_cause_diagnostics.py -v`; expected RED: module missing.

### Task 2: Implement standalone diagnostic script helpers

**Files:**
- Create: `scripts/diagnose_step_e_root_causes.py`

- [ ] Implement constants for required artifacts and output directory.
- [ ] Implement pure helpers used by tests: `current_sagittal_axis`, `flipped_sagittal_axis`, `velocity_frame_sample`, `should_run_5000_gate`, `percent_abs_error_gt_threshold_while_roll_stable`, `validate_required_artifacts`.
- [ ] Run helper tests; expected GREEN.

### Task 3: Implement Gate 1 wheel torque sign audit

**Files:**
- Modify: `scripts/diagnose_step_e_root_causes.py`

- [ ] Load `assets/robot/wheeled_biped_real.xml`, reset standing keyframe, calibrate root Z using same target contact distance as simulator.
- [ ] For pulse torques `[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]`, run 0.1 s first, then extend to 0.2 s only if both wheel-floor contact remains valid and `abs(pitch_x), abs(roll_y) < 0.35` rad.
- [ ] Record requested support position, pitch/roll, wheel velocities, contact validity, physical validity, and slopes to `wheel_torque_sign_audit.csv` and `.json`.

### Task 4: Implement Gate 2 axis ablation and Gate 3 velocity-frame audit

**Files:**
- Modify: `scripts/diagnose_step_e_root_causes.py`

- [ ] Implement `run_balance_core_diagnostic(axis_sign, steps, output_csv)` using identical initial state and gains for current/flipped axis; only sagittal axis sign differs.
- [ ] Keep WBC off in applied torques by using balance-core four-source composition only; legacy torque telemetry must remain zero.
- [ ] At the sagittal controller call site, set `actual_value_passed_to_controller_as_sagittal_velocity_m_s = raw_com_vy` and pass that exact variable.
- [ ] Log projected velocity, actual passed value, raw `com_vy`, and difference every control step.
- [ ] Write `axis_ablation_current.csv`, `axis_ablation_flipped.csv`, `axis_ablation_summary.json`, `velocity_frame_audit.csv`, `velocity_frame_audit.json`.

### Task 5: Implement Gate 4 posture audits and Gate 5 long-run gate

**Files:**
- Modify: `scripts/diagnose_step_e_root_causes.py`

- [ ] Generate `hip_roll_posture_audit.csv/.json` and `hip_yaw_posture_audit.csv/.json` from the 1000-step current-axis run.
- [ ] Include percentage of time `abs(hip_roll_error) > 0.10` while `abs(roll_y_rad) < 0.05`.
- [ ] If both 1000-step ablations survive, run 5000-step current/flipped and incorporate summary metrics; otherwise write `NOT RUN: one or both 1000-step axis ablation runs terminated` into report evidence.

### Task 6: Generate final report and validate artifacts

**Files:**
- Modify: `scripts/diagnose_step_e_root_causes.py`

- [ ] Write `step_e_root_cause_summary.json` with the exact requested schema.
- [ ] Write `step_e_root_cause_report.md` with executive summary, environment, H1-H4 sections, final decision matrix, exact next-step recommendation, safety constraints, and Missing artifacts section.
- [ ] Validate all required artifacts exist at the end; report missing artifacts if any.

### Task 7: Run diagnostics and tests

**Commands:**
- `pytest tests/test_step_e_root_cause_diagnostics.py -v`
- `python scripts/diagnose_step_e_root_causes.py`

**Expected:** All helper tests pass; all required diagnostic artifacts are created or explicitly marked `NOT RUN` where gated/environment-limited.
