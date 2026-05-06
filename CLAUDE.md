# CLAUDE.md

## Project overview

This repository is a MuJoCo MJX + JAX/PPO research codebase for a 10-actuated wheeled biped robot.
The robot has two legs and two wheels, with 5 actuated joints per leg:

- hip roll
- hip yaw
- hip pitch
- knee
- wheel

The repository is being pivoted from a **pure height-conditioned PPO balance policy** toward a **hybrid residual postural-control framework**:

```text
obs + task_cmd
    ↓
height-dependent LQR/IK nominal prior
    ↓
base_action

obs + task_cmd + base_action
    ↓
bounded PPO residual policy
    ↓
residual_action

final_action = base_action + residual_scale × residual_action
    ↓
smoothing / delay
    ↓
low-level PID
    ↓
robot
```

The new research thesis is:

> A height-dependent LQR/IK controller provides a structured nominal balance and posture prior, while a bounded PPO residual policy learns corrective actions for height-adaptive stabilization, commanded standing-squatting transitions, and push-disturbance recovery.

Keep the mental model of this repo as:

> a JAX/MJX-first wheeled-biped control research codebase, where residual RL must be implemented with explicit action semantics, controller metadata, train/eval parity, and quantitative evaluation.

This is not a generic robotics sandbox.

---

## Current repo status and truth

- The current repo is still mostly a **pure PPO balance prototype**.
- `BalanceEnv` is the implemented balance task with 42-dimensional observation.
- Existing `baseline_lqr.yaml` / LQR controller is currently an **evaluation baseline**, not yet a validated nominal prior for residual RL.
- `balance_robust`, `stand_up`, `wheeled_locomotion`, `walking`, `stair_climbing`, and `rough_terrain` may exist as configs/stubs, but they must not be claimed as completed unless trained and evaluated.
- Sim-to-real transfer has not been validated on hardware.
- Do not rewrite the project to PyTorch unless explicitly asked.
- Preserve MJX/JAX-first design.
- Preserve existing PPO/checkpoint/logging flow unless the task explicitly targets it.
- Prefer minimal, testable diffs over broad rewrites.

---

## New main method

The proposed method is **bounded residual PPO over a height-dependent LQR/IK prior**.

Canonical residual action semantics:

```text
base_action_abs ∈ [-1, 1]^10
residual_action ∈ [-1, 1]^10
residual_scale ∈ R^10

final_action_abs = clip(
    base_action_abs + residual_scale * residual_action,
    -1.0,
    1.0,
)
```

`final_action_abs` is then passed through:

```text
smoothing / delay → low-level PID → robot
```

Important:

- `base_action_abs` means the nominal normalized action target from LQR/IK.
- `residual_action` means the bounded PPO correction.
- `final_action_abs` means the normalized target after residual composition.
- Do not mix absolute action and pre-bias action semantics.
- Do not double-add `pid_action_bias`.
- Train/eval/visualize/inference must use the same action-composition function.

---

## Critical action and control invariants

- Action dimension is always 10 unless the robot model is intentionally changed.
- Joint/action order is:
  1. `l_hip_roll`
  2. `l_hip_yaw`
  3. `l_hip_pitch`
  4. `l_knee`
  5. `l_wheel`
  6. `r_hip_roll`
  7. `r_hip_yaw`
  8. `r_hip_pitch`
  9. `r_knee`
  10. `r_wheel`
- Leg joints use **position targets** through low-level PID.
- Wheel joints use **velocity targets** through low-level PID.
- Wheel actions must never be interpreted as position targets.
- Leg actions must never be interpreted as raw torque unless PID is explicitly disabled.
- `BalanceEnv` legacy pure PPO behavior should remain available as a baseline.
- `ResidualBalanceEnv` should be the main proposed method once implemented.
- Existing `pid_action_bias` is useful for pure PPO initialization, but residual composition must be explicit and bias-safe.

Recommended joint groups:

```python
LEG_POSITION_INDICES = [0, 1, 2, 3, 5, 6, 7, 8]
WHEEL_VELOCITY_INDICES = [4, 9]
HIP_ROLL_INDICES = [0, 5]
HIP_YAW_INDICES = [1, 6]
HIP_PITCH_KNEE_INDICES = [2, 3, 7, 8]
```

---

## Observation design

Current pure PPO `BalanceEnv` observation:

```text
obs_pure = obs_42
```

Proposed residual PPO observation:

```text
obs_residual = obs_42 + base_action_abs_10
obs_dim = 52
```

Residual observation should include:

- gravity in body frame
- body linear velocity, if used
- body angular velocity
- joint positions
- joint velocities
- previous final action, not previous residual action
- height command or broader task command
- current torso height
- yaw error
- `base_action_abs`

If body linear velocity is simulator-clean, mark it as a sim-only signal and discuss hardware estimation separately. Do not silently claim hardware-ready state estimation.

---

## Residual scale default

Use a vector residual scale, not one scalar.

Initial proposed residual scale:

```yaml
residual_scale:
  l_hip_roll: 0.10
  l_hip_yaw: 0.05
  l_hip_pitch: 0.15
  l_knee: 0.15
  l_wheel: 0.30
  r_hip_roll: 0.10
  r_hip_yaw: 0.05
  r_hip_pitch: 0.15
  r_knee: 0.15
  r_wheel: 0.30
```

Rationale:

- Wheels need larger residual authority for balancing and push recovery.
- Hip pitch/knee need moderate residual authority because IK already provides posture.
- Hip yaw should remain small to avoid excessive twisting.
- Hip roll can provide lateral balance correction.

Add residual-scale ablations when evaluating the paper.

---

## Naming rules

Do not call the prior “exact LQR” unless it is actually derived from validated per-height linearization and Riccati solves.

Use safer terms unless exact derivation exists:

- `height-dependent LQR/IK prior`
- `structured LQR/IK prior`
- `gain-scheduled LQR/IK prior`
- `model-based nominal prior`

Do not claim:

- hardware validation
- sim-to-real success
- first residual RL method
- first LQR/RL hybrid for wheeled robots
- completed stand-up recovery unless trained and evaluated
- completed locomotion/stair/rough-terrain tasks unless trained and evaluated
- safety for hardware

---

## Migration plan

Work phase-by-phase. Do not implement all phases at once.

### Phase A — Make action semantics explicit

Goal: create one canonical action pipeline without breaking existing pure PPO.

Tasks:

- Add controller/action types.
- Add shared residual composition function.
- Add tests for clipping, scaling, PID bias, wheel/leg semantics.
- Do not add `ResidualBalanceEnv` yet.
- Do not rewrite the paper yet.
- Do not train yet.

Suggested files:

- `wheeled_biped/controllers/action_codec.py` or `wheeled_biped/controllers/action_composer.py`
- `tests/test_action_codec.py` or `tests/test_action_composer.py`

Required functions/types:

- `ActionBreakdown`
- `ControllerMetadata`
- `ActionMode` or `PolicyType`
- `compose_residual_action(base_action_abs, residual_action, residual_scale, clip=True)`
- `validate_action_shape(x)`
- `validate_residual_scale(scale)`
- `clip_normalized_action(x)`
- `action_group_stats(x)`

Phase A pass criteria:

- zero residual returns base action
- zero base action plus residual returns scaled residual
- clipping works
- residual scale shape is validated
- output shape is 10
- no double-addition of `pid_action_bias`
- existing `BalanceEnv` behavior is unchanged

### Phase B — Validate LQR/IK prior

Goal: implement/validate a height-dependent nominal controller before training residual PPO.

Tasks:

- Implement `gain_scheduled_lqr.py` or equivalent.
- Implement/clean `lqr_ik_prior.py`.
- Add fixed-height tests.
- Run LQR-only evaluation before RL.

Validate:

- pitch sign convention
- wheel command sign
- yaw differential sign
- hip roll correction sign
- left/right joint mirroring
- IK monotonicity with height
- targets within joint limits
- LQR-only nominal rollout

### Phase C — Add residual environment

Goal: create the proposed training environment.

Tasks:

- Add `ResidualBalanceEnv`.
- Observation dim must be 52.
- Append `base_action_abs` to observation.
- Policy action is residual only.
- Info logs all action components.
- Add no-NaN rollout smoke test.

Required `info` keys:

- `base_action_abs`
- `residual_action`
- `residual_scaled`
- `final_action_abs`
- `control_action`
- `residual_norm`
- `residual_saturation_rate`

### Phase D — Add residual training/config

Tasks:

- Add `configs/training/balance_residual.yaml`.
- Add `configs/training/balance_residual_robust.yaml`.
- Update `scripts/train.py` stage mapping.
- Update checkpoint metadata.
- Train short seed 42 smoke run.
- Validate checkpoint.

Required checkpoint metadata:

- `policy_type: residual_ppo`
- `action_mode: residual`
- `obs_dim: 52`
- `action_dim: 10`
- `residual_scale`
- `base_controller_config`
- `base_action_in_obs: true`
- `pid_config_hash` if practical
- `smoothing_alpha`
- `action_delay_steps`

### Phase E — Paper evaluation suite

Tasks:

- Update `eval_balance.py`.
- Add residual metrics.
- Add `analyze_residual.py`.
- Add table exporters if needed.
- Ensure LQR/IK-only, pure PPO, and residual PPO are evaluated consistently.

### Phase F — Full experiments

Tasks:

- Train residual PPO over 3 seeds.
- Run LQR/IK-only comparison.
- Run pure PPO reference.
- Run push/height/robustness/ablation evaluations.

### Phase G — Rewrite paper

Only after Phase F produces data:

- rewrite abstract/contributions/method/results
- delete or quarantine unsupported claims
- fill tables/figures
- explicitly label stand-up/locomotion as future work if not evaluated

---

## Proposed repository architecture

Add or migrate toward:

```text
wheeled_biped/controllers/
├── action_codec.py              # canonical action composition and metadata
├── base_controller.py           # controller interface
├── lqr_ik_prior.py              # height IK and nominal action generation
├── gain_scheduled_lqr.py        # height-dependent LQR gains/utilities
├── hybrid_residual_controller.py# base + residual composition wrapper
└── controller_types.py          # shared enums/constants if not in action_codec.py

wheeled_biped/envs/
├── balance_env.py               # legacy pure PPO baseline
├── residual_balance_env.py      # proposed method
└── ...

wheeled_biped/rewards/
├── reward_functions.py
└── residual_reward_functions.py # optional if residual terms are separated

configs/controllers/
└── gain_scheduled_lqr.yaml

configs/training/
├── balance.yaml                 # pure PPO baseline
├── balance_residual.yaml        # proposed method
├── balance_residual_robust.yaml # push/disturbance fine-tune
└── ...

scripts/
├── train.py
├── eval_balance.py
├── validate_checkpoint.py
├── visualize.py
├── analyze_residual.py          # residual diagnostics
└── export_results.py
```

Do not create extra envs such as `height_transition_residual_env.py` or `standup_residual_env.py` until the base residual balance env works and the task truly needs separate reset/reward/termination logic.

---

## Reward design guidance

The current reward may contain redundant terms. Audit before adding more.

Potential overlaps:

- `body_level` vs `orientation`
- `heading` vs `yaw_rate`
- `legs_forward` / `legs_vertical` / `natural_pose` / `symmetry`
- `no_motion` / `position_drift` / `wheel_velocity`
- `action_rate` vs `final_action_rate` vs `residual_rate`

Use grouped rewards:

### Core task rewards

- `alive`
- `body_level` or pitch/roll stabilization
- `height_tracking`
- `position_drift`
- `heading`

### Posture regularization

- `symmetry`
- `ik_consistency` or reduced `natural_pose`
- avoid duplicate leg-orientation terms unless justified

### Effort and smoothness

- `joint_torque`
- `joint_velocity`
- `final_action_rate`
- weak or zero `wheel_speed` penalty during push recovery

### Residual-specific regularization

- `residual_magnitude`
- `residual_rate`
- `residual_saturation`

Do not over-penalize wheel velocity or action rate in push recovery. Recovery requires wheel motion.

---

## Evaluation requirements

The paper must compare:

1. `Gain-scheduled LQR/IK only`
2. `Proposed LQR/IK + bounded residual PPO`
3. `Pure PPO from scratch` as secondary/motivation baseline

Do not make pure PPO the main opponent unless it is tuned and trained fairly.

Core scenarios:

- nominal standing
- fixed-height balance at 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40 m
- random full-range height commands
- commanded height transitions
- friction low/high and friction sweep
- mass/damping perturbations
- sensor noise and action delay
- push recovery
- push sweep from 20 N to 200 N
- stand-up recovery only if implemented and evaluated

Core metrics:

- `survival_rate`
- `fall_rate`
- `episode_survival_time`
- `height_RMSE`
- `pitch_RMS_deg`
- `roll_RMS_deg`
- `yaw_error_RMS_deg`
- `xy_drift_max_m`
- `wheel_speed_RMS_rad_s`
- `torque_RMS_Nm`
- `energy_proxy_sum_abs_tau_qdot`
- `recovery_time_s`
- `max_recoverable_push_N`
- `peak_pitch_after_push_deg`
- `base_action_RMS`
- `residual_action_RMS`
- `final_action_RMS`
- `residual_to_base_ratio`
- `residual_rate`
- `residual_saturation_rate`
- `residual_RMS_by_joint_group`

---

## Required Results section for paper

The Results section should be quantitative, not only video/demo based.

Recommended structure:

```latex
\section{Experimental Results}

\subsection{Nominal Prior Verification}
% LQR/IK-only fixed-height and random-height results

\subsection{Training Convergence}
% residual PPO 3 seeds, reward, success, curriculum, residual RMS

\subsection{Random-Height Balance}
% LQR/IK vs residual PPO vs pure PPO reference

\subsection{Fixed-Height Balance Sweep}
% h = 0.70 ... 0.40 m

\subsection{Commanded Height Transitions}
% standing-squatting transitions

\subsection{Push-Disturbance Recovery}
% push recovery and push sweep

\subsection{Robustness to Model Uncertainty}
% friction/mass/damping/noise/delay

\subsection{Residual Action Analysis}
% base/residual/final action diagnostics

\subsection{Ablation Study}
% LQR-only, residual_scale, base_action obs, residual penalty

\subsection{Pure PPO Baseline Discussion}
% reference or appendix

\subsection{Stand-Up Recovery}
% only if implemented and evaluated
```

Minimum submit-ready result set:

1. Residual PPO over 3 seeds.
2. LQR/IK-only evaluation.
3. Pure PPO reference baseline or honest preliminary explanation.
4. Random-height balance table.
5. Fixed-height sweep table.
6. Height-transition figure and metrics.
7. Push-recovery table and time-series figure.
8. Residual action analysis.
9. LQR/IK-only vs residual ablation.
10. Residual-scale or base-action-observation ablation.

---

## Required paper tables

- Table I — Robot and Control Setup
- Table II — Controller Comparison
- Table III — Training Configuration
- Table IV — Nominal LQR/IK Prior Verification
- Table V — Main Random-Height Balance Results
- Table VI — Fixed-Height Balance Sweep
- Table VII — Commanded Height-Transition Performance
- Table VIII — Push-Disturbance Recovery
- Table IX — Robustness to Model Uncertainty
- Table X — Residual Action Analysis
- Table XI — Ablation Study
- Table XII — Pure PPO Baseline Discussion, optional or appendix

---

## Required paper figures

- Figure 1 — Robot morphology and joint layout
- Figure 2 — Hybrid residual control architecture
- Figure 3 — Nominal LQR/IK prior behavior
- Figure 4 — Training curves over 3 seeds
- Figure 5 — Random-height tracking time series
- Figure 6 — Commanded height transition time series
- Figure 7 — Push recovery time series
- Figure 8 — Push magnitude sweep
- Figure 9 — Robustness sweep
- Figure 10 — Residual action distribution
- Figure 11 — Qualitative snapshot sequence
- Figure 12 — Stand-up recovery only if implemented

Every claimed behavior must have at least one quantitative metric. Qualitative figures cannot replace tables.

---

## Files Claude should inspect first

For most tasks:

- `README.md`
- `CLAUDE.md`
- `paper/main.tex`
- `configs/training/balance.yaml`
- `configs/training/balance_robust.yaml`
- `configs/training/stand_up.yaml`
- `configs/baseline_lqr.yaml`
- `configs/curriculum.yaml`
- `wheeled_biped/envs/base_env.py`
- `wheeled_biped/envs/balance_env.py`
- `wheeled_biped/sim/low_level_control.py`
- `wheeled_biped/controllers/lqr_balance.py`
- `wheeled_biped/training/ppo.py`
- `wheeled_biped/training/curriculum.py`
- `wheeled_biped/rewards/reward_functions.py`
- `wheeled_biped/inference/unified_controller.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/eval_balance.py`
- `scripts/validate_checkpoint.py`
- `scripts/visualize.py`
- `scripts/export_results.py`
- `tests/`

Inspect the exact files touched by the task before editing.

---

## JAX / MJX coding rules

- Prefer pure functions.
- Preserve JIT-friendly and scan-friendly structure.
- Avoid Python-side loops in hot rollout/update paths when existing code already uses JAX patterns.
- Split RNG keys explicitly.
- Do not mutate JAX arrays in place.
- Keep base/action composition functions usable from env, eval, visualize, and inference.
- Avoid NumPy/SciPy inside JIT hot paths.
- If a controller needs offline gain computation, separate offline generation from runtime JAX execution.

---

## Working style

- Work one task/phase at a time.
- Show target files, intended change, risk points, and verification plan before editing.
- Prefer minimal diffs over opportunistic cleanup.
- Do not change public interfaces unless necessary.
- When proposing architecture changes, distinguish minimal patch from larger refactor.
- When uncertain, inspect code rather than assume.
- If task changes training logic, update evaluation and tests when appropriate.
- If task changes action semantics, update train/eval/validate/visualize/inference together or clearly document what remains pending.
- Do not train long runs until smoke tests and targeted tests pass.

---

## Subsystem guidance

### Environments

When editing env code:

- check obs size
- check action semantics
- check termination/reset behavior
- check reward coupling
- check whether task-specific observation extensions affect policy/controller logic
- ensure train/eval action path parity
- for residual env, log base/residual/final action components

### PPO / training

When editing PPO or rollout code:

- watch for NaNs
- preserve obs normalization semantics
- preserve checkpoint compatibility where possible
- verify rollout and minibatch shapes
- add metadata for residual policy type/action mode
- prefer smoke tests plus targeted invariants

### Curriculum

When editing curriculum:

- determine whether progression is budget-driven or performance-gated
- make progression logic explicit
- add/update tests if promotion/demotion logic changes
- for residual work, do not depend on stand-up before it is implemented and evaluated

### Evaluation

When editing evaluation:

- distinguish nominal evaluation from robustness benchmarking
- do not rely only on mean reward
- support LQR/IK-only, pure PPO, and residual PPO explicitly
- log residual-specific metrics for residual policies
- keep existing entrypoints working if possible

### Unified controller

When editing unified controller:

- treat observation semantics carefully across skills
- prefer explicit adapters over silent pad/cut logic
- support residual-aware inference before using residual checkpoints
- avoid switching to unsupported skills such as stand-up/locomotion unless evaluated

### Logging and metadata

When editing logging/checkpoints:

- preserve TensorBoard/WandB compatibility
- add reproducibility metadata where practical
- log action-mode metadata
- log residual metrics when residual policy is used
- do not silently load a residual checkpoint as pure PPO or vice versa

Required residual checkpoint metadata:

```yaml
policy_type: residual_ppo
action_mode: residual
obs_dim: 52
action_dim: 10
base_action_in_obs: true
residual_scale: [...]
base_controller_config: configs/controllers/gain_scheduled_lqr.yaml
smoothing_alpha: <value>
action_delay_steps: <value>
```

---

## Tests to add/update

Add or update tests for:

- base action shape and bounds
- residual composition correctness
- clipping
- residual scale shape
- zero residual returns base action
- PID bias semantics
- no double-addition of `pid_action_bias`
- obs dimension = 52 for residual env
- base action included in obs
- previous action = previous final action
- wheel actions interpreted as velocity targets
- leg actions interpreted as position targets
- residual reward terms
- no NaN rollout
- LQR sign convention
- height IK monotonicity and bounds
- final action clipping
- checkpoint metadata for residual policies
- evaluation action path matches training action path
- visualization action path matches training action path
- inference action path matches training action path

---

## Commands Claude should know

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### Fast checks

```bash
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v
```

### Slow smoke train

```bash
pytest tests/test_smoke_train.py -v -m slow
```

### Residual Phase A checks, after action codec is added

```bash
pytest tests/test_action_codec.py -v
```

or, if named differently:

```bash
pytest tests/test_action_composer.py -v
```

### Legacy LQR baseline evaluation

```bash
python scripts/eval_balance.py --controller baseline_lqr \
  --scenarios nominal --scenarios push_recovery \
  --num-episodes 20 --output-dir outputs/balance/lqr
```

### Future LQR/IK prior evaluation

```bash
python scripts/eval_balance.py --controller lqr_ik \
  --scenarios nominal --scenarios fixed_height_sweep \
  --num-episodes 20 --output-dir outputs/residual/lqr_ik_prior_eval
```

### Future residual smoke training

```bash
python scripts/train.py single \
  --stage balance_residual \
  --steps 100000 \
  --seed 42 \
  --num-envs 1024 \
  --output-dir outputs_residual_smoke
```

Do not run long training until Phase A, Phase B, and Phase C tests pass.
