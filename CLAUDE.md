# CLAUDE.md

## Project overview

This repo trains and evaluates a wheeled bipedal robot in MuJoCo MJX using JAX/PPO.
Current architecture already includes:
- vectorized envs with auto-reset
- PPO training with GAE, obs normalization, checkpoint/resume
- curriculum training
- telemetry/logger/evaluation utilities
- a heuristic unified controller for multi-skill switching
- balance training with low-level PID and push disturbance support

Main code lives in:
- `wheeled_biped/envs/`
- `wheeled_biped/training/`
- `wheeled_biped/rewards/`
- `wheeled_biped/inference/`
- `wheeled_biped/utils/`
- `scripts/`
- `tests/`

Keep the mental model of this repo as:
a JAX/MJX-first RL training codebase for a wheeled biped robot, not a generic robotics sandbox.

## Important repo truths

- Keep MJX/JAX-first design. Do not rewrite the training stack to PyTorch unless explicitly asked.
- Preserve existing PPO/checkpoint/logging flow unless a task explicitly targets it.
- Prefer minimal diffs over broad rewrites.
- Before editing, inspect the exact files touched by the task and explain the plan briefly.
- After editing, run the smallest relevant verification commands first, then broader checks if needed.
- If a task changes training logic, also update evaluation and tests when appropriate.

## Current known priorities

1. Curriculum manager should promote/demote based on actual performance, not just fixed stage progression.
2. Evaluation should become a benchmark suite, not just mean reward / episode length.
3. Domain randomization / perturbation / low-level control should be reusable across tasks, not concentrated in balance only.
4. Unified controller should reduce heuristic brittleness and avoid generic obs pad/cut where semantics differ.
5. Add tests for PPO trainer, curriculum logic, checkpoint invariants, and unified controller.
6. Improve experiment logging and sweep friendliness without unnecessary framework churn.

## Files Claude should inspect first

- `README.md`
- `agent.md`
- `wheeled_biped/training/ppo.py`
- `wheeled_biped/training/curriculum.py`
- `wheeled_biped/inference/unified_controller.py`
- `wheeled_biped/envs/base_env.py`
- `wheeled_biped/envs/`
- `wheeled_biped/rewards/reward_functions.py`
- `wheeled_biped/utils/config.py`
- `wheeled_biped/utils/logger.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/visualize.py`
- `tests/`

## Repo invariants

- Action dimension is 10 unless the robot model is intentionally changed.
- Base observation size is 39; task envs may extend this.
- Network input must match `env.obs_size`.
- Preserve current actuator/control semantics unless explicitly changing them.
- Default robot model path is `assets/robot/wheeled_biped_real.xml`.
- Do not casually change checkpoint format, CLI behavior, or config schema.

## JAX / MJX coding rules

- Prefer pure functions.
- Preserve JIT-friendly and scan-friendly structure.
- Avoid Python-side loops in hot rollout/update paths when existing code already uses JAX patterns.
- Split RNG keys explicitly.
- Do not mutate JAX arrays in place.

## Working style

- Work one task at a time.
- Show: target files, intended change, risk points, and verification plan before editing.
- Avoid changing public interfaces unless necessary.
- When proposing architecture changes, distinguish "minimal patch" vs "larger refactor".
- When uncertain, inspect code rather than assume.
- Prefer minimal diffs over opportunistic cleanup.

## Subsystem guidance

### Environments
When editing env code:
- check obs size
- check action semantics
- check termination/reset behavior
- check reward coupling
- check whether task-specific observation extensions affect policy/controller logic

### PPO / training
When editing PPO or rollout code:
- watch for NaNs
- preserve obs normalization semantics
- preserve checkpoint compatibility where possible
- verify rollout and minibatch shapes
- prefer smoke tests plus targeted invariants

### Curriculum
When editing curriculum:
- determine whether stage progression is budget-driven or performance-gated
- make progression logic explicit
- add/update tests if promotion/demotion logic changes

### Evaluation
When editing evaluation:
- distinguish nominal evaluation from robustness benchmarking
- do not rely only on mean reward if the task asks for stronger validation
- keep existing entrypoints working if possible

### Unified controller
When editing unified controller:
- treat observation semantics carefully across skills
- prefer explicit adapters over silent pad/cut logic
- prefer more stable switching logic over brittle thresholds

### Logging
When editing logging:
- preserve TensorBoard/WandB compatibility
- add useful reproducibility metadata where practical
- prefer extending current logger over adding a new framework

## Commands Claude should know

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"