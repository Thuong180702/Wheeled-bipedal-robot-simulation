# wheeled_biped/training/CLAUDE.md

This file supplements the root `CLAUDE.md`.
Follow the root file first. Use this file for training-specific rules.

## Purpose of this folder

This folder contains the RL training stack, including PPO, curriculum logic, rollout/update behavior, checkpointing, and training-time evaluation/logging integration.

This area is high-risk because small changes can silently degrade learning quality.

## Local priorities

Highest-value changes in this folder usually are:

1. curriculum progression based on measured performance
2. stronger evaluation / benchmark flow
3. checkpoint and resume correctness
4. training stability and shape correctness
5. better experiment logging and reproducibility

## Local invariants

- Keep the repo JAX/MJX-first.
- Do not rewrite training code into PyTorch unless explicitly asked.
- Preserve the existing PPO/checkpoint/logging flow unless the task explicitly targets it.
- Preserve rollout and minibatch shape assumptions.
- Preserve observation normalization semantics unless the task explicitly changes them.
- Avoid casual checkpoint format changes.
- If a training logic change affects evaluation, tests must be updated too.

## Editing rules for training code

Before editing:

1. identify the exact training entrypoint and target module
2. inspect config usage
3. inspect checkpoint save/load flow
4. inspect logger usage
5. inspect existing tests
6. identify whether the task is:
   - bugfix
   - stability fix
   - instrumentation change
   - behavior change
   - architectural refactor

When editing:

- prefer the smallest correct patch
- avoid mixing unrelated cleanup into a behavior-sensitive patch
- preserve existing CLI behavior where possible
- be explicit about shape assumptions, RNG handling, and normalization flow
- if you add new metrics, wire them cleanly into logging rather than printing ad hoc

After editing:

- run targeted tests first
- run a small smoke check if practical
- report what remains unverified
- be explicit if full training correctness was not empirically validated

## PPO rules

When touching PPO or rollout/update logic:

- watch for NaNs
- inspect rollout shapes carefully
- inspect minibatch splitting carefully
- inspect normalization updates carefully
- inspect terminal handling carefully
- inspect checkpoint save/load assumptions carefully
- preserve JIT-friendly structure where already present

Do not:

- change multiple stability-sensitive pieces at once without a strong reason
- silently alter loss scaling or advantage handling
- silently alter normalization behavior
- claim stability improvements without verification

## Curriculum rules

Treat curriculum changes as behavior changes, not simple refactors.

When editing curriculum:

- determine whether progression is budget-driven or performance-gated
- make promotion / hold / demotion conditions explicit
- if thresholds or windows exist, ensure they are actually used in the main control flow
- add tests for promote / hold / demote behavior
- do not hide curriculum state transitions inside loosely justified heuristics

Prefer:

- explicit metrics
- explicit thresholds
- explicit stage transition rules
- targeted tests

## Evaluation rules

Training and evaluation are coupled.

When editing evaluation code in or around this folder:

- distinguish nominal evaluation from robustness benchmarking
- prefer adding named eval modes instead of breaking the current entrypoint
- keep outputs structured enough for logging/comparison
- do not rely only on mean reward if the task asks for stronger validation

Useful evaluation targets include:

- success / failure rate
- fall rate
- time-to-failure
- command tracking error
- perturbation recovery behavior
- task completion metrics where applicable

## Checkpoint / resume rules

Checkpoint compatibility matters.

When touching save/load/resume:

- keep format stable unless a migration is explicitly intended
- document any incompatibility clearly
- verify that resume state is complete enough for the intended workflow
- think about optimizer state, normalization state, curriculum state, and global step assumptions

## Logger / experiment rules

This repo already has logging infrastructure. Prefer extending it over replacing it.

When adding metrics:

- make names clear and stable
- prefer structured logging over ad hoc console output
- include reproducibility metadata where practical
- keep sweep-friendliness in mind

Do not introduce a heavy framework migration unless explicitly requested.

## Files to inspect first for training tasks

- `wheeled_biped/training/ppo.py`
- `wheeled_biped/training/curriculum.py`
- other training modules in this folder
- `wheeled_biped/utils/logger.py`
- `wheeled_biped/utils/config.py`
- `scripts/train.py`
- `scripts/evaluate.py`
- relevant tests in `tests/`

## Good task framing for this folder

A good training task request should specify:

- exact training module(s)
- whether behavior should change or only instrumentation should change
- whether checkpoint compatibility must be preserved
- whether evaluation/logging/tests should be updated too
- what verification is expected

## What not to do here

- do not perform a broad training-stack rewrite for a focused fix
- do not change PPO, curriculum, and checkpoint format all in one patch unless explicitly requested
- do not silently change normalization behavior
- do not silently change resume semantics
- do not claim a training improvement without running at least a targeted verification path
- do not skip tests after changing behavior-sensitive logic
