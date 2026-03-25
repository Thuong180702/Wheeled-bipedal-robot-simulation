# wheeled_biped/envs/CLAUDE.md

This file supplements the root `CLAUDE.md`.
Follow the root file first. Use this file for environment-specific rules.

## Purpose of this folder

This folder defines the robot environments and task-specific environment behavior.
It is the source of truth for:

- observation construction
- action semantics at the env boundary
- reset / step / termination behavior
- reward coupling at the env level
- task-specific curriculum hooks where applicable

## Local invariants

- Preserve the current robot action semantics unless the task explicitly requires changing them.
- Action dimension is 10 unless the robot model itself is intentionally changed.
- Base observation size is 39; task environments may extend it.
- `env.obs_size` must stay consistent with the actual observation vector returned by the env.
- If observation structure changes, update all affected consumers:
  - policy/network input assumptions
  - inference controller / adapters
  - evaluation utilities
  - tests
- Do not silently add or remove observation terms without checking downstream effects.
- Keep reset and termination logic explicit and easy to reason about.

## Editing rules for env code

Before editing:

1. identify the exact env class and task config involved
2. inspect observation construction
3. inspect action mapping / scaling
4. inspect termination conditions
5. inspect reward coupling
6. inspect tests that cover the affected env behavior

When editing:

- prefer minimal, local changes
- preserve JAX-friendly data flow
- avoid Python-side stateful logic in hot paths if the existing code is already JAX-structured
- do not introduce silent shape fixes
- do not “temporarily” pad/cut observations in env code to satisfy a network mismatch
- do not mix unrelated reward redesign into a small env bugfix unless explicitly asked

After editing:

- verify observation shape
- verify action shape
- verify reset/step contract
- verify termination behavior
- verify no obvious NaN / invalid state behavior
- run the smallest relevant env tests first

## Observation rules

Treat observation semantics as high-risk.

Always verify:

- ordering of terms
- units / scaling
- whether terms are robot-state, command-state, contact-state, or task-specific extras
- whether the change affects controller logic or saved checkpoints

If a task env extends the base observation:

- make the extension explicit
- document the new terms in code comments if they are non-obvious
- verify policy input assumptions

## Action rules

When touching actions:

- preserve actuator ordering unless intentionally changed everywhere
- keep scaling / clipping explicit
- verify compatibility with low-level control assumptions
- do not hide control-policy mismatches by post-hoc clipping unless that is already the intended design

## Reset / termination rules

When touching reset / done logic:

- ensure auto-reset assumptions still hold where used
- ensure terminal reward / done semantics remain correct for training
- do not accidentally convert a terminal condition into a silent reset behavior
- think through effects on rollout collection and evaluation metrics

## Reward coupling rules

Even if reward functions live elsewhere, env edits often change reward meaning.

When changing env state, commands, contacts, or task signals:

- inspect relevant reward terms
- check whether reward inputs still match intended semantics
- update tests if the reward-visible behavior changes

## Domain randomization / perturbation

This repo wants reusable randomization and perturbation logic across tasks.

When editing envs:

- avoid embedding one-off randomization logic directly into a single task env if it should be shared
- prefer extracting shared pieces when the boundary is clean
- preserve current balance behavior when refactoring shared perturbation/control paths

## Files to inspect first for env tasks

- `wheeled_biped/envs/base_env.py`
- task env files in `wheeled_biped/envs/`
- `wheeled_biped/rewards/reward_functions.py`
- `wheeled_biped/utils/config.py`
- affected tests in `tests/`

## Good task framing for this folder

A good env task request should specify:

- target env(s)
- intended behavior change
- whether observation shape/semantics may change
- whether reward behavior is expected to change
- how to verify the change

## What not to do here

- do not casually change observation size
- do not silently reorder observation fields
- do not silently reorder action dimensions
- do not hide controller-policy-env mismatches
- do not bundle broad reward redesign into a focused env patch
- do not claim shape compatibility without checking `env.obs_size` and actual observation construction
