# wheeled_biped/inference/CLAUDE.md

This file supplements the root `CLAUDE.md`.
Follow the root file first. Use this file for inference/controller-specific rules.

## Purpose of this folder

This folder contains inference-time policy usage and multi-skill control logic.
Its most sensitive responsibility is making different trained policies work together safely and predictably.

This area is high-risk because shape compatibility is not enough; observation semantics must also match.

## Local priorities

The current direction for this folder is:

- reduce heuristic brittleness in the unified controller
- avoid unsafe generic observation pad/cut behavior
- make skill switching more stable and explicit
- preserve checkpoint-loading usability where practical

## Local invariants

- Treat observation semantics as more important than mere observation length.
- Do not assume two policies are interchangeable just because tensor shapes line up.
- Prefer explicit per-skill adapters over generic reshaping.
- Preserve current controller/checkpoint usability unless the task explicitly changes compatibility.
- When changing switching logic, think about stability, hysteresis, and failure behavior.

## Editing rules for inference code

Before editing:

1. identify the exact controller/policy-loading path
2. inspect how checkpoints are loaded
3. inspect how observations are adapted for each skill
4. inspect switching conditions and blending behavior
5. inspect failure paths for missing checkpoints or mismatched shapes
6. inspect relevant tests

When editing:

- prefer explicitness over convenience
- preserve existing behavior unless the task explicitly requires changing it
- isolate risky changes behind clear adapters or helper functions
- do not silently coerce inputs in a way that can hide semantic mismatches

After editing:

- run targeted controller tests first
- run a smoke path if practical
- report any remaining unverified runtime paths

## Observation adaptation rules

This is the most important rule in this folder:

Do not use generic pad/cut logic as the default solution when two skills expect different observation semantics.

Prefer:

- explicit adapters per skill
- documented assumptions for each adapter
- validation checks on required inputs
- clear failure messages when adaptation is not safe

If a policy expects task-specific extras:

- make those extras explicit
- document how they are derived
- do not silently inject zero-filled placeholders unless that is an explicitly approved fallback

## Skill switching rules

When editing switching logic:

- list all current heuristics before changing them
- reason about flicker / oscillation risk
- consider hysteresis or dwell-time if switching is unstable
- preserve safety-oriented behavior where possible
- prefer a small, testable stability improvement over a broad redesign

Do not:

- bury switching thresholds in scattered magic numbers
- make switching less interpretable unless explicitly asked
- claim robustness improvements without testing relevant cases

## Checkpoint loading rules

When touching inference-time loading:

- preserve existing checkpoint discovery/loading flow where possible
- be explicit about missing-skill behavior
- be explicit about shape/metadata mismatches
- prefer informative errors over silent fallback behavior

## Benchmark / validation rules

Inference changes should be validated against controller behavior, not only code execution.

Useful checks include:

- missing checkpoint handling
- shape mismatch handling
- stable switching behavior
- correct policy selection for representative cases
- behavior under boundary conditions near switching thresholds

## Files to inspect first for inference tasks

- `wheeled_biped/inference/unified_controller.py`
- any other policy-loading or inference helpers in this folder
- `scripts/visualize.py`
- `scripts/evaluate.py`
- relevant training/checkpoint utilities if loading behavior is involved
- relevant tests in `tests/`

## Good task framing for this folder

A good inference task request should specify:

- which controller path is affected
- whether switching behavior should change
- whether checkpoint compatibility must be preserved
- whether observation adaptation may change
- what smoke tests or controller tests should pass

## What not to do here

- do not treat shape compatibility as semantic compatibility
- do not use generic pad/cut logic as the default fix
- do not make switching logic more opaque without a strong reason
- do not silently swallow missing checkpoint or adapter errors
- do not bundle a major controller redesign into a small bugfix unless explicitly requested
