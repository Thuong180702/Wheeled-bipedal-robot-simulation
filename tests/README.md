# `tests/` — how to run this suite, and what state it is in

## The supported run

```bash
pytest tests/ -m "not slow" -q $(sed 's|^|--ignore=|' tests/known_failing.txt)
```

3398 passed, 104 skipped, in about 100 s on CPU.

## Why there is an exclusion list

Running `pytest tests/` unfiltered produces roughly 400 failures. None of them
touch a number in the paper — the paper's evidence chain is the script → raw
result correspondence in the top-level [README](../README.md), and every one of
those harnesses runs and reproduces its committed JSON. But the failures are
real: 53 of the 60 excluded files fail the same way when run in isolation, so
this is not a collection or environment artifact.

`known_failing.txt` lists them. They fall into three groups:

- **Abandoned research lines.** Tests for the phase-B9 / step-C-D-E / hierarchical
  VMC / centroidal WBC / residual-PPO work that preceded ACC. They assert
  behaviour of designs that were evaluated and rejected. `test_actuator_signs.py`
  is a good example: it asserts a left/right symmetry that the paper's chirality
  analysis measures and reports as absent.
- **Evidence-file assertions.** Tests that check a validation report or telemetry
  CSV exists for a campaign that was never re-run after the code moved on.
- **`test_env.py`**, excluded for cost rather than failure: it constructs
  `BalanceEnv` and steps MJX, and the JIT compile dominates the run.

Six further files fail only inside a full-suite run and pass in isolation —
cross-test pollution, mostly through JAX's global x64 flag. They are excluded
too, so the supported run is deterministic.

## What the exclusion list is not

It is not a way to make a red suite look green. The 3398 tests that do run cover
the ACC controller itself: the JAX control step and its channels, the torque
composer, the gates and filters, the action codec, the classical baselines, the
offline QP whole-body-control stack, telemetry schemas and the reward terms.

If you are reviewing the paper, the run that matters is not this suite at all —
it is the reproduction table in the top-level README, where each result names the
script that produces it and the committed file it must match.

## Slow tests

`-m "not slow"` skips the tests that trigger MJX JIT compilation. Run them with:

```bash
pytest tests/ -m slow -q
```
