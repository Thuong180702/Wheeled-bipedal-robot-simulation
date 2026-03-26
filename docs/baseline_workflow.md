# Benchmark Regression Baseline Workflow

This document describes how to generate a benchmark baseline and detect regressions
after code changes.

---

## Baseline format

A baseline is a plain JSON file produced by `scripts/evaluate.py`.
The file is the JSON serialisation of `BenchmarkResult.to_dict()` plus three metadata keys:

```json
{
  "mode": "nominal",
  "num_episodes": 100,
  "reward_mean": 42.31,
  "reward_std": 5.12,
  "reward_min": 18.4,
  "reward_p5":  22.1,
  "reward_p50": 43.0,
  "reward_p95": 60.2,
  "reward_max": 71.8,
  "episode_length_mean": 980.5,
  "episode_length_max": 1000,
  "success_rate": 0.87,
  "fall_rate": 0.13,
  "timeout_rate": 0.87,
  "mode_metrics": {},
  "checkpoint": "outputs/checkpoints/balance/final",
  "stage": "balance",
  "seed": 0
}
```

For `command_tracking` mode, `mode_metrics` additionally contains:
`overall_height_rmse` and `per_command` (list of per-command metrics).

---

## Step 1 — Generate a baseline

Train a policy, then evaluate it:

```bash
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/balance/final \
  --stage balance \
  --mode nominal \
  --num-episodes 200 \
  --seed 0
# → saves to outputs/checkpoints/balance/final/eval_results_nominal.json
```

Promote the JSON to a committed baseline:

```bash
mkdir -p baselines/
cp outputs/checkpoints/balance/final/eval_results_nominal.json \
   baselines/nominal_v1.json
git add baselines/nominal_v1.json
git commit -m "chore: add nominal benchmark baseline v1"
```

---

## Step 2 — Compare after a code change

After making changes, run evaluate again and compare:

```bash
# Re-evaluate (uses the same checkpoint — only code changed)
python scripts/evaluate.py \
  --checkpoint outputs/checkpoints/balance/final \
  --stage balance \
  --mode nominal \
  --num-episodes 200 \
  --seed 0

# Compare
python scripts/compare_baseline.py \
  --baseline baselines/nominal_v1.json \
  --current  outputs/checkpoints/balance/final/eval_results_nominal.json
```

Exit code **0** = no regressions. Exit code **1** = regressions detected.

### Example output

```
┌──────────────────────────────────────────────────────────────────┐
│           Baseline Comparison  [mode=nominal]                    │
├──────┬──────────────────────────────┬──────────┬────────┬───────┤
│      │ Metric                       │ Baseline │Current │ Delta │
├──────┼──────────────────────────────┼──────────┼────────┼───────┤
│ ❌   │ fall_rate                    │   0.1300 │ 0.2100 │+0.0800│
│ ⬆️   │ reward_mean                  │  42.3100 │ 47.0000│+4.6900│
│      │ success_rate                 │   0.8700 │ 0.7900 │-0.0800│  (within tol)
└──────┴──────────────────────────────┴──────────┴────────┴───────┘
❌  FAILED — 1 regression(s) detected.
```

---

## Step 3 — Optional: save comparison diff to JSON

```bash
python scripts/compare_baseline.py \
  --baseline baselines/nominal_v1.json \
  --current  eval_results_nominal.json \
  --save-json baseline_diff.json
```

---

## Regression tolerances

Tolerances are defined in `wheeled_biped/eval/baseline.py` (`_REGRESSION_SPECS`):

| Metric | Direction | Tolerance type | Default tolerance |
|---|---|---|---|
| `fall_rate` | lower is better | absolute | ±0.05 (5 pp) |
| `success_rate` | higher is better | absolute | ±0.05 (5 pp) |
| `reward_mean` | higher is better | relative | 5% |
| `reward_p5` | higher is better | relative | 5% |
| `overall_height_rmse` | lower is better | relative | 10% |
| `fall_after_push_rate` | lower is better | absolute | ±0.05 |

Override per-metric tolerances programmatically:

```python
from wheeled_biped.eval.baseline import compare_files

result = compare_files(
    current_path="eval_results_nominal.json",
    baseline_path="baselines/nominal_v1.json",
    tolerances={"fall_rate": 0.02, "reward_mean": 0.03},  # tighter
)
result.print_summary()
assert result.passed
```

---

## What is intentionally manual

- **Baseline selection**: which checkpoint to baseline is a project decision.
  The tool does not auto-select or auto-commit baselines.
- **Re-training**: if you want to update the baseline after an intentional
  improvement, re-run evaluate and commit the new JSON.
- **Multi-mode comparison**: run `compare_baseline.py` once per mode
  (nominal, push_recovery, domain_randomized, command_tracking).
