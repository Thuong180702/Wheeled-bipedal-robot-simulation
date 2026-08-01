# WBC post-fix campaign — raw evidence

Raw results backing §V-E and Appendix B of `paper/main.tex`. Every file here was
produced *after* the F1–F4 audit repairs (commit `433160f`, 2026-07-20). No
pre-repair artifact survives; see the Evidence Provenance subsection of
Appendix B.

| File | Backs |
|---|---|
| `full_batch_results.jsonl` | Table IX, full-batch panel. 225 scenarios, 481,000 control steps, three arms (V3/ACC, WBC-only, ACC+WBC assist). Merged from 70 shards by `scripts/merge_wbc_shards.py`. |
| `gated/ungated_low_small.jsonl` | Table IX, gate-open subset, `gate off` column. Slice of the campaign above (`random_push`, `low_small`, seeds 201–205). |
| `gated/assist_gated_low_small.jsonl` | Table IX, gate-open subset, `gate on` column. Same 5 scenarios re-run with `--adaptive`. |
| `gated/gate_trace_lowsmall.jsonl.gz` | Table XIII, bottom panel. Per-step α and gate factors, 10,775 steps. Produced by setting `WBC_GATE_TRACE=<path>` when running `scripts/phase3d_full_batch_execution.py`. |
| `ablation/*.jsonl` | Table XII. WBC-only arm at four task weightings over `random_push`/nominal/seeds 201–205. `balanced_default.jsonl` is the corresponding slice of the main campaign. |

## Reproducing the summaries

```bash
.venv/bin/python scripts/summarize_wbc_postfix.py    # Table IX full batch
.venv/bin/python scripts/summarize_wbc_ablation.py   # Table XII
```

## Two traps when reading these files

- `wbc_only_falls` / `v3_falls` count control **steps spent in a fallen state**,
  not fall events. The paper reports them as "fraction of episode fallen".
- `height_rms` is the RMS of **absolute** base-z, not a tracking error. A smaller
  value means the robot is closer to the floor, not that it tracked better.

The 70 raw per-shard directories (47 MB, `outputs/wbc_postfix_shards/`) are not
committed; `full_batch_results.jsonl` is their merge and is sufficient to
reproduce every number in the paper.
