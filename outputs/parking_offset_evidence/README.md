# Sagittal parking offset — raw evidence

Raw results backing Limitation 8 (§VI) and Appendix C of `paper/main.tex`: the
static torque balance of the 27 mm sagittal parking offset.

| File | Backs |
|---|---|
| `parking_offset_diag.json` | Appendix C, all four tables and Fig. 7. One entry per run, keyed by `--tag`. Produced by `scripts/diagnose_parking_offset.py`. |

Tags inside `parking_offset_diag.json`:

| Tag | Backs |
|---|---|
| `baseline_n10` | **The paper's baseline.** ACC as published, N=10 under the full idle protocol, nominal height h=0.4040 m. Appendix C-A torque table, C-B pitch decomposition, the `k_i=4` row of C-D, and Fig. 7a. Reproduces Table III's −27.0 ± 0.009 mm (measures −26.961 ± 0.0094). |
| `pitcheq_fix_n10` | Feedforward pitch reference recalibrated by −1.461°, N=10. The nominal row of the C-F before/after table and the dotted line in Fig. 7b. |
| `ki_10 … ki_160` | The rest of the 40× gain sweep, C-D and Fig. 7b (N=3 each). |
| `leak_5e4` | The leak sweep, C-E. |
| `heightchk_*` / `fix_*` | The four off-nominal heights of the C-F table, before/after (single trial each). |
| `acc_baseline`, `pitcheq_fix_n3` | The N=3 predecessors of the two N=10 runs. Agree to three decimals; kept as the audit trail. |
| `decomp`, `decomp2`, `pitcheq_fix` | Earlier single-trial probes, superseded. `acc_baseline` and `decomp` predate the pitch-axis fix, so their `pitch_deg` is the ZYX pitch (lateral) — see the second trap below. Every other field in them is axis-independent and still valid. |
| `push_acc_published_ki4.json` | Appendix C-F push table, row "ACC as published". Reproduces the 82.8 N weakest bearing of Table VI, validating the harness. |
| `push_ki40.json`, `push_ki160.json` | Appendix C-F push table, rows `k_i=40` and `k_i=160`. |
| `push_ff_pitch_recalibrated.json` | Appendix C-F push table, row "feedforward pitch recalibrated". |
| `push_measure.py` | The push harness used for the four files above: §V-B protocol, N=10 repetitions × 8 bearings, binary search 10–160 N at 5 N tolerance. |
| `robustness_ki20.json`, `robustness_ki40.json`, `robustness_ki80.json` | Appendix C, "Why the gain is not raised" — the falls column, arms `k_i = 20/40/80`. Each is the two noise+delay cells (`med` noise × 10 ms and 30 ms) at N=20, produced by `scripts/collect_robustness_sweep.py` in a detached worktree with only `anchor_position_ki` changed. |

The `k_i = 4` row of that table is the shipped profile, so its data is the
paper's main robustness file, `outputs/paper_statistics/robustness_sweep.json`
(all 12 cells, N=20; the `med_10ms` and `med_30ms` cells are the comparable
ones). The `k_i = 160` arm ran the full 12-cell grid rather than the two cells
and is archived under `outputs/ki_promotion_campaign_2026-08-01/`, together
with that arm's idle-ladder, push-ablation, flight/terrain, delay-stability,
rate-limit and compound re-measurements. All arms use identical trial seeds
(`BASE_SEED*100 + delay_steps*10 + trial`), so the comparison is paired.

Neither correction is promoted into the shipped `V3_ANCHOR` profile. For the
feedforward recalibration the reason is the unresolved 4.5 N F_med cost; for
`k_i` it is the measured fall-rate regression above (Cochran–Armitage trend on
log2 k_i: z = 2.34, p = 0.019).

## Reproducing

```bash
# torque decomposition + closed-form check, ACC as published (~45 s)
.venv/bin/python scripts/diagnose_parking_offset.py 10 --tag baseline_n10

# the same with the feedforward pitch reference recalibrated
.venv/bin/python scripts/diagnose_parking_offset.py 10 \
    --pitch-eq-bias=-1.461 --tag pitcheq_fix_n10

# one gain-sweep point
.venv/bin/python scripts/diagnose_parking_offset.py 3 --ki 40 --tag ki_40

# the noise+delay cost of one gain-sweep point (~10 min). Run from a detached
# worktree with anchor_position_ki edited, so the shipped profile is untouched:
#   git worktree add --detach /tmp/wt_ki40 HEAD && ln -s "$PWD/.venv" /tmp/wt_ki40/.venv
#   sed -i '' 's/anchor_position_ki=4\.0,/anchor_position_ki=40.0,/' \
#       /tmp/wt_ki40/wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py
# then restrict the grid to the two cells that carry the effect:
#   NOISE_LEVELS -> {"med": ...} and DELAY_STEPS = [1, 3] in collect_robustness_sweep.py
.venv/bin/mjpython scripts/collect_robustness_sweep.py --n-trials 20

# the feedforward correction at a non-nominal height (note: the bias flag
# needs the `=` form; the space form is silently dropped). --variant names
# match archive/.../balance_core_true_height_variants/variant_<NAME>__*.json
.venv/bin/python scripts/diagnose_parking_offset.py 1 --variant low_small \
    --pitch-eq-bias=-1.622 --tag fix_low_small

# figure + self-check (asserts the closed form tracks every swept point)
.venv/bin/python scripts/plot_parking_offset.py

# push validation, one arm (~20 min each). argv[1] = JSON overrides keyed by
# index into jax_params (88 = _IDX_ANCHOR_KI, k2_jax_controller.py:220),
# argv[2] = feedforward pitch bias in degrees. Result JSON on stdout.
.venv/bin/python outputs/parking_offset_evidence/push_measure.py \
    '{"88": 40.0}' 0 > push_ki40.json

# the feedforward-recalibration arm: no gain change, pitch map shifted instead
.venv/bin/python outputs/parking_offset_evidence/push_measure.py \
    '{}' -1.461 > push_ff_pitch_recalibrated.json
```

`diagnose_parking_offset.py` writes into
`outputs/paper_verification/parking_offset_diag.json`, accumulating one key per
`--tag`; the copy here is the snapshot the paper was written against. That is a
read-modify-write of a single file, so **run these sequentially** — two in
parallel will lose one of the results.

## Two traps when reading these files

- **`sag_pos_err_mm` is the controller's own position error, not the base
  offset.** The base offset is `sag_mm` (the −27.0 mm headline); it is larger
  by the 1–3 mm base-frame versus support-centre convention gap. The closed
  form of Eq. (27) — and `pred_err_mm` in these records — predicts the
  *controller* error, so compare it against `sag_pos_err_mm`.
- **Sagittal is world `y` on this platform, and the controller's sagittal
  attitude is `pitch_x` = rotation about world `x`** (`orientation_utils.py:36`,
  `body_pitch_x = roll`). The ZYX "pitch" is the *lateral* axis here. The `x`
  columns of these records are lateral. Both conventions have caused a wrong
  diagnosis in this repository before.
