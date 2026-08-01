# Sagittal parking offset — raw evidence

Raw results backing Limitation 8 (§VI) and Appendix C of `paper/main.tex`: the
static torque balance of the 27 mm sagittal parking offset.

| File | Backs |
|---|---|
| `parking_offset_diag.json` | Appendix C, all four tables and Fig. 7. One entry per run, keyed by `--tag`. `acc_baseline` = ACC as published, N=3, at the nominal height h=0.4040 m (Appendix C-A torque table, C-B pitch decomposition, and the `k_i=4` row of C-D); `ki_10 … ki_160` = the rest of the gain sweep (C-D); `leak_5e4` = the leak sweep (C-E); `heightchk_*` / `fix_*` = the four off-nominal heights of the C-F before/after table, and `pitcheq_fix_n3` its nominal-height row (N=3); `decomp`, `decomp2` are earlier single-trial probes, superseded by `acc_baseline`. Produced by `scripts/diagnose_parking_offset.py`. |
| `push_acc_published_ki4.json` | Appendix C-F push table, row "ACC as published". Reproduces the 82.8 N weakest bearing of Table VI, validating the harness. |
| `push_ki40.json`, `push_ki160.json` | Appendix C-F push table, rows `k_i=40` and `k_i=160`. |
| `push_ff_pitch_recalibrated.json` | Appendix C-F push table, row "feedforward pitch recalibrated". |
| `push_measure.py` | The push harness used for the four files above: §V-B protocol, N=10 repetitions × 8 bearings, binary search 10–160 N at 5 N tolerance. |

Neither correction is promoted into the shipped `V3_ANCHOR` profile — every
other table in the paper was measured on the published configuration.

## Reproducing

```bash
# torque decomposition + closed-form check, ACC as published
.venv/bin/python scripts/diagnose_parking_offset.py 3 --tag acc_baseline

# one gain-sweep point
.venv/bin/python scripts/diagnose_parking_offset.py 3 --ki 40 --tag ki_40

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
`--tag`; the copy here is the snapshot the paper was written against.

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
