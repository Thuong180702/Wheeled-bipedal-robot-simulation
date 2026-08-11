# Full-state LQR — corrected derivation

Evidence for the full-state LQR paragraph of §Classical Baselines and for
Limitation 4 in the paper. Every row lives in `results.json`.

## What was corrected

Two defects invalidated the earlier artifact (`outputs/fullstate_lqr/results.json`,
now superseded — do not cite it):

1. **Height reference.** The design used the shared `_build_height_ik`
   polynomial of `wheeled_biped/controllers/lqr_balance.py`, which saturates:
   0.60, 0.65 and 0.70 m all return near-straight legs that stand at ~0.73 m.
   The controller was therefore designed at, and stood at, a height it was
   never commanded, and its apparent survival came from not tracking height at
   all. `full_lqr.standing_pose` replaces it with a geometric bisection against
   the real wheel-contact radius (exact to 1e-6 over 0.40–0.70 m). The shared
   polynomial is deliberately left alone so the six published reduced-order
   baseline numbers do not move.
2. **Control-rate rebinding.** `scripts/eval_balance.py` rebound
   `full_lqr._N_PHYSICS_SUBSTEPS`, a name the module does not define, so
   `--control-hz` changed the rollout rate but left the linearization at its
   50 Hz default. Fixed to rebind `_SUBSTEPS`.

## Design numbers (r_scale = 1)

| quantity | 100 Hz | 50 Hz |
|---|---|---|
| equilibrium base z (commanded 0.65 m) | 0.649987 m | 0.649987 m |
| equilibrium residual force | 0.428 N of 79.46 N weight (0.54 %) | same |
| reduced state dim / rank | 30 / 30 | 30 / 30 |
| cond(A) | 2.59e7 | 6.48e9 |
| open-loop max abs eig | 1.0545 | 1.1119 |
| closed-loop max abs eig | 0.9911 | 0.9892 |
| PBH stabilizable | yes | yes |

The `cond(A) ~ 1e10` figure quoted in earlier drafts is a 50 Hz artifact, and
in neither case does it obstruct the Riccati solve.

## Outcome

`r_scale` swept over six decades (1e-2 … 1e4), N = 20 episodes x seeds
{0, 42, 123}, direct torque, 400 Nm/s rate limit.

- At a 20 s horizon, `r_scale` >= 100 shows 0 % falls at some commanded
  heights (best: 0.65 m at r = 1000, height RMSE 11.0 mm, pitch RMS 0.12 deg).
- **Extending the horizon to 60 s falls 100 % in every configuration tested**
  (22.2–57.0 s mean time-to-fall), with planar drift growing monotonically to
  0.43–0.63 m. The 20 s window truncates a slow divergence; it is not
  stability.
- No single control weighting holds the band: r = 100 stands 20 s at 0.60 and
  0.69 m but falls at 0.65 m; r = 1000 does the reverse.
- Random height in [0.40, 0.70] m: 45 % falls (r = 100), 65 % (r = 1000),
  80 % (r = 1e4).
- Push: 100 % falls under the standard 50 N impulse; max recoverable push
  10.0 N (r = 100) / 20.2 N (r = 1000), against ACC's F_min = 83 N.

## Reproduce

```bash
.venv/bin/python scripts/eval_balance.py --controller baseline_full_lqr \
  --scenarios nominal --num-episodes 20 --num-steps 6000 \
  --seeds 0 --seeds 42 --seeds 123 --control-hz 100 --no-binary-search \
  --output-dir outputs/fullstate_lqr_corrected/rerun
```

`r_scale` comes from `configs/baseline_full_lqr.yaml`; override by copying that
file and passing `--baseline-config`. `--num-steps 6000` at 100 Hz is the 60 s
horizon — a 1000-step run is only 10 s and will report survival that a longer
horizon removes.
