# Anchored Conditionally-Gated Control (ACC)

Controller, MuJoCo model, evaluation harnesses and raw result files for the paper

> **Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery**
> Van Thuong Nguyen, Nhu Thanh Vo — The University of Danang
> *Submitted to Robotics and Autonomous Systems.*

ACC is a hand-designed, fully interpretable torque assembly for a 10-DOF, 8.1 kg
wheeled biped. Its central mechanism is a **proximity-gated anchor**: a position
integral confined to a 15 cm neighbourhood of the commanded home and enabled by
an asymmetric envelope follower (23 ms attack, 1.5 s release) that separates
quiet stance from post-push ringdown. This resolves the standing trade-off in
which the stiffness that buys precision shrinks the push-recovery envelope,
while the integral action that removes equilibrium bias winds up during a
disturbance.

> **Scope.** Everything here is simulation. **No result in this repository has
> been validated on hardware.** The controller was built to serve as the nominal
> base action of a residual-learning stack; the paper reports the prior itself,
> and no residual policy is trained in it.

---

## Headline results

| Quantity | Value |
|---|---|
| Sagittal standing repeatability | 0.294 ± 0.015 mm (N = 10) |
| — vs. the proportional-only rung of the same ablation | 59× better |
| Sagittal parking offset (absolute accuracy) | 27.0 mm, held to 0.009 mm |
| — predicted by the closed-form model of Appendix C | to 0.92 % over a 40× gain sweep |
| Omnidirectional push recovery | 108 N median, 82 N worst case |
| Push-envelope chirality | −17.3 N, traced to the hip-roll/hip-yaw sign convention |
| Long-horizon standing | 30 min at 0.0004 mm/min drift |
| Drop recovery | up to 100 cm free fall |
| Terrain | curbs 10–20 cm, per-leg height split |
| Actuator-delay cliff | 6–8 ms (moved outward by a single retuned gain) |

Six classical baselines (four from a 4-state TWIP model, two from a 6-state
coupled model) and a full-state LQR linearized from the plant itself all fall in
every trial under the same protocol.

---

## Platform

10 DOF — hip roll, hip yaw, hip pitch, knee, drive wheel, per leg.

```
Mass        8.1 kg          Thigh   0.26 m      Wheel radius  0.06 m
Hip width   0.23 m          Shin    0.28 m
Model       SolidWorks -> URDF -> MuJoCo MJCF
Simulation  MuJoCo 500 Hz physics, 100 Hz control (5 substeps), raw torque
            commands, 400 Nm/s torque rate limit
Contact     implicitfast, 4 solver / 8 linesearch iterations; pyramidal cone on
            the body, elliptic on the wheels; mu_s = 1.2, mu_k = 0.8
```

| Joint | ctrlrange (Nm) | Joint range (rad) |
|---|---|---|
| Hip roll | ±30 | [−0.7, 0.7] |
| Hip yaw | ±30 | [−0.4, 0.4] |
| Hip pitch | ±150 | [−0.5, 1.8] |
| Knee | ±150 | [−0.5, 2.7] |
| Wheel | ±30 | unlimited |

**Height convention.** Two vertical references appear in this work and must not
be substituted for one another — swapping them injects the full 13.2 cm offset
as a reference error.

- **CoM-z** — the quantity ACC commands and regulates. Nominal stance 0.404 m.
- **base-z** — torso root-body height, MuJoCo `qpos[2]`. Same pose sits at 0.536 m.

All ACC results are CoM-z. The PPO reference and the WBC baselines command
base-z and are labelled as such.

**Axis convention.** Both wheel axles lie along world **X**; the rolling
(sagittal) direction is world **Y**, and world **X** is the lateral (track)
axis. Verified four independent ways (`scripts/verify_body_axes.py`).

---

## Controller structure

Two torque channels — wheel torque `tau_w` (2) and leg-joint torque `tau_q` (8) —
with flight and terrain entering as independent extensions:

```
tau_w = (1 - g_flight) * [ tau_balance + g_anchor * tau_anchor ] + g_flight * tau_flight
tau_q = tau_posture(h_cmd_L, h_cmd_R) + tau_lat + tau_yaw
```

Two structural points that are easy to misread as sums: the flight term is an
**override**, not a summand — at `g_flight = 1` it replaces the wheel torque —
and per-leg ground adaptation enters as a **reference split** on the posture
regulator, not as an extra torque. Every gate except the flight gate is a
smoothstep of a physical condition (proximity, quietness, attitude, terrain
disparity), never a discrete switch: at 100 Hz under a 400 Nm/s rate limit a
hard threshold would deliver the full gated component to the plant as a ramp.

What the factorial ablation localizes:

- **Gated damping** produces the steadiness. Reverting the two velocity-damping
  coefficients reinstates the full 25.2 mm limit cycle.
- **The anchor integral** produces the accuracy — it removes 4.2 mm of parking
  offset at a threefold cost in steadiness.
- **The proximity gate and the asymmetric envelope** are bit-for-bit inert in
  quiet stance and decisive under disturbance.
- **The pitch safety gate** is load-bearing even with no disturbance applied:
  removing it diverges from initial-condition noise alone in all ten trials.

### Where it lives

| Component | File |
|---|---|
| ACC profile (shipped default) | `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` → `K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR` |
| JAX control step | `wheeled_biped/controllers/k2_jax_controller.py` |
| Posture / height IK | `wheeled_biped/controllers/height_ik.py`, `centered_posture_height_schedule.py` |
| Yaw and lateral channels | `wheeled_biped/controllers/differential_wheel_yaw_stabilizer.py`, `lateral_roll_balance_controller.py` |
| Classical baselines | `wheeled_biped/controllers/lqr_balance.py`, `coupled_lqr_3d.py`, `full_lqr.py`, `pi_aw_baseline.py` |
| WBC baselines (offline QP) | `wheeled_biped/wbc/offline_task_stack.py`, `offline_qp_wbc.py` |
| Teleop command shaper | `wheeled_biped/teleop_shaper.py` |

`V3_ANCHOR` is the internal name of the profile the paper calls **ACC**. The two
are the same controller.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # requirements-macos.txt on Apple silicon
pip install -e ".[dev]"
```

Python ≥ 3.10, MuJoCo ≥ 3.1. The ACC evaluations run on CPU; only the PPO
reference of Section 5.4 needs a GPU.

---

## Reproducing the paper

Each table and figure names its producing script and its raw result file. Run the
script, then diff its output against the committed JSON.

| Paper item | Script | Raw result |
|---|---|---|
| Table 2 — idle standing ablation ladder | `scripts/collect_idle_ladder.py` | `outputs/paper_verification/idle_ladder.json` |
| 30 min / 5 min long-horizon idle | `scripts/idle_longhorizon.py` | `outputs/paper_verification/idle_longhorizon_{300,1800}s.json` |
| Table 3 — factorial push ablation (N = 30) | `scripts/replicate_ablation_n30.py` | `outputs/paper_statistics/ablation_n30_results.json` |
| Fig. — push polar map / chirality | `scripts/sweep_push_chirality.py` | `outputs/push_chirality/` |
| Flight recovery and terrain (N = 10) | `scripts/collect_flight_terrain_n10.py` | `outputs/flight_terrain_n10/results.json` |
| Curb / ledge courses | `scripts/ramp_step_tests.py` | `outputs/oblique_regime/` |
| Table 4 — classical baselines, rate-matched | `scripts/collect_rate_matched_baselines.py` | `outputs/rate_matched_baselines/results.json` |
| Full-state LQR control experiment | `scripts/eval_balance.py --controller baseline_full_lqr` | `outputs/fullstate_lqr_corrected/` |
| Coupled-model roll-coupling sweep | `scripts/sweep_b_roll_hip.py` | `outputs/b_roll_hip_sweep/results_100hz.json` |
| Table 5 — WBC three-arm counterfactual | `scripts/verify_wbc_audit_fixes.py` | `outputs/wbc_postfix_evidence/` |
| Robustness sweep (noise, mass, friction) | `scripts/collect_robustness_sweep.py` | `outputs/paper_statistics/robustness_sweep.json` |
| Delay cliff, 2 ms resolution + retune | `scripts/delay_cliff_resolution.py`, `scripts/delay_retune_sweep.py` | `outputs/robustness_sweep/delay_cliff/`, `.../delay_retune/` |
| Compound disturbance map | `scripts/compound_disturbance_corrected.py` | `outputs/compound_disturbance/results_corrected_map_shipped.json` |
| Torque rate-limit sweep | `scripts/sweep_rate_limit_corrected.py` | `outputs/rate_limit_sweep/results_corrected.json` |
| Table 6 — local stability certificate | `scripts/acc_stability_certificate.py` | `outputs/paper_verification/acc_stability_certificate.json` |
| Appendix C — parking-offset closed form | `scripts/diagnose_parking_offset.py` | `outputs/parking_offset_evidence/` |
| Control-step cost (27 664 flops) | `scripts/measure_control_step_cost.py` | `outputs/paper_verification/control_step_cost.json` |
| Axis-convention verification | `scripts/verify_body_axes.py` | — |
| Figures | `scripts/generate_paper_figures.py` | `paper/figures/` |

Two directories carry a `README.md` mapping every file to the table it produces:
`outputs/wbc_postfix_evidence/` and `outputs/parking_offset_evidence/`. The
latter also gives the worktree recipe for reproducing a gain arm without
disturbing the shipped profile.

Report the control-step cost as the **27 664-flop operation count**, not host
wall-clock — the latter drifts between runs.

### Supplementary video

The paper's 31-clip inventory lives under `paper/videos/` (the directory also
holds two teleop session recordings that are not part of it), regenerated in
full by:

```bash
python scripts/render_paper_videos.py
```

or one group at a time with `--only push drop curb ledge height idle`. Each clip
is a render of the evaluation itself, driven by the same harness that produced
the corresponding table at the same seeds, so a clip and the number it
illustrates cannot drift apart. `paper/videos/README.md` gives the per-clip
protocol and the measured outcome. Each clip is one trial at one seed and
therefore illustrates a regime rather than establishing a rate.

---

## Repository layout

```
wheeled_biped/
├── controllers/        ACC, its components, and the classical baselines
├── wbc/                offline QP whole-body-control baselines
├── envs/               MuJoCo/MJX environments
├── training/           PPO (used only for the preliminary reference run)
├── rewards/            reward terms for the RL reference
├── sim/                low-level control, physics helpers
└── inference/          unified controller entry point
assets/robot/           MJCF, URDF and STL meshes
configs/                robot, controller and training configuration
scripts/                evaluation harnesses, sweeps, figure and video renderers
outputs/                raw result files backing the paper
paper/bir/              manuscript (Elsevier CAS class)
paper/figures/          generated figures
paper/videos/           31 supplementary clips + protocol
tests/                  pytest suite
```

```bash
pytest tests/ -m "not slow" -q
```

---

## The RL side of this repository

The repository also contains a MuJoCo MJX + JAX PPO stack. Its role in the paper
is narrow and should not be overstated: a **single-seed, 5.4 M-step preliminary
run**, carrying neither domain randomization nor a hyperparameter search, used as
a difficulty probe rather than as a bound. It stands at one height for ~3.20 s
against 0.16–0.60 s for the classical baselines, but fails to generalize across
the 0.40–0.70 m base-z range, with height RMSE above 50 mm and falls on 85 % of
squat transitions. **It is not a tuned RL baseline and must not be cited as one.**

The residual-PPO-over-a-structured-prior framework that motivated ACC's design is
scaffolded here (`ResidualBalanceEnv`, 52-dim observation, `balance_residual*.yaml`)
but **is not trained or evaluated in the paper**. See `CLAUDE.md` for the state of
that work.

---

## Limitations

- Simulation only; no hardware validation of any kind.
- Every sub-millimetre idle figure is conditional on stiff ground contact.
  Softening `solref[0]` to 0.02 (approximately tire on asphalt) raises planar
  idle RMS by roughly 60 %, to about 1.2 mm.
- Standing precision is reported under clean sensing and zero actuator delay; the
  delay screen prices the cliff separately at 6–8 ms.
- The posture map is calibrated at two setups (CoM 0.354 m and 0.454 m) and
  linearly extrapolated to [0.254, 0.554] m outside that band.
- Locomotion, stair climbing and rough terrain exist as configuration stubs only
  and are neither trained nor evaluated.

---

## Citation

```bibtex
@article{nguyen2026acc,
  title   = {Anchored Conditionally-Gated Control for Wheeled Bipedal Balance
             and Disturbance Recovery},
  author  = {Nguyen, Van Thuong and Vo, Nhu Thanh},
  journal = {Robotics and Autonomous Systems},
  year    = {2026},
  note    = {Submitted}
}
```

---

## License

This repository is dual-licensed, because the paper and the software need
different terms.

| What | License | File |
|---|---|---|
| Software — everything under `wheeled_biped/`, `scripts/`, `tests/`, `configs/`, and the MJCF/URDF under `assets/` | [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) | [`LICENSE-CODE`](LICENSE-CODE) |
| Paper materials — `paper/` (manuscript, figures, the 31 supplementary clips) and the raw result files under `outputs/` | [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/) | [`LICENSE`](LICENSE) |

CC BY-NC-ND 4.0 matches the license the journal applies to the article itself.
It is deliberately **not** applied to the software: its NoDerivatives clause
would forbid the forking and modification that reproducing this work requires.
Where the two overlap, the software terms govern the software.

Previous releases of this repository were distributed under the MIT license;
that grant is not revoked for copies already obtained under it.
