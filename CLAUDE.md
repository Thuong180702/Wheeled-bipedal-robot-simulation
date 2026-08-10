# CLAUDE.md

## What this repository is

The controller, MuJoCo model, evaluation harnesses and raw result files behind

> **Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and
> Disturbance Recovery** — Van Thuong Nguyen, Nhu Thanh Vo, The University of
> Danang. Submitted to *Robotics and Autonomous Systems*.

ACC is a **hand-designed, fully interpretable torque assembly** for a 10-DOF,
8.1 kg wheeled biped. It contains no learned component. Its central mechanism is
a proximity-gated anchor: a position integral confined to a 15 cm neighbourhood
of the commanded home, enabled by an asymmetric envelope follower (23 ms attack,
1.5 s release) that separates quiet stance from post-push ringdown.

Hold this mental model:

> a simulation-only classical-control codebase whose deliverable is a paper.
> Every number in the manuscript is backed by a named script and a committed raw
> result file, and that correspondence is the thing to protect.

It is **not** a reinforcement-learning project. An MJX/JAX PPO stack exists here
(see *The RL side*) but is one preliminary run used as a difficulty probe.

`README.md` is the public description and is current. Prefer it for platform
numbers, headline results and the paper-item → script → raw-result table. This
file carries the working rules that do not belong in a public README.

---

## Ground truth and scope

- **Simulation only. No result here has been validated on hardware.** Never
  write, or let a document imply, sim-to-real transfer or deployment readiness.
- The shipped default profile is `K2_JAX_DEDICATED_DEFAULT_V3_ANCHOR`
  (`wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`,
  promoted 2026-07-21). **`V3_ANCHOR` is the internal name of what the paper
  calls ACC — the same controller.** Keep both names findable when editing.
- Locomotion, stair climbing and rough terrain are **configuration stubs**,
  neither trained nor evaluated.
- Release `v1.0.0` is archived on Zenodo. The paper cites the **version DOI**
  `10.5281/zenodo.21861318`; the concept DOI `10.5281/zenodo.21861316` resolves
  to the latest version. **Do not rewrite git history** — it would change the
  hash of the tag the minted DOI points at.

---

## Invariants that break results silently

### Height reference

Two vertical references exist and must never be substituted; swapping them
injects the full **13.2 cm** offset as a reference error.

- **CoM-z** — what ACC commands and regulates. Nominal stance **0.404 m**.
- **base-z** — torso root height, MuJoCo `qpos[2]`. Same pose, **0.536 m**.

All ACC results are CoM-z. The PPO reference and the WBC baselines command
base-z and must be labelled as such wherever reported.

### Axis convention

Both wheel axles lie along world **X**. The rolling (sagittal) direction is
world **Y**; world **X** is the lateral (track) axis. Re-run
`scripts/verify_body_axes.py` rather than reasoning from memory when a sign
looks wrong — it checks this four independent ways.

### Actuation

Ten actuators, in order:

```
0 l_hip_roll   1 l_hip_yaw   2 l_hip_pitch   3 l_knee   4 l_wheel
5 r_hip_roll   6 r_hip_yaw   7 r_hip_pitch   8 r_knee   9 r_wheel
```

ACC issues **raw torque** at 100 Hz control / 500 Hz physics (5 substeps) under
a **400 Nm/s** torque rate limit.

### Torque assembly

```
tau_w = (1 - g_flight) * [ tau_balance + g_anchor * tau_anchor ] + g_flight * tau_flight
tau_q = tau_posture(h_cmd_L, h_cmd_R) + tau_lat + tau_yaw
```

Two structures that read as sums but are not: the flight term is an **override**
(at `g_flight = 1` it replaces the wheel torque), and per-leg ground adaptation
enters as a **reference split** on the posture regulator, not an extra torque.

Every gate except the flight gate is a **smoothstep of a physical condition** —
never a discrete switch. At 100 Hz under a 400 Nm/s rate limit a hard threshold
delivers the whole gated component to the plant as a ramp. Do not "simplify" a
smoothstep into a threshold.

Every channel is assembled **inside** `k2_jax_controller_step`:
`k2_jax_shape_posture_compute`, `k2_jax_lateral_roll_compute`,
`k2_jax_yaw_compute`, `k2_jax_heading_hip_yaw_stabilizer`. The height schedule
behind them comes from `calibrated_outer_loop_functions_v2.py` and
`physics_equilibrium_feedforward.py`, sampled onto interpolation grids at import
time.

Standalone modules whose names suggest they do this work — `height_ik.py`,
`centered_posture_height_schedule.py`, `differential_wheel_yaw_stabilizer.py`,
`shape_posture_controller.py`, `lateral_roll_balance_controller.py` — predate the
JAX rewrite and are **not** on ACC's path. The last two are reachable only
because `controllers/__init__.py` re-exports them.

---

## Deleting files: six dependency mechanisms, five invisible to reading code

A 2026-08-10 cleanup untracked 2543 files. Getting the list right took five
correction rounds, because static import analysis finds only the first of these:

1. ordinary `import`
2. **runtime data under `archive/`** — fifteen paper harnesses and
   `wheeled_biped/teleop_shaper.py` load variant-setup JSON from
   `archive/cleanup_2026-06-13/output_summaries/balance_core_{true_height_variants,extended_height_range}/`
   at startup. `archive/` is not a junk directory.
3. **STL meshes** — `assets/robot/wheeled_biped_real.xml` loads eleven meshes
   from `assets/robot-urdf/meshes/`. Without them `MjModel.from_xml_path` raises
   and nothing simulates.
4. tests importing scripts by **bare module name** through `sys.path`
5. tests loading scripts by **assembled path**, e.g.
   `ROOT/"scripts"/"run_g1_….py"`, and asserting reports exist under `docs/validation/`
6. tests that **`pytest.skip("… not found")`** when a script is absent — the
   worst kind: deleting the file leaves the suite green while silently dropping
   coverage. About thirty script paths in `tests/` are guarded this way.

Also load-bearing: `docs/validation/V3_vs_V3_Assist_comparison_report.md`, which
both manuscripts `\path{}`-cite, and `paper/bir/thumbnails/`, which
`cas-common.sty` `\includegraphics` at load time.

**Never delete on the strength of a static scan.** Copy the tree with
`git archive HEAD`, delete the candidates in the copy, then run the paper
harnesses and the pytest suite there and diff both against the same runs on an
untouched copy.

---

## Known traps

- **Control-step cost:** report the **27 664-flop operation count**, never host
  wall-clock, which drifts between runs and has already had to be corrected once.
- **Torque rate limit in ad-hoc harnesses:** the correct value is 400 Nm/s.
  `scripts/run_v3_assist_comparison.py` still hardcodes `MAX_TORQUE_RATE = 100.0`;
  thresholds measured through it are not comparable to a paper number.
- **A height command needs `eq_joint` as well as `height_ref`.** Setting
  `height_ref` alone leaves the robot where it was.
- **`mj_contactForce` sign flips on box terrain.** Resolve it against the
  contact frame rather than assuming a fixed sign.
- **Shared controller context across trials** latches flight mode from one trial
  into the next and biases a bisection sweep. Build a fresh `ctx` per trial.
- **Before shipping a PDF, grep it for `??`.** A timed-out single-pass build once
  shipped with 159 unconverged references.
- **The posture map is calibrated at two setups only** (CoM 0.354 m and 0.454 m)
  and linearly extrapolated to [0.254, 0.554] m outside that band.

---

## Test suite

**The suite is already red on `main`** — measured on a pristine `git archive HEAD`
copy with CI's own flags:

```
404 failed, 4243 passed, 145 skipped, 4 errors   in 12:09
```

The comment in `.github/workflows/ci.yml` claiming "< 2 min total" is stale by
both measures. An absolute pass/fail count therefore proves nothing. To judge a
change, run the suite twice — untouched copy and changed tree — and diff the
sorted `FAILED`/`ERROR` node-id sets. **Watch the pass/skip split too**, because
mechanism 6 above converts coverage into skips without turning anything red.

```bash
pytest tests/ --ignore=tests/test_env.py -m "not slow" -q --timeout=120
```

---

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # requirements-macos.txt on Apple silicon
pip install -e ".[dev]"
```

```bash
cd paper/bir && latexmk -pdf main_v2.tex
```

---

## The RL side

`ResidualBalanceEnv` (52-dim observation), `configs/training/balance_residual*.yaml`
and `wheeled_biped/training/` scaffold a residual-PPO-over-a-structured-prior
framework. It **motivated ACC's design but is not trained or evaluated in the
paper**: one single-seed 5.4 M-step run, no domain randomization, no
hyperparameter search. It stands at one height for ~3.20 s against 0.16–0.60 s
for the classical baselines, but fails across the 0.40–0.70 m base-z range
(height RMSE > 50 mm, falls on 85 % of squat transitions). **Not a tuned RL
baseline; must not be cited as one.** Leave the scaffolding alone unless asked.

---

## JAX / MJX rules

- Prefer pure functions; preserve JIT- and scan-friendly structure.
- No Python-side loops in rollout or control hot paths where JAX patterns exist.
- Split RNG keys explicitly; never mutate JAX arrays in place.
- Keep NumPy/SciPy out of JIT hot paths; separate offline gain computation from
  runtime execution.
- The control step is shared by evaluation, visualization and teleop. Verify a
  change in all three, or state which remain unverified.

---

## Working style

- One task at a time. State target files, intended change, risk points and a
  verification plan before editing.
- Minimal diffs over opportunistic cleanup; don't change public interfaces
  unless the task requires it.
- Inspect the code rather than assume — several controller names here survive
  from superseded designs and no longer mean what they say.
- If a change touches a harness that produces a paper number, regenerate the
  committed raw result file or say plainly that it is now stale.
- On multi-stage experiments: report results and wait for confirmation before
  the next stage; batch manuscript edits into one pass at the end.
- A defect that has been repaired belongs in the main text as the result, or as
  a control experiment — never in Limitations as a caveat.

## Do not claim

- hardware validation, sim-to-real success, or hardware safety
- novelty of the form "first residual RL method" / "first LQR-RL hybrid"
- completed stand-up recovery, locomotion, stair climbing or rough terrain
- a tuned PPO baseline
- an "exact LQR" prior unless genuinely derived from per-height linearization
  and Riccati solves
