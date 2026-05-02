# Wheeled Bipedal Robot Simulation

Simulation and reinforcement learning training for a wheeled bipedal robot using **MuJoCo MJX + JAX**, with PPO and curriculum learning.

> **Status:** Active research prototype. Only the `balance` stage has been trained and
> evaluated to date. Sim-to-real transfer has not been validated on hardware.

---

## Overview

| Task               | Description                                    | Status       |
|--------------------|------------------------------------------------|--------------|
| **Balance**        | Stand upright and hold commanded height         | Implemented  |
| **Balance Robust** | Push-recovery fine-tuning (40 N disturbances)  | Config ready |
| **Wheeled Locomotion** | Wheel-driven forward/backward/turn          | Config ready |
| **Walking**        | Leg-stepping locomotion                        | Config ready |
| **Stair Climbing** | Step up/down                                   | Config ready |
| **Rough Terrain**  | Uneven surface traversal                       | Config ready |
| **Stand Up**       | Self-recovery from fallen pose                 | Config ready |

---

## Robot specs

```
Thigh:       26 cm     Wheel:      Ø12 cm
Shin:        28 cm     Total mass: ~8.1 kg
Actuators:   10 × BLDC  Sensors:   IMU + 10 encoders
Model:       SolidWorks → URDF → MuJoCo MJCF
```

Joints per leg (5 × 2 = 10 total):

| Joint     | Function        | ctrlrange (Nm) | Joint range (rad) |
|-----------|-----------------|----------------|-------------------|
| Hip Roll  | Lateral lean    | ±15            | [-0.7, 0.7]       |
| Hip Yaw   | Foot rotation   | ±15            | [-0.4, 0.4]       |
| Hip Pitch | Thigh flex      | ±30            | [-0.5, 1.8]       |
| Knee      | Knee flex       | ±30            | [-0.5, 2.7]       |
| Wheel     | Drive wheel     | ±15            | unlimited         |

> **Note:** Left hip_pitch / knee axes are mirrored vs. the right leg (SolidWorks URDF
> symmetry). The policy learns to compensate automatically.

---

## Project structure

```
├── assets/
│   ├── robot/
│   │   ├── wheeled_biped_real.xml     # MuJoCo MJCF (default — used by all scripts)
│   │   └── wheeled_biped.xml          # Legacy MJCF (alternative model)
│   └── robot-urdf/
│       ├── urdf/HOANTHIEN_TEST.urdf
│       ├── meshes/*.STL               # 11 STL mesh files
│       └── config/                    # ROS joint config
├── configs/
│   ├── robot.yaml                     # Robot parameters
│   ├── curriculum.yaml                # Multi-stage curriculum pipeline
│   ├── baseline_lqr.yaml              # Classical LQR baseline (eval only)
│   └── training/
│       ├── balance.yaml               # Pure standing balance (push_magnitude=0)
│       ├── balance_robust.yaml        # Push-recovery fine-tuning (40 N)
│       ├── wheeled_locomotion.yaml
│       ├── walking.yaml
│       ├── stair_climbing.yaml
│       ├── rough_terrain.yaml
│       └── stand_up.yaml
├── docs/
│   └── baseline_workflow.md           # How to generate and compare baselines
├── outputs/                           # All experiment artifacts (gitignored)
│   ├── balance/
│   │   ├── rl/
│   │   │   ├── seed42/                # One directory per independent run
│   │   │   │   ├── checkpoints/       # PPO checkpoints (step_N/, final/)
│   │   │   │   ├── balance_seed42_metrics.jsonl
│   │   │   │   ├── run_metadata.json
│   │   │   │   └── tb/                # TensorBoard event files
│   │   │   ├── seed113/
│   │   │   ├── seed999/
│   │   │   └── paper_eval/            # Aggregated eval output across seeds
│   │   └── lqr/                       # LQR baseline evaluation outputs
│   ├── balance_robust/rl/seed42/
│   ├── stand_up/rl/seed42/
│   └── <stage>/rl/seed<N>/
├── paper/                             # LaTeX manuscript
│   ├── main.tex
│   ├── refs.bib.backup      # BibTeX-format backup (paper uses embedded bibliography)
│   └── figures/
│       └── annotated_robot_joints.png
├── wheeled_biped/                     # Main package
│   ├── envs/
│   │   ├── base_env.py                # MJX base environment
│   │   ├── balance_env.py             # Balance task (42-dim obs: 39-base + height_cmd + current_height + yaw_error)
│   │   └── ...                        # Other env stubs
│   ├── eval/
│   │   ├── benchmark.py               # 4 benchmark modes
│   │   ├── baseline.py                # Baseline save / comparison utilities
│   │   └── standing_quality.py        # Posture quality signals (pure numpy)
│   ├── rewards/
│   │   └── reward_functions.py        # JAX reward components
│   ├── training/
│   │   ├── ppo.py                     # PPO + obs normalisation
│   │   ├── networks.py                # Actor-Critic (Flax)
│   │   ├── curriculum.py              # Eval-driven curriculum manager
│   │   └── live_viewer.py             # Real-time viewer during training
│   ├── inference/
│   │   └── unified_controller.py      # Multi-skill controller with hysteresis
│   ├── sim/
│   │   ├── domain_randomization.py    # Mass / friction / damping randomisation
│   │   ├── push_disturbance.py        # Periodic push helper (JAX-compatible)
│   │   ├── low_level_control.py       # PID joint controller helper
│   │   └── terrain_generator.py
│   └── utils/
│       ├── math_utils.py              # JAX quaternion / rotation utilities
│       ├── config.py                  # YAML loader + model path resolver
│       └── logger.py                  # TensorBoard + WandB logger
├── scripts/
│   ├── train.py                       # Training entry point
│   ├── evaluate.py                    # Evaluation (4 benchmark modes)
│   ├── eval_balance.py                # Research evaluation: per-step metrics, scenario sweeps
│   ├── visualize.py                   # MuJoCo viewer / video / interactive
│   ├── validate_checkpoint.py         # Standing validation (benchmark + posture quality)
│   ├── compare_baseline.py            # Regression comparison CLI
│   └── export_results.py              # Export logs → CSV/PNG; eval JSON → Markdown/LaTeX
├── tests/                             # 16 test files (pytest)
├── .github/workflows/ci.yml           # GitHub Actions CI
├── pyproject.toml
└── requirements.txt
```

### Artifact layout convention

All training runs write into `outputs/<stage>/<controller>/seed<seed>/`:

| Controller | Description | Output root |
|---|---|---|
| `rl` | PPO-trained policy | `outputs/<stage>/rl/seed<N>/` |
| `lqr` | Classical LQR baseline (eval only) | `outputs/<stage>/lqr/` |

Each RL run directory is self-contained:

```
outputs/balance/rl/seed42/
├── checkpoints/
│   ├── step_<env_steps>/checkpoint.pkl
│   └── final/checkpoint.pkl
├── balance_seed42_metrics.jsonl   ← training metrics (JSONL)
├── run_metadata.json              ← seed, config, git hash
└── tb/balance_seed42/             ← TensorBoard events
```

Final paper results use 3 independent seeds (42, 113, 999). Each seed is a fully
independent training run; results are aggregated post-hoc over the three runs.

---

## Installation

### Requirements

- Python 3.10 (required — JAX/jaxlib constraints)
- NVIDIA GPU + CUDA 12 (recommended; CPU fallback works but is slow for training)
- RAM ≥ 16 GB

### Install

```bash
git clone https://github.com/Thuong180702/Wheeled-bipedal-robot-simulation.git
cd Wheeled-bipedal-robot-simulation

python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux / macOS

pip install -r requirements.txt
```

> **Windows note:** If you encounter `uvloop` errors: install `orbax-checkpoint`,
> `flax`, `msgpack`, `rich`, and `PyYAML` manually with `--no-deps`.

### Verify

```bash
python -c "import mujoco; import jax; print(mujoco.__version__, jax.__version__)"
python scripts/visualize.py model          # open robot in MuJoCo viewer
pytest tests/ -v --ignore=tests/test_env.py  # run CPU-safe tests
```

### Reproducibility / compute stack

The table below records the environment used for the experiments reported in the paper.
Fill in the actual values from your own run with `pip freeze` and `nvidia-smi`.

| Component | Minimum required | Tested version |
|---|---|---|
| Python | 3.10 | 3.10.11 |
| JAX | 0.4.25 | 0.6.2 |
| jaxlib | 0.4.25 | 0.6.2 |
| MuJoCo | 3.1.0 | 3.5.0 |
| mujoco-mjx | 3.1.0 | 3.5.0 |
| Flax | 0.8.0 | 0.10.7 |
| Optax | 0.2.0 | 0.2.7 |
| NumPy | 1.26.0 | 2.2.6 |
| CUDA toolkit | 12.x | <!-- fill: e.g. 12.4 — run `nvcc --version` --> |
| NVIDIA driver | — | <!-- fill: e.g. 550.90.07 — run `nvidia-smi` --> |
| GPU model | — | <!-- fill: training GPU, e.g. NVIDIA RTX 3090 24 GB --> |
| OS | — | <!-- fill: e.g. Ubuntu 22.04 LTS or Windows 10 --> |

To capture your exact environment after a successful install:

```bash
pip freeze > requirements-lock.txt
```

Commit `requirements-lock.txt` alongside any final experiment checkpoints for full
reproducibility.

---

## Usage

### 1. View robot model

```bash
python scripts/visualize.py model
python scripts/visualize.py model --model-path path/to/custom.xml
```

### 2. Training

#### Single stage

```bash
# Pure standing balance — initial learning run (5M steps learns narrow height range)
python scripts/train.py single --stage balance --steps 5000000

# Same run with explicit seed — outputs go to outputs/balance/rl/seed42/
python scripts/train.py single --stage balance --steps 50000000 --seed 42

# Second seed for paper results (3 seeds total: 42, 113, 999)
python scripts/train.py single --stage balance --steps 50000000 --seed 113
python scripts/train.py single --stage balance --steps 50000000 --seed 999

# Stand-up / height-transition fine-tune (cross-stage warm-start from balance)
python scripts/train.py single --stage stand_up \
    --warm-start outputs/balance/rl/seed42/checkpoints/final \
    --steps 50000000 --seed 42

# Push-recovery fine-tune (cross-stage warm-start from stand_up)
python scripts/train.py single --stage balance_robust \
    --warm-start outputs/stand_up/rl/seed42/checkpoints/final \
    --steps 50000000 --seed 42

# Other stages
python scripts/train.py single --stage wheeled_locomotion --steps 5000000
python scripts/train.py single --stage walking           --steps 5000000
python scripts/train.py single --stage stair_climbing    --steps 5000000
python scripts/train.py single --stage rough_terrain     --steps 5000000
```

Outputs land in `outputs/<stage>/rl/seed<N>/`:
```
outputs/balance/rl/seed42/
├── checkpoints/final/checkpoint.pkl
├── balance_seed42_metrics.jsonl
└── run_metadata.json
```

Common options:

```bash
--num-envs 8192         # parallel environments (default 4096)
--output-dir my_outputs # output root (default: outputs/)
--seed 42               # random seed — determines output subdirectory
--warm-start <checkpoint> # cross-stage fine-tune; load weights/obs RMS only
--resume <checkpoint>   # exact same-run resume from a checkpoint directory
--additional-steps N    # when resuming, train N extra env-steps from checkpoint
```

#### Training with live viewer

```bash
python scripts/train.py single --stage balance --live-view
python scripts/train.py single --stage balance --live-view --view-interval 5
```

The viewer runs on the main thread; training runs in a background thread.
Closing the viewer window sets a stop flag on the trainer — training saves a checkpoint
and exits gracefully (the background thread is given up to 60 s to finish).

#### Full curriculum

```bash
python scripts/train.py curriculum --steps-per-stage 5000000
```

Stages run in order: Balance → Stand Up → Balance Robust.
Each stage warm-starts from the previous stage's checkpoint.

#### Cross-stage warm-start

Use `--warm-start` for staged fine-tuning (`balance → stand_up → balance_robust`).
It loads policy parameters and observation normalization from the source
checkpoint, then starts a fresh optimizer, RNG, environment state, and
destination-stage step counter. Use `--resume` only when continuing the exact
same stage/run.

#### Resume from checkpoint

```bash
python scripts/train.py single --stage balance --seed 42 \
    --resume outputs/balance/rl/seed42/checkpoints/final \
    --additional-steps 5000000
```

`--steps` is an absolute target total. If a checkpoint is already at step
1,000,000, `--steps 5000000` trains until roughly 5,000,000 total env-steps,
not 5,000,000 additional steps. Use `--additional-steps` for relative resume
training. Current checkpoints store policy parameters, optimizer state,
observation normalization, RNG, compact environment state, curriculum state,
and best-checkpoint trackers; older checkpoints still load but reset envs once.

Press **Ctrl+C** to stop at any time. The latest checkpoint is saved automatically.

### 3. Evaluate a trained policy

Four benchmark modes:

| Mode | Description | Key metrics |
|---|---|---|
| `nominal` | Standard env defaults | `reward_mean`, `fall_rate`, `success_rate` |
| `push_recovery` | Stronger / always-enabled push | `fall_after_push_rate` |
| `domain_randomized` | ±30% mass, ±50% friction | `height_error_mean`, `mass_perturb_pct`, `friction_perturb_pct` |
| `command_tracking` | Sweep target heights | `overall_height_rmse`, per-command RMSE |

```bash
python scripts/evaluate.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --stage balance \
  --mode nominal          # or push_recovery / domain_randomized / command_tracking
```

Results are saved to `outputs/balance/rl/seed42/checkpoints/final/eval_results_<mode>.json`.

**`success_rate` semantics:** A episode is a success if the env's own `time_limit` flag
fires (i.e., the robot survived the full episode) — *not* a comparison against the
benchmark's `max_steps`. `fall_rate + success_rate ≈ 1.0`.

### 4. Benchmark baseline comparison

```bash
# Save a baseline after training
mkdir -p baselines/
cp outputs/balance/rl/seed42/checkpoints/final/eval_results_nominal.json baselines/nominal_v1.json

# Compare a later run against the baseline (exit code 1 if regression)
python scripts/compare_baseline.py \
  --baseline baselines/nominal_v1.json \
  --current  outputs/balance/rl/seed42/checkpoints/final/eval_results_nominal.json
```

See [`docs/baseline_workflow.md`](docs/baseline_workflow.md) for the full workflow
and tolerance reference.

### 5. Standing validation (quality check)

Combines a nominal benchmark with a headless per-step rollout to surface
reward exploitation patterns that JSON metrics alone cannot detect.

```bash
# Default: clean obs (no sensor noise) — useful for debugging pure policy quality
python scripts/validate_checkpoint.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final

# Sim-to-real preparation: apply sensor noise from checkpoint config
python scripts/validate_checkpoint.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final --noise

# Custom height command and rollout length
python scripts/validate_checkpoint.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --height-cmd 0.65 --num-steps 1000 --save-csv
```

**Outputs** (written to the checkpoint directory by default):
`outputs/balance/rl/seed42/checkpoints/final/`

| File | Contents |
|---|---|
| `validation_report.json` | Benchmark metrics + all quality signals with WARN/OK |
| `telemetry_plot.png` | 6-panel per-step signal plot (position, orientation, velocity, joints, torques) |
| `telemetry.csv` | Raw per-step telemetry (only with `--save-csv`) |

**Quality signals checked and what they reveal:**

| Signal | Exploit / problem it catches |
|---|---|
| `wheel_spin_mean_rads` | Wheel-momentum balancing instead of posture |
| `height_std_m` | Vertical oscillation / height bouncing |
| `xy_drift_max_m` | Slow rolling / drifting while appearing stable |
| `roll/pitch_mean_abs_deg` | Chronic lean below the 46° termination threshold |
| `ctrl_jitter_mean_nm` | Chattering actuation (action_rate penalty too weak) |
| `leg_asymmetry_mean_rad` | Asymmetric crouching (one side lower than the other) |
| `ang_vel_rms_rads` | Chronic torso wobble below termination threshold |

Each WARN flag includes a plain-text description of the exploit pattern it suggests.

### 6. Research evaluation (paper metrics)

`scripts/eval_balance.py` produces per-step quantitative metrics
(pitch/roll RMS, torque RMS, drift, recovery time) across scenario groups
for paper tables and ablation studies.  It runs single-env CPU rollouts;
it is **not** the training-time curriculum eval (`PPOTrainer.eval_pass()`).

```bash
# Single checkpoint, all default scenarios
python scripts/eval_balance.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final

# Selected scenarios
python scripts/eval_balance.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --scenarios nominal --scenarios push_recovery \
  --scenarios friction_low --scenarios friction_high

# Push magnitude sweep (8 points: 20–200 N)
python scripts/eval_balance.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --scenarios push_sweep --num-episodes 10

# Friction sweep (6 points: 0.3×–1.8×)
python scripts/eval_balance.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --scenarios friction_sweep --num-episodes 10

# Paper evaluation: 3 seeds, 50 episodes each, results written to paper_eval/
python scripts/eval_balance.py \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --checkpoint outputs/balance/rl/seed113/checkpoints/final \
  --checkpoint outputs/balance/rl/seed999/checkpoints/final \
  --num-episodes 50 --seeds 0 --seeds 42 --seeds 123 \
  --output-dir outputs/balance/rl/paper_eval

# Classical LQR baseline (no checkpoint required)
python scripts/eval_balance.py \
  --controller baseline_lqr \
  --scenarios nominal --scenarios push_recovery \
  --num-episodes 20 \
  --output-dir outputs/balance/lqr
```

**Output files** (written to `--output-dir`, default = first checkpoint dir):

| File | Contents |
|---|---|
| `eval_results.csv` | All metrics, one row per checkpoint × scenario |
| `eval_results.json` | Structured results with full metadata |
| `summary_table.txt` | Paper-ready formatted summary table |

**Available scenarios:** `nominal`, `narrow_height`, `wide_height`, `full_range`,
`push_recovery`, `friction_low`, `friction_high`, `sensor_noise_delay`,
`push_sweep`, `friction_sweep`

---

### 7. Visualize a trained policy

```bash
# Watch policy in MuJoCo viewer
python scripts/visualize.py policy \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final

# Custom step count and slow-motion
python scripts/visualize.py policy \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --num-steps 5000 --slow-factor 2.0
```

### 8. Render video

```bash
python scripts/visualize.py render \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --output demo.mp4

# Custom resolution
python scripts/visualize.py render \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final \
  --output demo.mp4 --width 1920 --height 1080 --fps 60
```

### 9. Interactive keyboard control

```bash
python scripts/visualize.py interactive                                    # PD control only
python scripts/visualize.py interactive \
  --checkpoint outputs/balance/rl/seed42/checkpoints/final               # with policy
```

| Key | Function |
|---|---|
| ↑ / ↓ | Forward / backward |
| ← / → | Turn left / right |
| Q / E | Roll left / right |
| U / J | Increase / decrease target height |
| Space | Brake |
| [ / ] | Slow / fast |

### 10. Unified multi-skill controller

```bash
# Default: reads checkpoints from outputs/<stage>/rl/seed42/checkpoints/final/
python scripts/visualize.py unified
python scripts/visualize.py unified --checkpoint-dir outputs --no-auto-mode
```

Keys 1–6 force a specific skill; 0 returns to auto-detect. Skill switching uses
dwell-time hysteresis (3 consecutive frames) to avoid single-frame spurious transitions.

**Tilt detection** uses the gravity vector projected into the body frame
(`arccos(-g_body[2])`), which is yaw-invariant. A robot that only rotates around
the vertical axis will not falsely trigger balance fallback.

### 11. Run tests

```bash
# Fast test suite — CPU-safe, typically < 3 min (equivalent to CI)
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v

# Individual files
pytest tests/test_model.py -v              # MuJoCo model integrity
pytest tests/test_rewards.py -v            # Reward functions
pytest tests/test_ppo_trainer.py -v        # PPO invariants: no NaN, checkpoint round-trip, eval-gated curriculum
pytest tests/test_curriculum.py -v         # Curriculum promote/hold/demote
pytest tests/test_unified_controller.py -v # Tilt semantics, skill switching
pytest tests/test_benchmark.py -v          # Benchmark success/fall/timeout semantics
pytest tests/test_sim_helpers.py -v        # Push disturbance, PID control
pytest tests/test_noise_and_dr.py -v       # Sensor noise + per-episode DR
pytest tests/test_baseline.py -v           # Baseline comparison logic
pytest tests/test_standing_quality.py -v   # Standing quality signals (pure numpy, no JAX/MuJoCo)
pytest tests/test_validate_checkpoint.py -v # validate_checkpoint.py CLI smoke tests
pytest tests/test_latex_table.py -v        # LaTeX table generator
pytest tests/test_lqr_controller.py -v     # LQR balance controller (mocked MuJoCo)
pytest tests/test_eval_balance.py -v       # eval_balance.py data structures and scenario sweep

# End-to-end smoke test — verifies train() runs without error and produces a checkpoint
# Marked @pytest.mark.slow: 2–5 min on CPU (JAX JIT compile), < 30 s on GPU.
# Run explicitly; not included in the fast suite above.
pytest tests/test_smoke_train.py -v -m slow

# Full MJX rollout test (slow on CPU runners — excluded from CI)
pytest tests/test_env.py -v
```

---

## Quick reference

```bash
# ── View ──────────────────────────────────────────────────────────────────────
python scripts/visualize.py model
python scripts/visualize.py policy     --checkpoint outputs/balance/rl/seed42/checkpoints/final
python scripts/visualize.py render     --checkpoint outputs/balance/rl/seed42/checkpoints/final --output demo.mp4
python scripts/visualize.py interactive --checkpoint outputs/balance/rl/seed42/checkpoints/final
python scripts/visualize.py unified    --checkpoint-dir outputs

# ── Train ─────────────────────────────────────────────────────────────────────
python scripts/train.py single     --stage balance        --steps 5000000  --seed 42
python scripts/train.py single     --stage balance        --steps 50000000 --seed 113
python scripts/train.py single     --stage stand_up       --steps 50000000 --seed 42 \
    --warm-start outputs/balance/rl/seed42/checkpoints/final
python scripts/train.py single     --stage balance_robust --steps 50000000 --seed 42 \
    --warm-start outputs/stand_up/rl/seed42/checkpoints/final
python scripts/train.py single     --stage balance        --live-view
python scripts/train.py curriculum --steps-per-stage 5000000

# ── Resume ────────────────────────────────────────────────────────────────────
python scripts/train.py single --stage balance --seed 42 \
    --resume outputs/balance/rl/seed42/checkpoints/final \
    --additional-steps 5000000

# ── Evaluate (benchmark) ──────────────────────────────────────────────────────
python scripts/evaluate.py --checkpoint outputs/balance/rl/seed42/checkpoints/final --mode nominal
python scripts/evaluate.py --checkpoint outputs/balance/rl/seed42/checkpoints/final --mode push_recovery
python scripts/evaluate.py --checkpoint outputs/balance/rl/seed42/checkpoints/final --mode domain_randomized
python scripts/evaluate.py --checkpoint outputs/balance/rl/seed42/checkpoints/final --mode command_tracking
python scripts/validate_checkpoint.py --checkpoint outputs/balance/rl/seed42/checkpoints/final

# ── Research evaluation (paper metrics) ───────────────────────────────────────
python scripts/eval_balance.py --checkpoint outputs/balance/rl/seed42/checkpoints/final
python scripts/eval_balance.py \
    --checkpoint outputs/balance/rl/seed42/checkpoints/final \
    --checkpoint outputs/balance/rl/seed113/checkpoints/final \
    --checkpoint outputs/balance/rl/seed999/checkpoints/final \
    --scenarios nominal --scenarios push_recovery --scenarios push_sweep \
    --num-episodes 50 \
    --output-dir outputs/balance/rl/paper_eval
python scripts/eval_balance.py --controller baseline_lqr --scenarios nominal \
    --output-dir outputs/balance/lqr

# ── Baseline comparison ───────────────────────────────────────────────────────
# Requires baselines/nominal_v1.json from a previous nominal eval snapshot.
python scripts/compare_baseline.py --baseline baselines/nominal_v1.json \
    --current outputs/balance/rl/seed42/checkpoints/final/eval_results_nominal.json

# ── Tests ─────────────────────────────────────────────────────────────────────
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v     # fast suite
pytest tests/test_smoke_train.py -v -m slow                    # end-to-end smoke test

# ── Export (after training) ────────────────────────────────────────────────────
python scripts/export_results.py curves \
    outputs/balance/rl/seed42/balance_seed42_metrics.jsonl \
    --tags reward/mean curriculum/level curriculum/eval_per_step \
    --output outputs/balance/rl/seed42/training_curves.png
python scripts/export_results.py table \
    outputs/balance/rl/seed42/checkpoints/final/eval_results_command_tracking.json \
    --output outputs/balance/rl/seed42/tables/height_tracking.md
python scripts/export_results.py latex \
    outputs/balance/rl/paper_eval/eval_results.json \
    --output outputs/tables/balance_eval.tex

# ── LQR baseline evaluation ───────────────────────────────────────────────────
python scripts/eval_balance.py --controller baseline_lqr \
    --scenarios nominal --scenarios push_recovery \
    --scenarios friction_low --scenarios friction_high \
    --num-episodes 20 --output-dir outputs/balance/lqr
```

---

## Architecture

### Observation space (42 dims for BalanceEnv)

| Component | Dims | Description |
|---|---|---|
| Gravity in body frame | 3 | Body tilt — yaw-invariant tilt detection |
| Linear velocity | 3 | Body-frame linear velocity (simulator-exact in `clean` mode) |
| Angular velocity | 3 | Body-frame angular velocity |
| Joint positions | 10 | All 10 joint angles (encoders) |
| Joint velocities | 10 | All 10 joint velocities |
| Previous action | 10 | Last control output (policy's intended target, pre-delay) |
| Height command | 1 | Target height, normalised to [0, 1] over [0.40, 0.70] m |
| Current height | 1 | Current torso height, normalised to [0, 1] over [0.40, 0.70] m |
| Yaw error | 1 | Heading drift from episode start, wrapped to [−π, π] |

> Base 39-dim observation (`lin_vel_mode="clean"`) is shared across skills.
> `height_command`, `current_height`, and `yaw_error` are BalanceEnv extensions
> (dims 40–42).
> With `lin_vel_mode="disabled"` the base shrinks to 36 dims → total 39.
> The unified controller provides per-skill observation adapters with explicit
> schemas; generic zero-padding is an escape hatch that logs a warning.

`StandUpEnv` and `balance_robust` keep the same BalanceEnv-compatible 42-dim
observation contract and PID action semantics, so checkpoints can be
cross-stage warm-started without changing the policy architecture. The
commanded height range is `[0.40, 0.70]` m; stand-up reset poses may start
slightly outside this range as initial-state perturbations.

### Action space (10 dims, normalised to [−1, 1])

With `low_level_pid.enabled: true` (default in `balance.yaml`):
- Joints 0–3, 5–8 (hip/knee): policy output is interpreted as a **position target**,
  converted to torque by a PID controller.
- Joints 4, 9 (wheels): policy output is a **velocity target** (scaled by `wheel_vel_limit`).

Without PID: output is scaled directly to actuator torque range.

### PPO training

- Vectorised rollout using `jax.vmap` over 4096 parallel MJX environments.
- Observation running mean/std normalisation (Welford online update).
- GAE advantage estimation with γ=0.99, λ=0.95.
- Checkpoints save full same-run resume state: policy parameters, optimizer
  state, observation normalization, RNG, compact `EnvState`, curriculum state,
  and best-checkpoint trackers. Cross-stage curriculum warm-starts intentionally
  reset optimizer, training counters, and env state while reusing learned weights.
- Optional destructive-update guard: `ppo.max_policy_kl > 0` rejects a PPO
  update whose measured approximate KL exceeds the configured limit.

Two curriculum systems operate independently:

**Within-stage height curriculum** (`PPOTrainer`, balance only): expands
`curriculum_min_height` from 0.69 → 0.40 m over 29 levels. Default mode
(`use_eval_signal: true` in `balance.yaml`): on the default GPU config,
`eval_pass()` is called every 2 updates using the greedy policy (32 envs × 200 episodes,
obs_rms frozen). With `num_envs=4096 × rollout_length=128 = 524,288 steps/update`,
this fires the first eval at ~1 M env-steps — compatible with a 5 M-step run.
Advancement fires when `eval_reward_mean / episode_length >= reward_threshold`.
In `balance.yaml`, `reward_threshold: 0.65` means 65% of the positive reward
budget, i.e. about `6.825` reward/step from a `10.5` reward/step maximum.
Progress is logged as `curriculum/eval_per_step`, `curriculum/eval_success_rate`,
and `curriculum/eval_fall_rate`. Backward-compatible fallback (`use_eval_signal:
false`): rolling window of per-update `avg_reward` from training rollouts (noisier
signal, original behavior).

**Multi-stage pipeline** (`CurriculumManager`, `configs/curriculum.yaml`): gates stage
promotion/hold/demotion on `eval_reward_mean` from the end-of-training `eval_pass()`
call. Falls back to `best_reward` for checkpoints from older trainers.

### Balance reward design

Two training configs serve different objectives:

| Term | `balance.yaml` | `balance_robust.yaml` | Note |
|---|---|---|---|
| `no_motion` | 0.5 | **0.0** | Wheels must spin to recover from push |
| `wheel_velocity` | −0.002 | **0.0** | Wheels are primary balancing actuators under push |
| `action_rate` | −0.05 | **−0.005** | Rapid wheel burst needed immediately after impact |
| `body_level` | 1.5 | 1.5 | Unchanged — consistent standing objective |
| `height` | 2.5 | 1.5 | Reduced — push survival outweighs strict height tracking |
| `natural_pose` | 0.4 | 1.5 | Increased — stronger return-to-stance after recovery |

Both configs share `push_magnitude=0` (balance) or `40 N` (robust).

### Curriculum manager

`configs/curriculum.yaml` defines the pipeline. `CurriculumManager` in
`wheeled_biped/training/curriculum.py` gates stage promotion / hold / demotion using
`eval_reward_mean` from the trainer. The `_evaluate_promotion` method compares the
metric against the stage's `success_value` threshold over a sliding `promotion_window`.

### CI

GitHub Actions workflow at `.github/workflows/ci.yml`:
- **Ruff** lint + format check
- **Pytest** over 15 of 16 test files (excludes `test_env.py` — full MJX JIT compile is
  prohibitively slow on free runners). `test_smoke_train.py` is collected but its tests
  are marked `@pytest.mark.slow`; run them locally with `pytest tests/test_smoke_train.py -m slow`.
- CPU-only JAX (`jax[cpu]`) installed explicitly before the project package to avoid
  pulling in the cuda12 variant

---

## Configuration

All hyperparameters are in `configs/`:

- `configs/robot.yaml` — robot physics parameters
- `configs/curriculum.yaml` — multi-stage pipeline definition
- `configs/training/<stage>.yaml` — per-task hyperparameters

Key knobs:

| Parameter | Location | Effect |
|---|---|---|
| `task.num_envs` | `<stage>.yaml` | Parallel environments (GPU memory bound) |
| `ppo.learning_rate` | `<stage>.yaml` | Step size |
| `ppo.max_policy_kl` | `<stage>.yaml` | Reject overly large PPO policy updates (`0` disables) |
| `rewards.*` | `<stage>.yaml` | Per-component reward weights |
| `domain_randomization.push_magnitude` | `<stage>.yaml` | Push force (0 = disabled) |
| `termination.max_tilt_rad` | `<stage>.yaml` | Fall threshold (~0.8 rad = 46°) |
| `low_level_pid.enabled` | `<stage>.yaml` | PID low-level control mode |
| `sensor_noise.enabled` | `<stage>.yaml` | Enable Gaussian obs noise (sim-to-real realism) |
| `sensor_noise.ang_vel_std` | `<stage>.yaml` | Angular velocity noise std (rad/s) |
| `sensor_noise.joint_pos_std` | `<stage>.yaml` | Joint position noise std (rad) |

---

## Known limitations and gaps

- **Only `balance` has been trained.** All other stages are config-ready but untrained.
- **Sensor noise is conservative.** Gaussian noise is applied to IMU (angular velocity, gravity vector) and encoder (joint position, velocity) observations during training. The default standard deviations (`ang_vel_std: 0.05 rad/s`, `gravity_std: 0.02`, `joint_pos_std: 0.005 rad`, `joint_vel_std: 0.01 rad/s`) are initial estimates; hardware-calibrated values have not been validated. Disable with `sensor_noise.enabled: false`.
- **Sim-to-real not validated.** Domain randomisation and push disturbance are the only
  transfer bridges; no hardware tests have been performed.
- **`test_env.py` excluded from CI.** The full MJX rollout JIT-compiles to GPU code;
  compile time is impractical on CPU-only GitHub runners.
- **Unified controller requires all checkpoints.** Missing skill checkpoints are
  skipped gracefully, but the controller has only been tested in simulation.
- **Height curriculum is env-internal.** The `curriculum_min_height` value is carried
  through episodes via `EnvState.info`, not managed by `CurriculumManager`. The two
  curriculum systems are independent. Within-stage advancement uses an eval-gated
  `eval_pass()` signal by default; see Architecture > PPO training for details.

---

## Recommended pre-training workflow

Before starting a long balance training run, complete these steps in order:

```bash
# 1. Verify the stack compiles and produces a checkpoint on your hardware
pytest tests/test_smoke_train.py -v -m slow
#    Expected: 8 tests pass; checkpoint written to tmp dir; ~30 s on GPU, ~2-5 min on CPU.

# 2. Run the fast unit test suite to catch any regressions
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v

# 3. Confirm the robot model loads cleanly
python scripts/visualize.py model
```

Once a checkpoint exists, optionally validate it before continuing training:

```bash
python scripts/validate_checkpoint.py --checkpoint outputs/balance/rl/seed42/checkpoints/final
```

---

## Paper artifact generation

### Building the paper PDF

The manuscript is in `paper/main.tex` and uses an **embedded bibliography**
(`\begin{thebibliography}`) — no external `.bib` file or BibTeX/biber step is needed.
Build with a single `pdflatex` pass (or two passes for correct cross-references):

```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # second pass resolves \ref and \cite labels
```

Required LaTeX packages: `IEEEtran` document class, `amsmath`, `graphicx`, `booktabs`,
`siunitx`, `url`, `xcolor`, `cite`, `multirow`, `array`, `balance`.
All are included in a standard TeX Live / MiKTeX full installation.

> **Note:** `paper/refs.bib.backup` is a BibTeX-format backup of the embedded citations
> kept for reference. It is **not** loaded by the paper build.

### Exporting training results

After a successful run, use `scripts/export_results.py` to produce paper-ready outputs
from the training log and benchmark JSON without additional dependencies (matplotlib is
optional for PNG figures).

The training logger writes to `outputs/<stage>/rl/seed<seed>/<stage>_seed<seed>_metrics.jsonl`
(e.g. `outputs/balance/rl/seed42/balance_seed42_metrics.jsonl` for `--stage balance --seed 42`).

Three sub-commands: `curves`, `table`, `latex`.

```bash
# Training curves → CSV + PNG (per seed)
python scripts/export_results.py curves \
    outputs/balance/rl/seed42/balance_seed42_metrics.jsonl \
    --tags reward/mean curriculum/level curriculum/eval_per_step \
    --output outputs/balance/rl/seed42/training_curves.png

# CSV only (skip PNG, e.g. on a headless server without matplotlib)
python scripts/export_results.py curves \
    outputs/balance/rl/seed42/balance_seed42_metrics.jsonl \
    --no-plot

# Benchmark table → Markdown (from evaluate.py output)
python scripts/export_results.py table \
    outputs/balance/rl/seed42/checkpoints/final/eval_results_command_tracking.json \
    --output outputs/balance/rl/seed42/tables/height_tracking.md

# Print to stdout (no --output)
python scripts/export_results.py table \
    outputs/balance/rl/seed42/checkpoints/final/eval_results_nominal.json

# Research eval → LaTeX booktabs table (from eval_balance.py output over 3 seeds)
python scripts/export_results.py latex \
    outputs/balance/rl/paper_eval/eval_results.json \
    --output outputs/tables/balance_eval.tex \
    --caption "Balance evaluation results." \
    --label "tab:balance_eval"
```

Output files written alongside the source log/JSON by default when `--output` is omitted.

### 3-seed experiment protocol

Final paper results should be reported over 3 independent seeds (42, 113, 999):

```bash
# 1. Train all three seeds
python scripts/train.py single --stage balance --steps 50000000 --seed 42
python scripts/train.py single --stage balance --steps 50000000 --seed 113
python scripts/train.py single --stage balance --steps 50000000 --seed 999

# 2. Evaluate all three seeds
python scripts/eval_balance.py \
    --checkpoint outputs/balance/rl/seed42/checkpoints/final \
    --checkpoint outputs/balance/rl/seed113/checkpoints/final \
    --checkpoint outputs/balance/rl/seed999/checkpoints/final \
    --num-episodes 50 --num-steps 2000 --seeds 0 --seeds 42 --seeds 123 \
    --output-dir outputs/balance/rl/paper_eval

# 3. Evaluate LQR baseline
python scripts/eval_balance.py \
    --controller baseline_lqr \
    --scenarios nominal --scenarios push_recovery \
    --scenarios friction_low --scenarios friction_high \
    --num-episodes 50 --output-dir outputs/balance/lqr

# 4. Export to LaTeX table
python scripts/export_results.py latex \
    outputs/balance/rl/paper_eval/eval_results.json \
    --output outputs/tables/balance_eval.tex
```

Each seed is a fully independent training run with its own RNG state. Results are
aggregated (mean ± std) post-hoc; the three runs are never mixed during training.

---

## License

MIT
