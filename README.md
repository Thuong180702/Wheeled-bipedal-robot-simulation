# Wheeled Bipedal Robot Simulation

Simulation and reinforcement learning training for a wheeled bipedal robot using **MuJoCo MJX + JAX**, with PPO and curriculum learning.

> **Status:** Active research prototype. Only the `balance` stage has been trained and
> evaluated to date. Sim-to-real transfer has not been validated on hardware.

---

## Overview

| Task               | Description                                    | Status       |
|--------------------|------------------------------------------------|--------------|
| **Balance**        | Stand upright, hold target height, resist push | Implemented  |
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
│   │   ├── wheeled_biped.xml          # MuJoCo MJCF (primary model)
│   │   └── wheeled_biped_real.xml     # MJCF converted from URDF
│   └── robot-urdf/
│       ├── urdf/HOANTHIEN_TEST.urdf
│       ├── meshes/*.STL               # 11 STL mesh files
│       └── config/                    # ROS joint config
├── configs/
│   ├── robot.yaml                     # Robot parameters
│   ├── curriculum.yaml                # Multi-stage curriculum pipeline
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
├── wheeled_biped/                     # Main package
│   ├── envs/
│   │   ├── base_env.py                # MJX base environment
│   │   ├── balance_env.py             # Balance task (40-dim obs)
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
│   ├── visualize.py                   # MuJoCo viewer / video / interactive
│   ├── validate_checkpoint.py         # Standing validation (benchmark + posture quality)
│   ├── compare_baseline.py            # Regression comparison CLI
│   └── export_results.py              # Export training logs → CSV/PNG; eval JSON → Markdown table
├── tests/                             # 12 test files (pytest)
├── .github/workflows/ci.yml           # GitHub Actions CI
├── pyproject.toml
└── requirements.txt
```

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

# Pure standing balance — curriculum run (50M steps advances through all 29 height levels)
# With num_envs=4096×rollout=128=524K steps/update and eval_interval=2,
# the height curriculum can advance every ~1M steps.
python scripts/train.py single --stage balance --steps 50000000

# Push-recovery fine-tune (warm-start from balance checkpoint)
python scripts/train.py single --stage balance_robust --steps 3000000

# Other stages
python scripts/train.py single --stage wheeled_locomotion --steps 5000000
python scripts/train.py single --stage walking           --steps 5000000
python scripts/train.py single --stage stair_climbing    --steps 5000000
python scripts/train.py single --stage rough_terrain     --steps 5000000
python scripts/train.py single --stage stand_up          --steps 5000000
```

Common options:

```bash
--num-envs 8192         # parallel environments (default 4096)
--output-dir my_outputs # output directory (default: outputs/)
--seed 123              # random seed
--resume <checkpoint>   # resume from a saved checkpoint
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

Stages run in order: Balance → Wheeled Locomotion → Walking → Stair Climbing → Rough Terrain.
Each stage warm-starts from the previous stage's checkpoint.

#### Resume from checkpoint

```bash
python scripts/train.py single --stage balance --resume outputs/checkpoints/balance/step_1000000
```

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
  --checkpoint outputs/checkpoints/balance/final \
  --stage balance \
  --mode nominal          # or push_recovery / domain_randomized / command_tracking
```

Results are saved to `<checkpoint>/eval_results_<mode>.json`.

**`success_rate` semantics:** A episode is a success if the env's own `time_limit` flag
fires (i.e., the robot survived the full episode) — *not* a comparison against the
benchmark's `max_steps`. `fall_rate + success_rate ≈ 1.0`.

### 4. Benchmark baseline comparison

```bash
# Save a baseline after training
mkdir -p baselines/
cp outputs/checkpoints/balance/final/eval_results_nominal.json baselines/nominal_v1.json

# Compare a later run against the baseline (exit code 1 if regression)
python scripts/compare_baseline.py \
  --baseline baselines/nominal_v1.json \
  --current  outputs/checkpoints/balance/final/eval_results_nominal.json
```

See [`docs/baseline_workflow.md`](docs/baseline_workflow.md) for the full workflow
and tolerance reference.

### 5. Standing validation (quality check)

Combines a nominal benchmark with a headless per-step rollout to surface
reward exploitation patterns that JSON metrics alone cannot detect.

```bash
python scripts/validate_checkpoint.py \
  --checkpoint outputs/checkpoints/balance/final

# Custom height command and rollout length
python scripts/validate_checkpoint.py \
  --checkpoint outputs/checkpoints/balance/final \
  --height-cmd 0.65 --num-steps 1000 --save-csv
```

**Outputs** (written to the checkpoint directory by default):

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

---

### 6. Visualize a trained policy

```bash
# Watch policy in MuJoCo viewer
python scripts/visualize.py policy \
  --checkpoint outputs/checkpoints/balance/final

# Custom step count and slow-motion
python scripts/visualize.py policy \
  --checkpoint outputs/checkpoints/balance/final \
  --num-steps 5000 --slow-factor 2.0
```

### 7. Render video

```bash
python scripts/visualize.py render \
  --checkpoint outputs/checkpoints/balance/final \
  --output demo.mp4

# Custom resolution
python scripts/visualize.py render \
  --checkpoint outputs/checkpoints/balance/final \
  --output demo.mp4 --width 1920 --height 1080 --fps 60
```

### 8. Interactive keyboard control

```bash
python scripts/visualize.py interactive                                    # PD control only
python scripts/visualize.py interactive --checkpoint .../balance/final    # with policy
```

| Key | Function |
|---|---|
| ↑ / ↓ | Forward / backward |
| ← / → | Turn left / right |
| Q / E | Roll left / right |
| U / J | Increase / decrease target height |
| Space | Brake |
| [ / ] | Slow / fast |

### 9. Unified multi-skill controller

```bash
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints --no-auto-mode
```

Keys 1–6 force a specific skill; 0 returns to auto-detect. Skill switching uses
dwell-time hysteresis (3 consecutive frames) to avoid single-frame spurious transitions.

**Tilt detection** uses the gravity vector projected into the body frame
(`arccos(-g_body[2])`), which is yaw-invariant. A robot that only rotates around
the vertical axis will not falsely trigger balance fallback.

### 10. Run tests

```bash
# Fast test suite — CPU-safe, typically < 3 min (equivalent to CI)
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v

# Individual files
pytest tests/test_model.py -v             # MuJoCo model integrity
pytest tests/test_rewards.py -v           # Reward functions
pytest tests/test_ppo_trainer.py -v       # PPO invariants: no NaN, checkpoint round-trip, eval-gated curriculum
pytest tests/test_curriculum.py -v        # Curriculum promote/hold/demote
pytest tests/test_unified_controller.py -v # Tilt semantics, skill switching
pytest tests/test_benchmark.py -v         # Benchmark success/fall/timeout semantics
pytest tests/test_sim_helpers.py -v       # Push disturbance, PID control
pytest tests/test_noise_and_dr.py -v      # Sensor noise + per-episode DR
pytest tests/test_baseline.py -v          # Baseline comparison logic
pytest tests/test_standing_quality.py -v  # Standing quality signals (pure numpy, no JAX/MuJoCo)

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
python scripts/visualize.py policy     --checkpoint outputs/checkpoints/balance/final
python scripts/visualize.py render     --checkpoint outputs/checkpoints/balance/final --output demo.mp4
python scripts/visualize.py interactive --checkpoint outputs/checkpoints/balance/final
python scripts/visualize.py unified    --checkpoint-dir outputs/checkpoints

# ── Train ─────────────────────────────────────────────────────────────────────
python scripts/train.py single     --stage balance        --steps 5000000
python scripts/train.py single     --stage balance_robust --steps 3000000
python scripts/train.py single     --stage balance        --live-view
python scripts/train.py curriculum --steps-per-stage 5000000

# ── Evaluate ──────────────────────────────────────────────────────────────────
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --mode nominal
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --mode push_recovery
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --mode domain_randomized
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --mode command_tracking
python scripts/validate_checkpoint.py --checkpoint outputs/checkpoints/balance/final

# ── Baseline comparison ───────────────────────────────────────────────────────
python scripts/compare_baseline.py --baseline baselines/nominal_v1.json \
                                   --current  outputs/.../eval_results_nominal.json

# ── Tests ─────────────────────────────────────────────────────────────────────
pytest tests/ --ignore=tests/test_env.py -m "not slow" -v     # fast suite
pytest tests/test_smoke_train.py -v -m slow                    # end-to-end smoke test

# ── Export (after training) ────────────────────────────────────────────────────
python scripts/export_results.py curves \
    outputs/logs/balance_seed42_metrics.jsonl \
    --tags reward/mean curriculum/level curriculum/eval_per_step
python scripts/export_results.py table \
    outputs/checkpoints/balance/final/eval_results_command_tracking.json \
    --output outputs/tables/height_tracking.md
```

---

## Architecture

### Observation space (40 dims for BalanceEnv)

| Component | Dims | Description |
|---|---|---|
| Gravity in body frame | 3 | Body tilt — yaw-invariant tilt detection |
| Linear velocity | 3 | Body-frame linear velocity |
| Angular velocity | 3 | Body-frame angular velocity |
| Joint positions | 10 | All 10 joint angles (encoders) |
| Joint velocities | 10 | All 10 joint velocities |
| Previous action | 10 | Last control output |
| Height command | 1 | Target height, normalised to [0, 1] |

> Base 39-dim observation is shared across skills; `height_command` is the BalanceEnv
> extension. The unified controller provides per-skill observation adapters —
> generic zero-padding is an explicit fallback that logs a warning.

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

Two curriculum systems operate independently:

**Within-stage height curriculum** (`PPOTrainer`, balance only): expands
`curriculum_min_height` from 0.69 → 0.40 m over 29 levels. Default mode
(`use_eval_signal: true` in `balance.yaml`): `eval_pass()` is called every
`eval_interval=2` updates using the greedy policy (32 envs × 200 episodes,
obs_rms frozen). With `num_envs=4096 × rollout_length=128 = 524,288 steps/update`,
this fires the first eval at ~1 M env-steps — compatible with a 5 M-step run.
Advancement fires when `eval_reward_mean / episode_length >= reward_threshold`.
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
| `wheel_velocity` | −0.005 | **0.0** | Wheels are primary balancing actuators under push |
| `action_rate` | −0.05 | **−0.005** | Rapid wheel burst needed immediately after impact |
| `height`, `body_level`, `natural_pose` | shared | shared | Consistent standing objective |

Both configs share `push_magnitude=0` (balance) or `40 N` (robust).

### Curriculum manager

`configs/curriculum.yaml` defines the pipeline. `CurriculumManager` in
`wheeled_biped/training/curriculum.py` gates stage promotion / hold / demotion using
`eval_reward_mean` from the trainer. The `_evaluate_promotion` method compares the
metric against the stage's `success_value` threshold over a sliding `promotion_window`.

### CI

GitHub Actions workflow at `.github/workflows/ci.yml`:
- **Ruff** lint + format check
- **Pytest** over 11 of 12 test files (excludes `test_env.py` — full MJX JIT compile is
  prohibitively slow on free runners). `test_smoke_train.py` is collected but its tests
  are marked `@pytest.mark.slow`; run them locally with `pytest tests/test_smoke_train.py -m slow`.
- CPU-only JAX (`jax[cpu]`) — `jax[cuda12]` override applied before package install

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
python scripts/validate_checkpoint.py --checkpoint outputs/checkpoints/balance/final
```

---

## Paper artifact generation

After a successful run, use `scripts/export_results.py` to produce paper-ready outputs
from the training log and benchmark JSON without additional dependencies (matplotlib is
optional for PNG figures).

The training logger writes to `outputs/logs/<stage>_seed<seed>_metrics.jsonl`
(e.g. `balance_seed42_metrics.jsonl` for `--stage balance --seed 42`).

```bash
# Training curves → CSV + PNG
python scripts/export_results.py curves \
    outputs/logs/balance_seed42_metrics.jsonl \
    --tags reward/mean curriculum/level curriculum/eval_per_step \
    --output outputs/figures/training_curves.png

# CSV only (skip PNG, e.g. on a headless server without matplotlib)
python scripts/export_results.py curves \
    outputs/logs/balance_seed42_metrics.jsonl \
    --no-plot

# Benchmark table → Markdown
python scripts/export_results.py table \
    outputs/checkpoints/balance/final/eval_results_command_tracking.json \
    --output outputs/tables/height_tracking.md

# Print to stdout (no --output)
python scripts/export_results.py table \
    outputs/checkpoints/balance/final/eval_results_nominal.json
```

Output files written alongside the source log/JSON by default when `--output` is omitted.

---

## License

MIT
