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
│   │   └── baseline.py                # Baseline save / comparison utilities
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
│   └── compare_baseline.py            # Regression comparison CLI
├── tests/                             # 10 test files (pytest)
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
# Pure standing balance (no push disturbances — best starting point)
python scripts/train.py single --stage balance --steps 5000000

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

The viewer runs on the main thread; training continues in a background thread.
Close the viewer window at any time — training keeps running.

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
| `domain_randomized` | ±15% mass, ±40% friction | `height_error_mean`, `position_drift_mean` |
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
cp outputs/checkpoints/balance/final/eval_results_nominal.json baselines/nominal_v1.json

# Compare a later run against the baseline (exit code 1 if regression)
python scripts/compare_baseline.py \
  --baseline baselines/nominal_v1.json \
  --current  outputs/checkpoints/balance/final/eval_results_nominal.json
```

See [`docs/baseline_workflow.md`](docs/baseline_workflow.md) for the full workflow
and tolerance reference.

### 5. Visualize a trained policy

```bash
# Watch policy in MuJoCo viewer
python scripts/visualize.py policy \
  --checkpoint outputs/checkpoints/balance/final

# Custom step count and slow-motion
python scripts/visualize.py policy \
  --checkpoint outputs/checkpoints/balance/final \
  --num-steps 5000 --slow-factor 2.0
```

### 6. Render video

```bash
python scripts/visualize.py render \
  --checkpoint outputs/checkpoints/balance/final \
  --output demo.mp4

# Custom resolution
python scripts/visualize.py render \
  --checkpoint outputs/checkpoints/balance/final \
  --output demo.mp4 --width 1920 --height 1080 --fps 60
```

### 7. Interactive keyboard control

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

### 8. Unified multi-skill controller

```bash
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints --no-auto-mode
```

Keys 1–6 force a specific skill; 0 returns to auto-detect. Skill switching uses
dwell-time hysteresis (3 consecutive frames) to avoid single-frame spurious transitions.

**Tilt detection** uses the gravity vector projected into the body frame
(`arccos(-g_body[2])`), which is yaw-invariant. A robot that only rotates around
the vertical axis will not falsely trigger balance fallback.

### 9. Run tests

```bash
# Full CPU-safe test suite (used in CI)
pytest tests/ --ignore=tests/test_env.py -v

# Individual files
pytest tests/test_model.py -v             # MuJoCo model integrity
pytest tests/test_rewards.py -v           # Reward functions
pytest tests/test_ppo_trainer.py -v       # PPO invariants (no NaN, checkpoint round-trip)
pytest tests/test_curriculum.py -v        # Curriculum promote/hold/demote
pytest tests/test_unified_controller.py -v # Tilt semantics, skill switching
pytest tests/test_benchmark.py -v         # Benchmark success/fall/timeout semantics
pytest tests/test_sim_helpers.py -v       # Push disturbance, PID control
pytest tests/test_baseline.py -v          # Baseline comparison logic

# test_env.py is excluded from CI (full MJX rollout, slow on CPU runners)
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

# ── Baseline comparison ───────────────────────────────────────────────────────
python scripts/compare_baseline.py --baseline baselines/nominal_v1.json \
                                   --current  outputs/.../eval_results_nominal.json

# ── Tests ─────────────────────────────────────────────────────────────────────
pytest tests/ --ignore=tests/test_env.py -v
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
- Curriculum progression driven by `eval_reward_mean` — the mean of the last 50
  training update rewards (smoothed, not the all-time-max `best_reward`). Falls back
  to `best_reward` for checkpoints from older trainers.

### Balance reward design

Two training configs serve different objectives:

| Term | `balance.yaml` | `balance_robust.yaml` | Note |
|---|---|---|---|
| `no_motion` | 0.5 | **0.0** | Wheels must spin to recover from push |
| `wheel_velocity` | −0.0006 | **0.0** | Wheels are primary balancing actuators under push |
| `action_rate` | −0.025 | **−0.005** | Rapid wheel burst needed immediately after impact |
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
- **Pytest** over 7 of 9 test files (excludes `test_env.py` — full MJX JIT compile is
  prohibitively slow on free runners)
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

---

## Known limitations and gaps

- **Only `balance` has been trained.** All other stages are config-ready but untrained.
- **No sensor noise.** IMU and encoder noise are not modelled in simulation.
- **Sim-to-real not validated.** Domain randomisation and push disturbance are the only
  transfer bridges; no hardware tests have been performed.
- **`test_env.py` excluded from CI.** The full MJX rollout JIT-compiles to GPU code;
  compile time is impractical on CPU-only GitHub runners.
- **Unified controller requires all checkpoints.** Missing skill checkpoints are
  skipped gracefully, but the controller has only been tested in simulation.
- **Height curriculum is env-internal.** The `curriculum_min_height` value is carried
  through episodes via `EnvState.info`, not managed by `CurriculumManager`. The two
  curriculum systems are independent.

---

## License

MIT
