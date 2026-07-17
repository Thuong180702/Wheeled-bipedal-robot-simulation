# macOS Setup Guide

## Prerequisites

- macOS 15+ on Apple Silicon (ARM64)
- Internet connection for downloads

## Step 1 — Install Miniforge (conda-forge)

This provides Python without needing Homebrew or sudo access:

```bash
curl -fsSL -o /tmp/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash /tmp/miniforge.sh -b -p ~/miniforge3
```

## Step 2 — Create Python 3.10 conda environment

The repo expects Python 3.10. Python 3.13 has incompatibilities with `chex.dataclass` (stricter mutable default handling in dataclasses).

```bash
~/miniforge3/bin/conda create -n wheeled-biped python=3.10 -y
```

## Step 3 — Create virtual environment

```bash
cd /path/to/Wheeled-bipedal-robot-simulation
rm -rf .venv
~/miniforge3/envs/wheeled-biped/bin/python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

## Step 4 — Install macOS-safe dependencies

**Do NOT run `pip install -r requirements.txt`** — it includes `jax[cuda12]` which is for NVIDIA/CUDA and will fail on macOS.

Install the macOS-safe packages:

```bash
# JAX (CPU-only)
pip install -U jax jaxlib

# MuJoCo + MJX (MJX is a separate package in MuJoCo >= 3.2.0)
pip install "mujoco>=3.1.0" "mujoco-mjx>=3.0.0"

# JAX ecosystem
pip install "flax>=0.8.0" "optax>=0.2.0" "chex>=0.1.86" "distrax>=0.1.5" "jaxopt>=0.8.0,<0.9.0"

# Utilities
pip install "numpy>=1.26.0" "pyyaml>=6.0" "tqdm>=4.66.0" "typer>=0.9.0" "rich>=13.0.0"
pip install "tensorboardX>=2.6" "wandb>=0.16.0" "matplotlib>=3.8.0" "mediapy>=1.2.0"
pip install "scipy>=1.11" "osqp>=0.6"
pip install "pandas>=2.0" "tabulate>=0.9.0"

# Testing
pip install "pytest>=7.4.0" "pytest-timeout>=2.3.0" "pytest-xdist>=3.5.0" "ruff>=0.3.0"

# Install this repo in editable mode
pip install -e . --no-deps
```

## Step 5 — Verify imports

```bash
python -c "import jax; print('jax', jax.__version__, jax.default_backend(), jax.devices())"
python -c "import mujoco; print('mujoco', mujoco.__version__)"
python -c "import numpy, scipy, osqp; print('numpy/scipy/osqp ok')"
python -c "import flax, optax, chex, distrax, jaxopt; print('jax ecosystem ok')"
python -c "import wheeled_biped; print('wheeled_biped import ok')"
```

Expected output:
```
jax <version> cpu [CpuDevice(id=0)]
mujoco <version>
numpy/scipy/osqp ok
jax ecosystem ok
wheeled_biped import ok
```

## Step 6 — Run tests (CPU mode)

Set environment variables for CPU-safe testing:

```bash
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"

python -m pytest tests/ \
  --ignore=tests/test_env.py \
  -m "not slow" \
  -v --tb=short
```

## Step 7 — Verify MuJoCo model loads

```bash
python - <<'PY'
from wheeled_biped.utils.config import get_model_path
import mujoco
path = get_model_path()
model = mujoco.MjModel.from_xml_path(str(path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
print("MuJoCo model loaded: nq=%d, nv=%d, nu=%d" % (model.nq, model.nv, model.nu))
PY
```

Expected: model loads with nq=17, nv=16, nu=10.

## Step 8 — View model (optional, requires display)

```bash
python scripts/visualize.py model
```

If this fails with a display/window error, it's a GUI environment issue, not a repo setup failure.

## Step 9 — Phase 3D/WBC imports

```bash
python - <<'PY'
import scipy, osqp
import wheeled_biped.wbc.offline_qp_wbc
import wheeled_biped.wbc.phase3d3_incremental_qp
print("WBC/Phase 3D imports ok")
PY
```

## Known issues

### Python 3.13 incompatibility

Python 3.13 has stricter `dataclass` checking that rejects JAX array defaults in
`chex.dataclass(frozen=True)` decorators. The fix is to use Python 3.10 instead.

Error: `ValueError: mutable default <class 'jaxlib._jax.ArrayImpl'> for field base_quat is not allowed: use default_factory`

### MuJoCo MJX separation

In MuJoCo >= 3.2.0, `mjx` was moved from `mujoco.mjx` to a separate `mujoco-mjx` package.
If you see `ImportError: cannot import name 'mjx' from 'mujoco'`, install `mujoco-mjx`.

## Requirements file

Use `requirements-macos.txt` (included in this repo) for future installs:

```bash
pip install -r requirements-macos.txt
pip install -e . --no-deps
```
