"""
Đọc và quản lý cấu hình YAML cho dự án.

Hỗ trợ merge giữa config mặc định và config người dùng.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Đọc file YAML và trả về dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Merge sâu: override ghi đè lên base.

    Args:
        base: dict cơ sở.
        override: dict ghi đè.

    Returns:
        Dict đã merge.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_training_config(task_config_path: str | Path) -> dict[str, Any]:
    """Đọc config training, merge với robot config.

    Args:
        task_config_path: đường dẫn tới file config task (vd: balance.yaml).

    Returns:
        Dict config hoàn chỉnh bao gồm cả thông số robot.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    robot_cfg = load_yaml(project_root / "configs" / "robot.yaml")
    task_cfg = load_yaml(task_config_path)
    return deep_merge(robot_cfg, task_cfg)


def get_project_root() -> Path:
    """Trả về đường dẫn gốc của dự án."""
    return Path(__file__).resolve().parent.parent.parent


def get_model_path() -> Path:
    """Trả về đường dẫn tới file MJCF robot."""
    return get_project_root() / "assets" / "robot" / "wheeled_biped_real.xml"


class Config:
    """Wrapper cho dict config, hỗ trợ truy cập bằng attribute."""

    def __init__(self, cfg_dict: dict[str, Any]):
        for key, value in cfg_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển lại về dict."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self) -> str:
        return f"Config({self.to_dict()})"


def get_run_metadata(
    config: dict | None = None,
    seed: int | None = None,
    experiment_name: str | None = None,
) -> dict:
    """Collect reproducibility metadata for a training run.

    Returns a plain dict suitable for JSON serialisation. All fields use
    graceful fallbacks — this function never raises.

    Args:
        config: training config dict (extracts env name, num_envs, lr, etc.)
        seed: random seed used for the run.
        experiment_name: human-readable run identifier.

    Returns:
        Dict with keys: timestamp, python_version, platform, jax_version,
        jax_backend, jax_devices, mujoco_version, git_commit, seed,
        experiment_name, env_name, num_envs, learning_rate, rollout_length.
    """
    import datetime
    import platform
    import sys

    meta: dict = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.system(),
    }

    # JAX — version + backend
    try:
        import jax

        meta["jax_version"] = jax.__version__
        meta["jax_backend"] = jax.default_backend()
        meta["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        meta["jax_version"] = "unknown"
        meta["jax_backend"] = "unknown"

    # MuJoCo
    try:
        import mujoco

        meta["mujoco_version"] = mujoco.__version__
    except Exception:
        meta["mujoco_version"] = "unknown"

    # Git commit — best-effort, silent on failure (no git, detached HEAD, etc.)
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        meta["git_commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        meta["git_commit"] = "unknown"

    # Run-specific fields
    if seed is not None:
        meta["seed"] = seed
    if experiment_name is not None:
        meta["experiment_name"] = experiment_name

    # Config-derived fields
    if config:
        task = config.get("task", {})
        ppo = config.get("ppo", {})
        meta["env_name"] = task.get("env", "unknown")
        meta["num_envs"] = task.get("num_envs", "unknown")
        meta["learning_rate"] = ppo.get("learning_rate", "unknown")
        meta["rollout_length"] = ppo.get("rollout_length", "unknown")

    return meta
