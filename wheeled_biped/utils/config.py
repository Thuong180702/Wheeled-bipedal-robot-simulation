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
    with open(path, "r", encoding="utf-8") as f:
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
