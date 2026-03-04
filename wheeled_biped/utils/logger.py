"""
Logger cho training: hỗ trợ TensorBoard và WandB.

Tự động phát hiện backend có sẵn.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np


class TrainingLogger:
    """Ghi log metrics trong quá trình training.

    Hỗ trợ:
      - TensorBoard (tensorboardX)
      - Weights & Biases (wandb)
      - File JSON (luôn bật)
    """

    def __init__(
        self,
        log_dir: str | Path,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "wheeled-biped",
        config: dict[str, Any] | None = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self._step = 0
        self._start_time = time.time()
        self._metrics_buffer: list[dict] = []

        # TensorBoard
        self._tb_writer = None
        if use_tensorboard:
            try:
                from tensorboardX import SummaryWriter

                tb_path = self.log_dir / "tb" / experiment_name
                self._tb_writer = SummaryWriter(str(tb_path))
            except ImportError:
                print("[Logger] tensorboardX không khả dụng, bỏ qua TB.")

        # WandB
        self._wandb_run = None
        if use_wandb:
            try:
                import wandb

                self._wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config or {},
                    dir=str(self.log_dir),
                )
            except ImportError:
                print("[Logger] wandb không khả dụng, bỏ qua WandB.")

        # JSON log
        self._json_path = self.log_dir / f"{experiment_name}_metrics.jsonl"

    @property
    def step(self) -> int:
        return self._step

    def set_step(self, step: int) -> None:
        self._step = step

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Ghi một scalar metric."""
        s = step if step is not None else self._step
        # Chuyển JAX/numpy array sang float
        if hasattr(value, "item"):
            value = float(value.item())

        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, s)

        if self._wandb_run is not None:
            import wandb

            wandb.log({tag: value}, step=s)

        self._metrics_buffer.append({"step": s, "tag": tag, "value": value})

    def log_dict(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Ghi nhiều scalar cùng lúc."""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def log_histogram(
        self, tag: str, values: np.ndarray, step: int | None = None
    ) -> None:
        """Ghi histogram (chỉ TensorBoard)."""
        s = step if step is not None else self._step
        if self._tb_writer is not None:
            self._tb_writer.add_histogram(tag, values, s)

    def flush(self) -> None:
        """Đẩy buffer ra file."""
        if self._tb_writer is not None:
            self._tb_writer.flush()

        # Ghi JSON
        if self._metrics_buffer:
            with open(self._json_path, "a", encoding="utf-8") as f:
                for entry in self._metrics_buffer:
                    f.write(json.dumps(entry) + "\n")
            self._metrics_buffer.clear()

    def close(self) -> None:
        """Đóng tất cả writers."""
        self.flush()
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            import wandb

            wandb.finish()

    def get_elapsed_time(self) -> float:
        """Thời gian đã trôi (giây)."""
        return time.time() - self._start_time

    def print_summary(self, metrics: dict[str, float], prefix: str = "") -> None:
        """In tóm tắt metrics ra console."""
        elapsed = self.get_elapsed_time()
        parts = [f"[{prefix}] Step {self._step} | {elapsed:.0f}s"]
        for key, val in metrics.items():
            parts.append(f"{key}={val:.4f}")
        print(" | ".join(parts))
