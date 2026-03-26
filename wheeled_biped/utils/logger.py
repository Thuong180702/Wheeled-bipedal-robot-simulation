"""
Logger cho training: hỗ trợ TensorBoard và WandB.

Tự động phát hiện backend có sẵn.

Features:
  - TensorBoard (tensorboardX) — scalar, histogram, text
  - Weights & Biases (wandb) — scalar, text
  - File JSONL — always enabled, crash-safe via auto-flush (flush_every)
  - run_metadata.json — reproducibility fields written at init
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
        metadata: dict[str, Any] | None = None,
        flush_every: int = 50,
    ):
        """Khởi tạo logger.

        Args:
            log_dir: thư mục lưu log.
            experiment_name: tên thí nghiệm.
            use_tensorboard: bật TensorBoard.
            use_wandb: bật W&B.
            wandb_project: tên W&B project.
            config: dict config (gửi lên W&B, lưu vào metadata).
            metadata: dict reproducibility fields (git hash, seed, v.v.) —
                ghi vào run_metadata.json trong log_dir khi khởi tạo.
            flush_every: tự động flush JSONL sau N lần log_scalar.
                Ngăn mất dữ liệu khi crash. Mặc định 50.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self._step = 0
        self._start_time = time.time()
        self._metrics_buffer: list[dict] = []
        self._log_call_count = 0
        self._flush_every = max(1, flush_every)

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

        # Write reproducibility metadata at init (if provided)
        if metadata is not None:
            meta_payload = {"experiment_name": experiment_name, **metadata}
            if config is not None:
                meta_payload["config"] = config
            meta_path = self.log_dir / "run_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, indent=2, default=str)

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
        # Auto-flush to prevent data loss on crash
        self._log_call_count += 1
        if self._log_call_count % self._flush_every == 0:
            self.flush()

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

    def log_text(self, tag: str, text: str, step: int | None = None) -> None:
        """Ghi text summary (TensorBoard + WandB).

        Hữu ích để ghi ghi chú về quyết định curriculum, hyperparams, v.v.
        """
        s = step if step is not None else self._step
        if self._tb_writer is not None:
            self._tb_writer.add_text(tag, text, s)
        if self._wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=s)

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
