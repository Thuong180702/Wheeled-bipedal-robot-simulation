"""
Curriculum Manager - Quản lý pipeline training theo từng giai đoạn.

Workflow:
  1. Bắt đầu từ stage "balance"
  2. Khi đạt ngưỡng thành công → chuyển sang stage tiếp
  3. Nếu hiệu suất giảm quá thấp → quay lại stage trước
  4. Hỗ trợ warm-start từ checkpoint stage trước
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from wheeled_biped.envs import make_env
from wheeled_biped.training.ppo import PPOTrainer
from wheeled_biped.utils.config import load_training_config, load_yaml
from wheeled_biped.utils.logger import TrainingLogger


class CurriculumManager:
    """Quản lý luồng curriculum learning.

    Attributes:
        stages: danh sách các stage cấu hình.
        current_stage_idx: index stage hiện tại.
    """

    def __init__(
        self,
        curriculum_config_path: str | Path,
        output_dir: str | Path = "outputs",
    ):
        """Khởi tạo curriculum manager.

        Args:
            curriculum_config_path: đường dẫn tới curriculum.yaml.
            output_dir: thư mục lưu kết quả.
        """
        config = load_yaml(curriculum_config_path)
        curriculum_cfg = config.get("curriculum", {})

        self.stages = curriculum_cfg.get("stages", [])
        self.promotion_threshold = curriculum_cfg.get("promotion_threshold", 0.8)
        self.promotion_window = curriculum_cfg.get("promotion_window", 100)
        self.demotion_threshold = curriculum_cfg.get("demotion_threshold", 0.3)
        self.max_stage_steps = curriculum_cfg.get("max_stage_steps", 5_000_000)

        self.current_stage_idx = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lịch sử hiệu suất
        self._performance_history: list[float] = []

    @property
    def current_stage(self) -> dict:
        """Stage hiện tại."""
        return self.stages[self.current_stage_idx]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def is_complete(self) -> bool:
        """Đã hoàn thành tất cả các stage?"""
        return self.current_stage_idx >= self.num_stages

    def _create_trainer_for_stage(
        self, stage_idx: int
    ) -> tuple[PPOTrainer, TrainingLogger]:
        """Tạo trainer cho một stage cụ thể.

        Args:
            stage_idx: index của stage.

        Returns:
            (trainer, logger)
        """
        stage = self.stages[stage_idx]
        stage_name = stage["name"]

        # Load config
        config = load_training_config(stage["config"])

        # Tạo environment
        env_name = config.get("task", {}).get("env", "BalanceEnv")
        env = make_env(env_name, config=config)

        # Logger
        log_dir = self.output_dir / "logs"
        logger = TrainingLogger(
            log_dir=log_dir,
            experiment_name=f"stage_{stage_idx}_{stage_name}",
            use_tensorboard=True,
            config=config,
        )

        # Trainer
        trainer = PPOTrainer(env=env, config=config, logger=logger)

        # Warm-start từ stage trước (nếu có)
        pretrained_from = stage.get("pretrained_from")
        if pretrained_from and stage_idx > 0:
            prev_checkpoint = (
                self.output_dir / "checkpoints" / f"stage_{stage_idx - 1}" / "final"
            )
            if prev_checkpoint.exists():
                print(f"  ⟶ Warm-start từ: {prev_checkpoint}")
                trainer.load_checkpoint(str(prev_checkpoint))

        return trainer, logger

    def _evaluate_promotion(self, avg_reward: float) -> str:
        """Đánh giá xem có nên chuyển stage không.

        Returns:
            "promote" | "demote" | "continue"
        """
        self._performance_history.append(avg_reward)

        if len(self._performance_history) < self.promotion_window:
            return "continue"

        # Lấy n reward gần nhất
        recent = self._performance_history[-self.promotion_window :]

        stage = self.current_stage
        target = stage.get("success_value", float("inf"))

        success_rate = sum(1 for r in recent if r >= target) / len(recent)

        if success_rate >= self.promotion_threshold:
            return "promote"
        elif success_rate < self.demotion_threshold and self.current_stage_idx > 0:
            return "demote"
        else:
            return "continue"

    def promote(self) -> bool:
        """Chuyển sang stage tiếp theo.

        Returns:
            True nếu chuyển thành công.
        """
        if self.current_stage_idx < self.num_stages - 1:
            self.current_stage_idx += 1
            self._performance_history.clear()
            print(f"\n{'='*60}")
            print(
                f"  PROMOTED → Stage {self.current_stage_idx}: "
                f"{self.current_stage['name']}"
            )
            print(f"{'='*60}\n")
            return True
        print("\n✓ Đã hoàn thành tất cả stages!")
        return False

    def demote(self) -> bool:
        """Quay lại stage trước."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            self._performance_history.clear()
            print(f"\n{'='*60}")
            print(
                f"  DEMOTED → Stage {self.current_stage_idx}: "
                f"{self.current_stage['name']}"
            )
            print(f"{'='*60}\n")
            return True
        return False

    def run(self, total_steps_per_stage: int | None = None) -> dict[str, Any]:
        """Chạy toàn bộ curriculum pipeline.

        Args:
            total_steps_per_stage: số bước mỗi stage (mặc định dùng config).

        Returns:
            Dict kết quả tổng hợp.
        """
        results = {}

        while not self.is_complete:
            stage = self.current_stage
            stage_name = stage["name"]
            stage_steps = total_steps_per_stage or self.max_stage_steps

            print(f"\n{'═'*60}")
            print(
                f"  Stage {self.current_stage_idx}/{self.num_stages - 1}: "
                f"{stage_name}"
            )
            print(f"  {stage.get('description', '')}")
            print(f"  Max steps: {stage_steps:,}")
            print(f"{'═'*60}\n")

            # Tạo trainer
            trainer, logger = self._create_trainer_for_stage(self.current_stage_idx)

            # Train
            checkpoint_dir = str(
                self.output_dir / "checkpoints" / f"stage_{self.current_stage_idx}"
            )
            train_result = trainer.train(
                total_steps=stage_steps,
                checkpoint_dir=checkpoint_dir,
            )

            results[stage_name] = train_result

            # Đánh giá chuyển stage
            if not self.promote():
                break  # Đã hoàn thành tất cả

        return results

    def run_single_stage(self, stage_name: str, total_steps: int | None = None) -> dict:
        """Chạy một stage cụ thể (bỏ qua curriculum).

        Args:
            stage_name: tên stage.
            total_steps: số bước training.

        Returns:
            Dict kết quả.
        """
        # Tìm stage theo tên
        for idx, stage in enumerate(self.stages):
            if stage["name"] == stage_name:
                self.current_stage_idx = idx
                break
        else:
            raise ValueError(f"Stage '{stage_name}' không tồn tại.")

        trainer, logger = self._create_trainer_for_stage(self.current_stage_idx)

        steps = total_steps or self.max_stage_steps
        checkpoint_dir = str(
            self.output_dir / "checkpoints" / f"stage_{self.current_stage_idx}"
        )

        return trainer.train(
            total_steps=steps,
            checkpoint_dir=checkpoint_dir,
        )
