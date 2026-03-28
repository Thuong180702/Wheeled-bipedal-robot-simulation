"""
Live Viewer — Hiển thị robot real-time trong lúc training.

Kiến trúc:
  - Viewer chạy trên MAIN THREAD (bắt buộc bởi GLFW/OpenGL trên Windows).
  - Training chạy trên BACKGROUND THREAD.
  - Dùng shared buffer + threading.Event để đồng bộ.

Dùng qua hàm ``run_training_with_viewer(trainer, **kwargs)``
"""

from __future__ import annotations

import signal
import threading
import time
from typing import TYPE_CHECKING, Any

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from mujoco import mjx


class LiveTrainingViewer:
    """Viewer hiển thị trạng thái robot trong quá trình training.

    - ``run_on_main_thread()`` phải gọi từ main thread (GLFW yêu cầu).
    - Training thread gọi ``update()`` để đẩy dữ liệu mới.
    - Dùng ``viewer_ready`` event để training thread chờ viewer mở xong.
    """

    def __init__(self, mj_model: mujoco.MjModel, title: str = "Training Live"):
        self._model = mj_model
        self._data = mujoco.MjData(mj_model)
        self._title = title
        self._running = False
        self._lock = threading.Lock()

        # Đồng bộ: training thread chờ viewer sẵn sàng
        self.viewer_ready = threading.Event()
        # Đồng bộ: viewer chờ training xong
        self.training_done = threading.Event()

        # Buffer: training thread ghi → main thread đọc
        self._has_new_data = False
        self._buf_qpos: np.ndarray | None = None
        self._buf_qvel: np.ndarray | None = None
        self._buf_ctrl: np.ndarray | None = None
        self._status_text: str = "Đang khởi tạo..."

    # ── Gọi từ TRAINING thread ──

    def wait_for_viewer(self, timeout: float = 30.0) -> bool:
        """Chờ cho đến khi viewer mở xong (gọi từ training thread)."""
        return self.viewer_ready.wait(timeout=timeout)

    def update(
        self,
        mjx_data: mjx.Data,
        env_idx: int = 0,
        info: dict[str, str] | None = None,
    ) -> None:
        """Cập nhật trạng thái hiển thị (gọi từ training thread)."""
        if not self._running:
            return
        try:
            import jax

            qpos = np.array(jax.device_get(mjx_data.qpos[env_idx]))
            qvel = np.array(jax.device_get(mjx_data.qvel[env_idx]))
            ctrl = np.array(jax.device_get(mjx_data.ctrl[env_idx]))

            with self._lock:
                self._buf_qpos = qpos
                self._buf_qvel = qvel
                self._buf_ctrl = ctrl
                self._has_new_data = True
                if info:
                    parts = [f"{k}={v}" for k, v in info.items()]
                    self._status_text = " | ".join(parts)
        except Exception:
            pass

    def set_status(self, text: str) -> None:
        """Cập nhật status text hiển thị trên console."""
        with self._lock:
            self._status_text = text

    def request_stop(self) -> None:
        """Yêu cầu viewer dừng (gọi từ training thread khi train xong)."""
        self.training_done.set()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ── Gọi từ MAIN thread ──

    def run_on_main_thread(self) -> None:
        """Chạy viewer loop trên main thread (blocking).

        Thoát khi:
          - Người dùng đóng cửa sổ, HOẶC
          - ``request_stop()`` được gọi từ training thread.
        """
        if self._model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        mujoco.mj_forward(self._model, self._data)

        self._running = True
        self.viewer_ready.set()  # Báo training thread: viewer đã sẵn sàng

        frame_count = 0
        try:
            with mujoco.viewer.launch_passive(self._model, self._data) as viewer:
                while self._running and viewer.is_running():
                    with self._lock:
                        if self._has_new_data:
                            self._data.qpos[:] = self._buf_qpos
                            self._data.qvel[:] = self._buf_qvel
                            self._data.ctrl[:] = self._buf_ctrl
                            mujoco.mj_forward(self._model, self._data)
                            self._has_new_data = False

                        status = self._status_text

                    viewer.sync()

                    # In status mỗi ~1 giây (30 frames)
                    frame_count += 1
                    if frame_count % 30 == 0:
                        print(f"\r  [Viewer] {status}          ", end="", flush=True)

                    time.sleep(1 / 30)
        except Exception as e:
            print(f"\n[LiveViewer] Lỗi: {e}")
        finally:
            print()  # Newline sau \r
            self._running = False
            self.viewer_ready.set()  # Đảm bảo training thread không bị block mãi


def run_training_with_viewer(
    trainer: Any,
    mj_model: mujoco.MjModel,
    *,
    total_steps: int = 10_000_000,
    log_interval: int = 10,
    save_interval: int = 100,
    checkpoint_dir: str = "checkpoints",
    view_interval: int = 2,
) -> dict:
    """Chạy training + live viewer.

    - Training chạy ở background thread (non-daemon → không bị kill đột ngột).
    - Viewer chạy ở main thread (GLFW/OpenGL bắt buộc).
    - Ctrl+C dừng training an toàn, lưu checkpoint.

    Returns:
        Dict kết quả training.
    """
    viewer = LiveTrainingViewer(mj_model, title="PPO Training")
    result_holder: dict = {}
    error_holder: list = []

    def _train_worker():
        try:
            # Chờ viewer mở xong trước khi bắt đầu
            viewer.set_status("Chờ viewer mở...")
            if not viewer.wait_for_viewer(timeout=30):
                print("\n  ⚠️  Viewer không mở được, training tiếp tục không có viewer.")

            result = trainer.train(
                total_steps=total_steps,
                log_interval=log_interval,
                save_interval=save_interval,
                checkpoint_dir=checkpoint_dir,
                live_view=False,
                view_interval=view_interval,
                _external_viewer=viewer,
            )
            result_holder.update(result)
        except KeyboardInterrupt:
            print("\n  Training thread nhận Ctrl+C")
        except Exception as e:
            import traceback

            print(f"\n  ❌ Training thread lỗi: {e}")
            traceback.print_exc()
            error_holder.append(e)
        finally:
            viewer.request_stop()

    # NON-daemon thread — không bị kill khi main thread thoát
    train_thread = threading.Thread(target=_train_worker, daemon=False)
    train_thread.start()

    print("  🖥️  Live viewer đang mở...")
    print("      Đóng cửa sổ viewer hoặc Ctrl+C để dừng.")
    print("      (JAX JIT compilation lần đầu có thể mất vài phút)\n")

    # Xử lý Ctrl+C: dừng cả viewer + training an toàn
    original_sigint = signal.getsignal(signal.SIGINT)
    _sigint_count = [0]

    def _sigint_handler(signum, frame):
        _sigint_count[0] += 1
        print("\n\n  ⚠️  Ctrl+C — đang dừng training...")
        # Set stop flag trên trainer → training loop sẽ break
        if hasattr(trainer, "_stop_requested"):
            trainer._stop_requested = True
        viewer.request_stop()
        # Nếu Ctrl+C lần 2 → force exit
        if _sigint_count[0] >= 2:
            print("  ⚠️  Ctrl+C lần 2 — force exit!")
            import os

            os._exit(1)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        # Main thread chạy viewer (blocking)
        viewer.run_on_main_thread()
    finally:
        signal.signal(signal.SIGINT, original_sigint)

    # Set stop flag khi viewer đóng (user đóng cửa sổ)
    if hasattr(trainer, "_stop_requested"):
        trainer._stop_requested = True

    # Chờ training hoàn tất (cho thời gian lưu checkpoint)
    print("  Đang chờ training thread kết thúc (tối đa 60s)...")
    train_thread.join(timeout=60)
    if train_thread.is_alive():
        print("  ⚠️  Training thread chưa dừng sau 60s. Force exit.")
        import os

        os._exit(1)

    if error_holder:
        raise error_holder[0]

    return result_holder
