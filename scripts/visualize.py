"""
Script trực quan hóa robot trong MuJoCo viewer.

Cách dùng:
  # Xem robot ở tư thế mặc định
  python scripts/visualize.py model

  # Xem policy đã train
  python scripts/visualize.py policy --checkpoint outputs/checkpoints/balance/final

  # Render video
  python scripts/visualize.py render --checkpoint outputs/checkpoints/balance/final \
    --output video.mp4
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(help="Trực quan hóa robot trong MuJoCo.")
console = Console()


@app.command()
def model(
    model_path: str = typer.Option(
        None,
        help="Đường dẫn file MJCF (mặc định: assets/robot/wheeled_biped.xml).",
    ),
):
    """Mở MuJoCo viewer để xem model robot."""
    import mujoco
    import mujoco.viewer

    from wheeled_biped.utils.config import get_model_path

    path = model_path or str(get_model_path())
    console.print(f"Đang mở model: {path}")

    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    # Đặt tư thế đứng
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    console.print("[green]Viewer đang mở. Đóng cửa sổ để thoát.[/green]")

    # Mở viewer tương tác
    mujoco.viewer.launch(mj_model, mj_data)


@app.command()
def policy(
    checkpoint: str = typer.Option(..., help="Đường dẫn checkpoint."),
    num_steps: int = typer.Option(2000, help="Số bước mô phỏng."),
    slow_factor: float = typer.Option(1.0, help="Hệ số chậm (>1 = chậm hơn)."),
    log: bool = typer.Option(False, help="Ghi telemetry CSV + hiển thị biểu đồ sau khi chạy."),
    log_dir: str = typer.Option("outputs/telemetry", help="Thư mục lưu log."),
):
    """Mô phỏng robot với policy đã train trong viewer."""
    import pickle

    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.viewer
    import numpy as np

    from wheeled_biped.envs.balance_env import BalanceEnv
    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.config import get_model_path
    from wheeled_biped.utils.telemetry import TelemetryRecorder, plot_telemetry

    # Tải checkpoint
    ckpt_path = Path(checkpoint) / "checkpoint.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.device_put(ckpt["params"])
    obs_rms = jax.device_put(ckpt["obs_rms"])
    config = ckpt["config"]

    rng = jax.random.PRNGKey(0)
    obs_size = obs_rms.mean.shape[0]  # Suy ra từ running stats
    model, _ = create_actor_critic(
        obs_size=obs_size,
        action_size=10,
        config=config,
        rng=rng,
    )

    # Tải MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(mj_model)

    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    # Damped settle: robot từ từ hạ xuống mặt đất, không nảy
    for _ in range(500):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    console.print(f"[green]Đang chạy policy từ: {checkpoint}[/green]")
    console.print(f"  Steps: {num_steps}, Slow: {slow_factor}x")
    if log:
        console.print(f"  [cyan]Telemetry: ON → {log_dir}/[/cyan]")

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    # Height range và default height lấy theo env/config để đồng bộ với training
    min_h = float(getattr(BalanceEnv, "MIN_HEIGHT_CMD", 0.40))
    max_h = float(getattr(BalanceEnv, "MAX_HEIGHT_CMD", 0.70))
    default_h = float(config.get("task", {}).get("initial_min_height", 0.69))
    default_h = max(min(default_h, max_h), min_h)
    height_cmd_norm = jnp.array([(default_h - min_h) / (max_h - min_h)])

    # Low-level PID settings (nếu checkpoint bật)
    pid_cfg = config.get("low_level_pid", {})
    pid_enabled = bool(pid_cfg.get("enabled", False))
    pid_alpha = float(pid_cfg.get("action_smoothing_alpha", 0.0))
    pid_i_limit = float(pid_cfg.get("anti_windup_limit", 0.3))
    wheel_vel_limit = float(pid_cfg.get("wheel_vel_limit", 20.0))

    joint_names = [
        "l_hip_roll",
        "l_hip_yaw",
        "l_hip_pitch",
        "l_knee",
        "l_wheel",
        "r_hip_roll",
        "r_hip_yaw",
        "r_hip_pitch",
        "r_knee",
        "r_wheel",
    ]
    joint_mins = []
    joint_maxs = []
    for n in joint_names:
        jid = mj_model.joint(n).id
        jrange = mj_model.jnt_range[jid]
        joint_mins.append(float(jrange[0]))
        joint_maxs.append(float(jrange[1]))
    joint_mins = jnp.array(joint_mins, dtype=jnp.float32)
    joint_maxs = jnp.array(joint_maxs, dtype=jnp.float32)
    wheel_mask = jnp.array([1.0 if "wheel" in n else 0.0 for n in joint_names])

    default_kp = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
    default_ki = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
    default_kd = [3.0, 2.0, 4.0, 4.0, 0.2, 3.0, 2.0, 4.0, 4.0, 0.2]
    kp_cfg = pid_cfg.get("kp", default_kp)
    ki_cfg = pid_cfg.get("ki", default_ki)
    kd_cfg = pid_cfg.get("kd", default_kd)
    if not isinstance(kp_cfg, list) or len(kp_cfg) != 10:
        kp_cfg = default_kp
    if not isinstance(ki_cfg, list) or len(ki_cfg) != 10:
        ki_cfg = default_ki
    if not isinstance(kd_cfg, list) or len(kd_cfg) != 10:
        kd_cfg = default_kd
    pid_kp = jnp.array(kp_cfg, dtype=jnp.float32)
    pid_ki = jnp.array(ki_cfg, dtype=jnp.float32)
    pid_kd = jnp.array(kd_cfg, dtype=jnp.float32)

    ctrl_range = jnp.array(mj_model.actuator_ctrlrange)
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    pid_integral = jnp.zeros(10)

    # prev_action trong observation phải khớp training (action sau smoothing)
    prev_action = jnp.zeros(10)

    # Telemetry recorder
    recorder = TelemetryRecorder(control_dt=control_dt) if log else None

    from wheeled_biped.utils.math_utils import quat_to_euler, wrap_angle

    _initial_yaw = float(quat_to_euler(jnp.array(mj_data.qpos[3:7]))[2])

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for step in range(num_steps):
            if not viewer.is_running():
                break

            step_start = time.time()

            # Trích xuất observation - PHẢI GIỐNG CHÍNH XÁC balance_env obs (41 dims)
            from wheeled_biped.utils.math_utils import (
                get_gravity_in_body_frame,
                quat_conjugate,
                quat_rotate,
            )

            torso_quat = jnp.array(mj_data.qpos[3:7])
            yaw_error = jnp.array([wrap_angle(float(quat_to_euler(torso_quat)[2]) - _initial_yaw)])
            gravity_body = get_gravity_in_body_frame(torso_quat)

            # Body-frame velocity (giống training)
            quat_inv = quat_conjugate(torso_quat)
            world_lin_vel = jnp.array(mj_data.qvel[:3])
            world_ang_vel = jnp.array(mj_data.qvel[3:6])
            body_lin_vel = quat_rotate(quat_inv, world_lin_vel)
            body_ang_vel = quat_rotate(quat_inv, world_ang_vel)

            obs = jnp.concatenate(
                [
                    gravity_body,  # 3
                    body_lin_vel,  # 3 (body frame)
                    body_ang_vel,  # 3 (body frame)
                    jnp.array(mj_data.qpos[7:17]),  # 10
                    jnp.array(mj_data.qvel[6:16]),  # 10
                    prev_action,  # 10 (normalized [-1,1])
                    height_cmd_norm,  # 1 (height command normalized)
                    yaw_error,  # 1
                ]
            )

            # Normalize + get action
            obs_norm = normalize_obs(obs, obs_rms)
            dist, _ = model.apply(params, obs_norm)
            action = dist.loc  # deterministic
            action = jnp.clip(action, -1.0, 1.0)

            # Low-level PID path (giống training) hoặc direct path
            if pid_enabled and pid_alpha > 0.0:
                control_action = pid_alpha * prev_action + (1.0 - pid_alpha) * action
            else:
                control_action = action

            if pid_enabled:
                joint_pos = jnp.array(mj_data.qpos[7:17])
                joint_vel = jnp.array(mj_data.qvel[6:16])
                pos_target = joint_mins + (control_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
                vel_target_wheel = control_action * wheel_vel_limit
                pos_err = pos_target - joint_pos
                error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_wheel - joint_vel)
                d_error = -joint_vel
                pid_integral = jnp.clip(
                    pid_integral + error * control_dt,
                    -pid_i_limit,
                    pid_i_limit,
                )
                ctrl = jnp.clip(
                    pid_kp * error + pid_kd * d_error + pid_ki * pid_integral,
                    ctrl_min,
                    ctrl_max,
                )
            else:
                ctrl = ctrl_min + (control_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

            prev_action = control_action
            mj_data.ctrl[:] = np.array(ctrl)

            # Physics steps
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            # Ghi telemetry
            if recorder is not None:
                recorder.record(mj_data)

            viewer.sync()

            # Timing
            elapsed = time.time() - step_start
            sleep_time = control_dt * slow_factor - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Sau khi viewer đóng: lưu log + vẽ biểu đồ
    if recorder is not None and len(recorder.data) > 0:
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = recorder.save_csv(f"{log_dir}/policy_{ts}.csv")
        console.print(f"[green]Đã lưu telemetry: {csv_path}[/green]")
        plot_telemetry(
            recorder.to_numpy(),
            output_path=f"{log_dir}/policy_{ts}.png",
            show=True,
        )


@app.command()
def render(
    checkpoint: str = typer.Option(..., help="Đường dẫn checkpoint."),
    output: str = typer.Option("video.mp4", help="File video output."),
    num_steps: int = typer.Option(500, help="Số bước."),
    width: int = typer.Option(1280, help="Chiều rộng video."),
    height: int = typer.Option(720, help="Chiều cao video."),
    fps: int = typer.Option(50, help="FPS video."),
    log: bool = typer.Option(True, help="Ghi telemetry CSV + biểu đồ cạnh video."),
):
    """Render video từ policy đã train."""
    import pickle

    import jax
    import jax.numpy as jnp
    import mediapy
    import mujoco
    import numpy as np

    from wheeled_biped.envs.balance_env import BalanceEnv
    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.config import get_model_path
    from wheeled_biped.utils.math_utils import (
        get_gravity_in_body_frame,
        quat_conjugate,
        quat_rotate,
        quat_to_euler,
        wrap_angle,
    )
    from wheeled_biped.utils.telemetry import TelemetryRecorder, plot_telemetry

    # Tải checkpoint
    ckpt_path = Path(checkpoint) / "checkpoint.pkl"
    with open(ckpt_path, "rb") as f:
        ckpt = pickle.load(f)

    params = jax.device_put(ckpt["params"])
    obs_rms = jax.device_put(ckpt["obs_rms"])
    config = ckpt["config"]

    rng = jax.random.PRNGKey(0)
    obs_size = obs_rms.mean.shape[0]  # Suy ra từ running stats
    model, _ = create_actor_critic(
        obs_size=obs_size,
        action_size=10,
        config=config,
        rng=rng,
    )

    # MuJoCo
    mj_model = mujoco.MjModel.from_xml_path(str(get_model_path()))
    mj_data = mujoco.MjData(mj_model)

    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    for _ in range(500):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    frames = []
    min_h = float(getattr(BalanceEnv, "MIN_HEIGHT_CMD", 0.40))
    max_h = float(getattr(BalanceEnv, "MAX_HEIGHT_CMD", 0.70))
    default_h = float(config.get("task", {}).get("initial_min_height", 0.69))
    default_h = max(min(default_h, max_h), min_h)
    height_cmd_norm = jnp.array([(default_h - min_h) / (max_h - min_h)])

    pid_cfg = config.get("low_level_pid", {})
    pid_enabled = bool(pid_cfg.get("enabled", False))
    pid_alpha = float(pid_cfg.get("action_smoothing_alpha", 0.0))
    pid_i_limit = float(pid_cfg.get("anti_windup_limit", 0.3))
    wheel_vel_limit = float(pid_cfg.get("wheel_vel_limit", 20.0))
    joint_names = [
        "l_hip_roll",
        "l_hip_yaw",
        "l_hip_pitch",
        "l_knee",
        "l_wheel",
        "r_hip_roll",
        "r_hip_yaw",
        "r_hip_pitch",
        "r_knee",
        "r_wheel",
    ]
    joint_mins = []
    joint_maxs = []
    for n in joint_names:
        jid = mj_model.joint(n).id
        jrange = mj_model.jnt_range[jid]
        joint_mins.append(float(jrange[0]))
        joint_maxs.append(float(jrange[1]))
    joint_mins = jnp.array(joint_mins, dtype=jnp.float32)
    joint_maxs = jnp.array(joint_maxs, dtype=jnp.float32)
    wheel_mask = jnp.array([1.0 if "wheel" in n else 0.0 for n in joint_names])
    default_kp = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
    default_ki = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
    default_kd = [3.0, 2.0, 4.0, 4.0, 0.2, 3.0, 2.0, 4.0, 4.0, 0.2]
    kp_cfg = pid_cfg.get("kp", default_kp)
    ki_cfg = pid_cfg.get("ki", default_ki)
    kd_cfg = pid_cfg.get("kd", default_kd)
    if not isinstance(kp_cfg, list) or len(kp_cfg) != 10:
        kp_cfg = default_kp
    if not isinstance(ki_cfg, list) or len(ki_cfg) != 10:
        ki_cfg = default_ki
    if not isinstance(kd_cfg, list) or len(kd_cfg) != 10:
        kd_cfg = default_kd
    pid_kp = jnp.array(kp_cfg, dtype=jnp.float32)
    pid_ki = jnp.array(ki_cfg, dtype=jnp.float32)
    pid_kd = jnp.array(kd_cfg, dtype=jnp.float32)
    ctrl_range = jnp.array(mj_model.actuator_ctrlrange)
    ctrl_min = ctrl_range[:, 0]
    ctrl_max = ctrl_range[:, 1]
    pid_integral = jnp.zeros(10)
    prev_action = jnp.zeros(10)

    console.print(f"Rendering {num_steps} steps...")
    recorder = TelemetryRecorder(control_dt=control_dt) if log else None
    _initial_yaw = float(quat_to_euler(jnp.array(mj_data.qpos[3:7]))[2])

    for step in range(num_steps):
        torso_quat = jnp.array(mj_data.qpos[3:7])
        yaw_error = jnp.array([wrap_angle(float(quat_to_euler(torso_quat)[2]) - _initial_yaw)])
        gravity_body = get_gravity_in_body_frame(torso_quat)

        # Body-frame velocity (giống training)
        quat_inv = quat_conjugate(torso_quat)
        body_lin_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[:3]))
        body_ang_vel = quat_rotate(quat_inv, jnp.array(mj_data.qvel[3:6]))

        obs = jnp.concatenate(
            [
                gravity_body,
                body_lin_vel,
                body_ang_vel,
                jnp.array(mj_data.qpos[7:17]),
                jnp.array(mj_data.qvel[6:16]),
                prev_action,
                height_cmd_norm,  # height command normalized
                yaw_error,  # 1
            ]
        )

        obs_norm = normalize_obs(obs, obs_rms)
        dist, _ = model.apply(params, obs_norm)
        action = jnp.clip(dist.loc, -1.0, 1.0)

        if pid_enabled and pid_alpha > 0.0:
            control_action = pid_alpha * prev_action + (1.0 - pid_alpha) * action
        else:
            control_action = action

        if pid_enabled:
            joint_pos = jnp.array(mj_data.qpos[7:17])
            joint_vel = jnp.array(mj_data.qvel[6:16])
            pos_target = joint_mins + (control_action + 1.0) * 0.5 * (joint_maxs - joint_mins)
            vel_target_wheel = control_action * wheel_vel_limit
            pos_err = pos_target - joint_pos
            error = (1.0 - wheel_mask) * pos_err + wheel_mask * (vel_target_wheel - joint_vel)
            d_error = -joint_vel
            pid_integral = jnp.clip(
                pid_integral + error * control_dt,
                -pid_i_limit,
                pid_i_limit,
            )
            ctrl = jnp.clip(
                pid_kp * error + pid_kd * d_error + pid_ki * pid_integral,
                ctrl_min,
                ctrl_max,
            )
        else:
            ctrl = ctrl_min + (control_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

        prev_action = control_action
        mj_data.ctrl[:] = np.array(ctrl)

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        if recorder is not None:
            recorder.record(mj_data)

        renderer.update_scene(mj_data, camera="track")
        frames.append(renderer.render())

    # Lưu video
    mediapy.write_video(output, frames, fps=fps)
    console.print(f"[green]Đã lưu video: {output}[/green]")

    # Lưu telemetry cạnh video
    if recorder is not None and len(recorder.data) > 0:
        out_path = Path(output)
        csv_path = recorder.save_csv(out_path.with_suffix(".csv"))
        console.print(f"[green]Đã lưu telemetry: {csv_path}[/green]")
        plot_telemetry(
            recorder.to_numpy(),
            output_path=str(out_path.with_suffix(".png")),
            show=False,
        )
        console.print(f"[green]Đã lưu biểu đồ: {out_path.with_suffix('.png')}[/green]")


@app.command()
def interactive(
    model_path: str = typer.Option(
        None,
        help="Đường dẫn file MJCF (mặc định: assets/robot/wheeled_biped.xml).",
    ),
    checkpoint: str = typer.Option(
        None,
        help="Checkpoint policy để giữ thăng bằng (nếu có).",
    ),
    log: bool = typer.Option(False, help="Ghi telemetry CSV + biểu đồ khi đóng viewer."),
    log_dir: str = typer.Option("outputs/telemetry", help="Thư mục lưu log."),
):
    """Điều khiển robot bằng bàn phím trong MuJoCo viewer.

    \b
    Phím điều khiển:
      ↑ / ↓  : Tiến / Lùi (lực bánh xe)
      ← / →  : Rẽ trái / phải (chênh lệch lực bánh)
      Q / E  : Nghiêng trái / phải (hip roll)
      R      : Reset về tư thế đứng
      Space  : Dừng (phanh bánh xe)
      [ / ]  : Giảm / Tăng tốc độ
    """
    import pickle
    import threading

    import mujoco
    import mujoco.viewer
    import numpy as np

    from wheeled_biped.envs.balance_env import BalanceEnv
    from wheeled_biped.utils.config import get_model_path
    from wheeled_biped.utils.telemetry import TelemetryRecorder, plot_telemetry

    path = model_path or str(get_model_path())
    console.print(f"Đang mở model: {path}")

    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    def _settle_robot():
        """Damped settle: robot từ từ hạ xuống mặt đất, không nảy."""
        if mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
        for _ in range(500):
            mujoco.mj_step(mj_model, mj_data)
            mj_data.qvel[:] = 0
        mujoco.mj_forward(mj_model, mj_data)

    _settle_robot()
    # Mutable container so nested functions (_policy) can read and reset this.
    _initial_yaw: list[float] = [0.0]  # filled after imports inside `if checkpoint`

    MIN_H = float(getattr(BalanceEnv, "MIN_HEIGHT_CMD", 0.40))  # noqa: N806
    MAX_H = float(getattr(BalanceEnv, "MAX_HEIGHT_CMD", 0.70))  # noqa: N806
    default_height_cmd = max(min(0.69, MAX_H), MIN_H)
    KEY_H = default_height_cmd  # noqa: N806

    # ---------- trạng thái điều khiển ----------
    # Actuator order: l_hip_roll(0), l_hip_yaw(1), l_hip_pitch(2), l_knee(3), l_wheel(4)
    #                 r_hip_roll(5), r_hip_yaw(6), r_hip_pitch(7), r_knee(8), r_wheel(9)
    ctrl_state = {
        "forward": 0.0,  # -1 .. +1
        "turn": 0.0,  # -1 .. +1
        "roll": 0.0,  # -1 .. +1
        "speed": 5.0,  # wheel torque magnitude (Nm)
        "roll_gain": 3.0,  # hip roll torque magnitude
        "height_cmd": default_height_cmd,  # độ cao mục tiêu (m)
        "reset_requested": False,
    }
    _lock = threading.Lock()

    # ---------- PD standing gains ----------
    # Khi không có checkpoint, sử dụng PD controller đơn giản để giữ thăng bằng
    # Target joint positions (standing keyframe)
    standing_qpos = np.array(mj_data.qpos[7:17], copy=True) if mj_model.nkey > 0 else np.zeros(10)
    kp_joints = np.array(
        [8.0, 6.0, 20.0, 18.0, 0, 8.0, 6.0, 20.0, 18.0, 0]
    )  # hip_roll, hip_yaw, hip_pitch, knee, wheel (0 cho bánh)
    kd_joints = np.array([0.5, 0.3, 1.5, 1.2, 0, 0.5, 0.3, 1.5, 1.2, 0])

    # Tải policy nếu có
    policy_fn = None
    if checkpoint:
        try:
            import jax
            import jax.numpy as jnp

            from wheeled_biped.training.networks import create_actor_critic
            from wheeled_biped.training.ppo import normalize_obs
            from wheeled_biped.utils.math_utils import (
                get_gravity_in_body_frame,
                quat_conjugate,
                quat_rotate,
                quat_to_euler,
                wrap_angle,
            )

            ckpt_path = Path(checkpoint) / "checkpoint.pkl"
            with open(ckpt_path, "rb") as f:
                ckpt = pickle.load(f)

            params = jax.device_put(ckpt["params"])
            obs_rms = jax.device_put(ckpt["obs_rms"])
            config = ckpt["config"]
            rng = jax.random.PRNGKey(0)
            obs_size = obs_rms.mean.shape[0]
            network, _ = create_actor_critic(
                obs_size=obs_size,
                action_size=10,
                config=config,
                rng=rng,
            )

            _interactive_prev_action = jnp.zeros(10)
            _interactive_pid_integral = jnp.zeros(10)
            _interactive_height_cmd_norm = (default_height_cmd - MIN_H) / (MAX_H - MIN_H)
            _initial_yaw[0] = float(quat_to_euler(jnp.array(mj_data.qpos[3:7]))[2])

            pid_cfg = config.get("low_level_pid", {})
            pid_enabled = bool(pid_cfg.get("enabled", False))
            pid_alpha = float(pid_cfg.get("action_smoothing_alpha", 0.0))
            pid_i_limit = float(pid_cfg.get("anti_windup_limit", 0.3))
            wheel_vel_limit = float(pid_cfg.get("wheel_vel_limit", 20.0))
            joint_names = [
                "l_hip_roll",
                "l_hip_yaw",
                "l_hip_pitch",
                "l_knee",
                "l_wheel",
                "r_hip_roll",
                "r_hip_yaw",
                "r_hip_pitch",
                "r_knee",
                "r_wheel",
            ]
            joint_mins = []
            joint_maxs = []
            for n in joint_names:
                jid = mj_model.joint(n).id
                jrange = mj_model.jnt_range[jid]
                joint_mins.append(float(jrange[0]))
                joint_maxs.append(float(jrange[1]))
            joint_mins = jnp.array(joint_mins, dtype=jnp.float32)
            joint_maxs = jnp.array(joint_maxs, dtype=jnp.float32)
            wheel_mask = jnp.array([1.0 if "wheel" in n else 0.0 for n in joint_names])
            default_kp = [55.0, 40.0, 70.0, 70.0, 4.0, 55.0, 40.0, 70.0, 70.0, 4.0]
            default_ki = [0.8, 0.4, 1.0, 1.0, 0.1, 0.8, 0.4, 1.0, 1.0, 0.1]
            default_kd = [3.0, 2.0, 4.0, 4.0, 0.2, 3.0, 2.0, 4.0, 4.0, 0.2]
            kp_cfg = pid_cfg.get("kp", default_kp)
            ki_cfg = pid_cfg.get("ki", default_ki)
            kd_cfg = pid_cfg.get("kd", default_kd)
            if not isinstance(kp_cfg, list) or len(kp_cfg) != 10:
                kp_cfg = default_kp
            if not isinstance(ki_cfg, list) or len(ki_cfg) != 10:
                ki_cfg = default_ki
            if not isinstance(kd_cfg, list) or len(kd_cfg) != 10:
                kd_cfg = default_kd
            pid_kp = jnp.array(kp_cfg, dtype=jnp.float32)
            pid_ki = jnp.array(ki_cfg, dtype=jnp.float32)
            pid_kd = jnp.array(kd_cfg, dtype=jnp.float32)
            ctrl_range = jnp.array(mj_model.actuator_ctrlrange)
            ctrl_min = ctrl_range[:, 0]
            ctrl_max = ctrl_range[:, 1]

            def _policy(data, height_cmd_norm=None):
                nonlocal _interactive_prev_action, _interactive_height_cmd_norm  # noqa: E501
                nonlocal _interactive_pid_integral
                if height_cmd_norm is not None:
                    _interactive_height_cmd_norm = height_cmd_norm
                torso_quat = jnp.array(data.qpos[3:7])
                gravity_body = get_gravity_in_body_frame(torso_quat)
                quat_inv = quat_conjugate(torso_quat)
                body_lin_vel = quat_rotate(quat_inv, jnp.array(data.qvel[:3]))
                body_ang_vel = quat_rotate(quat_inv, jnp.array(data.qvel[3:6]))
                _cur_yaw = float(quat_to_euler(torso_quat)[2])
                yaw_error = jnp.array([wrap_angle(_cur_yaw - _initial_yaw[0])])
                obs = jnp.concatenate(
                    [
                        gravity_body,
                        body_lin_vel,
                        body_ang_vel,
                        jnp.array(data.qpos[7:17]),
                        jnp.array(data.qvel[6:16]),
                        _interactive_prev_action,
                        jnp.array([_interactive_height_cmd_norm]),
                        yaw_error,  # 1
                    ]
                )
                obs_norm = normalize_obs(obs, obs_rms)
                dist, _ = network.apply(params, obs_norm)
                action = jnp.clip(dist.loc, -1.0, 1.0)

                if pid_enabled and pid_alpha > 0.0:
                    control_action = (
                        pid_alpha * _interactive_prev_action + (1.0 - pid_alpha) * action
                    )
                else:
                    control_action = action

                if pid_enabled:
                    joint_pos = jnp.array(data.qpos[7:17])
                    joint_vel = jnp.array(data.qvel[6:16])
                    pos_target = joint_mins + (control_action + 1.0) * 0.5 * (
                        joint_maxs - joint_mins
                    )
                    vel_target_wheel = control_action * wheel_vel_limit
                    pos_err = pos_target - joint_pos
                    error = (1.0 - wheel_mask) * pos_err + wheel_mask * (
                        vel_target_wheel - joint_vel
                    )
                    d_error = -joint_vel
                    _interactive_pid_integral = jnp.clip(
                        _interactive_pid_integral + error * control_dt,
                        -pid_i_limit,
                        pid_i_limit,
                    )
                    ctrl = jnp.clip(
                        pid_kp * error + pid_kd * d_error + pid_ki * _interactive_pid_integral,
                        ctrl_min,
                        ctrl_max,
                    )
                else:
                    ctrl = ctrl_min + (control_action + 1.0) * 0.5 * (ctrl_max - ctrl_min)

                _interactive_prev_action = control_action
                return np.array(ctrl)

            policy_fn = _policy
            console.print(f"[green]Đã tải policy: {checkpoint}[/green]")
        except Exception as e:
            console.print(f"[yellow]Không tải được policy: {e}[/yellow]")
            console.print("[yellow]Sẽ dùng PD controller thay thế.[/yellow]")

    # ---------- keyboard callback ----------
    def key_callback(keycode):
        with _lock:
            # MuJoCo key codes: W=87, S=83, A=65, D=68, Q=81, E=69, R=82
            # Space=32, [=91, ]=93
            if keycode == 265:  # GLFW_KEY_UP / W
                ctrl_state["forward"] = 1.0
            elif keycode == 264:  # GLFW_KEY_DOWN / S
                ctrl_state["forward"] = -1.0
            elif keycode == 263:  # GLFW_KEY_LEFT / A
                ctrl_state["turn"] = -1.0
            elif keycode == 262:  # GLFW_KEY_RIGHT / D
                ctrl_state["turn"] = 1.0
            elif keycode == 81:  # Q
                ctrl_state["roll"] = -1.0
            elif keycode == 69:  # E
                ctrl_state["roll"] = 1.0
            elif keycode == 32:  # Space
                ctrl_state["forward"] = 0.0
                ctrl_state["turn"] = 0.0
                ctrl_state["roll"] = 0.0
            elif keycode == 93:  # ]
                ctrl_state["speed"] = min(ctrl_state["speed"] + 1.0, 10.0)
                print(f"  Speed: {ctrl_state['speed']:.1f}")
            elif keycode == 91:  # [
                ctrl_state["speed"] = max(ctrl_state["speed"] - 1.0, 1.0)
                print(f"  Speed: {ctrl_state['speed']:.1f}")
            elif keycode == 85:  # U — tăng chiều cao (+1cm)
                ctrl_state["height_cmd"] = min(ctrl_state["height_cmd"] + 0.01, MAX_H)
                print(f"  Chiều cao: {ctrl_state['height_cmd']:.2f}m")
            elif keycode == 74:  # J — giảm chiều cao (-1cm)
                ctrl_state["height_cmd"] = max(ctrl_state["height_cmd"] - 0.01, MIN_H)
                print(f"  Chiều cao: {ctrl_state['height_cmd']:.2f}m")
            elif keycode == 259:  # Backspace — reset robot
                ctrl_state["reset_requested"] = True

    console.print("\n[bold cyan]═══ Chế độ điều khiển tương tác ═══[/bold cyan]")
    console.print("  ↑/↓ : Tiến / Lùi")
    console.print("  ←/→ : Rẽ trái / phải")
    console.print("  Q/E : Nghiêng trái / phải (hip roll)")
    console.print("  U/J : Tăng / Giảm chiều cao robot")
    console.print("  Space : Dừng lại")
    console.print("  [ / ] : Giảm / Tăng tốc độ")
    console.print("  Backspace : Reset robot\n")
    if policy_fn:
        console.print("  [green]Mode: Policy + keyboard override[/green]")
    else:
        console.print("  [yellow]Mode: PD controller + keyboard[/yellow]")

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = max(1, int(control_dt / physics_dt))

    # Decay factor — phím giữ = giá trị 1, thả phím → decay dần về 0
    decay = 0.85

    # Smooth height interpolation — tránh giật khi thay đổi chiều cao
    smooth_h_cmd = default_height_cmd  # giá trị thực tế dùng để tính target (nội suy mượt, mét)
    height_smooth_rate = 0.1  # tốc độ nội suy mỗi step (nhỏ = mượt hơn)

    # Telemetry recorder
    recorder = TelemetryRecorder(control_dt=control_dt) if log else None
    if log:
        console.print(f"  [cyan]Telemetry: ON → {log_dir}/[/cyan]")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            with _lock:
                # Handle reset request
                if ctrl_state["reset_requested"]:
                    ctrl_state["reset_requested"] = False
                    ctrl_state["forward"] = 0.0
                    ctrl_state["turn"] = 0.0
                    ctrl_state["roll"] = 0.0
                    ctrl_state["height_cmd"] = default_height_cmd
                    smooth_h_cmd = default_height_cmd
                    _settle_robot()
                    if policy_fn is not None:
                        _interactive_prev_action = jnp.zeros(10)
                        _interactive_pid_integral = jnp.zeros(10)
                        _interactive_height_cmd_norm = (default_height_cmd - MIN_H) / (
                            MAX_H - MIN_H
                        )
                    viewer.sync()
                    continue

                fwd = ctrl_state["forward"]
                trn = ctrl_state["turn"]
                rll = ctrl_state["roll"]
                spd = ctrl_state["speed"]
                rg = ctrl_state["roll_gain"]
                h_cmd = ctrl_state["height_cmd"]  # mục tiêu độ cao (m)

                # Decay — giảm dần khi không nhấn
                ctrl_state["forward"] *= decay
                ctrl_state["turn"] *= decay
                ctrl_state["roll"] *= decay
                if abs(ctrl_state["forward"]) < 0.01:
                    ctrl_state["forward"] = 0.0
                if abs(ctrl_state["turn"]) < 0.01:
                    ctrl_state["turn"] = 0.0
                if abs(ctrl_state["roll"]) < 0.01:
                    ctrl_state["roll"] = 0.0

            # Smooth interpolation: nội suy smooth_h_cmd → h_cmd mượt mà
            smooth_h_cmd += (h_cmd - smooth_h_cmd) * height_smooth_rate
            if abs(smooth_h_cmd - h_cmd) < 0.001:
                smooth_h_cmd = h_cmd
            h_eff = smooth_h_cmd  # h_eff = độ cao mục tiêu mượt (mét)

            # Tính target khớp theo chiều cao (h_eff tính bằng mét)
            # Keyframe baseline = KEY_H, min = MIN_H, max = MAX_H
            HP_HIGH, KN_HIGH = 0.0, 0.0  # chân thẳng → cao nhất  # noqa: N806
            HP_KEY, KN_KEY = 0.3, 0.5  # keyframe  # noqa: N806
            HP_LOW, KN_LOW = 1.5, 2.5  # gập tối đa về trước → thấp nhất  # noqa: N806

            height_target = standing_qpos.copy()
            if h_eff >= KEY_H:
                # Nâng cao: nội suy từ keyframe → thẳng
                t = (h_eff - KEY_H) / max(MAX_H - KEY_H, 1e-6)  # 0→1
                height_target[2] = HP_KEY + t * (HP_HIGH - HP_KEY)  # l_hip_pitch
                height_target[7] = HP_KEY + t * (HP_HIGH - HP_KEY)  # r_hip_pitch
                height_target[3] = KN_KEY + t * (KN_HIGH - KN_KEY)  # l_knee
                height_target[8] = KN_KEY + t * (KN_HIGH - KN_KEY)  # r_knee
            else:
                # Hạ thấp: nội suy từ keyframe → gập max
                t = (KEY_H - h_eff) / max(KEY_H - MIN_H, 1e-6)  # 0→1
                height_target[2] = HP_KEY + t * (HP_LOW - HP_KEY)  # l_hip_pitch
                height_target[7] = HP_KEY + t * (HP_LOW - HP_KEY)  # r_hip_pitch
                height_target[3] = KN_KEY + t * (KN_LOW - KN_KEY)  # l_knee
                height_target[8] = KN_KEY + t * (KN_LOW - KN_KEY)  # r_knee

            if policy_fn is not None:
                # Tính height_command normalized [0,1] cho policy (h_eff đã là mét)
                h_cmd_norm = (h_eff - MIN_H) / (MAX_H - MIN_H)
                h_cmd_norm = max(0.0, min(1.0, h_cmd_norm))
                # Policy giữ thăng bằng
                base_ctrl = policy_fn(mj_data, height_cmd_norm=h_cmd_norm)
                # Override bánh xe
                desired_left = fwd * spd - trn * spd * 0.5
                desired_right = fwd * spd + trn * spd * 0.5
                if abs(fwd) > 0.01 or abs(trn) > 0.01:
                    base_ctrl[4] = np.clip(desired_left, -10, 10)  # l_wheel
                    base_ctrl[9] = np.clip(desired_right, -10, 10)
                # Override hip roll
                if abs(rll) > 0.01:
                    base_ctrl[0] = rll * rg
                    base_ctrl[5] = rll * rg
                # PD blend cho height control — mượt hơn, không giật
                # Blend policy output với PD theo |h_eff|: càng lệch keyframe → PD càng mạnh
                if abs(h_eff - KEY_H) > 0.005:
                    joint_pos = np.array(mj_data.qpos[7:17])
                    joint_vel = np.array(mj_data.qvel[6:16])
                    kp_h, kd_h = 8.0, 0.8
                    # blend: 0→1 theo mức lệch khỏi keyframe
                    if h_eff >= KEY_H:
                        blend = min((h_eff - KEY_H) / max(MAX_H - KEY_H, 1e-6), 1.0)
                    else:
                        blend = min((KEY_H - h_eff) / max(KEY_H - MIN_H, 1e-6), 1.0)
                    for idx in [2, 3, 7, 8]:  # hip_pitch, knee (2 bên)
                        pd_torque = (
                            kp_h * (height_target[idx] - joint_pos[idx]) - kd_h * joint_vel[idx]
                        )
                        pd_torque = np.clip(pd_torque, -10, 10)
                        # Blend: policy*(1-blend) + PD*blend
                        base_ctrl[idx] = (1 - blend) * base_ctrl[idx] + blend * pd_torque
                mj_data.ctrl[:] = np.clip(
                    base_ctrl,
                    mj_model.actuator_ctrlrange[:, 0],
                    mj_model.actuator_ctrlrange[:, 1],
                )
            else:
                # PD controller giữ tư thế + height control
                joint_pos = np.array(mj_data.qpos[7:17])
                joint_vel = np.array(mj_data.qvel[6:16])
                pd_ctrl = kp_joints * (height_target - joint_pos) - kd_joints * joint_vel

                # Overlay keyboard
                desired_left = fwd * spd - trn * spd * 0.5
                desired_right = fwd * spd + trn * spd * 0.5
                pd_ctrl[4] = desired_left  # l_wheel
                pd_ctrl[9] = desired_right  # r_wheel
                pd_ctrl[0] += rll * rg  # l_hip_roll
                pd_ctrl[5] += rll * rg  # r_hip_roll

                mj_data.ctrl[:] = np.clip(
                    pd_ctrl,
                    mj_model.actuator_ctrlrange[:, 0],
                    mj_model.actuator_ctrlrange[:, 1],
                )

            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            if recorder is not None:
                recorder.record(mj_data)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # Sau khi viewer đóng: lưu telemetry
    if recorder is not None and len(recorder.data) > 0:
        import datetime

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = recorder.save_csv(f"{log_dir}/interactive_{ts}.csv")
        console.print(f"\n[green]Đã lưu telemetry: {csv_path}[/green]")
        plot_telemetry(
            recorder.to_numpy(),
            output_path=f"{log_dir}/interactive_{ts}.png",
            show=True,
        )


@app.command()
def unified(
    checkpoint_dir: str = typer.Option(
        "outputs/checkpoints",
        help="Thư mục gốc chứa các checkpoint (balance/, wheeled_locomotion/, ...).",
    ),
    model_path: str = typer.Option(
        None,
        help="Đường dẫn MJCF (mặc định: assets/robot/wheeled_biped.xml).",
    ),
    auto_mode: bool = typer.Option(
        True,
        help="Tự động chuyển đổi skill (False = chọn thủ công bằng phím 1-5).",
    ),
):
    """Chạy unified controller — tự động chọn skill phù hợp + điều khiển bằng phím.

    \b
    Phím điều khiển:
      ↑/↓     : Tiến / Lùi
      ←/→     : Rẽ trái / phải
      U / J   : Tăng / Giảm chiều cao robot
      Space   : Dừng lại
      [ / ]   : Giảm / Tăng tốc
      1-5     : Ép chọn skill (1=Balance, 2=Locomotion, 3=Walking, 4=Stair, 5=Terrain)
      6       : Ép chọn Stand Up
      0       : Quay về auto-detect
      R       : Reset robot (trong viewer)
    """
    import threading

    import mujoco
    import mujoco.viewer
    import numpy as np

    from wheeled_biped.inference.unified_controller import (
        ControlCommand,
        Skill,
        UnifiedController,
    )
    from wheeled_biped.utils.config import get_model_path

    path = model_path or str(get_model_path())
    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    for _ in range(500):
        mujoco.mj_step(mj_model, mj_data)
        mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    console.print("\n[bold green]═══ Unified Controller ═══[/bold green]")
    console.print(f"  Checkpoint dir: {checkpoint_dir}")
    console.print(f"  Auto mode: {auto_mode}\n")

    controller = UnifiedController(
        checkpoint_dir=checkpoint_dir,
        mj_model=mj_model,
    )

    # ---------- trạng thái điều khiển ----------
    ctrl_cmd = {
        "vel_x": 0.0,
        "ang_vel_z": 0.0,
        "speed": 5.0,
        "force_skill": None,
        "height_cmd": 0.71,  # độ cao mục tiêu (m), range [0.38, 0.72]
    }
    _lock = threading.Lock()
    _auto = {"mode": auto_mode}

    _key_to_skill = {
        49: Skill.BALANCE,  # 1
        50: Skill.LOCOMOTION,  # 2
        51: Skill.WALKING,  # 3
        52: Skill.STAIR,  # 4
        53: Skill.TERRAIN,  # 5
        54: Skill.STAND_UP,  # 6
    }

    decay = 0.85

    def key_callback(keycode):
        with _lock:
            if keycode == 265:  # UP
                ctrl_cmd["vel_x"] = ctrl_cmd["speed"] * 0.3
            elif keycode == 264:  # DOWN
                ctrl_cmd["vel_x"] = -ctrl_cmd["speed"] * 0.3
            elif keycode == 263:  # LEFT
                ctrl_cmd["ang_vel_z"] = 1.0
            elif keycode == 262:  # RIGHT
                ctrl_cmd["ang_vel_z"] = -1.0
            elif keycode == 32:  # Space
                ctrl_cmd["vel_x"] = 0.0
                ctrl_cmd["ang_vel_z"] = 0.0
            elif keycode == 93:  # ]
                ctrl_cmd["speed"] = min(ctrl_cmd["speed"] + 1.0, 10.0)
                print(f"  Speed: {ctrl_cmd['speed']:.1f}")
            elif keycode == 91:  # [
                ctrl_cmd["speed"] = max(ctrl_cmd["speed"] - 1.0, 1.0)
                print(f"  Speed: {ctrl_cmd['speed']:.1f}")
            elif keycode == 85:  # U — tăng chiều cao (+1cm)
                ctrl_cmd["height_cmd"] = min(ctrl_cmd["height_cmd"] + 0.01, 0.72)
                print(f"  Chiều cao: {ctrl_cmd['height_cmd']:.2f}m")
            elif keycode == 74:  # J — giảm chiều cao (-1cm)
                ctrl_cmd["height_cmd"] = max(ctrl_cmd["height_cmd"] - 0.01, 0.38)
                print(f"  Chiều cao: {ctrl_cmd['height_cmd']:.2f}m")
            elif keycode == 48:  # 0 → auto
                _auto["mode"] = True
                ctrl_cmd["force_skill"] = None
                print("  Mode: AUTO")
            elif keycode in _key_to_skill:
                skill = _key_to_skill[keycode]
                if skill in controller.available_skills:
                    _auto["mode"] = False
                    ctrl_cmd["force_skill"] = skill
                    print(f"  Forced skill: {skill.name}")
                else:
                    print(f"  Skill {skill.name} chưa được tải!")

    console.print("[bold cyan]═══ Điều khiển ═══[/bold cyan]")
    console.print("  ↑/↓ : Tiến / Lùi       ←/→ : Rẽ")
    console.print("  U/J : Chiều cao         Space: Dừng")
    console.print("  [/] : Tốc độ            1-6 : Chọn skill")
    console.print("  0   : Auto\n")

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = max(1, int(control_dt / physics_dt))
    step_count = 0
    smooth_h_cmd_uni = 0.71  # smooth height cho unified mode (mét)
    height_smooth_rate_uni = 0.1

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            with _lock:
                vx = ctrl_cmd["vel_x"]
                wz = ctrl_cmd["ang_vel_z"]
                h_cmd = ctrl_cmd["height_cmd"]  # mục tiêu độ cao (m)
                forced = ctrl_cmd["force_skill"] if not _auto["mode"] else None

                ctrl_cmd["vel_x"] *= decay
                ctrl_cmd["ang_vel_z"] *= decay
                if abs(ctrl_cmd["vel_x"]) < 0.01:
                    ctrl_cmd["vel_x"] = 0.0
                if abs(ctrl_cmd["ang_vel_z"]) < 0.01:
                    ctrl_cmd["ang_vel_z"] = 0.0

            cmd = ControlCommand(vel_x=vx, ang_vel_z=wz, mode=forced)
            ctrl = controller.get_action(mj_data, cmd)

            # Smooth height interpolation
            smooth_h_cmd_uni += (h_cmd - smooth_h_cmd_uni) * height_smooth_rate_uni
            if abs(smooth_h_cmd_uni - h_cmd) < 0.001:
                smooth_h_cmd_uni = h_cmd
            h_eff = smooth_h_cmd_uni

            # Overlay height offset lên các khớp chân
            MIN_H, MAX_H = 0.38, 0.72  # noqa: N806
            if abs(h_eff - 0.71) > 0.005:
                joint_pos = np.array(mj_data.qpos[7:17])
                joint_vel = np.array(mj_data.qvel[6:16])
                # Nội suy target theo h_eff (mét)
                HP_HIGH, KN_HIGH = 0.0, 0.0  # noqa: N806
                HP_KEY, KN_KEY = 0.3, 0.5  # noqa: N806
                HP_LOW, KN_LOW = 1.5, 2.5  # noqa: N806
                if h_eff >= 0.71:
                    t = (h_eff - 0.71) / (MAX_H - 0.71)
                    hp_t = HP_KEY + t * (HP_HIGH - HP_KEY)
                    kn_t = KN_KEY + t * (KN_HIGH - KN_KEY)
                else:
                    t = (0.71 - h_eff) / (0.71 - MIN_H)
                    hp_t = HP_KEY + t * (HP_LOW - HP_KEY)
                    kn_t = KN_KEY + t * (KN_LOW - KN_KEY)
                kp_h = 8.0
                kd_h = 0.8
                if h_eff >= 0.71:
                    blend = min((h_eff - 0.71) / (MAX_H - 0.71), 1.0)
                else:
                    blend = min((0.71 - h_eff) / (0.71 - MIN_H), 1.0)
                for j_idx, target in [(2, hp_t), (7, hp_t), (3, kn_t), (8, kn_t)]:
                    pd = kp_h * (target - joint_pos[j_idx]) - kd_h * joint_vel[j_idx]
                    pd = np.clip(pd, -10, 10)
                    ctrl[j_idx] = (1 - blend) * ctrl[j_idx] + blend * pd

            mj_data.ctrl[:] = ctrl

            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            viewer.sync()

            # Hiển thị skill mỗi 50 step
            step_count += 1
            if step_count % 50 == 0:
                sk = controller.active_skill
                h = float(mj_data.qpos[2])
                print(
                    f"\r  Skill: {sk.name:<12s} | h={h:.2f}m | vx_cmd={vx:.2f} | wz_cmd={wz:.2f}",
                    end="",
                )

            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print()  # newline sau \r


if __name__ == "__main__":
    app()
