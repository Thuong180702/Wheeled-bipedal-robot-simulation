"""
Script trực quan hóa robot trong MuJoCo viewer.

Cách dùng:
  # Xem robot ở tư thế mặc định
  python scripts/visualize.py model

  # Xem policy đã train
  python scripts/visualize.py policy --checkpoint outputs/checkpoints/balance/final

  # Render video
  python scripts/visualize.py render --checkpoint outputs/checkpoints/balance/final --output video.mp4
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
):
    """Mô phỏng robot với policy đã train trong viewer."""
    import pickle

    import jax
    import jax.numpy as jnp
    import mujoco
    import mujoco.viewer

    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.config import get_model_path

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
    mujoco.mj_forward(mj_model, mj_data)

    console.print(f"[green]Đang chạy policy từ: {checkpoint}[/green]")
    console.print(f"  Steps: {num_steps}, Slow: {slow_factor}x")

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for step in range(num_steps):
            if not viewer.is_running():
                break

            step_start = time.time()

            # Trích xuất observation (simplified - CPU version)
            from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

            torso_quat = jnp.array(mj_data.qpos[3:7])
            gravity_body = get_gravity_in_body_frame(torso_quat)

            obs = jnp.concatenate(
                [
                    gravity_body,
                    jnp.array(mj_data.qvel[:3]),
                    jnp.array(mj_data.qvel[3:6]),
                    jnp.array(mj_data.qpos[7:17]),
                    jnp.array(mj_data.qvel[6:16]),
                    jnp.array(mj_data.ctrl[:10]),
                ]
            )

            # Normalize + get action
            obs_norm = normalize_obs(obs, obs_rms)
            dist, _ = model.apply(params, obs_norm)
            action = dist.loc  # deterministic
            action = jnp.clip(action, -1.0, 1.0)

            # Scale action
            ctrl_range = mj_model.actuator_ctrlrange
            ctrl = ctrl_range[:, 0] + (action + 1) * 0.5 * (
                ctrl_range[:, 1] - ctrl_range[:, 0]
            )
            mj_data.ctrl[:] = ctrl

            # Physics steps
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            viewer.sync()

            # Timing
            elapsed = time.time() - step_start
            sleep_time = control_dt * slow_factor - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


@app.command()
def render(
    checkpoint: str = typer.Option(..., help="Đường dẫn checkpoint."),
    output: str = typer.Option("video.mp4", help="File video output."),
    num_steps: int = typer.Option(500, help="Số bước."),
    width: int = typer.Option(1280, help="Chiều rộng video."),
    height: int = typer.Option(720, help="Chiều cao video."),
    fps: int = typer.Option(50, help="FPS video."),
):
    """Render video từ policy đã train."""
    import pickle

    import jax
    import jax.numpy as jnp
    import mediapy
    import mujoco
    import numpy as np

    from wheeled_biped.training.networks import create_actor_critic
    from wheeled_biped.training.ppo import normalize_obs
    from wheeled_biped.utils.config import get_model_path
    from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

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
    mujoco.mj_forward(mj_model, mj_data)

    # Renderer
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = int(control_dt / physics_dt)

    frames = []
    console.print(f"Rendering {num_steps} steps...")

    for step in range(num_steps):
        torso_quat = jnp.array(mj_data.qpos[3:7])
        gravity_body = get_gravity_in_body_frame(torso_quat)

        obs = jnp.concatenate(
            [
                gravity_body,
                jnp.array(mj_data.qvel[:3]),
                jnp.array(mj_data.qvel[3:6]),
                jnp.array(mj_data.qpos[7:17]),
                jnp.array(mj_data.qvel[6:16]),
                jnp.array(mj_data.ctrl[:10]),
            ]
        )

        obs_norm = normalize_obs(obs, obs_rms)
        dist, _ = model.apply(params, obs_norm)
        action = jnp.clip(dist.loc, -1.0, 1.0)

        ctrl_range = mj_model.actuator_ctrlrange
        ctrl = ctrl_range[:, 0] + (action + 1) * 0.5 * (
            ctrl_range[:, 1] - ctrl_range[:, 0]
        )
        mj_data.ctrl[:] = np.array(ctrl)

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        renderer.update_scene(mj_data, camera="track")
        frames.append(renderer.render())

    # Lưu video
    mediapy.write_video(output, frames, fps=fps)
    console.print(f"[green]Đã lưu video: {output}[/green]")


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
):
    """Điều khiển robot bằng bàn phím trong MuJoCo viewer.

    \b
    Phím điều khiển:
      W / S  : Tiến / Lùi (lực bánh xe)
      A / D  : Rẽ trái / phải (chênh lệch lực bánh)
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

    from wheeled_biped.utils.config import get_model_path

    path = model_path or str(get_model_path())
    console.print(f"Đang mở model: {path}")

    mj_model = mujoco.MjModel.from_xml_path(path)
    mj_data = mujoco.MjData(mj_model)

    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    # ---------- trạng thái điều khiển ----------
    # Actuator order: l_hip_roll(0), l_hip_pitch(1), l_knee(2), l_ankle(3), l_wheel(4)
    #                 r_hip_roll(5), r_hip_pitch(6), r_knee(7), r_ankle(8), r_wheel(9)
    ctrl_state = {
        "forward": 0.0,  # -1 .. +1
        "turn": 0.0,  # -1 .. +1
        "roll": 0.0,  # -1 .. +1
        "speed": 5.0,  # wheel torque magnitude (Nm)
        "roll_gain": 3.0,  # hip roll torque magnitude
        "height_offset": 0.0,  # \u0111i\u1ec1u ch\u1ec9nh chi\u1ec1u cao: -1 (ng\u1ed3i) .. 0 (\u0111\u1ee9ng) .. +0.3 (ki\u1ec5ng)
    }
    _lock = threading.Lock()

    # ---------- PD standing gains ----------
    # Khi không có checkpoint, sử dụng PD controller đơn giản để giữ thăng bằng
    # Target joint positions (standing keyframe)
    standing_qpos = (
        np.array(mj_data.qpos[7:17], copy=True) if mj_model.nkey > 0 else np.zeros(10)
    )
    kp_joints = np.array(
        [8.0, 12.0, 10.0, 6.0, 0, 8.0, 12.0, 10.0, 6.0, 0]
    )  # 0 cho bánh
    kd_joints = np.array([0.5, 0.8, 0.6, 0.3, 0, 0.5, 0.8, 0.6, 0.3, 0])

    # Tải policy nếu có
    policy_fn = None
    if checkpoint:
        try:
            import jax
            import jax.numpy as jnp

            from wheeled_biped.training.networks import create_actor_critic
            from wheeled_biped.training.ppo import normalize_obs
            from wheeled_biped.utils.math_utils import get_gravity_in_body_frame

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

            def _policy(data):
                torso_quat = jnp.array(data.qpos[3:7])
                gravity_body = get_gravity_in_body_frame(torso_quat)
                obs = jnp.concatenate(
                    [
                        gravity_body,
                        jnp.array(data.qvel[:3]),
                        jnp.array(data.qvel[3:6]),
                        jnp.array(data.qpos[7:17]),
                        jnp.array(data.qvel[6:16]),
                        jnp.array(data.ctrl[:10]),
                    ]
                )
                obs_norm = normalize_obs(obs, obs_rms)
                dist, _ = network.apply(params, obs_norm)
                action = jnp.clip(dist.loc, -1.0, 1.0)
                ctrl_range = mj_model.actuator_ctrlrange
                ctrl = ctrl_range[:, 0] + (action + 1) * 0.5 * (
                    ctrl_range[:, 1] - ctrl_range[:, 0]
                )
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
            elif keycode == 85:  # U — tăng chiều cao
                ctrl_state["height_offset"] = min(
                    ctrl_state["height_offset"] + 0.1, 0.3
                )
                print(f"  Height offset: {ctrl_state['height_offset']:+.1f}")
            elif keycode == 74:  # J — giảm chiều cao (gập chân)
                ctrl_state["height_offset"] = max(
                    ctrl_state["height_offset"] - 0.1, -1.0
                )
                print(f"  Height offset: {ctrl_state['height_offset']:+.1f}")

    console.print("\n[bold cyan]═══ Chế độ điều khiển tương tác ═══[/bold cyan]")
    console.print("  ↑/↓ : Tiến / Lùi")
    console.print("  ←/→ : Rẽ trái / phải")
    console.print("  Q/E : Nghiêng trái / phải (hip roll)")
    console.print("  U/J : Tăng / Giảm chiều cao robot")
    console.print("  Space : Dừng lại")
    console.print("  [ / ] : Giảm / Tăng tốc độ")
    console.print("  Nhấn R trong viewer để reset\n")
    if policy_fn:
        console.print("  [green]Mode: Policy + keyboard override[/green]")
    else:
        console.print("  [yellow]Mode: PD controller + keyboard[/yellow]")

    control_dt = 0.02
    physics_dt = mj_model.opt.timestep
    n_substeps = max(1, int(control_dt / physics_dt))

    # Decay factor — phím giữ = giá trị 1, thả phím → decay dần về 0
    decay = 0.85

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            with _lock:
                fwd = ctrl_state["forward"]
                trn = ctrl_state["turn"]
                rll = ctrl_state["roll"]
                spd = ctrl_state["speed"]
                rg = ctrl_state["roll_gain"]
                h_off = ctrl_state["height_offset"]  # -1 (ngồi) .. 0 (đứng) .. +0.3

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

            # Tính target khớp theo chiều cao
            # h_off < 0: gập đùi (hip_pitch +), gập gối (knee +), chỉnh ankle
            # Actuator order: hip_roll(0), hip_pitch(1), knee(2), ankle(3), wheel(4)
            height_target = standing_qpos.copy()
            if h_off != 0.0:
                # Gập đùi: hip_pitch tăng khi ngồi (h_off < 0 → gập nhiều hơn)
                height_target[1] += (-h_off) * 0.6   # l_hip_pitch
                height_target[6] += (-h_off) * 0.6   # r_hip_pitch
                # Gập gối: knee tăng khi ngồi
                height_target[2] += (-h_off) * 1.2   # l_knee
                height_target[7] += (-h_off) * 1.2   # r_knee
                # Chỉnh ankle để bàn chân vẫn phẳng
                height_target[3] += (-h_off) * (-0.6)  # l_ankle
                height_target[8] += (-h_off) * (-0.6)  # r_ankle

            if policy_fn is not None:
                # Dùng policy để giữ thăng bằng, overlay keyboard commands lên bánh xe
                base_ctrl = policy_fn(mj_data)
                # Override wheel torques
                wheel_left = fwd * spd - trn * spd * 0.5
                wheel_right = fwd * spd + trn * spd * 0.5
                base_ctrl[4] = np.clip(base_ctrl[4] + wheel_left, -10, 10)
                base_ctrl[9] = np.clip(base_ctrl[9] + wheel_right, -10, 10)
                # Override hip roll
                base_ctrl[0] += rll * rg
                base_ctrl[5] += rll * rg
                # Overlay height control — PD bổ sung trên các khớp chân
                if h_off != 0.0:
                    joint_pos = np.array(mj_data.qpos[7:17])
                    joint_vel = np.array(mj_data.qvel[6:16])
                    for idx in [1, 2, 3, 6, 7, 8]:  # hip_pitch, knee, ankle (2 bên)
                        delta = kp_joints[idx] * (height_target[idx] - joint_pos[idx]) \
                                - kd_joints[idx] * joint_vel[idx]
                        base_ctrl[idx] = np.clip(base_ctrl[idx] + delta, -10, 10)
                mj_data.ctrl[:] = np.clip(
                    base_ctrl,
                    mj_model.actuator_ctrlrange[:, 0],
                    mj_model.actuator_ctrlrange[:, 1],
                )
            else:
                # PD controller giữ tư thế + height control
                joint_pos = np.array(mj_data.qpos[7:17])
                joint_vel = np.array(mj_data.qvel[6:16])
                pd_ctrl = (
                    kp_joints * (height_target - joint_pos) - kd_joints * joint_vel
                )

                # Overlay keyboard
                wheel_left = fwd * spd - trn * spd * 0.5
                wheel_right = fwd * spd + trn * spd * 0.5
                pd_ctrl[4] = wheel_left  # l_wheel
                pd_ctrl[9] = wheel_right  # r_wheel
                pd_ctrl[0] += rll * rg  # l_hip_roll
                pd_ctrl[5] += rll * rg  # r_hip_roll

                mj_data.ctrl[:] = np.clip(
                    pd_ctrl,
                    mj_model.actuator_ctrlrange[:, 0],
                    mj_model.actuator_ctrlrange[:, 1],
                )

            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            viewer.sync()

            elapsed = time.time() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


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
    mujoco.mj_forward(mj_model, mj_data)

    console.print("\n[bold green]═══ Unified Controller ═══[/bold green]")
    console.print(f"  Checkpoint dir: {checkpoint_dir}")
    console.print(f"  Auto mode: {auto_mode}\n")

    controller = UnifiedController(
        checkpoint_dir=checkpoint_dir,
        mj_model=mj_model,
    )

    # ---------- trạng thái điều khiển ----------
    ctrl_cmd = {"vel_x": 0.0, "ang_vel_z": 0.0, "speed": 5.0, "force_skill": None,
                "height_offset": 0.0}
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
            elif keycode == 85:  # U — tăng chiều cao
                ctrl_cmd["height_offset"] = min(
                    ctrl_cmd["height_offset"] + 0.1, 0.3
                )
                print(f"  Height offset: {ctrl_cmd['height_offset']:+.1f}")
            elif keycode == 74:  # J — giảm chiều cao
                ctrl_cmd["height_offset"] = max(
                    ctrl_cmd["height_offset"] - 0.1, -1.0
                )
                print(f"  Height offset: {ctrl_cmd['height_offset']:+.1f}")
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

    with mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            with _lock:
                vx = ctrl_cmd["vel_x"]
                wz = ctrl_cmd["ang_vel_z"]
                h_off = ctrl_cmd["height_offset"]
                forced = ctrl_cmd["force_skill"] if not _auto["mode"] else None

                ctrl_cmd["vel_x"] *= decay
                ctrl_cmd["ang_vel_z"] *= decay
                if abs(ctrl_cmd["vel_x"]) < 0.01:
                    ctrl_cmd["vel_x"] = 0.0
                if abs(ctrl_cmd["ang_vel_z"]) < 0.01:
                    ctrl_cmd["ang_vel_z"] = 0.0

            cmd = ControlCommand(vel_x=vx, ang_vel_z=wz, mode=forced)
            ctrl = controller.get_action(mj_data, cmd)

            # Overlay height offset lên các khớp chân
            if h_off != 0.0:
                joint_pos = np.array(mj_data.qpos[7:17])
                joint_vel = np.array(mj_data.qvel[6:16])
                # Tính target từ height offset
                kp = np.array([8.0, 12.0, 10.0, 6.0, 0, 8.0, 12.0, 10.0, 6.0, 0])
                kd = np.array([0.5, 0.8, 0.6, 0.3, 0, 0.5, 0.8, 0.6, 0.3, 0])
                for idx, gain_h in [(1, 0.6), (2, 1.2), (3, -0.6),
                                    (6, 0.6), (7, 1.2), (8, -0.6)]:
                    target_delta = (-h_off) * gain_h
                    delta = kp[idx] * target_delta - kd[idx] * joint_vel[idx]
                    ctrl[idx] = np.clip(ctrl[idx] + delta, -10, 10)

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
