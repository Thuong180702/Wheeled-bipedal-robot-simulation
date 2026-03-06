"""
Reward functions cho các task huấn luyện robot.

Mỗi hàm nhận state/observation dạng JAX array và trả về scalar reward.
Tất cả tương thích jax.jit và jax.vmap.
"""

import jax
import jax.numpy as jnp

from wheeled_biped.utils.math_utils import (
    exp_kernel,
    get_gravity_in_body_frame,
    quat_to_euler,
    wrap_angle,
)


# ============================================================
# Reward: Đứng thẳng (upright)
# ============================================================


@jax.jit
def reward_upright(torso_quat: jnp.ndarray) -> jnp.ndarray:
    """Thưởng khi thân robot đứng thẳng.

    Dựa trên thành phần z của trọng lực trong body frame.
    Giá trị 1.0 khi hoàn toàn thẳng, giảm dần khi nghiêng.

    Args:
        torso_quat: quaternion hướng thân [w, x, y, z].

    Returns:
        Reward trong khoảng [0, 1].
    """
    # Trọng lực trong body frame: khi đứng thẳng, gz = -1
    g_body = get_gravity_in_body_frame(torso_quat)
    # cos(tilt_angle) = -gz (gz ~= -1 khi thẳng)
    upright = -g_body[..., 2]
    return jnp.clip(upright, 0.0, 1.0)


# ============================================================
# Reward: Thân nằm ngang (body level - phạt roll/pitch riêng)
# ============================================================


@jax.jit
def reward_body_level(
    torso_quat: jnp.ndarray,
    sigma_roll: float = 0.1,
    sigma_pitch: float = 0.1,
) -> jnp.ndarray:
    """Thưởng khi thân robot nằm ngang song song mặt đất.

    Phạt riêng roll và pitch qua exp_kernel — nhạy hơn reward_upright
    vì cos(10°) ≈ 0.985 (gần 1.0) nhưng exp(-(10°)²/σ²) giảm mạnh.

    Args:
        torso_quat: quaternion hướng thân [w, x, y, z].
        sigma_roll: sigma cho roll (nhỏ = phạt mạnh hơn).
        sigma_pitch: sigma cho pitch.

    Returns:
        Reward trong khoảng [0, 1]. 1.0 khi hoàn toàn nằm ngang.
    """
    euler = quat_to_euler(torso_quat)
    roll = euler[..., 0]
    pitch = euler[..., 1]
    return exp_kernel(roll, sigma_roll) * exp_kernel(pitch, sigma_pitch)


# ============================================================
# Reward: Duy trì chiều cao
# ============================================================


@jax.jit
def reward_height(
    torso_height: jnp.ndarray,
    target_height: float = 0.65,
    sigma: float = 0.1,
) -> jnp.ndarray:
    """Thưởng duy trì chiều cao mong muốn.

    Args:
        torso_height: chiều cao torso (z).
        target_height: chiều cao mục tiêu.
        sigma: độ rộng kernel.

    Returns:
        Reward trong khoảng [0, 1].
    """
    return exp_kernel(torso_height - target_height, sigma)


# ============================================================
# Reward: Bám vận tốc mong muốn (cho locomotion)
# ============================================================


@jax.jit
def reward_tracking_velocity(
    base_vel_x: jnp.ndarray,
    base_vel_y: jnp.ndarray,
    base_ang_vel_z: jnp.ndarray,
    cmd_vel_x: jnp.ndarray,
    cmd_vel_y: jnp.ndarray,
    cmd_ang_vel_z: jnp.ndarray,
    sigma_lin: float = 0.25,
    sigma_ang: float = 0.25,
) -> jnp.ndarray:
    """Thưởng bám theo vận tốc lệnh.

    Args:
        base_vel_x: vận tốc thực tế theo x.
        base_vel_y: vận tốc thực tế theo y.
        base_ang_vel_z: vận tốc xoay thực tế quanh z.
        cmd_*: vận tốc mong muốn.
        sigma_lin: sigma cho vận tốc tuyến tính.
        sigma_ang: sigma cho vận tốc góc.

    Returns:
        Reward trong khoảng [0, 1].
    """
    lin_error = jnp.sqrt(
        jnp.square(base_vel_x - cmd_vel_x) + jnp.square(base_vel_y - cmd_vel_y)
    )
    ang_error = jnp.abs(base_ang_vel_z - cmd_ang_vel_z)

    r_lin = exp_kernel(lin_error, sigma_lin)
    r_ang = exp_kernel(ang_error, sigma_ang)
    return 0.7 * r_lin + 0.3 * r_ang


# ============================================================
# Reward: Hướng đi (heading)
# ============================================================


@jax.jit
def reward_heading(torso_quat: jnp.ndarray, target_yaw: jnp.ndarray) -> jnp.ndarray:
    """Thưởng khi robot hướng đúng mục tiêu.

    Args:
        torso_quat: quaternion hướng thân.
        target_yaw: góc yaw mong muốn.

    Returns:
        Reward [0, 1].
    """
    euler = quat_to_euler(torso_quat)
    yaw_error = wrap_angle(euler[..., 2] - target_yaw)
    return exp_kernel(yaw_error, sigma=0.5)


# ============================================================
# Phạt: Mô-men xoắn (joint torque penalty)
# ============================================================


@jax.jit
def penalty_joint_torque(torques: jnp.ndarray) -> jnp.ndarray:
    """Phạt dùng mô-men lớn (tiết kiệm năng lượng).

    Args:
        torques: vector mô-men các khớp.

    Returns:
        Penalty (giá trị dương, cần nhân hệ số âm).
    """
    return jnp.sum(jnp.square(torques), axis=-1)


# ============================================================
# Phạt: Tốc độ khớp lớn
# ============================================================


@jax.jit
def penalty_joint_velocity(joint_vel: jnp.ndarray) -> jnp.ndarray:
    """Phạt vận tốc khớp quá lớn.

    Args:
        joint_vel: vận tốc các khớp.

    Returns:
        Penalty (giá trị dương).
    """
    return jnp.sum(jnp.square(joint_vel), axis=-1)


# ============================================================
# Phạt: Thay đổi action đột ngột (action rate)
# ============================================================


@jax.jit
def penalty_action_rate(
    current_action: jnp.ndarray,
    previous_action: jnp.ndarray,
) -> jnp.ndarray:
    """Phạt thay đổi action quá nhanh.

    Args:
        current_action: action hiện tại.
        previous_action: action bước trước.

    Returns:
        Penalty (giá trị dương).
    """
    return jnp.sum(jnp.square(current_action - previous_action), axis=-1)


# ============================================================
# Reward: Sống sót (alive bonus)
# ============================================================


@jax.jit
def reward_alive(is_alive: jnp.ndarray) -> jnp.ndarray:
    """Thưởng mỗi step robot còn hoạt động.

    Args:
        is_alive: boolean array (True = còn sống).

    Returns:
        1.0 nếu còn sống, 0.0 nếu ngã.
    """
    return is_alive.astype(jnp.float32)


# ============================================================
# Reward: Đứng yên (no linear motion)
# ============================================================


@jax.jit
def reward_no_motion(base_lin_vel: jnp.ndarray, sigma: float = 0.1) -> jnp.ndarray:
    """Thưởng khi robot đứng yên, không di chuyển.

    Args:
        base_lin_vel: vận tốc tuyến tính body frame (3,).
        sigma: độ rộng kernel.

    Returns:
        Reward trong khoảng [0, 1]. 1.0 khi đứng yên hoàn toàn.
    """
    vel_norm = jnp.sqrt(jnp.sum(jnp.square(base_lin_vel)))
    return exp_kernel(vel_norm, sigma)


# ============================================================
# Reward: Giữ tư thế mặc định (default pose)
# ============================================================


@jax.jit
def reward_default_pose(
    joint_pos: jnp.ndarray,
    default_pos: jnp.ndarray,
    mask: jnp.ndarray,
    sigma: float = 0.5,
) -> jnp.ndarray:
    """Thưởng khi tư thế khớp gần với tư thế đứng mặc định.

    Args:
        joint_pos: vị trí khớp hiện tại (10,).
        default_pos: vị trí khớp mặc định (10,).
        mask: mask loại trừ bánh xe (10,). 1.0 cho khớp cần so sánh, 0.0 cho bánh xe.
        sigma: độ rộng kernel.

    Returns:
        Reward trong khoảng [0, 1]. 1.0 khi đúng tư thế mặc định.
    """
    diff = (joint_pos - default_pos) * mask
    error = jnp.sqrt(jnp.sum(jnp.square(diff)))
    return exp_kernel(error, sigma)


# ============================================================
# Reward: Chân hướng thẳng về phía trước (hip_yaw ≈ 0)
# ============================================================


@jax.jit
def reward_legs_forward(
    joint_pos: jnp.ndarray,
    sigma: float = 0.15,
) -> jnp.ndarray:
    """Thưởng khi 2 chân hướng thẳng về phía trước (hip_yaw ≈ 0).

    Hip yaw ≠ 0 nghĩa là chân xoay sang bên → bánh xe không thẳng.

    Args:
        joint_pos: vị trí 10 khớp. Index 1=l_hip_yaw, 6=r_hip_yaw.
        sigma: độ rộng kernel.

    Returns:
        Reward [0, 1]. 1.0 khi cả 2 chân hướng thẳng trước.
    """
    yaw_left = joint_pos[..., 1]
    yaw_right = joint_pos[..., 6]
    return exp_kernel(yaw_left, sigma) * exp_kernel(yaw_right, sigma)


# ============================================================
# Reward: Chân vuông góc với mặt đất (hip_roll ≈ 0)
# ============================================================


@jax.jit
def reward_legs_vertical(
    joint_pos: jnp.ndarray,
    sigma: float = 0.15,
) -> jnp.ndarray:
    """Thưởng khi 2 chân vuông góc với mặt đất (hip_roll ≈ 0).

    Hip roll ≠ 0 nghĩa là chân nghiêng sang bên.

    Args:
        joint_pos: vị trí 10 khớp. Index 0=l_hip_roll, 5=r_hip_roll.
        sigma: độ rộng kernel.

    Returns:
        Reward [0, 1]. 1.0 khi cả 2 chân thẳng đứng.
    """
    roll_left = joint_pos[..., 0]
    roll_right = joint_pos[..., 5]
    return exp_kernel(roll_left, sigma) * exp_kernel(roll_right, sigma)


# ============================================================
# Phạt: Bánh xe quay (wheel velocity penalty)
# ============================================================


@jax.jit
def penalty_wheel_velocity(
    joint_vel: jnp.ndarray,
) -> jnp.ndarray:
    """Phạt bánh xe quay — giữ robot đứng yên tại chỗ.

    Args:
        joint_vel: vận tốc 10 khớp. Index 4=l_wheel, 9=r_wheel.

    Returns:
        Penalty (giá trị dương). Tổng bình phương vận tốc 2 bánh.
    """
    return jnp.square(joint_vel[..., 4]) + jnp.square(joint_vel[..., 9])


# ============================================================
# Reward: Tiếp xúc chân (foot contact)
# ============================================================


@jax.jit
def reward_foot_contact(
    left_contact: jnp.ndarray,
    right_contact: jnp.ndarray,
    desired_left: jnp.ndarray,
    desired_right: jnp.ndarray,
) -> jnp.ndarray:
    """Thưởng khi tiếp xúc chân đúng nhịp (cho walking).

    Args:
        left_contact: True nếu chân trái chạm đất.
        right_contact: True nếu chân phải chạm đất.
        desired_left: True nếu chân trái nên chạm.
        desired_right: True nếu chân phải nên chạm.

    Returns:
        Reward [0, 1].
    """
    l_match = (left_contact == desired_left).astype(jnp.float32)
    r_match = (right_contact == desired_right).astype(jnp.float32)
    return 0.5 * (l_match + r_match)


# ============================================================
# Reward: Nâng chân đủ cao (foot clearance - cho walking/stairs)
# ============================================================


@jax.jit
def reward_foot_clearance(
    foot_height: jnp.ndarray,
    target_clearance: float = 0.05,
    is_swing: jnp.ndarray = None,
) -> jnp.ndarray:
    """Thưởng nâng chân đủ cao khi ở pha swing.

    Args:
        foot_height: chiều cao bàn chân (z).
        target_clearance: chiều cao nâng mong muốn.
        is_swing: True nếu chân đang ở pha swing.

    Returns:
        Reward.
    """
    clearance_reward = exp_kernel(foot_height - target_clearance, sigma=0.02)
    if is_swing is not None:
        # Chỉ tính khi chân đang pha swing
        clearance_reward = clearance_reward * is_swing.astype(jnp.float32)
    return clearance_reward


# ============================================================
# Reward: Dáng đi đối xứng (gait symmetry)
# ============================================================


@jax.jit
def reward_gait_symmetry(
    left_joint_pos: jnp.ndarray,
    right_joint_pos: jnp.ndarray,
    phase_offset: float = jnp.pi,
) -> jnp.ndarray:
    """Thưởng dáng đi đối xứng giữa 2 chân.

    Hai chân nên lệch pha 180° (đối bước).

    Args:
        left_joint_pos: vị trí khớp chân trái.
        right_joint_pos: vị trí khớp chân phải.
        phase_offset: pha lệch mong muốn (π cho đi bộ).

    Returns:
        Reward [0, 1].
    """
    diff = jnp.sum(jnp.square(left_joint_pos - right_joint_pos), axis=-1)
    sim = exp_kernel(diff, sigma=0.5)
    # phase_offset=0: reward same position (standing/balance)
    # phase_offset=π: reward different positions (anti-phase walking)
    return jnp.where(phase_offset > jnp.pi / 2, 1.0 - sim, sim)


# ============================================================
# Reward: Đối xứng 2 chân (cho balance — 2 chân giống nhau)
# ============================================================


@jax.jit
def reward_leg_symmetry(
    joint_pos: jnp.ndarray,
    sigma: float = 0.5,
) -> jnp.ndarray:
    """Thưởng khi 2 chân có tư thế đối xứng (giống nhau).

    So sánh các khớp tương ứng trái-phải: hip_roll, hip_yaw, hip_pitch, knee.
    Không so sánh bánh xe (index 4, 9).

    Args:
        joint_pos: vị trí 10 khớp [l_hr, l_hy, l_hp, l_kn, l_wh, r_hr, r_hy, r_hp, r_kn, r_wh].
        sigma: độ rộng kernel.

    Returns:
        Reward trong khoảng [0, 1]. 1.0 khi 2 chân hoàn toàn giống nhau.
    """
    left = joint_pos[..., :4]  # l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee
    right = joint_pos[..., 5:9]  # r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee
    diff = jnp.sum(jnp.square(left - right), axis=-1)
    return exp_kernel(jnp.sqrt(diff), sigma)


# ============================================================
# Phạt: Body stability (phạt rung lắc)
# ============================================================


@jax.jit
def penalty_body_angular_velocity(
    angular_vel: jnp.ndarray,
) -> jnp.ndarray:
    """Phạt vận tốc góc thân quá lớn (rung lắc).

    Args:
        angular_vel: vận tốc góc thân [wx, wy, wz].

    Returns:
        Penalty (giá trị dương).
    """
    return jnp.sum(jnp.square(angular_vel), axis=-1)


# ============================================================
# Reward: Tiến bộ leo cầu thang (forward + height progress)
# ============================================================


@jax.jit
def reward_stair_progress(
    current_pos: jnp.ndarray,
    previous_pos: jnp.ndarray,
    forward_weight: float = 1.0,
    height_weight: float = 1.5,
) -> jnp.ndarray:
    """Thưởng dựa trên tiến bộ vị trí (tiến + leo cao).

    Args:
        current_pos: vị trí hiện tại [x, y, z].
        previous_pos: vị trí bước trước [x, y, z].
        forward_weight: trọng số tiến phía trước.
        height_weight: trọng số leo cao.

    Returns:
        Reward.
    """
    # Y axis = forward direction for this robot
    dy = current_pos[..., 1] - previous_pos[..., 1]
    dz = current_pos[..., 2] - previous_pos[..., 2]
    return forward_weight * dy + height_weight * jnp.clip(dz, 0.0, None)


# ============================================================
# Reward: Tiến bộ chiều cao (cho stand-up)
# ============================================================


@jax.jit
def reward_height_progress(
    current_height: jnp.ndarray,
    previous_height: jnp.ndarray,
    scale: float = 5.0,
) -> jnp.ndarray:
    """Thưởng khi robot nâng chiều cao lên (đứng dậy).

    Args:
        current_height: chiều cao torso hiện tại (z).
        previous_height: chiều cao torso bước trước (z).
        scale: hệ số scale cho delta height.

    Returns:
        Reward — dương khi đi lên, âm khi đi xuống.
    """
    dz = current_height - previous_height
    return scale * dz


@jax.jit
def reward_stand_up_phase(
    torso_height: jnp.ndarray,
    torso_quat: jnp.ndarray,
    target_height: float = 0.65,
    height_sigma: float = 0.15,
) -> jnp.ndarray:
    """Reward tổng hợp cho pha đứng dậy.

    Kết hợp: gần target height + đứng thẳng.
    Bonus cao khi đạt cả 2 điều kiện.

    Args:
        torso_height: chiều cao torso.
        torso_quat: quaternion hướng torso.
        target_height: chiều cao mục tiêu.
        height_sigma: sigma cho kernel chiều cao.

    Returns:
        Reward [0, 1].
    """
    r_h = exp_kernel(torso_height - target_height, height_sigma)
    r_up = reward_upright(torso_quat)
    # Nhân 2 thành phần để đòi hỏi cả 2
    return r_h * r_up


# ============================================================
# Tổng hợp: Hàm tính tổng reward
# ============================================================


def compute_total_reward(
    reward_components: dict[str, jnp.ndarray],
    reward_weights: dict[str, float],
) -> jnp.ndarray:
    """Tính tổng reward có trọng số.

    Args:
        reward_components: dict tên → giá trị reward.
        reward_weights: dict tên → trọng số.

    Returns:
        Tổng reward.
    """
    total = jnp.zeros(())
    for name, value in reward_components.items():
        weight = reward_weights.get(name, 0.0)
        total = total + weight * value
    return total
