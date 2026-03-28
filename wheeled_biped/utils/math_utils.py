"""
Các hàm toán học cho xử lý quaternion, rotation, vector - chạy trên JAX.

Tất cả hàm đều tương thích với jax.jit và jax.vmap.
"""

import jax
import jax.numpy as jnp

# ============================================================
# Quaternion Utilities (quat = [w, x, y, z])
# ============================================================


@jax.jit
def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Nhân hai quaternion: q1 * q2.

    Args:
        q1: quaternion [w, x, y, z]
        q2: quaternion [w, x, y, z]

    Returns:
        Quaternion tích [w, x, y, z].
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return jnp.stack([w, x, y, z], axis=-1)


@jax.jit
def quat_conjugate(q: jnp.ndarray) -> jnp.ndarray:
    """Liên hợp quaternion (nghịch đảo nếu đơn vị)."""
    return q * jnp.array([1.0, -1.0, -1.0, -1.0])


@jax.jit
def quat_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Xoay vector v bằng quaternion q.

    Args:
        q: quaternion [w, x, y, z]
        v: vector [x, y, z]

    Returns:
        Vector đã xoay [x, y, z].
    """
    # Mở rộng v thành quaternion thuần [0, vx, vy, vz]
    v_quat = jnp.concatenate([jnp.zeros_like(v[..., :1]), v], axis=-1)
    # q * v * q_conj
    result = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return result[..., 1:]


@jax.jit
def quat_to_euler(q: jnp.ndarray) -> jnp.ndarray:
    """Chuyển quaternion sang góc Euler (roll, pitch, yaw).

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        Góc Euler [roll, pitch, yaw] (rad).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jnp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = jnp.clip(sinp, -1.0, 1.0)
    pitch = jnp.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)

    return jnp.stack([roll, pitch, yaw], axis=-1)


@jax.jit
def quat_to_rot_matrix(q: jnp.ndarray) -> jnp.ndarray:
    """Chuyển quaternion sang ma trận xoay 3×3.

    Args:
        q: quaternion [w, x, y, z]

    Returns:
        Ma trận xoay (3, 3).
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)

    return jnp.stack(
        [
            jnp.stack([r00, r01, r02], axis=-1),
            jnp.stack([r10, r11, r12], axis=-1),
            jnp.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


# ============================================================
# Projection & Gravity
# ============================================================


@jax.jit
def get_gravity_in_body_frame(quat: jnp.ndarray) -> jnp.ndarray:
    """Tính vector trọng lực trong hệ tọa độ thân robot.

    Args:
        quat: quaternion hướng thân [w, x, y, z]

    Returns:
        Vector trọng lực trong body frame [gx, gy, gz].
    """
    gravity_world = jnp.array([0.0, 0.0, -1.0])
    return quat_rotate(quat_conjugate(quat), gravity_world)


@jax.jit
def project_gravity(quat: jnp.ndarray) -> jnp.ndarray:
    """Tính thành phần trọng lực chiếu lên trục x, y (nghiêng).

    Returns:
        [gx, gy] - dùng để đánh giá mức độ nghiêng.
    """
    g_body = get_gravity_in_body_frame(quat)
    return g_body[..., :2]


# ============================================================
# Normalization & Scaling
# ============================================================


@jax.jit
def normalize(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Chuẩn hóa vector về đơn vị."""
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + eps)


@jax.jit
def wrap_angle(angle: jnp.ndarray) -> jnp.ndarray:
    """Gói góc về khoảng [-π, π]."""
    return (angle + jnp.pi) % (2 * jnp.pi) - jnp.pi


@jax.jit
def smooth_abs(x: jnp.ndarray, eps: float = 0.01) -> jnp.ndarray:
    """Giá trị tuyệt đối mượt (differentiable ở x=0)."""
    return jnp.sqrt(x * x + eps * eps) - eps


@jax.jit
def exp_kernel(x: jnp.ndarray, sigma: float = 0.25) -> jnp.ndarray:
    """Kernel Gaussian: exp(-x²/σ²). Dùng cho reward shaping."""
    return jnp.exp(-jnp.square(x) / (sigma * sigma))
