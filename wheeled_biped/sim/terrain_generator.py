"""
Tạo địa hình (terrain) cho môi trường huấn luyện.

Hỗ trợ:
  - Mặt phẳng (flat)
  - Cầu thang (stairs)
  - Địa hình gồ ghề (heightfield - random bumps)

Tương thích MuJoCo MJCF và MJX.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def generate_stair_terrain(
    num_steps: int = 8,
    step_height: float = 0.15,
    step_depth: float = 0.30,
    step_width: float = 1.0,
    platform_length: float = 1.0,
) -> str:
    """Tạo XML snippet cho cầu thang.

    Cầu thang bắt đầu từ y=platform_length (robot hướng Y+), leo lên rồi xuống.

    Args:
        num_steps: số bậc thang.
        step_height: chiều cao mỗi bậc (m).
        step_depth: chiều sâu mỗi bậc (m).
        step_width: chiều rộng bậc (m).
        platform_length: chiều dài sàn phẳng trước cầu thang (m).

    Returns:
        Chuỗi XML chứa các geom bậc thang.
    """
    geoms = []

    # Sàn phẳng đầu (dọc trục Y)
    geoms.append(
        f'<geom name="platform_start" type="box" '
        f'pos="0 {platform_length / 2} {-step_height / 2}" '
        f'size="{step_width / 2} {platform_length / 2} {step_height / 2}" '
        f'rgba="0.6 0.6 0.6 1" condim="3" contype="1" conaffinity="1"/>'
    )

    # Bậc thang đi lên (dọc trục Y)
    for i in range(num_steps):
        y = platform_length + (i + 0.5) * step_depth
        z = (i + 1) * step_height - step_height / 2
        h = (i + 1) * step_height / 2
        geoms.append(
            f'<geom name="stair_up_{i}" type="box" '
            f'pos="0 {y:.4f} {z:.4f}" '
            f'size="{step_width / 2} {step_depth / 2} {h:.4f}" '
            f'rgba="0.55 0.55 0.58 1" condim="3" contype="1" conaffinity="1"/>'
        )

    # Sàn trên cùng
    top_y = platform_length + num_steps * step_depth + platform_length / 2
    top_z = num_steps * step_height - step_height / 2
    geoms.append(
        f'<geom name="platform_top" type="box" '
        f'pos="0 {top_y:.4f} {top_z:.4f}" '
        f'size="{step_width / 2} {platform_length / 2} {step_height / 2}" '
        f'rgba="0.6 0.6 0.6 1" condim="3" contype="1" conaffinity="1"/>'
    )

    # Bậc thang đi xuống (dọc trục Y)
    for i in range(num_steps):
        y = top_y + platform_length / 2 + (i + 0.5) * step_depth
        remaining = num_steps - i
        z = remaining * step_height - step_height / 2
        h = remaining * step_height / 2
        geoms.append(
            f'<geom name="stair_down_{i}" type="box" '
            f'pos="0 {y:.4f} {z:.4f}" '
            f'size="{step_width / 2} {step_depth / 2} {h:.4f}" '
            f'rgba="0.55 0.55 0.58 1" condim="3" contype="1" conaffinity="1"/>'
        )

    return "\n    ".join(geoms)


def generate_heightfield_data(
    size_x: float = 10.0,
    size_y: float = 4.0,
    resolution: float = 0.05,
    max_height: float = 0.05,
    frequency: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """Tạo dữ liệu heightfield cho địa hình gồ ghề.

    Dùng nhiễu Perlin-like (sum of sinusoids) để tạo terrain.

    Args:
        size_x: kích thước theo x (m).
        size_y: kích thước theo y (m).
        resolution: độ phân giải (m/pixel).
        max_height: chiều cao cực đại (m).
        frequency: tần số gợn sóng.
        seed: random seed.

    Returns:
        Ma trận heightfield (nrow, ncol) kiểu float32.
    """
    rng = np.random.RandomState(seed)

    ncol = int(size_x / resolution)
    nrow = int(size_y / resolution)

    x = np.linspace(0, size_x * frequency, ncol)
    y = np.linspace(0, size_y * frequency, nrow)
    xx, yy = np.meshgrid(x, y)

    # Tổng hợp nhiều tần số
    terrain = np.zeros((nrow, ncol), dtype=np.float32)
    for octave in range(4):
        freq = 2**octave
        amp = 1.0 / freq
        phase_x = rng.uniform(0, 2 * np.pi)
        phase_y = rng.uniform(0, 2 * np.pi)
        terrain += amp * np.sin(freq * xx + phase_x) * np.cos(freq * yy + phase_y)

    # Thêm nhiễu ngẫu nhiên
    terrain += 0.3 * rng.randn(nrow, ncol)

    # Chuẩn hóa về phạm vi [-max_height, max_height]
    terrain = terrain / (np.abs(terrain).max() + 1e-8) * max_height

    # Làm mượt biên để robot bắt đầu trên mặt phẳng
    fade_cols = min(20, ncol // 5)
    fade = np.linspace(0, 1, fade_cols)
    terrain[:, :fade_cols] *= fade[np.newaxis, :]
    terrain[:, -fade_cols:] *= fade[::-1][np.newaxis, :]

    return terrain.astype(np.float32)


def create_heightfield_xml(
    hfield_data: np.ndarray,
    size_x: float = 10.0,
    size_y: float = 4.0,
    max_height: float = 0.1,
) -> str:
    """Tạo XML snippet cho heightfield terrain.

    MuJoCo heightfield cần khai báo trong <asset> và <worldbody>.

    Args:
        hfield_data: ma trận heightfield.
        size_x: kích thước x (m).
        size_y: kích thước y (m).
        max_height: chiều cao cực đại (m).

    Returns:
        Tuple (asset_xml, body_xml).
    """
    nrow, ncol = hfield_data.shape

    # Chuyển sang format MuJoCo (uint16, 0-65535)
    normalized = (hfield_data - hfield_data.min()) / (
        hfield_data.max() - hfield_data.min() + 1e-8
    )
    data_uint16 = (normalized * 65535).astype(np.uint16)

    # Chiều cao tối đa để truyền cho MuJoCo
    elevation = max_height

    asset_xml = (
        f'<hfield name="terrain" nrow="{nrow}" ncol="{ncol}" '
        f'size="{size_x / 2} {size_y / 2} {elevation} 0.01"/>'
    )

    body_xml = (
        f'<geom name="terrain_geom" type="hfield" hfield="terrain" '
        f'pos="{size_x / 2} 0 0" '
        f'condim="3" contype="1" conaffinity="1" '
        f'rgba="0.4 0.35 0.3 1"/>'
    )

    return asset_xml, body_xml, data_uint16


def create_terrain_model_xml(
    terrain_type: str = "flat",
    terrain_config: dict[str, Any] | None = None,
) -> str:
    """Tạo file XML hoàn chỉnh cho terrain (dùng include vào model chính).

    Args:
        terrain_type: "flat" | "stairs" | "rough"
        terrain_config: thông số terrain.

    Returns:
        Chuỗi XML.
    """
    cfg = terrain_config or {}

    if terrain_type == "flat":
        return (
            "<mujoco>\n"
            "  <worldbody>\n"
            '    <geom name="floor" type="plane" size="50 50 0.1"\n'
            '          rgba="0.3 0.3 0.3 1" condim="3" contype="1" conaffinity="1"/>\n'
            "  </worldbody>\n"
            "</mujoco>"
        )

    elif terrain_type == "stairs":
        stair_geoms = generate_stair_terrain(
            num_steps=cfg.get("num_steps", 8),
            step_height=cfg.get("step_height", 0.15),
            step_depth=cfg.get("step_depth", 0.30),
            step_width=cfg.get("step_width", 1.0),
        )
        return (
            "<mujoco>\n"
            "  <worldbody>\n"
            f"    {stair_geoms}\n"
            "  </worldbody>\n"
            "</mujoco>"
        )

    elif terrain_type == "rough":
        hfield_data = generate_heightfield_data(
            size_x=cfg.get("size_x", 10.0),
            size_y=cfg.get("size_y", 4.0),
            resolution=cfg.get("resolution", 0.05),
            max_height=cfg.get("max_height", 0.05),
            frequency=cfg.get("frequency", 1.0),
        )
        asset_xml, body_xml, _ = create_heightfield_xml(
            hfield_data,
            size_x=cfg.get("size_x", 10.0),
            size_y=cfg.get("size_y", 4.0),
            max_height=cfg.get("max_height", 0.05),
        )
        return (
            "<mujoco>\n"
            f"  <asset>\n    {asset_xml}\n  </asset>\n"
            f"  <worldbody>\n    {body_xml}\n  </worldbody>\n"
            "</mujoco>"
        )

    else:
        raise ValueError(f"Terrain type không hợp lệ: {terrain_type}")
