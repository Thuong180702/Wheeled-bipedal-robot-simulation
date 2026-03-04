# Wheeled Bipedal Robot Simulation

Mô phỏng và huấn luyện robot hai chân có bánh xe (Wheeled Bipedal Robot) bằng **MuJoCo MJX + JAX** trên GPU, sử dụng Reinforcement Learning (PPO) với chiến lược **Curriculum Learning**.

## Tổng quan

Dự án xây dựng pipeline hoàn chỉnh từ mô hình vật lý đến huấn luyện AI, hướng tới ứng dụng sim-to-real:

| Task                   | Mô tả                                |
| ---------------------- | ------------------------------------ |
| **Balance**            | Đứng vững, chống nhiễu loạn          |
| **Wheeled Locomotion** | Di chuyển bằng bánh xe (tiến/lùi/rẽ) |
| **Walking**            | Bước chân đi bộ                      |
| **Stair Climbing**     | Leo lên/xuống cầu thang              |
| **Rough Terrain**      | Đi trên địa hình gồ ghề              |
| **Stand Up**           | Tự đứng dậy khi bị ngã/nằm xuống    |

## Thông số Robot

```
Đùi:        35 cm          Bánh xe:    Ø12 cm
Ống chân:   35 cm          Tổng KL:    ~7 kg
Động cơ:    10 × 400g      Cảm biến:   IMU + 10 encoder
```

**Cấu hình động cơ (5 loại × 2 bên = 10):**

- Hip Roll (xoay hông ngang)
- Hip Pitch (gập hông trước/sau)
- Knee Pitch (gập gối)
- Ankle Pitch (gập cổ chân)
- Wheel (bánh xe)

## Cấu trúc dự án

```
├── assets/
│   └── robot/
│       └── wheeled_biped.xml          # MuJoCo MJCF model
├── configs/
│   ├── robot.yaml                     # Thông số robot
│   ├── curriculum.yaml                # Cấu hình curriculum pipeline
│   └── training/
│       ├── balance.yaml
│       ├── wheeled_locomotion.yaml
│       ├── walking.yaml
│       ├── stair_climbing.yaml
│       ├── rough_terrain.yaml
│       └── stand_up.yaml
├── wheeled_biped/                     # Package chính
│   ├── envs/                          # MJX environments
│   │   ├── base_env.py                # Base env (MJX + JAX)
│   │   ├── balance_env.py             # Task đứng vững
│   │   ├── locomotion_env.py          # Task di chuyển bánh xe
│   │   ├── walking_env.py             # Task đi bộ
│   │   ├── stair_env.py               # Task leo cầu thang
│   │   ├── terrain_env.py             # Task địa hình gồ ghề
│   │   └── standup_env.py             # Task đứng dậy
│   ├── rewards/
│   │   └── reward_functions.py        # Reward components (JAX)
│   ├── training/
│   │   ├── ppo.py                     # PPO algorithm (JAX)
│   │   ├── networks.py                # Actor-Critic (Flax)
│   │   ├── curriculum.py              # Curriculum manager
│   │   └── live_viewer.py             # Live viewer khi training
│   ├── inference/
│   │   └── unified_controller.py      # Unified multi-skill controller
│   ├── sim/
│   │   ├── domain_randomization.py    # DR cho sim-to-real
│   │   └── terrain_generator.py       # Tạo terrain
│   └── utils/
│       ├── math_utils.py              # Quaternion, rotation (JAX)
│       ├── config.py                  # YAML config loader
│       └── logger.py                  # TensorBoard + WandB
├── scripts/
│   ├── train.py                       # Script training
│   ├── evaluate.py                    # Đánh giá model
│   └── visualize.py                   # Trực quan hóa + điều khiển
├── tests/                             # Unit tests
├── pyproject.toml
└── requirements.txt
```

---

## Cài đặt

### Yêu cầu hệ thống

- **Python 3.10** (bắt buộc — jaxlib chưa hỗ trợ 3.12 trên Windows)
- NVIDIA GPU + CUDA 12 (khuyến nghị, hoạt động được trên CPU)
- RAM ≥ 16 GB

### Cài đặt từ đầu

```bash
# Clone repository
git clone https://github.com/Thuong180702/Wheeled-bipedal-robot-simulation.git
cd Wheeled-bipedal-robot-simulation

# Tạo virtual environment (Python 3.10)
python -m venv .venv

# Kích hoạt venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # Linux / Mac

# Cài đặt dependencies
pip install -r requirements.txt

# Nếu gặp lỗi uvloop trên Windows:
pip install orbax-checkpoint --no-deps
pip install flax --no-deps
pip install msgpack rich PyYAML
```

### Kiểm tra cài đặt

```bash
# Kiểm tra MuJoCo + JAX
python -c "import mujoco; import jax; print(f'MuJoCo {mujoco.__version__}, JAX {jax.__version__}')"

# Xem robot model trong viewer
python scripts/visualize.py model

# Chạy unit tests
pytest tests/ -v
```

---

## Hướng dẫn sử dụng

### 1. Xem robot model

```bash
# Mở MuJoCo viewer để xem robot ở tư thế đứng
python scripts/visualize.py model

# Dùng file MJCF tùy chỉnh
python scripts/visualize.py model --model-path path/to/custom.xml
```

### 2. Training

#### Training một task đơn lẻ

```bash
# Task đứng vững (balance)
python scripts/train.py single --stage balance --steps 5000000

# Task di chuyển bằng bánh xe
python scripts/train.py single --stage wheeled_locomotion --steps 5000000

# Task đi bộ
python scripts/train.py single --stage walking --steps 5000000

# Task leo cầu thang
python scripts/train.py single --stage stair_climbing --steps 5000000

# Task địa hình gồ ghề
python scripts/train.py single --stage rough_terrain --steps 5000000

# Task đứng dậy (từ tư thế ngã/nằm)
python scripts/train.py single --stage stand_up --steps 5000000
```

#### Tùy chỉnh training

```bash
# Tùy chỉnh số environments song song
python scripts/train.py single --stage balance --num-envs 8192

# Thay đổi thư mục output
python scripts/train.py single --stage balance --output-dir my_outputs

# Đổi random seed
python scripts/train.py single --stage balance --seed 123
```

#### Training với live viewer (xem robot khi đang train)

```bash
# Mở cửa sổ MuJoCo viewer real-time trong lúc training
python scripts/train.py single --stage balance --live-view

# Tùy chỉnh tần suất cập nhật viewer (mỗi N updates)
python scripts/train.py single --stage balance --live-view --view-interval 5
```

> **Lưu ý:** Viewer chạy trên main thread, training chạy ở background thread.
> Đóng cửa sổ viewer bất cứ lúc nào — training vẫn tiếp tục.

#### Training đầy đủ Curriculum (tất cả 5 stages)

```bash
python scripts/train.py curriculum --steps-per-stage 5000000
```

Pipeline tự động chạy theo thứ tự:
`Balance → Wheeled Locomotion → Walking → Stair Climbing → Rough Terrain`

Mỗi stage kế thừa (warm-start) từ checkpoint stage trước.

#### Tiếp tục training từ checkpoint

```bash
python scripts/train.py single --stage balance --resume outputs/checkpoints/balance/step_1000000
```

#### Dừng training

Nhấn **Ctrl+C** để dừng bất cứ lúc nào. Checkpoint sẽ được tự động lưu trước khi thoát.

### 3. Đánh giá model

```bash
# Đánh giá policy đã train
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --stage balance
```

### 4. Trực quan hóa policy đã train

```bash
# Xem policy chạy trong MuJoCo viewer
python scripts/visualize.py policy --checkpoint outputs/checkpoints/balance/final

# Tùy chỉnh số bước và tốc độ
python scripts/visualize.py policy --checkpoint outputs/checkpoints/balance/final --num-steps 5000 --slow-factor 2.0
```

### 5. Render video

```bash
# Render video MP4 từ policy
python scripts/visualize.py render --checkpoint outputs/checkpoints/balance/final --output demo.mp4

# Tùy chỉnh resolution và FPS
python scripts/visualize.py render --checkpoint outputs/checkpoints/balance/final --output demo.mp4 --width 1920 --height 1080 --fps 60
```

### 6. Điều khiển robot bằng bàn phím (Interactive)

```bash
# Điều khiển bằng PD controller (không cần checkpoint)
python scripts/visualize.py interactive

# Điều khiển với policy giữ thăng bằng (cần checkpoint)
python scripts/visualize.py interactive --checkpoint outputs/checkpoints/balance/final
```

**Phím điều khiển:**

| Phím    | Chức năng                 |
| ------- | ------------------------- |
| ↑ / ↓   | Tiến / Lùi                |
| ← / →   | Rẽ trái / phải            |
| Q / E   | Nghiêng trái / phải       |
| U / J   | Tăng / Giảm chiều cao     |
| Space   | Dừng lại (phanh)          |
| \[ / \] | Giảm / Tăng tốc độ        |

### 7. Unified Controller (tự động chọn skill)

Sau khi train đủ các task, dùng unified controller để robot tự động chuyển skill:

```bash
# Chạy unified controller với auto-detect
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints

# Tắt auto, chọn thủ công bằng phím 1-5
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints --no-auto-mode
```

**Phím điều khiển:**

| Phím    | Chức năng              |
| ------- | ---------------------- |
| ↑ / ↓   | Tiến / Lùi             |
| ← / →   | Rẽ trái / phải         |
| U / J   | Tăng / Giảm chiều cao  |
| Space   | Dừng lại               |
| \[ / \] | Giảm / Tăng tốc độ     |
| 1       | Ép chọn Balance        |
| 2       | Ép chọn Locomotion     |
| 3       | Ép chọn Walking        |
| 4       | Ép chọn Stair Climbing |
| 5       | Ép chọn Rough Terrain  |
| 6       | Ép chọn Stand Up       |
| 0       | Quay về Auto-detect    |

### 8. Chạy tests

```bash
# Chạy tất cả tests
pytest tests/ -v

# Chạy test cụ thể
pytest tests/test_model.py -v
pytest tests/test_rewards.py -v
pytest tests/test_env.py -v
```

---

## Tham khảo nhanh — Tất cả lệnh

```bash
# ─── Xem model ───
python scripts/visualize.py model

# ─── Training ───
python scripts/train.py single --stage balance --steps 5000000
python scripts/train.py single --stage stand_up --steps 5000000
python scripts/train.py single --stage balance --live-view
python scripts/train.py curriculum --steps-per-stage 5000000

# ─── Đánh giá ───
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --stage balance

# ─── Xem policy ───
python scripts/visualize.py policy --checkpoint outputs/checkpoints/balance/final

# ─── Render video ───
python scripts/visualize.py render --checkpoint outputs/checkpoints/balance/final --output demo.mp4

# ─── Điều khiển tương tác ───
python scripts/visualize.py interactive
python scripts/visualize.py interactive --checkpoint outputs/checkpoints/balance/final

# ─── Unified controller ───
python scripts/visualize.py unified --checkpoint-dir outputs/checkpoints

# ─── Tests ───
pytest tests/ -v
```

---

## Kiến trúc kỹ thuật

### MuJoCo MJX + JAX

```
┌─────────────────────────────────────────────────────┐
│                    GPU (CUDA 12)                     │
│  ┌───────────┐  ┌───────────┐  ┌────────────────┐  │
│  │ MJX Model │  │ MJX Data  │  │ JAX JIT/VMAP   │  │
│  │ (Physics) │→ │ ×4096 env │→ │ (Vectorized)   │  │
│  └───────────┘  └───────────┘  └────────────────┘  │
│                       ↕                              │
│  ┌────────────────────────────────────────────────┐  │
│  │        PPO (Actor-Critic, Flax + Optax)        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │  │
│  │  │  Actor   │  │  Critic  │  │  GAE + Loss  │  │  │
│  │  │ (Policy) │  │ (Value)  │  │  Computation │  │  │
│  │  └──────────┘  └──────────┘  └─────────────┘  │  │
│  └────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

### Observation Space (39 chiều)

| Component    | Kích thước | Mô tả                      |
| ------------ | ---------- | -------------------------- |
| Gravity body | 3          | Trọng lực trong body frame |
| Linear vel   | 3          | Vận tốc tuyến tính thân    |
| Angular vel  | 3          | Vận tốc góc thân           |
| Joint pos    | 10         | Vị trí 10 khớp (encoder)   |
| Joint vel    | 10         | Vận tốc 10 khớp            |
| Prev action  | 10         | Action bước trước          |

### Action Space (10 chiều)

Output policy ∈ [-1, 1] được scale sang torque theo giới hạn từng motor.

### Curriculum Learning

```
Stage 0: Balance ────┐
                     │ success_rate > 80%
Stage 1: Wheeled ←───┘
     Locomotion ─────┐
                     │ success_rate > 80%
Stage 2: Walking ←───┘
                ─────┐
                     │
Stage 3: Stairs ←────┘
                ─────┐
                     │
Stage 4: Rough  ←────┘
        Terrain

Stage 5: Stand Up ←── warm-start từ Balance
         (đứng dậy khi ngã)
```

Mỗi stage kế thừa (warm-start) từ checkpoint stage trước.

### Sim-to-Real Transfer

- **Domain Randomization:** Ngẫu nhiên hóa khối lượng (±10%), ma sát (±30%), damping khớp
- **External Perturbation:** Đẩy ngẫu nhiên lên thân robot mỗi N bước
- **Sensor Noise:** Thêm nhiễu Gaussian vào IMU và encoder
- **Observation Normalization:** Running mean/std chuẩn hóa input

## Cấu hình

Tất cả hyperparameter được quản lý qua file YAML trong `configs/`:

- `configs/robot.yaml` — thông số vật lý robot
- `configs/curriculum.yaml` — pipeline curriculum
- `configs/training/*.yaml` — hyperparameter từng task

Có thể điều chỉnh:

- Số environments song song (`num_envs`)
- Learning rate, epochs, clip_epsilon, ...
- Trọng số reward
- Ngưỡng chuyển stage
- Cấu hình domain randomization

## License

MIT
