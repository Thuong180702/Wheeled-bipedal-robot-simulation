# AGENT GUIDE - Wheeled Bipedal Robot Simulation

Tai lieu nay danh cho AI coding agents lam viec trong repo nay. Muc tieu: hieu nhanh kien truc, thao tac an toan, va chay dung build/test command ngay lap tuc.

## 1) Muc tieu du an

- Mo phong va huan luyen robot hai chan co banh xe bang MuJoCo MJX + JAX.
- Thuat toan huan luyen chinh: PPO (Flax/Optax), ho tro curriculum multi-stage.
- Doi tuong code chinh: `wheeled_biped/`, cac script CLI trong `scripts/`, config YAML trong `configs/`.

## 2) Kien truc project (thuc te theo code)

### 2.1 Module chinh

- `wheeled_biped/envs/`
  - `base_env.py`: base class `WheeledBipedEnv`, `EnvState`, reset/step/vmap helper.
  - Cac env task:
    - `balance_env.py` (obs=40)
    - `standup_env.py` (obs=40)
    - `locomotion_env.py` (obs=41 = base39 + command2)
    - `walking_env.py` (obs=43 = base39 + command2 + gait_phase2)
    - `stair_env.py` (obs=42 = base39 + goal2 + progress1)
    - `terrain_env.py` (obs=41 = base39 + command2)
  - `__init__.py`: `ENV_REGISTRY` + `make_env(env_name, **kwargs)`.

- `wheeled_biped/training/`
  - `ppo.py`: rollout, GAE, PPO update, obs normalization, checkpoint save/load.
  - `networks.py`: `ActorCritic` (Flax), Gaussian policy + value head.
  - `curriculum.py`: `CurriculumManager` quan ly stage progression.
  - `live_viewer.py`: train + viewer.

- `wheeled_biped/rewards/`
  - `reward_functions.py`: reward/penalty JAX-first, tong hop qua `compute_total_reward`.

- `wheeled_biped/utils/`
  - `config.py`: load YAML, deep merge `robot.yaml <- task.yaml`, model path.
  - `math_utils.py`: quat/rotation helper.
  - `logger.py`: TensorBoard/WandB/JSONL logging.

- `scripts/`
  - `train.py`: CLI train (`curriculum`, `single`).
  - `evaluate.py`: danh gia checkpoint.
  - `visualize.py`: mode `model`, `policy`, `render`.

- `tests/`
  - `test_env.py`, `test_model.py`, `test_rewards.py`.

### 2.2 Luong chay huan luyen

1. `scripts/train.py` load config YAML.
2. Tao env qua `make_env(...)`.
3. Tao `PPOTrainer` (network + optimizer + obs_rms).
4. Rollout vectorized (`v_step`) -> tinh GAE -> PPO minibatch updates.
5. Log metrics/checkpoint vao `outputs/`.

### 2.3 Luong eval/visualize

- `scripts/evaluate.py`: load `checkpoint.pkl` -> build network/env theo config trong checkpoint -> chay episodes -> tong hop metrics.
- `scripts/visualize.py`:
  - `model`: xem model MJCF.
  - `policy`: inference trong viewer, co telemetry option.
  - `render`: render video MP4.

## 3) Quy tac van hanh bat buoc cho AI

### 3.1 Quy tac tong quat

- Khong sua file khi user chi yeu cau doc/giai thich/review.
- Neu can sua code: luon chay test truoc khi edit (pre-change baseline), sau do chay lai test sau khi edit.
- Khong doi cau truc thu muc chinh, build pipeline, hoac phu thuoc lon neu chua duoc user cho phep ro rang.
- Khong cham vao file third-party, file generated, hoac artifacts trong `outputs/`, `logs/`, `build/` (tru khi user yeu cau).

### 3.2 Quy tac khi sua code

1. Xac dinh pham vi anh huong (env/task/config/test nao lien quan).
2. Chay smoke test truoc khi sua.
3. Sua toi thieu, khong refactor rong neu user khong yeu cau.
4. Them/doi test tuong ung neu thay doi hanh vi.
5. Chay lai test + lint/toi thieu smoke test sau khi sua.
6. Bao cao ro: file thay doi, ly do, rui ro con lai.

## 4) Build, test, lint, type-check commands

Du an Python, khong co buoc "build" bat buoc nhu C/C++ de chay train/eval. "Build" o day duoc hieu la setup moi truong + verify code quality.

### 4.1 Setup moi truong

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Neu can bo dev tools:

```powershell
pip install -e ".[dev]"
```

### 4.2 Test (uu tien chay truoc moi thay doi)

Smoke test nhanh:

```powershell
python -m pytest tests/test_env.py -q -x
```

Chay full test suite:

```powershell
python -m pytest tests -v
```

### 4.3 Lint va format

```powershell
ruff check wheeled_biped scripts tests
ruff format wheeled_biped scripts tests
```

### 4.4 Type-check

```powershell
mypy wheeled_biped scripts --ignore-missing-imports
```

### 4.5 Chay train/eval/viz

Train 1 stage:

```powershell
python scripts/train.py single --stage balance --steps 5000000
```

Train curriculum:

```powershell
python scripts/train.py curriculum --config configs/curriculum.yaml
```

Evaluate:

```powershell
python scripts/evaluate.py --checkpoint outputs/checkpoints/balance/final --stage balance
```

Visualize model/policy:

```powershell
python scripts/visualize.py model
python scripts/visualize.py policy --checkpoint outputs/checkpoints/balance/final
```

## 5) Code convention (naming, format, style guide)

### 5.1 Naming

- Python naming: `snake_case` cho function/variable, `PascalCase` cho class, `UPPER_SNAKE_CASE` cho constants.
- Env class phai ket thuc bang `Env` va dang ky trong `ENV_REGISTRY`.
- Reward function naming:
  - `reward_*` cho thanh phan thuong duong.
  - `penalty_*` cho thanh phan phat (ham tra ve do lon duong, weight am trong config).

### 5.2 Typing

- Dung type hints day du (`dict[str, Any]`, `str | Path`, `tuple[...]`).
- Trang thai env dung `NamedTuple` (`EnvState`) de JAX-friendly, immutable semantics.

### 5.3 JAX/MJX style bat buoc

- Uu tien pure function + `@jax.jit` (hoac `@functools.partial(jax.jit, static_argnums=(0,))`).
- Khong mutation truc tiep tren JAX array, dung `.at[...]` hoac `.replace(...)`.
- Loop trong JIT dung `jax.lax.scan`, tranh Python loop trong duong hot path.
- Tach RNG bang `jax.random.split(...)`, khong tai su dung key.

### 5.4 Format va static checks

- Theo `pyproject.toml`:
  - `ruff` line-length: 100.
  - target Python: 3.10.
  - pytest pattern: `tests/test_*.py`.
- Comment ngan gon, dung vao logic kho; khong viet comment thua.

### 5.5 Config pattern

- Training config duoc merge: `configs/robot.yaml` <- `configs/training/*.yaml`.
- Dung `.get(..., default)` khi doc config de tranh KeyError.
- Neu them reward moi:
  - Them ham trong `reward_functions.py`.
  - Them weight trong YAML task lien quan.
  - Cap nhat test neu thay doi hanh vi.

## 6) Invariants quan trong (de tranh pha vo he thong)

- Action dimension la 10 (khong duoc doi neu khong doi model robot).
- Thu tu joint/action mac dinh:
  `[l_hip_roll, l_hip_yaw, l_hip_pitch, l_knee, l_wheel, r_hip_roll, r_hip_yaw, r_hip_pitch, r_knee, r_wheel]`.
- `base_env` observation base = 39 dims.
- Moi env mo rong obs theo task rieng, network input phai khop `env.obs_size`.
- Huong tien cua model real dang dung convention `-Y` la forward (da duoc dung trong locomotion/walking/terrain/stair logic).
- Model path mac dinh trong code la `assets/robot/wheeled_biped_real.xml`.

## 7) Checklist cho AI truoc/sau khi sua

### 7.1 Pre-change

- [ ] Chay test baseline it nhat 1 smoke test.
- [ ] Doc env/config/test lien quan truoc khi sua.
- [ ] Xac nhan change co can cap nhat reward/config/checkpoint compatibility hay khong.

### 7.2 Post-change

- [ ] Chay lai test lien quan + neu co the chay full `pytest tests -v`.
- [ ] Chay lint (`ruff check`) cho vung code da sua.
- [ ] Neu thay doi API/obs/reward logic: cap nhat test va tai lieu.
- [ ] Bao cao ro ket qua test da pass/fail va phan chua verify.

## 8) Huong dan ung xu cho AI sau nay

- Uu tien do chinh xac hanh vi hon la toi uu hoa dep code.
- Neu gap mau thuan giua README va code, tin code source truoc.
- Neu command train/test chay qua lau, bao cao roi de xuat smoke test thay the, khong im lang bo qua.
- Luon tra loi bang tieng Viet trong repo nay.

## 9) Quick reference

- Entry points (`pyproject.toml`):
  - `wb-train = scripts.train:app`
  - `wb-eval = scripts.evaluate:app`
  - `wb-viz = scripts.visualize:app`
- Thu muc output mac dinh:
  - Checkpoint: `outputs/checkpoints/...`
  - Logs: `outputs/logs/...`
  - Telemetry: `outputs/telemetry/...` (hoac `telemetry/` theo script chay).
