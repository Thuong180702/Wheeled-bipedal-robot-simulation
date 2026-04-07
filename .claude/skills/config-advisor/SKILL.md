---
name: config-advisor
description: >
  LUÔN dùng skill này khi user hỏi "config nào cần thay đổi?", "tại sao wheel spin?",
  "curriculum stuck phải làm gì?", hay có bất kỳ WARN/exploit flag nào từ eval-analyzer
  trong wheeled biped project. Biết cấu trúc balance.yaml, balance_robust.yaml,
  curriculum.yaml. Nhận symptom từ eval/training → đề xuất YAML diff cụ thể với lý do
  và dự đoán tác động. Luôn output minimal diff (1-2 params), không overhaul toàn bộ.
  Trigger ngay khi thấy: wheel_spin, xy_drift, ctrl_jitter, curriculum stuck,
  plateau, poor_friction, poor_push_recovery, high fall rate.
license: Project-internal skill
---

# Config Advisor Skill

## Tổng quan

Skill này giải quyết câu hỏi:

- "Tôi nên thay đổi config gì để fix symptom X?"
- "Tăng/giảm tham số nào khi curriculum stuck?"
- "Config nào khác nhau giữa balance và balance_robust, và tại sao?"
- "Thay đổi này sẽ tác động thế nào đến training?"

---

## Bước 1 — File map

```
configs/
├── curriculum.yaml          # Multi-stage pipeline
└── training/
    ├── balance.yaml          # Stage 1: pure balance, no push
    ├── balance_robust.yaml   # Stage 3: push recovery, warm-start từ stand_up
    ├── stand_up.yaml         # Stage 2: height transition + recovery from fall
    └── ...
```

---

## Bước 2 — Symptom → Config recipe

Catalog các triệu chứng phổ biến và config fix tương ứng:

### EXPLOIT PATTERNS

**wheel_spin_exploit** — wheel spin liên tục thay vì dùng posture

```yaml
# configs/training/balance.yaml
rewards:
  wheel_velocity: -0.012 # was: -0.006 | tăng penalty x2
  no_motion: 0.7 # was: 0.5   | thưởng đứng yên mạnh hơn
# ⚠️ KHÔNG áp dụng cho balance_robust (wheel cần spin để recover)
```

Tác động: giảm wheel_spin sau 2–5M steps. Có thể tạm giảm reward khi policy relearn.

---

**xy_drift_exploit** — robot từ từ drift ra khỏi vị trí ban đầu

```yaml
# configs/training/balance.yaml
rewards:
  position_drift: 2.5 # was: 1.5 | tăng 67%
  wheel_velocity: -0.010 # was: -0.006 | bổ sung penalty
```

Tác động: giảm drift sau 3–7M steps.
Cẩn thận: tăng quá mạnh → robot rigid, không recover được từ push.

---

**ctrl_jitter** — action thay đổi đột ngột giữa các steps

```yaml
# configs/training/balance.yaml
rewards:
  action_rate: -0.10 # was: -0.06 | tăng penalty 67%
```

Tác động: smooth action sau 2–4M steps. Limit: không vượt -0.15.

---

**leg_asymmetry** — một chân cao hơn chân kia

```yaml
# configs/training/balance.yaml
rewards:
  symmetry: 1.5 # was: 1.0 | tăng 50%
```

---

**height_oscillation** — robot bouncing lên xuống

```yaml
# configs/training/balance.yaml
rewards:
  height: 3.0 # was: 2.5
ppo:
  entropy_coeff: 0.002 # was: 0.004 | giảm exploration
```

---

### CURRICULUM ISSUES

**curriculum_stuck_level** — within-stage curriculum không advance (min_height không giảm)

```yaml
# configs/training/balance.yaml
curriculum:
  reward_threshold: 0.68 # was: 0.75 | giảm ngưỡng advance
  eval_interval: 20 # was: 50   | eval thường xuyên hơn
```

Tác động: advance nhanh hơn. Không giảm dưới 0.55 (advance sớm quá).

---

**curriculum_stuck_stage** — multi-stage max_retries reached, không promote

```yaml
# configs/curriculum.yaml
curriculum:
  stages:
    - name: balance
      success_value: 6.5 # was: 7.0 | giảm 0.5 reward/step
  max_retries_per_stage: 5 # was: 3   | thêm 2 attempts
```

Cẩn thận: không giảm success_value xuống < 6.0 cho balance.

---

**curriculum_demotion_loop** — oscillating promote/demote liên tục

```yaml
# configs/curriculum.yaml
curriculum:
  demotion_threshold: 0.2 # was: 0.3  | chỉ demote khi thực sự rất tệ
  promotion_window: 150 # was: 100  | window dài hơn → stable signal
```

---

### ROBUSTNESS ISSUES

**poor_friction_generalization** — performance giảm mạnh trên friction_low/high

```yaml
# configs/training/balance.yaml
domain_randomization:
  friction_range: [0.5, 1.5] # was: [0.7, 1.3] | mở rộng từ ±30% → ±50%
```

Nominal performance có thể giảm nhẹ. Không giảm dưới 0.4 (không còn realistic).

---

**poor_push_recovery** — max_recoverable_push_n thấp (< 40N)

```yaml
# configs/training/balance.yaml (thêm push nhỏ vào balance stage)
domain_randomization:
  push_magnitude: 20 # was: 0   | push nhỏ để quen với disturbance
  push_interval: 300 # was: 500 | push thường xuyên hơn
```

Chỉ thêm sau khi balance đã converge tốt. Cho push recovery mạnh → train balance_robust.

---

**high_fall_rate_nominal** — fall rate > 20% trên nominal

```yaml
# configs/training/balance.yaml
rewards:
  body_level: 2.0 # was: 1.5
  alive: 0.5 # was: 0.3
ppo:
  entropy_coeff: 0.002 # was: 0.004
```

---

### CONVERGENCE ISSUES

**reward_plateau_early** — plateau sớm (< 10M steps)

```yaml
# configs/training/balance.yaml
ppo:
  learning_rate: 3.0e-4 # was: 2.0e-4 | tăng LR nhẹ
  entropy_coeff: 0.003 # was: 0.004  | giảm nhẹ entropy
```

**reward_plateau_late** — plateau muộn (> 30M steps)

```yaml
# configs/training/balance.yaml
ppo:
  entropy_coeff: 0.006 # was: 0.004 | tăng exploration để thoát local min
curriculum:
  reward_threshold: 0.65 # was: 0.75  | mở rộng training distribution
```

---

### BALANCE ROBUST SPECIFIC

**balance_robust_wont_recover** — không recover sau push mặc dù warm-start tốt

```yaml
# configs/training/balance_robust.yaml
rewards:
  natural_pose: 2.0 # was: 1.5 | tăng return-to-stance
domain_randomization:
  push_magnitude: 30 # was: 40  | giảm push để học dần
```

Verify: no_motion=0.0 và wheel_velocity=0.0 (phải đúng rồi, không thay đổi).

---

## Bước 3 — Phân tích config hiện tại

```python
import yaml

def analyze_config(path: str, stage: str) -> list[str]:
    """Đọc config và highlight điểm cần chú ý."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    flags = []
    rewards = cfg.get("rewards", {})
    ppo     = cfg.get("ppo", {})
    dr      = cfg.get("domain_randomization", {})
    cur     = cfg.get("curriculum", {})

    # Wheel spin risk
    if rewards.get("wheel_velocity", 0) > -0.005 and stage == "balance":
        flags.append(f"WEAK wheel_velocity={rewards['wheel_velocity']} → risk exploit")

    # Entropy
    if ppo.get("entropy_coeff", 0.004) > 0.01:
        flags.append(f"HIGH entropy_coeff={ppo['entropy_coeff']} → noisy gradient")
    if ppo.get("entropy_coeff", 0.004) < 0.001:
        flags.append(f"CRITICAL entropy_coeff={ppo['entropy_coeff']} < 0.001 → risk collapse")

    # Narrow DR
    friction = dr.get("friction_range", [1.0, 1.0])
    if friction[1] - friction[0] < 0.4 and dr.get("enabled"):
        flags.append(f"NARROW friction_range={friction} → poor generalization")

    # Curriculum eval sparsity
    if cur.get("enabled") and cur.get("eval_interval", 50) > 30:
        flags.append(f"SPARSE eval_interval={cur['eval_interval']} → slow curriculum gate")

    return flags
```

---

## Bước 4 — Quick reference: parameter ranges

```
REWARDS (balance.yaml):
  body_level        : 1.0 – 2.5    default 1.5
  height            : 1.5 – 4.0    default 2.5
  natural_pose      : 0.2 – 1.0    default 0.4
  no_motion         : 0.2 – 1.0    default 0.5  → 0.0 trong balance_robust
  wheel_velocity    : -0.02 – -0.003 default -0.006 → 0.0 trong balance_robust
  action_rate       : -0.15 – -0.02 default -0.06
  position_drift    : 0.5 – 3.0    default 1.5

PPO:
  learning_rate     : 1e-4 – 5e-4  default 2e-4
  entropy_coeff     : 0.001 – 0.01 default 0.004  KHÔNG < 0.001
  num_minibatches   : 8 – 64       default 32

DOMAIN_RANDOMIZATION:
  friction_range    : [0.4,1.6] max | [0.7,1.3] conservative
  push_magnitude    : 0 (balance) | 20–50 (robust)

CURRICULUM — balance.yaml (within-stage):
  reward_threshold  : 0.55 – 0.80  default 0.75
  eval_interval     : 2 – 50       default 50

CURRICULUM — curriculum.yaml (multi-stage):
  success_value (balance)      : 6.0 – 7.5   default 7.0
  success_value (balance_robust): 5.0 – 6.5  default 6.0
  success_value (stand_up)     : 4.0 – 5.5   default 5.0
  promotion_threshold          : 0.7 – 0.9   default 0.8
  demotion_threshold           : 0.15 – 0.4  default 0.3
  max_retries_per_stage        : 3 – 8       default 3
```

---

## Bước 5 — Compatibility rules (bắt buộc kiểm tra)

```
1. balance_robust.yaml: no_motion=0.0 và wheel_velocity=0.0 — KHÔNG thay đổi
   (wheel cần spin để recover từ push)

2. Warm-start compatibility: low_level_pid.enabled và gains phải khớp
   giữa balance.yaml và balance_robust.yaml

3. success_value đơn vị: reward/STEP (không phải episode sum)
   eval_per_step = eval_reward_mean / episode_length
   Max ~10.5/step cho balance. Không đặt success_value > 9.0

4. entropy_coeff: KHÔNG < 0.001 — risk mode collapse

5. action_rate: KHÔNG < -0.20 — quá restrictive, policy không reactive
```

---

## Quy tắc khi dùng skill này

1. **Đọc config hiện tại trước** — `analyze_config()` để có baseline, không đoán.
2. **Chỉ thay 1–2 params mỗi lần** — thay nhiều cùng lúc khó diagnose kết quả.
3. **Không đụng balance_robust rewards** (no_motion, wheel_velocity) — đây là intentional design.
4. **success_value đơn vị là reward/step** — không nhầm với episode sum.
5. **Resume thay vì retrain** sau config change — trừ khi thay đổi architecture/action semantics.
6. **Compatibility check** trước mọi thay đổi liên quan đến warm-start hoặc multi-stage.
7. **Sau config change**, đề xuất verify command cụ thể để confirm hiệu quả.
