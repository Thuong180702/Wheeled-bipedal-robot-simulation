---
name: training-decision
description: >
  Dùng skill này SAU KHI đã có output từ eval-analyzer skill.
  Nhận metrics + flags → đưa ra quyết định train cụ thể: tiếp tục, resume từ checkpoint,
  train lại từ đầu, thay đổi config, hoặc advance stage.
  Output: quyết định có lý do + command sẵn sàng chạy + config diff nếu cần.
  Luôn dùng eval-analyzer trước skill này.
license: Project-internal skill
---

# Training Decision Skill

## Tổng quan

Skill này nhận kết quả từ `eval-analyzer` và đưa ra **một trong 6 quyết định**:

| Quyết định | Ký hiệu | Mô tả |
|---|---|---|
| Continue | `CONTINUE` | Train tiếp từ checkpoint cuối, không đổi gì |
| Resume-tweak | `RESUME_TWEAK` | Resume + thay đổi config nhỏ |
| Resume-checkpoint | `RESUME_CKPT` | Resume từ một checkpoint cũ hơn |
| Retrain-stage | `RETRAIN_STAGE` | Train lại stage này từ đầu (giữ warm-start) |
| Retrain-scratch | `RETRAIN_SCRATCH` | Train lại từ đầu hoàn toàn |
| Advance-stage | `ADVANCE_STAGE` | Chuyển sang stage tiếp theo của curriculum |

---

## Bước 1 — Đọc input từ eval-analyzer

Input cần có:

```python
eval_report = {
    # Từ eval_balance.json
    "scenarios": [
        {"scenario": "nominal", "fall_rate": 0.05, "overall_status": "OK", ...},
        {"scenario": "friction_low", "fall_rate": 0.28, "overall_status": "WARN", ...},
    ],
    "warns": ["friction_low.fall_rate=0.28 [WARN]"],

    # Từ evaluate.json (nếu có)
    "benchmark": {
        "eval_per_step": 7.3,
        "fall_rate": 0.08,
        "success_rate": 0.92,
    },

    # Từ metrics.jsonl (nếu có)
    "training_trend": {
        "reward_trend": "plateau",   # improving | plateau | declining
        "latest_eval_reward": 7250,
        "latest_fall_rate": 0.08,
        "total_steps": 12_000_000,
    },

    # Metadata
    "checkpoint_dir": "outputs/balance/rl/seed42/checkpoints/final",
    "current_stage": "balance",   # balance | balance_robust | stand_up | ...
}
```

---

## Bước 2 — Decision tree

### Tree chính

```python
def make_decision(report: dict) -> dict:
    benchmark = report.get("benchmark", {})
    trend = report.get("training_trend", {})
    scenarios = report.get("scenarios", [])
    warns = report.get("warns", [])
    current_stage = report.get("current_stage", "balance")

    eval_per_step = benchmark.get("eval_per_step", 0.0)
    fall_rate = benchmark.get("fall_rate", 1.0)
    success_rate = benchmark.get("success_rate", 0.0)
    reward_trend = trend.get("reward_trend", "unknown")
    total_steps = trend.get("total_steps", 0)

    # ── CRITICAL: không thể train tiếp ──────────────────────────────────────
    nominal = next((s for s in scenarios if s["scenario"] == "nominal"), None)
    if nominal and nominal["fall_rate"] > 0.5:
        return _decide_retrain_scratch(report, reason="fall_rate nominal > 50% — policy collapse")

    if reward_trend == "declining" and total_steps > 5_000_000:
        return _decide_retrain_scratch(report, reason="reward declining sau 5M steps — có thể diverge")

    # ── ADVANCE: đủ điều kiện lên stage tiếp ────────────────────────────────
    SUCCESS_VALUES = {"balance": 7.0, "balance_robust": 6.0, "stand_up": 5.0}
    success_threshold = SUCCESS_VALUES.get(current_stage, 7.0)

    if (eval_per_step >= success_threshold
            and success_rate >= 0.80
            and fall_rate <= 0.15
            and len([w for w in warns if "CRITICAL" in w]) == 0):
        return _decide_advance_stage(report, eval_per_step, success_threshold)

    # ── CONTINUE: đang tốt, chưa đủ steps ───────────────────────────────────
    if (reward_trend == "improving"
            and fall_rate <= 0.20
            and len(warns) <= 2):
        return _decide_continue(report)

    # ── PLATEAU: phân tích nguyên nhân ──────────────────────────────────────
    if reward_trend == "plateau":
        return _analyze_plateau(report, eval_per_step, success_threshold, warns)

    # ── WARN scenarios: config tweak ────────────────────────────────────────
    if warns:
        return _analyze_warns(report, warns, scenarios)

    # default
    return _decide_continue(report, reason="không đủ thông tin để quyết định khác")
```

---

### Sub-decisions: logic chi tiết

#### Phân tích plateau

```python
def _analyze_plateau(report, eval_per_step, success_threshold, warns):
    trend = report.get("training_trend", {})
    total_steps = trend.get("total_steps", 0)
    scenarios = report.get("scenarios", [])
    validation = report.get("validation", {})

    # Kiểm tra exploit patterns từ validation_report
    exploit_signals = []
    if validation:
        val_warns = validation.get("warns", {})
        if "wheel_spin_mean_rads" in val_warns:
            exploit_signals.append("wheel_spin")
        if "xy_drift_max_m" in val_warns:
            exploit_signals.append("xy_drift")
        if "ctrl_jitter_mean_nm" in val_warns:
            exploit_signals.append("ctrl_jitter")

    # Plateau vì exploit → cần reward tweak
    if exploit_signals:
        return _decide_resume_tweak(
            report,
            reason=f"Plateau do exploit: {exploit_signals}",
            config_changes=_exploit_config_fixes(exploit_signals)
        )

    # Plateau dù eval_per_step gần threshold → gần đạt, train thêm
    gap = success_threshold - eval_per_step
    if gap < 0.5 and total_steps < 30_000_000:
        return _decide_continue(
            report,
            reason=f"Plateau nhưng eval_per_step={eval_per_step:.2f} gần threshold={success_threshold}. "
                   f"Thiếu {gap:.2f} reward/step. Train thêm steps."
        )

    # Plateau và eval_per_step xa threshold → cần config change
    if gap >= 0.5:
        return _decide_resume_tweak(
            report,
            reason=f"Plateau, gap={gap:.2f} reward/step so với threshold. Cần điều chỉnh reward weights.",
            config_changes=_reward_boost_config(report.get("current_stage", "balance"))
        )

    # Plateau sau nhiều steps → retrain stage
    if total_steps > 40_000_000:
        return _decide_retrain_stage(
            report,
            reason=f"Plateau sau {total_steps/1e6:.0f}M steps, gần max budget. Retrain stage."
        )

    return _decide_continue(report, reason="Plateau nhưng chưa đủ căn cứ thay đổi")
```

#### Phân tích warns từ scenarios

```python
def _analyze_warns(report, warns, scenarios):
    # Phân loại warns theo scenario
    friction_warns = [w for w in warns if "friction_low" in w or "friction_high" in w]
    push_warns = [w for w in warns if "push_recovery" in w or "max_recoverable_push" in w]
    nominal_warns = [w for w in warns if "nominal" in w]
    full_range_warns = [w for w in warns if "full_range" in w]

    config_changes = {}

    if nominal_warns:
        # Vấn đề trên nominal là vấn đề cốt lõi
        return _decide_resume_tweak(
            report,
            reason="Warn trên nominal scenario — vấn đề cơ bản của policy",
            config_changes={"domain_randomization.push_magnitude": 0, "ppo.entropy_coeff": 0.01}
        )

    if friction_warns:
        config_changes["domain_randomization.friction_multiplier_range"] = "[0.5, 1.5]"
        config_changes["comment"] = "Tăng friction randomization để generalize"

    if push_warns:
        push_val = _get_max_push(scenarios)
        if push_val < 40:
            config_changes["domain_randomization.push_magnitude"] = 30
            config_changes["comment"] = "Policy yếu với push — bật push disturbance nhẹ"

    if full_range_warns and not nominal_warns:
        config_changes["curriculum.reward_threshold"] = 0.70
        config_changes["comment"] = "Giảm curriculum threshold để có thêm time ở full_range"

    if config_changes:
        return _decide_resume_tweak(report, reason=f"Warns: {warns[:3]}", config_changes=config_changes)

    return _decide_continue(report, reason=f"Warns nhỏ ({len(warns)}), tiếp tục train")
```

---

## Bước 3 — Config change recipes

### Exploit fixes

```python
def _exploit_config_fixes(exploit_signals: list) -> dict:
    fixes = {}
    if "wheel_spin" in exploit_signals:
        fixes["rewards.wheel_velocity"] = -0.01   # tăng penalty (default -0.005)
        fixes["rewards.no_motion"] = 0.6           # tăng thưởng đứng yên (default 0.5)
    if "xy_drift" in exploit_signals:
        fixes["rewards.wheel_velocity"] = -0.015
    if "ctrl_jitter" in exploit_signals:
        fixes["rewards.action_rate"] = -0.08       # tăng penalty jitter (default -0.05)
    return fixes

def _reward_boost_config(stage: str) -> dict:
    if stage == "balance":
        return {
            "rewards.height": 3.0,        # tăng từ 2.5 → 3.0
            "rewards.body_level": 1.8,    # tăng từ 1.5 → 1.8
            "ppo.entropy_coeff": 0.003,   # giảm entropy để exploit ít hơn
            "comment": "Boost height + body_level rewards để thoát plateau"
        }
    elif stage == "balance_robust":
        return {
            "rewards.natural_pose": 2.0,  # tăng return-to-stance
            "domain_randomization.push_magnitude": 50,
            "comment": "Tăng push magnitude và natural_pose để push recovery tốt hơn"
        }
    return {}
```

---

## Bước 4 — Format output

### Decision output format

```python
def format_decision(decision: dict) -> str:
    """Format decision thành báo cáo rõ ràng."""
    lines = [
        "=" * 60,
        f"🎯 TRAINING DECISION: {decision['action']}",
        "=" * 60,
        f"Lý do: {decision['reason']}",
        "",
    ]

    if decision.get("config_changes"):
        lines.append("📝 Config thay đổi đề xuất:")
        for k, v in decision["config_changes"].items():
            if k != "comment":
                lines.append(f"   {k}: {v}")
        if "comment" in decision["config_changes"]:
            lines.append(f"   # {decision['config_changes']['comment']}")
        lines.append("")

    lines.append("⚡ Command để thực thi:")
    for cmd in decision["commands"]:
        lines.append(f"   {cmd}")

    if decision.get("next_eval"):
        lines.append("")
        lines.append(f"📊 Sau khi chạy, eval lại bằng:")
        lines.append(f"   {decision['next_eval']}")

    lines.append("=" * 60)
    return "\n".join(lines)
```

### Ví dụ output thực tế

```
============================================================
🎯 TRAINING DECISION: RESUME_TWEAK
============================================================
Lý do: Plateau do exploit: wheel_spin — wheel đang spin nhiều để balance thay vì dùng posture

📝 Config thay đổi đề xuất:
   rewards.wheel_velocity: -0.01
   rewards.no_motion: 0.6
   # Tăng wheel penalty và no_motion reward để ngăn wheel-momentum exploit

⚡ Command để thực thi:
   # 1. Sửa configs/training/balance.yaml:
   #    rewards.wheel_velocity: -0.01
   #    rewards.no_motion: 0.6

   python scripts/train.py single --stage balance --seed 42 \
       --resume outputs/balance/rl/seed42/checkpoints/step_10000000

📊 Sau khi chạy, eval lại bằng:
   python scripts/eval_balance.py \
       --checkpoint outputs/balance/rl/seed42/checkpoints/final \
       --scenarios nominal push_recovery
   python scripts/validate_checkpoint.py \
       --checkpoint outputs/balance/rl/seed42/checkpoints/final
============================================================
```

---

## Bước 5 — Command templates

### CONTINUE
```bash
python scripts/train.py single --stage {stage} --seed {seed} \
    --resume {checkpoint_dir}
```

### RESUME_TWEAK (sau khi sửa config)
```bash
# Kiểm tra config diff trước
git diff configs/training/{stage}.yaml

python scripts/train.py single --stage {stage} --seed {seed} \
    --resume {last_checkpoint}
```

### RESUME_CKPT (quay về checkpoint cũ hơn)
```bash
# Liệt kê checkpoints có sẵn
ls outputs/balance/rl/seed{seed}/checkpoints/

python scripts/train.py single --stage {stage} --seed {seed} \
    --resume outputs/balance/rl/seed{seed}/checkpoints/step_{N}
```

### RETRAIN_STAGE (train lại stage, giữ warm-start nếu có)
```bash
python scripts/train.py single --stage {stage} --seed {new_seed} \
    --steps {steps}
# Dùng seed mới để tránh lặp lại path cũ
```

### RETRAIN_SCRATCH
```bash
python scripts/train.py single --stage balance --seed {new_seed} \
    --steps 50000000
# Không dùng --resume
```

### ADVANCE_STAGE
```bash
# Chạy curriculum manager từ stage tiếp theo
python scripts/train.py curriculum \
    --steps-per-stage 10000000

# Hoặc chạy stage cụ thể với warm-start
python scripts/train.py single --stage balance_robust --seed 42 \
    --steps 5000000 \
    --resume outputs/balance/rl/seed42/checkpoints/final
```

---

## Bước 6 — Special cases cần xử lý thêm

### Curriculum stuck (max_retries reached)

Nếu log có `⚠️ max_retries_per_stage reached`:
```python
config_changes = {
    "curriculum.success_value": current_success_value - 0.5,  # giảm threshold
    "curriculum.max_retries_per_stage": current_max + 2,
    "comment": "Curriculum stuck — giảm success_value hoặc tăng max_retries"
}
```

### 3-seed divergence

Nếu so sánh 3 seeds và 1 seed outlier:
- Bỏ seed đó, train lại seed mới
- Không average kết quả của seed outlier vào paper metrics

### Eval_per_step đang tính sai

Luôn kiểm tra: `eval_per_step = eval_reward_mean / episode_length`
- `episode_length` default = 1000 nếu không có trong result dict
- Nếu eval_per_step trông quá thấp (<1.0), kiểm tra lại episode_length

---

## Quy tắc khi dùng skill này

1. **Không dùng skill này mà không có output từ eval-analyzer** — thiếu thông tin sẽ ra quyết định sai.
2. **Nominal scenario là ưu tiên số 1** — nếu nominal fail thì không cần xem scenario khác.
3. **Luôn kèm theo lý do** — không chỉ nói "retrain" mà giải thích metric nào trigger quyết định đó.
4. **Config change phải minimal** — chỉ thay đổi 1-2 hyperparameter, không overhaul toàn bộ config.
5. **Sau mỗi quyết định, nêu rõ eval command** để verify sau khi thực hiện.
6. **Phân biệt 2 curriculum systems**:
   - Within-stage height curriculum (tự động trong PPOTrainer) → theo dõi qua `curriculum/eval_per_step` trong JSONL
   - Multi-stage curriculum (CurriculumManager) → theo dõi qua `eval_per_step` vs `success_value` trong curriculum.yaml