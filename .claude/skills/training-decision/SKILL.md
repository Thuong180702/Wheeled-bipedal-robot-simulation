---
name: training-decision
description: >
  LUÔN dùng skill này ngay sau eval-analyzer khi user hỏi "train tiếp hay dừng?",
  "có nên resume không?", "cần thay đổi gì?", hay bất cứ câu hỏi nào liên quan
  đến quyết định training của wheeled biped project. Nhận report từ eval-analyzer
  (validation signals + training trend) → đưa ra 1 trong 6 quyết định cụ thể:
  CONTINUE, RESUME_TWEAK, RESUME_CKPT, RETRAIN_STAGE, RETRAIN_SCRATCH, ADVANCE_STAGE.
  Output: quyết định có lý do + command sẵn sàng chạy + config diff YAML nếu cần.
  Không dùng skill này khi chưa có output từ eval-analyzer.
license: Project-internal skill
---

# Training Decision Skill

## Tổng quan

Skill này nhận kết quả từ `eval-analyzer` và đưa ra **một trong 6 quyết định**:

| Quyết định        | Ký hiệu           | Mô tả                                       |
| ----------------- | ----------------- | ------------------------------------------- |
| Continue          | `CONTINUE`        | Train tiếp từ checkpoint cuối, không đổi gì |
| Resume-tweak      | `RESUME_TWEAK`    | Resume + thay đổi config nhỏ                |
| Resume-checkpoint | `RESUME_CKPT`     | Resume từ một checkpoint cũ hơn             |
| Retrain-stage     | `RETRAIN_STAGE`   | Train lại stage này từ đầu (giữ warm-start) |
| Retrain-scratch   | `RETRAIN_SCRATCH` | Train lại từ đầu hoàn toàn                  |
| Advance-stage     | `ADVANCE_STAGE`   | Chuyển sang stage tiếp theo của curriculum  |

---

## Bước 1 — Đọc input từ eval-analyzer

Input cần có (thứ tự ưu tiên nguồn dữ liệu):

```python
eval_report = {
    # Từ validation_report.json (PRIMARY — validate_checkpoint.py)
    # ⚠️ Đây là nguồn CHÍNH để ra quyết định — có thể quyết định mà không cần scenarios
    "validation": {
        "benchmark": {
            # benchmark nominal từ validate_checkpoint.py
            "reward_mean": 7250.0,
            "fall_rate": 0.08,
            "success_rate": 0.92,
            "episode_length_mean": 980,
        },
        "warn_signals": {
            # chỉ có key nếu signal vượt threshold — dict rỗng = không có exploit
            # "wheel_spin_mean_rads": 4.1,
            # "xy_drift_max_m": 0.45,
            # "ctrl_jitter_mean_nm": 2.1,
            # "leg_asymmetry_mean_rad": 0.18,
            # "ang_vel_rms_rads": 0.7,
        },
        "num_suspicious": 0,
        "flags": [],   # human-readable strings
    },

    # Từ metrics.jsonl (PRIMARY — training log)
    "training_trend": {
        "reward_trend": "plateau",   # improving | plateau | declining
        "latest_eval_reward": 7250,
        "latest_fall_rate": 0.08,
        "total_steps": 12_000_000,
    },

    # Từ evaluate.json (Secondary — nếu có, override validation.benchmark)
    "benchmark": {
        "eval_per_step": 7.3,
        "fall_rate": 0.08,
        "success_rate": 0.92,
        "source": "evaluate.json",   # hoặc "validation_report.benchmark"
    },

    # Từ eval_balance.json (Secondary — paper metrics, multi-scenario)
    # KHÔNG bắt buộc để ra quyết định
    "scenarios": [
        {"scenario": "nominal", "fall_rate": 0.05, "overall_status": "OK"},
        {"scenario": "friction_low", "fall_rate": 0.28, "overall_status": "WARN"},
    ],
    "warns": [
        "exploit.wheel_spin_mean_rads=0.0 [WARN]",   # từ validation
        "friction_low.fall_rate=0.28 [WARN]",        # từ eval_balance
    ],

    # Metadata
    "checkpoint_dir": "outputs/balance/rl/seed42/checkpoints/final",
    "current_stage": "balance",
}
```

---

## Bước 2 — Decision tree

> **Note về helper functions**: Các hàm `_decide_continue()`, `_decide_resume_tweak()`,
> `_decide_retrain_scratch()`, `_decide_retrain_stage()`, `_decide_advance_stage()`
> là **pseudocode** — Claude tự xây dựng response dict phù hợp khi gặp từng case.
> Mỗi hàm trả về `{"action": "<ACTION>", "reason": "...", "commands": [...], "next_eval": "..."}`.

### Tree chính

```python
def make_decision(report: dict) -> dict:
    validation    = report.get("validation", {})
    benchmark     = report.get("benchmark", {})
    trend         = report.get("training_trend", {})
    scenarios     = report.get("scenarios", [])
    warns         = report.get("warns", [])
    current_stage = report.get("current_stage", "balance")

    reward_trend  = trend.get("reward_trend", "unknown")
    total_steps   = trend.get("total_steps", 0)

    # ── Lấy fall_rate và eval_per_step từ nguồn tốt nhất ────────────────────
    # Thứ tự ưu tiên: benchmark (evaluate.py) → validation.benchmark → scenarios nominal
    val_bench    = validation.get("benchmark", {})
    fall_rate    = benchmark.get("fall_rate") or val_bench.get("fall_rate", 1.0)
    success_rate = benchmark.get("success_rate") or val_bench.get("success_rate", 0.0)
    eval_per_step = benchmark.get("eval_per_step", 0.0)

    # ── EARLY EXIT: exploit patterns từ validation — KIỂM TRA ĐẦU TIÊN ──────
    # validate_checkpoint.py phát hiện exploit trực tiếp từ per-step telemetry.
    # Không cần đợi plateau hay eval_balance để confirm — act ngay.
    warn_sigs = validation.get("warn_signals", {})
    early_exploits = []
    if warn_sigs.get("wheel_spin_mean_rads", 0) > 3.0:
        early_exploits.append("wheel_spin")
    if warn_sigs.get("xy_drift_max_m", 0) > 0.3:
        early_exploits.append("xy_drift")
    if warn_sigs.get("ctrl_jitter_mean_nm", 0) > 1.5:
        early_exploits.append("ctrl_jitter")
    if warn_sigs.get("leg_asymmetry_mean_rad", 0) > 0.15:
        early_exploits.append("leg_asymmetry")
    if warn_sigs.get("ang_vel_rms_rads", 0) > 0.5:
        early_exploits.append("torso_wobble")

    if early_exploits:
        return _decide_resume_tweak(
            report,
            reason=f"Exploit patterns phát hiện bởi validate_checkpoint: {early_exploits}. "
                   f"Cần fix config trước khi tiếp tục train.",
            config_changes=_exploit_config_fixes(early_exploits),
            symptom_key=early_exploits[0] + "_exploit",
        )

    # ── CRITICAL: không thể train tiếp ──────────────────────────────────────
    # Dùng fall_rate từ validation.benchmark (trustworthy — nominal rollout)
    if fall_rate > 0.5:
        source = "validation benchmark" if val_bench else "evaluate"
        return _decide_retrain_scratch(
            report,
            reason=f"fall_rate={fall_rate:.1%} > 50% — policy collapse ({source})"
        )

    # Fallback: kiểm tra scenarios nếu validation benchmark chưa có
    if not val_bench:
        nominal_scen = next((s for s in scenarios if s["scenario"] == "nominal"), None)
        if nominal_scen and nominal_scen["fall_rate"] > 0.5:
            return _decide_retrain_scratch(
                report, reason="fall_rate nominal > 50% — policy collapse (eval_balance)"
            )

    if reward_trend == "declining" and total_steps > 5_000_000:
        return _decide_retrain_scratch(
            report, reason="reward declining sau 5M steps — có thể diverge"
        )

    # ── ADVANCE: đủ điều kiện lên stage tiếp ────────────────────────────────
    SUCCESS_VALUES = {"balance": 7.0, "balance_robust": 6.0, "stand_up": 5.0}
    success_threshold = SUCCESS_VALUES.get(current_stage, 7.0)

    no_validation_flags = validation.get("num_suspicious", 0) == 0
    no_critical_warns   = len([w for w in warns if "CRITICAL" in w]) == 0

    if (eval_per_step >= success_threshold
            and success_rate >= 0.80
            and fall_rate <= 0.15
            and no_critical_warns
            and no_validation_flags):
        return _decide_advance_stage(report, eval_per_step, success_threshold)

    # ── CONTINUE: đang tốt, chưa đủ steps ───────────────────────────────────
    if (reward_trend == "improving"
            and fall_rate <= 0.20
            and len(warns) <= 2
            and validation.get("num_suspicious", 0) == 0):
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
    trend       = report.get("training_trend", {})
    total_steps = trend.get("total_steps", 0)
    scenarios   = report.get("scenarios", [])
    validation  = report.get("validation", {})

    # Kiểm tra exploit từ validation — dùng warn_signals (đúng key)
    # (Nếu có exploit ở đây, early-exit ở trên đã xử lý rồi.
    #  Đây là lần check thứ 2 với threshold rộng hơn để detect subtle exploits.)
    exploit_signals = []
    warn_sigs = validation.get("warn_signals", {})
    if warn_sigs.get("wheel_spin_mean_rads", 0) > 2.0:   # rộng hơn early-exit (3.0)
        exploit_signals.append("wheel_spin")
    if warn_sigs.get("xy_drift_max_m", 0) > 0.2:         # rộng hơn early-exit (0.3)
        exploit_signals.append("xy_drift")
    if warn_sigs.get("ctrl_jitter_mean_nm", 0) > 1.0:    # rộng hơn early-exit (1.5)
        exploit_signals.append("ctrl_jitter")
    if warn_sigs.get("height_std_m", 0) > 0.03:          # height oscillation
        exploit_signals.append("height_oscillation")

    if exploit_signals:
        return _decide_resume_tweak(
            report,
            reason=f"Plateau do subtle exploit từ validate_checkpoint: {exploit_signals}",
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
            reason=f"Plateau, gap={gap:.2f} reward/step so với threshold. "
                   f"Cần điều chỉnh reward weights.",
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

#### Phân tích warns từ scenarios (eval_balance)

```python
def _analyze_warns(report, warns, scenarios):
    # Tách warns theo nguồn gốc
    exploit_warns  = [w for w in warns if w.startswith("exploit.")]
    friction_warns = [w for w in warns if "friction_low" in w or "friction_high" in w]
    push_warns     = [w for w in warns if "push_recovery" in w or "max_recoverable_push" in w]
    nominal_warns  = [w for w in warns if "nominal" in w and not w.startswith("exploit.")]
    full_range_warns = [w for w in warns if "full_range" in w]

    config_changes = {}

    # Exploit warns từ validation đã được xử lý ở early-exit
    # Nếu vẫn còn ở đây là subtle exploit (threshold rộng hơn ở _analyze_plateau)
    if exploit_warns and not nominal_warns:
        return _decide_resume_tweak(
            report,
            reason=f"Exploit signals từ validate_checkpoint: {exploit_warns[:2]}",
            config_changes=_exploit_config_fixes_from_warns(exploit_warns)
        )

    if nominal_warns:
        return _decide_resume_tweak(
            report,
            reason="Warn trên nominal scenario — vấn đề cơ bản của policy",
            config_changes={"domain_randomization.push_magnitude": 0, "ppo.entropy_coeff": 0.01}
        )

    if friction_warns:
        config_changes["domain_randomization.friction_range"] = "[0.5, 1.5]"
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
        return _decide_resume_tweak(
            report, reason=f"Warns: {warns[:3]}", config_changes=config_changes
        )

    return _decide_continue(report, reason=f"Warns nhỏ ({len(warns)}), tiếp tục train")
```

---

## Bước 3 — Config change recipes

### Exploit fixes (từ validate_checkpoint signals)

```python
def _exploit_config_fixes(exploit_signals: list) -> dict:
    fixes = {}
    if "wheel_spin" in exploit_signals:
        fixes["rewards.wheel_velocity"] = -0.01    # tăng penalty (default -0.006)
        fixes["rewards.no_motion"]      = 0.6      # tăng thưởng đứng yên (default 0.5)
    if "xy_drift" in exploit_signals:
        fixes["rewards.position_drift"] = 2.5      # tăng drift penalty (default 1.5)
        fixes["rewards.wheel_velocity"] = -0.012
    if "ctrl_jitter" in exploit_signals:
        fixes["rewards.action_rate"]    = -0.08    # tăng penalty jitter (default -0.06)
    if "leg_asymmetry" in exploit_signals:
        fixes["rewards.symmetry"]       = 1.5      # tăng symmetry reward (default 1.0)
    if "height_oscillation" in exploit_signals:
        fixes["rewards.height"]         = 3.0      # tăng height tracking (default 2.5)
        fixes["ppo.entropy_coeff"]      = 0.002    # giảm exploration
    if "torso_wobble" in exploit_signals:
        fixes["rewards.body_level"]     = 2.0      # tăng body level reward (default 1.5)
        fixes["rewards.orientation"]    = 1.2      # tăng orientation reward (default 0.8)
    return fixes

def _exploit_config_fixes_from_warns(exploit_warns: list) -> dict:
    """Convert exploit warn strings thành config fixes."""
    fixes = {}
    for w in exploit_warns:
        if "wheel_spin" in w:
            fixes.update(_exploit_config_fixes(["wheel_spin"]))
        elif "xy_drift" in w:
            fixes.update(_exploit_config_fixes(["xy_drift"]))
        elif "ctrl_jitter" in w:
            fixes.update(_exploit_config_fixes(["ctrl_jitter"]))
    return fixes

def _reward_boost_config(stage: str) -> dict:
    if stage == "balance":
        return {
            "rewards.height":      3.0,    # tăng từ 2.5
            "rewards.body_level":  1.8,    # tăng từ 1.5
            "ppo.entropy_coeff":   0.003,  # giảm entropy
            "comment": "Boost height + body_level rewards để thoát plateau"
        }
    elif stage == "balance_robust":
        return {
            "rewards.natural_pose":                    2.0,
            "domain_randomization.push_magnitude":     50,
            "comment": "Tăng push magnitude và natural_pose để push recovery tốt hơn"
        }
    return {}
```

---

## Bước 4 — Format output

### Decision output format

```python
def format_decision(decision: dict) -> str:
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
        lines.append("📊 Sau khi chạy, verify bằng:")
        lines.append(f"   {decision['next_eval']}")

    lines.append("=" * 60)
    return "\n".join(lines)
```

### Ví dụ output — exploit từ validate_checkpoint

```
============================================================
🎯 TRAINING DECISION: RESUME_TWEAK
============================================================
Lý do: Exploit patterns phát hiện bởi validate_checkpoint: ['wheel_spin', 'xy_drift']
       Cần fix config trước khi tiếp tục train.

📝 Config thay đổi đề xuất:
   rewards.wheel_velocity: -0.012
   rewards.no_motion: 0.6
   rewards.position_drift: 2.5
   # Fix wheel-momentum và drift exploit

⚡ Command để thực thi:
   # 1. Sửa configs/training/balance.yaml theo config changes trên
   python scripts/train.py single --stage balance --seed 42 \
       --resume outputs/balance/rl/seed42/checkpoints/final

📊 Sau khi chạy, verify bằng:
   python scripts/validate_checkpoint.py \
       --checkpoint outputs/balance/rl/seed42/checkpoints/final
============================================================
```

---

## Bước 5 — Command templates

| Action          | Command                                                                                                                                                   |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CONTINUE        | `python scripts/train.py single --stage {stage} --seed {seed} --resume {ckpt}`                                                                            |
| RESUME_TWEAK    | `git diff configs/training/{stage}.yaml` rồi same resume command                                                                                          |
| RESUME_CKPT     | `--resume outputs/{stage}/rl/seed{seed}/checkpoints/step_{N}`                                                                                             |
| RETRAIN_STAGE   | Same command, **seed mới** (tránh overwrite path cũ), không `--resume`                                                                                    |
| RETRAIN_SCRATCH | `--stage balance --seed {new_seed} --steps 50000000` — không `--resume`                                                                                   |
| ADVANCE_STAGE   | `python scripts/train.py curriculum --steps-per-stage 10000000` hoặc `single --stage balance_robust --resume outputs/balance/rl/seed42/checkpoints/final` |

---

## Bước 6 — Special cases

**Chỉ có validation, chưa có eval_balance**: Vẫn ra được quyết định từ `validation.benchmark`. Note trong output: "chưa có multi-scenario eval".

**Curriculum stuck** (`max_retries reached` trong log):

```yaml
curriculum.success_value: <current - 0.5>
curriculum.max_retries_per_stage: <current + 2>
```

**3-seed divergence**: Bỏ seed outlier (> 2σ từ mean), train seed mới. Không average outlier vào paper.

**eval_per_step sai**: Kiểm tra `eval_per_step = eval_reward_mean / episode_length` — nếu < 1.0 thì episode_length đang dùng sai default.

---

## Quy tắc khi dùng skill này

1. **Không dùng skill này mà không có output từ eval-analyzer** — thiếu thông tin sẽ ra quyết định sai.
2. **`validation.warn_signals` là early-exit trigger** — exploit từ validate_checkpoint được xử lý TRƯỚC mọi thứ khác.
3. **Lấy fall_rate từ `validation.benchmark` trước** — trustworthy hơn scenarios từ eval_balance.
4. **ADVANCE_STAGE yêu cầu `num_suspicious == 0`** — không advance nếu còn exploit signals.
5. **Luôn kèm theo lý do** — giải thích metric nào trigger quyết định.
6. **Config change phải minimal** — chỉ 1-2 hyperparameter, không overhaul toàn bộ config.
7. **Sau mỗi quyết định, next_eval là validate_checkpoint** — không phải eval_balance.
8. **Phân biệt 2 curriculum systems**:
   - Within-stage height curriculum → theo dõi qua `curriculum/eval_per_step` trong JSONL
   - Multi-stage curriculum → `eval_per_step` vs `success_value` trong curriculum.yaml
