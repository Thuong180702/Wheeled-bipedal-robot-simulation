---
name: ConfigTuningAgent
description: >
  Agent nhận symptom (từ eval-analyzer, training-decision, hoặc user mô tả)
  và tạo ra config diff YAML chính xác, kiểm tra compatibility, và sinh lệnh
  train/resume sẵn sàng chạy. Orchestrates: config-advisor skill (recipe lookup)
  + compatibility checks + config file read/write. Không bao giờ thay đổi
  balance_robust reward semantics (no_motion/wheel_velocity). Output: YAML diff
  + commands + verification plan.
skills_used:
  - config-advisor
  - eval-analyzer   # optional: để đọc lại sau khi train với config mới
license: Project-internal agent
---

# ConfigTuningAgent

## Mục đích

Trả lời câu hỏi: **"Config của tôi cần thay đổi gì cụ thể, và thay đổi đó có an toàn không?"**

Khác với `config-advisor` skill (catalog recipes), agent này:
1. Đọc config hiện tại từ disk
2. Áp dụng recipe vào giá trị hiện tại (không hardcode old value)
3. Kiểm tra compatibility đầy đủ
4. Tạo file config mới với diff rõ ràng
5. Sinh lệnh train resume và verify

---

## Trigger patterns

- `"Fix wheel spin exploit trong balance"`
- `"Curriculum bị stuck, sửa config như nào?"`
- `"Tăng push robustness cho balance stage"`
- `"Apply recipe: poor_friction_generalization"`
- `"Tạo config mới từ balance với push=20N"`
- `"Balance_robust không recover, config cần thay đổi gì?"`
- `"Config diff cho training-decision: RESUME_TWEAK"`

---

## Inputs

```python
@dataclass
class ConfigTuningInput:
    # Symptom — bắt buộc ít nhất 1
    symptom_keys: list[str] | None = None   # từ config-advisor catalog
    free_text_request: str | None = None    # mô tả tự do nếu không có key

    # Context
    stage: str = "balance"
    seed: int = 42
    checkpoint_to_resume: str | None = None  # None = không resume
    target_steps: int = 5_000_000

    # Output control
    write_config: bool = False   # True = ghi config file mới
    config_variant_suffix: str = ""  # e.g. "_v2" → balance_v2.yaml
    outputs_dir: str = "outputs"
```

---

## Workflow

```
┌────────────────────────────────────────────────────────────────┐
│                   ConfigTuningAgent                             │
│                                                                 │
│  Phase 1: READ CURRENT CONFIG                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Load configs/training/<stage>.yaml                      │  │
│  │  Load configs/curriculum.yaml (nếu liên quan)            │  │
│  │  Phân tích config: detect flags, print summary           │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                        │                                        │
│  Phase 2: RESOLVE SYMPTOMS                                      │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │  symptom_keys → config-advisor recipes                   │  │
│  │  free_text → map tới gần nhất trong catalog              │  │
│  │  Merge nhiều recipes nếu có xung đột → resolve           │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                        │                                        │
│  Phase 3: APPLY & VALIDATE                                      │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │  Apply changes lên current config dict                   │  │
│  │  Chạy compatibility checks                               │  │
│  │  Compute config diff (old vs new)                        │  │
│  │  Predict training impact                                 │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                        │                                        │
│  Phase 4: OUTPUT                                                │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │  YAML diff + new config snippet                          │  │
│  │  Compatibility warnings                                  │  │
│  │  Train/resume commands                                   │  │
│  │  Verify commands (eval sau khi train)                    │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: READ CURRENT CONFIG

```python
import yaml, copy, difflib
from pathlib import Path
from dataclasses import dataclass, field

STAGE_CONFIG_MAP = {
    "balance":          "configs/training/balance.yaml",
    "balance_robust":   "configs/training/balance_robust.yaml",
    "stand_up":         "configs/training/stand_up.yaml",
    "wheeled_locomotion": "configs/training/wheeled_locomotion.yaml",
    "walking":          "configs/training/walking.yaml",
    "stair_climbing":   "configs/training/stair_climbing.yaml",
    "rough_terrain":    "configs/training/rough_terrain.yaml",
}

def read_config(stage: str) -> tuple[dict, str]:
    """Load config từ disk, trả về (config_dict, file_path)."""
    path = STAGE_CONFIG_MAP.get(stage)
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"Config không tồn tại: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg, path

def read_curriculum_config() -> dict:
    path = "configs/curriculum.yaml"
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)

def summarize_config(cfg: dict, stage: str) -> list[str]:
    """Tóm tắt config hiện tại, highlight các điểm đáng chú ý."""
    lines = []
    rewards = cfg.get("rewards", {})
    ppo = cfg.get("ppo", {})
    dr = cfg.get("domain_randomization", {})
    cur = cfg.get("curriculum", {})

    lines.append(f"Config: {STAGE_CONFIG_MAP.get(stage, 'unknown')}")
    lines.append(f"Task: {cfg.get('task', {}).get('name', '?')} | "
                 f"episode_length={cfg.get('task', {}).get('episode_length', '?')} | "
                 f"num_envs={cfg.get('task', {}).get('num_envs', '?')}")
    lines.append(f"PPO: lr={ppo.get('learning_rate', '?')} | "
                 f"entropy={ppo.get('entropy_coeff', '?')} | "
                 f"epochs={ppo.get('num_epochs', '?')}")

    if rewards:
        # Highlight key rewards
        key_rewards = ["body_level", "height", "natural_pose", "no_motion",
                       "wheel_velocity", "action_rate", "position_drift"]
        r_parts = [f"{k}={rewards[k]}" for k in key_rewards if k in rewards]
        lines.append("Rewards: " + " | ".join(r_parts))

    if dr:
        lines.append(f"DR: enabled={dr.get('enabled')} | "
                     f"friction={dr.get('friction_range')} | "
                     f"push_magnitude={dr.get('push_magnitude', 0)}")

    if cur:
        lines.append(f"Curriculum: enabled={cur.get('enabled')} | "
                     f"threshold={cur.get('reward_threshold')} | "
                     f"eval_interval={cur.get('eval_interval')}")

    # Flags từ config-advisor
    flags = analyze_config(cfg, stage)  # from config-advisor skill
    if flags:
        lines.append("⚠️  Flags: " + " | ".join(flags))

    return lines
```

---

## Phase 2: RESOLVE SYMPTOMS → RECIPES

```python
# Mapping free-text keywords → symptom_keys
FREE_TEXT_MAP = {
    "wheel spin":           "wheel_spin_exploit",
    "wheel spinning":       "wheel_spin_exploit",
    "drift":                "xy_drift_exploit",
    "xy drift":             "xy_drift_exploit",
    "jitter":               "ctrl_jitter",
    "chattering":           "ctrl_jitter",
    "asymmetry":            "leg_asymmetry",
    "bouncing":             "height_oscillation",
    "oscillat":             "height_oscillation",
    "curriculum stuck":     "curriculum_stuck_level",
    "level stuck":          "curriculum_stuck_level",
    "stage stuck":          "curriculum_stuck_stage",
    "promote":              "curriculum_stuck_stage",
    "demote loop":          "curriculum_demotion_loop",
    "oscillat.*stage":      "curriculum_demotion_loop",
    "friction":             "poor_friction_generalization",
    "push recovery":        "poor_push_recovery",
    "max push":             "poor_push_recovery",
    "fall rate":            "high_fall_rate_nominal",
    "falling":              "high_fall_rate_nominal",
    "plateau early":        "reward_plateau_early",
    "plateau":              "reward_plateau_late",
    "plateau late":         "reward_plateau_late",
    "not recover":          "balance_robust_wont_recover",
    "robust":               "balance_robust_wont_recover",
}

def resolve_symptoms(inp: ConfigTuningInput) -> list[str]:
    """Chuyển symptom_keys + free_text thành danh sách symptom keys đã resolve."""
    keys = list(inp.symptom_keys or [])

    if inp.free_text_request:
        text_lower = inp.free_text_request.lower()
        for keyword, key in FREE_TEXT_MAP.items():
            import re
            if re.search(keyword, text_lower):
                if key not in keys:
                    keys.append(key)

    return keys if keys else ["reward_plateau_late"]  # default nếu không rõ

def merge_recipes(recipes: list[dict]) -> dict:
    """
    Merge nhiều recipes thành một dict changes.
    Nếu xung đột, dùng recipe sau (override).
    Đặc biệt: không cho phép wheel_velocity/no_motion != 0 cho balance_robust.
    """
    merged = {}
    for recipe in recipes:
        for param, change in recipe.get("changes", {}).items():
            merged[param] = change
    return merged
```

---

## Phase 3: APPLY & VALIDATE

```python
def apply_changes_to_config(cfg: dict, changes: dict, stage: str) -> tuple[dict, list[str]]:
    """
    Apply changes dict lên config. Trả về (new_cfg, warnings).
    changes: {
        "rewards.wheel_velocity": {"new": -0.012, "reason": "..."},
        "ppo.entropy_coeff": {"new": 0.006, "reason": "..."},
        ...
    }
    """
    new_cfg = copy.deepcopy(cfg)
    warnings = []

    for param_path, change in changes.items():
        new_val = change["new"] if isinstance(change, dict) else change
        parts = param_path.split(".")
        section = parts[0]
        sub_key = ".".join(parts[1:]) if len(parts) > 1 else None

        if sub_key is None:
            new_cfg[section] = new_val
        else:
            # Handle nested keys
            nested_parts = sub_key.split(".")
            target = new_cfg.setdefault(section, {})
            for p in nested_parts[:-1]:
                target = target.setdefault(p, {})
            old_val = target.get(nested_parts[-1])
            target[nested_parts[-1]] = new_val

            if old_val is not None and old_val != new_val:
                pass  # Will be shown in diff

    return new_cfg, warnings

COMPATIBILITY_RULES = [
    {
        "name": "balance_robust_no_motion_zero",
        "check": lambda cfg, stage: (
            stage != "balance_robust" or
            cfg.get("rewards", {}).get("no_motion", 0) == 0.0
        ),
        "error": "balance_robust.yaml: rewards.no_motion PHẢI = 0.0",
        "severity": "CRITICAL",
    },
    {
        "name": "balance_robust_wheel_velocity_zero",
        "check": lambda cfg, stage: (
            stage != "balance_robust" or
            cfg.get("rewards", {}).get("wheel_velocity", 0) == 0.0
        ),
        "error": "balance_robust.yaml: rewards.wheel_velocity PHẢI = 0.0",
        "severity": "CRITICAL",
    },
    {
        "name": "entropy_coeff_minimum",
        "check": lambda cfg, stage: (
            cfg.get("ppo", {}).get("entropy_coeff", 0.004) >= 0.001
        ),
        "error": "entropy_coeff < 0.001 → risk mode collapse",
        "severity": "CRITICAL",
    },
    {
        "name": "action_rate_maximum",
        "check": lambda cfg, stage: (
            cfg.get("rewards", {}).get("action_rate", -0.06) >= -0.20
        ),
        "error": "action_rate < -0.20 quá restrictive → policy không reactive",
        "severity": "WARN",
    },
    {
        "name": "pid_enabled_for_warm_start",
        "check": lambda cfg, stage: (
            stage not in ("balance_robust",) or
            cfg.get("low_level_pid", {}).get("enabled", False) is True
        ),
        "error": "low_level_pid.enabled phải = true cho warm-start từ balance",
        "severity": "CRITICAL",
    },
    {
        "name": "success_value_unit_sanity",
        "check": lambda cfg, stage: True,  # cannot check curriculum.yaml here
        "error": "success_value phải là reward/step (max ~10.5 cho balance)",
        "severity": "INFO",
        "note": "Kiểm tra curriculum.yaml: success_value là reward/step, không phải episode sum",
    },
]

def run_compatibility_checks(new_cfg: dict, stage: str) -> list[dict]:
    """Chạy tất cả compatibility checks."""
    violations = []
    for rule in COMPATIBILITY_RULES:
        try:
            ok = rule["check"](new_cfg, stage)
        except Exception:
            ok = True  # skip nếu lỗi
        if not ok:
            violations.append({
                "name": rule["name"],
                "error": rule["error"],
                "severity": rule["severity"],
            })
    return violations

def compute_yaml_diff(old_cfg: dict, new_cfg: dict) -> str:
    """Tạo YAML diff string."""
    old_yaml = yaml.dump(old_cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    new_yaml = yaml.dump(new_cfg, default_flow_style=False, allow_unicode=True, sort_keys=False)
    diff = list(difflib.unified_diff(
        old_yaml.splitlines(), new_yaml.splitlines(),
        fromfile="config (current)", tofile="config (proposed)", n=3
    ))
    return "\n".join(diff)

def predict_impact(changes: dict, stage: str) -> list[str]:
    """Dự đoán tác động của từng thay đổi."""
    impacts = []
    IMPACT_MAP = {
        "rewards.wheel_velocity": "Giảm wheel spin exploit. Có thể tạm giảm reward trong 1-3M steps.",
        "rewards.no_motion":      "Tăng incentive đứng yên. Monitor wheel_spin_mean sau 2M steps.",
        "rewards.action_rate":    "Smooth action hơn. Monitor ctrl_jitter sau 2M steps.",
        "rewards.position_drift": "Giảm xy drift. Monitor xy_drift_max sau 3M steps.",
        "rewards.height":         "Tăng height tracking. Monitor height_rmse sau 3M steps.",
        "rewards.body_level":     "Tăng stability cơ bản. Monitor fall_rate sau 5M steps.",
        "rewards.natural_pose":   "Tăng posture quality. Monitor leg_asymmetry sau 3M steps.",
        "ppo.entropy_coeff":      "Thay đổi exploration. Monitor reward_trend sau 5M steps.",
        "ppo.learning_rate":      "Thay đổi tốc độ học. Monitor loss stability sau 2M steps.",
        "domain_randomization.friction_range": "Cải thiện friction generalization. Nominal performance có thể giảm nhẹ.",
        "domain_randomization.push_magnitude": "Training với push disturbance. Monitor fall_rate sau 5M steps.",
        "curriculum.reward_threshold":          "Curriculum advance dễ/khó hơn. Monitor curriculum level.",
        "curriculum.success_value":             "Stage promotion threshold thay đổi.",
        "curriculum.max_retries_per_stage":     "Cho thêm attempts trước khi force-promote.",
    }
    for param in changes:
        if param in IMPACT_MAP:
            impacts.append(f"  {param}: {IMPACT_MAP[param]}")
    return impacts
```

---

## Phase 4: OUTPUT FORMAT

```python
def format_output(inp: ConfigTuningInput,
                   old_cfg: dict, new_cfg: dict,
                   config_path: str,
                   resolved_symptoms: list[str],
                   violations: list[dict],
                   yaml_diff: str,
                   impacts: list[str],
                   all_changes: dict) -> str:
    lines = []

    critical_violations = [v for v in violations if v["severity"] == "CRITICAL"]
    warn_violations = [v for v in violations if v["severity"] == "WARN"]

    status = "✅ SAFE" if not critical_violations else "🔴 BLOCKED"
    lines += [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  CONFIG TUNING AGENT — {inp.stage:<37}║",
        f"║  Status: {status:<52}║",
        "╚══════════════════════════════════════════════════════════════╝", "",
    ]

    # ── Current config summary ─────────────────────────────────────────────
    lines += ["📋 CURRENT CONFIG", "─" * 62]
    for l in summarize_config(old_cfg, inp.stage):
        lines.append(f"  {l}")
    lines.append("")

    # ── Symptoms resolved ─────────────────────────────────────────────────
    lines += [f"🔍 SYMPTOMS ({len(resolved_symptoms)})", "─" * 62]
    for key in resolved_symptoms:
        recipe = SYMPTOM_RECIPES.get(key, {})  # from config-advisor skill
        desc = recipe.get("description", key)
        lines.append(f"  • {key}: {desc[:70]}")
    lines.append("")

    # ── Violations ────────────────────────────────────────────────────────
    if violations:
        lines += [f"{'🔴 COMPATIBILITY VIOLATIONS (BLOCKED)' if critical_violations else '⚠️  WARNINGS'}", "─" * 62]
        for v in violations:
            icon = "🔴" if v["severity"] == "CRITICAL" else "⚠️ "
            lines.append(f"  {icon} {v['name']}: {v['error']}")
        if critical_violations:
            lines.append("\n  ⛔ Không thể apply changes vì có CRITICAL violations.")
            lines.append("     Sửa violations trước rồi gọi lại agent.")
            lines.append("─" * 62)
            return "\n".join(lines)
        lines.append("")

    # ── Changes ─────────────────────────────────────────────────────────
    lines += [f"📝 CONFIG CHANGES ({len(all_changes)})", "─" * 62]
    for param, change in all_changes.items():
        if isinstance(change, dict):
            old_v = _get_nested(old_cfg, param)
            new_v = change["new"]
            reason = change.get("reason", "")
        else:
            old_v = _get_nested(old_cfg, param)
            new_v = change
            reason = ""
        lines.append(f"  {param}:")
        lines.append(f"    {old_v} → {new_v}   # {reason}")
    lines.append("")

    # ── YAML snippet (copy-paste ready) ─────────────────────────────────
    lines += ["📄 YAML CHANGES (copy-paste vào config file)", "─" * 62]
    # Group by section
    by_section = {}
    for param, change in all_changes.items():
        parts = param.split(".")
        section = parts[0]
        sub = ".".join(parts[1:])
        new_v = change["new"] if isinstance(change, dict) else change
        by_section.setdefault(section, []).append((sub, new_v,
            change.get("reason", "") if isinstance(change, dict) else ""))

    for section, items in by_section.items():
        lines.append(f"  {section}:")
        for sub, val, reason in items:
            comment = f"   # was: {_get_nested(old_cfg, f'{section}.{sub}')}"
            if reason:
                comment += f" | {reason[:50]}"
            lines.append(f"    {sub}: {val}{comment}")
    lines.append("")

    # ── Impact prediction ─────────────────────────────────────────────────
    if impacts:
        lines += ["📈 EXPECTED IMPACT", "─" * 62]
        lines.extend(impacts)
        lines.append("")

    # ── YAML diff ────────────────────────────────────────────────────────
    if yaml_diff:
        lines += ["🔍 FULL YAML DIFF", "─" * 62]
        for dl in yaml_diff.splitlines()[:50]:
            lines.append(f"  {dl}")
        diff_lines = [l for l in yaml_diff.splitlines()
                      if l.startswith(("+","-")) and not l.startswith(("+++","---"))]
        if len(diff_lines) > 50:
            lines.append(f"  ... ({len(diff_lines)-50} more lines)")
        lines.append("")

    # ── Commands ─────────────────────────────────────────────────────────
    lines += ["⚡ COMMANDS", "─" * 62]

    # Apply config
    lines += [
        "  1️⃣  Apply config changes:",
        f"     # Sửa {config_path} với YAML snippet ở trên",
        f"     # Hoặc nếu write_config=True, agent tự ghi file",
        "",
    ]

    # Verify thay đổi
    lines += [
        "  2️⃣  Verify config không có syntax error:",
        f"     python -c \"import yaml; yaml.safe_load(open('{config_path}'))\"",
        "",
    ]

    # Resume command
    seed = inp.seed
    stage = inp.stage
    steps = inp.target_steps

    if inp.checkpoint_to_resume:
        resume_flag = f"--resume {inp.checkpoint_to_resume}"
    else:
        resume_flag = f"--resume outputs/{stage}/rl/seed{seed}/checkpoints/final"

    lines += [
        "  3️⃣  Resume training với config mới:",
        f"     python scripts/train.py single \\",
        f"         --stage {stage} --seed {seed} \\",
        f"         --steps {steps} \\",
        f"         {resume_flag}",
        "",
    ]

    # Verify sau khi train
    verify_scenarios = _pick_verify_scenarios(resolved_symptoms)
    lines += [
        "  4️⃣  Eval sau khi train để verify fix:",
        f"     python scripts/eval_balance.py \\",
        f"         --checkpoint outputs/{stage}/rl/seed{seed}/checkpoints/final \\",
        f"         --scenarios {' '.join(verify_scenarios)} \\",
        f"         --num-episodes 20",
        "",
        "  5️⃣  Validate exploit patterns:",
        f"     python scripts/validate_checkpoint.py \\",
        f"         --checkpoint outputs/{stage}/rl/seed{seed}/checkpoints/final",
        "",
    ]

    lines += ["─" * 62]
    return "\n".join(lines)

def _get_nested(cfg: dict, param_path: str):
    """Lấy giá trị nested từ config dict."""
    parts = param_path.split(".")
    obj = cfg
    for p in parts:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return "N/A"
    return obj

def _pick_verify_scenarios(symptom_keys: list[str]) -> list[str]:
    """Chọn scenarios phù hợp để verify fix."""
    scenario_map = {
        "wheel_spin_exploit":           ["nominal"],
        "xy_drift_exploit":             ["nominal"],
        "ctrl_jitter":                  ["nominal"],
        "leg_asymmetry":                ["nominal", "full_range"],
        "poor_friction_generalization": ["friction_low", "friction_high", "nominal"],
        "poor_push_recovery":           ["push_recovery", "nominal"],
        "high_fall_rate_nominal":       ["nominal"],
        "curriculum_stuck_level":       ["nominal"],
        "curriculum_stuck_stage":       ["nominal"],
        "reward_plateau_early":         ["nominal"],
        "reward_plateau_late":          ["nominal"],
        "balance_robust_wont_recover":  ["push_recovery", "nominal"],
    }
    scenarios = set()
    for key in symptom_keys:
        scenarios.update(scenario_map.get(key, ["nominal"]))
    return list(scenarios) or ["nominal"]
```

---

## Entrypoint

```python
def run(inp: ConfigTuningInput) -> str:
    print(f"📖 Phase 1: Reading config for stage={inp.stage}...")
    try:
        old_cfg, config_path = read_config(inp.stage)
    except FileNotFoundError as e:
        return f"❌ {e}"

    config_summary = summarize_config(old_cfg, inp.stage)
    print("  Config loaded. Flags: " + str(
        [f for f in config_summary if "⚠️" in f][:2]
    ))

    print("🔍 Phase 2: Resolving symptoms...")
    symptoms = resolve_symptoms(inp)
    print(f"  Symptoms: {symptoms}")

    # Get recipes from config-advisor skill
    all_changes = {}
    for key in symptoms:
        recipe = SYMPTOM_RECIPES.get(key, {})
        changes = recipe.get("changes", {})
        all_changes.update(changes)

    if not all_changes:
        return (
            f"❌ Không tìm thấy recipe cho symptoms: {symptoms}\n"
            f"Available: {list(SYMPTOM_RECIPES.keys())}"
        )

    print(f"📝 Phase 3: Applying {len(all_changes)} changes + validation...")
    new_cfg, apply_warnings = apply_changes_to_config(old_cfg, all_changes, inp.stage)
    violations = run_compatibility_checks(new_cfg, inp.stage)
    yaml_diff = compute_yaml_diff(old_cfg, new_cfg)
    impacts = predict_impact(all_changes, inp.stage)

    if inp.write_config and not any(v["severity"] == "CRITICAL" for v in violations):
        suffix = inp.config_variant_suffix
        if suffix:
            new_path = config_path.replace(".yaml", f"{suffix}.yaml")
        else:
            new_path = config_path
        # Backup + write
        Path(new_path + ".bak").write_text(
            Path(config_path).read_text(encoding="utf-8"), encoding="utf-8"
        )
        with open(new_path, "w") as f:
            yaml.dump(new_cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        print(f"  Config written: {new_path} (backup: {new_path}.bak)")

    return format_output(
        inp, old_cfg, new_cfg, config_path,
        symptoms, violations, yaml_diff, impacts, all_changes
    )
```

---

## Ví dụ output

```
╔══════════════════════════════════════════════════════════════╗
║  CONFIG TUNING AGENT — balance                               ║
║  Status: ✅ SAFE                                             ║
╚══════════════════════════════════════════════════════════════╝

📋 CURRENT CONFIG
──────────────────────────────────────────────────────────────
  Config: configs/training/balance.yaml
  Task: balance | episode_length=1000 | num_envs=4096
  PPO: lr=0.0002 | entropy=0.004 | epochs=4
  Rewards: body_level=1.5 | height=2.5 | natural_pose=0.4 | no_motion=0.5 | wheel_velocity=-0.006 | action_rate=-0.06 | position_drift=1.5
  DR: enabled=True | friction=[0.7, 1.3] | push_magnitude=0

🔍 SYMPTOMS (1)
──────────────────────────────────────────────────────────────
  • poor_friction_generalization: performance giảm mạnh trên friction_low/high so với nominal

📝 CONFIG CHANGES (1)
──────────────────────────────────────────────────────────────
  domain_randomization.friction_range:
    [0.7, 1.3] → [0.5, 1.5]   # Mở rộng friction randomization từ ±30% → ±50%

📄 YAML CHANGES (copy-paste vào config file)
──────────────────────────────────────────────────────────────
  domain_randomization:
    friction_range: [0.5, 1.5]   # was: [0.7, 1.3] | Mở rộng từ ±30% → ±50%

📈 EXPECTED IMPACT
──────────────────────────────────────────────────────────────
  domain_randomization.friction_range: Cải thiện friction generalization. Nominal có thể giảm nhẹ.

⚡ COMMANDS
──────────────────────────────────────────────────────────────
  1️⃣  Apply config changes:
     # Sửa configs/training/balance.yaml với YAML snippet ở trên

  2️⃣  Verify config:
     python -c "import yaml; yaml.safe_load(open('configs/training/balance.yaml'))"

  3️⃣  Resume training:
     python scripts/train.py single \
         --stage balance --seed 42 \
         --steps 5000000 \
         --resume outputs/balance/rl/seed42/checkpoints/final

  4️⃣  Eval sau khi train:
     python scripts/eval_balance.py \
         --checkpoint outputs/balance/rl/seed42/checkpoints/final \
         --scenarios friction_low friction_high nominal \
         --num-episodes 20

  5️⃣  Validate exploit patterns:
     python scripts/validate_checkpoint.py \
         --checkpoint outputs/balance/rl/seed42/checkpoints/final
──────────────────────────────────────────────────────────────
```

---

## Quy tắc của agent

1. **CRITICAL violations = không apply** — blocked hoàn toàn cho đến khi user sửa.
2. **balance_robust reward semantics bảo vệ tuyệt đối** — no_motion và wheel_velocity không bao giờ được thay khác 0.
3. **Không overwrite config không cần backup** — luôn tạo .bak trước khi ghi.
4. **Merge recipe conflicts: recipe sau override recipe trước** — user được cảnh báo về xung đột.
5. **Chỉ 1-2 params mỗi lần** — nếu có > 3 params thay đổi, hỏi user confirm trước.
6. **Verify scenarios phải match symptom** — không eval nominal nếu symptom là friction.
7. **write_config=False là default** — không ghi file khi chưa được user bật.