---
name: CheckpointReviewAgent
description: >
  Agent tự động review một checkpoint (hoặc toàn bộ outputs/ của một stage),
  chạy validate_checkpoint nếu cần, tổng hợp metrics, và đưa ra quyết định
  train cụ thể kèm command sẵn sàng thực thi. Orchestrates 3 skills:
  checkpoint-manager → eval-analyzer → training-decision
  (→ config-advisor nếu cần config change).
skills_used:
  - checkpoint-manager
  - eval-analyzer
  - training-decision
  - config-advisor # optional, chỉ khi decision là RESUME_TWEAK hoặc CONFIG_CHANGE
license: Project-internal agent
---

# CheckpointReviewAgent

## Mục đích

Trả lời câu hỏi: **"Checkpoint này đang ở đâu, có tốt không, và tôi nên làm gì tiếp theo?"**

Thay vì phải tự chạy từng script và đọc output, agent này tự động:

1. Tìm và đọc tất cả thông tin sẵn có của checkpoint
2. Chạy `validate_checkpoint.py` nếu chưa có `validation_report.json`
3. Phân tích exploit signals và metrics, ra quyết định
4. Gợi ý chạy `eval_balance.py` nếu cần multi-scenario analysis

---

## Trigger patterns

Agent này được kích hoạt khi người dùng nói:

- `"Review checkpoint này cho tôi: <path>"`
- `"Checkpoint seed42 đang thế nào rồi?"`
- `"Tôi nên train tiếp hay train lại?"`
- `"Check xem balance stage có ổn không"`
- `"Phân tích outputs/balance/rl/seed42"`
- `"Validate xong rồi, giờ phải làm gì?"`

---

## Inputs

```python
@dataclass
class CheckpointReviewInput:
    # Bắt buộc có ít nhất 1 trong 2
    checkpoint_path: str | None = None   # e.g. "outputs/balance/rl/seed42/checkpoints/final"
    stage_dir: str | None = None         # e.g. "outputs/balance" — scan tất cả seeds

    # Optional
    stage: str = "balance"
    seed: int | None = None
    run_eval_if_missing: bool = True
    force_rerun_eval: bool = False
```

---

## Workflow chính

```
┌─────────────────────────────────────────────────────────────────┐
│                   CheckpointReviewAgent                         │
│                                                                  │
│  Phase 1: DISCOVER                                               │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ checkpoint-manager skill                                 │    │
│  │  • scan_checkpoints() → tìm tất cả ckpt                 │    │
│  │  • _read_checkpoint_meta() → step, reward, curriculum   │    │
│  │  • enrich_with_jsonl() → trend từ training log          │    │
│  │  • find_eval_files() → kiểm tra files có chưa           │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 2: EVAL CHECK (thứ tự ưu tiên mới)                        │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │  REQUIRED: validation_report.json                        │    │
│  │    → Nếu thiếu: sinh validate_checkpoint command        │    │
│  │    → Có thể proceed khi chỉ có validation + JSONL       │    │
│  │                                                          │    │
│  │  OPTIONAL: eval_balance eval_results.json               │    │
│  │    → Chỉ cần nếu muốn multi-scenario / paper metrics    │    │
│  │    → Sinh command nếu thiếu, nhưng KHÔNG block proceed  │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 3: ANALYZE                                                │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │ eval-analyzer skill                                      │    │
│  │  • load_validation_report() → exploit signals (PRIMARY) │    │
│  │  • extract_training_trend() từ JSONL (PRIMARY)          │    │
│  │  • load_eval_balance() nếu có (Secondary)               │    │
│  │  • build_report() → structured report dict              │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 4: DECIDE                                                 │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │ training-decision skill                                  │    │
│  │  • make_decision(report)                                 │    │
│  │    1. Early-exit: exploit từ validation.warn_signals    │    │
│  │    2. CRITICAL: fall_rate từ validation.benchmark       │    │
│  │    3. ADVANCE / CONTINUE / PLATEAU / WARNS              │    │
│  │  • Nếu RESUME_TWEAK → gọi config-advisor skill          │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 5: OUTPUT                                                 │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │  • Quality signals summary (từ validation)              │    │
│  │  • Scenario metrics (từ eval_balance, nếu có)           │    │
│  │  • Training trend                                        │    │
│  │  • Quyết định + lý do                                   │    │
│  │  • Commands: train + next_eval (validate trước)         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: DISCOVER — checkpoint-manager skill

```python
from pathlib import Path

def phase1_discover(input: CheckpointReviewInput) -> dict:
    if input.checkpoint_path:
        ckpt_dir     = Path(input.checkpoint_path)
        stage        = input.stage or _infer_stage_from_path(ckpt_dir)
        seed         = input.seed  or _infer_seed_from_path(ckpt_dir)
        outputs_root = ckpt_dir.parent.parent.parent.parent
    elif input.stage_dir:
        outputs_root = Path(input.stage_dir).parent
        stage        = Path(input.stage_dir).name
        seed         = input.seed
    else:
        raise ValueError("Cần cung cấp checkpoint_path hoặc stage_dir")

    all_ckpts = scan_checkpoints(
        outputs_dir=str(outputs_root),
        stage=stage,
        seeds=[seed] if seed else None,
    )
    if not all_ckpts:
        return {"error": f"Không tìm thấy checkpoint nào trong {outputs_root}/{stage}"}

    timeline     = enrich_with_jsonl(all_ckpts, outputs_dir=str(outputs_root))
    target_ckpts = [c for c in all_ckpts if c.path == str(ckpt_dir)] \
                   if input.checkpoint_path else all_ckpts
    if not target_ckpts:
        target_ckpts = all_ckpts

    best_per_seed = pick_best_per_seed(target_ckpts, stage)

    eval_status = {}
    for seed_num, ckpt in best_per_seed.items():
        found = find_eval_files(ckpt.path)
        eval_status[seed_num] = {
            "checkpoint":      ckpt,
            "eval_files":      found,
            "has_validation":  "validation" in found,   # PRIMARY — bắt buộc
            "has_eval_balance": "eval_balance" in found, # Secondary — optional
            "has_jsonl":       "jsonl" in found,
        }

    return {
        "stage": stage,
        "outputs_root": str(outputs_root),
        "all_checkpoints": all_ckpts,
        "best_per_seed": best_per_seed,
        "timeline": timeline,
        "eval_status": eval_status,
    }


def _infer_stage_from_path(path: Path) -> str:
    for i, part in enumerate(path.parts):
        if part == "rl" and i > 0:
            return path.parts[i - 1]
    return "balance"

def _infer_seed_from_path(path: Path) -> int | None:
    import re
    for part in path.parts:
        m = re.match(r"seed(\d+)", part)
        if m:
            return int(m.group(1))
    return None
```

---

## Phase 2: EVAL CHECK — validation là required, eval_balance là optional

```python
def phase2_eval_check(discovery: dict,
                       input: CheckpointReviewInput) -> dict:
    """
    Thứ tự ưu tiên mới:
    1. validation_report.json — REQUIRED để ra quyết định
    2. eval_results.json (eval_balance) — OPTIONAL, chỉ cần cho multi-scenario
    """
    eval_status   = discovery["eval_status"]
    stage         = discovery["stage"]
    needs_eval    = []
    eval_commands = []

    for seed_num, status in eval_status.items():
        ckpt      = status["checkpoint"]
        ckpt_path = ckpt.path

        # ── REQUIRED: validation_report.json ─────────────────────────────────
        if input.force_rerun_eval or not status["has_validation"]:
            needs_eval.append(seed_num)
            eval_commands.append(
                f"# Seed {seed_num} — REQUIRED: validate quality signals\n"
                f"python scripts/validate_checkpoint.py \\\n"
                f"    --checkpoint {ckpt_path}"
            )

        # ── OPTIONAL: eval_balance — chỉ suggest, không block ─────────────────
        if not status["has_eval_balance"]:
            eval_commands.append(
                f"# Seed {seed_num} — OPTIONAL: multi-scenario metrics (paper use)\n"
                f"python scripts/eval_balance.py \\\n"
                f"    --checkpoint {ckpt_path} \\\n"
                f"    --scenarios nominal push_recovery friction_low friction_high \\\n"
                f"    --num-episodes 20"
            )

    # Có thể proceed nếu: có validation (required) + JSONL
    # eval_balance thiếu không block
    has_validation_any = any(s["has_validation"] for s in eval_status.values())
    has_jsonl_any      = any(s["has_jsonl"] for s in eval_status.values())

    can_proceed = (
        has_validation_any or              # có validation → proceed
        not input.run_eval_if_missing      # user không muốn block
    )

    return {
        "needs_validation": needs_eval,    # seeds thiếu validation (REQUIRED)
        "eval_commands":    eval_commands,
        "can_proceed":      can_proceed,
        "has_validation_any":  has_validation_any,
        "has_eval_balance_any": any(s["has_eval_balance"] for s in eval_status.values()),
        "has_jsonl_any":    has_jsonl_any,
    }
```

---

## Phase 3: ANALYZE — eval-analyzer skill

```python
def phase3_analyze(discovery: dict) -> dict:
    """
    Chạy eval-analyzer trên tất cả files tìm được.
    Có thể analyze khi chỉ có validation + JSONL (không cần eval_balance).
    """
    eval_status      = discovery["eval_status"]
    timeline         = discovery["timeline"]
    stage            = discovery["stage"]
    reports_by_seed  = {}

    for seed_num, status in eval_status.items():
        # Proceed nếu có validation HOẶC eval_balance (không cần cả hai)
        has_data = status["has_validation"] or status["has_eval_balance"]
        if not has_data:
            continue

        found = status["eval_files"]
        ckpt  = status["checkpoint"]

        # build_report() xử lý đúng thứ tự ưu tiên:
        # validation → JSONL → evaluate → eval_balance
        report = build_report(
            eval_balance_path=found.get("eval_balance"),
            evaluate_path=found.get("evaluate_nominal"),
            validation_path=found.get("validation"),
            jsonl_path=found.get("jsonl"),
        )

        report["seed"]              = seed_num
        report["checkpoint_path"]   = ckpt.path
        report["checkpoint_step"]   = ckpt.step
        report["checkpoint_label"]  = ckpt.label
        report["eval_per_step_pkl"] = ckpt.eval_per_step
        report["curriculum_min_height"] = ckpt.curriculum_min_height
        report["current_stage"]     = stage

        # Note nếu chỉ có partial data
        report["data_completeness"] = {
            "has_validation":   status["has_validation"],
            "has_eval_balance": status["has_eval_balance"],
            "has_jsonl":        status["has_jsonl"],
            "note": "" if (status["has_validation"] and status["has_eval_balance"])
                    else "⚠️ Thiếu eval_balance — quyết định dựa trên validation + JSONL"
                         if status["has_validation"]
                    else "⚠️ Thiếu validation — quyết định chỉ dựa trên eval_balance"
        }

        # Enrich training trend nếu JSONL có nhưng build_report chưa nhận
        if not report.get("training_trend") and timeline:
            jsonl_metrics = get_metrics_at_step(timeline, stage, seed_num, ckpt.step)
            if jsonl_metrics:
                report["training_trend"] = {
                    "latest_eval_reward": jsonl_metrics.get("eval_reward_mean"),
                    "latest_fall_rate":   jsonl_metrics.get("eval_fall_rate"),
                    "reward_trend":       _estimate_trend(timeline, stage, seed_num),
                    "total_steps":        ckpt.step,
                }

        reports_by_seed[seed_num] = report

    return {
        "reports_by_seed": reports_by_seed,
        "stage":           stage,
        "analyzed_seeds":  list(reports_by_seed.keys()),
        "skipped_seeds":   [s for s in discovery["eval_status"] if s not in reports_by_seed],
    }


def _estimate_trend(timeline: dict, stage: str, seed: int) -> str:
    records   = timeline.get((stage, seed), [])
    eval_pts  = [(r["step"], r["eval/reward_mean"])
                 for r in records if "eval/reward_mean" in r]
    if len(eval_pts) < 3:
        return "insufficient_data"
    steps, rewards = zip(*eval_pts[-6:])
    delta = rewards[-1] - rewards[0]
    base  = max(abs(rewards[0]), 1.0)
    pct   = delta / base
    if pct > 0.03:    return "improving"
    elif pct < -0.03: return "declining"
    else:             return "plateau"
```

---

## Phase 4: DECIDE — training-decision skill

```python
def phase4_decide(analysis: dict, discovery: dict) -> dict:
    reports          = analysis["reports_by_seed"]
    decisions_by_seed = {}
    config_advices   = {}

    for seed_num, report in reports.items():
        decision = make_decision(report)
        decisions_by_seed[seed_num] = decision

        if decision["action"] in ("RESUME_TWEAK", "CONFIG_CHANGE"):
            symptom = decision.get("symptom_key")
            if symptom:
                advice = format_advice(symptom, current_stage=report["current_stage"])
                config_advices[seed_num] = advice

    primary_seed     = _pick_primary_seed(decisions_by_seed, discovery)
    primary_decision = decisions_by_seed.get(primary_seed)

    return {
        "decisions_by_seed": decisions_by_seed,
        "primary_seed":      primary_seed,
        "primary_decision":  primary_decision,
        "config_advices":    config_advices,
        "all_agree":         len(set(d["action"] for d in decisions_by_seed.values())) == 1,
    }


def _pick_primary_seed(decisions: dict, discovery: dict) -> int:
    ACTION_PRIORITY = {
        "RETRAIN_SCRATCH": 5,
        "RETRAIN_STAGE":   4,
        "RESUME_TWEAK":    3,
        "RESUME_CKPT":     3,
        "ADVANCE_STAGE":   2,
        "CONTINUE":        1,
    }
    if not decisions:
        return list(discovery["eval_status"].keys())[0]
    return max(decisions.keys(),
               key=lambda s: ACTION_PRIORITY.get(decisions[s]["action"], 0))
```

---

## Phase 5: OUTPUT — format báo cáo cuối

```python
def phase5_format_output(
    discovery: dict,
    eval_check: dict,
    analysis: dict,
    decision_result: dict,
) -> str:
    stage  = discovery["stage"]
    lines  = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  CHECKPOINT REVIEW REPORT — stage: {stage:<25}║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]

    # ── Section 1: Checkpoint Overview ────────────────────────────────────────
    lines += ["📦 CHECKPOINT OVERVIEW", "─" * 62]
    best_per_seed = discovery["best_per_seed"]
    for seed_num, ckpt in sorted(best_per_seed.items()):
        phase  = _curriculum_phase(ckpt.curriculum_min_height)
        ep_str = f"{ckpt.eval_per_step:.2f}" if ckpt.eval_per_step else "—"
        lines.append(
            f"  seed{seed_num:3d} │ {ckpt.label:<16} │ "
            f"step {ckpt.step:>10,} │ eval/step={ep_str} │ {phase}"
        )
    lines.append("")

    # ── Section 2: Eval Status ─────────────────────────────────────────────────
    lines += ["📊 EVAL STATUS", "─" * 62]

    if eval_check["needs_validation"]:
        lines.append(f"  ⚠️  Seeds cần validate trước: {eval_check['needs_validation']}")
        lines.append("")
        lines.append("  Chạy lệnh sau rồi gọi lại agent:")
        lines.append("")
        # Hiện validate commands trước
        for cmd in eval_check["eval_commands"]:
            if "validate_checkpoint" in cmd:
                for cmd_line in cmd.splitlines():
                    lines.append(f"    {cmd_line}")
                lines.append("")

        if not eval_check["can_proceed"]:
            lines.append("  ⏸  Dừng tại đây — cần validation_report.json trước khi quyết định.")
            return "\n".join(lines)

    # Gợi ý eval_balance (optional, không block)
    optional_cmds = [c for c in eval_check["eval_commands"] if "eval_balance" in c]
    if optional_cmds and not eval_check["has_eval_balance_any"]:
        lines.append("  ℹ️  eval_balance chưa chạy (optional — cần cho multi-scenario analysis):")
        for cmd in optional_cmds[:1]:  # chỉ show 1 example
            for cmd_line in cmd.splitlines():
                lines.append(f"    {cmd_line}")
        lines.append("")

    if analysis["skipped_seeds"]:
        lines.append(f"  ℹ️  Seeds không có data (skipped): {analysis['skipped_seeds']}")
    lines.append("")

    # ── Section 3: Quality Signals (validation — PRIMARY) ─────────────────────
    reports = analysis["reports_by_seed"]
    has_any_validation = any(r.get("validation") for r in reports.values())

    if has_any_validation:
        lines += ["🔍 QUALITY SIGNALS (validate_checkpoint — PRIMARY)", "─" * 62]
        for seed_num, report in sorted(reports.items()):
            val = report.get("validation", {})
            if not val:
                continue
            num_sus   = val.get("num_suspicious", 0)
            warn_sigs = val.get("warn_signals", {})
            lines.append(f"  seed{seed_num}: suspicious={num_sus} "
                         f"{'🔴' if num_sus > 0 else '✅'}")
            if warn_sigs:
                for sig, v in warn_sigs.items():
                    lines.append(f"    ⚠️  {sig}: {v:.2f}")
            else:
                bench = val.get("benchmark", {})
                fr    = bench.get("fall_rate", float("nan"))
                sr    = bench.get("success_rate", float("nan"))
                import math
                fr_str = f"{fr:.1%}" if not math.isnan(fr) else "—"
                sr_str = f"{sr:.1%}" if not math.isnan(sr) else "—"
                lines.append(f"    benchmark: fall_rate={fr_str}  success={sr_str}")
        lines.append("")

    # ── Section 4: Scenario Metrics (eval_balance — Secondary) ────────────────
    has_scenarios = any(r.get("scenarios") for r in reports.values())
    if has_scenarios:
        lines += ["📈 SCENARIO METRICS (eval_balance — optional)", "─" * 62]
        lines.append(
            f"  {'SEED':>5} │ {'SCENARIO':<18} │ "
            f"{'FALL%':>6} │ {'SURV(s)':>7} │ "
            f"{'PITCH°':>6} │ {'H_ERR':>6} │ "
            f"{'MAX_PUSH':>8} │ STATUS"
        )
        lines.append("  " + "─" * 80)

        for seed_num, report in sorted(reports.items()):
            for scenario_data in report.get("scenarios", []):
                import math
                fr   = scenario_data.get("fall_rate",            float("nan"))
                surv = scenario_data.get("survival_time_mean_s", float("nan"))
                pit  = scenario_data.get("pitch_rms_deg",        float("nan"))
                herr = scenario_data.get("height_rmse_m",        float("nan"))
                push = scenario_data.get("max_recoverable_push_n", float("nan"))
                status      = scenario_data.get("overall_status", "OK")
                status_icon = {"OK": "✅", "WARN": "⚠️ ", "CRITICAL": "🔴"}.get(status, "  ")
                lines.append(
                    f"  {seed_num:>5} │ {scenario_data['scenario']:<18} │ "
                    f"{'—' if math.isnan(fr) else f'{fr*100:.0f}%':>6} │ "
                    f"{'—' if math.isnan(surv) else f'{surv:.1f}':>7} │ "
                    f"{'—' if math.isnan(pit) else f'{pit:.2f}':>6} │ "
                    f"{'—' if math.isnan(herr) else f'{herr:.3f}':>6} │ "
                    f"{'—' if math.isnan(push) else f'{push:.0f}N':>8} │ "
                    f"{status_icon} {status}"
                )
            lines.append("  " + "─" * 80)

        all_warns = []
        for seed_num, report in reports.items():
            for w in report.get("warns", []):
                all_warns.append(f"  seed{seed_num}: {w}")
        if all_warns:
            lines += ["", "  ⚠️  WARNINGS:"]
            lines.extend(all_warns)
        lines.append("")

    # ── Section 5: Training Trend ──────────────────────────────────────────────
    has_trend = any(r.get("training_trend") for r in reports.values())
    if has_trend:
        lines += ["📉 TRAINING TREND", "─" * 62]
        for seed_num, report in sorted(reports.items()):
            trend = report.get("training_trend", {})
            if not trend:
                continue
            rew   = trend.get("latest_eval_reward")
            fr    = trend.get("latest_fall_rate")
            t     = trend.get("reward_trend", "unknown")
            steps = trend.get("total_steps", 0)
            icon  = {"improving": "📈", "plateau": "➡️ ", "declining": "📉",
                     "insufficient_data": "❓"}.get(t, "❓")
            rew_str = f"{rew:.1f}" if rew else "—"
            fr_str  = f"{fr:.1%}" if fr else "—"
            lines.append(
                f"  seed{seed_num}: {icon} {t.upper()}  "
                f"eval_reward={rew_str}  fall_rate={fr_str}  @ {steps:,} steps"
            )
        lines.append("")

    # ── Section 6: Decision ────────────────────────────────────────────────────
    lines += ["🎯 DECISION", "─" * 62]

    primary      = decision_result["primary_decision"]
    primary_seed = decision_result["primary_seed"]
    all_agree    = decision_result["all_agree"]

    if not primary:
        lines.append("  ❌ Không đủ dữ liệu để ra quyết định. Chạy validate_checkpoint trước.")
    else:
        if not all_agree and len(decision_result["decisions_by_seed"]) > 1:
            lines.append("  ⚠️  Các seeds có quyết định khác nhau:")
            for s, d in sorted(decision_result["decisions_by_seed"].items()):
                lines.append(f"    seed{s}: {d['action']} — {d['reason'][:60]}")
            lines.append(f"  → Hiển thị quyết định cho seed{primary_seed}:")
            lines.append("")

        action_icons = {
            "CONTINUE":        "▶️ ",
            "RESUME_TWEAK":    "🔧",
            "RESUME_CKPT":     "⏮️ ",
            "RETRAIN_STAGE":   "🔄",
            "RETRAIN_SCRATCH": "🆕",
            "ADVANCE_STAGE":   "⏭️ ",
        }
        icon = action_icons.get(primary["action"], "❓")
        lines += [f"  {icon} {primary['action']}", f"  Lý do: {primary['reason']}", ""]

        if decision_result["config_advices"]:
            lines.append("  📝 CONFIG CHANGES ĐỀ XUẤT:")
            for seed_num, advice in decision_result["config_advices"].items():
                lines.append(f"  (seed{seed_num})")
                for adv_line in advice.splitlines():
                    lines.append(f"    {adv_line}")
            lines.append("")

    # ── Section 7: Commands ────────────────────────────────────────────────────
    lines += ["⚡ COMMANDS", "─" * 62]

    if primary:
        commands = primary.get("commands", [])
        for cmd in commands:
            lines.append(f"  {cmd}")
        if commands:
            lines.append("")

        # next_eval: validate_checkpoint TRƯỚC eval_balance
        lines.append("  📊 Sau khi chạy, verify bằng:")
        if primary_seed and primary_seed in discovery["best_per_seed"]:
            ckpt_path = discovery["best_per_seed"][primary_seed].path
            lines += [
                f"  python scripts/validate_checkpoint.py \\",
                f"      --checkpoint {ckpt_path}",
            ]

    lines += [
        "",
        "─" * 62,
        "  💡 Để review lại sau khi train thêm:",
        "     → Gọi lại agent với cùng checkpoint path",
        "─" * 62,
    ]

    return "\n".join(lines)


def _curriculum_phase(min_height: float | None) -> str:
    if min_height is None:   return "curriculum=unknown"
    if min_height >= 0.65:   return f"Phase A ({min_height:.2f}m)"
    elif min_height >= 0.50: return f"Phase B ({min_height:.2f}m)"
    elif min_height > 0.40:  return f"Phase C ({min_height:.2f}m)"
    else:                    return f"✅ Complete ({min_height:.2f}m)"
```

---

## Entrypoint chính

```python
def run(input: CheckpointReviewInput) -> str:
    print("🔍 Phase 1: Discovering checkpoints...")
    discovery = phase1_discover(input)
    if "error" in discovery:
        return f"❌ ERROR: {discovery['error']}"

    print("📋 Phase 2: Checking eval status...")
    eval_check = phase2_eval_check(discovery, input)

    # Nếu không có validation và không có eval_balance → không analyze được gì
    if not eval_check["can_proceed"] and \
       not eval_check["has_validation_any"] and \
       not eval_check["has_eval_balance_any"]:
        partial = phase5_format_output(
            discovery, eval_check,
            {"reports_by_seed": {}, "analyzed_seeds": [], "skipped_seeds": []},
            {"decisions_by_seed": {}, "primary_seed": None, "primary_decision": None,
             "config_advices": {}, "all_agree": True},
        )
        return partial

    print("📊 Phase 3: Analyzing metrics...")
    analysis = phase3_analyze(discovery)

    print("🎯 Phase 4: Making decision...")
    decision_result = phase4_decide(analysis, discovery)

    print("📝 Phase 5: Formatting output...")
    return phase5_format_output(discovery, eval_check, analysis, decision_result)
```

---

## Edge cases

### Case 1: Chỉ có validation_report.json, chưa có eval_balance

Proceed bình thường. Decision dựa trên exploit signals + benchmark nominal.
Output có note: "⚠️ Thiếu eval_balance — quyết định dựa trên validation + JSONL".

### Case 2: Chỉ có JSONL, chưa có cả validation lẫn eval_balance

Sinh validate_checkpoint command và dừng. JSONL trend chưa đủ để ra quyết định.

### Case 3: Nhiều seeds với quyết định khác nhau

`_pick_primary_seed()` chọn seed có action nghiêm trọng nhất.
Output hiển thị tất cả decisions + highlight primary.

### Case 4: Curriculum chưa hoàn thành

Phát hiện qua `curriculum_min_height > 0.41`.
Thêm note vào decision: "curriculum chưa xong — cần train thêm steps".

---

## Quy tắc của agent

1. **validation_report.json là REQUIRED** — không ra quyết định khi chưa có.
2. **eval_balance là OPTIONAL** — không block proceed, chỉ suggest chạy.
3. **Không chạy eval tự động** — chỉ sinh commands, để user kiểm soát.
4. **Multi-seed awareness** — luôn review tất cả seeds có sẵn.
5. **Curriculum state luôn được report** — dù metrics tốt, nếu curriculum chưa xong thì không advance stage.
6. **next_eval luôn là validate_checkpoint** — không phải eval_balance.
7. **Commands phải copy-paste ready** — không có placeholder trong output cuối.
8. **Preserve checkpoint compatibility** — không suggest thay đổi phá vỡ checkpoint format.
