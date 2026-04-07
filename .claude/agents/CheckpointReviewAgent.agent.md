---
name: CheckpointReviewAgent
description: >
  Agent tự động review một checkpoint (hoặc toàn bộ outputs/ của một stage),
  chạy eval nếu cần, tổng hợp metrics, và đưa ra quyết định train cụ thể kèm
  command sẵn sàng thực thi. Orchestrates 3 skills: checkpoint-manager →
  eval-analyzer → training-decision (→ config-advisor nếu cần config change).
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
2. Chạy eval nếu chưa có kết quả
3. Phân tích metrics và phát hiện vấn đề
4. Ra quyết định cụ thể với command thực thi

---

## Trigger patterns

Agent này được kích hoạt khi người dùng nói:

- `"Review checkpoint này cho tôi: <path>"`
- `"Checkpoint seed42 đang thế nào rồi?"`
- `"Tôi nên train tiếp hay train lại?"`
- `"Check xem balance stage có ổn không"`
- `"Phân tích outputs/balance/rl/seed42"`
- `"Eval xong rồi, giờ phải làm gì?"`

---

## Inputs

```python
@dataclass
class CheckpointReviewInput:
    # Bắt buộc có ít nhất 1 trong 2
    checkpoint_path: str | None = None   # e.g. "outputs/balance/rl/seed42/checkpoints/final"
    stage_dir: str | None = None         # e.g. "outputs/balance" — scan tất cả seeds

    # Optional
    stage: str = "balance"               # balance | balance_robust | stand_up | ...
    seed: int | None = None              # None = tất cả seeds
    run_eval_if_missing: bool = True     # Tự chạy eval_balance.py nếu chưa có results
    eval_scenarios: list[str] = None     # None = ["nominal", "push_recovery", "friction_low", "friction_high"]
    force_rerun_eval: bool = False       # Rerun kể cả khi đã có eval results
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
│  │  • find_eval_files() → kiểm tra eval results có chưa   │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 2: EVAL (nếu thiếu)                                       │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │ Nếu run_eval_if_missing=True và chưa có eval_results:   │    │
│  │   → sinh eval command, báo user, dừng chờ               │    │
│  │ Nếu đã có eval_results:                                  │    │
│  │   → tiếp tục sang Phase 3                               │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 3: ANALYZE                                                │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │ eval-analyzer skill                                      │    │
│  │  • load eval_results.json / eval_results_*.json         │    │
│  │  • load validation_report.json nếu có                   │    │
│  │  • flag_metric() → WARN/CRITICAL trên từng metric       │    │
│  │  • extract_training_trend() từ JSONL                    │    │
│  │  • build_report() → structured report dict             │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 4: DECIDE                                                 │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │ training-decision skill                                  │    │
│  │  • make_decision(report) → action + reason              │    │
│  │  • Nếu RESUME_TWEAK → gọi config-advisor skill          │    │
│  │  • format_decision() → final output                     │    │
│  └───────────────────┬─────────────────────────────────────┘    │
│                      │                                           │
│  Phase 5: OUTPUT                                                 │
│  ┌───────────────────▼─────────────────────────────────────┐    │
│  │  • Bảng metrics tóm tắt                                 │    │
│  │  • Danh sách WARN/CRITICAL                              │    │
│  │  • Quyết định + lý do                                   │    │
│  │  • Command thực thi (copy-paste ready)                  │    │
│  │  • Next eval command để verify                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: DISCOVER — checkpoint-manager skill

```python
from pathlib import Path

def phase1_discover(input: CheckpointReviewInput) -> dict:
    """
    Thu thập toàn bộ thông tin có sẵn về checkpoint mà không chạy eval.
    """
    # 1a. Xác định scope
    if input.checkpoint_path:
        ckpt_dir = Path(input.checkpoint_path)
        stage = input.stage or _infer_stage_from_path(ckpt_dir)
        seed  = input.seed  or _infer_seed_from_path(ckpt_dir)
        outputs_root = ckpt_dir.parent.parent.parent.parent  # outputs/
    elif input.stage_dir:
        outputs_root = Path(input.stage_dir).parent
        stage = Path(input.stage_dir).name
        seed  = input.seed
    else:
        raise ValueError("Cần cung cấp checkpoint_path hoặc stage_dir")

    # 1b. Scan checkpoints (checkpoint-manager skill)
    all_ckpts = scan_checkpoints(
        outputs_dir=str(outputs_root),
        stage=stage,
        seeds=[seed] if seed else None,
    )
    if not all_ckpts:
        return {"error": f"Không tìm thấy checkpoint nào trong {outputs_root}/{stage}"}

    # 1c. Enrich với JSONL
    timeline = enrich_with_jsonl(all_ckpts, outputs_dir=str(outputs_root))

    # 1d. Xác định checkpoint cần review
    if input.checkpoint_path:
        target_ckpts = [c for c in all_ckpts if c.path == str(ckpt_dir)]
        if not target_ckpts:
            # Không match chính xác → dùng nearest
            target_ckpts = all_ckpts
    else:
        target_ckpts = all_ckpts

    # 1e. Pick best per seed
    best_per_seed = pick_best_per_seed(target_ckpts, stage)

    # 1f. Tìm eval files
    eval_status = {}
    for seed_num, ckpt in best_per_seed.items():
        found = find_eval_files(ckpt.path)
        eval_status[seed_num] = {
            "checkpoint": ckpt,
            "eval_files": found,
            "has_eval_balance": "eval_balance" in found,
            "has_validation": "validation" in found,
            "has_jsonl": "jsonl" in found,
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
    """Suy ra stage từ đường dẫn checkpoint."""
    parts = path.parts
    # outputs/<stage>/rl/seed<N>/checkpoints/<label>
    for i, part in enumerate(parts):
        if part == "rl" and i > 0:
            return parts[i - 1]
    return "balance"  # default

def _infer_seed_from_path(path: Path) -> int | None:
    import re
    for part in path.parts:
        m = re.match(r"seed(\d+)", part)
        if m:
            return int(m.group(1))
    return None
```

---

## Phase 2: EVAL — kiểm tra và sinh eval command

```python
def phase2_eval_check(discovery: dict,
                       input: CheckpointReviewInput) -> dict:
    """
    Kiểm tra eval results có đủ không.
    Nếu thiếu và run_eval_if_missing=True → sinh eval commands.
    """
    eval_status = discovery["eval_status"]
    stage = discovery["stage"]
    needs_eval = []
    eval_commands = []

    # Scenarios mặc định cho review nhanh (không chạy full)
    default_scenarios = input.eval_scenarios or [
        "nominal", "push_recovery", "friction_low", "friction_high"
    ]

    for seed_num, status in eval_status.items():
        ckpt = status["checkpoint"]
        needs = []

        if input.force_rerun_eval or not status["has_eval_balance"]:
            needs.append("eval_balance")
        if not status["has_validation"]:
            needs.append("validation")

        if needs:
            needs_eval.append(seed_num)
            ckpt_path = ckpt.path

            if "eval_balance" in needs:
                scenarios_str = " \\\n        ".join(
                    f"--scenarios {s}" for s in default_scenarios
                )
                eval_commands.append(
                    f"# Seed {seed_num}: eval_balance\n"
                    f"python scripts/eval_balance.py \\\n"
                    f"    --checkpoint {ckpt_path} \\\n"
                    f"    {scenarios_str} \\\n"
                    f"    --num-episodes 20"
                )

            if "validation" in needs:
                eval_commands.append(
                    f"# Seed {seed_num}: validate checkpoint\n"
                    f"python scripts/validate_checkpoint.py \\\n"
                    f"    --checkpoint {ckpt_path}"
                )

    return {
        "needs_eval": needs_eval,
        "eval_commands": eval_commands,
        "can_proceed": len(needs_eval) == 0 or not input.run_eval_if_missing,
        "has_partial_eval": any(
            s["has_eval_balance"] for s in eval_status.values()
        ),
    }
```

---

## Phase 3: ANALYZE — eval-analyzer skill

```python
def phase3_analyze(discovery: dict) -> dict:
    """
    Chạy eval-analyzer skill trên tất cả eval files tìm được.
    Trả về report tổng hợp cho từng seed, và report aggregate.
    """
    eval_status = discovery["eval_status"]
    timeline = discovery["timeline"]
    stage = discovery["stage"]
    reports_by_seed = {}

    for seed_num, status in eval_status.items():
        if not status["has_eval_balance"]:
            continue  # skip nếu chưa có eval

        found = status["eval_files"]
        ckpt = status["checkpoint"]

        # Chạy eval-analyzer build_report()
        report = build_report(
            eval_balance_path=found.get("eval_balance"),
            evaluate_path=found.get("evaluate_nominal"),
            validation_path=found.get("validation"),
            jsonl_path=found.get("jsonl"),
        )

        # Thêm metadata
        report["seed"] = seed_num
        report["checkpoint_path"] = ckpt.path
        report["checkpoint_step"] = ckpt.step
        report["checkpoint_label"] = ckpt.label
        report["eval_per_step_pkl"] = ckpt.eval_per_step
        report["curriculum_min_height"] = ckpt.curriculum_min_height
        report["current_stage"] = stage

        # Enrich training trend nếu JSONL có nhưng report chưa có
        if not report.get("training_trend") and timeline:
            jsonl_metrics = get_metrics_at_step(
                timeline, stage, seed_num, ckpt.step
            )
            if jsonl_metrics:
                report["training_trend"] = {
                    "latest_eval_reward": jsonl_metrics.get("eval_reward_mean"),
                    "latest_fall_rate": jsonl_metrics.get("eval_fall_rate"),
                    "reward_trend": _estimate_trend(timeline, stage, seed_num),
                    "total_steps": ckpt.step,
                }

        reports_by_seed[seed_num] = report

    return {
        "reports_by_seed": reports_by_seed,
        "stage": stage,
        "analyzed_seeds": list(reports_by_seed.keys()),
        "skipped_seeds": [
            s for s in discovery["eval_status"]
            if s not in reports_by_seed
        ],
    }


def _estimate_trend(timeline: dict, stage: str, seed: int) -> str:
    """Ước tính trend từ các điểm eval_reward_mean trong JSONL."""
    records = timeline.get((stage, seed), [])
    eval_pts = [(r["step"], r["eval/reward_mean"])
                for r in records if "eval/reward_mean" in r]
    if len(eval_pts) < 3:
        return "insufficient_data"
    steps, rewards = zip(*eval_pts[-6:])  # last 6 points
    delta = rewards[-1] - rewards[0]
    if delta > 0.5:
        return "improving"
    elif delta < -0.5:
        return "declining"
    else:
        return "plateau"
```

---

## Phase 4: DECIDE — training-decision skill

```python
def phase4_decide(analysis: dict, discovery: dict) -> dict:
    """
    Chạy training-decision skill trên từng seed và tổng hợp.
    Nếu có nhiều seeds, pick seed cần action nhất.
    """
    reports = analysis["reports_by_seed"]
    decisions_by_seed = {}
    config_advices = {}

    for seed_num, report in reports.items():
        decision = make_decision(report)
        decisions_by_seed[seed_num] = decision

        # Nếu decision cần config change → gọi config-advisor
        if decision["action"] in ("RESUME_TWEAK", "CONFIG_CHANGE"):
            symptom = decision.get("symptom_key")
            if symptom:
                advice = format_advice(symptom, current_stage=report["current_stage"])
                config_advices[seed_num] = advice

    # Tổng hợp: nếu nhiều seeds, quyết định nào quan trọng nhất?
    primary_seed = _pick_primary_seed(decisions_by_seed, discovery)
    primary_decision = decisions_by_seed.get(primary_seed)

    return {
        "decisions_by_seed": decisions_by_seed,
        "primary_seed": primary_seed,
        "primary_decision": primary_decision,
        "config_advices": config_advices,
        "all_agree": len(set(d["action"] for d in decisions_by_seed.values())) == 1,
    }


def _pick_primary_seed(decisions: dict, discovery: dict) -> int:
    """
    Chọn seed 'chính' để hiển thị quyết định:
    - Ưu tiên seed có action nghiêm trọng nhất (RETRAIN > RESUME > CONTINUE)
    - Nếu bằng nhau, chọn seed có eval_per_step cao nhất
    """
    ACTION_PRIORITY = {
        "RETRAIN_SCRATCH": 5,
        "RETRAIN_STAGE": 4,
        "RESUME_TWEAK": 3,
        "RESUME_CKPT": 3,
        "ADVANCE_STAGE": 2,
        "CONTINUE": 1,
    }
    if not decisions:
        return list(discovery["eval_status"].keys())[0]

    return max(
        decisions.keys(),
        key=lambda s: ACTION_PRIORITY.get(decisions[s]["action"], 0)
    )
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
    """
    Format toàn bộ kết quả thành một báo cáo có cấu trúc rõ ràng.
    """
    stage = discovery["stage"]
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  CHECKPOINT REVIEW REPORT — stage: {stage:<25}║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]

    # ── Section 1: Checkpoint Overview ────────────────────────────────────────
    lines.append("📦 CHECKPOINT OVERVIEW")
    lines.append("─" * 62)
    best_per_seed = discovery["best_per_seed"]
    for seed_num, ckpt in sorted(best_per_seed.items()):
        phase = _curriculum_phase(ckpt.curriculum_min_height)
        ep_str = f"{ckpt.eval_per_step:.2f}" if ckpt.eval_per_step else "—"
        lines.append(
            f"  seed{seed_num:3d} │ {ckpt.label:<16} │ "
            f"step {ckpt.step:>10,} │ "
            f"eval/step={ep_str} │ "
            f"curriculum={phase}"
        )
    lines.append("")

    # ── Section 2: Eval Status ─────────────────────────────────────────────────
    lines.append("📊 EVAL STATUS")
    lines.append("─" * 62)

    if eval_check["needs_eval"]:
        lines.append(f"  ⚠️  Seeds cần eval trước: {eval_check['needs_eval']}")
        lines.append("")
        lines.append("  Chạy các lệnh sau rồi gọi lại agent:")
        lines.append("")
        for cmd in eval_check["eval_commands"]:
            for cmd_line in cmd.splitlines():
                lines.append(f"    {cmd_line}")
            lines.append("")

        if not eval_check["can_proceed"]:
            lines.append("  ⏸  Dừng tại đây — cần eval xong trước khi quyết định.")
            return "\n".join(lines)

    if analysis["skipped_seeds"]:
        lines.append(f"  ℹ️  Seeds không có eval data (skipped): {analysis['skipped_seeds']}")
    lines.append("")

    # ── Section 3: Metrics Summary ─────────────────────────────────────────────
    reports = analysis["reports_by_seed"]
    if reports:
        lines.append("📈 METRICS SUMMARY")
        lines.append("─" * 62)

        # Header row
        lines.append(
            f"  {'SEED':>5} │ {'SCENARIO':<18} │ "
            f"{'FALL%':>6} │ {'SURV(s)':>7} │ "
            f"{'PITCH°':>6} │ {'H_ERR':>6} │ "
            f"{'MAX_PUSH':>8} │ STATUS"
        )
        lines.append("  " + "─" * 80)

        for seed_num, report in sorted(reports.items()):
            for scenario_data in report.get("scenarios", []):
                fr   = scenario_data.get("fall_rate", float("nan"))
                surv = scenario_data.get("survival_time_mean_s", float("nan"))
                pit  = scenario_data.get("pitch_rms_deg", float("nan"))
                herr = scenario_data.get("height_rmse_m", float("nan"))
                push = scenario_data.get("max_recoverable_push_n", float("nan"))
                status = scenario_data.get("overall_status", "OK")
                status_icon = {"OK": "✅", "WARN": "⚠️ ", "CRITICAL": "🔴"}.get(status, "  ")

                import math
                lines.append(
                    f"  {seed_num:>5} │ {scenario_data['scenario']:<18} │ "
                    f"{'—' if math.isnan(fr) else f'{fr*100:.0f}%':>6} │ "
                    f"{'—' if math.isnan(surv) else f'{surv:.1f}':>7} │ "
                    f"{'—' if math.isnan(pit)  else f'{pit:.2f}':>6} │ "
                    f"{'—' if math.isnan(herr) else f'{herr:.3f}':>6} │ "
                    f"{'—' if math.isnan(push) else f'{push:.0f}N':>8} │ "
                    f"{status_icon} {status}"
                )
            lines.append("  " + "─" * 80)

        # Warnings list
        all_warns = []
        for seed_num, report in reports.items():
            for w in report.get("warns", []):
                all_warns.append(f"  seed{seed_num}: {w}")
        if all_warns:
            lines.append("")
            lines.append("  ⚠️  WARNINGS:")
            lines.extend(all_warns)
        lines.append("")

    # ── Section 4: Training Trend ──────────────────────────────────────────────
    has_trend = any(
        r.get("training_trend") for r in reports.values()
    )
    if has_trend:
        lines.append("📉 TRAINING TREND")
        lines.append("─" * 62)
        for seed_num, report in sorted(reports.items()):
            trend = report.get("training_trend", {})
            if not trend:
                continue
            rew = trend.get("latest_eval_reward")
            fr  = trend.get("latest_fall_rate")
            t   = trend.get("reward_trend", "unknown")
            steps = trend.get("total_steps", 0)
            trend_icon = {"improving": "📈", "plateau": "➡️ ", "declining": "📉", "insufficient_data": "❓"}.get(t, "❓")
            lines.append(
                f"  seed{seed_num}: {trend_icon} {t.upper()}"
                f"  eval_reward={rew:.1f if rew else '—'}"
                f"  fall_rate={fr:.1%if fr else '—'}"
                f"  @ {steps:,} steps"
            )
        lines.append("")

    # ── Section 5: Decision ────────────────────────────────────────────────────
    lines.append("🎯 DECISION")
    lines.append("─" * 62)

    primary = decision_result["primary_decision"]
    primary_seed = decision_result["primary_seed"]
    all_agree = decision_result["all_agree"]

    if not primary:
        lines.append("  ❌ Không đủ dữ liệu để ra quyết định. Chạy eval trước.")
    else:
        if not all_agree and len(decision_result["decisions_by_seed"]) > 1:
            lines.append("  ⚠️  Các seeds có quyết định khác nhau:")
            for s, d in sorted(decision_result["decisions_by_seed"].items()):
                lines.append(f"    seed{s}: {d['action']} — {d['reason'][:60]}")
            lines.append(f"  → Hiển thị quyết định cho seed{primary_seed} (ưu tiên cao nhất):")
            lines.append("")

        action_icons = {
            "CONTINUE":       "▶️ ",
            "RESUME_TWEAK":   "🔧",
            "RESUME_CKPT":    "⏮️ ",
            "RETRAIN_STAGE":  "🔄",
            "RETRAIN_SCRATCH":"🆕",
            "ADVANCE_STAGE":  "⏭️ ",
        }
        icon = action_icons.get(primary["action"], "❓")
        lines += [
            f"  {icon} {primary['action']}",
            f"  Lý do: {primary['reason']}",
            "",
        ]

        # Config advice nếu có
        if decision_result["config_advices"]:
            lines.append("  📝 CONFIG CHANGES ĐỀ XUẤT:")
            for seed_num, advice in decision_result["config_advices"].items():
                lines.append(f"  (seed{seed_num})")
                for adv_line in advice.splitlines():
                    lines.append(f"    {adv_line}")
            lines.append("")

    # ── Section 6: Commands ────────────────────────────────────────────────────
    lines.append("⚡ COMMANDS")
    lines.append("─" * 62)

    if primary:
        commands = primary.get("commands", [])
        if commands:
            for cmd in commands:
                lines.append(f"  {cmd}")
            lines.append("")

        next_eval = primary.get("next_eval")
        if next_eval:
            lines.append("  📊 Sau khi chạy, verify bằng:")
            lines.append(f"  {next_eval}")
            lines.append("")

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines += [
        "─" * 62,
        "  💡 Để review lại sau khi train thêm:",
        f"     → Gọi lại agent với cùng checkpoint path",
        "─" * 62,
    ]

    return "\n".join(lines)


def _curriculum_phase(min_height: float | None) -> str:
    if min_height is None:
        return "unknown"
    if min_height >= 0.65:
        return f"Phase A ({min_height:.2f}m)"
    elif min_height >= 0.50:
        return f"Phase B ({min_height:.2f}m)"
    elif min_height > 0.40:
        return f"Phase C ({min_height:.2f}m)"
    else:
        return f"✅ Complete ({min_height:.2f}m)"
```

---

## Entrypoint chính

```python
def run(input: CheckpointReviewInput) -> str:
    """
    Main entrypoint của CheckpointReviewAgent.
    Orchestrates toàn bộ workflow từ discover → decide → output.
    """
    print("🔍 Phase 1: Discovering checkpoints...")
    discovery = phase1_discover(input)
    if "error" in discovery:
        return f"❌ ERROR: {discovery['error']}"

    print("📋 Phase 2: Checking eval status...")
    eval_check = phase2_eval_check(discovery, input)

    if not eval_check["can_proceed"] and not eval_check["has_partial_eval"]:
        # Không có gì để phân tích → chỉ show eval commands
        partial_report = phase5_format_output(
            discovery, eval_check,
            {"reports_by_seed": {}, "analyzed_seeds": [], "skipped_seeds": []},
            {"decisions_by_seed": {}, "primary_seed": None, "primary_decision": None,
             "config_advices": {}, "all_agree": True},
        )
        return partial_report

    print("📊 Phase 3: Analyzing metrics...")
    analysis = phase3_analyze(discovery)

    print("🎯 Phase 4: Making decision...")
    decision_result = phase4_decide(analysis, discovery)

    print("📝 Phase 5: Formatting output...")
    return phase5_format_output(discovery, eval_check, analysis, decision_result)
```

---

## Ví dụ output

```
╔══════════════════════════════════════════════════════════════╗
║  CHECKPOINT REVIEW REPORT — stage: balance                   ║
╚══════════════════════════════════════════════════════════════╝

📦 CHECKPOINT OVERVIEW
──────────────────────────────────────────────────────────────
  seed 42 │ final            │ step 12,000,000 │ eval/step=7.48 │ curriculum=✅ Complete (0.40m)
  seed113 │ final            │ step 12,000,000 │ eval/step=7.22 │ curriculum=Phase C (0.42m)
  seed999 │ step_10000000    │ step 10,000,000 │ eval/step=6.91 │ curriculum=Phase C (0.45m)

📊 EVAL STATUS
──────────────────────────────────────────────────────────────
  ⚠️  Seeds cần eval trước: [999]

  Chạy các lệnh sau rồi gọi lại agent:

    # Seed 999: eval_balance
    python scripts/eval_balance.py \
        --checkpoint outputs/balance/rl/seed999/checkpoints/step_10000000 \
        --scenarios nominal \
        --scenarios push_recovery \
        --scenarios friction_low \
        --scenarios friction_high \
        --num-episodes 20

📈 METRICS SUMMARY
──────────────────────────────────────────────────────────────
   SEED │ SCENARIO           │  FALL% │ SURV(s) │ PITCH° │  H_ERR │ MAX_PUSH │ STATUS
  ──────────────────────────────────────────────────────────────────────────────────
     42 │ nominal            │     5% │    19.8 │   1.21 │  0.012 │       — │ ✅ OK
     42 │ push_recovery      │    12% │    18.3 │   2.10 │  0.019 │     87N │ ✅ OK
     42 │ friction_low       │    28% │    14.1 │   3.50 │  0.026 │     52N │ ⚠️  WARN
     42 │ friction_high      │     8% │    19.2 │   1.40 │  0.014 │     79N │ ✅ OK
  ──────────────────────────────────────────────────────────────────────────────────
    113 │ nominal            │     7% │    19.1 │   1.45 │  0.014 │       — │ ✅ OK
    113 │ push_recovery      │    18% │    17.8 │   2.30 │  0.021 │     71N │ ✅ OK
    113 │ friction_low       │    35% │    12.9 │   4.10 │  0.031 │     45N │ ⚠️  WARN
    113 │ friction_high      │    10% │    18.8 │   1.60 │  0.016 │     68N │ ✅ OK
  ──────────────────────────────────────────────────────────────────────────────────

  ⚠️  WARNINGS:
  seed42:  friction_low.fall_rate=0.28 [WARN]
  seed113: friction_low.fall_rate=0.35 [WARN]
  seed113: friction_low.height_rmse_m=0.031 [WARN]

📉 TRAINING TREND
──────────────────────────────────────────────────────────────
  seed42:  ➡️  PLATEAU  eval_reward=7480  fall_rate=5.2%  @ 12,000,000 steps
  seed113: 📈 IMPROVING eval_reward=7220  fall_rate=7.1%  @ 12,000,000 steps

🎯 DECISION
──────────────────────────────────────────────────────────────
  ⚠️  Các seeds có quyết định khác nhau:
    seed42:  RESUME_TWEAK — friction_low warns, cần mở rộng domain randomization
    seed113: CONTINUE — đang improving, chưa đủ steps

  → Hiển thị quyết định cho seed42 (ưu tiên cao nhất):

  🔧 RESUME_TWEAK
  Lý do: Warns trên friction_low (fall_rate=28%, 35%) — friction_range quá hẹp

  📝 CONFIG CHANGES ĐỀ XUẤT:
  (seed42)
    # Fix: poor_friction_generalization
    domain_randomization:
      friction_range: [0.5, 1.5]   # was: [0.7, 1.3]

⚡ COMMANDS
──────────────────────────────────────────────────────────────
  # 1. Sửa configs/training/balance.yaml:
  #    domain_randomization.friction_range: [0.5, 1.5]

  python scripts/train.py single --stage balance --seed 42 \
      --resume outputs/balance/rl/seed42/checkpoints/final \
      --steps 5000000

  📊 Sau khi chạy, verify bằng:
  python scripts/eval_balance.py \
      --checkpoint outputs/balance/rl/seed42/checkpoints/final \
      --scenarios friction_low friction_high nominal

──────────────────────────────────────────────────────────────
  💡 Để review lại sau khi train thêm:
     → Gọi lại agent với cùng checkpoint path
──────────────────────────────────────────────────────────────
```

---

## Edge cases và xử lý đặc biệt

### Case 1: Checkpoint chưa có eval nào

```python
# Phase 2 phát hiện → sinh eval commands → dừng chờ user
# Output chỉ có Section 1 (Overview) + Section 2 (Eval commands)
# Không cố gắng ra quyết định khi chưa có data
```

### Case 2: Chỉ có JSONL, chưa có eval_results.json

```python
# Phase 3: dùng training trend từ JSONL để phân tích sơ bộ
# training-decision nhận training_trend thay vì eval scenarios
# Decision có thể là CONTINUE (if improving) hoặc "Cần eval để quyết định chính xác"
```

### Case 3: Nhiều seeds với quyết định khác nhau

```python
# Phase 4: _pick_primary_seed() chọn seed có action ưu tiên cao nhất
# Output Section 5 hiển thị tất cả decisions + highlight primary
# Commands được gen cho primary seed
```

### Case 4: Curriculum chưa hoàn thành

```python
# Phát hiện qua curriculum_min_height > 0.41
# Thêm note vào decision: "curriculum chưa xong — cần train thêm steps"
# eval_per_step có thể đã đạt threshold nhưng chưa nên advance stage
```

### Case 5: checkpoint_path là step_N thay vì final

```python
# Phase 1: scan thêm các checkpoints khác trong cùng seed
# Highlight nếu final/ tốt hơn hoặc kém hơn step_N đang review
# Gợi ý dùng final/ hoặc step_N phù hợp nhất
```

---

## Quy tắc của agent

1. **Không bao giờ ra quyết định khi chưa có eval data** — luôn sinh eval commands trước.
2. **Không chạy eval tự động** — chỉ sinh commands, để user kiểm soát việc chạy.
3. **Multi-seed awareness** — luôn review tất cả seeds có sẵn, không chỉ seed được chỉ định.
4. **Curriculum state luôn được report** — dù nominal metrics tốt, nếu curriculum chưa xong thì không advance stage.
5. **Config advice chỉ khi cần** — không spam config suggestions nếu decision là CONTINUE hoặc ADVANCE_STAGE.
6. **Commands phải copy-paste ready** — không có placeholder như `<checkpoint_dir>` trong output cuối.
7. **Preserve checkpoint compatibility** — không bao giờ suggest thay đổi phá vỡ checkpoint format.
