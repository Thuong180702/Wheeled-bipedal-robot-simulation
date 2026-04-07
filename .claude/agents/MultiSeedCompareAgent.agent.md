---
name: MultiSeedCompareAgent
description: >
  Agent so sánh kết quả training của nhiều seeds (42, 113, 999), xác định seed
  nào tốt nhất/tệ nhất/outlier, tổng hợp mean ± std sẵn sàng cho paper, và
  đưa ra recommendation: seed nào dùng cho paper, seed nào cần train thêm,
  seed nào nên bỏ và chạy lại. Orchestrates: checkpoint-manager → eval-analyzer
  → (tự aggregate) → training-decision. Output gồm comparison table, paper
  readiness verdict, và commands cụ thể.
skills_used:
  - checkpoint-manager
  - eval-analyzer
  - training-decision
license: Project-internal agent
---

# MultiSeedCompareAgent

## Mục đích

Trả lời câu hỏi: **"3 seeds của tôi đang ở đâu, có đủ để báo cáo paper chưa, và seed nào có vấn đề?"**

Agent này được thiết kế để chạy sau khi ít nhất 2 seeds đã train xong, trước khi
viết kết quả vào paper. Output trực tiếp feed vào `paper-updater` skill.

---

## Trigger patterns

- `"So sánh 3 seeds cho tôi"`
- `"Seed nào tốt nhất?"`
- `"Paper readiness check"`
- `"Có đủ 3 seeds để báo cáo chưa?"`
- `"Seed 113 có outlier không?"`
- `"Tổng hợp mean ± std để điền vào paper"`

---

## Inputs

```python
@dataclass
class MultiSeedCompareInput:
    stage: str = "balance"
    outputs_dir: str = "outputs"
    seeds: list[int] = None          # None = tự tìm, thường [42, 113, 999]
    paper_seeds: list[int] = None    # None = [42, 113, 999]
    scenarios: list[str] = None      # None = tất cả scenarios có trong eval_results
    run_eval_if_missing: bool = True
    outlier_std_threshold: float = 2.0  # seed bị flagged là outlier nếu > 2σ từ mean
```

---

## Workflow

```
┌──────────────────────────────────────────────────────────────────┐
│                  MultiSeedCompareAgent                            │
│                                                                   │
│  Phase 1: COLLECT                                                 │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │ checkpoint-manager + eval-analyzer (per seed)            │    │
│  │  • scan_checkpoints() → best checkpoint mỗi seed         │    │
│  │  • find_eval_files()  → kiểm tra eval status            │    │
│  │  • load_eval_balance() → ScenarioMetrics per seed        │    │
│  │  • enrich_with_jsonl() → training curve per seed         │    │
│  └─────────────────────┬────────────────────────────────────┘    │
│                         │                                         │
│  Phase 2: EVAL GAP CHECK                                          │
│  ┌─────────────────────▼────────────────────────────────────┐    │
│  │  Seeds thiếu eval → sinh commands, báo user              │    │
│  │  Có thể tiếp tục với seeds đã có eval (partial compare)  │    │
│  └─────────────────────┬────────────────────────────────────┘    │
│                         │                                         │
│  Phase 3: AGGREGATE                                               │
│  ┌─────────────────────▼────────────────────────────────────┐    │
│  │  Per scenario + per metric:                              │    │
│  │  • mean, std, min, max across seeds                      │    │
│  │  • detect outlier seeds                                  │    │
│  │  • compute seed-vs-seed delta (regression check)         │    │
│  └─────────────────────┬────────────────────────────────────┘    │
│                         │                                         │
│  Phase 4: PAPER READINESS                                         │
│  ┌─────────────────────▼────────────────────────────────────┐    │
│  │  Check từng paper requirement:                           │    │
│  │  • n_seeds == 3 (paper_seeds đều có eval)                │    │
│  │  • eval_per_step >= success_value (7.0)                  │    │
│  │  • curriculum completed (min_height <= 0.41)             │    │
│  │  • no outlier seeds                                      │    │
│  │  • std không quá cao (noisy results)                     │    │
│  └─────────────────────┬────────────────────────────────────┘    │
│                         │                                         │
│  Phase 5: OUTPUT                                                  │
│  ┌─────────────────────▼────────────────────────────────────┐    │
│  │  • Per-seed status table                                 │    │
│  │  • Cross-seed comparison table (mean ± std per scenario) │    │
│  │  • Outlier report                                        │    │
│  │  • Paper readiness verdict + missing items               │    │
│  │  • Commands: eval missing, export latex, update paper    │    │
│  └──────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: COLLECT — thu thập data từng seed

```python
import json, math, pickle
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

PAPER_SEEDS     = [42, 113, 999]
SUCCESS_VALUES  = {"balance": 7.0, "balance_robust": 6.0, "stand_up": 5.0}
PAPER_SCENARIOS = ["nominal", "push_recovery", "friction_low", "friction_high", "full_range"]

@dataclass
class SeedData:
    seed: int
    checkpoint_path: str
    checkpoint_step: int
    checkpoint_label: str
    eval_per_step: float | None        # từ pkl (best_eval_per_step)
    curriculum_min_height: float | None
    has_eval_balance: bool
    has_validation: bool
    scenarios: dict[str, dict]         # scenario → metrics dict
    training_trend: str                # improving|plateau|declining|unknown
    latest_eval_reward: float | None
    latest_fall_rate: float | None
    total_steps: int

def collect_seed_data(stage: str, outputs_dir: str,
                       seeds: list[int]) -> dict[int, SeedData]:
    """Thu thập SeedData cho mỗi seed."""
    all_ckpts = scan_checkpoints(outputs_dir=outputs_dir, stage=stage, seeds=seeds)
    timeline  = enrich_with_jsonl(all_ckpts, outputs_dir=outputs_dir)
    best      = pick_best_per_seed(all_ckpts, stage)
    result    = {}

    for seed, ckpt in best.items():
        found = find_eval_files(ckpt.path)

        # Load scenario metrics nếu có
        scenarios = {}
        if "eval_balance" in found:
            eb = load_eval_balance(found["eval_balance"])
            for r in eb.get("results", []):
                scenarios[r["scenario"]] = r

        # Training trend từ JSONL
        jsonl_metrics = get_metrics_at_step(timeline, stage, seed, ckpt.step)
        trend = _estimate_trend_from_timeline(timeline, stage, seed)

        result[seed] = SeedData(
            seed=seed,
            checkpoint_path=ckpt.path,
            checkpoint_step=ckpt.step,
            checkpoint_label=ckpt.label,
            eval_per_step=ckpt.eval_per_step,
            curriculum_min_height=ckpt.curriculum_min_height,
            has_eval_balance=bool(scenarios),
            has_validation="validation" in found,
            scenarios=scenarios,
            training_trend=trend,
            latest_eval_reward=jsonl_metrics.get("eval_reward_mean"),
            latest_fall_rate=jsonl_metrics.get("eval_fall_rate"),
            total_steps=ckpt.step,
        )

    return result


def _estimate_trend_from_timeline(timeline, stage, seed) -> str:
    records = timeline.get((stage, seed), [])
    pts = [(r["step"], r["eval/reward_mean"])
           for r in records if "eval/reward_mean" in r]
    if len(pts) < 4:
        return "unknown"
    _, rewards = zip(*pts[-8:])
    delta = rewards[-1] - rewards[0]
    if   delta >  0.5: return "improving"
    elif delta < -0.5: return "declining"
    else:               return "plateau"
```

---

## Phase 2: EVAL GAP CHECK

```python
def eval_gap_check(seed_data: dict[int, SeedData],
                    paper_seeds: list[int],
                    stage: str) -> dict:
    """Kiểm tra seeds nào còn thiếu eval."""
    missing_eval   = [s for s in paper_seeds if s not in seed_data or not seed_data[s].has_eval_balance]
    missing_seeds  = [s for s in paper_seeds if s not in seed_data]
    partial_seeds  = [s for s in paper_seeds if s in seed_data and seed_data[s].has_eval_balance]
    eval_commands  = []

    for seed in missing_eval:
        if seed in seed_data:
            ckpt_path = seed_data[seed].checkpoint_path
        else:
            # Dự đoán đường dẫn
            ckpt_path = f"outputs/{stage}/rl/seed{seed}/checkpoints/final"

        eval_commands.append(
            f"# Seed {seed} — chạy eval_balance trước\n"
            f"python scripts/eval_balance.py \\\n"
            f"    --checkpoint {ckpt_path} \\\n"
            f"    --scenarios nominal push_recovery friction_low friction_high full_range \\\n"
            f"    --num-episodes 50 --num-steps 2000 --seeds 0 42 123"
        )

    return {
        "missing_eval":  missing_eval,
        "missing_seeds": missing_seeds,
        "partial_seeds": partial_seeds,
        "eval_commands": eval_commands,
        "can_compare":   len(partial_seeds) >= 2,
        "full_compare":  len(partial_seeds) == len(paper_seeds),
    }
```

---

## Phase 3: AGGREGATE — tính mean ± std và phát hiện outlier

```python
@dataclass
class AggregatedMetric:
    scenario: str
    metric: str
    values: dict[int, float]       # seed → value
    mean: float
    std: float
    min_val: float
    max_val: float
    outlier_seeds: list[int]       # seeds > threshold * std từ mean
    cv: float                      # coefficient of variation = std/mean (nếu mean != 0)

def aggregate_across_seeds(seed_data: dict[int, SeedData],
                             scenarios: list[str],
                             outlier_threshold: float = 2.0) -> dict:
    """
    Tính mean ± std cho từng (scenario, metric) trên các seeds có eval.
    """
    METRICS_TO_AGGREGATE = [
        ("fall_rate",              "lower"),
        ("survival_rate",          "higher"),
        ("survival_time_mean_s",   "higher"),
        ("pitch_rms_deg",          "lower"),
        ("roll_rms_deg",           "lower"),
        ("height_rmse_m",          "lower"),
        ("torque_rms_nm",          "lower"),
        ("max_recoverable_push_n", "higher"),
        ("wheel_speed_rms_rads",   "lower"),
        ("xy_drift_max_m",         "lower"),
    ]

    seeds_with_eval = [s for s, d in seed_data.items() if d.has_eval_balance]
    aggregated = {}   # (scenario, metric) → AggregatedMetric

    for scen in scenarios:
        for metric, direction in METRICS_TO_AGGREGATE:
            vals = {}
            for seed in seeds_with_eval:
                scen_data = seed_data[seed].scenarios.get(scen, {})
                v = scen_data.get(metric, float("nan"))
                if not (math.isnan(v) or math.isinf(v)):
                    vals[seed] = v

            if not vals:
                continue

            arr  = np.array(list(vals.values()))
            mean = float(np.mean(arr))
            std  = float(np.std(arr))
            cv   = abs(std / mean) if abs(mean) > 1e-9 else float("nan")

            # Outlier: > threshold * std từ mean
            outliers = [
                seed for seed, v in vals.items()
                if abs(v - mean) > outlier_threshold * std and std > 1e-9
            ]

            aggregated[(scen, metric)] = AggregatedMetric(
                scenario=scen,
                metric=metric,
                values=vals,
                mean=mean,
                std=std,
                min_val=float(np.min(arr)),
                max_val=float(np.max(arr)),
                outlier_seeds=outliers,
                cv=cv,
            )

    return {"by_scenario_metric": aggregated, "seeds_with_eval": seeds_with_eval}


def find_outlier_seeds(aggregated: dict,
                        outlier_threshold: float = 2.0) -> dict[int, list[str]]:
    """
    Tổng hợp: seed nào bị outlier trên bao nhiêu (scenario, metric)?
    Trả về {seed: ["nominal.fall_rate", "friction_low.height_rmse_m", ...]}
    """
    outlier_map = {}
    for (scen, metric), agg in aggregated["by_scenario_metric"].items():
        for seed in agg.outlier_seeds:
            key = f"{scen}.{metric}"
            outlier_map.setdefault(seed, []).append(key)
    return outlier_map
```

---

## Phase 4: PAPER READINESS CHECK

```python
PAPER_REQUIREMENTS = {
    "n_seeds":              3,
    "min_eval_per_step":    None,   # lấy từ SUCCESS_VALUES[stage]
    "curriculum_complete":  0.41,   # min_height <= 0.41
    "max_nominal_fall_rate": 0.15,
    "max_cv_threshold":     0.30,   # CV > 30% → results quá noisy
    "max_outlier_metrics":  2,      # seed bị flag là outlier trên > 2 metrics → suspect
}

@dataclass
class PaperReadinessResult:
    verdict: str                  # READY | NOT_READY | PARTIAL
    n_seeds_ready: int
    missing_items: list[str]      # danh sách điều kiện chưa đạt
    warnings: list[str]           # soft warnings (không block nhưng cần chú ý)
    seed_status: dict[int, str]   # seed → "READY"|"NEEDS_EVAL"|"NEEDS_TRAIN"|"OUTLIER"
    mean_std_summary: dict        # scenario → {metric: "mean ± std"} dùng cho paper

def check_paper_readiness(seed_data: dict[int, SeedData],
                           aggregated: dict,
                           paper_seeds: list[int],
                           stage: str) -> PaperReadinessResult:
    success_value = SUCCESS_VALUES.get(stage, 7.0)
    missing = []
    warnings = []
    seed_status = {}
    outlier_map = find_outlier_seeds(aggregated)

    # ── Check từng paper seed ─────────────────────────────────────────────────
    for seed in paper_seeds:
        if seed not in seed_data:
            seed_status[seed] = "NEEDS_TRAIN"
            missing.append(f"seed{seed}: chưa có checkpoint")
            continue

        sd = seed_data[seed]

        if not sd.has_eval_balance:
            seed_status[seed] = "NEEDS_EVAL"
            missing.append(f"seed{seed}: chưa có eval_results (chạy eval_balance.py)")
            continue

        issues = []

        # eval_per_step
        ep = sd.eval_per_step or 0.0
        if ep < success_value:
            issues.append(f"eval_per_step={ep:.2f} < {success_value}")

        # curriculum
        ch = sd.curriculum_min_height or 1.0
        if ch > PAPER_REQUIREMENTS["curriculum_complete"]:
            issues.append(f"curriculum chưa xong (min_height={ch:.2f})")

        # nominal fall rate (từ eval data)
        nom = sd.scenarios.get("nominal", {})
        fr  = nom.get("fall_rate", float("nan"))
        if not math.isnan(fr) and fr > PAPER_REQUIREMENTS["max_nominal_fall_rate"]:
            issues.append(f"nominal fall_rate={fr:.1%} > 15%")

        # outlier check
        n_outlier = len(outlier_map.get(seed, []))
        if n_outlier > PAPER_REQUIREMENTS["max_outlier_metrics"]:
            issues.append(f"outlier trên {n_outlier} metrics")

        if issues:
            seed_status[seed] = "NEEDS_TRAIN" if any("curriculum" in i or "eval_per_step" in i for i in issues) else "WARN"
            missing.extend([f"seed{seed}: {i}" for i in issues])
        else:
            seed_status[seed] = "READY"

    # ── CV check trên aggregated metrics ─────────────────────────────────────
    high_cv = []
    for (scen, metric), agg in aggregated["by_scenario_metric"].items():
        if not math.isnan(agg.cv) and agg.cv > PAPER_REQUIREMENTS["max_cv_threshold"]:
            high_cv.append(f"{scen}.{metric}: CV={agg.cv:.1%}")
    if high_cv:
        warnings.append(f"High variance metrics (CV > 30%): {', '.join(high_cv[:4])}")

    # ── Tính mean ± std summary cho paper ────────────────────────────────────
    mean_std_summary = {}
    for scen in PAPER_SCENARIOS:
        mean_std_summary[scen] = {}
        for metric in ["survival_rate", "height_rmse_m", "pitch_rms_deg",
                        "torque_rms_nm", "max_recoverable_push_n"]:
            key = (scen, metric)
            if key in aggregated["by_scenario_metric"]:
                agg = aggregated["by_scenario_metric"][key]
                n = len(aggregated["seeds_with_eval"])
                if n > 1:
                    mean_std_summary[scen][metric] = f"{agg.mean:.3f} ± {agg.std:.3f}"
                else:
                    mean_std_summary[scen][metric] = f"{agg.mean:.3f} (1 seed)"

    # ── Verdict ───────────────────────────────────────────────────────────────
    n_ready = sum(1 for s in paper_seeds if seed_status.get(s) == "READY")
    if n_ready == len(paper_seeds) and not missing:
        verdict = "READY"
    elif n_ready >= 2:
        verdict = "PARTIAL"
    else:
        verdict = "NOT_READY"

    return PaperReadinessResult(
        verdict=verdict,
        n_seeds_ready=n_ready,
        missing_items=missing,
        warnings=warnings,
        seed_status=seed_status,
        mean_std_summary=mean_std_summary,
    )
```

---

## Phase 5: OUTPUT

```python
def format_output(seed_data: dict[int, SeedData],
                   aggregated: dict,
                   paper_readiness: PaperReadinessResult,
                   eval_gap: dict,
                   paper_seeds: list[int],
                   stage: str) -> str:
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    verdict_icon = {"READY": "✅", "PARTIAL": "⚠️ ", "NOT_READY": "❌"}.get(
        paper_readiness.verdict, "❓"
    )
    lines += [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  MULTI-SEED COMPARISON — stage: {stage:<27}║",
        f"║  Paper readiness: {verdict_icon} {paper_readiness.verdict:<39}║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]

    # ── Section 1: Per-seed status ────────────────────────────────────────────
    lines += ["📦 PER-SEED STATUS", "─" * 66]
    status_icons = {
        "READY":       "✅",
        "NEEDS_EVAL":  "📊",
        "NEEDS_TRAIN": "🏋️ ",
        "WARN":        "⚠️ ",
        "OUTLIER":     "🔴",
    }
    lines.append(f"  {'SEED':>5} │ {'STATUS':<12} │ {'STEP':>12} │ {'EVAL/STEP':>9} │ {'CURR_H':>7} │ TREND")
    lines.append("  " + "─" * 64)
    for seed in paper_seeds:
        status = paper_readiness.seed_status.get(seed, "UNKNOWN")
        icon   = status_icons.get(status, "❓")
        sd     = seed_data.get(seed)
        if sd:
            ep_str  = f"{sd.eval_per_step:.2f}" if sd.eval_per_step else "—"
            ch_str  = f"{sd.curriculum_min_height:.2f}" if sd.curriculum_min_height else "—"
            trend_icon = {"improving":"📈","plateau":"➡️ ","declining":"📉","unknown":"❓"}.get(sd.training_trend,"❓")
            lines.append(
                f"  {seed:>5} │ {icon} {status:<10} │ "
                f"{sd.total_steps:>12,} │ {ep_str:>9} │ {ch_str:>7} │ "
                f"{trend_icon} {sd.training_trend}"
            )
        else:
            lines.append(f"  {seed:>5} │ {icon} {status:<10} │ {'—':>12} │ {'—':>9} │ {'—':>7} │ —")
    lines.append("")

    # ── Section 2: Cross-seed comparison table ─────────────────────────────────
    seeds_with_eval = aggregated.get("seeds_with_eval", [])
    if len(seeds_with_eval) >= 2:
        lines += ["📊 CROSS-SEED METRICS (mean ± std)", "─" * 66]
        header_seeds = "  ".join(f"seed{s}" for s in sorted(seeds_with_eval))
        lines.append(f"  {'SCENARIO':<18} │ {'METRIC':<22} │ {header_seeds:<30} │ MEAN ± STD")
        lines.append("  " + "─" * 80)

        display_metrics = [
            ("fall_rate",              "Fall rate",    lambda v: f"{v:.1%}"),
            ("survival_time_mean_s",   "Survival (s)", lambda v: f"{v:.1f}"),
            ("pitch_rms_deg",          "Pitch RMS°",   lambda v: f"{v:.2f}"),
            ("height_rmse_m",          "Height RMSE",  lambda v: f"{v:.3f}"),
            ("max_recoverable_push_n", "Max push (N)", lambda v: f"{v:.0f}"),
        ]
        prev_scen = None
        for scen in PAPER_SCENARIOS:
            for metric, label, fmt_fn in display_metrics:
                key = (scen, metric)
                if key not in aggregated["by_scenario_metric"]:
                    continue
                agg = aggregated["by_scenario_metric"][key]

                scen_display = scen if scen != prev_scen else ""
                prev_scen = scen

                per_seed_vals = "  ".join(
                    fmt_fn(agg.values.get(s, float("nan")))
                    if s in agg.values else "  —"
                    for s in sorted(seeds_with_eval)
                )
                outlier_flag = "🔴" if agg.outlier_seeds else "  "
                lines.append(
                    f"  {scen_display:<18} │ {label:<22} │ "
                    f"{per_seed_vals:<30} │ "
                    f"{outlier_flag} {agg.mean:.3f} ± {agg.std:.3f}"
                )
        lines.append("")

    # ── Section 3: Outlier report ─────────────────────────────────────────────
    outlier_map = find_outlier_seeds(aggregated)
    if outlier_map:
        lines += ["🔴 OUTLIER REPORT", "─" * 66]
        for seed, metrics in sorted(outlier_map.items()):
            lines.append(f"  seed{seed}: outlier trên {len(metrics)} metric(s)")
            for m in metrics[:5]:
                lines.append(f"    • {m}")
            if len(metrics) > 5:
                lines.append(f"    • ... +{len(metrics)-5} more")
        lines.append("")

    # ── Section 4: Paper readiness detail ──────────────────────────────────────
    lines += [f"{'✅' if paper_readiness.verdict == 'READY' else '❌'} PAPER READINESS DETAIL", "─" * 66]
    lines.append(f"  Seeds ready: {paper_readiness.n_seeds_ready}/{len(paper_seeds)}")

    if paper_readiness.missing_items:
        lines.append("  Missing / failing:")
        for item in paper_readiness.missing_items:
            lines.append(f"    ❌ {item}")

    if paper_readiness.warnings:
        lines.append("  Warnings (non-blocking):")
        for w in paper_readiness.warnings:
            lines.append(f"    ⚠️  {w}")

    if paper_readiness.verdict == "READY":
        lines.append("  🎉 Tất cả seeds sẵn sàng — có thể điền số vào paper!")
    lines.append("")

    # ── Section 5: Mean ± std summary (paper-ready) ────────────────────────────
    if paper_readiness.verdict in ("READY", "PARTIAL") and paper_readiness.mean_std_summary:
        n = len(seeds_with_eval)
        lines += [f"📋 MEAN ± STD SUMMARY (n={n} seeds, dùng cho paper)", "─" * 66]
        lines.append(f"  {'SCENARIO':<18} │ {'SURV%':>14} │ {'H_RMSE':>14} │ {'PITCH°':>14} │ {'TORQUE':>14} │ {'MAX_PUSH':>14}")
        lines.append("  " + "─" * 90)
        for scen in PAPER_SCENARIOS:
            ms = paper_readiness.mean_std_summary.get(scen, {})
            if not ms:
                continue
            surv  = ms.get("survival_rate", "—")
            herr  = ms.get("height_rmse_m", "—")
            pitch = ms.get("pitch_rms_deg", "—")
            torq  = ms.get("torque_rms_nm", "—")
            push  = ms.get("max_recoverable_push_n", "—")
            lines.append(f"  {scen:<18} │ {surv:>14} │ {herr:>14} │ {pitch:>14} │ {torq:>14} │ {push:>14}")
        lines.append("")

    # ── Section 6: Commands ─────────────────────────────────────────────────────
    lines += ["⚡ COMMANDS", "─" * 66]

    # Missing eval commands
    if eval_gap["eval_commands"]:
        lines.append("  1️⃣  Chạy eval cho seeds còn thiếu:")
        for cmd in eval_gap["eval_commands"]:
            for l in cmd.splitlines():
                lines.append(f"    {l}")
            lines.append("")

    # Multi-seed eval command (khi tất cả đã train xong)
    all_finals = " \\\n                 ".join(
        f"outputs/{stage}/rl/seed{s}/checkpoints/final"
        for s in paper_seeds
    )
    lines += [
        "  2️⃣  Eval tổng hợp 3 seeds cho paper (sau khi tất cả train xong):",
        f"    python scripts/eval_balance.py \\",
        f"        --checkpoint {all_finals} \\",
        f"        --num-episodes 50 --num-steps 2000 --seeds 0 42 123 \\",
        f"        --output-dir outputs/{stage}/rl/paper_eval",
        "",
    ]

    # Export to LaTeX
    if paper_readiness.verdict in ("READY", "PARTIAL"):
        lines += [
            "  3️⃣  Export kết quả ra LaTeX:",
            f"    python scripts/export_results.py latex \\",
            f"        outputs/{stage}/rl/paper_eval/eval_results.json \\",
            f"        --output outputs/tables/{stage}_eval.tex",
            "",
            "  4️⃣  Sau đó dùng paper-updater skill để điền vào paper/main.tex",
            "",
        ]

    lines += ["─" * 66]
    return "\n".join(lines)
```

---

## Entrypoint

```python
def run(inp: MultiSeedCompareInput) -> str:
    seeds       = inp.seeds or PAPER_SEEDS
    paper_seeds = inp.paper_seeds or PAPER_SEEDS
    scenarios   = inp.scenarios or PAPER_SCENARIOS

    print(f"📊 Collecting data for {len(seeds)} seeds...")
    seed_data = collect_seed_data(inp.stage, inp.outputs_dir, seeds)

    print("📋 Checking eval gaps...")
    eval_gap = eval_gap_check(seed_data, paper_seeds, inp.stage)

    seeds_with_eval = [s for s in paper_seeds
                       if s in seed_data and seed_data[s].has_eval_balance]

    if not seeds_with_eval:
        # Không có gì để so sánh
        lines = [
            "❌ Không có seed nào có eval data.",
            "Chạy các lệnh sau trước:",
            "",
        ]
        for cmd in eval_gap["eval_commands"]:
            lines.append(cmd)
        return "\n".join(lines)

    print(f"🔢 Aggregating across {len(seeds_with_eval)} seeds...")
    aggregated = aggregate_across_seeds(seed_data, scenarios)

    print("✅ Checking paper readiness...")
    readiness = check_paper_readiness(seed_data, aggregated, paper_seeds, inp.stage)

    return format_output(seed_data, aggregated, readiness, eval_gap, paper_seeds, inp.stage)
```

---

## Quy tắc của agent

1. **Không merge kết quả seeds chưa train xong** — seed còn đang training không được đưa vào mean ± std.
2. **Outlier = seed > 2σ từ mean trên > 2 metrics** — flagged, không tự loại bỏ; để user quyết định.
3. **Partial compare vẫn có giá trị** — nếu chỉ có 2/3 seeds, vẫn so sánh và báo cáo mean của 2 seeds đó với note "(2/3 seeds)".
4. **Curriculum state bắt buộc check** — seed có eval tốt nhưng curriculum chưa xong (min_height > 0.41) thì vẫn NEEDS_TRAIN.
5. **Mean ± std chỉ sinh khi >= 2 seeds** — 1 seed thì ghi số đơn với note "(1 seed, preliminary)".
6. **Commands luôn copy-paste ready** — paths thực tế, không có placeholder.