---
name: eval-analyzer
description: >
  Dùng skill này để đọc, parse và tóm tắt kết quả validation của wheeled biped project.
  Kích hoạt khi user nhắc đến: checkpoint, eval results, training trend, fall rate,
  exploit pattern, wheel spin, friction score, push recovery, WARN flag, metrics,
  hay bất cứ lúc nào cần hiểu policy đang tốt/xấu thế nào.
  Biết xử lý 4 loại file: validation_report.json (validate_checkpoint.py — PRIMARY),
  *_metrics.jsonl (training log — PRIMARY), eval_results_*.json (evaluate.py),
  eval_results.json (eval_balance.py — paper metrics). Tự tìm file, tìm
  validation_report.json trước. Không ra quyết định train — đó là việc của
  training-decision skill.
license: Project-internal skill
---

# Eval Analyzer Skill

## Tổng quan nhanh — 4 loại file, vai trò rõ ràng

| File                     | Script nguồn             | Hàm đọc                    | Vai trò trong quyết định            |
| ------------------------ | ------------------------ | -------------------------- | ----------------------------------- |
| `validation_report.json` | `validate_checkpoint.py` | `load_validation_report()` | **PRIMARY** — exploit + benchmark   |
| `*_metrics.jsonl`        | Training logger          | `load_jsonl_log()`         | **PRIMARY** — reward trend          |
| `eval_results_*.json`    | `evaluate.py`            | `load_evaluate()`          | Secondary — eval_per_step chính xác |
| `eval_results.json`      | `eval_balance.py`        | `load_eval_balance()`      | Secondary — paper metrics           |

> **Quy tắc ưu tiên**: `validation_report.json` + JSONL → đủ để ra quyết định train.
> `eval_results.json` (eval_balance) KHÔNG bắt buộc — chỉ cần khi muốn multi-scenario hoặc paper metrics.

> ⚠️ **Key quan trọng**: `validation_report.json` dùng key `"standing_quality"` (KHÔNG phải `"quality_signals"`).

---

## Bước 0 — Tìm files tự động

```python
from pathlib import Path

def find_eval_files(checkpoint_dir: str) -> dict:
    """
    Tự động tìm tất cả eval files. Gọi đầu tiên, trước khi làm gì khác.
    Tìm validation_report.json TRƯỚC — đây là primary source.
    """
    base = Path(checkpoint_dir)
    found = {}

    # 1. validation_report.json — PRIMARY, tìm trước
    p = base / "validation_report.json"
    if p.exists():
        found["validation"] = str(p)

    # 2. JSONL training log — PRIMARY, ở seed dir (parent của checkpoints/)
    seed_dir = base.parent.parent  # outputs/<stage>/rl/seed<N>/
    for pattern in ["*_metrics.jsonl", "metrics.jsonl"]:
        matches = list(seed_dir.glob(pattern))
        if matches:
            found["jsonl"] = str(sorted(matches)[-1])  # lấy mới nhất
            break

    # 3. eval_balance.py output — Secondary (paper metrics)
    for p in [
        base / "eval_results.json",
        base.parent.parent / "paper_eval" / "eval_results.json",
    ]:
        if p.exists():
            found["eval_balance"] = str(p)
            break

    # 4. evaluate.py outputs — Secondary (eval_per_step trustworthy)
    for mode in ["nominal", "push_recovery", "domain_randomized", "command_tracking"]:
        p = base / f"eval_results_{mode}.json"
        if p.exists():
            found[f"evaluate_{mode}"] = str(p)

    return found
```

---

## Bước 1 — Đọc từng loại file

### validation_report.json — PRIMARY

```python
import json, math
from pathlib import Path

def load_validation_report(path: str) -> dict:
    """
    ⚠️ KEY QUAN TRỌNG: dùng "standing_quality", KHÔNG phải "quality_signals".

    Cấu trúc thực tế của file:
    {
      "standing_quality": {
        "wheel_spin_mean_rads": 2.3,
        "height_std_m": 0.01,
        "xy_drift_max_m": 0.08,
        "roll_mean_abs_deg": 1.2,
        "pitch_mean_abs_deg": 0.9,
        "ctrl_jitter_mean_nm": 0.3,
        "leg_asymmetry_mean_rad": 0.05,
        "ang_vel_rms_rads": 0.2,
        "flags": ["HIGH wheel spin (4.1 rad/s) — ..."],
        "num_suspicious": 1
      },
      "benchmark": {
        "reward_mean": 7250, "fall_rate": 0.08, "success_rate": 0.92,
        "episode_length_mean": 980, ...
      },
      "checkpoint": "outputs/balance/rl/seed42/checkpoints/final",
      "stage": "balance",
      "height_cmd_m": 0.65,
    }
    """
    with open(path) as f:
        data = json.load(f)

    quality = data.get("standing_quality", {})
    flags   = quality.get("flags", [])

    # Thresholds từ wheeled_biped/eval/standing_quality.py
    VALIDATION_THRESHOLDS = {
        "wheel_spin_mean_rads":   3.0,   # wheel-momentum exploit
        "height_std_m":           0.05,  # vertical oscillation
        "xy_drift_max_m":         0.3,   # slow drift exploit
        "roll_mean_abs_deg":      5.0,   # chronic lean
        "pitch_mean_abs_deg":     5.0,   # chronic lean
        "ctrl_jitter_mean_nm":    1.5,   # chattering actuation
        "leg_asymmetry_mean_rad": 0.1,   # asymmetric crouching
        "ang_vel_rms_rads":       0.5,   # torso wobble
    }

    # warn_signals: chỉ chứa signal đã vượt threshold
    warn_signals = {}
    for signal, threshold in VALIDATION_THRESHOLDS.items():
        val = quality.get(signal)
        if val is not None and not math.isnan(float(val)) and float(val) > threshold:
            warn_signals[signal] = float(val)

    return {
        "benchmark":       data.get("benchmark", {}),
        "quality_signals": quality,       # raw dict với tất cả float values
        "flags":           flags,         # list human-readable warning strings
        "warn_signals":    warn_signals,  # {signal: value} chỉ khi vượt threshold
        "num_suspicious":  quality.get("num_suspicious", 0),
    }
```

### eval_results.json (eval_balance.py) — Secondary

```python
def load_eval_balance(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Structure: {controller, checkpoints, scenarios, results: [ScenarioMetrics...]}
    return data

def parse_scenario_metrics(results: list[dict]) -> list[dict]:
    parsed = []
    for r in results:
        parsed.append({
            "checkpoint":             Path(r.get("checkpoint", "")).name,
            "scenario":               r["scenario"],
            "fall_rate":              r.get("fall_rate",              float("nan")),
            "survival_rate":          r.get("survival_rate",          float("nan")),
            "survival_time_mean_s":   r.get("survival_time_mean_s",   float("nan")),
            "pitch_rms_deg":          r.get("pitch_rms_deg",          float("nan")),
            "roll_rms_deg":           r.get("roll_rms_deg",           float("nan")),
            "height_rmse_m":          r.get("height_rmse_m",          float("nan")),
            "wheel_speed_rms_rads":   r.get("wheel_speed_rms_rads",   float("nan")),
            "xy_drift_max_m":         r.get("xy_drift_max_m",         float("nan")),
            "torque_rms_nm":          r.get("torque_rms_nm",          float("nan")),
            "recovery_time_s":        r.get("recovery_time_s",        float("nan")),
            "max_recoverable_push_n": r.get("max_recoverable_push_n", float("nan")),
        })
    return parsed
```

### eval*results*\*.json (evaluate.py) — Secondary

```python
def load_evaluate(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    ep_len = data.get("episode_length_mean", 1000) or 1000
    return {
        "checkpoint":          data.get("checkpoint", ""),
        "mode":                data.get("mode", ""),
        "reward_mean":         data.get("reward_mean",    float("nan")),
        "reward_std":          data.get("reward_std",     float("nan")),
        "fall_rate":           data.get("fall_rate",      float("nan")),
        "success_rate":        data.get("success_rate",   float("nan")),
        "episode_length_mean": ep_len,
        # eval_per_step để so với success_value trong curriculum.yaml
        "eval_per_step":       data.get("reward_mean", 0) / ep_len,
        "mode_metrics":        data.get("mode_metrics", {}),
    }
```

### \*\_metrics.jsonl (training log) — PRIMARY

```python
def load_jsonl_log(path: str) -> list[dict]:
    """Đọc training log JSONL. Keys dùng slash: 'eval/reward_mean', 'eval/fall_rate'."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

def extract_training_trend(records: list[dict]) -> dict:
    """
    Tính trend từ JSONL log.
    eval/reward_mean là EPISODE SUM — dùng % change, không dùng absolute delta.
    """
    eval_rewards   = [(r["step"], r["eval/reward_mean"])
                      for r in records if "eval/reward_mean" in r]
    fall_rates     = [(r["step"], r["eval/fall_rate"])
                      for r in records if "eval/fall_rate" in r]
    curriculum_pts = [(r["step"], r["curriculum/eval_per_step"])
                      for r in records if "curriculum/eval_per_step" in r]

    if not eval_rewards:
        return {"status": "no_eval_data", "reward_trend": "unknown"}

    steps, rewards = zip(*eval_rewards)
    recent_n = min(6, len(rewards))
    recent   = rewards[-recent_n:]

    base      = max(abs(recent[0]), 1.0)
    delta_pct = (recent[-1] - recent[0]) / base

    if delta_pct > 0.03:
        trend = "improving"
    elif delta_pct < -0.03:
        trend = "declining"
    else:
        trend = "plateau"

    return {
        "total_steps":        max(steps),
        "latest_eval_reward": rewards[-1],
        "latest_fall_rate":   fall_rates[-1][1] if fall_rates else float("nan"),
        "reward_trend":       trend,
        "delta_pct":          delta_pct * 100,
        "recent_rewards":     list(recent),
        "curriculum_pts":     curriculum_pts[-10:],
        "num_eval_points":    len(eval_rewards),
    }
```

---

## Bước 2 — Flags WARN/CRITICAL

```python
BALANCE_THRESHOLDS = {
    "fall_rate":               {"warn": 0.20, "critical": 0.40, "direction": "lower_is_better"},
    "survival_time_mean_s":    {"warn": 15.0, "critical": 8.0,  "direction": "higher_is_better"},
    "pitch_rms_deg":           {"warn": 5.0,  "critical": 10.0, "direction": "lower_is_better"},
    "roll_rms_deg":            {"warn": 3.0,  "critical": 6.0,  "direction": "lower_is_better"},
    "height_rmse_m":           {"warn": 0.03, "critical": 0.06, "direction": "lower_is_better"},
    "wheel_speed_rms_rads":    {"warn": 3.0,  "critical": 6.0,  "direction": "lower_is_better"},
    "xy_drift_max_m":          {"warn": 0.3,  "critical": 0.8,  "direction": "lower_is_better"},
    "torque_rms_nm":           {"warn": 8.0,  "critical": 12.0, "direction": "lower_is_better"},
    "max_recoverable_push_n":  {"warn": 40.0, "critical": 20.0, "direction": "higher_is_better"},
    "recovery_time_s":         {"warn": 3.0,  "critical": 6.0,  "direction": "lower_is_better"},
}

def flag_metric(value: float, metric_name: str) -> str:
    if math.isnan(value) or math.isinf(value):
        return "N/A"
    thresh = BALANCE_THRESHOLDS.get(metric_name)
    if not thresh:
        return "OK"
    if thresh["direction"] == "lower_is_better":
        if value >= thresh["critical"]: return "CRITICAL"
        if value >= thresh["warn"]:     return "WARN"
    else:
        if value <= thresh["critical"]: return "CRITICAL"
        if value <= thresh["warn"]:     return "WARN"
    return "OK"
```

---

## Bước 3 — So sánh nhiều checkpoints

```python
def compare_checkpoints(reports: list[dict], label_key: str = "checkpoint") -> dict:
    if len(reports) < 2:
        return {"error": "Cần ít nhất 2 checkpoints để so sánh"}

    by_scenario = {}
    for r in reports:
        scen = r["scenario"]
        by_scenario.setdefault(scen, []).append(r)

    comparison  = {}
    regressions = []
    improvements = []

    for scenario, rows in by_scenario.items():
        if len(rows) < 2:
            continue
        base, curr = rows[0], rows[-1]
        deltas = {}
        for metric in BALANCE_THRESHOLDS:
            bval = base.get(metric, float("nan"))
            cval = curr.get(metric, float("nan"))
            if math.isnan(bval) or math.isnan(cval):
                continue
            delta = cval - bval
            direction = BALANCE_THRESHOLDS[metric]["direction"]
            is_better = delta < 0 if direction == "lower_is_better" else delta > 0
            deltas[metric] = {"base": bval, "curr": cval, "delta": delta, "better": is_better}
            if not is_better and abs(delta) > 0.01:
                regressions.append(f"{scenario}.{metric}: {bval:.3f} → {cval:.3f}")
            elif is_better and abs(delta) > 0.01:
                improvements.append(f"{scenario}.{metric}: {bval:.3f} → {cval:.3f} ✅")
        comparison[scenario] = deltas

    reg_count = len(regressions)
    imp_count = len(improvements)
    recommend = (
        "Dùng checkpoint mới nhất (không có regression)" if reg_count == 0
        else f"Dùng checkpoint mới ({imp_count} improvements vs {reg_count} regressions)"
             if imp_count > reg_count
        else f"⚠️ Dùng checkpoint cũ hơn ({reg_count} regressions trong checkpoint mới)"
    )

    return {
        "by_scenario":    comparison,
        "regressions":    regressions,
        "improvements":   improvements,
        "verdict":        "REGRESSION" if regressions else "IMPROVEMENT" if improvements else "UNCHANGED",
        "recommendation": recommend,
    }
```

---

## Bước 4 — Curriculum phase và advance check

```python
STAGE_SUCCESS_VALUES = {"balance": 7.0, "balance_robust": 6.0, "stand_up": 5.0}

def curriculum_phase(min_height: float | None) -> str:
    if min_height is None:       return "unknown"
    if min_height >= 0.65:       return f"Phase A — narrow [{min_height:.2f}, 0.70]m"
    elif min_height >= 0.50:     return f"Phase B — moderate [{min_height:.2f}, 0.70]m"
    elif min_height > 0.40:      return f"Phase C — full range [{min_height:.2f}, 0.70]m"
    else:                        return "✅ Completed [0.40, 0.70]m"

def check_advance_readiness(eval_per_step: float,
                             stage: str = "balance",
                             curriculum_min_height: float | None = None,
                             eval_success_rate: float | None = None,
                             fall_rate: float | None = None) -> dict:
    success_val = STAGE_SUCCESS_VALUES.get(stage, 7.0)
    checks = {}

    checks["eval_per_step"] = {
        "value": eval_per_step, "threshold": success_val,
        "passed": eval_per_step >= success_val,
        "note": f"{eval_per_step:.2f} {'≥' if eval_per_step >= success_val else '<'} {success_val}",
    }
    if stage == "balance" and curriculum_min_height is not None:
        checks["curriculum"] = {
            "value": curriculum_min_height, "threshold": 0.41,
            "passed": curriculum_min_height <= 0.41,
            "note": curriculum_phase(curriculum_min_height),
        }
    if eval_success_rate is not None:
        checks["success_rate"] = {
            "value": eval_success_rate, "threshold": 0.80,
            "passed": eval_success_rate >= 0.80,
        }
    if fall_rate is not None:
        checks["fall_rate"] = {
            "value": fall_rate, "threshold": 0.15,
            "passed": fall_rate <= 0.15,
        }

    all_passed = all(c["passed"] for c in checks.values())
    return {
        "stage": stage, "checks": checks, "ready_to_advance": all_passed,
        "verdict": "✅ Đủ điều kiện advance stage" if all_passed
                   else "⏳ Chưa đủ điều kiện — xem chi tiết bên dưới",
    }
```

---

## Bước 5 — Tổng hợp báo cáo

```python
def build_report(eval_balance_path: str = None,
                 evaluate_path: str = None,
                 validation_path: str = None,
                 jsonl_path: str = None) -> dict:
    """
    Tổng hợp report từ nhiều nguồn theo thứ tự ưu tiên:
      1. validation_report.json  → PRIMARY: exploit detection + basic benchmark
      2. *_metrics.jsonl         → PRIMARY: training trend
      3. eval_results_*.json     → Secondary: override eval_per_step
      4. eval_results.json       → Secondary: multi-scenario paper metrics

    Có thể analyze và ra quyết định chỉ với validation + JSONL.
    eval_balance KHÔNG bắt buộc.
    """
    report = {"sources": [], "scenarios": [], "warns": [], "training_trend": None}

    # ── PRIMARY 1: validation_report.json ────────────────────────────────────
    if validation_path and Path(validation_path).exists():
        val = load_validation_report(validation_path)
        report["validation"] = val

        # Exploit warns → report["warns"] ngay, không cần đợi eval_balance
        for sig, v in val["warn_signals"].items():
            report["warns"].append(f"exploit.{sig}={v:.2f} [WARN]")

        # benchmark từ validation làm fallback cho quyết định train
        val_bench = val.get("benchmark", {})
        if val_bench:
            ep_len = val_bench.get("episode_length_mean", 1000) or 1000
            report["benchmark"] = {
                "reward_mean":   val_bench.get("reward_mean",   float("nan")),
                "fall_rate":     val_bench.get("fall_rate",     float("nan")),
                "success_rate":  val_bench.get("success_rate",  float("nan")),
                "eval_per_step": val_bench.get("reward_mean", 0) / ep_len,
                "source":        "validation_report.benchmark",
            }
        report["sources"].append("validation_report.json")

    # ── PRIMARY 2: JSONL training log ─────────────────────────────────────────
    if jsonl_path and Path(jsonl_path).exists():
        records = load_jsonl_log(jsonl_path)
        report["training_trend"] = extract_training_trend(records)
        report["sources"].append("metrics.jsonl")

    # ── Secondary: evaluate.py (override benchmark nếu có — trustworthy hơn) ─
    if evaluate_path and Path(evaluate_path).exists():
        bench = load_evaluate(evaluate_path)
        bench["fall_rate_flag"] = flag_metric(bench["fall_rate"], "fall_rate")
        report["benchmark"] = bench   # override validation.benchmark
        report["sources"].append("evaluate.json")

    # ── Secondary: eval_balance.py (multi-scenario, paper metrics) ────────────
    if eval_balance_path and Path(eval_balance_path).exists():
        data    = load_eval_balance(eval_balance_path)
        metrics = parse_scenario_metrics(data.get("results", []))
        for m in metrics:
            flags = {k: flag_metric(m[k], k) for k in BALANCE_THRESHOLDS if k in m}
            worst = ("CRITICAL" if "CRITICAL" in flags.values()
                     else "WARN" if "WARN" in flags.values() else "OK")
            m["flags"] = flags
            m["overall_status"] = worst
            if worst != "OK":
                for k, v in flags.items():
                    if v in ("WARN", "CRITICAL"):
                        report["warns"].append(f"{m['scenario']}.{k}={m[k]:.3f} [{v}]")
        report["scenarios"] = metrics
        report["sources"].append("eval_balance.json")

    return report
```

### Format output chuẩn

```
=== EVAL ANALYZER REPORT — seed42/final ===
Sources: validation_report.json, metrics.jsonl

QUALITY SIGNALS (validate_checkpoint — PRIMARY):
  wheel_spin_mean_rads : 0.90 rad/s  ✅ OK   (threshold: 3.0)
  xy_drift_max_m       : 0.08m       ✅ OK   (threshold: 0.3)
  ctrl_jitter_mean_nm  : 0.30 Nm     ✅ OK   (threshold: 1.5)
  leg_asymmetry_mean   : 0.04 rad    ✅ OK   (threshold: 0.1)
  Suspicious: 0 signals

BENCHMARK (nominal, từ validation):
  reward_mean=7250  fall_rate=8%  success_rate=92%  eval_per_step=7.25

TREND (JSONL — PRIMARY):
  ➡️ PLATEAU (+1.2%)  eval_reward=7480  fall=5.2%  @12M steps

SCENARIO METRICS (eval_balance — optional, paper use):
  SCENARIO        FALL%  SURV(s)  PITCH°  H_ERR  MAX_PUSH  STATUS
  nominal          5%     19.8     1.2    0.012     —       ✅ OK
  friction_low    28%     14.1     3.5    0.025    52N      ⚠️ WARN

⚠️ WARNS: friction_low.fall_rate=0.28
```

---

## Khi không có file — thứ tự ưu tiên chạy eval

```
Bước 1 — REQUIRED để ra quyết định train:
  python scripts/validate_checkpoint.py \
      --checkpoint <checkpoint_dir>
  # → tạo validation_report.json (exploit + benchmark)

Bước 2 — OPTIONAL, chỉ cần cho paper metrics / multi-scenario:
  python scripts/eval_balance.py \
      --checkpoint <checkpoint_dir> \
      --scenarios nominal push_recovery friction_low friction_high \
      --num-episodes 20
  # → tạo eval_results.json
```

---

## Quy tắc khi dùng skill này

1. **Luôn gọi `find_eval_files()` đầu tiên** — tìm `validation_report.json` trước.
2. **`validation_report.json` là PRIMARY** — có thể ra quyết định chỉ với validation + JSONL, không cần eval_balance.
3. **`validation_report.json` dùng key `"standing_quality"`** — không phải `"quality_signals"`.
4. **`eval/reward_mean` trong JSONL là episode sum** — detect trend bằng % change (≥3%).
5. **eval_per_step = reward_mean / episode_length** — so với `success_value` (7.0/6.0/5.0).
6. **Không tự ra quyết định train** — chỉ flag và tóm tắt. Dùng training-decision skill sau.
7. **Khi so sánh checkpoints** — dùng `compare_checkpoints()`, highlight regression trước.
8. **Không nhầm curriculum gate** (PPOTrainer.eval_pass()) với research eval (`eval_balance.py`).
