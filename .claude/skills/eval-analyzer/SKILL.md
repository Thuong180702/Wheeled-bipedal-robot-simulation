---
name: eval-analyzer
description: >
  Dùng skill này để đọc, parse và tóm tắt kết quả evaluation của wheeled biped project.
  Kích hoạt khi user nhắc đến: checkpoint, eval results, training trend, fall rate,
  exploit pattern, wheel spin, friction score, push recovery, WARN flag, metrics,
  hay bất cứ lúc nào cần hiểu policy đang tốt/xấu thế nào.
  Biết xử lý 4 loại file: eval_results.json (eval_balance.py), eval_results_*.json
  (evaluate.py), validation_report.json (validate_checkpoint.py), *_metrics.jsonl
  (training log). Tự tìm file — không hỏi lại đường dẫn trừ khi không có context.
  Output: bảng metrics WARN/OK/CRITICAL, phát hiện exploit, training trend.
  Không ra quyết định train — đó là việc của training-decision skill.
license: Project-internal skill
---

# Eval Analyzer Skill

## Tổng quan nhanh — 4 loại file, 4 hàm đọc

| File                     | Script nguồn             | Hàm đọc                    | Key dùng                             |
| ------------------------ | ------------------------ | -------------------------- | ------------------------------------ |
| `eval_results.json`      | `eval_balance.py`        | `load_eval_balance()`      | `.results[]`                         |
| `eval_results_*.json`    | `evaluate.py`            | `load_evaluate()`          | `.reward_mean`, `.fall_rate`         |
| `validation_report.json` | `validate_checkpoint.py` | `load_validation_report()` | `.standing_quality` ⚠️               |
| `*_metrics.jsonl`        | Training logger          | `load_jsonl_log()`         | `eval/reward_mean`, `eval/fall_rate` |

> ⚠️ **Key quan trọng**: `validation_report.json` dùng key `"standing_quality"` (KHÔNG phải `"quality_signals"`).

---

## Bước 0 — Tìm files tự động

```python
from pathlib import Path

def find_eval_files(checkpoint_dir: str) -> dict:
    """Tự động tìm tất cả eval files. Gọi đầu tiên, trước khi làm gì khác."""
    base = Path(checkpoint_dir)
    found = {}

    # eval_balance.py lưu output vào checkpoint dir (hoặc --output-dir)
    # Thứ tự ưu tiên: trong checkpoint dir → trong seed dir
    for p in [
        base / "eval_results.json",
        base.parent.parent / "paper_eval" / "eval_results.json",
    ]:
        if p.exists():
            found["eval_balance"] = str(p)
            break

    # evaluate.py outputs — nhiều modes
    for mode in ["nominal", "push_recovery", "domain_randomized", "command_tracking"]:
        p = base / f"eval_results_{mode}.json"
        if p.exists():
            found[f"evaluate_{mode}"] = str(p)

    # validation report (validate_checkpoint.py)
    p = base / "validation_report.json"
    if p.exists():
        found["validation"] = str(p)

    # JSONL training log — ở seed dir (parent của checkpoints/)
    seed_dir = base.parent.parent  # outputs/<stage>/rl/seed<N>/
    for pattern in ["*_metrics.jsonl", "metrics.jsonl"]:
        matches = list(seed_dir.glob(pattern))
        if matches:
            found["jsonl"] = str(sorted(matches)[-1])  # lấy mới nhất
            break

    return found
```

---

## Bước 1 — Đọc từng loại file

### eval_results.json (eval_balance.py)

```python
import json, math
from pathlib import Path

def load_eval_balance(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Structure: {controller, checkpoints: [...], scenarios: [...], results: [ScenarioMetrics...]}
    return data

def parse_scenario_metrics(results: list[dict]) -> list[dict]:
    """Trích xuất metrics từ list results của eval_balance.py."""
    parsed = []
    for r in results:
        parsed.append({
            "checkpoint": Path(r.get("checkpoint", "")).name,
            "scenario": r["scenario"],
            # Core metrics
            "fall_rate":              r.get("fall_rate",              float("nan")),
            "survival_rate":          r.get("survival_rate",          float("nan")),
            "survival_time_mean_s":   r.get("survival_time_mean_s",   float("nan")),
            # Posture quality
            "pitch_rms_deg":          r.get("pitch_rms_deg",          float("nan")),
            "roll_rms_deg":           r.get("roll_rms_deg",           float("nan")),
            "height_rmse_m":          r.get("height_rmse_m",          float("nan")),
            # Exploit signals
            "wheel_speed_rms_rads":   r.get("wheel_speed_rms_rads",   float("nan")),
            "xy_drift_max_m":         r.get("xy_drift_max_m",         float("nan")),
            "torque_rms_nm":          r.get("torque_rms_nm",          float("nan")),
            # Push recovery
            "recovery_time_s":        r.get("recovery_time_s",        float("nan")),
            "max_recoverable_push_n": r.get("max_recoverable_push_n", float("nan")),
        })
    return parsed
```

### eval*results*\*.json (evaluate.py)

```python
def load_evaluate(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    ep_len = data.get("episode_length_mean", 1000) or 1000  # guard div/0
    return {
        "checkpoint":         data.get("checkpoint", ""),
        "mode":               data.get("mode", ""),
        "reward_mean":        data.get("reward_mean",        float("nan")),
        "reward_std":         data.get("reward_std",         float("nan")),
        "fall_rate":          data.get("fall_rate",          float("nan")),
        "success_rate":       data.get("success_rate",       float("nan")),
        "episode_length_mean": ep_len,
        # eval_per_step dùng để so với success_value trong curriculum.yaml
        "eval_per_step":      data.get("reward_mean", 0) / ep_len,
        "mode_metrics":       data.get("mode_metrics", {}),
    }
```

### validation_report.json (validate_checkpoint.py)

```python
def load_validation_report(path: str) -> dict:
    """
    ⚠️ KEY QUAN TRỌNG: dùng "standing_quality", KHÔNG phải "quality_signals".
    Cấu trúc thực tế:
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
        "flags": ["HIGH wheel spin (4.1 rad/s)..."],
        "num_suspicious": 1
      },
      "benchmark": {...},
      "checkpoint": "...",
    }
    """
    with open(path) as f:
        data = json.load(f)

    quality = data.get("standing_quality", {})
    flags   = quality.get("flags", [])

    # Map signal name → WARN nếu có trong flags
    from wheeled_biped.eval.standing_quality import THRESHOLDS
    warn_signals = {}
    for signal, threshold in THRESHOLDS.items():
        val = quality.get(signal)
        if val is not None and not math.isnan(float(val)) and float(val) > threshold:
            warn_signals[signal] = float(val)

    return {
        "benchmark":      data.get("benchmark", {}),
        "quality_signals": quality,   # raw dict với float values
        "flags":           flags,     # list of human-readable strings
        "warn_signals":    warn_signals,
        "num_suspicious":  quality.get("num_suspicious", 0),
    }
```

### \*\_metrics.jsonl (training log)

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
    Lưu ý: eval/reward_mean là EPISODE SUM (không phải per-step).
    Dùng % change để detect plateau, không dùng absolute threshold.
    """
    eval_rewards  = [(r["step"], r["eval/reward_mean"])
                     for r in records if "eval/reward_mean" in r]
    fall_rates    = [(r["step"], r["eval/fall_rate"])
                     for r in records if "eval/fall_rate" in r]
    curriculum_pts = [(r["step"], r["curriculum/eval_per_step"])
                      for r in records if "curriculum/eval_per_step" in r]

    if not eval_rewards:
        return {"status": "no_eval_data", "reward_trend": "unknown"}

    steps, rewards = zip(*eval_rewards)
    recent_n = min(6, len(rewards))
    recent = rewards[-recent_n:]

    # Dùng % change (rewards là episode sum ~ hàng nghìn, 0.1 là không đáng kể)
    base = max(abs(recent[0]), 1.0)
    delta_pct = (recent[-1] - recent[0]) / base

    if delta_pct > 0.03:       # > 3% improvement
        trend = "improving"
    elif delta_pct < -0.03:    # > 3% decline
        trend = "declining"
    else:
        trend = "plateau"

    return {
        "total_steps":         max(steps),
        "latest_eval_reward":  rewards[-1],
        "latest_fall_rate":    fall_rates[-1][1] if fall_rates else float("nan"),
        "reward_trend":        trend,
        "delta_pct":           delta_pct * 100,  # e.g. "+5.2%"
        "recent_rewards":      list(recent),
        "curriculum_pts":      curriculum_pts[-10:],
        "num_eval_points":     len(eval_rewards),
    }
```

---

## Bước 2 — Flags WARN/CRITICAL

### Thresholds cho balance task

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
    """
    So sánh nhiều checkpoint reports, tính delta, highlight regression.
    reports: list of dicts từ parse_scenario_metrics(), mỗi dict có 'checkpoint' field.
    """
    if len(reports) < 2:
        return {"error": "Cần ít nhất 2 checkpoints để so sánh"}

    # Group by scenario
    by_scenario = {}
    for r in reports:
        scen = r["scenario"]
        by_scenario.setdefault(scen, []).append(r)

    comparison = {}
    regressions = []
    improvements = []

    for scenario, rows in by_scenario.items():
        if len(rows) < 2:
            continue
        # Sắp xếp theo thứ tự được cung cấp
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
            deltas[metric] = {"base": bval, "curr": cval, "delta": delta,
                               "better": is_better}
            if not is_better and abs(delta) > 0.01:
                regressions.append(f"{scenario}.{metric}: {bval:.3f} → {cval:.3f}")
            elif is_better and abs(delta) > 0.01:
                improvements.append(f"{scenario}.{metric}: {bval:.3f} → {cval:.3f} ✅")
        comparison[scenario] = deltas

    # Recommend: checkpoint nào tốt hơn overall
    reg_count = len(regressions)
    imp_count = len(improvements)
    if reg_count == 0:
        recommend = f"Dùng checkpoint mới nhất (ít regression hơn)"
    elif imp_count > reg_count:
        recommend = f"Dùng checkpoint mới ({imp_count} improvements vs {reg_count} regressions)"
    else:
        recommend = f"⚠️ Dùng checkpoint cũ hơn ({reg_count} regressions trong checkpoint mới)"

    return {
        "by_scenario": comparison,
        "regressions": regressions,
        "improvements": improvements,
        "verdict": "REGRESSION" if regressions else "IMPROVEMENT" if improvements else "UNCHANGED",
        "recommendation": recommend,
    }
```

---

## Bước 4 — Curriculum phase và advance check

```python
# success_value theo từng stage (từ curriculum.yaml)
STAGE_SUCCESS_VALUES = {
    "balance":        7.0,
    "balance_robust": 6.0,
    "stand_up":       5.0,
}

def curriculum_phase(min_height: float | None) -> str:
    """Map curriculum_min_height → phase label."""
    if min_height is None:
        return "unknown"
    if min_height >= 0.65:
        return f"Phase A — narrow [{min_height:.2f}, 0.70]m"
    elif min_height >= 0.50:
        return f"Phase B — moderate [{min_height:.2f}, 0.70]m"
    elif min_height > 0.40:
        return f"Phase C — full range [{min_height:.2f}, 0.70]m"
    else:
        return "✅ Completed [0.40, 0.70]m"

def check_advance_readiness(eval_per_step: float,
                             stage: str = "balance",
                             curriculum_min_height: float | None = None,
                             eval_success_rate: float | None = None,
                             fall_rate: float | None = None) -> dict:
    """
    Kiểm tra điều kiện advance sang stage tiếp theo.
    eval_per_step: từ evaluate.py (reward_mean / episode_length)
    """
    success_val = STAGE_SUCCESS_VALUES.get(stage, 7.0)
    checks = {}

    # Check 1: eval_per_step đủ ngưỡng
    checks["eval_per_step"] = {
        "value": eval_per_step,
        "threshold": success_val,
        "passed": eval_per_step >= success_val,
        "note": f"{eval_per_step:.2f} {'≥' if eval_per_step >= success_val else '<'} {success_val}",
    }

    # Check 2: curriculum hoàn thành (chỉ balance stage)
    if stage == "balance" and curriculum_min_height is not None:
        completed = curriculum_min_height <= 0.41
        checks["curriculum"] = {
            "value": curriculum_min_height,
            "threshold": 0.41,
            "passed": completed,
            "note": curriculum_phase(curriculum_min_height),
        }

    # Check 3: success_rate và fall_rate (optional)
    if eval_success_rate is not None:
        checks["success_rate"] = {
            "value": eval_success_rate,
            "threshold": 0.80,
            "passed": eval_success_rate >= 0.80,
        }
    if fall_rate is not None:
        checks["fall_rate"] = {
            "value": fall_rate,
            "threshold": 0.15,
            "passed": fall_rate <= 0.15,
        }

    all_passed = all(c["passed"] for c in checks.values())
    return {
        "stage": stage,
        "checks": checks,
        "ready_to_advance": all_passed,
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
    report = {"sources": [], "scenarios": [], "warns": [], "training_trend": None}

    if eval_balance_path and Path(eval_balance_path).exists():
        data = load_eval_balance(eval_balance_path)
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
                        report["warns"].append(
                            f"{m['scenario']}.{k}={m[k]:.3f} [{v}]"
                        )
        report["scenarios"] = metrics
        report["sources"].append("eval_balance.json")

    if evaluate_path and Path(evaluate_path).exists():
        bench = load_evaluate(evaluate_path)
        bench["fall_rate_flag"] = flag_metric(bench["fall_rate"], "fall_rate")
        report["benchmark"] = bench
        report["sources"].append("evaluate.json")

    if validation_path and Path(validation_path).exists():
        val = load_validation_report(validation_path)  # dùng đúng key "standing_quality"
        report["validation"] = val
        if val["warn_signals"]:
            for sig, v in val["warn_signals"].items():
                report["warns"].append(f"exploit.{sig}={v:.2f} [WARN]")
        report["sources"].append("validation_report.json")

    if jsonl_path and Path(jsonl_path).exists():
        records = load_jsonl_log(jsonl_path)
        report["training_trend"] = extract_training_trend(records)
        report["sources"].append("metrics.jsonl")

    return report
```

### Format output chuẩn

```
=== EVAL ANALYZER REPORT — final ===
SCENARIO        FALL%  SURV(s)  PITCH°  H_ERR  MAX_PUSH  STATUS
───────────────────────────────────────────────────────────────
nominal          5%     19.8     1.2    0.012     —       ✅ OK
friction_low    28%     14.1     3.5    0.025    52N      ⚠️ WARN
full_range      22%     15.8     3.1    0.031     —       ⚠️ WARN

⚠️ WARNS: friction_low.fall_rate=0.28, full_range.height_rmse_m=0.031
EXPLOIT: wheel_spin=0.9 ✅  xy_drift=0.08m ✅  jitter=0.3Nm ✅
TREND: ➡️ PLATEAU (+1.2%)  eval_reward=7480  fall=5.2%  @12M steps
```

---

## Khi không có file

```
Không tìm thấy eval files trong <checkpoint_dir>.
Chạy trước:
  python scripts/eval_balance.py \
      --checkpoint <checkpoint_dir> \
      --scenarios nominal push_recovery friction_low friction_high \
      --num-episodes 20
  python scripts/validate_checkpoint.py \
      --checkpoint <checkpoint_dir>
```

---

## Quy tắc khi dùng skill này

1. **Luôn gọi `find_eval_files()` đầu tiên** — không hỏi user đường dẫn trừ khi không có context.
2. **`validation_report.json` dùng key `"standing_quality"`** — không phải `"quality_signals"`.
3. **`eval/reward_mean` trong JSONL là episode sum** — detect trend bằng % change (≥3%), không dùng absolute delta.
4. **eval_per_step = reward_mean / episode_length** — so với `success_value` (7.0/6.0/5.0 cho balance/robust/stand_up).
5. **Không tự ra quyết định train** — chỉ flag và tóm tắt. Dùng training-decision skill sau.
6. **Khi so sánh checkpoints** — dùng `compare_checkpoints()`, highlight regression trước improvement.
7. **Nominal scenario là anchor** — nếu nominal CRITICAL thì không cần xem scenarios khác.
8. **Không nhầm curriculum gate** (PPOTrainer.eval_pass(), in-training) với research eval (`eval_balance.py`, offline).
