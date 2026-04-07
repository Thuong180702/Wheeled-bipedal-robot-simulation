---
name: TrainingMonitorAgent
description: >
  Agent theo dõi sức khoẻ của một training run đang diễn ra hoặc vừa kết thúc.
  Đọc JSONL log, phát hiện sớm các vấn đề (plateau, divergence, curriculum stuck,
  exploit patterns), và cảnh báo với action cụ thể — trước khi lãng phí thêm
  GPU time. Không cần eval_results.json — chỉ cần JSONL log file.
  Orchestrates: eval-analyzer (trend) → training-decision (early action).
skills_used:
  - eval-analyzer    # phần extract_training_trend và JSONL parsing
  - training-decision
  - config-advisor   # nếu phát hiện symptom cần fix config
license: Project-internal agent
---

# TrainingMonitorAgent

## Mục đích

Trả lời câu hỏi: **"Training đang khoẻ không? Có nên dừng lại và làm gì không?"**

Khác với `CheckpointReviewAgent` (review sau khi train xong), agent này được thiết
kế để chạy **trong khi đang train** hoặc **ngay sau khi một run kết thúc/bị dừng**,
trước khi quyết định có resume hay không.

---

## Trigger patterns

- `"Xem log training seed42 đang thế nào"`
- `"balance_seed42_metrics.jsonl có vấn đề gì không?"`
- `"Training đang chạy, có nên dừng không?"`
- `"Curriculum có đang advance không?"`
- `"Tại sao reward không tăng nữa?"`
- `"Kiểm tra xem có exploit không"`

---

## Inputs

```python
@dataclass
class TrainingMonitorInput:
    # Bắt buộc ít nhất 1 trong 2
    jsonl_path: str | None = None    # e.g. "outputs/balance/rl/seed42/balance_seed42_metrics.jsonl"
    seed_dir: str | None = None      # e.g. "outputs/balance/rl/seed42" — tự tìm JSONL

    # Optional context
    stage: str = "balance"
    seed: int | None = None
    target_steps: int = 50_000_000   # tổng steps dự kiến
    check_window: int = 10           # số eval points cuối để tính trend
```

---

## Metric keys trong JSONL

```python
# Từ PPOTrainer.train() logging (wheeled_biped/training/ppo.py)
TRAINING_METRICS = {
    # Training metrics (mỗi log_interval updates)
    "reward/mean":               "avg per-rollout reward (noisy)",
    "reward/std":                "reward std",
    "training/fps":              "environment steps per second",
    "training/global_step":      "total env steps so far",
    "training/update_time_s":    "seconds per PPO update",

    # Eval metrics (end-of-stage eval_pass)
    "eval/reward_mean":          "held-out eval reward (trustworthy)",
    "eval/fall_rate":            "fraction of eval episodes that fell",
    "eval/success_rate":         "fraction of eval episodes that survived",
    "eval/reward_std":           "eval reward std",

    # Curriculum metrics (mỗi eval_interval updates, nếu curriculum enabled)
    "curriculum/level":          "current height level (1–29)",
    "curriculum/min_height":     "current minimum commanded height (m)",
    "curriculum/eval_per_step":  "eval_reward_mean / episode_length",
    "curriculum/reward_threshold": "advancement threshold (reward/step)",
    "curriculum/eval_success_rate": "success rate tại curriculum eval",
    "curriculum/eval_fall_rate":    "fall rate tại curriculum eval",

    # PPO loss breakdown
    "loss/policy":               "policy gradient loss",
    "loss/value":                "value function loss",
    "loss/entropy":              "entropy term",
    "loss/total":                "total loss",
}
```

---

## Phase 1: LOAD & PARSE JSONL

```python
import json, math
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

@dataclass
class TrainingSnapshot:
    """Toàn bộ thông tin từ một JSONL log file."""
    jsonl_path: str
    total_steps: int
    total_eval_points: int             # số lần eval/reward_mean được log
    total_curriculum_evals: int        # số lần curriculum/eval_per_step được log

    # Time series (step, value)
    reward_mean_ts:           list[tuple]  # (step, reward/mean)
    eval_reward_ts:           list[tuple]  # (step, eval/reward_mean)
    eval_fall_rate_ts:        list[tuple]  # (step, eval/fall_rate)
    curriculum_level_ts:      list[tuple]  # (step, curriculum/level)
    curriculum_min_height_ts: list[tuple]  # (step, curriculum/min_height)
    curriculum_eval_per_step_ts: list[tuple]  # (step, curriculum/eval_per_step)
    fps_ts:                   list[tuple]  # (step, training/fps)
    loss_policy_ts:           list[tuple]  # (step, loss/policy)
    loss_value_ts:            list[tuple]  # (step, loss/value)
    loss_entropy_ts:          list[tuple]  # (step, loss/entropy)

    # Derived (tính sau khi load)
    latest_reward_mean: float
    latest_eval_reward: float
    latest_eval_fall_rate: float
    latest_curriculum_level: int | None
    latest_curriculum_min_height: float | None
    latest_eval_per_step: float | None
    latest_fps: float | None


def load_jsonl_snapshot(jsonl_path: str) -> TrainingSnapshot:
    """Parse toàn bộ JSONL log thành TrainingSnapshot."""
    by_tag = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                tag   = r.get("tag", "")
                step  = r.get("step", 0)
                value = r.get("value")
                if tag and value is not None:
                    by_tag.setdefault(tag, []).append((step, float(value)))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

    # Sort tất cả time series theo step
    for tag in by_tag:
        by_tag[tag].sort(key=lambda x: x[0])

    def get_ts(tag):
        return by_tag.get(tag, [])

    def latest(ts):
        return ts[-1][1] if ts else float("nan")

    def latest_int(ts):
        v = latest(ts)
        return int(v) if not math.isnan(v) else None

    reward_ts    = get_ts("reward/mean")
    eval_rew_ts  = get_ts("eval/reward_mean")
    eval_fr_ts   = get_ts("eval/fall_rate")
    cur_level_ts = get_ts("curriculum/level")
    cur_h_ts     = get_ts("curriculum/min_height")
    cur_ep_ts    = get_ts("curriculum/eval_per_step")
    fps_ts       = get_ts("training/fps")

    all_steps = [s for ts in by_tag.values() for s, _ in ts]
    total_steps = max(all_steps) if all_steps else 0

    return TrainingSnapshot(
        jsonl_path=jsonl_path,
        total_steps=total_steps,
        total_eval_points=len(eval_rew_ts),
        total_curriculum_evals=len(cur_ep_ts),
        reward_mean_ts=reward_ts,
        eval_reward_ts=eval_rew_ts,
        eval_fall_rate_ts=eval_fr_ts,
        curriculum_level_ts=cur_level_ts,
        curriculum_min_height_ts=cur_h_ts,
        curriculum_eval_per_step_ts=cur_ep_ts,
        fps_ts=fps_ts,
        loss_policy_ts=get_ts("loss/policy"),
        loss_value_ts=get_ts("loss/value"),
        loss_entropy_ts=get_ts("loss/entropy"),
        latest_reward_mean=latest(reward_ts),
        latest_eval_reward=latest(eval_rew_ts),
        latest_eval_fall_rate=latest(eval_fr_ts),
        latest_curriculum_level=latest_int(cur_level_ts),
        latest_curriculum_min_height=latest(cur_h_ts) if cur_h_ts else None,
        latest_eval_per_step=latest(cur_ep_ts) if cur_ep_ts else None,
        latest_fps=latest(fps_ts) if fps_ts else None,
    )


def find_jsonl(seed_dir: str, stage: str = "balance", seed: int = None) -> str | None:
    """Tự tìm JSONL file trong seed_dir."""
    base = Path(seed_dir)
    # Pattern: <stage>_seed<N>_metrics.jsonl
    candidates = list(base.glob("*_metrics.jsonl"))
    if candidates:
        return str(candidates[0])
    # Fallback: bất kỳ .jsonl nào
    candidates = list(base.glob("*.jsonl"))
    return str(candidates[0]) if candidates else None
```

---

## Phase 2: HEALTH CHECKS

```python
@dataclass
class HealthIssue:
    severity: str          # CRITICAL | WARN | INFO
    category: str          # plateau | divergence | curriculum_stuck | exploit |
                           # low_fps | data_sparse | training_complete
    title: str
    detail: str
    action: str            # action đề xuất
    symptom_key: str | None = None   # key để pass vào config-advisor nếu cần


def run_health_checks(snap: TrainingSnapshot,
                       stage: str,
                       target_steps: int,
                       check_window: int) -> list[HealthIssue]:
    """Chạy tất cả health checks và trả về danh sách issues."""
    issues = []

    issues += _check_data_quality(snap)
    issues += _check_reward_trend(snap, check_window)
    issues += _check_fall_rate(snap, check_window)
    issues += _check_divergence(snap)
    issues += _check_curriculum(snap, stage)
    issues += _check_loss_health(snap)
    issues += _check_fps(snap)
    issues += _check_progress(snap, target_steps)

    return issues


# ── Check 1: Data quality (đủ data để phân tích không?) ──────────────────────

def _check_data_quality(snap: TrainingSnapshot) -> list[HealthIssue]:
    issues = []
    if snap.total_eval_points == 0:
        issues.append(HealthIssue(
            severity="WARN", category="data_sparse",
            title="Chưa có eval data",
            detail="JSONL chưa có eval/reward_mean. Training có thể chưa đủ dài để trigger eval.",
            action="Kiểm tra eval_interval trong config, hoặc đợi training chạy thêm.",
        ))
    elif snap.total_eval_points < 5:
        issues.append(HealthIssue(
            severity="INFO", category="data_sparse",
            title=f"Ít eval points ({snap.total_eval_points})",
            detail="Chưa đủ data để phân tích trend chính xác.",
            action="Đợi thêm training steps trước khi kết luận.",
        ))
    return issues


# ── Check 2: Reward trend (plateau / improvement) ─────────────────────────────

def _check_reward_trend(snap: TrainingSnapshot,
                          window: int) -> list[HealthIssue]:
    issues = []
    ts = snap.eval_reward_ts
    if len(ts) < 4:
        return issues

    recent = ts[-window:]
    steps, rewards = zip(*recent)
    delta = rewards[-1] - rewards[0]
    n_steps = steps[-1] - steps[0]

    # Plateau: change < 1% của current value trong window
    rel_change = abs(delta) / max(abs(rewards[0]), 1e-9)
    if rel_change < 0.01 and len(recent) >= window:
        issues.append(HealthIssue(
            severity="WARN", category="plateau",
            title=f"Reward plateau (Δ={delta:+.2f} trong {n_steps/1e6:.1f}M steps)",
            detail=(
                f"eval_reward không tăng đáng kể trong {window} eval points cuối.\n"
                f"  Latest: {rewards[-1]:.2f} | First in window: {rewards[0]:.2f} | Δ={delta:+.2f} ({rel_change:.1%})"
            ),
            action="Xem xét: tăng entropy_coeff, điều chỉnh reward weights, hoặc resume từ checkpoint khác.",
            symptom_key="reward_plateau_late" if snap.total_steps > 20_000_000 else "reward_plateau_early",
        ))

    # Declining: reward giảm liên tục
    if delta < -0.5 and len(recent) >= 6:
        issues.append(HealthIssue(
            severity="CRITICAL", category="divergence",
            title=f"Reward đang giảm (Δ={delta:+.2f})",
            detail=(
                f"eval_reward giảm đáng kể trong {window} eval points cuối.\n"
                f"  Latest: {rewards[-1]:.2f} | Peak in window: {max(rewards):.2f}"
            ),
            action="DỪNG TRAINING. Kiểm tra divergence. Có thể resume từ checkpoint cũ hơn.",
        ))

    return issues


# ── Check 3: Fall rate ─────────────────────────────────────────────────────────

def _check_fall_rate(snap: TrainingSnapshot, window: int) -> list[HealthIssue]:
    issues = []
    ts = snap.eval_fall_rate_ts
    if len(ts) < 3:
        return issues

    recent_fr = [v for _, v in ts[-window:]]
    latest_fr = recent_fr[-1]
    mean_fr   = float(np.mean(recent_fr))

    if latest_fr > 0.50:
        issues.append(HealthIssue(
            severity="CRITICAL", category="divergence",
            title=f"Fall rate rất cao: {latest_fr:.1%}",
            detail=(
                f"Policy đang ngã > 50% episodes — có dấu hiệu policy collapse.\n"
                f"  Latest fall_rate: {latest_fr:.1%} | Mean (last {len(recent_fr)}): {mean_fr:.1%}"
            ),
            action="DỪNG TRAINING ngay. Quay về checkpoint trước đó. Kiểm tra config thay đổi gần đây.",
        ))
    elif latest_fr > 0.30:
        issues.append(HealthIssue(
            severity="WARN", category="plateau",
            title=f"Fall rate cao: {latest_fr:.1%}",
            detail=f"Policy không ổn định — fall_rate={latest_fr:.1%} > 30%.",
            action="Cân nhắc tăng rewards.body_level, giảm entropy_coeff.",
            symptom_key="high_fall_rate_nominal",
        ))

    # Tăng đột biến (spike): fall_rate tăng nhiều so với mean
    if latest_fr > mean_fr * 1.8 and mean_fr < 0.15 and latest_fr > 0.20:
        issues.append(HealthIssue(
            severity="WARN", category="instability",
            title=f"Fall rate tăng đột biến ({mean_fr:.1%} → {latest_fr:.1%})",
            detail="Fall rate tăng đột ngột so với baseline gần đây. Có thể do curriculum advance.",
            action="Theo dõi 2–3 eval points tiếp. Nếu không hồi phục, xem xét giảm curriculum threshold.",
        ))

    return issues


# ── Check 4: Divergence / NaN ─────────────────────────────────────────────────

def _check_divergence(snap: TrainingSnapshot) -> list[HealthIssue]:
    issues = []

    # Policy loss explode
    if snap.loss_policy_ts:
        recent_policy_loss = [v for _, v in snap.loss_policy_ts[-10:]]
        if any(math.isnan(v) or math.isinf(v) for v in recent_policy_loss):
            issues.append(HealthIssue(
                severity="CRITICAL", category="divergence",
                title="NaN/Inf trong policy loss",
                detail="loss/policy chứa NaN hoặc Inf — gradient explosion.",
                action="DỪNG TRAINING. Giảm learning_rate. Kiểm tra max_grad_norm.",
            ))
        elif recent_policy_loss and max(recent_policy_loss) > 10.0:
            issues.append(HealthIssue(
                severity="WARN", category="divergence",
                title=f"Policy loss cao bất thường: {max(recent_policy_loss):.2f}",
                detail="Loss/policy rất cao — có thể sắp diverge.",
                action="Theo dõi. Nếu tiếp tục tăng, giảm learning_rate hoặc tăng max_grad_norm.",
            ))

    # Value loss explode
    if snap.loss_value_ts:
        recent_v = [v for _, v in snap.loss_value_ts[-10:]]
        if recent_v and max(recent_v) > 100.0:
            issues.append(HealthIssue(
                severity="WARN", category="divergence",
                title=f"Value loss cao: {max(recent_v):.1f}",
                detail="loss/value rất cao — value function không fit được.",
                action="Tăng ppo.value_loss_coeff hoặc giảm learning_rate.",
            ))

    # Entropy collapse
    if snap.loss_entropy_ts and len(snap.loss_entropy_ts) >= 5:
        recent_ent = [v for _, v in snap.loss_entropy_ts[-10:]]
        if recent_ent and min(recent_ent) < 0.001:
            issues.append(HealthIssue(
                severity="WARN", category="divergence",
                title=f"Entropy collapse: {min(recent_ent):.4f}",
                detail="Entropy quá thấp — policy đang bị deterministic hóa quá mức.",
                action="Tăng ppo.entropy_coeff (tối thiểu 0.001, recommend 0.003–0.006).",
            ))

    return issues


# ── Check 5: Curriculum health ────────────────────────────────────────────────

def _check_curriculum(snap: TrainingSnapshot, stage: str) -> list[HealthIssue]:
    issues = []
    if not snap.curriculum_eval_per_step_ts:
        return issues   # curriculum không enabled hoặc chưa fire

    cur_ts  = snap.curriculum_eval_per_step_ts
    lvl_ts  = snap.curriculum_level_ts

    # Curriculum stuck: level không tăng trong nhiều eval points
    if len(lvl_ts) >= 10:
        recent_levels = [v for _, v in lvl_ts[-15:]]
        level_range   = max(recent_levels) - min(recent_levels)
        if level_range == 0:
            latest_level = int(recent_levels[-1])
            latest_ep    = cur_ts[-1][1] if cur_ts else float("nan")
            issues.append(HealthIssue(
                severity="WARN", category="curriculum_stuck",
                title=f"Curriculum stuck ở level {latest_level}",
                detail=(
                    f"Level không tăng trong 15 curriculum evals cuối.\n"
                    f"  Level: {latest_level}/29 | eval_per_step: {latest_ep:.3f}"
                ),
                action=(
                    "Xem xét giảm curriculum.reward_threshold trong balance.yaml "
                    "(e.g. 0.75 → 0.68). Kiểm tra eval_interval."
                ),
                symptom_key="curriculum_stuck_level",
            ))

    # Curriculum đang advance tốt (INFO)
    if len(lvl_ts) >= 5:
        first_level = lvl_ts[0][1]
        last_level  = lvl_ts[-1][1]
        if last_level > first_level + 3:
            issues.append(HealthIssue(
                severity="INFO", category="curriculum_ok",
                title=f"Curriculum advancing: level {int(first_level)} → {int(last_level)}",
                detail=f"min_height: {snap.latest_curriculum_min_height:.2f}m",
                action="Không cần hành động.",
            ))

    # eval_per_step vs threshold
    if snap.curriculum_eval_per_step_ts:
        latest_ep = snap.latest_eval_per_step or 0.0
        # Lấy threshold từ log nếu có
        threshold_ts = snap.reward_mean_ts  # hack: xem có key curriculum/reward_threshold không
        # (nếu không có, dùng default)
        threshold = 0.75 * 10.5  # 75% * max_reward ~= 7.875 reward/step target
        if latest_ep >= threshold:
            issues.append(HealthIssue(
                severity="INFO", category="curriculum_ok",
                title=f"eval_per_step={latest_ep:.3f} ≥ threshold — curriculum healthy",
                detail="Policy đang vượt threshold đều đặn.",
                action="Không cần hành động.",
            ))

    return issues


# ── Check 6: Loss health ──────────────────────────────────────────────────────

def _check_loss_health(snap: TrainingSnapshot) -> list[HealthIssue]:
    issues = []
    if len(snap.loss_entropy_ts) < 5:
        return issues

    # Entropy trend — nên stable hoặc giảm nhẹ theo thời gian
    recent_ent = [v for _, v in snap.loss_entropy_ts[-20:]]
    if recent_ent:
        ent_mean = np.mean(recent_ent)
        ent_last = recent_ent[-1]
        if ent_last < ent_mean * 0.3 and ent_last < 0.005:
            issues.append(HealthIssue(
                severity="WARN", category="entropy_decay",
                title=f"Entropy giảm nhanh ({ent_mean:.4f} → {ent_last:.4f})",
                detail="Policy converging quá nhanh — có thể stuck ở local minimum sớm.",
                action="Tăng nhẹ ppo.entropy_coeff (e.g. × 1.5).",
            ))
    return issues


# ── Check 7: FPS / throughput ─────────────────────────────────────────────────

def _check_fps(snap: TrainingSnapshot) -> list[HealthIssue]:
    issues = []
    if not snap.fps_ts or len(snap.fps_ts) < 3:
        return issues

    recent_fps = [v for _, v in snap.fps_ts[-10:]]
    mean_fps   = float(np.mean(recent_fps))
    latest_fps = recent_fps[-1]

    # FPS drop đột ngột > 30%
    if latest_fps < mean_fps * 0.7 and mean_fps > 1000:
        issues.append(HealthIssue(
            severity="WARN", category="low_fps",
            title=f"FPS giảm: {mean_fps:.0f} → {latest_fps:.0f}",
            detail="Throughput giảm đáng kể — có thể do memory pressure hoặc JIT recompile.",
            action="Kiểm tra GPU memory usage. Nếu OOM, giảm num_envs.",
        ))

    return issues


# ── Check 8: Training progress ────────────────────────────────────────────────

def _check_progress(snap: TrainingSnapshot, target_steps: int) -> list[HealthIssue]:
    issues = []
    pct = snap.total_steps / max(target_steps, 1)

    if pct >= 1.0:
        issues.append(HealthIssue(
            severity="INFO", category="training_complete",
            title=f"Training đã đạt target steps ({snap.total_steps/1e6:.1f}M/{target_steps/1e6:.0f}M)",
            detail="Đã đạt hoặc vượt target. Có thể đánh giá kết quả.",
            action="Chạy eval_balance.py và validate_checkpoint.py để đánh giá policy.",
        ))
    elif pct > 0.8:
        issues.append(HealthIssue(
            severity="INFO", category="near_complete",
            title=f"Gần đạt target: {pct:.0%} ({snap.total_steps/1e6:.1f}M/{target_steps/1e6:.0f}M)",
            detail="Training sắp xong.",
            action="Chuẩn bị eval sau khi hoàn thành.",
        ))

    # Warn nếu training dừng sớm (ít steps mà không có reason)
    if snap.total_steps < 1_000_000 and snap.total_eval_points == 0:
        issues.append(HealthIssue(
            severity="INFO", category="early_stage",
            title=f"Training mới bắt đầu ({snap.total_steps/1e6:.2f}M steps)",
            detail="Quá ít data để phân tích trend.",
            action="Đợi training đạt ít nhất 5M steps trước khi đánh giá.",
        ))

    return issues
```

---

## Phase 3: TRAINING SUMMARY STATS

```python
@dataclass
class TrainingSummary:
    total_steps: int
    pct_complete: float
    latest_eval_reward: float
    eval_reward_trend: str           # improving | plateau | declining | unknown
    eval_reward_change_5m: float     # thay đổi eval_reward trong 5M steps gần nhất
    latest_fall_rate: float
    fall_rate_trend: str
    curriculum_level: int | None     # None nếu không có curriculum
    curriculum_phase: str            # Phase A/B/C hoặc "Complete"
    curriculum_min_height: float | None
    curriculum_completion_pct: float # % levels đã advance (0–100)
    latest_fps: float | None
    estimated_time_remaining_h: float | None  # dựa trên FPS hiện tại

def compute_summary(snap: TrainingSnapshot, target_steps: int) -> TrainingSummary:
    def trend_of(ts, window=10) -> str:
        if len(ts) < 4:
            return "unknown"
        recent = [v for _, v in ts[-window:]]
        delta  = recent[-1] - recent[0]
        if   delta > 0.3:  return "improving"
        elif delta < -0.3: return "declining"
        else:               return "plateau"

    def change_last_n_steps(ts, n_steps=5_000_000) -> float:
        if len(ts) < 2:
            return 0.0
        cutoff = snap.total_steps - n_steps
        before = [v for s, v in ts if s <= cutoff]
        after  = [v for s, v in ts if s > cutoff]
        if not before or not after:
            return 0.0
        return after[-1] - before[-1]

    # Curriculum phase
    min_h = snap.latest_curriculum_min_height
    if min_h is None:
        phase = "N/A"
        cur_pct = 0.0
    elif min_h <= 0.41:
        phase = "✅ Complete"
        cur_pct = 100.0
    elif min_h >= 0.65:
        phase = "Phase A"
        cur_pct = (0.69 - min_h) / (0.69 - 0.40) * 100
    elif min_h >= 0.50:
        phase = "Phase B"
        cur_pct = (0.69 - min_h) / (0.69 - 0.40) * 100
    else:
        phase = "Phase C"
        cur_pct = (0.69 - min_h) / (0.69 - 0.40) * 100

    # ETA
    eta_h = None
    if snap.latest_fps and snap.latest_fps > 0:
        remaining_steps = max(0, target_steps - snap.total_steps)
        eta_s = remaining_steps / snap.latest_fps
        eta_h = eta_s / 3600

    return TrainingSummary(
        total_steps=snap.total_steps,
        pct_complete=snap.total_steps / max(target_steps, 1),
        latest_eval_reward=snap.latest_eval_reward,
        eval_reward_trend=trend_of(snap.eval_reward_ts),
        eval_reward_change_5m=change_last_n_steps(snap.eval_reward_ts),
        latest_fall_rate=snap.latest_eval_fall_rate,
        fall_rate_trend=trend_of(snap.eval_fall_rate_ts),
        curriculum_level=snap.latest_curriculum_level,
        curriculum_phase=phase,
        curriculum_min_height=min_h,
        curriculum_completion_pct=cur_pct,
        latest_fps=snap.latest_fps,
        estimated_time_remaining_h=eta_h,
    )
```

---

## Phase 4: OUTPUT

```python
def format_monitor_output(snap: TrainingSnapshot,
                            summary: TrainingSummary,
                            issues: list[HealthIssue],
                            stage: str,
                            target_steps: int) -> str:
    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    criticals = [i for i in issues if i.severity == "CRITICAL"]
    warns     = [i for i in issues if i.severity == "WARN"]
    infos     = [i for i in issues if i.severity == "INFO"]

    overall = "🔴 CRITICAL" if criticals else "⚠️  WARN" if warns else "✅ HEALTHY"
    lines += [
        "╔══════════════════════════════════════════════════════════════╗",
        f"║  TRAINING MONITOR — {stage} seed                            ║",
        f"║  Status: {overall:<52}║",
        "╚══════════════════════════════════════════════════════════════╝",
        "",
    ]

    # ── Section 1: Progress ───────────────────────────────────────────────────
    bar_len  = 30
    filled   = int(summary.pct_complete * bar_len)
    bar      = "█" * filled + "░" * (bar_len - filled)
    eta_str  = f"{summary.estimated_time_remaining_h:.1f}h remaining" if summary.estimated_time_remaining_h else "ETA unknown"
    fps_str  = f"{summary.latest_fps:.0f} fps" if summary.latest_fps else "—"
    lines += [
        "📈 TRAINING PROGRESS",
        "─" * 62,
        f"  [{bar}] {summary.pct_complete:.1%}",
        f"  Steps: {summary.total_steps:>12,} / {target_steps:,}",
        f"  FPS:   {fps_str:<12}  ETA: {eta_str}",
        "",
    ]

    # ── Section 2: Key metrics ────────────────────────────────────────────────
    tr_icon = {"improving":"📈","plateau":"➡️ ","declining":"📉","unknown":"❓"}.get
    eval_r   = f"{summary.latest_eval_reward:.2f}" if not math.isnan(summary.latest_eval_reward) else "—"
    eval_fr  = f"{summary.latest_fall_rate:.1%}"   if not math.isnan(summary.latest_fall_rate)   else "—"
    ep_str   = f"{snap.latest_eval_per_step:.3f}"  if snap.latest_eval_per_step                   else "—"
    lines += [
        "📊 KEY METRICS (latest)",
        "─" * 62,
        f"  eval_reward_mean : {eval_r:>10}  {tr_icon(summary.eval_reward_trend,'❓')} {summary.eval_reward_trend}  "
        f"(Δ5M steps: {summary.eval_reward_change_5m:+.2f})",
        f"  eval_fall_rate   : {eval_fr:>10}  {tr_icon(summary.fall_rate_trend,'❓')} {summary.fall_rate_trend}",
        f"  eval_per_step    : {ep_str:>10}  (curriculum gate threshold ~7.0 for balance)",
        f"  eval points      : {snap.total_eval_points}",
        "",
    ]

    # ── Section 3: Curriculum ────────────────────────────────────────────────
    if snap.curriculum_level_ts:
        lines += ["🎓 CURRICULUM", "─" * 62]
        bar2  = "█" * int(summary.curriculum_completion_pct / 100 * 29) + "░" * (29 - int(summary.curriculum_completion_pct / 100 * 29))
        lines += [
            f"  Phase: {summary.curriculum_phase}",
            f"  Level: {summary.curriculum_level or '?'}/29  [{bar2}] {summary.curriculum_completion_pct:.0f}%",
            f"  Min height: {summary.curriculum_min_height:.2f}m" if summary.curriculum_min_height else "  Min height: —",
            f"  Curriculum evals fired: {snap.total_curriculum_evals}",
        ]
        # Show last few levels (progression)
        if len(snap.curriculum_level_ts) >= 3:
            recent_lvls = [(s, int(v)) for s, v in snap.curriculum_level_ts[-5:]]
            lvl_str = " → ".join(f"L{v}@{s/1e6:.1f}M" for s, v in recent_lvls)
            lines.append(f"  Recent: {lvl_str}")
        lines.append("")

    # ── Section 4: Issues ─────────────────────────────────────────────────────
    if criticals or warns:
        lines += [f"{'🔴' if criticals else '⚠️ '} ISSUES DETECTED", "─" * 62]
        for issue in criticals + warns:
            sev_icon = "🔴" if issue.severity == "CRITICAL" else "⚠️ "
            lines += [
                f"  {sev_icon} [{issue.category.upper()}] {issue.title}",
                f"     {issue.detail.replace(chr(10), chr(10) + '     ')}",
                f"     → {issue.action}",
                "",
            ]

    if infos:
        lines += ["ℹ️  INFO", "─" * 62]
        for issue in infos:
            lines.append(f"  ℹ️  {issue.title}: {issue.detail[:80]}")
        lines.append("")

    # ── Section 5: Actionable commands ────────────────────────────────────────
    lines += ["⚡ RECOMMENDED NEXT STEPS", "─" * 62]
    seed_dir   = Path(snap.jsonl_path).parent
    stage_path = seed_dir.parent.parent  # outputs/<stage>
    seed_label = seed_dir.name           # seed42

    if any(i.severity == "CRITICAL" for i in issues):
        lines += [
            "  🛑 DỪNG TRAINING:",
            "     Ctrl+C hoặc kill training process",
            "",
            "  Sau đó review checkpoint tốt nhất:",
            f"    python scripts/validate_checkpoint.py \\",
            f"        --checkpoint {seed_dir}/checkpoints/final",
            "",
        ]
    elif summary.pct_complete >= 1.0:
        lines += [
            "  ✅ Training xong — chạy eval:",
            f"    python scripts/eval_balance.py \\",
            f"        --checkpoint {seed_dir}/checkpoints/final \\",
            f"        --scenarios nominal push_recovery friction_low friction_high \\",
            f"        --num-episodes 20",
            "",
            f"    python scripts/validate_checkpoint.py \\",
            f"        --checkpoint {seed_dir}/checkpoints/final",
            "",
        ]
    else:
        issue_with_symptom = next((i for i in issues if i.symptom_key), None)
        if issue_with_symptom:
            lines += [
                "  ⚠️  Có vấn đề cần xử lý — xem config-advisor:",
                f"     symptom: {issue_with_symptom.symptom_key}",
                f"     Gọi config-advisor skill với symptom_key='{issue_with_symptom.symptom_key}'",
                "",
            ]
        else:
            lines += [
                "  ▶️  Tiếp tục training — không có vấn đề nghiêm trọng",
                f"     Eval point tiếp theo sau khoảng {snap.total_steps/1e6 + 2:.1f}M steps",
                "",
            ]

    lines += ["─" * 62]
    return "\n".join(lines)
```

---

## Entrypoint

```python
def run(inp: TrainingMonitorInput) -> str:
    # Tìm JSONL file
    if inp.jsonl_path:
        jsonl_path = inp.jsonl_path
    elif inp.seed_dir:
        jsonl_path = find_jsonl(inp.seed_dir, inp.stage, inp.seed)
        if not jsonl_path:
            return f"❌ Không tìm thấy JSONL file trong {inp.seed_dir}"
    else:
        return "❌ Cần cung cấp jsonl_path hoặc seed_dir"

    print(f"📂 Loading: {jsonl_path}")
    snap = load_jsonl_snapshot(jsonl_path)

    print(f"   {snap.total_steps/1e6:.2f}M steps, {snap.total_eval_points} eval points")

    print("📋 Computing summary...")
    summary = compute_summary(snap, inp.target_steps)

    print("🔍 Running health checks...")
    issues = run_health_checks(snap, inp.stage, inp.target_steps, inp.check_window)

    return format_monitor_output(snap, summary, issues, inp.stage, inp.target_steps)
```

---

## Ví dụ output

```
╔══════════════════════════════════════════════════════════════╗
║  TRAINING MONITOR — balance seed                             ║
║  Status: ⚠️  WARN                                            ║
╚══════════════════════════════════════════════════════════════╝

📈 TRAINING PROGRESS
──────────────────────────────────────────────────────────────
  [████████████████░░░░░░░░░░░░░░] 55.0%
  Steps:   27,500,000 / 50,000,000
  FPS:   52,400 fps   ETA: 43.0h remaining

📊 KEY METRICS (latest)
──────────────────────────────────────────────────────────────
  eval_reward_mean :       6820  ➡️  plateau   (Δ5M steps: +12.00)
  eval_fall_rate   :      8.2%   ➡️  plateau
  eval_per_step    :      6.820  (curriculum gate threshold ~7.0 for balance)
  eval points      : 27

🎓 CURRICULUM
──────────────────────────────────────────────────────────────
  Phase: Phase C
  Level: 24/29  [███████████████████████░░░░░░] 82%
  Min height: 0.44m
  Curriculum evals fired: 63
  Recent: L21@18.2M → L22@21.3M → L23@24.1M → L24@27.0M

⚠️  ISSUES DETECTED
──────────────────────────────────────────────────────────────
  ⚠️  [PLATEAU] Reward plateau (Δ=+12.00 trong 25M steps)
     eval_reward không tăng đáng kể trong 10 eval points cuối.
       Latest: 6820.00 | First in window: 6808.00 | Δ=+12.00 (0.2%)
     → Xem xét: tăng entropy_coeff, điều chỉnh reward weights, hoặc resume từ checkpoint khác.

ℹ️  INFO
──────────────────────────────────────────────────────────────
  ℹ️  Curriculum advancing: level 18 → 24: min_height: 0.44m

⚡ RECOMMENDED NEXT STEPS
──────────────────────────────────────────────────────────────
  ⚠️  Có vấn đề cần xử lý — xem config-advisor:
     symptom: reward_plateau_late
     Gọi config-advisor skill với symptom_key='reward_plateau_late'

──────────────────────────────────────────────────────────────
```

---

## Quy tắc của agent

1. **Không cần eval_results.json** — chỉ cần JSONL, hoạt động ngay trong khi đang train.
2. **CRITICAL issues = dừng training** — luôn output "DỪNG" rõ ràng, không ambiguous.
3. **Không tự chạy bất kỳ script nào** — chỉ phân tích và sinh commands.
4. **Curriculum health là ưu tiên** — nếu curriculum stuck, report ngay dù reward trend OK.
5. **Threshold reference luôn được ghi rõ** — không để user đoán eval_per_step=7.0 là tốt hay xấu.
6. **ETA chỉ là ước tính** — dựa trên FPS hiện tại, không tính eval overhead.
7. **Sau CRITICAL alert** — luôn suggest validate_checkpoint trước khi resume.