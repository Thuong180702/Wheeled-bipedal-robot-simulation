---
name: checkpoint-manager
description: >
  LUÔN dùng skill này khi user hỏi "checkpoint nào tốt nhất?", "tôi có những
  checkpoint nào?", "resume từ đâu?", "seed nào đang ở step bao nhiêu?", hay
  cần tìm checkpoint cụ thể trong wheeled biped project. Biết cấu trúc
  outputs/STAGE/rl/seedN/checkpoints/, đọc checkpoint.pkl metadata không load
  params vào RAM, kết hợp JSONL log để lấy metrics theo step. Output: bảng
  checkpoint x metrics, gợi ý checkpoint tốt nhất để resume/eval/paper.
  Dùng trước eval-analyzer khi chưa biết checkpoint nào cần review.
license: Project-internal skill
---

# Checkpoint Manager Skill

## Tổng quan

Skill này giải quyết các câu hỏi:

- "Tôi có những checkpoint nào?"
- "Checkpoint nào tốt nhất để resume?"
- "Checkpoint nào nên dùng cho paper?"
- "Step N có checkpoint không, hay phải quay về step nào gần nhất?"
- "Seed nào trong 3 seeds đang có kết quả tốt nhất?"

---

## Bước 1 — Hiểu cấu trúc thư mục

### Layout chuẩn của project

```
outputs/
└── <stage>/                         # balance | balance_robust | stand_up | ...
    ├── rl/
    │   ├── seed42/
    │   │   ├── checkpoints/
    │   │   │   ├── step_1000000/    # checkpoint tại step 1M
    │   │   │   │   └── checkpoint.pkl
    │   │   │   ├── step_5000000/
    │   │   │   │   └── checkpoint.pkl
    │   │   │   ├── step_10000000/
    │   │   │   │   └── checkpoint.pkl
    │   │   │   └── final/           # checkpoint cuối hoặc tốt nhất
    │   │   │       └── checkpoint.pkl
    │   │   ├── balance_seed42_metrics.jsonl   # training log
    │   │   └── run_metadata.json
    │   ├── seed113/
    │   │   └── ...
    │   ├── seed999/
    │   │   └── ...
    │   └── paper_eval/              # kết quả eval tổng hợp 3 seeds
    └── lqr/
        └── eval_results.json        # LQR baseline results
```

### Scan toàn bộ checkpoints

```python
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class CheckpointInfo:
    path: str
    stage: str
    seed: int
    step: int            # global_step trong checkpoint.pkl
    label: str           # "step_1000000" hoặc "final"
    best_reward: float
    eval_per_step: Optional[float]    # best_eval_per_step nếu có
    eval_success: Optional[float]     # best_eval_success nếu có
    curriculum_min_height: Optional[float]
    curriculum_level: Optional[int]
    has_eval_results: bool            # có eval_results.json bên cạnh không
    has_validation: bool              # có validation_report.json không

def scan_checkpoints(outputs_dir: str = "outputs",
                     stage: str = None,
                     seeds: list[int] = None) -> list[CheckpointInfo]:
    """Quét toàn bộ checkpoints trong outputs/."""
    base = Path(outputs_dir)
    infos = []

    # Tìm tất cả stage dirs
    stage_dirs = [base / stage] if stage else [d for d in base.iterdir() if d.is_dir()]

    for stage_dir in stage_dirs:
        rl_dir = stage_dir / "rl"
        if not rl_dir.exists():
            continue

        stage_name = stage_dir.name

        # Tìm seed dirs
        for seed_dir in rl_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed"):
                continue

            try:
                seed_num = int(seed_dir.name.replace("seed", ""))
            except ValueError:
                continue

            if seeds and seed_num not in seeds:
                continue

            ckpt_dir = seed_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue

            for ckpt_subdir in sorted(ckpt_dir.iterdir()):
                pkl_path = ckpt_subdir / "checkpoint.pkl"
                if not pkl_path.exists():
                    continue

                try:
                    meta = _read_checkpoint_meta(str(pkl_path))
                except Exception as e:
                    print(f"  [skip] {pkl_path}: {e}")
                    continue

                infos.append(CheckpointInfo(
                    path=str(ckpt_subdir),
                    stage=stage_name,
                    seed=seed_num,
                    step=meta.get("global_step", 0),
                    label=ckpt_subdir.name,
                    best_reward=meta.get("best_reward", float("nan")),
                    eval_per_step=meta.get("best_eval_per_step"),
                    eval_success=meta.get("best_eval_success"),
                    curriculum_min_height=meta.get("curriculum_min_height"),
                    curriculum_level=None,   # không lưu trong pkl, đọc từ JSONL
                    has_eval_results=(ckpt_subdir / "eval_results.json").exists(),
                    has_validation=(ckpt_subdir / "validation_report.json").exists(),
                ))

    return sorted(infos, key=lambda x: (x.stage, x.seed, x.step))


def _read_checkpoint_meta(pkl_path: str) -> dict:
    """Đọc metadata từ checkpoint.pkl mà không load params (tiết kiệm RAM)."""
    with open(pkl_path, "rb") as f:
        ckpt = pickle.load(f)
    # Trả về tất cả keys trừ params/opt_state/obs_rms (nặng)
    return {k: v for k, v in ckpt.items()
            if k not in ("params", "opt_state", "obs_rms")}
```

---

## Bước 2 — Kết hợp với JSONL log

Checkpoint.pkl lưu `best_reward`, `best_eval_per_step`, `best_eval_success` tại thời điểm save.
Để xem **tất cả** metrics theo thời gian, đọc thêm JSONL:

```python
def enrich_with_jsonl(infos: list[CheckpointInfo],
                      outputs_dir: str = "outputs") -> dict:
    """
    Trả về dict: (stage, seed) → list of (step, metrics) từ JSONL log.
    Dùng để tìm metrics tại step gần nhất với mỗi checkpoint.
    """
    timeline = {}
    base = Path(outputs_dir)

    for info in infos:
        key = (info.stage, info.seed)
        if key in timeline:
            continue

        # Tìm JSONL file
        seed_dir = base / info.stage / "rl" / f"seed{info.seed}"
        jsonl_files = list(seed_dir.glob("*_metrics.jsonl"))
        if not jsonl_files:
            continue

        records = []
        with open(jsonl_files[0]) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "step" in r:
                        records.append(r)
                except json.JSONDecodeError:
                    continue

        timeline[key] = sorted(records, key=lambda r: r["step"])

    return timeline


def get_metrics_at_step(timeline: dict, stage: str, seed: int,
                         target_step: int) -> dict:
    """Lấy metrics JSONL gần nhất với target_step."""
    key = (stage, seed)
    records = timeline.get(key, [])
    if not records:
        return {}

    # Tìm record gần nhất
    closest = min(records, key=lambda r: abs(r["step"] - target_step))
    return {
        "step": closest["step"],
        "eval_reward_mean": closest.get("eval/reward_mean"),
        "eval_fall_rate": closest.get("eval/fall_rate"),
        "eval_success_rate": closest.get("eval/success_rate"),
        "curriculum_level": closest.get("curriculum/level"),
        "curriculum_eval_per_step": closest.get("curriculum/eval_per_step"),
        "train_reward": closest.get("train/reward_mean_recent"),
    }
```

---

## Bước 3 — Chọn checkpoint tốt nhất

### Tiêu chí lựa chọn

```python
# Thứ tự ưu tiên (từ cao xuống thấp):
SELECTION_CRITERIA = [
    # 1. best_eval_per_step (saved trong pkl) — trustworthy nhất
    #    Vì được tính từ held-out eval_pass(), không phải training reward
    "eval_per_step",
    # 2. best_eval_success (% episodes không fall)
    "eval_success",
    # 3. best_reward (rolling training reward) — noisy hơn
    "best_reward",
]

def pick_best_checkpoint(infos: list[CheckpointInfo],
                          stage: str = None,
                          seed: int = None,
                          prefer_final: bool = True) -> CheckpointInfo:
    """
    Chọn checkpoint tốt nhất theo tiêu chí.
    - prefer_final=True: nếu 'final' tồn tại và tốt, ưu tiên nó.
    """
    candidates = [i for i in infos
                  if (stage is None or i.stage == stage)
                  and (seed is None or i.seed == seed)]

    if not candidates:
        raise ValueError(f"Không tìm thấy checkpoint cho stage={stage}, seed={seed}")

    # Ưu tiên final nếu prefer_final
    if prefer_final:
        finals = [c for c in candidates if c.label == "final"]
        if finals:
            return finals[0]

    # Sắp xếp theo eval_per_step → eval_success → best_reward
    def sort_key(c: CheckpointInfo):
        return (
            c.eval_per_step or float("-inf"),
            c.eval_success or float("-inf"),
            c.best_reward,
        )

    return max(candidates, key=sort_key)


def pick_best_per_seed(infos: list[CheckpointInfo],
                        stage: str) -> dict[int, CheckpointInfo]:
    """Chọn best checkpoint cho từng seed của một stage."""
    seeds = set(i.seed for i in infos if i.stage == stage)
    return {
        seed: pick_best_checkpoint(infos, stage=stage, seed=seed)
        for seed in sorted(seeds)
    }
```

---

## Bước 4 — Format báo cáo

### Bảng tổng quan checkpoints

```
=== CHECKPOINT MANAGER REPORT ===
Stage: balance  |  Found: 9 checkpoints across 3 seeds

SEED   LABEL          STEP      EVAL/STEP  EVAL_SUCC  FALL%   CURR_H  EVAL?  VALID?
──────────────────────────────────────────────────────────────────────────────────────
42     step_1000000   1,000,000   4.21       62%       28%     0.69    —      —
42     step_5000000   5,000,000   6.15       81%       12%     0.62    —      —
42     step_10000000 10,000,000   7.31       91%        6%     0.51    ✓      —
42  ★  final         12,000,000   7.48       93%        5%     0.40    ✓      ✓
──────
113    step_5000000   5,000,000   5.89       78%       15%     0.65    —      —
113 ★  final         12,000,000   7.22       90%        7%     0.42    ✓      —
──────
999    step_5000000   5,000,000   6.01       80%       14%     0.63    —      —
999 ★  final         12,000,000   7.35       91%        6%     0.41    ✓      —

★ = best checkpoint per seed

RECOMMENDED FOR PAPER: seed42/final
  eval_per_step=7.48 (threshold=7.0 ✅), eval_success=93%, fall_rate=5%
  curriculum_min_height=0.40 (full range completed ✅)

RECOMMENDED FOR RESUME: seed113/final
  Reason: seed113 vẫn cần thêm steps (eval_per_step=7.22 < 7.48 của seed42)
```

### Python: generate report

```python
def format_checkpoint_report(infos: list[CheckpointInfo],
                               timeline: dict,
                               stage: str,
                               success_value: float = 7.0) -> str:
    stage_infos = [i for i in infos if i.stage == stage]
    if not stage_infos:
        return f"Không tìm thấy checkpoints cho stage '{stage}'"

    lines = [
        f"=== CHECKPOINT MANAGER REPORT ===",
        f"Stage: {stage}  |  Found: {len(stage_infos)} checkpoints",
        "",
        f"{'SEED':<6} {'LABEL':<16} {'STEP':>12}  {'EVAL/STEP':>9}  "
        f"{'SUCC':>6}  {'FALL':>5}  {'CURR_H':>7}  {'EVAL':>5}  {'VALID':>6}",
        "─" * 88,
    ]

    best_per_seed = pick_best_per_seed(stage_infos, stage)

    seeds = sorted(set(i.seed for i in stage_infos))
    for seed in seeds:
        seed_infos = sorted([i for i in stage_infos if i.seed == seed],
                             key=lambda x: x.step)
        for info in seed_infos:
            is_best = best_per_seed.get(seed) == info
            star = "★" if is_best else " "

            # Enrich từ JSONL nếu eval_per_step không có trong pkl
            jsonl_metrics = get_metrics_at_step(timeline, stage, seed, info.step)
            ep = info.eval_per_step or jsonl_metrics.get("curriculum_eval_per_step")
            es = info.eval_success or jsonl_metrics.get("eval_success_rate")
            fr = jsonl_metrics.get("eval_fall_rate")

            ep_str = f"{ep:.2f}" if ep is not None else "—"
            es_str = f"{es*100:.0f}%" if es is not None else "—"
            fr_str = f"{fr*100:.0f}%" if fr is not None else "—"
            ch_str = f"{info.curriculum_min_height:.2f}" if info.curriculum_min_height else "—"
            ev_str = "✓" if info.has_eval_results else "—"
            vl_str = "✓" if info.has_validation else "—"

            lines.append(
                f"{seed:<4} {star}  {info.label:<14} {info.step:>12,}  "
                f"{ep_str:>9}  {es_str:>6}  {fr_str:>5}  {ch_str:>7}  "
                f"{ev_str:>5}  {vl_str:>6}"
            )
        lines.append("──────")

    # Recommendations
    lines.append("")
    lines.append("RECOMMENDATIONS:")

    # Paper: seed với eval_per_step cao nhất
    all_bests = list(best_per_seed.values())
    paper_ckpt = max(all_bests,
                     key=lambda x: (x.eval_per_step or 0, x.eval_success or 0))
    threshold_ok = "✅" if (paper_ckpt.eval_per_step or 0) >= success_value else "⚠️"
    full_range_ok = "✅" if (paper_ckpt.curriculum_min_height or 1.0) <= 0.41 else "⚠️"
    lines.append(f"  Paper: seed{paper_ckpt.seed}/{paper_ckpt.label}")
    lines.append(f"    eval_per_step={paper_ckpt.eval_per_step:.2f} (threshold={success_value}) {threshold_ok}")
    lines.append(f"    curriculum_min_height={paper_ckpt.curriculum_min_height} {full_range_ok}")

    return "\n".join(lines)
```

---

## Bước 5 — Các use cases thường gặp

### Use case 1: "Tôi muốn resume từ checkpoint tốt nhất"

```python
best = pick_best_checkpoint(infos, stage="balance", seed=42)
print(f"Resume command:")
print(f"  python scripts/train.py single --stage balance --seed 42 \\")
print(f"      --resume {best.path}")
```

### Use case 2: "Curriculum đang ở đâu rồi?"

```python
for info in infos:
    if info.label == "final" and info.stage == "balance":
        ch = info.curriculum_min_height
        if ch is not None:
            phase = "A (narrow)" if ch >= 0.65 else "B (moderate)" if ch >= 0.50 else "C (full range)"
            print(f"  seed{info.seed}: curriculum_min_height={ch:.2f} → Phase {phase}")
```

Mapping phases:

- `curriculum_min_height >= 0.65` → **Phase A** (levels 1–5): narrow band `[0.65, 0.70]`
- `0.50 <= curriculum_min_height < 0.65` → **Phase B** (levels 6–20): widening
- `curriculum_min_height < 0.50` → **Phase C** (levels 21–29): full range `[0.40, 0.70]`
- `curriculum_min_height = 0.40` → **Completed**: toàn bộ curriculum xong

### Use case 3: "Checkpoint ở step X có không?"

```python
def find_nearest_checkpoint(infos: list[CheckpointInfo],
                              stage: str, seed: int,
                              target_step: int) -> CheckpointInfo:
    candidates = [i for i in infos
                  if i.stage == stage and i.seed == seed]
    if not candidates:
        raise ValueError(f"Không có checkpoint nào cho seed{seed}")
    return min(candidates, key=lambda x: abs(x.step - target_step))
```

### Use case 4: "3 seeds có đủ để báo cáo paper chưa?"

```python
PAPER_SEEDS = [42, 113, 999]
PAPER_REQUIREMENTS = {
    "min_eval_per_step": 7.0,        # success_value trong curriculum.yaml
    "min_eval_success": 0.80,
    "max_fall_rate": 0.15,
    "curriculum_completed": 0.41,    # min_height <= 0.41 tức là đạt full range
    "has_eval_results": True,        # đã chạy eval_balance.py
}

def check_paper_readiness(infos: list[CheckpointInfo],
                           stage: str = "balance") -> dict:
    best = pick_best_per_seed(infos, stage)
    report = {"ready": True, "seeds_ready": [], "seeds_not_ready": [], "details": {}}

    for seed in PAPER_SEEDS:
        if seed not in best:
            report["seeds_not_ready"].append(seed)
            report["details"][seed] = "Không có checkpoint"
            report["ready"] = False
            continue

        ckpt = best[seed]
        issues = []

        ep = ckpt.eval_per_step or 0
        if ep < PAPER_REQUIREMENTS["min_eval_per_step"]:
            issues.append(f"eval_per_step={ep:.2f} < {PAPER_REQUIREMENTS['min_eval_per_step']}")

        ch = ckpt.curriculum_min_height or 1.0
        if ch > PAPER_REQUIREMENTS["curriculum_completed"]:
            issues.append(f"curriculum chưa xong (min_height={ch:.2f} > 0.41)")

        if not ckpt.has_eval_results:
            issues.append("chưa chạy eval_balance.py")

        if issues:
            report["seeds_not_ready"].append(seed)
            report["details"][seed] = "; ".join(issues)
            report["ready"] = False
        else:
            report["seeds_ready"].append(seed)
            report["details"][seed] = "OK"

    return report
```

---

## Quy tắc khi dùng skill này

1. **Luôn scan toàn bộ** trước khi nói "checkpoint X không tồn tại" — dùng `scan_checkpoints()`.
2. **Đọc pkl metadata nhẹ** — không load `params`/`opt_state` vào RAM khi chỉ cần so sánh.
3. **`eval_per_step` từ pkl > từ JSONL** — pkl lưu `best_eval_per_step` (value tốt nhất trong quá trình train), JSONL lưu value tại từng step.
4. **`final/` không nhất thiết là tốt nhất** — trainer lưu `final/` khi kết thúc, có thể là best hoặc chỉ là last. So sánh với `step_N/` khác.
5. **Không tự chạy eval** — nếu checkpoint chưa có `eval_results.json`, báo cáo thiếu và đề xuất command chạy eval, không tự phán xét chất lượng.
6. **Paper readiness check** cần cả 3 seeds (42, 113, 999) đều pass — không dùng 1 seed outlier.
