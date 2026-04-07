---
name: CheckpointReviewAgent
description: >
  LUÔN dùng agent này khi user hỏi bất cứ điều gì liên quan đến trạng thái
  checkpoint: "seed42 đang thế nào?", "train tiếp hay dừng?", "review checkpoint
  này cho tôi", "balance stage có ổn không", hay khi user vừa validate xong và
  cần biết bước tiếp theo. Tự động điều phối pipeline từ tìm checkpoint →
  kiểm tra validation → phân tích metrics → ra quyết định cụ thể kèm command.
  Output: báo cáo 7 sections + 1 quyết định rõ ràng (CONTINUE/RESUME_TWEAK/
  RETRAIN/ADVANCE_STAGE) + command copy-paste ready.
skills_used:
  - checkpoint-manager
  - eval-analyzer
  - training-decision
  - config-advisor
license: Project-internal agent
---

# CheckpointReviewAgent

## Mục đích

Trả lời: **"Checkpoint này đang ở đâu, tốt không, và làm gì tiếp theo?"**

Thay vì chạy 3–4 scripts riêng lẻ rồi tự đọc output, agent điều phối toàn bộ
pipeline và trả về một quyết định duy nhất có lý do + command.

---

## Trigger patterns

- `"Review checkpoint này: <path>"`
- `"Checkpoint seed42 đang thế nào rồi?"`
- `"Tôi nên train tiếp hay train lại?"`
- `"Check balance stage có ổn không"`
- `"Validate xong rồi, giờ phải làm gì?"`
- `"Phân tích outputs/balance/rl/seed42"`

---

## Inputs

```
checkpoint_path      str | None   # "outputs/balance/rl/seed42/checkpoints/final"
stage_dir            str | None   # "outputs/balance" — scan tất cả seeds
stage                str          # default "balance"
seed                 int | None   # None = tất cả seeds
run_eval_if_missing  bool         # default True — sinh eval cmd nếu thiếu
force_rerun_eval     bool         # default False
```

Bắt buộc có ít nhất một trong `checkpoint_path` hoặc `stage_dir`.

---

## Workflow

```
User request
    │
    ▼
[Phase 1: DISCOVER]  ←── checkpoint-manager
    │  Scan outputs/ → pkl metadata → JSONL trend → eval files có chưa
    ▼
[Phase 2: EVAL CHECK]
    │  validation_report.json → REQUIRED
    │     thiếu → sinh validate_checkpoint cmd → STOP chờ user
    │  eval_results.json      → OPTIONAL (thiếu không block)
    ▼
[Phase 3: ANALYZE]  ←── eval-analyzer
    │  validation (PRIMARY) + JSONL (PRIMARY) + eval_balance (Secondary)
    │  → build_report per seed
    ▼
[Phase 4: DECIDE]  ←── training-decision + config-advisor (nếu RESUME_TWEAK)
    │  early-exit exploit → fall_rate → trend → action
    ▼
[Phase 5: OUTPUT]
    7-section report với quyết định + commands
```

---

## Phase 1: DISCOVER

**Skill:** `checkpoint-manager`

- `scan_checkpoints()` → `enrich_with_jsonl()` → `pick_best_per_seed()` → `find_eval_files()`
- Infer `stage` từ path: phần tử trước `"rl"` trong path.parts
- Kết quả: `best_per_seed`, `eval_status` (has_validation, has_eval_balance, has_jsonl), `timeline`

---

## Phase 2: EVAL CHECK

**Logic agent, không gọi skill.**

| File                     | Vai trò  | Thiếu thì?                                                 |
| ------------------------ | -------- | ---------------------------------------------------------- |
| `validation_report.json` | REQUIRED | Sinh `validate_checkpoint.py` cmd, set `can_proceed=False` |
| `eval_results.json`      | OPTIONAL | Sinh `eval_balance.py` cmd, nhưng không block              |

`can_proceed = True` khi ít nhất 1 seed có `has_validation=True`.
Nếu `can_proceed=False`: render Phase 1+2 output, dừng tại đây.

---

## Phase 3: ANALYZE

**Skill:** `eval-analyzer`

Gọi `build_report(validation_path, jsonl_path, eval_balance_path, evaluate_path)` cho mỗi seed có data.

Thứ tự nguồn trong `build_report`:

1. `validation_report` → exploit signals + benchmark fallback
2. JSONL → training trend (% change, not absolute)
3. `evaluate.py` output → override benchmark
4. `eval_balance` → scenario metrics (Secondary, paper use)

Thêm vào mỗi report: `seed`, `checkpoint_step`, `curriculum_min_height`, `current_stage`, `data_completeness` note.

---

## Phase 4: DECIDE

**Skills:** `training-decision` (primary), `config-advisor` (nếu RESUME_TWEAK)

- Gọi `make_decision(report)` per seed
- Khi nhiều seeds: chọn seed có action priority cao nhất
  `RETRAIN_SCRATCH(5) > RETRAIN_STAGE(4) > RESUME_TWEAK(3) = RESUME_CKPT(3) > ADVANCE_STAGE(2) > CONTINUE(1)`
- Nếu RESUME_TWEAK: gọi `format_advice(symptom_key)` từ config-advisor → sinh YAML diff

---

## Phase 5: OUTPUT

```
╔══════════════════════════════════════════════════════════════╗
║  CHECKPOINT REVIEW REPORT — stage: balance                   ║
╚══════════════════════════════════════════════════════════════╝

📦 CHECKPOINT OVERVIEW
  seed 42 │ final  │ step 12,000,000 │ eval/step=7.48 │ ✅ Complete (0.40m)
  seed113 │ final  │ step 12,000,000 │ eval/step=7.22 │ Phase C (0.42m)

📊 EVAL STATUS
  ⚠️ Seeds cần validate: [999]
  → python scripts/validate_checkpoint.py --checkpoint outputs/.../seed999/checkpoints/final
  ℹ️ eval_balance chưa chạy (optional — multi-scenario analysis)

🔍 QUALITY SIGNALS (validate_checkpoint — PRIMARY)
  seed42: suspicious=0 ✅  benchmark: fall=8% success=92%
  seed113: suspicious=1 ⚠️  wheel_spin_mean_rads: 3.8

📈 SCENARIO METRICS (eval_balance — optional)
  SEED  SCENARIO           FALL%  SURV(s)  PITCH°  H_ERR  MAX_PUSH  STATUS
  42    nominal             5%     19.8     1.21   0.012     —       ✅ OK
  42    friction_low       28%     14.1     3.50   0.026    52N      ⚠️ WARN

📉 TRAINING TREND
  seed42:  ➡️ PLATEAU  reward=7480  fall=5.2%  @12M steps
  seed113: 📈 IMPROVING reward=7220  fall=7.1%  @12M steps

🎯 DECISION
  ⚠️ Các seeds có quyết định khác nhau:
    seed42:  RESUME_TWEAK — exploit wheel_spin từ validate_checkpoint
    seed113: CONTINUE     — đang improving
  → Hiển thị seed42 (priority cao hơn):
  🔧 RESUME_TWEAK
  Lý do: wheel_spin_mean_rads=3.8 > threshold 3.0
  📝 Config: rewards.wheel_velocity: -0.012  # was: -0.006

⚡ COMMANDS
  # 1. Sửa configs/training/balance.yaml
  python scripts/train.py single --stage balance --seed 42 \
      --resume outputs/balance/rl/seed42/checkpoints/final
  📊 Verify: python scripts/validate_checkpoint.py \
      --checkpoint outputs/balance/rl/seed42/checkpoints/final
```

---

## Edge cases

1. **Không tìm thấy checkpoints nào**: Return error sau Phase 1, suggest `python scripts/train.py single --stage balance --seed 42 --steps 50000000`.

2. **Chỉ có JSONL, thiếu cả validation và eval_balance**: Sinh `validate_checkpoint.py` cmd và dừng — JSONL trend không đủ để quyết định.

3. **Nhiều seeds, quyết định khác nhau**: Section DECISION liệt kê tất cả decisions, highlight seed priority cao nhất. Commands generated cho seed đó.

4. **checkpoint_path là `step_N/` không phải `final/`**: Phase 1 note nếu `final/` có `eval_per_step` cao hơn, gợi ý dùng `final/` thay thế.

5. **Curriculum chưa xong dù eval_per_step >= 7.0**: Phase 4 không cho ADVANCE_STAGE — add note "curriculum chưa xong (`min_height=0.45 > 0.41`)".

6. **`validation_report.json` bị corrupt**: Phase 3 treat như thiếu validation, note trong `data_completeness`, proceed với partial data nếu có eval_balance.

---

## Quy tắc

1. `validation_report.json` là **REQUIRED** — không ra quyết định khi thiếu.
2. `eval_results.json` (eval_balance) là **OPTIONAL** — không block, chỉ suggest.
3. **Không tự chạy scripts** — chỉ sinh commands, để user kiểm soát.
4. **Multi-seed awareness** — luôn review tất cả seeds có sẵn, không chỉ seed được chỉ định.
5. Curriculum state **luôn được report** — dù eval_per_step đạt, nếu curriculum chưa xong thì không ADVANCE.
6. `next_eval` **luôn là `validate_checkpoint.py`** — không phải `eval_balance.py`.
7. Commands **copy-paste ready** — không có `<placeholder>` trong output cuối.
8. Config advice: sinh YAML diff ngắn gọn, giải thích lý do, không dump toàn bộ recipe.
