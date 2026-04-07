---
name: PaperSyncAgent
description: >
  Agent tự động đồng bộ kết quả training/eval vào paper/main.tex và README.md.
  Nhận eval_results.json → tìm tất cả placeholder [TODO] trong LaTeX → điền số
  liệu thực, cập nhật limitations đã được giải quyết, sync README status table.
  Output: diff trước khi apply + commands để compile/verify. Luôn preview, không
  bao giờ ghi file mà không có user confirmation.
  Orchestrates: paper-updater skill + readme-updater skill.
skills_used:
  - paper-updater
  - readme-updater
license: Project-internal agent
---

# PaperSyncAgent

## Mục đích

**"Có kết quả mới rồi — cần cập nhật những gì trong paper và README?"**

Agent chạy sau khi MultiSeedCompareAgent xác nhận paper readiness, hoặc bất cứ khi nào có eval_results.json mới. Output là diff cụ thể, không tự ghi file.

---

## Trigger patterns

- `"Cập nhật paper với kết quả mới"`
- `"Điền số liệu vào paper/main.tex"`
- `"Sync eval results vào paper"`
- `"Balance stage xong rồi, update README và paper"`
- `"TODO trong paper còn bao nhiêu?"`

---

## Inputs

```python
@dataclass
class PaperSyncInput:
    paper_eval_dir: str | None = None   # "outputs/balance/rl/paper_eval"
    rl_eval_path:   str | None = None   # path cụ thể tới eval_results.json (RL)
    lqr_eval_path:  str | None = None   # path tới LQR eval_results.json
    jsonl_paths: list[str] | None = None

    tex_path:    str = "paper/main.tex"
    readme_path: str = "README.md"
    stage:       str = "balance"
    n_seeds:     int = 3

    dry_run:       bool = True
    apply_readme:  bool = True
    apply_paper:   bool = True
```

---

## Workflow

```
Phase 1: AUDIT
  paper-updater: scan_placeholders()
  → tìm tất cả [TODO], TODO-RESULT, TODO-TABLE, ...
  → phân loại: fillable vs blocked (TODO-CLAIM)
  → báo cáo: còn bao nhiêu placeholder

Phase 2: DATA RESOLVE
  → tìm và load eval_results.json (RL + LQR)
  → extract total_steps từ JSONL nếu có
  → xác định n_seeds thực tế trong data
  → kiểm tra data completeness

Phase 3: PAPER UPDATE  (paper-updater skill)
  → fill_rl_vs_lqr_table()       — bảng RL vs LQR chính
  → fill_balance_results_table() — per-height table
  → update_training_steps()      — điền total steps
  → resolve_limitation()         — xóa limitations đã xong
  → preview_diff()

Phase 4: README UPDATE  (readme-updater skill)
  → update_overview_table()      — status các tasks
  → update_status_badge()        — dòng status badge
  → update_three_seed_protocol() — seed status table
  → preview_diff()

Phase 5: OUTPUT & CONFIRM
  → Summary: N placeholders filled, M remain
  → Diff preview (paper + README, capped 60 lines)
  → Remaining TODOs list
  → Commands: apply + pdflatex compile check
```

---

## Phase 1: AUDIT

```python
from pathlib import Path
import re

@dataclass
class PlaceholderAudit:
    total: int
    fillable: int          # có thể điền tự động nếu có data
    blocked: int           # TODO-CLAIM, TODO-ABLATION — không tự điền
    by_kind: dict          # kind → count
    details: list[dict]

def audit_placeholders(tex_path: str) -> PlaceholderAudit:
    """Scan paper/main.tex và phân loại placeholders."""
    # Gọi scan_placeholders() từ paper-updater skill
    placeholders = scan_placeholders(tex_path)

    FILLABLE  = {"TODO", "TODO-RESULT", "TODO-TABLE"}
    BLOCKED   = {"TODO-CLAIM", "TODO-ABLATION"}

    by_kind = {}
    for p in placeholders:
        by_kind[p.kind] = by_kind.get(p.kind, 0) + 1

    fillable = sum(c for k, c in by_kind.items() if k in FILLABLE)
    blocked  = sum(c for k, c in by_kind.items() if k in BLOCKED)

    return PlaceholderAudit(
        total=sum(by_kind.values()),
        fillable=fillable,
        blocked=blocked,
        by_kind=by_kind,
        details=[{
            "kind":    p.kind,
            "line":    p.line_no,
            "desc":    p.description[:60],
            "inline":  p.is_inline,
            "blocked": p.kind in BLOCKED,
        } for p in placeholders],
    )
```

---

## Phase 2: DATA RESOLVE

```python
import json

@dataclass
class DataSources:
    rl_results:         dict | None
    lqr_results:        dict | None
    total_steps:        int | None
    n_seeds_actual:     int
    per_height_metrics: dict         # {0.40: metrics_dict, ...}
    missing:            list[str]

def resolve_data_sources(inp: PaperSyncInput) -> DataSources:
    rl_results  = None
    lqr_results = None
    missing     = []

    # ── RL eval results (auto-discover) ──────────────────────────────────────
    rl_candidates = []
    if inp.rl_eval_path:
        rl_candidates.append(inp.rl_eval_path)
    if inp.paper_eval_dir:
        rl_candidates.append(f"{inp.paper_eval_dir}/eval_results.json")
    rl_candidates.append(f"outputs/{inp.stage}/rl/paper_eval/eval_results.json")

    for path in rl_candidates:
        if Path(path).exists():
            with open(path) as f:
                rl_results = json.load(f)
            break
    if not rl_results:
        missing.append("RL eval_results.json — chạy 3-seed eval_balance.py trước")

    # ── LQR eval results (optional) ──────────────────────────────────────────
    lqr_candidates = []
    if inp.lqr_eval_path:
        lqr_candidates.append(inp.lqr_eval_path)
    lqr_candidates.append(f"outputs/{inp.stage}/lqr/eval_results.json")

    for path in lqr_candidates:
        if Path(path).exists():
            with open(path) as f:
                lqr_results = json.load(f)
            break

    # ── Total steps từ JSONL (auto-discover) ─────────────────────────────────
    total_steps = None
    for seed in [42, 113, 999]:
        jpath = f"outputs/{inp.stage}/rl/seed{seed}/{inp.stage}_seed{seed}_metrics.jsonl"
        if Path(jpath).exists():
            # Đọc step cuối cùng từ JSONL
            last_step = 0
            with open(jpath) as f:
                for line in f:
                    try:
                        r = json.loads(line.strip())
                        if r.get("step", 0) > last_step:
                            last_step = r["step"]
                    except Exception:
                        pass
            if last_step > (total_steps or 0):
                total_steps = last_step

    # ── n_seeds thực tế ───────────────────────────────────────────────────────
    n_seeds_actual = 1
    if rl_results:
        checkpoints = rl_results.get("checkpoints", [])
        n_seeds_actual = len(checkpoints) if checkpoints else inp.n_seeds

    # ── Per-height metrics cho tab:balance_results ────────────────────────────
    per_height = {}
    if rl_results:
        HEIGHT_MAP = {"narrow_height": 0.69, "nominal": 0.65, "wide_height": 0.60}
        for r in rl_results.get("results", []):
            h = HEIGHT_MAP.get(r.get("scenario", ""))
            if h:
                per_height[h] = r

    return DataSources(
        rl_results=rl_results,
        lqr_results=lqr_results,
        total_steps=total_steps,
        n_seeds_actual=n_seeds_actual,
        per_height_metrics=per_height,
        missing=missing,
    )
```

---

## Phase 3: PAPER UPDATE

```python
@dataclass
class PaperUpdateResult:
    original_tex:           str
    updated_tex:            str
    diff:                   str
    changes_made:           list[str]
    placeholders_filled:    int
    placeholders_remaining: int
    skipped:                list[str]

def build_paper_update(tex_path: str,
                        data: DataSources,
                        n_seeds: int) -> PaperUpdateResult:
    """Apply all fillable updates to paper/main.tex."""
    original = Path(tex_path).read_text(encoding="utf-8")
    current  = original
    changes  = []
    skipped  = []

    # 1. tab:rl_vs_lqr — RL rows
    if data.rl_results:
        before  = current
        current = fill_rl_vs_lqr_table(current, data.rl_results, n_seeds=n_seeds)
        if current != before:
            n = before.count("[TODO]") - current.count("[TODO]")
            changes.append(f"tab:rl_vs_lqr RL rows: {n} cells filled")
    else:
        skipped.append("tab:rl_vs_lqr RL: thiếu rl_results")

    # 2. tab:rl_vs_lqr — LQR rows (optional)
    if data.lqr_results:
        before  = current
        current = fill_rl_vs_lqr_table(current, data.lqr_results, n_seeds=1)
        if current != before:
            n = before.count("[TODO]") - current.count("[TODO]")
            changes.append(f"tab:rl_vs_lqr LQR rows: {n} cells filled")

    # 3. tab:balance_results — per-height
    if data.per_height_metrics:
        before  = current
        current = fill_balance_results_table(current, data.per_height_metrics, n_seeds=n_seeds)
        if current != before:
            n = before.count("[TODO]") - current.count("[TODO]")
            changes.append(f"tab:balance_results: {n} cells filled")
    else:
        skipped.append("tab:balance_results: thiếu per-height metrics")

    # 4. Training steps
    if data.total_steps:
        before  = current
        current = update_training_steps(current, data.total_steps)
        if current != before:
            changes.append(f"Training steps: {data.total_steps:,}")
    else:
        skipped.append("Training steps: thiếu JSONL data")

    # 5. Limitations — chỉ resolve khi đủ 3 seeds
    if data.rl_results and data.n_seeds_actual >= 3:
        before  = current
        current = resolve_limitation(current, "no_results")
        current = resolve_limitation(current, "multi_seed_done")
        if current != before:
            changes.append("Limitations: no_results + multi_seed_done resolved")
    elif data.rl_results:
        skipped.append(
            f"Limitations: chỉ {data.n_seeds_actual}/3 seeds — chưa resolve multi_seed_done"
        )

    # 6. Remove inline textit TODOs trong sections đã có content
    before  = current
    current = remove_inline_todo(current, "sec:results")
    if current != before:
        changes.append("Inline TODO placeholders removed")

    # Stats
    orig_count  = len(re.findall(r'\[TODO\]', original))
    final_count = len(re.findall(r'\[TODO\]', current))
    filled      = max(0, orig_count - final_count)

    all_todo_remaining = len(re.findall(r'TODO', current))

    import difflib
    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="main.tex (current)",
        tofile="main.tex (proposed)",
        n=3,
    ))

    return PaperUpdateResult(
        original_tex=original,
        updated_tex=current,
        diff=diff,
        changes_made=changes,
        placeholders_filled=filled,
        placeholders_remaining=all_todo_remaining,
        skipped=skipped,
    )
```

---

## Phase 4: README UPDATE

```python
import difflib

@dataclass
class ReadmeUpdateResult:
    original_readme: str
    updated_readme:  str
    diff:            str
    changes_made:    list[str]

def build_readme_update(readme_path: str,
                         stage: str,
                         data: DataSources,
                         n_seeds: int) -> ReadmeUpdateResult:
    original = Path(readme_path).read_text(encoding="utf-8")
    current  = original
    changes  = []

    # 1. Overview table status
    if data.rl_results:
        new_status = "Paper ready" if n_seeds >= 3 and not data.missing else "Evaluated"
        # readme-updater skill: update_overview_table()
        before  = current
        pattern = rf'(\|\s*\*\*{stage.replace("_", " ").title()}\*\*[^|]*\|[^|]*\|)\s*\w[^|]*\|'
        current = re.sub(pattern, rf'\g<1> {new_status} |', current, flags=re.IGNORECASE)
        if current != before:
            changes.append(f"Overview table: {stage} → {new_status}")

    # 2. Status badge
    badge_text = (
        '> **Status:** Active research prototype. `balance` stage fully trained and evaluated\n'
        '> over 3 seeds. Results reported in paper. Sim-to-real transfer not yet validated.'
        if n_seeds >= 3 and not data.missing else
        '> **Status:** Active research prototype. `balance` stage trained and evaluated.\n'
        '> Sim-to-real transfer has not been validated on hardware.'
    )
    before = current
    current = re.sub(r'>\s*\*\*Status:\*\*[^\n]*(?:\n>[^\n]*)*', badge_text, current)
    if current != before:
        changes.append("Status badge updated")

    # 3. Seed status table (trong 3-seed protocol section)
    if data.rl_results:
        checkpoints = data.rl_results.get("checkpoints", [])
        steps_str   = f"{data.total_steps//1_000_000}M" if data.total_steps else "?"
        table_rows  = ["| Seed | Status |", "|------|--------|"]
        for seed in [42, 113, 999]:
            in_eval = any(f"seed{seed}" in str(c) for c in checkpoints)
            status  = f"✅ Trained & evaluated ({steps_str} steps)" if in_eval else "⏸ Not in eval set"
            table_rows.append(f"| {seed} | {status} |")
        new_table = "\n".join(table_rows)
        before = current
        current = re.sub(
            r'\| Seed \| Status \|.*?(?=\nEach seed|\n\n---)',
            new_table,
            current,
            flags=re.DOTALL,
        )
        if current != before:
            changes.append("3-seed protocol: seed status table updated")

    diff = "".join(difflib.unified_diff(
        original.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="README.md (current)",
        tofile="README.md (proposed)",
        n=3,
    ))
    return ReadmeUpdateResult(
        original_readme=original,
        updated_readme=current,
        diff=diff,
        changes_made=changes,
    )
```

---

## Phase 5: OUTPUT

```python
def format_sync_output(audit: PlaceholderAudit,
                        data: DataSources,
                        paper: PaperUpdateResult,
                        readme: ReadmeUpdateResult,
                        inp: PaperSyncInput) -> str:
    lines = []
    lines += [
        "╔══════════════════════════════════════════════════════════════╗",
        "║  PAPER SYNC AGENT REPORT                                     ║",
        f"║  Mode: {'DRY RUN (preview only)' if inp.dry_run else 'APPLY CHANGES':<43}║",
        "╚══════════════════════════════════════════════════════════════╝", "",
    ]

    # Section 1: Placeholder audit
    lines += ["📋 PLACEHOLDER AUDIT", "─" * 62]
    for kind, count in sorted(audit.by_kind.items()):
        lock = " 🔒" if kind in ("TODO-CLAIM", "TODO-ABLATION") else ""
        lines.append(f"  [{kind}] × {count}{lock}")
    lines += ["", f"  Fillable: {audit.fillable} | Blocked: {audit.blocked} | Total: {audit.total}", ""]

    # Section 2: Data sources
    lines += ["📁 DATA SOURCES", "─" * 62]
    lines.append(f"  RL eval_results : {'✅ ' + str(data.n_seeds_actual) + ' seeds' if data.rl_results else '❌ not found'}")
    lines.append(f"  LQR eval_results: {'✅ found' if data.lqr_results else '⚠️  not found (optional)'}")
    lines.append(f"  Total steps     : {'✅ ' + f'{data.total_steps:,}' if data.total_steps else '⚠️  not found'}")
    if data.missing:
        lines += ["", "  ❌ Missing:"]
        for m in data.missing:
            lines.append(f"    • {m}")
    lines.append("")

    # Section 3: Paper changes
    lines += ["📝 PAPER CHANGES (paper/main.tex)", "─" * 62]
    if paper.changes_made:
        lines.append(f"  ✅ {paper.placeholders_filled} placeholder(s) filled:")
        for c in paper.changes_made:
            lines.append(f"    • {c}")
    else:
        lines.append("  ℹ️  Không có thay đổi nào được áp dụng")
    if paper.skipped:
        lines.append(f"  ⏭  Skipped ({len(paper.skipped)}):")
        for s in paper.skipped[:4]:
            lines.append(f"    • {s}")
    lines += ["", f"  Remaining TODOs: ~{paper.placeholders_remaining}", ""]

    # Section 4: README changes
    lines += ["📄 README CHANGES", "─" * 62]
    if readme.changes_made:
        for c in readme.changes_made:
            lines.append(f"  ✅ {c}")
    else:
        lines.append("  ℹ️  Không có thay đổi")
    lines.append("")

    # Section 5: Diff preview (capped 60 lines)
    if paper.diff:
        lines += ["🔍 PAPER DIFF PREVIEW (first 60 lines)", "─" * 62]
        diff_lines = paper.diff.splitlines()
        for dl in diff_lines[:60]:
            lines.append(f"  {dl}")
        if len(diff_lines) > 60:
            lines.append(f"  ... ({len(diff_lines) - 60} more lines)")
        lines.append("")

    # Section 6: Remaining blocked TODOs
    blocked = [d for d in audit.details if d["blocked"]]
    if blocked:
        lines += ["🔒 REMAINING BLOCKED TODOs (cần evidence)", "─" * 62]
        for d in blocked[:6]:
            lines.append(f"  L{d['line']}: {d['kind']} — {d['desc']}")
        lines.append("")

    # Section 7: Commands
    lines += ["⚡ COMMANDS", "─" * 62]
    if inp.dry_run:
        lines += ["  ℹ️  DRY RUN. Để apply, gọi lại với dry_run=False", ""]
    else:
        lines += ["  ✅ Changes applied.", ""]

    lines += [
        "  # Compile paper để verify không có lỗi LaTeX:",
        "  cd paper && pdflatex -interaction=nonstopmode main.tex",
        "",
        f"  # Export LaTeX table (nếu chưa):",
        f"  python scripts/export_results.py latex \\",
        f"      outputs/{inp.stage}/rl/paper_eval/eval_results.json \\",
        f"      --output outputs/tables/{inp.stage}_eval.tex",
        "",
        "─" * 62,
    ]
    return "\n".join(lines)
```

---

## Entrypoint

```python
def run(inp: PaperSyncInput) -> str:
    print("📋 Phase 1: Auditing placeholders...")
    audit = audit_placeholders(inp.tex_path)

    print("📁 Phase 2: Resolving data sources...")
    data = resolve_data_sources(inp)

    print("📝 Phase 3: Building paper update...")
    paper_result = build_paper_update(inp.tex_path, data, inp.n_seeds)

    print("📄 Phase 4: Building README update...")
    readme_result = build_readme_update(inp.readme_path, inp.stage, data, inp.n_seeds)

    output = format_sync_output(audit, data, paper_result, readme_result, inp)

    if not inp.dry_run:
        if paper_result.changes_made and inp.apply_paper:
            apply_update(inp.tex_path, paper_result.updated_tex)
        if readme_result.changes_made and inp.apply_readme:
            apply_update(inp.readme_path, readme_result.updated_readme)

    return output
```

---

## Quy tắc của agent

1. **TODO-CLAIM và TODO-ABLATION không bao giờ tự điền** — hiển thị rõ trong audit, giải thích lý do.
2. **Dry run mặc định** — không ghi file cho đến khi user confirm diff là OK.
3. **Không merge partial data** — chỉ có 1–2 seeds thì ghi số với note "(preliminary, N=X)".
4. **LQR missing không block** — LQR optional; RL rows vẫn được điền.
5. **Backup trước khi apply** — paper-updater skill tự tạo `main.tex.bak`.
6. **Sau khi apply, suggest pdflatex compile** để verify không có LaTeX error.
7. **Diff capped ở 60 lines** — không dump toàn bộ, chỉ preview.