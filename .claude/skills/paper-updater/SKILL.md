---
name: paper-updater
description: >
  LUÔN dùng skill này khi user muốn cập nhật paper/main.tex, điền số liệu vào
  bảng LaTeX, hay hỏi "còn bao nhiêu TODO trong paper?". Biết tìm và điền tất cả
  placeholder: [TODO] (table cells), [TODO-RESULT], [TODO-TABLE], [TODO-FIGURE],
  [TODO-CLAIM], [TODO-ABLATION]. Đọc trực tiếp từ eval_results.json để điền đúng
  giá trị. Output: diff LaTeX minimal — chỉ thay đúng placeholder, không đụng text khác.
  Luôn preview diff trước khi apply. Tuyệt đối không tự xóa TODO-CLAIM còn pending.
license: Project-internal skill
---

# Paper Updater Skill

## Tổng quan

`paper/main.tex` có các loại placeholder cần xử lý khác nhau:

| Loại                     | Vị trí trong file | Khi nào điền                                 |
| ------------------------ | ----------------- | -------------------------------------------- |
| `[TODO]` bare            | Trong table cell  | Sau khi có eval_results.json                 |
| `% [TODO-RESULT: ...]`   | Comment block     | Sau khi có số liệu cụ thể                    |
| `% [TODO-TABLE: ...]`    | Comment block     | Sau khi có đủ data cho bảng                  |
| `% [TODO-FIGURE: ...]`   | Comment block     | Sau khi figure được generate                 |
| `% [TODO-CLAIM: ...]`    | Comment block     | **Không tự xóa** — cần experimental evidence |
| `% [TODO-ABLATION: ...]` | Comment block     | Chỉ điền khi ablation đã chạy                |
| `\textit{[TODO-...]}`    | Inline visible    | Thay bằng text thực hoặc xóa khi done        |

---

## Bước 1 — Scan toàn bộ placeholders

```python
import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class TexPlaceholder:
    kind: str          # TODO | TODO-RESULT | TODO-TABLE | TODO-FIGURE | TODO-CLAIM | TODO-ABLATION
    line_no: int       # 1-indexed
    raw_line: str
    description: str
    context: str       # 2 dòng trước + dòng này
    is_inline: bool    # True = visible in PDF (\textit{...})
    is_table_cell: bool

def scan_placeholders(tex_path: str = "paper/main.tex") -> list[TexPlaceholder]:
    text = Path(tex_path).read_text(encoding="utf-8")
    lines = text.splitlines()
    placeholders = []

    for i, line in enumerate(lines):
        ctx = "\n".join(lines[max(0, i-2):i+1])

        # [TODO] bare — table cell
        if re.search(r'\[TODO\]', line) and "TODO-" not in line:
            placeholders.append(TexPlaceholder(
                kind="TODO", line_no=i+1, raw_line=line,
                description="table cell value", context=ctx,
                is_inline=False, is_table_cell=True,
            ))
            continue

        # % [TODO-XXXX: description]
        m = re.search(r'%\s*\[(TODO-\w+):\s*(.+?)(?:\]|$)', line)
        if m:
            placeholders.append(TexPlaceholder(
                kind=m.group(1), line_no=i+1, raw_line=line,
                description=m.group(2).strip()[:120], context=ctx,
                is_inline=False, is_table_cell=False,
            ))
            continue

        # \textit{[TODO-...]} — visible placeholder
        m = re.search(r'\\textit\{.*?\[(TODO[^\]]*)\].*?\}', line)
        if m:
            placeholders.append(TexPlaceholder(
                kind=m.group(1).split(":")[0], line_no=i+1, raw_line=line,
                description=m.group(1)[:100], context=ctx,
                is_inline=True, is_table_cell=False,
            ))

    return placeholders


def summarize_placeholders(placeholders: list[TexPlaceholder]) -> str:
    by_kind = {}
    for p in placeholders:
        by_kind.setdefault(p.kind, []).append(p)

    lines = ["=== PAPER PLACEHOLDER STATUS ===", ""]
    for kind in ["TODO", "TODO-RESULT", "TODO-TABLE", "TODO-FIGURE", "TODO-ABLATION", "TODO-CLAIM"]:
        items = by_kind.get(kind, [])
        if not items:
            continue
        lock = " 🔒 DO NOT AUTO-FILL" if kind in ("TODO-CLAIM", "TODO-ABLATION") else ""
        lines.append(f"  [{kind}] × {len(items)}{lock}")
        for p in items[:4]:
            lines.append(f"    L{p.line_no}: {p.description[:80]}")
        if len(items) > 4:
            lines.append(f"    ... +{len(items)-4} more")
    return "\n".join(lines)
```

---

## Bước 2 — Điền bảng tab:rl_vs_lqr

Đây là bảng chính có nhiều `[TODO]` nhất. Cột: Scenario | Surv.(%) | e_h(m) | Pitch(deg) | Torque(Nm) | Push(N).

```python
import math, json
import numpy as np

# Map scenario key (từ eval_balance.py) → row label trong LaTeX
RL_VS_LQR_ROW_MAP = {
    "nominal":       "Nominal",
    "narrow_height": "Narrow",
    "full_range":    "Full range",
    "push_recovery": "Push recov.",
    "friction_low":  "Friction low",
}

def _fmt(val: float, decimals: int = 2, is_pct: bool = False,
          nan_str: str = "---") -> str:
    if math.isnan(val) or math.isinf(val):
        return nan_str
    v = val * 100 if is_pct else val
    return f"{v:.{decimals}f}"

def _fmt_ms(mean: float, std: float, decimals: int = 2,
             is_pct: bool = False) -> str:
    """mean ± std, hoặc chỉ mean nếu std=0/nan."""
    if math.isnan(mean):
        return "---"
    m = mean * 100 if is_pct else mean
    s = std  * 100 if is_pct else std
    if math.isnan(s) or s < 1e-9:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"

def _aggregate(results: list[dict], scenario: str, metric: str) -> tuple[float, float]:
    """Aggregate metric over multiple seeds → (mean, std)."""
    vals = [r.get(metric, float("nan"))
            for r in results if r.get("scenario") == scenario]
    vals = [v for v in vals if not (math.isnan(v) or math.isinf(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))

def fill_rl_vs_lqr_table(tex: str, eval_results: dict,
                           n_seeds: int = 1) -> str:
    """
    Điền bảng tab:rl_vs_lqr.
    eval_results: parsed từ eval_results.json (eval_balance.py output).
    n_seeds: số seeds trong results — dùng mean±std nếu >1.
    """
    results = eval_results.get("results", [])
    lines = tex.splitlines()
    new_lines = list(lines)

    in_table = False
    for i, line in enumerate(lines):
        if r"\label{tab:rl_vs_lqr}" in line:
            in_table = True
        if in_table and r"\end{table}" in line:
            in_table = False
        if not in_table or "[TODO]" not in line:
            continue

        for scen_key, row_label in RL_VS_LQR_ROW_MAP.items():
            if f"& {row_label}" not in line:
                continue

            surv_m,  surv_s  = _aggregate(results, scen_key, "survival_rate")
            eh_m,    eh_s    = _aggregate(results, scen_key, "height_rmse_m")
            pitch_m, pitch_s = _aggregate(results, scen_key, "pitch_rms_deg")
            torq_m,  torq_s  = _aggregate(results, scen_key, "torque_rms_nm")
            push_m,  push_s  = _aggregate(results, scen_key, "max_recoverable_push_n")

            if n_seeds > 1:
                surv  = _fmt_ms(surv_m,  surv_s,  1, is_pct=True)
                eh    = _fmt_ms(eh_m,    eh_s,    3)
                pitch = _fmt_ms(pitch_m, pitch_s, 2)
                torque = _fmt_ms(torq_m, torq_s,  2)
                push  = _fmt_ms(push_m,  push_s,  1)
            else:
                surv   = _fmt(surv_m,  1, is_pct=True)
                eh     = _fmt(eh_m,    3)
                pitch  = _fmt(pitch_m, 2)
                torque = _fmt(torq_m,  2)
                push   = _fmt(push_m,  1)

            # Replace [TODO] cells left-to-right
            new_line = line
            for val in [surv, eh, pitch, torque, push]:
                new_line = new_line.replace("[TODO]", val, 1)
            new_lines[i] = new_line
            break

    return "\n".join(new_lines)
```

---

## Bước 3 — Điền bảng tab:balance_results (variable-height)

```python
# Bảng cột: h_cmd | Surv.(%) | e_h(m) | Pitch(deg) | Torque(Nm)
BALANCE_HEIGHT_ROWS = [0.40, 0.50, 0.60, 0.65, 0.69]

def fill_balance_results_table(tex: str,
                                 per_height_metrics: dict[float, dict],
                                 n_seeds: int = 1) -> str:
    """
    per_height_metrics: {0.40: {survival_rate, height_rmse_m, pitch_rms_deg, torque_rms_nm}, ...}
    Lấy từ eval_balance.py với scenarios full_range / narrow_height / wide_height,
    hoặc từ evaluate.py mode=command_tracking.
    """
    lines = tex.splitlines()
    new_lines = list(lines)
    in_table = False

    for i, line in enumerate(lines):
        if r"\label{tab:balance_results}" in line:
            in_table = True
        if in_table and r"\end{table}" in line:
            in_table = False
        if not in_table or "[TODO]" not in line:
            continue

        stripped = line.strip()
        for h in BALANCE_HEIGHT_ROWS:
            if not stripped.startswith(f"{h:.2f}"):
                continue
            m = per_height_metrics.get(h, {})
            if not m:
                break

            surv   = _fmt(m.get("survival_rate",  float("nan")), 1, is_pct=True)
            eh     = _fmt(m.get("height_rmse_m",  float("nan")), 3)
            pitch  = _fmt(m.get("pitch_rms_deg",  float("nan")), 2)
            torque = _fmt(m.get("torque_rms_nm",  float("nan")), 2)

            indent = len(line) - len(line.lstrip())
            new_lines[i] = " " * indent + f"{h:.2f} & {surv} & {eh} & {pitch} & {torque} \\\\"
            break

    return "\n".join(new_lines)
```

---

## Bước 4 — Cập nhật Training Configuration (total steps)

```python
def update_training_steps(tex: str, total_steps: int) -> str:
    """Điền số steps thực tế vào TODO-RESULT trong Training Configuration."""
    pattern = (
        r'%\s*\[TODO-RESULT: report actual total training steps once complete[^\]]*\]'
    )
    replacement = f"% Training completed: {total_steps:,} environment steps."
    new_tex = re.sub(pattern, replacement, tex, flags=re.DOTALL)
    if new_tex == tex:
        print("  ⚠️  Không tìm thấy TODO-RESULT training steps placeholder.")
    return new_tex
```

---

## Bước 5 — Cập nhật Current Limitations

```python
RESOLVED_LIMITATIONS = {
    "no_results": {
        "find": r"\\textbf\{No quantitative results yet\.\}[\s\S]*?(?=\\item|\Z)",
        "replace": (
            "\\textbf{Quantitative results available.}\n"
            "    Training complete. Balance stage results reported in "
            "Section~\\ref{sec:results}.\n\n"
        ),
    },
    "multi_seed_done": {
        "find": r"\\textbf\{Multi-seed training planned[^}]*\}[\s\S]*?(?=\\item|\Z)",
        "replace": (
            "\\textbf{Multi-seed training complete.}\n"
            "    Seeds 42, 113, 999 completed. Results aggregated as mean "
            "$\\pm$ std in Section~\\ref{sec:results}.\n\n"
        ),
    },
}

def resolve_limitation(tex: str, key: str) -> str:
    cfg = RESOLVED_LIMITATIONS.get(key)
    if not cfg:
        return tex
    # Tìm \item block chứa pattern
    item_pattern = r'(\s*\\item\s+)' + cfg["find"]
    match = re.search(item_pattern, tex, re.DOTALL)
    if not match:
        print(f"  ⚠️  Không tìm thấy limitation '{key}'.")
        return tex
    return tex[:match.start()] + match.group(1) + cfg["replace"] + tex[match.end():]
```

---

## Bước 6 — Xóa \textit{[TODO-...]} khi section đã được điền

```python
def remove_inline_todo(tex: str, section_label: str) -> str:
    """
    Xóa dòng \textit{[TODO-...]} trong subsection khi đã có content thực.
    section_label: label của subsection, e.g. "sec:results"
    Chỉ xóa nếu dòng TRƯỚC nó KHÔNG phải comment (tức là đã có content).
    """
    lines = tex.splitlines()
    new_lines = []
    skip_next_textit = False

    for i, line in enumerate(lines):
        # Detect \textit{[TODO-...]} dòng standalone
        if re.match(r'\s*\\textit\{.*?\[TODO', line):
            # Chỉ xóa nếu dòng liền trước là content thực (không phải comment/blank)
            prev = lines[i-1].strip() if i > 0 else ""
            if prev and not prev.startswith("%") and prev != "":
                continue  # skip dòng placeholder này
        new_lines.append(line)

    return "\n".join(new_lines)
```

---

## Bước 7 — Workflow end-to-end

```python
import difflib

def preview_diff(old: str, new: str) -> str:
    return "".join(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile="main.tex (current)",
        tofile="main.tex (proposed)",
        n=3,
    ))

def apply_update(tex_path: str, new_tex: str, backup: bool = True) -> None:
    old = Path(tex_path).read_text(encoding="utf-8")
    if old == new_tex:
        print("Không có thay đổi.")
        return
    if backup:
        Path(tex_path + ".bak").write_text(old, encoding="utf-8")
    Path(tex_path).write_text(new_tex, encoding="utf-8")
    n = sum(1 for l in preview_diff(old, new_tex).splitlines()
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))
    print(f"✅ main.tex updated ({n} lines changed). Backup: {tex_path}.bak")

# ─── Ví dụ sử dụng sau khi có paper_eval ──────────────────────────
# tex = Path("paper/main.tex").read_text()
#
# # Scan tình trạng
# print(summarize_placeholders(scan_placeholders()))
#
# # Load results
# rl  = json.load(open("outputs/balance/rl/paper_eval/eval_results.json"))
# lqr = json.load(open("outputs/balance/lqr/eval_results.json"))
#
# # Điền bảng
# new_tex = fill_rl_vs_lqr_table(tex,     rl,  n_seeds=3)
# new_tex = fill_rl_vs_lqr_table(new_tex, lqr, n_seeds=1)
# new_tex = update_training_steps(new_tex, 50_000_000)
# new_tex = resolve_limitation(new_tex, "no_results")
# new_tex = resolve_limitation(new_tex, "multi_seed_done")
#
# # Review và apply
# print(preview_diff(tex, new_tex))
# apply_update("paper/main.tex", new_tex)
```

---

## Quy tắc khi dùng skill này

1. **`TODO-CLAIM` KHÔNG BAO GIỜ tự xóa** — chỉ xóa khi user xác nhận có experimental evidence.
2. **`TODO-ABLATION` không tự điền** — chỉ điền khi ablation đã chạy xong và có kết quả.
3. **`TODO-FIGURE` comment** — chỉ xóa comment khi figure file thực sự tồn tại.
4. **Luôn preview diff** trước khi apply — không ghi file khi chưa được confirm.
5. **Không reformat LaTeX** — giữ nguyên indentation, alignment cột bảng, comment style.
6. **mean ± std chỉ khi đủ 3 seeds** — 1-2 seeds thì ghi số đơn và note "(preliminary)".
7. **Verify compile** sau khi update: `pdflatex -interaction=nonstopmode paper/main.tex`.
8. **Sau paper update**, đề xuất sync README.md nếu status thay đổi.
