---
name: readme-updater
description: >
  LUÔN dùng skill này bất cứ khi nào cần cập nhật README.md của wheeled biped project:
  stage mới train xong, eval results mới có, status task thay đổi, command mới,
  hay user nói "update README", "đánh dấu stage này là done", "thêm kết quả vào README".
  Biết chính xác cấu trúc từng section của README hiện tại (status table, quick reference,
  3-seed protocol, known limitations). Luôn tạo diff rõ ràng trước khi apply —
  không overwrite nguyên file. Sau khi update README, đề xuất sync paper/main.tex nếu liên quan.
license: Project-internal skill
---

# README Updater Skill

## Tổng quan

README có các section cần update định kỳ:

| Section                        | Update khi nào                 |
| ------------------------------ | ------------------------------ |
| **Status badge** (dòng đầu)    | Stage mới được train/eval      |
| **Overview table**             | Task status thay đổi           |
| **Quick reference**            | Command mới, path thay đổi     |
| **3-seed experiment protocol** | Sau khi train xong seeds       |
| **Known limitations**          | Khi limitation được giải quyết |
| **Paper artifact generation**  | Sau khi có paper eval results  |

---

## Bước 1 — Đọc và parse README hiện tại

### Xác định sections bằng anchors

```python
import re
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ReadmeSection:
    name: str           # tên section để reference
    start_line: int
    end_line: int       # exclusive
    content: str

def parse_readme(readme_path: str = "README.md") -> dict[str, ReadmeSection]:
    """
    Parse README thành dict của sections theo heading.
    Dùng để locate chính xác đoạn cần thay đổi.
    """
    text = Path(readme_path).read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # Map tên section → anchor patterns để tìm
    SECTION_ANCHORS = {
        "status_badge":       r"^\s*>\s*\*\*Status:\*\*",
        "overview_table":     r"^\|\s*\*\*Balance\*\*",
        "quick_reference":    r"^##\s+Quick reference",
        "three_seed_protocol":r"^###\s+3-seed experiment protocol",
        "known_limitations":  r"^##\s+Known limitations",
        "paper_artifacts":    r"^##\s+Paper artifact generation",
        "usage_training":     r"^###\s+2\. Training",
        "usage_evaluate":     r"^###\s+3\.",
    }

    sections = {}
    for name, pattern in SECTION_ANCHORS.items():
        for i, line in enumerate(lines):
            if re.search(pattern, line):
                # Tìm end của section (next heading cùng level hoặc EOF)
                end = _find_section_end(lines, i, line)
                sections[name] = ReadmeSection(
                    name=name,
                    start_line=i,
                    end_line=end,
                    content="".join(lines[i:end])
                )
                break

    return sections

def _find_section_end(lines, start, start_line):
    """Tìm line kết thúc của section (heading tiếp theo cùng level)."""
    # Xác định level của heading bắt đầu
    m = re.match(r"^(#{1,4})\s", start_line)
    if not m:
        # Không phải heading — tìm blank line tiếp theo
        for i in range(start + 1, len(lines)):
            if lines[i].strip() == "" and i > start + 2:
                return i + 1
        return len(lines)

    level = len(m.group(1))
    for i in range(start + 1, len(lines)):
        m2 = re.match(r"^(#{1,4})\s", lines[i])
        if m2 and len(m2.group(1)) <= level:
            return i
    return len(lines)
```

---

## Bước 2 — Update templates cho từng section

### 2a. Overview table — Task status

```python
# Các giá trị hợp lệ cho cột Status:
STATUS_VALUES = {
    "not_started":  "Config ready",
    "training":     "Training",
    "trained":      "Implemented",
    "evaluated":    "Evaluated",
    "paper_ready":  "Paper ready",
}

# Cấu trúc bảng hiện tại:
OVERVIEW_TABLE_TEMPLATE = """\
| Task               | Description                                    | Status       |
|--------------------|------------------------------------------------|--------------|
| **Balance**        | Stand upright, hold target height, resist push | {balance}    |
| **Balance Robust** | Push-recovery fine-tuning (40 N disturbances)  | {balance_robust} |
| **Wheeled Locomotion** | Wheel-driven forward/backward/turn          | {wheeled_locomotion} |
| **Walking**        | Leg-stepping locomotion                        | {walking}    |
| **Stair Climbing** | Step up/down                                   | {stair_climbing} |
| **Rough Terrain**  | Uneven surface traversal                       | {rough_terrain} |
| **Stand Up**       | Self-recovery from fallen pose                 | {stand_up}   |
"""

def update_overview_table(readme_path: str,
                           task_updates: dict[str, str]) -> str:
    """
    task_updates: {"balance": "Paper ready", "balance_robust": "Implemented"}
    Trả về nội dung README mới (chưa ghi file).
    """
    content = Path(readme_path).read_text(encoding="utf-8")

    # Đọc trạng thái hiện tại từ README
    current_statuses = _parse_task_statuses(content)
    # Merge với updates
    current_statuses.update(task_updates)

    # Tìm và replace bảng
    old_table = _extract_overview_table(content)
    new_table = OVERVIEW_TABLE_TEMPLATE.format(**current_statuses)

    return content.replace(old_table, new_table, 1)


def _parse_task_statuses(content: str) -> dict:
    """Đọc status hiện tại từ bảng Overview."""
    pattern = r"\|\s*\*\*([^*]+)\*\*\s*\|[^|]+\|\s*([^|\n]+)\s*\|"
    matches = re.findall(pattern, content)
    result = {}
    key_map = {
        "Balance Robust": "balance_robust",
        "Balance": "balance",
        "Wheeled Locomotion": "wheeled_locomotion",
        "Walking": "walking",
        "Stair Climbing": "stair_climbing",
        "Rough Terrain": "rough_terrain",
        "Stand Up": "stand_up",
    }
    for task_name, status in matches:
        key = key_map.get(task_name.strip())
        if key:
            result[key] = status.strip()
    return result
```

### 2b. Status badge

```python
STATUS_BADGE_PATTERNS = {
    # (pattern để match, text mới)
    "balance_only": (
        r"> \*\*Status:\*\*.*",
        '> **Status:** Active research prototype. Only the `balance` stage has been trained and\n'
        '> evaluated to date. Sim-to-real transfer has not been validated on hardware.'
    ),
    "balance_robust_trained": (
        r"> \*\*Status:\*\*.*",
        '> **Status:** Active research prototype. `balance` and `balance_robust` stages trained.\n'
        '> Sim-to-real transfer not yet validated on hardware.'
    ),
    "multi_stage_trained": (
        r"> \*\*Status:\*\*.*",
        '> **Status:** Active research prototype. Core locomotion stages trained in simulation.\n'
        '> Sim-to-real transfer not yet validated on hardware.'
    ),
}

def update_status_badge(readme_path: str, new_status_key: str) -> str:
    """Cập nhật dòng status badge."""
    content = Path(readme_path).read_text(encoding="utf-8")
    pattern, replacement = STATUS_BADGE_PATTERNS[new_status_key]
    # Chỉ replace dòng đầu tiên match
    new_content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
    return new_content
```

### 2c. Known limitations — đánh dấu đã giải quyết

```python
LIMITATION_KEYS = {
    "balance_only": (
        "Only `balance` has been trained",
        "Only the `balance` has been trained"
    ),
    "no_results": "No quantitative results yet",
    "no_sim_to_real": "Sim-to-real not validated",
    "multi_seed_planned": "Multi-seed training planned",
}

def mark_limitation_resolved(readme_path: str,
                               limitation_key: str,
                               replacement_note: str) -> str:
    """
    Thay thế một limitation item bằng note mới (hoặc xóa nó).

    Ví dụ:
        mark_limitation_resolved(
            "README.md",
            "no_results",
            "~~No quantitative results yet.~~ Balance stage results reported in paper."
        )
    """
    content = Path(readme_path).read_text(encoding="utf-8")
    search_text = LIMITATION_KEYS.get(limitation_key, limitation_key)
    if isinstance(search_text, tuple):
        for s in search_text:
            if s in content:
                search_text = s
                break

    # Tìm dòng chứa limitation text và replace toàn bộ bullet item
    lines = content.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if search_text in line:
            lines[i] = line.replace(line.rstrip(), f"- {replacement_note}")
            break

    return "".join(lines)
```

### 2d. Quick reference — thêm command mới

```python
def add_quick_reference_command(readme_path: str,
                                  section: str,
                                  new_command: str,
                                  after_pattern: str = None) -> str:
    """
    Thêm command mới vào Quick reference section.

    section: "Train" | "Evaluate" | "Research evaluation" | "Export"
    new_command: command string (multi-line OK)
    after_pattern: nếu có, thêm command sau dòng match pattern này
    """
    content = Path(readme_path).read_text(encoding="utf-8")

    # Tìm section trong quick reference
    section_pattern = rf"# ── {re.escape(section)}"
    match = re.search(section_pattern, content)
    if not match:
        # Thử tìm gần đúng hơn
        for line in content.splitlines():
            if section.lower() in line.lower() and "──" in line:
                break
        return content  # không tìm thấy, không thay đổi

    # Tìm vị trí insert (cuối section hoặc sau after_pattern)
    if after_pattern:
        ap_match = re.search(re.escape(after_pattern), content[match.start():])
        if ap_match:
            insert_pos = match.start() + ap_match.end()
            return content[:insert_pos] + "\n" + new_command + content[insert_pos:]

    return content   # fallback: không thêm nếu không tìm được vị trí chính xác
```

### 2e. 3-seed experiment protocol — update sau khi có results

````python
THREE_SEED_PROTOCOL_TEMPLATE = """\
### 3-seed experiment protocol

Final paper results should be reported over 3 independent seeds (42, 113, 999):

```bash
# 1. Train all three seeds
python scripts/train.py single --stage balance --steps 50000000 --seed 42
python scripts/train.py single --stage balance --steps 50000000 --seed 113
python scripts/train.py single --stage balance --steps 50000000 --seed 999

# 2. Evaluate all three seeds
python scripts/eval_balance.py \\
    --checkpoint outputs/balance/rl/seed42/checkpoints/final \\
                 outputs/balance/rl/seed113/checkpoints/final \\
                 outputs/balance/rl/seed999/checkpoints/final \\
    --num-episodes 50 --num-steps 2000 --seeds 0 42 123 \\
    --output-dir outputs/balance/rl/paper_eval

# 3. Evaluate LQR baseline
python scripts/eval_balance.py \\
    --controller baseline_lqr \\
    --scenarios nominal push_recovery friction_low friction_high \\
    --num-episodes 50 --output-dir outputs/balance/lqr

# 4. Export to LaTeX table
python scripts/export_results.py latex \\
    outputs/balance/rl/paper_eval/eval_results.json \\
    --output outputs/tables/balance_eval.tex
````

**Seed status:**
{seed_status_table}

Each seed is a fully independent training run with its own RNG state. Results are
aggregated (mean ± std) post-hoc; the three runs are never mixed during training.
"""

def update_three_seed_protocol(readme_path: str,
seed_statuses: dict[int, str]) -> str:
"""
seed_statuses: {42: "✅ Trained (50M steps)", 113: "⏳ Training", 999: "⏸ Not started"}
""" # Format seed status table
table_lines = ["| Seed | Status |", "|------|--------|"]
for seed in sorted(seed_statuses):
table_lines.append(f"| {seed} | {seed_statuses[seed]} |")
seed_table = "\n".join(table_lines)

    new_section = THREE_SEED_PROTOCOL_TEMPLATE.format(seed_status_table=seed_table)

    content = Path(readme_path).read_text(encoding="utf-8")
    # Tìm và replace section cũ
    old_section_match = re.search(
        r"### 3-seed experiment protocol.*?(?=\n---|\Z)",
        content, re.DOTALL
    )
    if old_section_match:
        return content[:old_section_match.start()] + new_section + content[old_section_match.end():]
    return content

````

---

## Bước 3 — Workflow cập nhật an toàn

### Nguyên tắc: LUÔN tạo diff trước khi ghi

```python
import difflib

def preview_diff(old_content: str, new_content: str,
                  fromfile: str = "README.md (current)",
                  tofile: str = "README.md (proposed)") -> str:
    """Tạo unified diff để review trước khi apply."""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
        n=3,   # context lines
    )
    return "".join(diff)


def apply_readme_update(readme_path: str,
                          new_content: str,
                          backup: bool = True) -> None:
    """
    Ghi README mới với optional backup.
    Chỉ ghi nếu có thay đổi thực sự.
    """
    old_content = Path(readme_path).read_text(encoding="utf-8")

    if old_content == new_content:
        print("README không thay đổi — bỏ qua.")
        return

    if backup:
        backup_path = readme_path + ".bak"
        Path(backup_path).write_text(old_content, encoding="utf-8")
        print(f"Backup: {backup_path}")

    Path(readme_path).write_text(new_content, encoding="utf-8")
    # Đếm dòng thay đổi
    diff_lines = [l for l in preview_diff(old_content, new_content).splitlines()
                  if l.startswith("+") or l.startswith("-")]
    print(f"README updated: {len(diff_lines)} lines changed.")
````

---

## Bước 4 — Các update scenarios phổ biến

### Scenario A: Balance stage vừa train xong + có eval results

```python
# Bước 1: Cập nhật status table
new_content = update_overview_table("README.md", {"balance": "Evaluated"})

# Bước 2: Cập nhật status badge
new_content = update_status_badge_content(new_content,
    '> **Status:** Active research prototype. `balance` stage trained and evaluated.\n'
    '> Sim-to-real transfer not yet validated on hardware.')

# Bước 3: Cập nhật seed status
new_content_str = update_three_seed_protocol_content(
    new_content,
    seed_statuses={
        42:  "✅ Trained (50M steps), eval done",
        113: "✅ Trained (50M steps), eval done",
        999: "✅ Trained (50M steps), eval done",
    }
)

# Bước 4: Xóa limitation "no results"
new_content_str = mark_limitation_resolved_content(
    new_content_str,
    "no_results",
    "~~No quantitative results yet.~~ Balance stage results now reported — see paper."
)

# Preview
print(preview_diff(Path("README.md").read_text(), new_content_str))
# Nếu OK: apply_readme_update("README.md", new_content_str)
```

### Scenario B: Thêm stage mới được train (ví dụ balance_robust)

```python
new_content = update_overview_table("README.md", {
    "balance_robust": "Implemented"
})
new_content = update_status_badge_content(new_content,
    '> **Status:** Active research prototype. `balance` and `balance_robust` trained.\n'
    '> Sim-to-real transfer not yet validated.')
```

### Scenario C: Cập nhật sau khi thêm script/command mới

Không tự thêm command vào README — hỏi user xem command mới là gì và
muốn đặt vào section nào, rồi dùng `add_quick_reference_command()`.

---

## Bước 5 — Checklist review trước khi commit

Sau khi update, kiểm tra những điểm sau:

```
[ ] Status badge phản ánh đúng stage đã train
[ ] Overview table: tất cả status values hợp lệ (xem STATUS_VALUES)
[ ] Quick reference: các paths trong commands còn valid
[ ] Known limitations: không xóa limitation chưa được giải quyết
[ ] 3-seed protocol: seed status table up to date
[ ] Không thay đổi các section không liên quan (robot specs, project structure, ...)
[ ] Diff hợp lý — không có thay đổi ngoài ý muốn (trailing whitespace, encoding, ...)
```

---

## Quy tắc khi dùng skill này

1. **Minimal diff** — chỉ thay đổi đúng section cần update, không reformat toàn file.
2. **Preview trước, ghi sau** — luôn show diff và hỏi confirm trước khi ghi file.
3. **Không tự thay đổi training commands** — các commands trong Quick reference phản ánh code thực tế; chỉ update khi scripts thay đổi.
4. **Không xóa Known limitations** khi chúng chưa được giải quyết — chỉ đánh dấu resolved.
5. **Giữ nguyên formatting** của file — không thêm/bỏ blank lines ngoài vùng edit.
6. **Status thứ tự**: `Config ready` → `Training` → `Implemented` → `Evaluated` → `Paper ready` — không skip bậc.
7. **Sau khi update README**, đề xuất cũng kiểm tra `paper/main.tex` nếu thay đổi liên quan đến results (dùng `paper-updater` skill).
