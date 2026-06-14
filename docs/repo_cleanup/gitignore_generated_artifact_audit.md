# Gitignore and Generated Artifact Audit

## Date: 2026-06-13

---

## 1. Current .gitignore Coverage

### 1.1 Python Bytecode
```
__pycache__/
*.py[cod]
*$py.class
*.pyo
*.pyd
```
**Status**: ✓ Good

### 1.2 Virtual Environments
```
.venv/
venv/
env/
ENV/
.env
.claude/worktrees/
```
**Status**: ✓ Good

### 1.3 Build / Packaging
```
build/
dist/
*.egg-info/
*.egg
MANIFEST
```
**Status**: ✓ Good

### 1.4 Training Outputs
```
outputs/
runs/
wandb/
logs/
outputs_residual_smoke/
outputs_residual_test/
```
**Status**: ✓ Good

### 1.5 MuJoCo Logs
```
MUJOCO_LOG.TXT
mujoco_log.txt
```
**Status**: ✓ Good

### 1.6 Testing & Linting Caches
```
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
```
**Status**: ✓ Good

### 1.7 Jupyter
```
.ipynb_checkpoints/
```
**Status**: ✓ Good

### 1.8 OS / Editor Artifacts
```
.DS_Store
Thumbs.db
*.swp
*.swo
*~
```
**Status**: ✓ Good

### 1.9 Binary Outputs
```
*.mp4
*.avi
*.gif
```
**Status**: ✓ Good

### 1.10 LaTeX Build Artifacts
```
*.aux
*.out
*.toc
*.bbl
*.blg
*.fls
*.fdb_latexmk
*.lof
*.lot
*.run.xml
*.bcf
*.synctex.gz
*_backup_*.tex
*_backup_*.pdf
*_backup_*.aux
*_backup_*.log
*_backup_*.synctex.gz
```
**Status**: ✓ Good

### 1.11 Misc
```
*.log
*.bak
*.tmp
```
**Status**: ✓ Good (but see issues below)

---

## 2. Issues with Current .gitignore

### 2.1 Issue: `*.log` is too broad

The current `.gitignore` has `*.log` which is too broad. It catches:
- Run logs (t6a_run.log, t6b_run.log, etc.) - **should be ignored**
- But also might catch legitimate log files if they exist

**Actual tracked log files**: None found (run logs are ignored)

### 2.2 Issue: Missing paper build artifacts

`paper/main.log` is ignored (`!! paper/main.log`), but other paper artifacts may not be:
- `paper/*.blg` (biblatex log)
- `paper/*.fls` (file list)
- `paper/*.out` (output)
- `paper/*.synctex.gz` (sync)

These are already covered by `*.log`, `*.out`, `*.synctex.gz`, etc.

### 2.3 Issue: Log files in root directory

Currently ignored (!! markers show them are ignored):
```
t6a_run.log
t6b_5000_run.log
t6b_run.log
t6c_run.log
t6d_run.log
t6e_run.log
t6h_t6i_500_T5_run.log
t6h_t6i_500_T6F_run.log
t6h_t6i_500_T6H_run.log
t6h_t6i_500_T6I_run.log
```

**Status**: ✓ Properly ignored

---

## 3. Tracked Files That Should Be Ignored

### 3.1 paper/main.aux
```
paper/main.aux
```
**Issue**: Tracked LaTeX auxiliary file that should be regenerated on build.

**Recommendation**: Add `*.aux` to .gitignore if not already tracked by the more specific `*_backup_*.aux` pattern. But since LaTeX needs it, keep it tracked for reproducibility.

**Decision**: Keep tracked - necessary for paper reproducibility.

### 3.2 paper/main.pdf
```
paper/main.pdf
```
**Issue**: Tracked binary PDF file.

**Decision**: Keep tracked - the main paper output.

### 3.3 assets/robot-urdf/export.log
```
assets/robot-urdf/export.log
```
**Issue**: Tracked log file from URDF export.

**Recommendation**: Add to ignore list or remove from tracking.

**Decision**: Recommend untracking.

---

## 4. Untracked/Ignored Files Analysis

### 4.1 Ignored Directories (Correct)
```
.claude/worktrees/        ✓ Good
.pytest_cache/           ✓ Good
.ruff_cache/              ✓ Good
outputs/                  ✓ Good
outputs_residual_smoke/   ✓ Good
outputs_residual_test/    ✓ Good
scripts/__pycache__/      ✓ Good
tests/__pycache__/        ✓ Good
wheeled_biped/__pycache__/ ✓ Good
```

### 4.2 Ignored Files (Correct)
```
MUJOCO_LOG.TXT            ✓ Good
paper/main.log            ✓ Good
```

### 4.3 Egg-info (Ignored)
```
wheeled_biped_sim.egg-info/
```
**Status**: ✓ Good

---

## 5. Recommended .gitignore Improvements

### 5.1 Minor Improvements

```gitignore
# Add these specific exclusions:

# URDF export log (not needed in repo)
assets/robot-urdf/export.log

# Optional: ignore specific experiment run logs more explicitly
*_run.log
```

### 5.2 Current .gitignore is Sufficient

The current .gitignore covers most cases well. The only minor improvement would be to explicitly ignore the export.log file.

---

## 6. Generated Artifact Locations

### 6.1 Output Directories (All gitignored)
```
outputs/                    # Training outputs
outputs_residual_smoke/     # Smoke test outputs
outputs_residual_test/      # Test outputs
```

### 6.2 Cache Directories (All gitignored)
```
.pytest_cache/
.mypy_cache/
.ruff_cache/
__pycache__/               # All locations
```

### 6.3 Run Logs (All gitignored)
```
t6*_run.log
MUJOCO_LOG.TXT
paper/main.log
```

### 6.4 LaTeX Build Artifacts (Properly ignored)
All LaTeX temporary files are ignored.

---

## 7. Summary

| Category | Status | Action Needed |
|----------|--------|---------------|
| Python bytecode | ✓ Covered | None |
| Virtual environments | ✓ Covered | None |
| Build artifacts | ✓ Covered | None |
| Training outputs | ✓ Covered | None |
| MuJoCo logs | ✓ Covered | None |
| Test caches | ✓ Covered | None |
| Jupyter | ✓ Covered | None |
| OS artifacts | ✓ Covered | None |
| Binary outputs | ✓ Covered | None |
| LaTeX artifacts | ✓ Covered | None |
| Run logs | ✓ Covered | None |
| Export logs | ⚠ Tracked | Recommend untracking `assets/robot-urdf/export.log` |

**Overall .gitignore assessment**: Good - comprehensive and correct.

---

*Generated: 2026-06-13*