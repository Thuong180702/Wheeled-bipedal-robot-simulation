# Repository Audit Baseline

## Date
2026-06-13

## Repository State
- **Branch**: main
- **HEAD**: ef532bb Add t6j
- **Working Directory**: F:\ROBOTCUATAO\Wheeled-bipedal-robot-simulation
- **Total Tracked Files**: 1072

## Recent Commits
```
ef532bb Add t6j
3ecbbc9 Add evaluation scripts for Step C and Step E sagittal schedule candidates
64367b0 fix: Step E height-variant position hold robustness
092dff6 Add Step C height recovery validation module
e8f15ef Add Step C Height Recovery Specification document
```

## Git Status
Working tree is clean (no uncommitted changes).

## Key Documentation Files
- README.md (exists)
- CLAUDE.md (project instructions, checked into codebase)

## Project Type
MuJoCo MJX + JAX research codebase for wheeled bipedal robot control.
Current focus: balance control using hierarchical controller with LQR/WBC/VMC.

## Gitignore Coverage
Standard Python + MuJoCo + training artifacts covered.
Files ignored: outputs/, runs/, wandb/, logs/, __pycache__/, .venv/, etc.

## Phase B Progress (from CLAUDE.md)
- Phase B.6 complete: Height-scheduled dynamic LQR/IK prior
- Phase C complete: ResidualBalanceEnv implemented
- Ready for Phase D: 1M+ step residual training

## Current Best Controller
**T6J_centering_bias_trim** (in `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py`)
- Based on T6I_phase_aware_release
- Profile added in commit ef532bb
- Passed full staged validation better than T6I

## Repository Size Indicators
- 129 test files in main tests/ directory
- 1856 test functions
- 336 script files in main scripts/ directory
- 339 documentation markdown files in main docs/ (excluding .claude/worktrees)

## Top-Level Directory Inventory
```
.claude/         # Claude Code worktrees (ignored)
.github/         # GitHub workflows
assets/          # Robot URDF and meshes
backup_checkpoints/  # Day1, Day2 backups (possibly obsolete)
baselines/       # Empty directory
configs/         # YAML configs
docs/            # 339 markdown docs (excluding worktrees)
outputs/         # Training outputs (gitignored)
outputs_residual_smoke/  # Smoke test outputs (gitignored)
outputs_residual_test/   # Test outputs (gitignored)
paper/           # LaTeX paper
scripts/         # 336 scripts
tests/           # 129 test files
wheeled_biped/   # Main package
```

## Baseline Verification Files
- repo_tracked_files.txt: Full list of 1072 tracked files
- repo_status_ignored.txt: List of ignored files