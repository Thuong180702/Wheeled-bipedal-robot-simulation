# Controller Semantic Rename — Audit

**Date:** 2026-06-13
**Branch:** `repo-cleanup-t6j`
**Scope:** Scan of all `T6J`/`T6I`/`T6F`/`APCR1ND` symbolic names to determine rename scope. Archive docs excluded.

---

## Rename mapping

| Semantic primary | Legacy alias | Old string (profile_name) | Meaning |
|---|---|---|---|
| `SUPPORT_CENTERING_BIAS_TRIM` | `T6J_CENTERING_BIAS_TRIM` | `T6J_centering_bias_trim` | Stable support-centering profile with slow bounded bias trim |
| `PHASE_AWARE_AUTHORITY_RELEASE` | `T6I_PHASE_AWARE_RELEASE` | `T6I_phase_aware_release` | Phase-aware cap decay when support error converges |
| `EMERGENCY_BUDGET_CAP_RAISE` | `T6F_BUDGET_CAP_RAISE` | `T6F_budget_cap_raise` | Safe high-height emergency recenter with budget cap raise |
| `BAND_LIMITED_SUPPORT_RECENTER` | `APCR1ND_T5_BAND_LIMITED_BALANCED` | `APCR1nD_T5_band_limited_balanced` | Band-limited support recenter baseline |

---

## Files requiring changes

### Runtime (must change)
| File | Change type |
|---|---|
| `wheeled_biped/controllers/sagittal_velocity_damped_balance_controller.py` | Rename 4 constants, 4 `profile_name` strings, 4 dict keys, add aliases, update comments |

### CLI / scripts (must change)
| File | Change type |
|---|---|
| `scripts/simulate_hierarchical_controller.py` | Add semantic names to CLI choices + profile map; keep aliases |

### Tests (must change)
| File | Change type |
|---|---|
| `tests/test_t6j_centering_bias_trim.py` | Keep imports as aliases; update profile_name string assertions |
| `tests/test_t6h_t6i_variants.py` | T6I import may remain as alias |

### Scripts (rename)
| File | Change type |
|---|---|
| `scripts/run_t6j_height_ladder.py` | `git mv` → `run_support_centering_height_ladder.py`; keep wrapper |
| `scripts/analyze_t6j_height_ladder.py` | `git mv` → `analyze_support_centering_height_ladder.py`; keep wrapper |

### Active docs (must change)
| File | Change type |
|---|---|
| `docs/repo_cleanup/execution/deep_outputs_cleanup_report.md` | Add legacy label note |
| `docs/validation/support_centering_bias_trim_final_report.md` | Create semantic-named report |

### Archive (do not change)
All `archive/cleanup_2026-06-13/` contents — not touched per policy.

---

## Files NOT needing change

| File | Reason |
|---|---|
| `CLAUDE.md` | No `T6J` references found |
| `README.md` | No `T6J` command references found |
| `wheeled_biped/controllers/shape_posture_controller.py` | No T6J references |
| `configs/` | No T6J references |
| `wheeled_biped/training/` | No T6J references |
| `wheeled_biped/eval/` | No T6J references |

---

## Telemetry variable names (NOT renamed)

The controller uses internal telemetry variable names like `t6j_bias_trim_tau_nm`, `t6i_phase_aware_cap_nm`, `t6f_emergency_cap_nm`. These are **private telemetry field names**, not user-facing profile identifiers. They are not renamed — changing them would break saved telemetry CSV column headers.

---

## Strategy

1. In the controller: define semantic primary constants, add legacy aliases pointing to the same objects, add semantic + legacy string keys in `JOINT_FIX_PROFILES`.
2. In the sim CLI: add semantic names as primary choices, map legacy names to the same constants.
3. In tests: update profile_name string assertions to use the new canonical name; keep alias imports for backward compatibility.
4. In scripts: git mv the two T6J scripts with deprecation wrappers.
5. No numeric parameters changed; no behavior change.