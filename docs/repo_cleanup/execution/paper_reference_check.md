# Paper Reference Check

**Date:** 2026-06-13
**Branch:** repo-cleanup-t6j

## Method

```
grep -R "docs/validation" -n paper/ docs/ README.md
grep -R "apcr1|t5_|t6f_sign|t6h_|phase_b|superpowers" -n paper/ README.md docs/*.md
```

## Findings

1. **`paper/main.tex` references to archived reports:** NONE.
   No `docs/validation/*` path is cited anywhere in `paper/` or `README.md`.

2. **Only matches found** are internal cross-references inside `paper/notes/phase_b6_stronger_classical_prior.md`:
   - line 152: `paper/notes/phase_b4_lqr_ik_prior_validation.md`
   - line 153: `paper/notes/phase_b5_classical_prior_variants.md`
   - lines 66/73: `outputs/phase_b6_tuning/...`, `outputs/phase_b6_eval/...`

   These all point inside `paper/notes/` (KEPT) or `outputs/` (gitignored, untouched). They do **not** depend on any `docs/` report being archived.

## Conclusion

No manuscript dependency blocks archiving. All `docs/phase_b*.md`, `docs/task_*.md`,
`docs/upstream_controller_debug_plan.md`, `docs/superpowers/**`, and
`docs/validation/{apcr,t5,t6f_sign,t6h,e1,e2,f1,f2,g1,phase_*}*.md` can be archived
(not hard-deleted) with no impact on the paper.

`paper/notes/` is preserved in full.
