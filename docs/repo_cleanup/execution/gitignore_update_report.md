# .gitignore Update Report

**Date:** 2026-06-13
**Branch:** repo-cleanup-t6j

## Changes

Added explicit exclusions under the `Misc` block:

```gitignore
# ─── Repo cleanup 2026-06-13: explicit exclusions ────────────────────────────
# (already matched by *.log above, listed explicitly for clarity)
assets/robot-urdf/export.log
*_run.log
```

## Notes

- Both new patterns are already functionally covered by the pre-existing `*.log`
  rule. They are listed explicitly for documentation clarity and to make the
  intent durable for future contributors.
- `assets/robot-urdf/export.log` was untracked via `git rm --cached`
  (local file preserved on disk; see delete_log.csv).
- `archive/` is **NOT** ignored — per the cleanup policy, the archive is part of
  this commit's history and must be tracked.

## Untrack action

```
git rm --cached assets/robot-urdf/export.log
```

Verified: file removed from index, local copy retained (1.38 MB).
