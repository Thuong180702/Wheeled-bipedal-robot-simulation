Status: DONE

Commits made:
- feat: analysis & gate evaluation for Step D push validation

  Creates `scripts/analyze_step_d.py` — standalone analysis tool that reads
  the combined CSV `outputs/step_d_all/step_d_all_metrics.csv`, reuses the
  `safety_ok` logic from `run_outer_loop_step_d_push.py`, computes gate
  metrics (must_not_fall_pass, any_hard_fail, max_drift_C, c_not_worse_count,
  total_cases), writes a JSON summary and a markdown report.

  Creates `tests/test_step_d_analysis.py` — 4 pytest integration tests:
  - Full dummy CSV (all 6 cases, profiles A and C, safe values) yields
    STEP_D_RANDOM_PUSH_PASS classification.
  - Markdown report is generated with Gate Summary and Decision sections.
  - Missing CSV exits with code 1.
  - Empty CSV exits with code 1.

Concerns: None
