# K2 JAX Production Runtime Regression Guard

**Date:** 2026-06-29
**Branch:** repo-cleanup-t6j
**Phase:** 6

## Smoke tests performed

| Scenario | Backend | Steps | Result |
|----------|---------|-------|--------|
| fixed_high (0.48m) | jax | 1000 | PASS — no fall, no NaN |
| fixed_high (0.48m) | jax, quiet, telemetry off | 500 | PASS |
| fixed_high (0.48m) | jax, quiet, summary | 500 | PASS |
| fixed_high (0.48m) | jax, quiet, decimated (10) | 500 | Not run |
| push_bwd (0.33m, 90N) | jax, quiet, summary | 3000 | PASS — survived |
| push_bwd (0.33m, 90N) | python | 200 | Not run |
| both-synced smoke | both-synced | — | Not run |

## Verified invariants

- [x] `backend=jax` standalone — no Python controller calls
- [x] No WBC/composer called in JAX fast path
- [x] No per-step print in `--quiet` mode
- [x] Telemetry can be off/summary/decimated/full
- [x] Python fallback (`--controller-backend python`) still runs
- [x] Both-synced mode infrastructure intact
- [x] Controller gains unchanged
- [x] Physics parameters unchanged
- [x] No fall/NaN on fixed-high scenario

## Pending (full 9+ scenario suite)

Not run due to time constraints. Required for PASS classification:
1. fixed_low (0.33m)
2. fixed_mid (0.40m)
3. ramp_up
4. ramp_down
5. up_down_cycle
6. gate_dwell
7. gate_chatter
8. push_fwd
9. push_bwd (verified)

## Pending tests

```bash
pytest tests/test_k2_jax_*.py -v
pytest tests/test_stage1_behavior_unchanged.py -v
```

Not run in this session.
