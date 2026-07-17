# K2_JAX_DEDICATED_DEFAULT_V3 — Promotion Notice

**Date:** 2026-07-01  
**Status:** **PROMOTED** — now the official default controller

---

## What Changed

`K2_JAX_DEDICATED_DEFAULT_V3` replaces `K2_JAX_DEDICATED_DEFAULT_V2` as the default controller.

### Three evidence-backed changes from V2/V3 base:

1. **Drift height gate fixed** (`drift_hgate_vel_low/high`: 0.08/0.35 → 2.0/12.0 cm)
   - V2's drift gate compared cm position error against sub-cm thresholds → gate closed 99.8% of time
   - Fix: gate now opens above 2 cm height error, closing above 12 cm
   - Result: drift controller fully operational (100% vs V2's 0.2%)

2. **Heading hip-yaw gain optimized** (`heading_hy_kp`: 0.15/0.40 → 0.55 Nm/rad)
   - 5-point micro-ablation (kp=0.40, 0.55, 0.70, 0.85, 1.00) revealed non-monotonic response
   - kp=0.55 achieves near-zero yaw at fixed mid height (-0.50° vs V2's 5.27°)
   - Self-limiting via error gate: as yaw→0, gate closes, preventing oscillation

3. **Dynamic q_ref blend improved** (`dynamic_q_ref_blend_alpha`: 0.40 → 0.60)
   - 60/40 dynamic/static blend improves height tracking without compromising stability
   - Result: +8% CoM Z max (0.404→0.436m), -56% dynamic displacement (3.09→1.37m)

### Preserved from V3:
- All other gains, gates, and signs
- V2 velocity-only drift damping (no position return, no heading hold)
- V3 differential hip-yaw heading sign (left=+tau, right=-tau)
- No wheel-differential heading
- No lateral velocity damping
- Continuous gates (no discrete height buckets)

---

## Validation Summary

| Criterion | Result |
|-----------|--------|
| Falls | **0** (37/37 scenarios) ✅ |
| SAFETY_FAIL | **0** ✅ |
| Step C (random heights) | **7/7 WITHIN_TOLERANCE** ✅ |
| Step D (push sweep) | **12/12 WITHIN_TOLERANCE** ✅ |
| Step E (fixed heights) | **7/10 PASS**, 3 SAFE_BUT_WORSE |
| Dynamic height | **1/5 PASS**, 4 SAFE_BUT_WORSE |
| Realtime ≥50 Hz | **120-127 Hz** ✅ |

---

## Key Metrics vs Old Default

| Metric | V2 Default | V3 Default |
|--------|-----------|------------|
| Fixed yaw error (°) | 5.27 | **-0.50** |
| Lateral drift mid (m) | -0.028 | **-0.030** |
| Drift gate operational | 0.2% | **100%** |
| Dynamic displacement (m) | 3.09 | **1.37** |
| Performance (Hz) | ≥50 | **≥120** |

---

## Rollback

To use the previous default:
```bash
python scripts/run_k2_jax_realtime.py --profile K2_JAX_DEDICATED_DEFAULT_V2 ...
```

V2 profile remains available as `K2_JAX_DEDICATED_DEFAULT_V2`.

---

## Known Tradeoffs

1. **Push yaw +1.8° vs V3:** Systemic tradeoff of fixed drift gate. All profiles with operational drift gate show this regression.
2. **High-height yaw 16-18°:** Hip-yaw stabilizer less effective at extreme heights. Future: wheel-differential heading.
3. **Dynamic lateral drift 1.37m:** Reduced 56% from V3 but still present. Future: lateral velocity damping.

---

## Related Reports
- `docs/validation/k2_v3_audit_report.md` — original V3 root-cause audit
- `docs/validation/k2_v3_audit_fix_v2_report.md` — heading-gain micro-ablation investigation
- `docs/validation/k2_v3_audit_fix_v2_final_report.md` — this candidate's full validation
