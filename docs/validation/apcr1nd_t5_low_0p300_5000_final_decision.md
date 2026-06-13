# APCR1nD T5 Low 0.300m 5000-Step Final Decision

**Date:** 2026-06-12  
**Profile:** APCR1nD_T5_band_limited_balanced  
**Height Variant:** low_0p300  
**Steps:** 5000  

---

## Classification

**APCR1ND_T5_LOW_0P300_5000_PASS_READY_FOR_HIGH_0P480**

---

## Decision Criteria

| Criterion | Target | T5 Result | Status |
|-----------|--------|-----------|--------|
| Survives ≥ 4900 | ≥ 4900 | 5000 | ✅ PASS |
| Outside ±0.08 m | ≤ 30% | 20.2% | ✅ PASS |
| Outside ±0.10 m | ≤ 10% | 2.0% | ✅ PASS |
| Outside ±0.15 m | ≤ 5% | 1.0% | ✅ PASS |
| Max \|e\| | ≤ 0.20 m | 0.171 m | ✅ PASS |
| Drift accumulation | < 1.5 | 0.865 | ✅ PASS |
| Wheel velocity spikes | < 50 steps > 7 rad/s | 0 | ✅ PASS |
| Contact/height/roll | Stable | Stable | ✅ PASS |
| No WBC/hidden/ownership | None | None | ✅ PASS |

**Result:** All 9 gates passed

---

## Key Findings

### Survival
- **5000/5000 steps** (100%)
- No fall, no termination
- Completed successfully

### Drift Control
- **Outside ±0.08 m:** 20.2% (vs 38.4% baseline, -47.4%)
- **Outside ±0.10 m:** 2.0% (vs 25.9% baseline, -92.3%)
- **Max |e|:** 0.171 m (bounded)
- **Accumulation ratio:** 0.865 (IMPROVING, not accumulating)

### Stability
- **Wheel RMS:** 1.24 rad/s (vs 2.41 baseline, -48.5%)
- **Pitch RMS:** 0.049 deg (vs 3.52 baseline, -98.6%)
- **0 steps** > 5 rad/s (excellent)

### Comparison to 2000-Step
- Outside ±0.08 improved from 25.1% → 20.2% (-19.5%)
- Outside ±0.10 improved from 5.0% → 2.0% (-60.0%)
- Performance IMPROVED with longer run

---

## 15 Questions Summary

| # | Question | Answer |
|---|----------|--------|
| 1 | Survived 5000? | Yes (5000/5000) |
| 2 | Remained bounded? | Yes (ratio 0.865) |
| 3 | Outside ±0.08 ≤ 30%? | Yes (20.2%) |
| 4 | Outside ±0.10 ≤ 10%? | Yes (2.0%) |
| 5 | Outside ±0.15 ≤ 5%? | Yes (1.0%) |
| 6 | Max \|e\| ≤ 0.20? | Yes (0.171 m) |
| 7 | Drift accumulated? | No (improved) |
| 8 | Worst window? | Window 3 (1000-1500) |
| 9 | Contact/height/roll stable? | Yes (excellent) |
| 10 | Wheel/pitch acceptable? | Yes (0 spikes) |
| 11 | Tuned logic correct? | Cannot verify (telemetry missing) |
| 12 | Improved over APCR1n? | Yes (all metrics) |
| 13 | Proceed to high_0p480? | **Yes** |
| 14 | Current best low_0p300? | **Yes** |
| 15 | ±0.08 fully achieved? | No (79.8% inside) |

---

## Next Steps

### Immediate
**Run T5 at high_0p480** for Step E extreme height validation

**Command:**
```bash
python scripts/simulate_hierarchical_controller.py \
  --controller-mode balance-core \
  --sagittal-controller velocity-damped \
  --vd-sagittal-authority-profile APCR1nD_T5_band_limited_balanced \
  --height-variant-setup outputs/physical_target_height_setups/high_0p480_setup.json \
  --steps 5000 \
  --telemetry-decimation 1 \
  --failure-window-steps 5000 \
  --write-run-summary-sidecar
```

### Do NOT
- Do NOT run Step C (different objective)
- Do NOT run Step D (different objective)
- Do NOT commit yet (per user restriction)
- Do NOT enable WBC
- Do NOT enable HY2-DIV
- Do NOT relax Step E gates

### Future Work
- Add tuned telemetry fields to CSV
- Investigate window 3 peak drift pattern
- Consider T5 as default profile for low_0p300

---

## Conclusion

T5 Band-Limited Balanced successfully passes all 5000-step validation criteria at low_0p300:

✅ Survived 5000 steps  
✅ No drift accumulation  
✅ All band targets achieved  
✅ All stability gates passed  
✅ Superior to APCR1n baseline  
✅ Improved from 2000-step  
✅ Ready for high_0p480  

**Classification:** PASS_READY_FOR_HIGH_0P480

**Status:** Phase 0-8 complete. Awaiting user decision on high_0p480 validation.

---

**Date:** 2026-06-12  
**Phase:** 8 (Final Classification) COMPLETE
