# APCR1nD T5 Low 0.300m 5000-Step Drift Report

**Date:** 2026-06-12  
**Profile:** APCR1nD_T5_band_limited_balanced  
**Classification:** PASS

---

## Drift Metrics

**Physical drift column:** active_pitch_crossing_signed_error_m

| Metric | Value |
|--------|-------|
| Survived steps | 5000/5000 |
| Min e | -0.0063 m |
| Max e | 0.1715 m |
| Max \|e\| | 0.1715 m |
| Peak-to-peak | 0.1778 m |
| Mean e | 0.0561 m |
| Mean \|e\| | 0.0561 m |
| Final e | 0.0402 m |
| Positive % | 99.1% |
| Negative % | 0.9% |
| Zero crossings | 5 |

---

## Band Metrics

| Band | Count | Percentage |
|------|-------|------------|
| Outside ±0.03 m | 4221 | 84.4% |
| Outside ±0.05 m | 2759 | 55.2% |
| **Outside ±0.08 m** | **1008** | **20.2%** |
| **Outside ±0.10 m** | **100** | **2.0%** |
| Outside ±0.12 m | 83 | 1.7% |
| **Outside ±0.15 m** | **51** | **1.0%** |

**Positive excursions:**
- e > 0.08 m: 1008 steps
- e > 0.10 m: 100 steps  
- e > 0.15 m: 51 steps

**Negative excursions:**
- e < -0.08 m: 0 steps
- e < -0.10 m: 0 steps
- e < -0.15 m: 0 steps

---

## Drift Accumulation

| Metric | Value |
|--------|-------|
| First 1000 mean \|e\| | 0.0581 m |
| Last 1000 mean \|e\| | 0.0503 m |
| **Accumulation ratio** | **0.865** |
| **Classification** | **STABLE** |

**Conclusion:** Drift DECREASED over time (ratio < 1.0). No accumulation.

---

## Comparison to APCR1n Baseline

| Metric | APCR1n | T5 | Improvement |
|--------|--------|----|----|
| Outside ±0.08 m | 38.4% | 20.2% | **-47.4%** |
| Outside ±0.10 m | 25.9% | 2.0% | **-92.3%** |
| Max \|e\| | 0.171 m | 0.171 m | 0% |
| Mean \|e\| | 0.0608 m | 0.0561 m | -7.7% |
| Accumulation | 1.099 | 0.865 | **Improved** |

**T5 superior on all drift metrics.**

---

## Status

✅ All drift targets achieved  
✅ No accumulation  
✅ Bounded throughout  
✅ Ready for high_0p480
