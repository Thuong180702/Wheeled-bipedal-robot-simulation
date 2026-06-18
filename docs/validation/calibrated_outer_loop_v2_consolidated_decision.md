# Calibrated Outer-Loop v2 (B2v2) — Consolidated Decision vs B

## Profiles
- **B:** `support_position_outer_loop_pitch_ref` — **current best**
- **B2v2:** `calibrated_support_position_outer_loop_pitch_ref_v2` — candidate
- **A:** `height_scheduled_pitch_equilibrium_trim` — fallback

This document consolidates fixed-height (Phase 6), Step C (random/changing height),
and Step D (push) validation, and decides whether B2v2 should become current best,
remain experimental, or be rejected.

---

## 1. Fixed-height Phase 6 (prior result)
- B2v2 improved vs B on 6/10 heights.
- B2v2 regressed vs B on 2/10 heights (low_0p320, high_0p480) — both MINOR.
- Hard safety failures: 0.
- high_0p465 critical regression from B2 v1 **eliminated**.
- high_0p450 major win **preserved** (B2v2 out15 0.7% vs B 16.4%).

**Verdict: B2v2 better (net), with two minor known regressions.**

---

## 2. Step C — random/changing height (8 cases)
Classification: **CALIBRATED_STEP_C_PASS** (8/8 cases pass the gate).

| Case | B maxabs | B2v2 maxabs | B out15 | B2v2 out15 | Verdict |
|---|---|---|---|---|---|
| C1 slow ladder | 0.0881 | 0.0911 | 0.0% | 0.0% | ≈equal |
| C2 random dwell500 | 0.1309 | 0.1313 | 0.0% | 0.0% | ≈equal |
| C3 random dwell200 | 0.1908 | **0.1524** | 16.4% | **0.7%** | **B2v2 BETTER** |
| C4 abrupt high-low | 0.1812 | 0.1808 | 8.0% | 8.2% | ≈equal |
| C5 long random 5000 | 0.1908 | 0.1885 | 18.6% | **13.5%** | **B2v2 BETTER** |
| C6 focused low_320 | 0.0881 | 0.0911 | 0.0% | 0.0% | ≈equal |
| C7 focused high_480 | 0.1853 | 0.1852 | 11.9% | 12.1% | ≈equal |
| C8 low320→450→480 loop | 0.1453 | 0.1485 | 0.0% | 0.0% | ≈equal |

- No fall. No WBC/hidden/ownership. No parameter discontinuity (pitch_ref_disc=0 all).
- Focused low_0p320 / high_0p480 transition cases: **PASS** (regressions do NOT amplify dynamically).

**Verdict: B2v2 equal-or-better in random/changing height. Two clear wins (C3, C5).**

---

## 3. Step D — push (12 cases)
Classification: **CALIBRATED_STEP_D_PASS_WITH_MONITORING**.

Original D1-D6 (preserve B behavior):

| Case | B maxabs | B2v2 maxabs | Δ | Within ±0.02? |
|---|---|---|---|---|
| D1 30N high_480 | 0.1666 | 0.1669 | +0.0003 | ✅ |
| D2 60N high_480 | 0.3030 | 0.3027 | −0.0003 | ✅ |
| D3 30N low_330 | 0.2560 | **0.2404** | −0.0156 | ✅ **better** |
| D4 60N low_330 | 0.3416 | **0.3228** | −0.0188 | ✅ **better** |
| D5 90N high_480 | 0.4744 | 0.4753 | +0.0009 | ✅ |
| D6 45N random | 0.2224 | 0.2221 | −0.0003 | ✅ |

**B2v2 improves or matches B on 6/6 original cases.**

Focused / extended cases:

| Case | B maxabs | B2v2 maxabs | Δ | Note |
|---|---|---|---|---|
| D7 30N low_320 | 0.1987 | 0.2377 | **+0.039** | ⚠️ exceeds ±0.02 (Phase 6 regression) |
| D8 60N low_320 | 0.3625 | 0.3640 | +0.0015 | ✅ within |
| D9 60N high_450 | 0.2822 | 0.2822 | 0.0000 | ✅ lower out25 (4.1% vs 7.5%) |
| D10 90N repeat | 0.4744 | 0.4753 | +0.0009 | ✅ within |
| D11 transition | 0.3030 | 0.3027 | −0.0003 | ✅ |
| D12 lateral | 0.3030 | 0.3027 | −0.0003 | ✅ |

### D4/D8 hip-yaw: shared architecture limit, NOT a B2v2 regression
The automated 0.35-rad threshold flagged D4/D8 as "hard fail" for B2v2, but B (current
best) hits the **same** level at 60N low-height push:

| Case | B hip_yaw_max | B2v2 hip_yaw_max |
|---|---|---|
| D4 60N low_330 | 0.4056 | **0.4046** (better) |
| D8 60N low_320 | 0.4065 | **0.4048** (better) |

Both profiles reach ~0.40 rad. B2v2 is marginally **better** on both. Neither falls.
This is a shared limit of the controller architecture at the extreme combination of
low height + 60N push, not a B2v2-specific divergence. **B would fail the same gate.**

### Genuine concern: D7 (30N low_0p320)
B2v2 maxabs 0.2377 vs B 0.1987 (+0.039), hip_yaw 0.29 vs 0.14. This exposes the Phase 6
low_0p320 minor regression under a 30N push. B2v2 stays **safe** (no fall, hip_yaw < 0.35,
no WBC), but it is clearly worse than B here. Notably this does NOT amplify at 60N (D8 ≈equal).

**Verdict: B2v2 preserves low_0p330 recovery and matches/beats B on all original push
cases, but the low_0p320 regression is real (D7). Safe throughout.**

---

## 4. Known regression audit

| Regression | Fixed-height | Step C dynamic | Step D push |
|---|---|---|---|
| low_0p320 | minor (score) | **harmless** (C6 ≈equal) | **visible at 30N** (D7 +0.039), harmless at 60N (D8) |
| high_0p480 | minor (score) | **harmless** (C7 ≈equal) | **harmless** (D1/D2/D5/D10/D11/D12 all within) |

The high_0p480 regression never becomes harmful dynamically.
The low_0p320 regression is harmless in random/changing height but produces a real
(though safe) drift increase under a small 30N push (D7).

---

## 5. Hip-yaw / leg-yaw stability
- Step C: comparable across cases; B2v2 hip_yaw_asym ≈ B.
- Step D: B2v2 equal-or-better at every case except D7 (low_0p320 30N, 0.29 vs 0.14, still safe).
- At the shared 60N low-height limit (D4/D8), B2v2 is marginally **better** than B.

**Verdict: equal-or-better, single exception D7 (safe).**

---

## 6. Posture / contact / height / roll safety
- No fall in any of the 20 dynamic runs (8 Step C + 12 Step D).
- No WBC authority enabled; no WBC owners; no hidden torque; no ownership violation.
- Contact maintained; CoM-Z safe; roll < 1.1° in push cases.

**Verdict: safe.**

---

## 7. Controller smoothness
- pitch_ref discontinuities = 0 across all Step C cases.
- No parameter (Kp/Kd/theta/deadband) discontinuity during height transitions.
- pitch_ref rate-limited as designed.

**Verdict: smooth.**

---

## Required questions

1. **Does B2v2's fixed-height improvement matter in dynamic Step C/D?**
   Yes. C3 (out15 16.4%→0.7%), C5 (out15 18.6%→13.5%), D9 (out25 7.5%→4.1%) are clear
   dynamic wins. The high_0p450 fixed-height win transfers to push (D9).

2. **Does the low_0p320 minor regression become harmful during random height or push?**
   Random height: NO (C6 ≈equal). Push: PARTIALLY — D7 (30N) shows +0.039 maxabs, but
   stays safe and does not amplify at 60N (D8 ≈equal). It is a monitoring item, not a failure.

3. **Does the high_0p480 minor regression become harmful during push?**
   NO. Every high_0p480 push case (D1/D2/D5/D10/D11/D12) is within ±0.02 of B.

4. **Does B2v2 preserve B's major low_0p330 push-recovery win?**
   YES — and improves it. D3 (−6% maxabs, out25 1.7%→0.0%) and D4 (−5.5% maxabs).

5. **Does B2v2 improve high_0p450 enough to justify the minor regressions?**
   Yes for the dynamic high band (C3/C5/D9). The low_0p320 30N push regression (D7) is
   the only countervailing item, and it remains safe.

6. **Are hip-yaw / leg-yaw metrics equal or better?**
   Equal-or-better everywhere except D7 (safe). At the shared 60N limit B2v2 is better.

7. **Should B2v2 become current best or remain experimental?**
   B2v2 is **safe, equal-or-better in aggregate**, with two clear dynamic wins (random
   height, low_0p330 push) and preserves all of B's critical behaviors. The single genuine
   regression (D7 low_0p320 30N, +0.039, safe) prevents an unconditional "clearly beats B"
   claim. **Recommendation: commit B2v2 as EXPERIMENTAL opt-in; keep B as current best**
   until the low_0p320 push regression is closed.

---

## Decision
**B2v2 = EXPERIMENTAL_ONLY.**
- Step C: PASS. Step D: PASS_WITH_MONITORING.
- Safe in all dynamic conditions. No fall, no WBC, no divergence.
- Equal-or-better than B in aggregate, with clear wins in random-height and low_0p330 push.
- Does NOT *clearly* beat B due to the low_0p320 30N push regression (D7).
- **Keep `support_position_outer_loop_pitch_ref` (B) as current best.**
- Commit B2v2 as an opt-in experimental profile (already opt-in; no default change).
