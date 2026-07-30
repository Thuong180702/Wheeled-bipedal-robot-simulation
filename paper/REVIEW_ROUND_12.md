# 📋 Editorial Decision — Full Peer Review Panel (7 Agents) — Round 12

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen (Independent Researcher)
**Review Date:** 2026-07-29
**Review Mode:** `full` — 5-reviewer panel + field analysis + editorial synthesis
**Skill Version:** academic-paper-reviewer v1.10.0
**Review Instruction:** Do NOT read prior review files (Rounds 1–11) to maintain neutrality.

---

## Review Panel Provenance

All five reviewer personas ran on a single model family. Persona diversity is not model diversity — blind spots may be correlated across reviewers (Ren et al. 2026, arXiv:2607.13104 §5.2).

---

## Decision

### **MAJOR REVISION** — Unanimous (5/5)

All 5 reviewers independently converged on Major Revision. The core contribution (proximity-gated anchor with asymmetric envelope follower) is genuine and well-validated internally. However, comparative claims systematically overreach the evidence, and 4 CRITICAL issues must be resolved before resubmission.

---

## Reviewer Summary

| Reviewer | Role | Identity | Recommendation | Confidence |
|----------|------|----------|---------------|------------|
| EIC | Editor-in-Chief, IEEE RA-L | Senior editor, wheeled-legged robots & torque control | **Major Revision** | 5/5 |
| R1 | Methodology | Control Systems Researcher, torque control architectures for legged robots | **Major Revision** | 5/5 |
| R2 | Domain Expert | Wheeled Bipedal Robotics (Ascento/SKATER/Swiss-Mile lineage) | **Major Revision** | 5/5 |
| R3 | Cross-Disciplinary | Process Control (chemical engineering) + Robotics | **Major Revision** | 4/5 |
| DA | Devil's Advocate | Adversarial stress-test — 4 CRITICAL items | **Major Revision** | 5/5 |

---

## Final Scores

### Weighted Aggregate (EIC — quality_rubrics.md)

| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Originality | 68 | 0.20 | 13.60 |
| Methodological Rigor | 62 | 0.25 | 15.50 |
| Evidence Sufficiency | 60 | 0.25 | 15.00 |
| Argument Coherence | 78 | 0.15 | 11.70 |
| Writing Quality | 74 | 0.15 | 11.10 |
| **FINAL WEIGHTED SCORE** | | | **66.90** |

> Rubric maps 65–79 → Minor Revision, but qualitative criteria (4 validated DA CRITICAL items + unanimous Major from 5 reviewers) override to **MAJOR REVISION**.

### Cross-Reviewer Score Correlation

| Reviewer | Overall Score | Primary Concern |
|----------|--------------|-----------------|
| EIC | 70.5 | Journal fit, latency sensitivity |
| R1 (Methodology) | 47/100 | Baseline fairness (LQR strawmen, PPO undertrained, missing PI+AW), reproducibility |
| R2 (Domain Expert) | 51.7/100 | Literature positioning, theoretical framework, platform comparison fairness |
| R3 (Cross-Disciplinary) | ~57/100 | Sim-to-real gap, novelty calibration, 0.5ms latency margin |
| DA | N/A (qualitative) | 4 CRITICAL items validated |

**Spread (47–71):** Genuine disagreement on severity, not on substance. R1 is harshest because baseline fairness and reproducibility are the paper's weakest dimensions. EIC is most generous because the asymmetric envelope-gated anchor is genuinely novel in wheeled-biped control.

---

## DA CRITICAL Items Adjudication

Per Iron Rule #4: every DA CRITICAL must be adjudicated visibly. All 4 validated — none rejected by EIC.

| DA Issue | Verdict | Blocks Accept? | Adjudication |
|----------|---------|---------------|-------------|
| **C1: Strawman LQR baselines** | ✅ VALIDATED (5/5 concur) | ✅ YES | All LQR variants use a 4-state TWIP model, not a full 10-DOF linearization of the articulated-leg plant. Limitation 6 partially admits this but abstract/conclusion overstate: "all classical baselines fail" → should be "the specific off-the-shelf LQR formulations tested fail." |
| **C2: Undertrained PPO** | ✅ VALIDATED (5/5 concur) | ✅ YES | 5.4M steps is ~10× below field standard (50–200M). Single seed, no learning curves. Claim "pure RL proved insufficient" is premature. Either retrain to 50M+ steps (3 seeds) or relabel as "preliminary." |
| **C3: Misleading 0.73mm headline** | ✅ VALIDATED (4/5 concur) | ✅ YES | Abstract and Table I feature 0.73mm without the idealized-conditions qualifier. Own robustness data shows degradation: ~1.2mm (tire), 2.30mm (10ms delay), 34mm (medium noise). Qualifier must precede the number in abstract. |
| **C4: No stability analysis** | ✅ VALIDATED (3/5 concur) | ⚠️ Does not block alone | Hybrid systems framing (Liberzon 2003) creates expectation of formal analysis. Minimum: linearized eigenvalue check at anchor-engaged fixed point (~5 lines of math). Compounds other issues. |

**Resolution:** All 4 CRITICAL items require action. C1–C3 block Accept. C4 compounds but doesn't independently block at RA-L level if addressed with eigenvalue check.

---

## Consensus Analysis

### UNANIMOUS AGREEMENT (5/5 reviewers)

1. ✅ **Asymmetric envelope-gated anchor is a genuinely useful contribution** — well-motivated by measured failure modes, convincingly validated by factorial ablation
2. ✅ **LQR baselines are not properly derived** — all use simplified 4-state TWIP model; comparative claims overreach
3. ✅ **PPO at 5.4M steps is undertrained** — insufficient to support "RL proved insufficient"
4. ✅ **0.73mm headline requires conditionality qualifiers** — degrades to 1.2mm (tire), 2.30mm (delay), 34mm (noise)
5. ✅ **Ablation methodology is a strength** — additive + subtractive, N=10, CI reporting, noise floor measurement
6. ✅ **Honest limitations are commendable** — 8–9 enumerated with quantitative specifics
7. ✅ **No stability analysis** — hybrid systems citation is decorative; Liberzon 2003 cited but zero formal machinery used

### CONSENSUS (4/5 reviewers)

8. ✅ **Novelty framing is inflated** — individual components standard (conditional integration, envelope detection, smoothstep blending); synthesis is the contribution, not invention of new primitives
9. ✅ **Flight/terrain are demos, not contributions** — N=10, no baselines, no ablation; should be labeled "Preliminary Extensibility Demonstrations"
10. ✅ **Residual RL claim is aspirational** — zero residual RL experiments; Whleaper already demonstrated LQR+PPO residual
11. ✅ **0.5ms latency margin is a deployment risk** — not a "Path to Hardware"; consumed by any single jitter source

---

## Top Blocking Issues (4, ranked)

| Rank | Blocking Issue | Source | Evidence Anchor | Resolves via |
|------|---------------|--------|-----------------|-------------|
| **1** | **LQR baselines are strawmen — all use 4-state TWIP model, not full 10-DOF linearization.** The paper's own Limitation 6 admits "a properly derived gain matrix from the complete plant is deferred." Claim "all classical baselines fail" must be scoped to "the specific off-the-shelf LQR formulations tested fail." | DA-C1, EIC, R1, R2, R3 | Abstract line 50; Conclusion line 693; Limitation 6 (line 685) | Scoping text change (Option B) OR derive full-state LQR (Option A) |
| **2** | **PPO at 5.4M steps is ~10× below field standard — "RL proved insufficient" claim is premature.** Single seed, no learning curves, curriculum designed for 10M-step eval intervals. A partially-trained policy's failure does not demonstrate RL insufficiency. | DA-C2, R1, R2 | Abstract line 50; Table V; Section 5.4 | Retrain 50M+ steps (3 seeds) OR relabel all PPO results as "preliminary" |
| **3** | **0.73mm headline in abstract lacks conditionality qualifier.** Own robustness data: ~1.2mm (tire), 2.30mm (10ms delay), 34mm (medium noise), projected 1–3mm (hardware). Abstract must state idealized conditions BEFORE the number. | DA-C3, EIC, R3 | Abstract line 49; Table VII; Table II footnote | Text change: add qualifier before number |
| **4** | **Missing PI + dead-zone baseline — 53× precision claim benchmarks against P-only (no integral at all).** A controller that cannot cancel DC bias by design will always lose to one that can. The simplest fair comparison (PI + spatial dead-zone on integral @ ±15cm) was not measured. | R1, DA | Abstract "53× RMS improvement"; Table II P-only row; Appendix A (prose only) | Add one experimental row |

---

## Revision Roadmap

### BLOCKING (CRITICAL — must fix before resubmission to any venue)

| # | Issue | Action | Effort |
|---|-------|--------|--------|
| **CR-1** | LQR baselines are strawmen | **(Option A)** Derive LQR from finite-difference linearization of full 10-DOF MuJoCo model at standing equilibrium. **(Option B — recommended)** Add prominent qualification throughout: "Four specific off-the-shelf LQR formulations using a simplified 4-state TWIP model fail; a fully-derived coupled LQR from the complete articulated-leg plant is deferred (Limitation 6)." Update abstract, Section 5.4, and Conclusion. | Low (B) / High (A) |
| **CR-2** | PPO undertrained | **(Option A)** Train PPO for 50M+ steps, 3 seeds, show learning curves + reward decomposition. **(Option B — recommended for now)** Relabel all PPO results as "preliminary (5.4M steps)" throughout paper including abstract. Replace "pure RL proved insufficient" with "preliminary PPO training (5.4M steps) showed limited height generalization, consistent with known difficulty of model-free RL on coupled pitch–height dynamics; extended training is deferred." | Low (B) / High (A) |
| **CR-3** | 0.73mm headline unqualified | Add "Under idealized conditions (clean sensors, 0 ms delay, stiff contact, MuJoCo)" BEFORE "0.73±0.07 mm" in abstract. Add hardware-projected range (1–3 mm) in same sentence: "…degrading to a projected 1–3 mm on hardware (encoder quantization, backlash, stiction, tire compliance, IMU noise)." | Low |
| **CR-4** | No stability analysis | Compute eigenvalues of linearized closed-loop system (pitch-forward Jacobian) at anchor-engaged equilibrium. If dominant eigenvalue time constant matches ~9s ringdown, provides analytical validation. If not, report and discuss discrepancy. ~5 lines of math + 1 paragraph. | Low–Medium |

### REQUIRED (MAJOR — strongly recommended for RA-L/TMECH)

| # | Issue | Action | Effort |
|---|-------|--------|--------|
| **MA-1** | Novelty framing inflated | Reframe contribution as synthesis: "We synthesize established techniques (conditional integration, envelope detection, smoothstep blending) with physically-motivated gating surfaces derived from wheeled-biped failure modes." Add explicit mapping table: ACC component → process-control analog (Åström 2005, Shinskey 1996). | Low |
| **MA-2** | Flight/terrain as contributions | Relabel as "Preliminary Extensibility Demonstrations" or move to appendix. Remove from contribution list (item 2). Add: "These are feasibility demonstrations (N=10 each); systematic evaluation with baselines and ablations is deferred." | Low |
| **MA-3** | Residual RL overclaim | Move ALL residual RL framing to Future Work. Add explicit statement: "Residual RL experiments using ACC as a prior have not been conducted and are not claimed as results of this paper. Whleaper [Zhu 2024] demonstrated LQR+PPO residual works on a similar morphology; whether ACC's anchor provides a stronger prior is an open hypothesis for future work." | Low |
| **MA-4** | 0.5ms latency margin | Reframe "Path to Hardware" as "Deployment Risks and Mitigations." Add timing budget table (sensor readout → estimation → computation → actuation → total, with worst-case jitter per stage). Acknowledge 0.5ms margin is consumed by any single jitter source. Present RTOS + dedicated MCU as a mitigation strategy, not a solved problem. | Low–Medium |
| **MA-5** | ACC runs at 100Hz vs LQR at 50Hz | Either report ACC performance at 50Hz, or add explicit acknowledgment: "ACC benefits from a 2× control bandwidth advantage (100 Hz vs. 50 Hz for LQR/PPO baselines). The contribution of control architecture vs. control rate cannot be separated in the current comparison." | Medium |
| **MA-6** | Missing PI + dead-zone baseline | Add a simple PI controller with spatial dead-zone (±15cm gate) on the position integral term, measured under identical conditions (same simulation, same push protocol, same metrics). This replaces the P-only strawman in the 53× comparison. | Medium |

### RECOMMENDED (MINOR)

| # | Issue | Action | Effort |
|---|-------|--------|--------|
| **MI-1** | Platform comparison table (Table I) selective | Add rows: Locomotion speed, Jumping capability. Change "—" footnote: "— indicates value not reported in cited work; does not imply the capability is absent." Remove boldface from rows where ACC's advantage is absence-of-data rather than quantitative superiority. Change Ascento push recovery to "Qual. (video)" with existing footnote. | Low |
| **MI-2** | CI computation bug in push_ci.py | Fix `push_ci.py` lines 180–181: replace `1.96` with `t.ppf(0.975, df=n_reps-1)` from `scipy.stats`. Already correct in `replicate_ablation_n5.py` (t=2.776) and `replicate_ablation_n10.py` (t=2.262). | Low |
| **MI-3** | Version-control ablation configs | Commit named YAML profiles for every ablation row (L0–L3, S0–S5) so third parties can reproduce Table IV with `push_ci.py --profile=<name>`. Currently only L0, L1, L3 have profiles in push_ci.py. | Low–Medium |
| **MI-4** | Envelope follower citations | Add signal processing references (e.g., Zölzer 2008, "Digital Audio Signal Processing") alongside Åström 2005 for the envelope detector lineage. | Low |
| **MI-5** | EMA initialization | Standardize on EMA₀=1.0 (hardware-recommended) for all reported measurements, or document the settle-time sensitivity sweep (3s vs. 5s producing 0.42mm vs. 0.73mm). | Low |
| **MI-6** | Whleaper quantitative engagement | Add explicit statement: "Whleaper does not report quantitative push-recovery force thresholds, precluding direct comparison of push survival. Idle precision is similarly unreported." | Low |
| **MI-7** | "Lessons Learned" section | Extract 3–4 design principles from the study: (1) envelope-detector regime detection is fragile to sensor noise — design for the degraded case, not the clean case; (2) latency sensitivity in gated controllers demands explicit phase-margin budgeting; (3) α_r dominance (20× next parameter) suggests single-parameter tuning suffices for porting. | Low |
| **MI-8** | Define "ringdown" at first use in main text | Currently defined in Table II caption before main text. Define in Section 5.1. | Low |
| **MI-9** | Parameter table (Table III) use continuous-time values | Lead with τ_attack=0.023s, τ_release=1.5s. Derive discrete α values from these. The rate-invariant specification is primary. | Low |
| **MI-10** | L2.5 ablation row | The additive path L2→L3 bundles 4 mechanisms simultaneously. Add L2.5 (L2 + proximity gate only) to decompose the additive confound. Currently acknowledged as "deferred" in footnote 7. | Medium |

---

## Venue-Specific Scorecard

| Venue | Current Score | After CRITICAL Fixes | After ALL Fixes | Verdict |
|-------|--------------|---------------------|-----------------|---------|
| **ICRA/IROS** | 68/100 | 78/100 (Accept) | 85/100 (Strong Accept) | ✅ **Ready with CRITICAL fixes** |
| **IEEE RA-L** | 62/100 | 72/100 (Minor Revision) | 78/100 (Accept boundary) | ⚠️ **Target after CRITICAL + MAJOR fixes** |
| **IEEE/ASME TMECH** | 52/100 | 62/100 (Major Revision) | 72/100 (Minor Revision) | ⚠️ **Needs hardware validation for strong case** |
| **Robotica** | 58/100 | 68/100 (Minor Revision) | 75/100 (Minor Revision) | ⚠️ **Viable after CRITICAL fixes** |
| **IEEE T-RO** | 38/100 | 48/100 (Reject) | 58/100 (Major Revision) | ❌ **Not a fit — needs formal theory + hardware** |

---

## Recommended Strategy

1. **Fix 4 CRITICAL items first** (1–2 weeks, mostly text/scoping — Option B for C1/C2, no new experiments required)
2. **Submit to IEEE RA-L** — best fit for the contribution level. Simulation-only is accepted with honest qualification. The letter format suits a single-mechanism contribution.
3. **If RA-L rejects:** ICRA/IROS is a strong fallback — the paper would be accepted there with minor revisions.
4. **In parallel:** begin hardware validation for a follow-up TMECH submission (demonstrate 1–3mm hardware precision, validate the 5-step tuning methodology on a second platform or physical robot).

---

## Strengths to Preserve

1. **Proximity-gated anchor with asymmetric EMA** — genuinely novel in wheeled-biped control. The specific coupling of spatial gating + temporal envelope for integral conditioning is creative and well-motivated.
2. **Factorial ablation (additive L0→L3 + subtractive S0→S5)** — methodologically exemplary. Two-direction design provides internal replication. S5 finding (pitch gate removal → parametric oscillation) demonstrates the method can detect unexpected interactions.
3. **Gate failure mode mapping** — stuck-at-1 → S4, stuck-at-0 → L0, stuck-half-open → S5. Implicit FMEA coverage.
4. **Honest limitations** — 8–9 items with quantitative specifics. Noise floor measured (0.002mm, SNR 289×). Between-trial CV reported. This transparency builds reader trust.
5. **Gate sensitivity hierarchy** — α_r dominates (220.4, 20× next parameter). 5-step porting procedure is actionable.
6. **Compound disturbance evaluation** — Scenario A (push during squat: 84% degradation) and Scenario B (sequential forward→backward: 5/5 survival) test realistic multi-axis loading.

---

## Bottom Line

**The paper has a publishable core** — the proximity-gated anchor with asymmetric envelope follower is a genuine engineering contribution that no prior wheeled-biped paper has demonstrated. The ablation methodology is exemplary. The honest limitations are commendable.

**The comparative claims need rescoping** — the LQR baselines, PPO baseline, and 0.73mm headline all overreach. Fixing these is a matter of honest scoping (text changes), not new experiments. Once the 4 CRITICAL items are addressed, the paper is competitive at IEEE RA-L.

**Estimated time to resubmission-ready:** 1–2 weeks for CRITICAL items; 4–6 weeks for CRITICAL + MAJOR items.

---

*Review completed 2026-07-29 by full 7-agent academic-paper-reviewer panel (v1.10.0). All 5 reviewers ran independently (Iron Rule #2). No prior review files (Rounds 1–11) were consulted — this review is neutral and independent.*
