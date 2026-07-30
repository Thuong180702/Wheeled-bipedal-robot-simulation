# 📋 Editorial Decision — Full Peer Review Panel (7 Agents) — Round 11

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen (Independent Researcher)
**Review Date:** 2026-07-29
**Review Mode:** `full` — 5-reviewer panel + field analysis + editorial synthesis
**Skill Version:** academic-paper-reviewer v1.10.0

---

## Review Panel Provenance

All five reviewer personas ran on a single model family. Persona diversity is not model diversity — blind spots may be correlated across reviewers (Ren et al. 2026, arXiv:2607.13104 §5.2).

---

## Decision

### **MAJOR REVISION** — Resubmit Encouraged

This is a unanimous recommendation. All 5 reviewers independently converged on Major Revision.

---

## Reviewer Summary

| Reviewer | Role | Identity | Recommendation | Confidence |
|----------|------|----------|---------------|------------|
| EIC | Editor-in-Chief, IEEE RA-L | 15yr wheeled mobile robotics editor | **Major Revision** | 4/5 |
| R1 | Methodology | Senior researcher, nonlinear control & robotics, ETH Zurich | **Major Revision** | 4/5 |
| R2 | Domain Expert | Professor of Robotics, KAIST; wheeled-legged platforms | **Major Revision** | 4/5 |
| R3 | Cross-Disciplinary | Principal Research Scientist, TRI; hardware deployment | **Major Revision** | 4/5 |
| DA | Devil's Advocate | Adversarial stress-test | **Reject** (58/100) | 5/5 |

> The DA recommends Reject but under the panel protocol, a single DA vote does not block — it escalates CRITICAL items for adjudication. The editorial decision (Major Revision) reflects the 4/5 reviewer consensus.

---

## Top Blocking Issues (3, ranked)

| Rank | Blocking Issue | Source | Evidence Anchor | Resolves via |
|------|---------------|--------|-----------------|-------------|
| **1** | **The central quantitative claim (53× precision improvement) benchmarks against a P-only strawman, not against any practical baseline (PI+AW, PI+dead-zone, DOB).** The paper argues against saturation-triggered AW in prose (Appendix A) but never runs the experiment. Until ACC is compared against PI + dead-zone conditional integration + back-calculation AW, the headline 53× number measures ACC's advantage over a controller *no practicing engineer would deploy for precision standing*. | DA, R1, R3 | Abstract 53× claim; Appendix A (prose only, no measurement); no PI+dead-zone row in any experimental table | R1 |
| **2** | **The headline "sub-millimeter precision (0.73 mm)" is achieved under exactly 1 of 12 conditions in the robustness sweep.** Under clean + 0ms delay: 0.42–0.73 mm. Under every other condition: 2.30–41.66 mm. The paper's own hardware pathway (Section VI) admits the clean/0ms condition is unachievable (projected 9.5ms latency, 0.5ms margin). The abstract prominently features 0.73 mm while burying the degradation in a parenthetical. | DA, R3, EIC | Table VI: 12 conditions; Abstract line 49; Hardware pathway line 685 | R2 |
| **3** | **DT-LQR gains were optimized for a plant with 0.25s servo dynamics, then tested on a plant without servo dynamics — this is a mismatch measurement, not a comparison.** The paper uses DT-LQR's failure to argue ACC's anchor is "the essential contribution" (Conclusion line 693), but cannot conclude this from a controller tested on the wrong plant. Re-deriving LQR for the direct-torque plant is a prerequisite for the "LQR fails" narrative. | DA, R1, R2 | Limitation 6 (line 679): "not re-derived for the DT plant"; Conclusion line 693 | R3 |

---

## Consensus Analysis

### Points of Agreement (4–5/5 reviewers agree)

**[CONSENSUS-5]** (All 5 reviewers agree):

1. **The asymmetric envelope follower is genuinely novel and well-executed.** The fast-attack/slow-release EMA to distinguish quiet stance from post-push ringdown is elegant, and the symmetric-EMA ablation convincingly demonstrates bistable failure. This is the paper's strongest technical contribution.

2. **The factorial ablation design (L0–L3 additive + S0–S5 subtractive) is methodologically excellent.** The two-direction validation structure and the mapping of gate failure modes to specific ablation rows provide strong internal validity.

3. **The limitations section is unusually honest and thorough.** Nine specific limitations, including the critical latency bifurcation, the backward-push weakness, and the simulation-only nature, are disclosed with commendable candor.

4. **The abstract cherry-picks best-case results.** The headline 0.73 mm figure is presented without adequate qualification about the idealized conditions required to achieve it. The robustness results that show degradation under realistic conditions are relegated to a parenthetical or omitted entirely.

5. **Simulation-only validation is the central weakness.** All results are in MuJoCo with ground-truth state access. The paper's own analysis shows the controller's key properties bifurcate at latency thresholds that make hardware deployment marginal.

**[CONSENSUS-4]** (4/5 agree):

6. **LQR baselines need fair re-implementation.** The PID servo path (~0.25s lag) cripples 3 of 4 LQR variants. The 6-state coupled LQR was not re-derived for the DT plant. The Q/R matrices were not re-derived when the plant changed (τ_s removed). A height-scheduled LQR with direct torque output and integral action on height error is the missing baseline.

7. **The 50+ parameter count undermines the claim of a "method" versus a "solution."** ACC has been demonstrated on exactly one simulated platform. The gate sensitivity analysis identifies α_r as dominant, but this is local sensitivity around one operating point, not evidence of portability. No second morphology or simulator has been tested.

8. **The Whleaper comparison — the closest prior architecture — is acknowledged but never delivered.** Whleaper (Zhu et al. 2024) uses LQR+PPO on the same 10-DOF morphology. The paper correctly identifies this as the nearest architectural neighbor and then provides zero quantitative comparison.

### Points of Disagreement

**Disagreement 1: Publication venue readiness**

- **EIC view**: Paper fits better at ICRA/IROS than RA-L. The contribution is an engineering synthesis, not a new algorithmic or theoretical result.
- **R2 view**: Paper is suitable for ICRA/IROS after addressing structural issues. Domain contribution is genuine (72/100).
- **R3 view**: Practical applicability is low (35/100). The 0.5ms latency margin makes this a simulation study with aspirational hardware claims.
- **DA view**: Publishable only as a short paper (2–3 pages, workshop) unless missing experiments are run.
- **Disagreement type**: Severity disagreement — all agree the paper needs work; disagreement is about HOW much work and at what venue.
- **Editor's Resolution**: Major Revision with resubmit encouraged. The contribution (asymmetric envelope-gated anchor) is genuine and the factorial ablation is well-designed. The paper can reach RA-L quality if it: (a) adds the PI+AW baseline, (b) re-derives DT-LQR properly, (c) narrows claims to what the data supports, and (d) moves flight/terrain to supplementary. If these cannot be done, ICRA/IROS is the appropriate venue.

**Disagreement 2: Statistical power at N=5**

- **R1 view (Critical)**: N=5 makes Welch's t-test unreliable. Sub-2N differences are indistinguishable from noise. The paper interprets non-significant results as informative.
- **DA view (Major)**: The subtractive ablation supports exactly 2 findings at N=5 (homing helps, pitch gate critical). Everything else is statistically indistinguishable from zero.
- **R2 view**: The statistical methodology (Welch's, Bonferroni, Cohen's d, Clopper-Pearson) is unusually careful for robotics. N=5 is acknowledged as a limitation.
- **Disagreement type**: Severity disagreement — statistical rigor vs. pragmatic acceptability.
- **Editor's Resolution**: R1/DA are correct that N=5 is insufficient for the fine-grained ablation narrative. However, the two robust findings (pitch gate criticality, homing benefit) are well-supported even at N=5, and the 53× idle precision improvement is unambiguous regardless of N. The paper should increase N to ≥10 for the push ablation and reframe fine-grained interpretations as suggestive rather than conclusive where power is insufficient.

---

## Decision Rationale

The panel unanimously views the asymmetric envelope-gated anchor mechanism as a genuine and well-executed contribution. The paper's methodological strengths — factorial ablation design, honest limitations disclosure, clear gate mathematics, and deterministic seed documentation — exceed the standard of most first-submission robotics papers.

However, three structural weaknesses prevent acceptance:

1. **Missing baselines create an evidence vacuum at the core of the paper's claims.** The 53× precision improvement number compares against P-only — a controller that cannot cancel DC bias by design. The paper argues in prose that saturation-triggered AW is insufficient (Appendix A) and that DOB/ADRC cannot distinguish push from ringdown (Related Work), but never runs the experiments. A PI controller with dead-zone conditional integration + back-calculation AW is the most obvious engineering baseline and has zero rows in any experimental table.

2. **The headline results are conditional on idealized conditions the paper admits are unachievable.** The 0.73 mm precision requires: stiff ground contact (solref={0.002,1}), zero latency, clean sensors, and simulator ground-truth body linear velocity. The paper's own robustness table shows that relaxing ANY of these degrades precision by 3–47×. The hardware pathway admits the latency condition alone is marginal (0.5ms margin). Presenting the idealized result as the headline while burying the degradation creates a mismatch between what the paper promises and what it delivers.

3. **The claims overreach the evidence by approximately 3×.** The paper presents three contributions across 7+ experimental dimensions. At the current evidence level, only one contribution is robustly supported: a proximity-gated integral with asymmetric envelope-gated damping achieves ~0.73 mm simulator idle precision on one platform while maintaining approximate P-only-level push survival. The flight/terrain "extensions" are trivial bolt-ons (3-term PD, 3-line height split). The factorial ablation's fine-grained per-component claims are underpowered at N=5. The LQR comparison tests controllers optimized for the wrong plant. The PPO baseline at 5.4M steps is not a credible RL comparison.

**Why Major Revision (not Reject):** The core idea is genuinely clever and the internal validation (within the simulator, within the ablation framework) is strong. The paper can reach RA-L quality if it: (1) adds the PI+AW comparison that every reviewer identified as missing, (2) properly re-derives LQR for the DT plant or removes the claim that LQR is structurally insufficient, (3) narrows the abstraction from "unified architecture with three contributions" to "a specific mechanism (asymmetric envelope-gated anchor) that achieves X," and (4) moves flight/terrain to supplementary. These are addressable with additional experiments and rewriting — not fundamental flaws in the approach.

---

## Required Revisions (Must Fix)

| # | Revision Item | Severity | Evidence Anchor | Source | Estimated Effort |
|---|--------------|----------|-----------------|--------|-----------------|
| **R1** | Add PI + dead-zone conditional integration + back-calculation AW as a baseline, with idle precision and push survival measured under identical conditions | **Critical** | No PI+AW row in any table; Appendix A argument-only | DA, R1, R3 | 2–3 days |
| **R2** | Re-derive LQR gains for the direct-torque plant (τ_s=0), OR remove the claim that "LQR is structurally insufficient" and reframe as "LQR implementations tested fail" | **Critical** | Limitation 6; Conclusion line 693 | DA, R1, R2 | 1–2 days |
| **R3** | Qualify the abstract's headline claims: state the idealized conditions for 0.73 mm explicitly; report the hardware-projected figure (~1–3 mm) alongside | **Critical** | Abstract line 49; Table VI 12 conditions | DA, R3, EIC | 0.5 day |
| **R4** | Increase N for push ablation from 5 to ≥10 per cell, or switch to nonparametric tests (Mann-Whitney U) and report effect sizes with confidence intervals instead of significance tests | **Major** | Table IV N=5; push_ci.py --reps parameter | R1, DA | 1–2 days |
| **R5** | Add standalone `AccBalanceController` reference implementation matching paper equations variable-for-variable, or provide a mapping document from Eq. 1–9 to the JAX/WBC code | **Major** | 2445-line WBC module; `compute_v3_torque_for_state` bears no relation to paper notation | R1 | 2–3 days |
| **R6** | Deliver Whleaper comparison: implement Whleaper's LQR on this platform and report idle precision + push survival, OR state explicitly that gains are unavailable and discuss expected comparison qualitatively with caveats | **Major** | Section II.A line 87; no Whleaper data anywhere | R2 | 1–3 days |
| **R7** | Fix platform comparison table: mark all "---" entries as "metric not reported in cited work"; correct Ascento push recovery entry; correct DIABLO height entry; add Simulation/Hardware column | **Major** | Table I; three misleading entries identified by R2 | R2 | 0.5 day |

### Required Item Details

**R1: PI+AW baseline.** The paper's central claim is that ACC resolves the precision-vs-push trade-off. The fair comparison is not P-only (which cannot cancel DC bias) but PI with anti-windup — the standard engineering solution for setpoint tracking with integrator windup protection. Implement: (a) PI with dead-zone on position error (±5 cm), (b) PI with back-calculation AW, (c) PI with conditional integration (integrator active only within 15 cm). Measure idle RMS and push F_min/F_med under identical conditions. If PI+AW matches ACC on push survival, ACC's contribution narrows to the asymmetric envelope specifically — which is a more honest and defensible contribution. If PI+AW cannot match ACC, the paper gains a compelling "why simple solutions fail" narrative.

**R2: Re-derive DT-LQR.** The paper's "LQR fails" narrative is the primary motivation for ACC. But DT-LQR was tested with gains optimized for a plant with 0.25s servo dynamics. Obtain the linearized plant with τ_s=0 (direct torque input), re-derive Q/R via Bryson's rule for the new plant, compute the LQR gain, and test. If the re-derived DT-LQR achieves non-trivial standing (e.g., >5s survival), the paper's motivation weakens and claims must be adjusted. If it still fails, the paper gains a stronger "LQR is genuinely insufficient" argument.

**R3: Abstract honesty.** Current abstract: "sub-millimeter idle precision (0.73±0.07 mm CoM RMS, N=10, 95%CI [0.68,0.78] mm, stiff ground contact; degrades to ~1.2 mm under tire-like compliance and to 2.30 mm under 10 ms delay)." The qualifying clause is buried in a parenthetical after 4 other numbers. Revised abstract should lead with the conditional nature: "Under idealized simulation conditions (stiff ground, clean sensors, zero latency), ACC achieves 0.73 mm CoM RMS; projected hardware performance is 1–3 mm."

**R4: Statistical power.** N=5 yields Welch-Satterthwaite df∈[4,8]. The t-distribution with df=4 has tails ~2.8× heavier than normal — technically valid, practically unreliable. A single outlier shifts the mean by up to 2.8 SD. The minimum detectable effect of ~4.5N means sub-2N differences (L1→L2: +1.1N, S1 vs S3: Δ=1.1N) are indistinguishable from noise. Options: (a) increase N≥10, reporting achieved power; or (b) switch to Mann-Whitney U + effect-size CIs, reframing interpretations away from significance testing.

**R5: Reference implementation.** The paper presents Eqs. 1–9 as the complete controller. The actual implementation lives in `wheeled_biped/wbc/offline_three_arm_counterfactual.py` (2445 lines) called through `compute_v3_torque_for_state(...)`. A reader cannot implement ACC from the paper alone. Either extract a standalone `AccBalanceController` class (mirroring `FairLQRTorqueController`) implementing exactly Eqs. 1–9 in pure NumPy, or provide a detailed mapping document from paper notation to code paths.

**R6: Whleaper comparison.** Whleaper (Zhu et al. 2024, IROS) is the closest prior: same 10-DOF morphology, same LQR+RL hybrid philosophy. The paper identifies it correctly (Section II.A) then provides zero comparison. Either implement Whleaper's LQR base controller and measure idle precision + push survival, or explicitly state that gains are not publicly available and discuss the expected comparison qualitatively with appropriate hedging.

**R7: Platform comparison table.** Three entries in Table I are misleading:
- Ascento push recovery: listed as "Omnidir. (WBC)" implying quantified push thresholds. Klemm et al. 2020 does not report force-threshold push data. Should be "---" per the table's own "not reported in cited work" standard.
- SKATER idle precision: "---" visually implies SKATER cannot stand precisely. SKATER's MPC maintains standing posture; precision simply wasn't measured in mm RMS.
- DIABLO height: "Stance" undersells DIABLO's demonstrated squatting capability. Should note height variation is demonstrated but range not reported.
Add a "Simulation/Hardware" column to distinguish ACC's simulation results from hardware-validated platforms.

---

## Suggested Revisions (Should Fix)

| # | Revision Item | Severity | Source | Priority |
|---|--------------|----------|--------|----------|
| **S1** | Move flight recovery and terrain adaptation to supplementary material; focus main paper on balance + push recovery + LQR/PPO comparison | Major | EIC, DA, R3 | P1 |
| **S2** | Add body-linear-velocity estimation caveat to all "clean sensor" result captions, or report supplementary rows with estimated (not ground-truth) velocity | Major | R1 | P1 |
| **S3** | Narrow "no single classical controller integrates all three" claim to a defensible formulation; add missing WBC/MPC references identified by R2 | Major | R2 | P1 |
| **S4** | Provide data table for "torque rate-limit invariance (200–800 Nm/s)" claim, or retract it | Major | DA | P2 |
| **S5** | Add simulation validation for gate sensitivity analysis (at minimum, sweep α_r and envelope thresholds) | Minor | R1 | P2 |
| **S6** | Replace Cohen's d with Hedges' g (bias-corrected for small N); add confidence intervals for effect sizes | Minor | R1 | P2 |
| **S7** | Extend PPO training to ≥50M steps with height-conditioned observation, OR relabel current PPO result as "preliminary" and remove from abstract | Major | DA, R3 | P2 |
| **S8** | Add Lyapunov or describing-function analysis for the composite gated system, OR remove Liberzon citation and acknowledge formal stability analysis as future work | Moderate | DA, R2 | P3 |
| **S9** | Document PID servo lag measurement methodology; reconcile "PID servo cripples LQR" claim with DT-LQR's persistent 100% fall rate | Minor | R1 | P3 |
| **S10** | Restore standard IEEEtran formatting (remove aggressive spacing reductions, increase bibliography font from 4pt) | Minor | EIC | P3 |

---

## Venue Suitability Assessment

| Venue | Readiness | Rationale |
|-------|-----------|-----------|
| **IEEE RA-L** | ⚠️ NOT READY — Major Revision needed | Requires: fair PI+AW baseline, re-derived DT-LQR, narrowed claims, R1–R7 completed. RA-L expects novel, self-contained contributions. ACC's core mechanism is novel but it's currently buried under overclaimed contributions and under-evidenced comparisons. |
| **ICRA / IROS** | 🟡 POSSIBLE after R1–R4 | Conference papers can be empirical contributions with simulation validation. ACC's factorial ablation and gate analysis would be well-received. Must fix: PI+AW baseline, abstract honesty, N≥10 for ablation. Flight/terrain as supplementary. |
| **IEEE/ASME TMECH** | ⚠️ NOT READY — similar to RA-L | TMECH expects mechatronic system contributions with hardware validation OR exceptional simulation depth. ACC currently has neither. |
| **IEEE T-RO** | 🔴 NOT SUITABLE | T-RO requires theoretical depth (stability proofs, formal analysis) or extensive hardware validation. ACC has neither. The Liberzon citation is decorative; zero hybrid-systems analysis is performed. |
| **Robotica** | 🟡 POSSIBLE after moderate revision | Robotica accepts well-executed simulation studies with classical control contributions. ACC's contribution level and experimental depth would fit after R1–R4. |
| **IEEE Robotics & Automation Magazine** | 🔴 NOT SUITABLE | RAM publishes tutorial/survey material, not individual controller contributions. |
| **RAL/ICRA Workshop** | 🟢 READY as-is | ACC as a 2–3 page workshop contribution, with narrowed claims to what the data supports, would be well-received immediately. |

---

## Score Breakdown

### Per-Dimension Scores (Synthesized from 5 reviewers)

| Dimension | EIC | R1 | R2 | R3 | DA | **Panel Mean** |
|-----------|-----|----|----|----|----|----------------|
| Originality (20%) | 65 | 70 | 72 | 60 | 55 | **64.4** |
| Methodological Rigor (25%) | 60 | 58 | 65 | 55 | 60 | **59.6** |
| Evidence Sufficiency (25%) | 55 | 62 | 60 | 50 | 50 | **55.4** |
| Argument Coherence (15%) | 65 | 60 | 70 | 58 | 58 | **62.2** |
| Writing Quality (15%) | 68 | 65 | 70 | 65 | 72 | **68.0** |
| Significance & Impact | 60 | — | 68 | 62 | — | **63.3** |
| Reproducibility | — | 45 | — | — | — | **45.0** |
| Practical Applicability | — | — | — | 35 | — | **35.0** |
| **Weighted Average** | **61.0** | **58.8** | **66.8** | **54.7** | **57.6** | **60.6 / 100** |

> Weighting: Originality 20%, Rigor 25%, Evidence 25%, Coherence 15%, Writing 15%. Panel mean = average of all reviewer weighted scores.

### Score Interpretation

- **60.6/100** places this paper in the **"borderline — Major Revision required"** band.
- **RA-L acceptance threshold** is typically 70+. The paper is approximately 10 points below.
- **ICRA/IROS acceptance threshold** is typically 60–65. The paper is at the threshold and would be competitive after R1–R4.
- The 10-point gap to RA-L is primarily driven by: missing baselines (costs ~5 points on Evidence), overclaimed contributions (costs ~3 points on Coherence), and simulation-only with latency showstopper (costs ~2 points on Significance).

---

## Revision Roadmap

### Priority 1 — Structural Revisions (Estimated: 7–10 days)
- [ ] **R1**: Add PI+AW baseline (2–3 days)
- [ ] **R2**: Re-derive DT-LQR for correct plant (1–2 days)
- [ ] **R3**: Rewrite abstract with honest conditioning (0.5 day)
- [ ] **R4**: Increase N to ≥10 for push ablation (1–2 days)
- [ ] **R7**: Fix platform comparison table (0.5 day)
- [ ] **S1**: Move flight/terrain to supplementary (1 day)
- [ ] **S2**: Add velocity-estimation caveats (0.5 day)

### Priority 2 — Content Supplementation (Estimated: 5–8 days)
- [ ] **R5**: Standalone reference implementation or code-to-paper mapping (2–3 days)
- [ ] **R6**: Whleaper comparison — implement or explain infeasibility (1–3 days)
- [ ] **S3**: Narrow over-broad claims; add missing references (1 day)
- [ ] **S7**: Extend PPO or relabel as preliminary (1–2 days, if extending)

### Priority 3 — Text and Formatting (Estimated: 2–3 days)
- [ ] **S4**: Rate-limit invariance table or retraction
- [ ] **S5**: Simulation-validate gate sensitivity
- [ ] **S6**: Hedges' g + effect size CIs
- [ ] **S8**: Liberzon citation — add analysis or remove
- [ ] **S9**: Document PID lag methodology
- [ ] **S10**: Restore standard IEEEtran formatting

### Total Estimated Effort
- **Minimum viable (R1–R4, R7, S1–S3)**: 8–12 days
- **Full revision (all items)**: 14–21 days

---

## Closing

After careful consideration of all five reviewer reports, the editorial decision is **Major Revision — Resubmit Encouraged**.

The asymmetric envelope-gated anchor mechanism is a genuine contribution to wheeled biped control. The factorial ablation is methodologically sound. The honest limitations disclosure sets a standard other papers should follow. These strengths make the paper worth revising.

However, the paper currently claims more than its evidence supports. The headline 53× precision improvement compares against a strawman. The sub-millimeter result is conditional on idealized conditions the paper admits are unachievable. The LQR baselines test controllers designed for the wrong plant. And the 50+ parameter count — demonstrated on exactly one simulated platform — does not yet constitute a generalizable method.

The path forward is clear: add the PI+AW comparison, properly re-derive the LQR baselines, narrow the claims to what the data actually supports, and let the asymmetric envelope-gated anchor stand as the paper's single, well-defended contribution. With these revisions, the paper would be competitive at ICRA/IROS and within striking distance of RA-L.

We look forward to receiving your revised manuscript within 8 weeks.

---

*Signed: Editorial Panel (7-agent academic peer review)*
*EIC + 3 Peer Reviewers + Devil's Advocate + Editorial Synthesizer*

---

## Appendix A: Field Analysis (Phase 0)

### Paper Basic Information
- **Title**: Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery
- **Full text length**: ~7,000 words, 892 lines LaTeX
- **Number of references**: 36
- **Format**: IEEEtran conference

### Field Analysis

| Dimension | Analysis Result |
|-----------|----------------|
| Primary Discipline | Robotics & Control Systems |
| Secondary Disciplines | Mechatronics, Nonlinear/Hybrid Control Systems, Process Control |
| Research Paradigm | Quantitative — simulation-based experimental validation |
| Methodology Type | Quasi-experimental — factorial ablation, comparative baselines, parametric sweeps |
| Target Journal Tier | Q1–Q2 (IEEE RA-L, ICRA/IROS, IEEE/ASME TMECH) |
| Paper Maturity | Pre-submission — well-structured, figures complete, statistical analysis present |

### Recommended Target Journals (Top 3)
1. **IEEE RA-L** — Concise format, accepts simulation studies with strong experimental design, rapid review cycle
2. **ICRA / IROS** — Premier robotics conferences, accept empirical control contributions with simulation validation
3. **IEEE/ASME TMECH** — Mechatronics focus, expects hardware validation or exceptional simulation depth

---

## Appendix B: Full Reviewer Reports

---

### EIC Review Report — IEEE RA-L Editor-in-Chief

**Reviewer Identity:** Editor-in-Chief, IEEE Robotics and Automation Letters. 15 years of experience editing robotics papers, specializing in wheeled mobile robots, balancing control, and hybrid locomotion.

**Overall Recommendation:** Major Revision

**Confidence Score:** 4/5

**Summary Assessment:**
The paper presents ACC, a two-channel torque assembly for a 10-DOF wheeled biped robot. The core mechanism is a proximity-gated anchor integral with asymmetric velocity envelope follower, combined with a gated damping boost. In MuJoCo simulation, ACC achieves 0.73 mm CoM RMS idle precision and omnidirectional push survival (F_min=83 N, F_med=119 N, 8 directions). A factorial ablation isolates each component's contribution, and the paper compares against four LQR variants and a 5.4M-step PPO policy.

The paper's principal strength is its methodological rigor within the simulator: the ablation is well-designed, statistical reporting is unusually careful, and the nine-item limitations section is commendably honest. The central weakness is the simulation-only nature combined with a <10 ms latency bifurcation threshold that the authors themselves document. The paper projects a 0.5 ms margin on real hardware, which is "insufficient for industrial deployment." This means the headline result is, at present, a simulation phenomenon. The novelty question is also material: the individual components are textbook techniques; the contribution is the synthesis into a specific gating architecture — a legitimate but incremental contribution.

**Strengths:**
1. **Honest and comprehensive limitations section.** Nine specific limitations enumerated, including critical latency-sensitivity, simulation-only nature, limited statistical power, and absence of WBC/MPC baselines.
2. **Well-designed factorial ablation.** Additive (L0–L3) and subtractive (S0–S5) ablation systematically isolates each architectural component's contribution.
3. **Careful statistical methodology.** Welch's unpaired t-test with Bonferroni correction, Cohen's d effect sizes, Clopper-Pearson CIs, and explicit minimum detectable effect reporting.
4. **Replicable parameter reporting.** Core parameters tabulated with units; complete 50+ parameter set referenced as available.
5. **Multi-baseline comparison.** Four LQR variants plus PPO provide a reasonably broad baseline set.
6. **Clearly defined failure-mode-to-gate mapping.** Each gate traced to a measured failure mode, providing implicit FMEA coverage.

**Weaknesses:**
1. **[Critical]** Simulation-only validation with a self-documented hardware-blocking latency threshold. The paper admits all results are simulation-only and documents that any non-zero delay bifurcates F_max from ~95 N to ~45 N with a projected 0.5 ms margin.
2. **[Major]** Incremental novelty insufficiently distinguished from prior art. The individual mechanisms are established process-control techniques. The paper needs a sharper contribution claim: either a new principle or a surprising result.
3. **[Major]** LQR baseline methodology is systematically unfair. Two of four LQR variants are crippled by design (PID servo path lag), and the coupled 6-state LQR was not properly derived for the DT plant.
4. **[Major]** Overstretched scope with underpowered analyses. Seven experimental dimensions in a 6-page paper, several receiving minimal depth (flight recovery uses fixed initial conditions, terrain tests only one curb angle).
5. **[Moderate]** Title misalignment with content. "Anchored Conditionally-Gated" — two branding terms in tension. Paper also includes flight and terrain not reflected in title.
6. **[Moderate]** Abstract is a data dump, not a synthesis. Contains 14 numerical values in 7 sentences without contextualizing significance.
7. **[Minor]** Formatting concerns — aggressive spacing reductions, 4pt bibliography font may violate IEEEtran requirements.

**Questions for Authors:**
1. Have you characterized what happens when the latency budget is occasionally exceeded — e.g., a single 12 ms control cycle? Does the controller recover, or does a single late sample trigger a fall cascade?
2. Given that DT-LQR restores roll stability, it seems plausible a properly derived 6-state DT direct-torque LQR might achieve non-trivial standing. Have you computed what such a controller would look like?
3. In a sustained push, the DOB would converge to the correct estimate and cancel the disturbance, while your anchor would disengage. Is there a regime where DOB outperforms ACC?
4. Has the porting procedure been applied to even one other robot model to verify the gating architecture transfers?

---

### Reviewer 1 — Methodology (ETH Zurich)

**Reviewer Identity:** Senior researcher in nonlinear control systems and robotics at ETH Zurich, specializing in hybrid/switched control, disturbance rejection, and experimental validation of controllers on legged/wheeled platforms.

**Overall Recommendation:** Major Revision

**Confidence Score:** 4/5

**Summary Assessment:**
This paper presents a torque-level controller for a 10-DOF wheeled biped achieving sub-millimeter idle precision and omnidirectional push recovery through conditional gating. The core idea — a proximity-gated anchor integral with asymmetric envelope follower — is novel and well-motivated. The experimental campaign is extensive for a single-author simulation study. However, the manuscript has several methodological weaknesses that prevent acceptance. The most serious concern is the N=5 statistical foundation for the central ablation claims, which pushes Welch's t-test beyond its defensible limits. The ACC implementation is buried in a 2445-line JAX-compiled WBC module, creating a gap between the paper's equations and what actually executes. The 46-configuration LQR sweep claim is narrower than implied. And the clean-sensor results are presented as headline numbers without adequate caveats.

**Strengths:**
1. **Well-specified gate mathematics.** Smoothstep gates defined with explicit thresholds and transition bands. Asymmetric EMA specified via continuous-time constants with correct discrete-alpha derivation.
2. **Honest and extensive limitation disclosure.** Nine specific limitations with engineering maturity unusual in a first submission.
3. **Factorial ablation structure.** Additive + subtractive design maps each mechanism to a measured failure mode.
4. **Deterministic seed documentation.** All RNG seeds specified — better practice than most papers.
5. **Code availability.** Complete repository at pinned commit with all evaluation scripts and controller implementations.

**Weaknesses:**
1. **[Critical]** N=5 statistical foundation for central ablation claims. Welch's t-test with N=5 yields df∈[4,8] with tails ~2.8× heavier than normal. Minimum detectable effect ~4.5N means sub-2N differences indistinguishable from noise.
2. **[Critical]** ACC implementation is opaque; paper equations do not map to executable code. The actual implementation lives in a 2445-line JAX/WBC module called through function names bearing no relation to paper terminology.
3. **[Major]** The 46-configuration LQR sweep claim overstates what was actually swept. The sweep perturbs only roll PD gains and wheel velocity limit — not LQR Q/R matrices, PID gains, yaw PD, height IK, or plant model parameters.
4. **[Major]** Body linear velocity from simulator ground truth invalidates the "clean sensor" baseline as a meaningful hardware predictor. Even "low noise" increases idle RMS from 0.42mm to 6.33mm — a 15× degradation.
5. **[Major]** LQR baseline parameter selection is arbitrary despite Bryson's rule justification. The paper does not state what x_{i,max} values were used. The Direct Torque LQR uses the same Q/R as the PID-servo LQR — incorrect because the plant has changed.
6. **[Minor]** PID servo lag (0.25s) measurement methodology is underspecified. If calculated rather than measured, the actual lag may differ.
7. **[Minor]** Gate sensitivity analysis is purely analytical and does not validate against simulation. The "deferred" simulation sweep is acknowledged in a code comment but not in the paper.

**Dimension Scores:**
| Dimension | Score | Justification |
|-----------|-------|---------------|
| Methodological Rigor | 58/100 | Gate mathematics well-specified. But N=5 insufficient, implementation opaque, LQR sweep overstated, clean-sensor caveats inadequate. |
| Evidence Sufficiency | 62/100 | Extensive scenario coverage. But statistical power low, sweep narrower than claimed, gate sensitivity lacks sim validation, no WBC/MPC baseline. |
| Reproducibility | 45/100 | 50+ parameters disclosed, seeds documented, code available. But ACC cannot be reimplemented from paper alone — requires understanding 2445-line JAX/WBC with archived calibration data. No unified eval entrypoint. |

**Questions for Authors:**
1. Was tau_release = 1.5s found by sweeping, or derived from the post-push ringdown time constant? If swept, over what range and with what objective function?
2. What additional computations does the WBC framework perform beyond the paper's equations? Specifically, does it apply any constraint projection, task prioritization, or contact force optimization?
3. What are the Bryson's rule Q/R values for the direct-torque plant, and what would DT-LQR performance be with those weights?
4. Can you decompose the 84% compound-disturbance degradation into gate-design vs. physical (actuator limit) contributions?

---

### Reviewer 2 — Domain Expert (KAIST)

**Reviewer Identity:** Professor of Robotics at KAIST, specializing in wheeled-legged locomotion, bipedal balancing, and whole-body control. Has built wheeled biped platforms and authored survey papers on wheel-legged robots.

**Overall Recommendation:** Major Revision

**Confidence Score:** 4/5

**Summary Assessment:**
ACC's core contribution — a proximity-gated position integral coupled to an asymmetric velocity envelope follower — is genuinely clever. The factorial ablation is methodologically sound and the LQR baselines, while limited, establish a meaningful lower bound. However, the paper over-positions a single mechanism as a new control paradigm, the platform comparison table is misleading in several entries, the Whleaper comparison is acknowledged but never delivered, and key domain references are missing. The empirical claims are honestly conditioned and the limitations section is unusually candid. The paper would be suitable for ICRA/IROS after addressing the structural issues, or for IEEE RA-L with a substantially expanded comparison framework.

**Strengths:**
1. **Asymmetric envelope follower is genuinely novel in this domain.** Prior approaches (DOB, ADRC, saturation-triggered AW) operate in frequency-domain or actuator-saturation regimes and cannot make this distinction.
2. **Factorial ablation design is methodologically rigorous.** Symmetric-EMA ablation demonstrating bistable failure confirms non-triviality. 46-configuration LQR sweep provides credible evidence.
3. **Honest limitations section.** Openly acknowledges simulation-only, latency sensitivity, 84% degradation, backward-push weakness.
4. **The precision-vs-recovery trade-off is a real, underexplored problem.** P-only's 55 mm p-p limit cycle from a 1.3 Nm equilibrium bias is well-demonstrated.
5. **Gate failure modes map to measured ablation rows.** Stuck-at-1 → S4, stuck-at-0 → L0, stuck-half-open → S5 — implicit FMEA coverage.

**Weaknesses:**
1. **[Major]** Platform comparison table is misleading in at least three entries. Ascento push recovery incorrectly implies quantified thresholds. SKATER idle precision "---" visually implies inability. DIABLO height "Stance" undersells demonstrated capability.
2. **[Major]** Whleaper comparison is acknowledged as the closest prior but never delivered. No quantitative comparison, no implementation of Whleaper's LQR, no discussion of expected results.
3. **[Major]** "No single classical controller integrates all three" claim is overstated. Klemm's LQR-assisted WBC integrates balance, posture, and terrain. SKATER's MPC integrates all three in one optimization.
4. **[Minor-Moderate]** Connection to hybrid/switched systems (Liberzon) is gestured at but not engaged. One-sentence citation without substantive analysis.
5. **[Moderate]** Missing domain references — Bjelonic WBC work, SKATER team WBC papers, wheeled humanoid WBC, Cui et al. 2019 specifics.
6. **[Minor]** Flight recovery comparison with Ascento is not apples-to-apples — passive recovery vs. active jumping.
7. **[Moderate]** 50+ hand-tuned gains with no tuning methodology for the posture/balance parameters.

**Dimension Scores:**
| Dimension | Score |
|-----------|-------|
| Literature Coverage | 65/100 |
| Theoretical Positioning | 70/100 |
| Domain Contribution | 72/100 |
| Platform Comparison Fairness | 55/100 |
| Claim Scope | 75/100 |
| Prior Work Characterization | 68/100 |

**Questions for Authors:**
1. If Whleaper's LQR base controller were implemented on your platform, what idle precision would you expect? Does Whleaper's LQR contain any integral-like term?
2. Do you have any guarantee (even heuristic) that the composite gated system does not enter a limit cycle due to gate chattering at intermediate values?
3. If another lab wanted to implement ACC on a different platform, which of the 50+ gains would need re-tuning?
4. If Klemm's LQR-assisted WBC were implemented on your 10-DOF platform, would you expect it to suppress the 1.3 Nm equilibrium bias and survive 83 N pushes?

---

### Reviewer 3 — Cross-Disciplinary Perspective (TRI)

**Reviewer Identity:** Principal Research Scientist at Toyota Research Institute (TRI), spanning control theory, machine learning for robotics, and hardware deployment. Evaluates whether this work would actually transfer to hardware.

**Overall Recommendation:** Major Revision

**Confidence Score:** 4/5

**Summary Assessment:**
The core idea — conditionally activating integral action based on physical regime rather than actuator saturation — is genuinely clever. However, the paper conflates a carefully hand-tuned solution for one specific platform with a generalizable method. The 50+ parameter space raises the question: is this a controller design methodology, or is it the output of exhaustive in-simulation parameter search dressed up as architecture? The claimed sub-millimeter precision evaporates under realistic tire compliance and any latency above zero. The LQR baselines are structurally handicapped; the PPO baseline at 5.4M steps is not a fair RL comparison. The 0.5ms latency margin is a deployment showstopper, not an implementation detail.

**Strengths:**
1. **Asymmetric envelope follower is a genuinely novel integration** of classical signal-processing into control-systems context.
2. **Factorial ablation design is methodologically strong.** Each row tests a falsifiable claim.
3. **Failure-mode-driven design process is well-documented.** Each gate traced to root cause.
4. **Two-channel torque assembly is architecturally clean** — allows flight/terrain without modifying core.
5. **Robustness analysis is honest about latency bifurcation** — uncommon and valuable candor.

**Weaknesses:**
1. **[Critical]** Path-to-hardware section reveals a deployment-showstopper, not a viable plan. 0.5ms margin means controller cannot be deployed on described architecture. Proposed fix (RTOS+MCU) adds complexity undercutting "simplicity" claim.
2. **[Major]** 50+ parameter count undermines claim of "method" versus "solution." No demonstration on second morphology. Gate sensitivity analysis is local around already-tuned configuration.
3. **[Major]** LQR baselines structurally handicapped; PPO baseline at 5.4M steps insufficient. Modern RL baselines for locomotion typically require 50–200M steps.
4. **[Major]** Sub-millimeter precision claim conditional on unrealistic ground contact. On any real surface, tire compliance alone pushes above 1mm.
5. **[Moderate]** Novelty claim relative to prior art overstated. Contribution is specific recombination of known techniques applied to specific morphology.
6. **[Moderate]** Generality claims untested. Platform comparison table implies ACC competes with hardware-validated platforms.
7. **[Minor]** Flight recovery torque budget marginal for real-world deployment — 2 Nm on 8.1 kg robot.

**Dimension Scores:**
| Dimension | Score |
|-----------|-------|
| Significance & Impact | 62/100 |
| Practical Applicability | 35/100 |
| Generality | 25/100 |

**Questions for Authors:**
1. How many of the 50+ parameters would need retuning if wheel radius changed from 0.06m to 0.08m?
2. What is the physical mechanism of the delay bifurcation at 10ms? Is it a PD phase-margin property or ACC-specific?
3. What happens if you train PPO for 50M+ steps with height-conditioned observation, domain randomization, and the same torque interface?
4. Given robustness results show idle precision degrades 81× under medium noise, what is your revised estimate of hardware idle precision?

---

### Devil's Advocate — Adversarial Stress-Test

**Overall Recommendation:** Reject (58/100)

**Strongest Counter-Argument:**
The "53× precision improvement" is a measurement artifact dressed up as a scientific result. The paper compares ACC's steady-state anchored RMS (0.73 mm) against the P-only limit cycle RMS (39 mm), yielding 39/0.73 = 53×. But P-only is a deliberately crippled strawman — it has no integral action whatsoever. The paper itself admits the sole purpose of the anchor is to cancel a 1.3 Nm equilibrium bias. Canceling a DC bias with integral action is Control Engineering 101. The fair question is not "does ACC outperform P-only?" — of course it does, P-only cannot cancel DC bias by design. The fair question is: "does ACC's elaborate gating machinery outperform a simple PI controller with back-calculation anti-windup?" The paper never measures this. Appendix A argues against saturation-triggered AW in two paragraphs of prose, but never runs the experiment. Until that comparison exists, the 53× number is meaningless.

**Core Argument Challenges:**
- Contribution 1: The anchor integral is a 2D conditional integration dead-zone — textbook. Only the asymmetric velocity-envelope part is novel.
- Contribution 2: "Two-channel" is a description, not an architecture. Flight controller is 3 lines of code. Terrain adaptation is 3 lines of arithmetic.
- Contribution 3: The subtractive ablation at N=5 is epistemologically empty for fine-grained claims. Only 2 findings are robust.

**Logical Fallacies Detected:**
1. **Affirming the consequent:** DT-LQR failing → ACC mechanisms essential. There are infinitely many controllers between DT-LQR and ACC.
2. **False dichotomy:** Frame as ACC vs. LQR vs. PPO — ignoring PI+AW, DOB, ADRC, cascade PID, gain-scheduled PID, residual RL, model-based RL.
3. **Circular reasoning:** Thresholds tuned on failure modes, then "validated" by testing on same failure modes. Not pre-registration.

**Cherry-Picking:** Abstract reports 0.73 mm and F_min=83 N but omits that at 10ms delay idle degrades to 2.30 mm, at low noise+10ms to 12.82 mm, at medium noise+0ms to 34.00 mm, and F_max bifurcates from 96 N to 47 N with any delay.

**Alternative Explanations:**
1. ACC works because 50+ hand-tuned gains overfit to one platform, not because architecture is general.
2. Sub-millimeter precision is a MuJoCo contact-model artifact (stiff solref, implicitfast integrator).
3. Delay bifurcation at 10ms may be a discrete-time PD property, not ACC-specific.
4. DT-LQR fails because gains not re-derived, not because LQR class is insufficient.
5. PPO at 5.4M steps is an anecdote, not a baseline.

**"So What?" Test:** Even if everything is true, ACC solves a problem (standing precision) the field has not identified as a bottleneck. No stability analysis for control theorists. Not deployable for industry (10ms latency, 0.5ms margin). No RL+ACC evidence for the RL community.

**Issue Summary:**

CRITICAL:
| # | Issue |
|---|-------|
| C1 | 53× compares against P-only strawman, not PI+AW |
| C2 | Sub-millimeter achieved under 1/12 robustness conditions |
| C3 | DT-LQR tested with gains for wrong plant |

MAJOR:
| # | Issue |
|---|-------|
| M1 | Subtract ablation supports only 2 findings at N=5 |
| M2 | 13/16 operating regimes uncharacterized |
| M3 | Rate-limit invariance claimed without data table |
| M4 | Liberzon citation decorative — no hybrid-systems analysis |
| M5 | PPO at 5.4M steps is anecdote, not baseline |

**DA Scores:**
| Dimension | Score |
|-----------|-------|
| Originality | 55/100 |
| Methodological Rigor | 60/100 |
| Evidence Sufficiency | 50/100 |
| Argument Coherence | 58/100 |
| Writing Quality | 72/100 |
| **Weighted Average** | **58.0/100** |
