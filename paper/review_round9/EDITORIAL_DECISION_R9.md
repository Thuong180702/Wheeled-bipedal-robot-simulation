Now I have all five reports. Let me synthesize the complete Editorial Decision Package.

---

# Editorial Decision

## Manuscript Information
- **Title:** Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery
- **Author:** Van Thuong Nguyen
- **Review Round:** Independent pre-submission review (Round 8)
- **Decision Date:** 2026-07-29

## Review Panel Provenance
All five reviewer personas ran on a single model family. Persona diversity is not model diversity — blind spots may be correlated across reviewers.

## Decision
### Major Revision

The decision is one grade above the 4/5 reviewer consensus (all four primary reviewers recommended Minor Revision) for three reasons that collectively exceed the scope of text-level fixes: (1) the two contradictory 95% confidence intervals for the headline idle-precision claim (R1-W1, flagged also by R2 minor-issue-11) constitute a factual reporting error that could change reader interpretation of the paper's central quantitative result; (2) the Devil's Advocate's two CRITICAL findings (C1: architecture-description mismatch from retaining the scheduled-k_p despite contradicting data; C2: logical contradiction between the Zamani comparison and the Direct Torque LQR evidence) require substantive resolution — not merely softened language — because they affect the paper's core architectural narrative and its central bottleneck claim; and (3) the Cross-Disciplinary reviewer's W1 and W2 (both rated Major by that reviewer with Confidence 4-5) demand re-framing of the "conditionally-gated" contribution relative to established process-control practice and anti-windup taxonomy that goes beyond a one-sentence qualifier. These are all resolvable within one revision cycle. None challenge the empirical validity of ACC's performance data.

## Top Blocking Issues (3, ranked)
| Rank | Blocking issue | Source reviewer(s) | Evidence anchor | Resolving roadmap item |
|------|---------------|-------------------|-----------------|----------------------|
| 1 | **Contradictory 95% CIs for idle precision.** Abstract reports 95% CI [0.68, 0.78] mm; Table IV footnote reports [0.50, 0.62] mm. These intervals do not overlap and have different centers (0.73 vs. 0.56). Only one can be correct. This is the paper's headline quantitative claim. | R1 (W1), R2 (minor-issue-11) | Abstract line 49 vs. Table IV footnote; R1 verified the abstract CI mathematically: 0.73 +/- 2.262*(0.07/sqrt(10)) = [0.68, 0.78] but [0.50, 0.62] is off-center by ~2.4 sigma. | Reconcile the two CIs: document which variance estimator each uses, correct the erroneous one, and apply the correction consistently. If the table CI uses between-trial variance of trial-level means rather than SEM of the grand mean, document and explain the difference. |
| 2 | **Logical contradiction: Zamani comparison vs. Direct Torque LQR evidence.** The deepened Zamani comparison (P2-R12) argues servo lag is the bottleneck because Zamani's direct-drive platform succeeds. But ACC's own Direct Torque LQR (no servo lag) still fails 100% with only 35% survival improvement. Three facts — (a) Zamani works without servo lag, (b) ACC's 6-state LQR fails with servo lag, (c) ACC's Direct Torque LQR fails without servo lag — cannot all be explained by "servo lag is the dominant bottleneck." | DA (C2), EIC (W2), R2 (W2) | Lines 87-88 (Zamani comparison); Table V LQR DT row (0.70 s survival, 100% fall rate) | Resolve explicitly. Options: (a) acknowledge platform differences (rigid two-wheeled body, no articulated legs, different inertia/authority) as the primary factor in Zamani's success, making the servo-lag comparison inconclusive; (b) demonstrate the DT LQR gains were not re-optimized for the direct-torque plant and the result is therefore conservative; (c) run and report coupled 6-state LQR in direct-torque mode (currently deferred, line 596). At minimum, add text reconciling the three data points. |
| 3 | **Architecture-description mismatch: scheduled k_p "retained for completeness" despite data showing fixed k_p=50 is equivalent or better.** S1 (fixed k_p=50) achieves F_min=83.1 N and F_med=118.8 N vs. S0 (scheduled k_p) at 80.4 N and 110.1 N. The paper recommends fixed k_p (line 188) but retains the schedule in the architecture diagram, Equation 1, and the "Full ACC" reference configuration. A component documented as introducing an implicit algebraic loop (Limitation 7) and providing no empirical benefit cannot be justified by "retained for completeness." | DA (C1), EIC (Round 7 DA-C1 verification: resolved, but DA argues the resolution is incomplete), R3 (W5), R2 (Q3) | Table IV S0 vs. S1 comparison; line 188-189; line 676 (Limitation 7); architecture Figure 2 | Either (a) remove the k_p schedule from the canonical architecture, re-label S1 as "Full ACC," and delete the scheduled variant from Equation 1 — the cleanest fix; or (b) provide a principled justification (e.g., safety margin under conditions not yet tested, or necessity for a different platform morphology). "Retained for completeness" is insufficient. |

## Reviewer Summary
| Reviewer | Role | Recommendation | Confidence |
|----------|------|---------------|------------|
| EIC | Senior Editor, IEEE RA-L | Minor Revision | 4 (High) |
| R1 | Methodology / Statistical Validation | Minor Revision | 4 (High) |
| R2 | Domain Expert (Wheeled Bipeds) | Minor Revision | 4 (High) |
| R3 | Cross-Disciplinary / Process Control | Minor Revision | 4 (High) |
| DA | Devil's Advocate (Stress-Test) | Major Revision (inferred: two CRITICAL + six MAJOR findings) | N/A (adversarial role) |

## Consensus Analysis
### Points of Agreement (Consensus)
1. **Core contribution is genuine (5/5).** The proximity-gated anchor with asymmetric envelope follower is a novel synthesis for wheeled-biped control, well-validated by the factorial ablation. [EIC S1, R1 S5, R2 S2, R3 S1, DA observations]
2. **Factorial ablation design is exemplary (5/5).** The additive (L0-L3) + subtractive (S0-S5) two-direction validation is methodologically stronger than typical robotics ablations. [EIC S2, R1 S5, R2 S1, R3 S3, DA observations]
3. **Statistical honesty is a distinguishing strength (5/5).** Explicit disclosure of Bonferroni correction, minimum detectable effect at N=5, Clopper-Pearson bounds for N=3, and 10 enumerated limitations set a commendable standard. [EIC S4, R1 S2, R2 S3, R3 S3, DA observations]
4. **LQR baseline evaluation is thorough within its paradigm (4/5).** Four variants with controlled ablations, 46-configuration gain sweep, and direct-torque isolation of the servo path represent above-average classical-baseline evaluation. [EIC S3, R1 S4, R2 S1; DA M3 partially dissents on sweep scope]
5. **10 ms delay bifurcation is the most practically significant and under-discussed result (4/5).** The collapse of F_max from ~95 N to ~45 N at any non-zero delay, combined with an estimated hardware latency of ~9.5 ms (0.5 ms margin), represents the single largest risk to hardware transfer. [EIC W5, R2 detailed comments, DA M5; R1 and R3 do not specifically flag]
6. **Platform comparison table (Table I) overclaims or mischaracterizes competitor capabilities (3/5).** Ascento push recovery characterization, SKATER height capability, and quantitative-vs-qualitative asymmetry are identified. [EIC W4, R2 W1; R1, R3, DA do not specifically address]
7. **"ACC as precondition for RL" claim is unsupported (4/5).** Zero RL+ACC experimental evidence. [EIC W3, DA M2, R2 detailed comments; R1 and R3 do not specifically flag]
8. **Gate activation figure (Fig. 6) is pedagogically excellent (4/5).** [EIC S6, DA observations, R3 S4 implicitly; R1, R2 do not specifically comment]

### Points of Disagreement
1. **Severity of DA-C1 (k_p schedule retention):** EIC marks it as "FIXED" in Round 7 verification. DA marks it as CRITICAL in Round 8, arguing that retaining a documented-harmful component in the canonical architecture is a description mismatch. R3 (W5) and R2 (Q3) also question the rationale for retention. **Resolution:** The EIC's verification was about the *acknowledgment* that fixed k_p=50 is recommended — that acknowledgment is in place. But the DA's argument is about the *architectural specification* — the paper still describes an architecture it does not recommend. This is a valid distinction. The paper should either remove the schedule from the architecture or provide a principled retention rationale. This is rated Blocking Issue #3.

2. **Severity of DA-C2 (Zamani contradiction):** EIC marks it as "PARTIALLY FIXED" (improved from "THE bottleneck" to "dominant bottleneck"). DA argues the deepended comparison created a new logical contradiction that was absent before. R2 (W2) independently identifies the same tension: the missing coupled-6-state-direct-torque experiment. **Resolution:** The contradiction is real and is not resolved by the softening from "THE" to "dominant." The paper must explicitly reconcile the three data points (Zamani success, ACC 6-state LQR failure, ACC DT LQR failure). This is rated Blocking Issue #2.

3. **Novelty of "conditionally-gated" framing:** EIC and R2 treat the framing as acceptable with minor adjustments. R3 (W1, rated Major) argues the paper should more explicitly acknowledge that continuous blending of control modes is standard in split-range control, and that ACC's contribution is the *specific synthesis of physically-motivated gating conditions* rather than the blending concept itself. **Resolution:** R3's concern is about scholarly credit assignment, not empirical validity. The paper should add one explicit acknowledgment paragraph as R3 suggests. This wording change is substantive enough to affect how the paper positions its contribution. Ranked as Required Revision #4.

4. **R1-W2 (test type: paired vs. independent):** R1 identifies that the Table IV footnote claims "paired t-test" but the measurement protocol uses independent seeds across configurations. Only R1 flags this. **Resolution:** This is a straightforward correction. Either change to independent-samples (Welch's) t-test, or document the pairing mechanism if one exists. Given N=5 per group and similar variances, the p-value shift will be modest, but the correction is necessary for statistical accuracy. Ranked as Required Revision #5.

## Decision Rationale

This manuscript has traversed eight review rounds and emerged as a substantially stronger paper. The core empirical contribution — a proximity-gated anchor integral with asymmetric envelope follower achieving 0.73 mm idle precision with 80 N omnidirectional push survival — is genuine and well-supported by factorial ablation. The LQR baseline evaluation (four variants, 46-configuration gain sweep) is unusually thorough for a wheeled-biped paper. The statistical disclosure (Bonferroni correction, minimum detectable effect, Clopper-Pearson bounds, 10 enumerated limitations) sets a standard that exceeds field norms.

Four of five reviewers (EIC, R1, R2, R3) independently recommended Minor Revision with high confidence (all at 4/5). Their remaining concerns center on framing precision (e.g., "dominant bottleneck" vs. "primary identifiable bottleneck," the "conditionally-gated" label's relationship to established process-control practice, the platform comparison table's characterization of competitors), text-level omissions (e.g., abstract omits push degradation under delay), and one statistical reporting error (contradictory 95% CIs for idle precision).

However, the Devil's Advocate's stress-test identified two structural issues that the other reviewers' concerns corroborate but rate differently. First, the deepened Zamani et al. (2020) comparison — added in Round 7 to strengthen the bottleneck argument — has created a logical tension with the Direct Torque LQR evidence: the paper uses Zamani's success on a direct-drive platform to argue servo lag is the bottleneck, while its own Direct Torque LQR (which also lacks servo lag) still fails 100%. R2 independently flags this gap as "Medium-High" severity, noting the missing coupled-6-state-direct-torque experiment. This is not a text-level fix; it requires explicit reconciliation of three data points that currently point in different directions. Second, the scheduled-k_p variant is now acknowledged as providing no benefit over fixed k_p=50 while introducing a documented algebraic loop, yet it remains in the architecture diagram and defines the "Full ACC" reference configuration. "Retained for completeness" is not a scientific justification for retaining a component that the paper's own data shows should not be used.

The CI inconsistency (R1-W1, corroborated by R2 minor-issue-11) is the most straightforward issue but also the most consequential for the paper's headline claim. Two 95% confidence intervals for the same quantity do not overlap and have different centers. One of them is incorrect. This must be resolved before acceptance.

The Cross-Disciplinary reviewer's concerns about positioning relative to process-control practice (W1: "conditionally-gated" relabels established techniques; W2: AW taxonomy distinctiveness claim needs qualification) are about scholarly framing, not empirical validity, but they matter for how the paper will be received by the control-theory community. Given that both are rated Major with high confidence by a domain expert, they warrant careful attention.

The collective weight of these issues — a factual reporting error affecting the headline claim, a logical contradiction in the central bottleneck argument, an architecture-description mismatch, and imprecise positioning relative to established control paradigms — exceeds what can be resolved through Minor Revision. Major Revision is appropriate because these are resolvable within one revision cycle and none challenges the validity of ACC's empirical performance data. The paper's intellectual honesty (10 limitations, statistical power disclosure, explicit labeling of qualitative demonstrations) remains its strongest asset and should be preserved as the revisions are made.

## Venue Readiness Assessment
1. **IEEE Robotics and Automation Letters (RA-L)** — Needs Major Revision. The core empirical contribution fits RA-L's scope and length. The CI inconsistency and Zamani contradiction must be resolved before archival publication. After revision, the paper is RA-L ready.

2. **ICRA/IROS conference** — Ready with Revision. Conference review is less intensive than journal review; the current manuscript would be competitive with the blocking issues resolved. The shorter format would compress some of the thorough ablation detail, but the core results would translate well.

3. **IEEE/ASME TMECH** — Needs Major Revision. TMECH expects stronger analytical treatment (e.g., stability analysis, describing-function analysis of the gate nonlinearity) than this empirically-focused paper currently provides. The process-control positioning issues (R3-W1, R3-W2) would face additional scrutiny. The practical impact narrative would need strengthening with more complete hardware degradation analysis (R2-W3).

4. **IEEE T-RO** — Not Ready. T-RO expects formal analysis (Lyapunov, passivity, or describing-function) for a contribution framed as a control architecture, hardware validation for claims of practical deployability, and comparison against optimization-based methods (MPC) rather than only classical LQR variants. The current manuscript's empirical-only approach and simulation-only validation fall short of T-RO standards. The DA's "missing stakeholder perspective" from a control-theorist is directly relevant here.

5. **Robotica** — Ready with Revision. Robotica accepts empirically-focused, simulation-validated control contributions with practical orientation. The honesty about limitations and the practical tuning guidance would be valued. The main gaps (CI inconsistency, Zamani contradiction, AW taxonomy precision) would still need fixing but the bar is more aligned with the paper's current level.

## Overall Score (0-100)
| Dimension | Weight | Component Scores | Weighted |
|-----------|--------|------------------|----------|
| Originality | 20% | EIC 78, R2 82, R3 72 = avg 77.3 | 15.5 |
| Methodological Rigor | 25% | EIC 82, R1 82, R3 78 = avg 80.7 | 20.2 |
| Evidence Sufficiency | 25% | EIC 72, R1 78, R2 75 = avg 75.0 | 18.8 |
| Argument Coherence | 15% | EIC 85, R3 82 = avg 83.5 | 12.5 |
| Writing Quality | 15% | EIC 80, R2 80 = avg 80.0 | 12.0 |
| **Weighted Total** | | | **79.0** |

Scores are synthesized from reviewer dimension ratings where available. EIC provided a full 5-dimension breakdown (78.7). R1 focused on methodology-specific dimensions (81). R2 used a 6-dimension breakdown (76.6). R3 used a 6-dimension breakdown (71). The panel-weighted score of 79 reflects: genuine originality in the wheeled-biped context (score pulled up by EIC and R2), strong methodological rigor (all raters >78), adequate but incomplete evidence sufficiency (limited by absent alternative-paradigm baseline and single-platform validation), high argument coherence, and strong writing quality. The R3 score of 71, which is the lowest, reflects process-control-domain expectations that the paper does not fully meet (control-theoretic soundness scored at 58, literature integration at 65). These are weighting decisions — a control-theory journal would rate the paper lower; a robotics-engineering venue would rate it higher. The score of 79 represents the consensus across reviewer perspectives.

## Required Revisions (Must Fix)
| # | Revision Item | Severity | Evidence Anchor | Source Reviewer | Section | Estimated Effort |
|---|--------------|----------|-----------------|-----------------|---------|-----------------|
| R1 | Reconcile contradictory 95% CIs for idle precision. Abstract CI [0.68, 0.78] vs. Table IV CI [0.50, 0.62] — one is incorrect. Document which variance estimator each uses, correct the error, ensure consistency. | Critical | Abstract l.49 vs. Table IV footnote | R1-W1, R2 minor-issue-11 | Abstract, Table IV | 1-2 hours |
| R2 | Resolve Zamani vs. Direct Torque logical contradiction. Reconcile three data points: Zamani succeeds without servo lag, ACC 6-state LQR fails with servo lag, ACC DT LQR fails without servo lag. Either (a) acknowledge platform differences as primary factor, (b) demonstrate DT LQR gains were not re-optimized, or (c) add coupled-6-state-DT experiment. | Critical | Lines 87-88 vs. Table V DT row | DA-C2, EIC-W2, R2-W2 | Section II-A, Section V-D | 2-8 hours (text) or 1-2 days (experiment) |
| R3 | Resolve architecture-description mismatch for scheduled k_p. Either remove from canonical architecture and relabel S1 as Full ACC, or provide principled retention rationale beyond "for completeness." | Critical | Table IV S0 vs S1, lines 188-189, Fig 2 | DA-C1, R3-W5, R2-Q3 | Section IV-B, Fig 2, Eq 1 | 2-4 hours |
| R4 | Add explicit acknowledgment that continuous blending is standard in split-range control; reposition ACC's contribution as the specific synthesis of physically-motivated gating conditions, not the blending concept itself. | Major | R3-W1 lines 88-89, 139 | R3-W1 | Section II-A, Section IV-A | 2-3 hours |
| R5 | Correct test type from "paired t-test" to independent-samples (Welch's) t-test, or document pairing mechanism if measurements are actually paired. Report test type and df explicitly. | Major | Table IV footnote l.485 | R1-W2 | Table IV, Section V-B | 1-2 hours |
| R6 | Qualify AW distinctiveness claim in Appendix A: acknowledge that state-gated conditional integration is already practiced in process control; specify that ACC's distinction is the envelope-detected, boost-coupled form. | Major | Appendix A lines 704-706 | R3-W2 | Appendix A | 1-2 hours |
| R7 | Hedge generality claims in 5-step porting procedure: acknowledge single-platform validation, single-mode assumption, and that cross-platform transfer is a hypothesis pending testing. | Major | Line 647 | R3-W3 | Section V-E | 1 hour |
| R8 | Add qualifier to abstract LQR failure statement: "...all fail (100% fall rate) among classical LQR variants evaluated" to align with Limitation 10. | Moderate | Abstract line 50 | EIC-W1 | Abstract | 15 min |
| R9 | Add push degradation to abstract: mention F_max collapse from ~95 N to ~47 N under delay alongside the precision degradation already stated. | Moderate | Abstract line 49 | EIC minor-issue, DA-m2 | Abstract | 15 min |
| R10 | Explain L2 (global-I survives) vs. S4 (global-I windups) discrepancy: the damping boost in S4 likely amplifies integral windup, while L2 has no boost. | Moderate | Table IV L2 vs S4 rows | R1-W5 | Section V-B | 1 hour |
| R11 | Report LQR survival time SD at higher precision or explain all episodes terminated at same controller timestep; +/- 0.00 across N=60 with randomized init is mechanically implausible. | Moderate | Table V | R1-W4 | Table V | 30 min |
| R12 | Add "Locomotion speed" and "Stair climbing" rows to Table I; verify SKATER height capability entry; add footnote clarifying competitor capabilities are summarized from cited work and may use different protocols. | Moderate | Table I | R2-W1, EIC-W4 | Table I | 2-3 hours |
| R13 | Rephrase "ACC is thus a precondition for RL" to forward-looking hypothesis: "ACC is hypothesized as a structured prior for future residual RL." | Moderate | Line 66 | EIC-W3, DA-M2 | Section I | 15 min |
| R14 | Provide justification or measurement for ~0.25 s servo lag figure; report PID gains used in LQR baselines. Without this, the bottleneck claim rests on an unverified tuning assumption.  | Major | Line 563 | DA-M4 | Section V-D | 2-4 hours |

## Suggested Revisions (Should Fix)
| # | Revision Item | Severity | Source Reviewer | Priority | Section |
|---|--------------|----------|-----------------|----------|---------|
| S1 | Report Hedges' g alongside Cohen's d for N=5 effect sizes (correction factor ~0.90) | Minor | R1-W3 | High | Section VI |
| S2 | Add combined hardware degradation estimate (soft contact + encoder quantization + sensor noise) rather than individual bounds; tighten the projected 1-3 mm range | Medium | R2-W3 | High | Section VI |
| S3 | Increase flight/terrain N from 3-4 to at least 10 for key conditions, or downgrade language from "feasibility proof" to "preliminary qualitative demonstration" | Low-Medium | R2-W4 | Medium | Section V-C |
| S4 | Discuss whether 10 ms delay bifurcation relates to phase margin at Nyquist frequency; estimate critical delay analytically | Medium | R2-Q2 | Medium | Section V-E |
| S5 | Add proprioceptive state estimation references for wheeled bipeds (SKATER, Ascento WBC) | Minor | R2 detailed comments | Medium | Section II, Section VI |
| S6 | Move statistical power discussion (Bonferroni, minimum detectable effect) to before the first results table, not buried in Discussion | Minor | R2 minor-issue-9 | Medium | Section V |
| S7 | Qualify non-monotonic noise explanation as post-hoc hypothesis ("we hypothesize") rather than confirmed mechanism | Minor | R1 minor-issue-7 | Medium | Section V-E |
| S8 | Add perturbation size to gate parameter sensitivity gradient magnitudes | Minor | R1 minor-issue-4 | Medium | Section V-E |
| S9 | Verify all references are cited in text (ferreira2021ollie, dicarlo2018mpc, lee2020rloc, kim2017velocity flagged as potentially uncited) | Minor | EIC minor-issue | Medium | References |
| S10 | Clarify whether Zamani theta_RMS/phi_RMS values are from the cited paper or from re-implementing the controller | Minor | R2 minor-issue-3 | Low | Section II-A |
| S11 | Explain the 7.7x factor for hip-roll gains (ratio of LQR-optimal kp_roll to decoupled PD kp_roll?) | Minor | R2 minor-issue-2 | Low | Section II-A |
| S12 | Cite envelope-detection signal-processing lineage (e.g., Giannoulis et al. 2012, JAES) | Minor | R3-M4 | Low | Section IV-C |
| S13 | Define "ringdown" at first use in main text body | Minor | R3-M3 | Low | Section I or IV-C |
| S14 | Discuss whether intermittent delay excursions (jitter) cause transient degradation or cumulative instability | Minor | R2 detailed comments | Low | Section V-E |

## Revision Roadmap
### Priority 1 — Structural Revisions (must complete before resubmission)
These items affect the paper's core claims, logical coherence, and statistical accuracy. Estimated effort: 12-28 hours.

- **R1:** Reconcile contradictory 95% CIs (1-2 h)
- **R2:** Resolve Zamani vs. Direct Torque contradiction (2-8 h text; 1-2 days if experiment needed)
- **R3:** Resolve k_p schedule architecture mismatch (2-4 h)
- **R4:** Add process-control positioning acknowledgment (2-3 h)
- **R5:** Correct test type from paired to independent (1-2 h)
- **R6:** Qualify AW distinctiveness claim (1-2 h)
- **R14:** Justify ~0.25 s servo lag figure (2-4 h)

### Priority 2 — Content Supplementation (strongly recommended)
These items add missing evidence, context, or analysis that strengthen the paper without changing its core claims. Estimated effort: 8-16 hours.

- **R7:** Hedge porting procedure generality claims (1 h)
- **R8-R9, R13:** Abstract and introduction text fixes (1 h total)
- **R10:** Explain L2 vs. S4 discrepancy (1 h)
- **R11:** Report LQR survival time SD correctly (30 min)
- **R12:** Revise platform comparison table (2-3 h)
- **S2:** Combined hardware degradation estimate (2-4 h)
- **S4:** Frequency-domain analysis of delay bifurcation (2-4 h)

### Priority 3 — Text and Formatting (should fix if time permits)
These items improve precision, completeness, and readability. Estimated effort: 4-8 hours.

- **S1:** Hedges' g correction (30 min)
- **S3:** Increase N or downgrade language for flight/terrain (1-4 h depending on path)
- **S5-S14:** Minor text improvements, citations, clarifications (3-4 h)

### Total Estimated Effort
24-52 hours spread across one revision cycle. The upper bound assumes the coupled-6-state-DT experiment is run; the lower bound assumes text-only resolution of the Zamani contradiction.

## Closing

This manuscript has made substantial progress across eight review rounds. The core empirical contribution — a proximity-gated anchor with asymmetric envelope follower delivering 0.73 mm idle precision with 80 N omnidirectional push survival — is genuine, well-ablated, and honestly contextualized. The factorial ablation design and statistical disclosure set a standard that the field should emulate.

The decision of Major Revision rather than Minor Revision reflects not a judgment that the paper is far from ready, but rather that the remaining issues — a factual CI inconsistency, a logical tension in the central bottleneck argument, and an architecture-description mismatch — require substantive resolution that goes beyond text-level adjustments. All are resolvable within one revision cycle. None challenges the validity of ACC's empirical performance data. The paper's distinguishing strength — its intellectual honesty — should be preserved throughout the revision; the 10 enumerated limitations, statistical power disclosure, and explicit labeling of qualitative demonstrations must not be weakened.

The authors are commended for their engagement with reviewer feedback across eight rounds. The paper is approaching RA-L quality.

---

## Appendix: Venue-by-Venue Detailed Assessment

### 1. IEEE Robotics and Automation Letters (RA-L)
- **Fit score:** 7/10
- **Key strengths:** Letter-format length is appropriate; empirical focus matches RA-L expectations; factorial ablation and statistical rigor exceed typical RA-L standards; 10 ms latency bifurcation finding is practically significant for the robotics community.
- **Key gaps:** CI inconsistency (factual error); Zamani contradiction in bottleneck argument; missing alternative-paradigm baseline (RL/MPC), though Limitation 10 now accepts this scope; "conditionally-gated" framing needs recalibration vis-a-vis process control.
- **Recommended action:** Major Revision then resubmit. The CI error, Zamani contradiction, and k_p schedule mismatch must be resolved. After these fixes, the paper is RA-L ready.

### 2. ICRA/IROS Conference
- **Fit score:** 7/10
- **Key strengths:** The 6-page format would compress the thorough ablation detail but the core results (0.73 mm precision, 80 N push, factorial ablation) would communicate clearly; the gate activation figure is conference-presentation ready; the practical tuning guidance has workshop/industry-track value.
- **Key gaps:** Conference reviewers may be less thorough than the current panel but may also be less forgiving of the CI inconsistency; the Zamani contradiction would be flagged in rebuttal; the LQR bottleneck argument is nuanced and may not survive compression to conference length.
- **Recommended action:** Ready with Revision. The shorter format reduces the impact of some statistical nuance issues. Fix CI, Zamani contradiction, and k_p schedule first.

### 3. IEEE/ASME TMECH
- **Fit score:** 6/10
- **Key strengths:** Practical engineering contribution aligns with TMECH's applied focus; the staged commissioning plan and hardware degradation discussion show deployment thinking; the 5-step porting procedure is practitioner-oriented.
- **Key gaps:** TMECH expects stronger analytical treatment (stability analysis, describing-function analysis of gate nonlinearity) for a contribution framed as a control architecture; process-control positioning issues (R3-W1, R3-W2) would face additional scrutiny from control-engineering reviewers; the absence of any formal stability guarantee for the multi-gate composite system is a gap; experimental validation is simulation-only, which is acceptable for TMECH but weakens the "hardware-aligned" narrative.
- **Recommended action:** Needs Major Revision. Add analytical treatment (at minimum, a describing-function analysis of the gate nonlinearity or a Lyapunov-based boundedness argument for the anchor mechanism). Strengthen hardware degradation analysis per R2-W3. Resolve all blocking issues first.

### 4. IEEE T-RO
- **Fit score:** 4/10
- **Key strengths:** The factorial ablation and statistical rigor would be appreciated; the contribution is genuinely novel within the wheeled-biped domain.
- **Key gaps:** T-RO expects: (a) formal stability analysis for a control architecture contribution, not just empirical validation; (b) hardware validation for claims of practical deployability; (c) comparison against optimization-based methods (MPC), not only classical LQR variants; (d) generalizability evidence across multiple platforms or morphologies. The current manuscript provides none of these. The DA's "missing stakeholder perspective" from a control-theorist and from an experimental roboticist with hardware experience are directly relevant to T-RO expectations.
- **Recommended action:** Not Ready. Would need: formal analysis, hardware validation, MPC baseline, and preferably multi-platform evidence. The gap between current state and T-RO standards is too large for a single revision cycle. Consider after hardware validation and RL+ACC co-training results.

### 5. Robotica
- **Fit score:** 7/10
- **Key strengths:** Empirically-focused, simulation-validated control contributions with practical orientation fit Robotica's scope; the 5-step tuning procedure and practical deployment discussion are valued; the honesty about limitations aligns with Robotica's applied ethos; the paper does not overclaim theoretical contributions.
- **Key gaps:** Same CI inconsistency and Zamani contradiction as for all venues; simulation-only validation is acceptable for Robotica but the hardware degradation projection would need tightening; the LQR baseline comparison, while thorough within paradigm, would benefit from at least one non-LQR reference point.
- **Recommended action:** Ready with Revision. Lower bar than T-RO or TMECH for formal analysis. Fix CI inconsistency, Zamani contradiction, and k_p schedule. The paper could be submitted to Robotica after one revision cycle.