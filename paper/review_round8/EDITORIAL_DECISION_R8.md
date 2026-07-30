# Editorial Decision -- Round 8

**Paper:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"
**Author:** Van Thuong Nguyen
**Decision Date:** 2026-07-28
**Decision Type:** Minor Revision
**Synthesized from:** EIC + R1 (Methodology) + R2 (Domain Expert) + R3 (Cross-Disciplinary / Process Control) + DA (Devil's Advocate)

---

## 1. Consensus Matrix

| Reviewer | Recommendation | Confidence | Key Strength Emphasized |
|----------|---------------|------------|------------------------|
| **EIC** | Minor Revision | 4 | Asymmetric envelope follower + proximity-gated anchor is genuinely novel; factorial ablation design is exemplary; 10-item limitations section is exceptional candor |
| **R1 (Methodology)** | Minor Revision | 4 | Pre-registered thresholds; honest statistical power analysis; complete reproducibility infrastructure (scripts, configs, commit hash, raw data); correct P2 fix application |
| **R2 (Domain)** | Minor Revision | 4 | Rigorous LQR bottleneck analysis with controlled ablations; asymmetric envelope follower is genuine innovation; 46-configuration gain sweep is thorough; honest statistical reporting |
| **R3 (Cross-Disc)** | Minor Revision | 4 | Honest engagement with process control lineage; coupled 6-state 3D LQR is fair comparison; Bonferroni context properly integrated; parameter sensitivity hierarchy is practically valuable |
| **DA** | N/A (stress-test role) | N/A | Disclosure quality exceptionally high; factorial ablation methodologically strong; gate activation figure pedagogically excellent; gate parameter sensitivity hierarchy is a genuine contribution |

**Consensus:** All four recommending reviewers converge on **Minor Revision**. The DA report identifies structural concerns (C1, C2) that warrant careful attention but does not dispute the core empirical contribution. No reviewer recommends Reject or Major Revision.

**Weighted Score Range:** 71 (R3) -- 81 (R1). Median: 77.3. The spread is attributable to R3's lower scores on control-theoretic soundness (58) and literature integration (65), reflecting the process-control perspective that the architectural framing overclaims relative to established practice. The other three reviewers score the paper in the 76-81 range.

---

## 2. Weakness Sub-Claim Inventory

Each reviewer weakness is decomposed into atomic sub-claims. Consensus level is marked as:
- **STRONG CONSENSUS** (4-5/5 reviewers)
- **CONSENSUS** (3/5)
- **DIVERGENT** (2/5 disagree)
- **SINGLE-REVIEWER**

| ID | Sub-Claim | EIC | R1 | R2 | R3 | DA | Consensus |
|----|-----------|-----|----|----|----|----|-----------|
| **SC-W1** | Alternative-paradigm baseline (RL/MPC/DOB) absent; abstract LQR failure statement needs scoping qualifier | W1 | -- | -- | -- | M2 (partial) | CONSENSUS (EIC+DA; R1/R2/R3 note but don't elevate to weakness) |
| **SC-W2** | "Dominant bottleneck" claim overstated relative to Direct Torque evidence (35% improvement, 100% fall persists) | W2 | -- | -- | -- | m4 | CONSENSUS (EIC+DA; R2 W2 related) |
| **SC-W3** | "ACC as precondition for RL" claim unevidenced (zero RL+ACC experiment) | W3 | -- | -- | -- | M2 | CONSENSUS (EIC+DA) |
| **SC-W4** | Abstract omits push degradation under delay (F_max collapses from ~95N to ~45N) | Minor Issue | -- | -- | -- | m2 | CONSENSUS (EIC+DA) |
| **SC-W5** | Platform comparison table (Table I) asymmetrically reports quantitative metrics for ACC, qualitative for competitors; missing capability rows | W4 | -- | W1 | -- | -- | CONSENSUS (EIC+R2) |
| **SC-W6** | Latency sensitivity bifurcation (<10 ms): hardware latency budget (~9.5 ms) estimated, not validated; 0.5 ms margin unrealistic | W5 | -- | Comments | -- | M5 | CONSENSUS (EIC+R2+DA) |
| **SC-W7** | 95% CI inconsistency: abstract reports [0.68, 0.78] mm vs Table IV footnote reports [0.50, 0.62] mm -- intervals do not overlap | -- | W1 | Minor 11 | -- | -- | CONSENSUS (R1+R2) |
| **SC-W8** | "Paired t-test" claimed for independent samples; should be Welch's t-test with df=8 | -- | W2 | -- | -- | -- | SINGLE-REVIEWER (R1) |
| **SC-W9** | Effect sizes reported as Cohen's d without Hedges' g small-sample correction for N=5 | -- | W3 | -- | -- | -- | SINGLE-REVIEWER (R1) |
| **SC-W10** | LQR baseline survival time SD = 0.00 implausible given randomized initial conditions across 60 episodes | -- | W4 | -- | -- | -- | SINGLE-REVIEWER (R1) |
| **SC-W11** | L2 (global-I, works) vs S4 (global-I, windups) discrepancy unexplained -- likely boost interaction | -- | W5 | -- | -- | -- | SINGLE-REVIEWER (R1) |
| **SC-W12** | Direct-torque LQR signal under-analyzed: 35% survival improvement + 10x torque reduction are significant; coupled 6-state direct-torque experiment missing | -- | -- | W2 | -- | -- | SINGLE-REVIEWER (R2) |
| **SC-W13** | Contact stiffness sensitivity analysis incomplete: single-factor (solref only), no combined degradation estimate, envelope band retuning for compliant surfaces not analyzed | -- | -- | W3 | -- | -- | SINGLE-REVIEWER (R2) |
| **SC-W14** | Flight/terrain demonstrations below evidence standards: N=3-4 insufficient even for "qualitative feasibility proof"; Clopper-Pearson CI [0.29, 1.0] | -- | -- | W4 | -- | -- | SINGLE-REVIEWER (R2) |
| **SC-W15** | "Conditionally-gated control" relabels established process control practice (split-range, override with bumpless transfer) without fully acknowledging depth of prior art | -- | -- | -- | W1 | -- | SINGLE-REVIEWER (R3) |
| **SC-W16** | AW taxonomy distinctiveness claim ("gating on physical state rather than control error") overstates novelty; state-gated conditional integration already practiced in industry | -- | -- | -- | W2 | -- | SINGLE-REVIEWER (R3) |
| **SC-W17** | 5-step porting procedure generality claims untested; single-platform validation, single-mode ringdown assumption | -- | -- | -- | W3 | -- | SINGLE-REVIEWER (R3) |
| **SC-W18** | k_p schedule "retained for completeness" despite contradicting data (S1 outperforms S0); creates architecture-description mismatch | -- | -- | -- | W5 | C1 | CONSENSUS (R3+DA; EIC notes fix was applied but doesn't re-flag) |
| **SC-W19** | Zamani comparison now contradicts bottleneck claim: Zamani succeeds without servo lag, but ACC Direct Torque LQR (also no servo lag) fails 100% | -- | -- | -- | -- | C2 | SINGLE-REVIEWER (DA; R2 W2 related but doesn't identify contradiction) |
| **SC-W20** | Gate-state bifurcation (partial-engagement regime worse than P-only) is systematic fragility, not tuning corner case; controller has no mechanism to detect/avoid pathological middle regime | -- | -- | -- | -- | M1 | SINGLE-REVIEWER (DA) |
| **SC-W21** | LQR 46-config gain sweep validates decoupled PD parameter space, not coupled 6-state LQR's Q/R matrices; sweep scope doesn't match claim | -- | -- | -- | -- | M3 | SINGLE-REVIEWER (DA) |
| **SC-W22** | 0.25s servo lag figure unjustified: no PID gain values, step response data, or settling time measurement provided | -- | -- | -- | -- | M4 | SINGLE-REVIEWER (DA; R2 comments echo concern) |
| **SC-W23** | 50+ hand-tuned gains without systematic tuning methodology; reproducibility concern | -- | -- | -- | -- | M6 | SINGLE-REVIEWER (DA; carried from Round 7 M4) |

**Summary:** The only sub-claims reaching CONSENSUS (3+ reviewers) are SC-W1 through SC-W7. The remaining 16 sub-claims are SINGLE-REVIEWER findings, reflecting the diverse expertise across the panel (methodology, wheeled-biped domain, process control, adversarial review). This distribution is healthy -- it means each specialist found issues in their domain that others did not, rather than multiple reviewers independently identifying the same fundamental flaws.

---

## 3. DA CRITICAL Adjudication

### C1: k_p Schedule Architectural Inconsistency

**DA Claim:** S1 (fixed k_p=50) outperforms S0 (scheduled k_p) on push survival (83.1 N vs 80.4 N), yet the scheduled variant remains in the architecture diagram, equation (1), and defines the "Full ACC" (S0) reference configuration. "Retained for completeness" is not a scientific justification. The paper describes an architecture it does not recommend.

**Adjudication: VALIDATED -- Severity downgraded from CRITICAL to MAJOR.**

**Rationale:**

The DA's factual premise is correct and verified against the manuscript:
- Table IV, S0 vs S1: S1 (fixed k_p=50) achieves F_min=83.1 N, F_med=118.8 N vs S0 (scheduled k_p) at 80.4 N, 110.1 N.
- Line 188: "Fixed k_p=50 is recommended: the scheduled variant...did not improve push survival over fixed k_p=50 (S1 vs. S0, Table IV)."
- Line 189: "The schedule is retained for completeness."
- Equation (1) does not directly contain k_p(h,v) -- the k_p schedule appears in Eq. 6 (balance core), not in the two-channel assembly equation (Eq. 1). However, the architecture figure (Fig. 2) and the "Full ACC" reference configuration (S0) do include it.

The DA's argument that "the paper describes an architecture it does not recommend" is logically sound. A reader encountering "Full ACC" as S0 expects S0 to be the recommended configuration; finding that S1 (a variant the paper itself recommends) outperforms S0 creates confusion.

However, the severity is MODERATE rather than CRITICAL for three reasons:
1. The paper **honestly discloses** the finding rather than hiding it. The Round 7 DA-C1 fix was explicitly to reconcile the scheduled k_p with the ablation data, and the paper now does so transparently.
2. The scheduled variant is presented as an **optional softening mechanism** within the balance core, not as a structurally distinct component. Removing it would change one parameter (k_p becomes constant 50) without altering the architecture's component inventory.
3. Limitation 7 (line 676-677) explicitly acknowledges the bidirectional coupling and implicit algebraic loop that the scheduled variant introduces, further undermining any claim that the schedule is essential.

**Verdict:** The scheduling mechanism should either be (a) removed from the canonical architecture (re-label S1 as "Full ACC," delete Eq. 6, simplify text), or (b) moved to an appendix with a clear statement that it is deprecated in favor of fixed k_p=50. "Retained for completeness" is insufficient for archival publication. This is a text-level architectural cleanup, achievable in one revision cycle.

---

### C2: Zamani / Direct Torque Logical Contradiction

**DA Claim:** The deepened Zamani comparison (P2-R12 fix) argues that Zamani's coupled LQR succeeds because their platform is direct-drive (no PID servo lag), and ACC's 6-state LQR fails because of servo lag. But ACC's Direct Torque LQR variant -- which also has no PID servo lag -- still fails 100%, with only 35% survival improvement. The three data points are mutually inconsistent: (1) Zamani works without servo lag; (2) ACC's 6-state LQR fails WITH servo lag; (3) ACC's Direct Torque LQR fails WITHOUT servo lag. These cannot all be explained by "servo lag is the dominant bottleneck."

**Adjudication: VALIDATED -- Severity confirmed as MAJOR.**

**Rationale:**

The DA has identified a genuine logical tension. The three data points are:

| Data Point | Platform | Servo Lag? | Outcome |
|-----------|----------|------------|---------|
| Zamani et al. (2020) | Rigid two-wheeled, direct-drive | No | Survives (theta_RMS ~1.5 deg) |
| ACC 6-State LQR | Legged morphology, PID servo | Yes | Falls in 0.52 s (100%) |
| ACC Direct Torque LQR | Legged morphology, direct torque | No | Falls in 0.70 s (100%) |

If servo lag were the sole explanatory variable, Data Point 3 should look more like Data Point 1 than Data Point 2. It does not -- Direct Torque LQR improves only 35% and still falls. Therefore, platform differences beyond servo architecture (rigid body vs articulated legs, mass distribution, actuator authority) must be contributing factors that the paper's current narrative does not acknowledge.

The paper's text at lines 87-88 states: "...the ~0.25 s servo lag -- absent in Zamani's direct-drive platform -- introduces phase margin loss that the coupled LQR cannot overcome, confirming the servo path as the dominant bottleneck." This sentence implies that the **presence** of servo lag in ACC and its **absence** in Zamani's platform is the key difference explaining the outcome gap. But the Direct Torque data shows that removing servo lag from ACC's platform does not close the gap. The sentence is therefore incomplete: it correctly identifies servo lag as **a** bottleneck but incorrectly implies it is **the** distinguishing factor between the two platforms' outcomes.

The paper does not currently resolve this tension:
- Option (a): Acknowledge that platform differences (rigid two-wheeled body, no articulated legs, different mass/inertia, different actuator authority) are likely primary factors in Zamani's success, and the servo-lag comparison is therefore suggestive rather than conclusive.
- Option (b): Note that Direct Torque LQR gains were optimized for the servo plant, not re-optimized for the direct-torque plant -- making the Direct Torque result a conservative lower bound.
- Option (c): Run the coupled 6-state direct-torque experiment (deferred to future work, line 596).

**Verdict:** The paper must explicitly resolve this contradiction. The minimum fix is to add 1-2 sentences acknowledging that while servo lag is the dominant bottleneck **among the factors tested on ACC's platform**, the Zamani comparison also reflects platform-level differences (rigid body, direct-drive actuators, different mass distribution) that the servo-lag variable alone cannot explain. Option (b) -- noting that Direct Torque gains were not re-optimized -- would also partially resolve the tension by acknowledging the Direct Torque result as conservative. The current text implies a clean causal chain (Zamani succeeds because no servo lag -> ACC fails because servo lag) that the Direct Torque data contradicts.

---

## 4. Round 7 P2 Fix Verification Summary

| Fix ID | Description | Status | Verifier(s) | Notes |
|--------|-------------|--------|-------------|-------|
| **P2-R9** | Non-monotonic noise explained (partial-engagement resonance) | **RESOLVED** | EIC, R1, DA | Text-level fix adequate. DA notes the explanation reveals deeper fragility (see SC-W20) but the disclosure itself is sufficient. |
| **P2-R10** | Contact stiffness sensitivity quantified | **PARTIALLY RESOLVED** | EIC, R2 | solref[0]=0.002->0.02 experiment done (~60% degradation). R2 W3 identifies missing: combined degradation analysis, envelope band retuning for compliant surfaces, and combined hardware degradation estimate. |
| **P2-R11** | Bonferroni-adjusted thresholds adjacent to p-values | **RESOLVED** | EIC, R1, R3 | Bonferroni alpha=0.007 in Table IV footnote, cross-referenced to Section VI. All three verifiers confirm. |
| **P2-R12** | Zamani et al. comparison deepened with quantitative metrics | **RESOLVED** (text) / **NEW TENSION** (logic) | EIC, DA | Quantitative metrics added (theta_RMS ~1.5 deg, phi_RMS ~2.0 deg). However, DA C2 identifies that the deepened comparison now creates a logical contradiction with the Direct Torque data. The text-level fix is complete; the logical tension is a new issue. |
| **P2-R13** | 46-configuration sweep arithmetic corrected (35+5+6=46) | **RESOLVED** | EIC, R1, DA | Arithmetic now consistent. DA M3 notes a separate concern: sweep tests decoupled PD parameters, not coupled 6-state LQR Q/R matrices -- but this is a scope issue, not an arithmetic error. |

**Round 7 P0 Critical Fix Verification:**

| Fix ID | Description | Status |
|--------|-------------|--------|
| **P0-R1** | Reconcile scheduled k_p with ablation data (DA-C1, Round 7) | **RESOLVED** (disclosure) / **NEW CONCERN** (architectural inconsistency, DA C1 Round 8) |
| **P0-R2** | Recalibrate LQR bottleneck claim (DA-C2, Round 7) | **PARTIALLY RESOLVED** -- "THE bottleneck" -> "the dominant bottleneck" is improved but still overstated per EIC W2, DA m4; new logical contradiction with Zamani comparison (DA C2, Round 8) |
| **P0-R3** | Add alternative-paradigm baseline (SC1, Rounds 6-7) | **DEFERRED** -- scoped via Limitation 10; abstract still unqualified (EIC W1) |

---

## 5. Editorial Decision

### Final Recommendation: **Minor Revision**

### Score: **76 / 100**

**Breakdown:**
- Originality: 78 (novel synthesis for wheeled bipeds; individual components have process-control lineage)
- Methodological Rigor: 82 (exemplary factorial ablation, pre-registered thresholds, comprehensive LQR comparison)
- Evidence Sufficiency: 72 (core claims well-supported; flight/terrain under-evidenced; alternative-paradigm baseline absent)
- Argument Coherence: 80 (clear narrative; two unresolved logical tensions -- DA C1, DA C2 -- prevent 85+)
- Writing/Presentation: 75 (clear technical prose; CI inconsistency, platform table asymmetry, and process-control framing issues)
- Practical Significance: 68 (sub-millimeter precision + push survival addresses real trade-off; latency sensitivity <10 ms is serious deployability barrier; generality untested)

### Rationale (275 words)

This manuscript has traversed eight review rounds and emerged as a substantially stronger paper. The core contribution -- a proximity-gated anchor with asymmetric envelope follower enabling sub-millimeter idle precision (0.73 mm CoM RMS) without sacrificing omnidirectional push survival (F_min=80 N) -- is genuine, well-validated by factorial ablation, and honestly contextualized within 10 enumerated limitations. All five Round 7 P2 fixes are correctly implemented, and both Round 7 DA CRITICAL findings are addressed in the text.

The panel converges on Minor Revision. Three issues require attention before acceptance:

First, two logical tensions introduced or sharpened by the Round 7 revisions must be resolved: (a) the k_p schedule is "retained for completeness" despite the paper's own data showing fixed k_p=50 outperforms it -- creating an architecture that the paper does not recommend; and (b) the deepened Zamani comparison now argues servo lag is the distinguishing bottleneck, but the paper's own Direct Torque data shows removing servo lag does not prevent failure -- creating an unresolved contradiction about what distinguishes Zamani's success from ACC's LQR failure.

Second, a genuine CI inconsistency exists between the abstract (95% CI [0.68, 0.78] mm) and Table IV footnote ([0.50, 0.62] mm) for the headline idle precision claim. These intervals do not overlap and must be reconciled.

Third, several framing issues need adjustment: the abstract's LQR failure statement needs scoping to "among classical LQR variants evaluated"; the "ACC as precondition for RL" claim needs rephrasing as a hypothesis; the platform comparison table needs asymmetry mitigation; and the process-control lineage (split-range, override with bumpless transfer) needs fuller acknowledgment.

These are all text-level or single-experiment fixes achievable within one revision cycle. The empirical contribution is not in question.

---

## 6. Revision Roadmap

### P0 (BLOCKING) -- Must fix before resubmission

| ID | Description | Effort | Raised By |
|----|-------------|--------|-----------|
| **P0-R1** | Reconcile 95% CI inconsistency: abstract [0.68, 0.78] mm vs Table IV footnote [0.50, 0.62] mm. Determine which CI is correct, apply consistently throughout paper, document the variance estimator used. | 1-2 hours | R1 (W1), R2 (Minor 11) |
| **P0-R2** | Resolve Zamani / Direct Torque logical contradiction (DA C2). Add explicit acknowledgment that (a) platform differences beyond servo architecture contribute to Zamani's success, and/or (b) Direct Torque LQR gains were not re-optimized for the direct-torque plant, making results conservative. The "dominant bottleneck" claim must be reconciled with the Direct Torque 100% fall rate. | 2-4 hours (text) | DA (C2), EIC (W2) |
| **P0-R3** | Resolve k_p schedule architectural inconsistency (DA C1). Either: (a) remove schedule from canonical architecture, re-label S1 as "Full ACC," delete or move Eq. 6 to appendix; or (b) provide a principled, data-supported reason to retain it. "Retained for completeness" is insufficient. | 2-3 hours (text + figure update) | DA (C1), R3 (W5) |

### P1 (HIGH) -- Strongly recommended

| ID | Description | Effort | Raised By |
|----|-------------|--------|-----------|
| **P1-R4** | Add scoping qualifier to abstract LQR failure statement: "...all fail (100% fall rate) among the classical LQR variants evaluated." Aligns abstract with Limitation 10. | 15 minutes | EIC (W1) |
| **P1-R5** | Add push degradation to abstract: mention F_max collapse from ~95 N to ~45 N under 10 ms delay alongside the precision degradation (2.30 mm). Currently the abstract selectively reports only precision impact. | 15 minutes | EIC (Minor Issue), DA (m2) |
| **P1-R6** | Rephrase "ACC is thus a precondition for RL" (line 66) as "ACC is designed to serve as a height-general structured prior that a residual policy can later augment -- a hypothesis pending RL+ACC co-training experiments." | 15 minutes | EIC (W3), DA (M2) |
| **P1-R7** | Fix "paired t-test" to independent-samples Welch's t-test (or document the pairing mechanism if measurements are genuinely paired). Report test type and degrees of freedom explicitly. | 1 hour | R1 (W2) |
| **P1-R8** | Mitigate platform comparison table asymmetry (Table I). Add: (a) "Locomotion speed" row; (b) change Ascento push recovery to "WBC (qual.)"; (c) footnote clarifying "---" means "not reported in cited work, not that capability is absent"; (d) verify SKATER "Contin." height entry against citation. | 2-3 hours | R2 (W1), EIC (W4) |
| **P1-R9** | Add explicit acknowledgment that continuous blending of control modes is standard in split-range control (Shinskey 1996, Ch. 7) and bumpless-transfer override control (Astrom & Hagglund 2005, Sec. 10.5). Position ACC as a specific, well-engineered application of these principles to wheeled-biped balancing with physically-motivated gating conditions. | 1-2 hours (text) | R3 (W1) |
| **P1-R10** | Qualify AW taxonomy distinctiveness claim (Appendix A). Acknowledge that state-gated conditional integration is practiced in industry (e.g., temperature-zone-gated integrals in batch reactors). Distinguish ACC's anchor by: (i) asymmetric velocity envelope as gating variable, and (ii) coupled integral-plus-damping boost mechanism. | 1 hour | R3 (W2) |
| **P1-R11** | Hedge 5-step porting procedure generality claims. Add: "This procedure has been validated only on the 8.1 kg, 0.54 m CoM platform; validation on substantially different morphologies is future work." Acknowledge single-mode ringdown assumption. | 30 minutes | R3 (W3) |
| **P1-R12** | Clarify that the 46-configuration LQR gain sweep tests the decoupled PD parameter space, not the coupled 6-state LQR's Q/R matrices. Add a sentence: "The sweep covers the decoupled-PD parameter space; a sweep over the 6-state LQR's Q/R cost matrices (which determine Riccati-derived gains) is deferred." | 30 minutes | DA (M3), R2 (Detailed Comment) |

### P2 (MEDIUM) -- Improves quality

| ID | Description | Effort | Raised By |
|----|-------------|--------|-----------|
| **P2-R13** | Report Hedges' g alongside or instead of Cohen's d for N=5 effect sizes, noting the small-sample correction factor (~10% reduction). | 30 minutes | R1 (W3) |
| **P2-R14** | Address LQR survival time zero-variance: either report at higher precision (0.52 +/- 0.003 s), note discretization at 0.01 s timestep, or confirm zero variance is genuine via re-measurement. | 1 hour | R1 (W4) |
| **P2-R15** | Explain L2 vs S4 global integral discrepancy: L2 (additive chain, no boost) survives with global integral; S4 (subtractive chain, retains boost) windups. Likely explanation: damping boost amplifies integral windup. Add one-sentence footnote. | 30 minutes | R1 (W5) |
| **P2-R16** | Provide combined hardware degradation estimate (encoder quantization + backlash + tire compliance + IMU noise) via Monte Carlo or worst-case linear sum, replacing individual bounds. Tighten the 1-3 mm projected range. | 4-8 hours (simulation) | R2 (W3) |
| **P2-R17** | Increase flight/terrain N to at least 10 for key conditions (ledge 50 cm, 20 cm curb) OR downgrade language from "feasibility proof" to "preliminary qualitative demonstration." | 4-8 hours (simulation) if increasing N; 30 minutes if downgrading language | R2 (W4) |
| **P2-R18** | Discuss gate-state bifurcation (partial-engagement regime worse than P-only) as a fundamental limitation of threshold-based gating, not merely an anomaly. Note that the controller has no mechanism to detect or avoid the pathological middle regime. | 1 hour (text) | DA (M1) |
| **P2-R19** | Provide derivation/measurement for ~0.25 s servo lag figure: report PID gains, show step-response measurement, or acknowledge conservative tuning. Without this, the bottleneck claim rests on an unverified assumption. | 2-4 hours (measurement) | DA (M4), R2 (Detailed Comment) |
| **P2-R20** | Document ablation config paths in supplementary material (e.g., `configs/ablations/` or appendix listing parameter modifications for L0-L3, S0-S5). | 1 hour | R1 (Minor 5) |
| **P2-R21** | Cite proprioceptive state estimation literature for wheeled bipeds (SKATER, Ascento WBC) where velocity and contact estimation are discussed. | 1 hour | R2 (Detailed Comment) |
| **P2-R22** | Deepen Hyon et al. (2007) comparison: 2-3 sentences explaining what ACC's smoothstep gates add beyond discrete switching, and whether Hyon's support-polygon gating has any analogue in ACC's proximity/envelope gates. | 1 hour | R2 (Detailed Comment) |
| **P2-R23** | Move statistical power discussion to appear before first results table (beginning of Section V), so readers have context when interpreting Tables III-VI. | 30 minutes | R2 (Minor 9) |
| **P2-R24** | Qualify partial-engagement resonance explanation as post-hoc hypothesis: "we hypothesize" or "a likely mechanism is..." to distinguish from pre-registered threshold tests. | 15 minutes | R1 (Minor 7), DA (P2-R9 assessment) |
| **P2-R25** | Report gate sensitivity perturbation size (e.g., +/- 1% or +/- 0.01) to make finite-difference gradient magnitudes interpretable. | 15 minutes | R1 (Minor 4) |
| **P2-R26** | Clarify Bryson's rule Q_roll=3.0 vs expected 44.4: explain rationale or recompute with consistent scaling. | 30 minutes | R2 (Detailed Comment) |
| **P2-R27** | Cite envelope-detection signal processing lineage (e.g., Giannoulis et al., 2012, "Digital Dynamic Range Compressor Design") for the asymmetric EMA. | 30 minutes | R3 (M4) |
| **P2-R28** | Add frequency-domain justification for 10 ms delay bifurcation: the additional phase lag at 100 Hz from a 10 ms delay is 36 degrees -- a brief analysis would strengthen the discussion. | 2-3 hours | R2 (Detailed Comment) |
| **P2-R29** | Discuss whether intermittent delay excursions (jitter at 12-15 ms) cause transient degradation or cumulative instability, given that real embedded systems have 1-5 ms scheduling jitter. | 1-2 hours (text) | R2 (Detailed Comment), DA (M5) |

### P3 (LOW) -- Optional improvements

| ID | Description | Effort | Raised By |
|----|-------------|--------|-----------|
| **P3-R30** | Clarify C0 vs C1 practical implications: C0 continuity ensures torque continuity; torque rate limit (400 Nm/s) provides additional smoothing. | 15 minutes | R3 (W4) |
| **P3-R31** | Consider removing or demoting the scheduled-k_p variant entirely (related to P0-R3). | (covered by P0-R3) | R3 (W5) |
| **P3-R32** | Define "ringdown" at first use in main text body. | 15 minutes | R3 (M3) |
| **P3-R33** | Report gate sensitivity as elasticity (percentage change per percentage change in parameter) for cross-parameter comparability. | 30 minutes | R3 (M5) |
| **P3-R34** | Add footnote to high-noise robustness rows noting that F_max in Table VI (forward push only) is not directly comparable to F_min in Table IV (omnidirectional, worst-case direction). | 15 minutes | DA (m1) |
| **P3-R35** | Verify all references are cited in text (dicarlo2018mpc, ferreira2021ollie, lee2020rloc, kim2017velocity flagged as potentially uncited). | 30 minutes | EIC (Minor Issue), R2 (Minor 10) |
| **P3-R36** | Verify compact spacing overrides (lines 16-27) comply with target journal style guide. | 15 minutes | EIC (Minor Issue) |
| **P3-R37** | Add cross-reference to AW validation script in Table V footnote for standalone readability. | 15 minutes | R1 (Minor 3) |
| **P3-R38** | Specify 7.7x factor derivation (ratio of LQR-optimal kp_roll to decoupled PD kp_roll=0.4) explicitly in text. | 15 minutes | R2 (Minor 2) |
| **P3-R39** | Note that EMA init transient (~7.5 s) exceeds 5 s settle period; show supplementary time-series confirming stabilization by t=5 s. | 1-2 hours | R1 (Minor 1) |

---

## 7. Comparison with Round 7

### What Improved

1. **All five P2 fixes correctly implemented.** P2-R9 (non-monotonic noise), P2-R10 (contact stiffness sensitivity), P2-R11 (Bonferroni context), P2-R12 (Zamani comparison deepened), and P2-R13 (46-config arithmetic) are all verifiable against the manuscript. The EIC, R1, and R3 confirm these independently.

2. **DA CRITICAL findings from Round 7 addressed.** DA-C1 (scheduled k_p contradiction): the paper now explicitly states fixed k_p=50 is recommended, with S1 vs S0 data in Table IV. DA-C2 (bottleneck overstatement): softened from "THE bottleneck" to "the dominant bottleneck." The SC1 alternative-baseline consensus is now scoped via Limitation 10.

3. **Process control engagement significantly deepened.** The paper now cites Shinskey (1996) and Astrom & Hagglund (2005, 2006) for override/selector-based control, and explicitly positions ACC's smoothstep gates as extending "from discrete selection to continuous blending." R3 acknowledges this as "a genuine improvement."

4. **LQR baseline comparison substantially strengthened.** The coupled 6-state 3D LQR baseline, 46-configuration gain sweep (now with correct arithmetic), and controlled ablations (servo path, integral channel, state dimensionality) constitute one of the most thorough classical-control baseline evaluations seen in a wheeled-biped paper.

5. **Statistical reporting matured.** Bonferroni-adjusted thresholds, Cohen's d effect sizes, explicit minimum-detectable-effect analysis, Clopper-Pearson CIs for small-N binary outcomes, and honest acknowledgment of which comparisons survive correction are now integrated throughout.

6. **Limitations section expanded.** Now 10 items, covering statistical power, latency sensitivity, compound-disturbance degradation, scheduled-k_p algebraic loop, architectural generality, and baseline comparison scope.

### What Stayed the Same

1. **Alternative-paradigm baseline still absent.** Despite being the strongest consensus item across Rounds 6-7 (SC1, 4/5 reviewers), no RL/MPC/DOB baseline has been added. The paper now scopes claims via Limitation 10, which is acceptable for a letter-format submission if the abstract is similarly scoped -- but the abstract still states the LQR failure without qualification.

2. **"ACC as precondition for RL" claim persists.** Identified as M3 in Round 7 and listed as P3-R18, this claim remains in Section I unchanged. Zero RL+ACC experimental evidence exists.

3. **0.25 s servo lag figure still unjustified.** Raised as m5 in Round 7, carried forward as DA M4 in Round 8. No PID gain values, step response data, or settling time measurement have been provided.

4. **50+ hand-tuned gains without systematic methodology.** Raised as M4 in Round 7, carried forward as DA M6. The gate parameter sensitivity analysis (identifying alpha_r as dominant) partially addresses this but does not document the tuning process for the remaining parameters.

5. **Flight/terrain demonstrations remain at low N.** N=3-4 for key conditions persists from earlier rounds. The Clopper-Pearson CI [0.29, 1.0] is now explicitly reported, which is an improvement in transparency but does not strengthen the evidence.

### What Newly Emerged

1. **CI inconsistency (P0-R1).** The abstract reports 95% CI [0.68, 0.78] mm for idle precision while Table IV footnote reports [0.50, 0.62] mm. These intervals do not overlap. This is a new finding by R1 that was not identified in any previous round. It may be a reporting error (two different variance estimators used) or a genuine discrepancy requiring reconciliation. Either way, it must be fixed -- it affects the paper's headline quantitative claim.

2. **Zamani / Direct Torque logical contradiction (DA C2).** This tension was not present in the Round 7 version because the Zamani comparison was less specific. The P2-R12 fix -- which deepened the Zamani comparison with quantitative metrics and the explicit servo-lag mechanism claim -- inadvertently created a contradiction with the Direct Torque data. This is a new issue created by a Round 7 fix. It requires explicit resolution (see P0-R2).

3. **k_p schedule architectural inconsistency (DA C1).** While the Round 7 DA-C1 fix correctly reconciled the scheduled k_p with the ablation data (the paper now honestly reports that fixed k_p=50 is recommended), the fix itself created a new problem: the paper now describes an architecture (S0 = "Full ACC" with scheduled k_p) that its own data shows should not be used (S1 = fixed k_p=50 is recommended). This is a new framing issue requiring architectural cleanup (see P0-R3).

4. **"Paired t-test" statistical concern (R1 W2).** R1's methodology review identified that the push_ci.py protocol uses independent random seeds per configuration, making the measurements independent, not paired. The test type affects standard errors and p-values. This is a new statistical reporting issue not identified in prior rounds.

5. **Process-control framing precision (R3 W1, W2).** R3's cross-disciplinary review identified that the paper's claims about novelty relative to process control practice (split-range continuous blending, state-gated conditional integration) overstate the distinctiveness of the architectural concept while understating the depth of prior art. This is a new dimension of scrutiny not present in earlier rounds, which lacked a dedicated process-control reviewer.

6. **Platform comparison table issues (R2 W1).** R2's domain-expert review identified several characterizations in Table I that are overly favorable to ACC or omit documented capabilities of comparison platforms. These issues were not flagged in earlier rounds.

**Net assessment:** The paper has improved substantially in disclosure quality, statistical rigor, and LQR baseline thoroughness. However, the Round 7 fixes have introduced two new logical tensions (DA C1, DA C2) that must be resolved, and the addition of specialist reviewers (R1 methodology, R3 process control, R2 domain expert) has surfaced new issues in their respective domains that earlier generalist review rounds did not catch. The overall trajectory is strongly positive -- eight rounds of revision have produced a substantially stronger manuscript -- but the three P0 blocking items must be addressed before acceptance.

---

*Editorial decision prepared: 2026-07-28. Synthesized from 5 independent Phase 1 reviewer reports. Every synthesis point traces to a specific reviewer report as documented in the Weakness Sub-Claim Inventory (Section 2). No new review comments have been fabricated.*
