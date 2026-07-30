# Peer Review Report -- EIC (Round 8)

## Reviewer Identity
Senior Editor, *IEEE Robotics and Automation Letters*, specializing in novel control architectures for legged and wheeled mobile robots. Formerly led the Dynamic Locomotion group at a European robotics institute. Has handled review cycles for Ascento, ANYmal, and SKATER submissions.

---

## Overall Assessment

### Recommendation
**Minor Revision**

### Confidence Score
**4** -- High confidence. The control architecture (gated PD + asymmetric envelope follower + anchor integral) falls squarely within my expertise. Confidence is bounded at 4 rather than 5 because all evidence remains simulation-only, and the projected real-world transfer hinges on latency and velocity estimation assumptions that are plausible but unverified.

### Summary Assessment
This eighth-round manuscript has addressed all five Round 7 P2 fixes correctly, resolved both DA CRITICAL findings (DA-C1 and DA-C2) from Round 7, and materially improved framing, clarity, and evidentiary support across the board. The scheduled k_p is now honestly demoted to an incidental variant (fixed k_p=50 is the recommended configuration, and S1 data showing it outperforms S0 is explicitly discussed). The bottleneck claim has been recalibrated from "THE bottleneck" to "the dominant bottleneck," which is a defensible intermediate position given the Direct Torque and coupled 6-state LQR evidence. The five P2 fixes -- non-monotonic noise explanation via partial-engagement resonance (P2-R9), contact stiffness sensitivity quantification (P2-R10), Bonferroni-adjusted thresholds adjacent to p-values (P2-R11), deepened Zamani et al. comparison with quantitative coupling to servo-lag hypothesis (P2-R12), and clarified 46-configuration sweep arithmetic (P2-R13) -- are all correctly implemented and verifiable against the manuscript text.

The one remaining substantive gap is the absence of an alternative-paradigm baseline (RL/MPC/DOB), which was the strongest consensus item across Rounds 6 and 7 (SC1, 4/5 reviewers). The paper now explicitly scopes its comparative claims to classical LQR variants (Limitation 10), which narrows the contribution claim acceptably for a letter-format submission. However, the abstract still states "Four classical LQR baselines...all fail (100% fall rate)" without the scoping qualifier present in the limitations section -- a minor framing misalignment that should be corrected.

The engineering contribution is genuine, the ablation methodology is exemplary, and the intellectual honesty (10 enumerated limitations, statistical power disclosure, explicit labeling of qualitative feasibility demonstrations) remains a distinguishing strength. The manuscript is approaching RA-L quality; the remaining issues are text-level framing adjustments achievable within one revision cycle.

---

## Strengths

1. **S1: Asymmetric envelope follower with proximity-gated anchor remains the genuinely novel core contribution.** The mechanism -- fast-attack (30 ms) / slow-release (1.5 s) EMA distinguishing quiet stance from post-push ringdown, coupled with spatial proximity gating (15 cm neighborhood) and a pitch-stability-gated damping boost -- has no direct analogue in the wheeled biped control literature. The symmetric-EMA ablation confirming bistable failure, the S5 parametric oscillation finding (boost without pitch gate creates 56 mm oscillation), and the alpha_r sensitivity dominance (20x the next parameter) collectively validate that this is not merely a tuned PID variant but a mechanism with identifiable structure. **Evidence: Table IV (push ablation), Table III (standing), gate sensitivity analysis (Section V-E).**

2. **S2: The factorial ablation design (additive L0-L3 + subtractive S0-S5) is methodologically exemplary.** Each component's marginal contribution is individually attributable. The two-direction validation (additive builds up, subtractive tears down) addresses the confound of compound intervention. The S5 pitch-gate ablation revealing parametric oscillation (56 mm) when boost lacks pitch gating is a genuinely insightful finding that demonstrates the ablation design can detect unexpected interactions, not just confirm predictions. **Evidence: Table IV (push ablation), Section V-B discussion.**

3. **S3: The four-variant LQR comparison (PID-servo, integral-AW, direct-torque, coupled 6-state 3D) is among the most thorough classical-control baseline evaluations I have seen in a wheeled-biped paper.** The direct-torque variant eliminates the PID-layer confound. The 46-configuration gain sweep confirms the failure is structural rather than parametric. The coupled 6-state 3D LQR (with LQR-optimal hip-roll gains ~7.7x stronger than the decoupled PD) performing identically to the 4-state variant provides non-trivial evidence for the servo-path bottleneck hypothesis. The Zamani et al. (2020) comparison is now deepened with specific quantitative context: their coupled LQR achieved theta_RMS ~1.5 deg, phi_RMS ~2.0 deg on a rigid direct-drive platform; the ~0.25 s servo lag absent in Zamani's platform prevents stabilization on this legged morphology. **Evidence: Table V (LQR baselines), Section V-D, footnote with 46-configuration breakdown.**

4. **S4: Exceptionally candid limitations section (10 enumerated items) and statistical power disclosure.** The paper explicitly acknowledges when N is insufficient (N=3 terrain: Clopper-Pearson 95% CI [0.29, 1.0]), when pairwise differences fall below the detection threshold (sub-2 N at N=5), when the Bonferroni correction renders comparisons non-significant (only L0 to L1 survives at alpha=0.007), and when claims are conditional (0.73 mm idle precision conditional on stiff ground contact). This level of self-critical disclosure is rare in robotics submissions and significantly strengthens the paper's credibility. **Evidence: Section VI (Discussion), Table IV footnotes, robustness table footnote.**

5. **S5: Complete parameter disclosure, version-controlled code, and explicit commit hash (f92640c).** All 50+ gains enumerated in Table III. Ablation configurations implemented as base-controller parameter modifications and documented. LQR baseline configs, controller implementations, and AW validation script are version-controlled. The gate parameter sensitivity analysis provides a tuning hierarchy (alpha_r dominates, pitch stability gates are insensitive safety interlocks) and a 5-step porting heuristic. **Evidence: Table III (parameters), Acknowledgment section, gate sensitivity analysis.**

6. **S6: The gate activation figure (Fig. 6) is pedagogically excellent.** The three-panel time series (raw velocity + EMA envelope, individual gate states, composite boost + effective stiffness) during a 90 N push-recovery cycle makes the controller's dynamic behavior transparent. This visualization standard should be emulated in classical control papers. **Evidence: Figure 6 (gate_activation.pdf).**

---

## Weaknesses

1. **W1: Alternative-paradigm baseline still absent despite 4/5 reviewer consensus across two review rounds.** The paper compares ACC against four LQR variants -- a thorough evaluation within the classical PD-servo paradigm -- but provides no comparison against a fundamentally different approach (RL, MPC, or DOB-based control). Limitation 10 now acknowledges this: "Baselines compared are classical LQR variants; comparison against learning-based (PPO) or optimization-based (MPC) methods is deferred." This narrows the claim scope acceptably, but the abstract's unqualified statement "Four classical LQR baselines...all fail (100% fall rate)" still implies these four variants exhaust the relevant comparison space.
    - **Severity:** Major
    - **Evidence Anchor:** `absence: Section V -- expected RL/MPC/DOB baseline; text: Limitation 10; text: Abstract line 50`
    - **Why it matters:** The central empirical claim ("all classical alternatives fail") is supported only within one paradigm. For RA-L archival publication, the scope of the empirical claim should match the scope of the empirical evidence.
    - **Suggestion:** Add a one-clause qualifier to the abstract's LQR failure statement: "...all fail (100% fall rate) among classical LQR variants evaluated." This is a one-sentence text fix that aligns the abstract with Limitation 10.
    - **Confidence:** 5

2. **W2: The "dominant bottleneck" claim, while improved from Round 7's "THE bottleneck," still overstates relative to the Direct Torque evidence.** Removing the PID servo lag improves survival by only 35% (0.52 s to 0.70 s) with 100% fall rate persisting. This demonstrates the servo path is a *contributing* bottleneck, but the residual failure after its removal suggests another bottleneck -- likely fundamental authority limits of linear control on this nonlinear torque-constrained platform -- remains. The coupled 6-state LQR evidence (identical performance to 4-state, 100% fall) confirms the servo path prevents LQR-optimal hip-roll gains from stabilizing the platform, but does not establish that the servo path is the *dominant* constraint -- only that it is *a* binding constraint among several.
    - **Severity:** Minor (improved from Round 7's Critical)
    - **Evidence Anchor:** `Table V: LQR DT survival 0.70 +/- 0.04 s, 100% fall rate; text: Section V-D "the dominant bottleneck"`
    - **Why it matters:** Precision in causal attribution distinguishes hypothesis from finding. "Dominant bottleneck" implies a rank-ordering among multiple bottlenecks that the data do not fully support.
    - **Suggestion:** "The PID servo action path is a primary identifiable bottleneck; removing it via direct torque output improves survival 35% but does not alone prevent the fall, indicating additional constraints -- likely actuator authority limits for linear control on this nonlinear platform -- also contribute."
    - **Confidence:** 4

3. **W3: The RL motivation ("ACC as precondition for RL") remains stated in the introduction without any RL+ACC evidence.** The paper states: "ACC is thus a *precondition* for RL on this platform: a height-general structured prior that a residual policy can later augment" (Section I). This is a forward-looking claim about future work, not a finding supported by the paper's evidence. The paper evaluates only classical controllers. The project CLAUDE.md confirms residual RL is planned but not trained.
    - **Severity:** Minor
    - **Evidence Anchor:** `text: Section I "ACC is thus a precondition for RL on this platform"; absence: Section V -- no RL+ACC experiment`
    - **Why it matters:** The introduction's framing influences how reviewers assess the paper's scope. A claim about RL that is unsupported by the paper's own evidence invites scrutiny that distracts from the paper's actual contributions.
    - **Suggestion:** Rephrase as a future-work motivation: "ACC is designed to serve as a height-general structured prior that a residual policy can later augment -- a hypothesis pending RL+ACC co-training experiments." This preserves the motivation without claiming a finding.
    - **Confidence:** 5

4. **W4: The platform comparison table (Table I) still asymmetrically reports quantitative metrics for ACC and qualitative descriptions for competitors.** ACC entries include specific numbers (0.73 mm, 110 N, 100 cm drop, 20 cm curb) while competitor entries use qualitative descriptors ("LQR," "Cascaded LQR," "Convex MPC," "---"). The footnote states "--- = not addressed in cited work," which is honest, but the visual impression is that ACC is the only platform with measurable capabilities. This is a reporting asymmetry, not a performance asymmetry.
    - **Severity:** Minor
    - **Evidence Anchor:** `Table I (Platform Comparison)`
    - **Why it matters:** A reader scanning the table may infer that competing platforms lack these capabilities entirely, when in many cases (e.g., Ascento's omnidirectional WBC-based balancing, SKATER's terrain adaptation) the capability exists but was measured under different protocols or reported in different metrics.
    - **Suggestion:** Either: (a) add a column explicitly stating "Metric reported in source?" with yes/no/not-measured, or (b) add a sentence to the table caption: "Quantitative comparisons across platforms are limited by differing evaluation protocols; ACC metrics are reported from the present work, competitor capabilities are summarized from cited literature and may not be directly comparable."
    - **Confidence:** 3

5. **W5: The latency sensitivity bifurcation (<10 ms threshold) is the most practically consequential result in the paper, yet the hardware latency budget remains an estimate without experimental validation.** The robustness table shows F_max drops from ~95 N to ~45 N at any non-zero delay, and idle RMS increases 5.5x at 10 ms. The paper estimates end-to-end latency at ~9.5 ms (IMU ~1 ms + filter ~3 ms + control ~4.5 ms + bus ~1 ms), which is within 5% of the 10 ms threshold. This means the controller, as designed, operates at the very edge of its viable regime. The paper's "Path to Hardware" discussion (Section VI) addresses encoder quantization, backlash, stiction, tire compliance, and IMU noise, but does not experimentally validate the 9.5 ms estimate or test ACC with realistic velocity-estimation latency (IMU+odometry complementary filters typically add 5-20 ms).
    - **Severity:** Minor
    - **Evidence Anchor:** `Table VI (Robustness): F_max 96 N at 0 ms vs 47 N at 10 ms; text: Section VI "End-to-end latency ~9.5 ms"`
    - **Why it matters:** If the projected 9.5 ms latency cannot be achieved on hardware, ACC's precision advantage may not survive the transition. This is the single largest risk to the paper's hardware-readiness narrative.
    - **Suggestion:** Add one sentence to the hardware degradation discussion: "The projected 9.5 ms end-to-end latency places ACC within 5% of its measured 10 ms degradation threshold; validating this latency budget on hardware, or hardening the controller against 10-20 ms latency through increased envelope release time, is the highest-priority hardware risk to retire."
    - **Confidence:** 4

---

## Detailed Comments

### Title and Abstract
The title "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery" is descriptive, distinctive, and accurately scoped. The abstract is dense with quantitative claims -- appropriate for RA-L. Two minor issues: (a) the LQR failure statement (line 50) should carry the "among classical LQR variants evaluated" qualifier present in Limitation 10, and (b) "degrades to 2.30 mm under 10 ms delay" (line 49) mentions precision degradation but omits the push performance degradation (F_max drops from ~95 N to ~45 N), which is a material omission given that "omnidirectional push recovery" is a headline claim.

### Introduction
The problem framing (stiffness-precision-windup trilemma) is clear and well-motivated. The PPO failure narrative (100M steps, failed height generalization) is used to motivate ACC as a structured prior for future RL. This is acceptable as motivational framing, but the claim "ACC is thus a precondition for RL" (line 66) should be rephrased as a forward-looking hypothesis (see Weakness W3). The two enumerated contributions accurately reflect what the paper demonstrates.

### Related Work
Coverage is now comprehensive. The addition of process control citations (Shinskey 1996, Astrom and Hagglund 2005/2006) and the deepened Zamani et al. (2020) comparison (now with quantitative metrics and servo-lag hypothesis) address the Round 7 P2-R12 and P1-R4 fixes. The distinction from DOB/ADRC (Section II-B) is clearly articulated: frequency-domain estimation-speed-vs-noise trade-off versus time-domain attack-vs-release trade-off. The connection to Hyon et al. (2007) on gated humanoid balancing provides appropriate lineage. One observation: the "C¹" smoothstep claim is appropriately qualified with the footnote acknowledging C⁰ composite behavior at EMA switching points -- this addresses the Round 4/7 continuity concern.

### Robot Platform
The morphology description is complete. The MuJoCo simulation parameters (solref, armature, solver settings, integration method) are now documented in full, addressing the Round 4/7 opacity concerns. The contact stiffness sensitivity is explicitly quantified: softening solref[0] from 0.002 to 0.02 increases idle RMS ~60% to ~1.2 mm, making the 0.73 mm claim conditional on stiff ground contact (P2-R10 fix verified). The tire compliance estimate (0.1-0.5 mm on real surfaces) and hardware degradation projection (1-3 mm) provide honest bounds on the precision claim.

### ACC Formulation
The two-channel torque assembly (Equation 1) is clearly specified. The "conditionally-gated" terminology is now connected to process-control lineage (selector-based, override, split-range control) with the explicit claim that ACC's smoothstep gates "extend this principle from discrete selection to continuous blending for robotics" -- this addresses the Round 7 SC2 consensus. The individual components are well-derived:

- **Balance core (Section IV-B):** The P-only limitation (1.3 Nm equilibrium bias sustaining a 55 mm limit cycle) motivates the anchor. The anti-windup discussion (Appendix A) now acknowledges the full AW taxonomy (conditional integration, tracking back-calculation, observer-based AW) and correctly narrows the claim: the simplest conditional-integration AW is insufficient because it triggers on saturation, not state-dependent windup. The scheduled k_p is honestly presented: "Fixed k_p=50 is recommended: the scheduled variant...did not improve push survival over fixed k_p=50 (S1 vs. S0, Table IV)" -- this directly addresses the Round 7 DA-C1 critical finding.
- **Anchor mechanism (Section IV-C):** The asymmetric EMA, proximity gate, envelope gate, pitch stability gate, and damping boost are mathematically specified. The symmetric-EMA bistability argument is clearly explained. The EMA initialization discussion (EMA_0 = 0 biases g_env = 1 for ~7.5 s; hardware should initialize EMA_0 = 1.0) is a commendable practical detail.
- **Flight recovery (Section IV-E):** The flight gate uses discrete load-based hysteresis (2-step engage, 5-step release) to prevent re-engaging balance torque during landing bounce. The distinction between free-fall drop (stress test) and ledge drive-off (use case) shows careful experimental design.
- **Terrain adaptation (Section IV-F):** Per-leg contact-point height estimation with closed-loop leveling integrator. The unloaded-wheel freeze-and-seek logic is clearly specified.

### Experimental Validation
The evaluation is organized around four scenario classes, each mapped to the component it tests:

- **Anchored standing (Section V-A):** ACC achieves 0.73 +/- 0.07 mm CoM RMS (N=10, 95% CI [0.68, 0.78] mm). The P-only limit cycle (55 mm p-p, 39 mm RMS) provides a clear baseline. The symmetric-EMA, global-I, and no-boost ablations isolate the anchor's contribution. The noise floor measurement (0.002 mm, SNR 289x) properly contextualizes the result.
- **Omnidirectional push recovery (Section V-B):** F_min = 80 N, F_med = 110 N across 8 directions via binary search (5 N tolerance). The factorial ablation (Table IV) is the paper's strongest evidence. The Bonferroni-adjusted significance discussion is now adjacent to the p-values (P2-R11 fix verified). The S5 collapse (F_min = 9.0 N without pitch gate) is striking and important.
- **Flight and terrain (Section V-C):** Correctly labeled as qualitative feasibility demonstrations with explicit N values (N=3-10). The Clopper-Pearson CI for N=3 binary outcomes is reported.
- **LQR baselines (Section V-D):** Four variants evaluated. The 46-configuration sweep is now arithmetically transparent (35 roll PD + 5 wheel velocity + 6 cross-coupling = 46; P2-R13 fix verified). The coupled 6-state LQR performing identically to the 4-state variant is evidence for the servo-path hypothesis. The identical LQR/LQR+AW values are explained (AW never triggers because wheels stay within velocity range during the 0.52 s pre-fall period) -- this is a non-obvious but correct observation.
- **Robustness analysis (Section V-E):** The noise x delay factorial sweep is comprehensive. The non-monotonic noise response is now explained (P2-R9 fix verified): high noise holds EMA above threshold, disengaging the anchor and avoiding partial-engagement resonance; low noise at 10 ms creates partial-engagement resonance; at 30 ms the EMA stays above threshold. The gate sensitivity analysis establishing alpha_r dominance (220.4, 20x next parameter) is a genuine contribution to systematizing the tuning process.

### Discussion and Conclusion
The limitations section (10 items) is exceptional. The statistical power discussion (minimum detectable effect ~4.5 N at N=5, Bonferroni-adjusted alpha = 0.007, Clopper-Pearson CI for N=3) is thorough. The hardware degradation projection (1-3 mm from 0.73 mm) and staged commissioning plan demonstrate deployment thinking. The conclusion correctly restates the main results and future work directions.

---

## Verification of Round 7 P2 Fixes

| Fix ID | Description | Status | Evidence Location |
|--------|-------------|--------|-------------------|
| **P2-R9** | Non-monotonic noise explained (partial-engagement resonance) | **FIXED** | Table VI footnote, lines 640-641: "The apparent improvement from medium to high noise...occurs because high noise holds the EMA permanently above threshold, disengaging the anchor...The non-monotonic low-noise idle RMS...reflects partial-engagement resonance: 10 ms phase-shifts the anchor into intermittent engage/release; at 30 ms the EMA stays above threshold, avoiding resonance." |
| **P2-R10** | Contact stiffness sensitivity discussed | **FIXED** | Section III, lines 108-109: "Softening solref[0] to 0.02 (tire-on-asphalt) increases idle RMS ~60% to ~1.2 mm, so the 0.73 mm claim is conditional on stiff ground contact; tire compliance adds 0.1-0.5 mm on real surfaces (see Section VI)." |
| **P2-R11** | Bonferroni context near p-values | **FIXED** | Table IV footnote, line 485: "Marginal (paired t-test, N=5; uncorrected, Bonferroni-adjusted alpha=0.007, see Section VI)." Also discussed in Section VI, line 666: "With 7 pairwise comparisons, a Bonferroni-adjusted threshold (alpha=0.05/7 approx 0.007) renders only L0 to L1 (p<0.001) significant." |
| **P2-R12** | Zamani et al. comparison deepened | **FIXED** | Section II-A, lines 87-88: "Zamani et al. demonstrated coupled pitch-roll LQR (theta_RMS ~1.5 deg, phi_RMS ~2.0 deg) on a rigid two-wheeled platform without articulated legs, where torque commands act directly on wheels. Our 6-state LQR replicates their coupled structure on a legged morphology and finds the PID servo path prevents stabilization despite ~7.7x stronger hip-roll gains: the ~0.25 s servo lag---absent in Zamani's direct-drive platform---introduces phase margin loss that the coupled LQR cannot overcome." |
| **P2-R13** | 46-configuration sweep arithmetic corrected | **FIXED** | Section V-D footnote, line 595: "Swept 35 roll PD combinations (kp_roll in {0.4,0.8,1.2,1.6,2.0,3.0,5.0}, kd_roll in {0.08,0.2,0.5,1.0,2.0}), 5 wheel velocity limits ({20,30,40,60,100} rad/s), and 6 roll-wheel cross-coupling gains ({0.0,0.5,1.0,2.0,5.0,10.0})." Arithmetic: 7x5 = 35 + 5 + 6 = 46 (separate sweeps, not full factorial). |

All five P2 fixes are correctly implemented and verifiable against the manuscript.

---

## Verification of Round 7 P0 Critical Fixes

| Fix ID | Description | Status | Assessment |
|--------|-------------|--------|------------|
| **P0-R1** | Reconcile scheduled k_p with ablation data (DA-C1) | **FIXED** | The paper now explicitly states "Fixed k_p=50 is recommended: the scheduled variant...did not improve push survival over fixed k_p=50 (S1 vs. S0, Table IV)" (Section IV-B). The schedule is retained "for completeness" but no longer claimed as essential. |
| **P0-R2** | Recalibrate LQR bottleneck claim (DA-C2) | **PARTIALLY FIXED** | Changed from "THE bottleneck" to "the dominant bottleneck." The Direct Torque evidence (35% improvement, 100% fall persists) is still not fully reconciled with "dominant" -- see Weakness W2 above. |
| **P0-R3** | Add alternative-paradigm baseline (SC1) | **DEFERRED** | No RL/MPC/DOB baseline added. The paper now scopes claims to classical LQR variants (Limitation 10). This is acceptable for a letter-format submission if the abstract is similarly scoped -- see Weakness W1 above. |

---

## Questions for Authors

1. **Latency budget validation:** You project 9.5 ms end-to-end latency, which places ACC within 5% of its measured 10 ms degradation threshold. Have you characterized the latency of your planned IMU+odometry complementary filter for v_sag estimation? Standard complementary filters on microcontrollers often add 5-20 ms of group delay. If the filter contributes more than the budgeted 3 ms, ACC would operate in its degraded push regime (F_max ~45 N rather than ~95 N). Is there a path to hardening the controller against 10-20 ms latency, e.g., by increasing tau_release?

2. **Direct-torque coupled LQR:** You demonstrate that the coupled 6-state LQR through a PID servo path fails identically to the 4-state variant, and that direct-torque output improves the 4-state LQR's survival by 35%. What would you predict for a coupled 6-state LQR with direct torque output -- the configuration that combines both fixes? This is listed as deferred work (Limitation 6). Do you have a reasoned hypothesis based on the evidence collected?

3. **Non-monotonic noise as systematic fragility:** Your explanation of the non-monotonic noise response (partial-engagement resonance at medium noise being worse than full disengagement at high noise) is convincing for this specific controller tuning. Do you consider this non-monotonicity a fundamental property of envelope-gated integral controllers (i.e., any controller with a velocity-threshold gate will exhibit a worst-case intermediate noise regime), or is it tunable by adjusting the envelope band (g_env: 0.25-0.50 m/s)?

4. **Contact stiffness and the 0.73 mm claim:** You report that softening solref[0] from 0.002 to 0.02 increases idle RMS by ~60% (to ~1.2 mm). Have you characterized the precision sensitivity to other contact parameters -- specifically solref[1] (damping) and solimp? For the 0.73 mm claim to be meaningful, the reader needs to know whether it is robust to plausible variations in contact modeling or specific to one choice of solver parameters.

---

## Minor Issues

- **Abstract, line 50:** "Four classical LQR baselines...all fail (100% fall rate)" -- add "among the classical LQR variants evaluated" to align with Limitation 10.
- **Abstract, line 49:** "degrades to 2.30 mm under 10 ms delay" mentions precision degradation but omits push degradation (F_max drops from ~95 N to ~45 N). Given that push recovery is a headline claim, this omission should be corrected.
- **Section I, line 66:** "ACC is thus a precondition for RL on this platform" -- rephrase as "ACC is designed to serve as a structured prior for future residual RL" (see Weakness W3).
- **Section V-D, line 593:** "the dominant bottleneck" -- consider the more precise formulation suggested in Weakness W2.
- **Table I (Platform Comparison):** The quantitative-vs-qualitative asymmetry should be addressed (see Weakness W4).
- **Reference verification:** References [29] (ferreira2021ollie), [30] (dicarlo2018mpc), [31] (lee2020rloc), and [32] (kim2017velocity) appear in the bibliography but their in-text citations could not be located in a scan of the manuscript. Verify all references are cited in the text.
- **Line 108:** "solref" formatting -- consider using `\texttt{solref}` consistently for MuJoCo parameter names (already done for the most part, verify consistency).
- **Compact spacing overrides (lines 16-27):** The aggressive `\setlength` commands compress the layout beyond IEEEtran conference defaults. Verify compliance with the target journal's style guide.

---

## Dimension Scores

| Dimension | Score (0-100) | Notes |
|-----------|--------------|-------|
| Originality (20%) | 78 | Asymmetric envelope follower + proximity-gated anchor is genuinely novel in wheeled-biped control. The S5 parametric oscillation finding adds depth. The "conditionally-gated" framing still slightly overstates relative to established process-control lineage, but the Round 7 connections to override/selector-based control partially address this. |
| Methodological Rigor (25%) | 82 | Factorial ablation (additive + subtractive) is exemplary. Four-variant LQR comparison with controlled ablations (servo path, integral channel, state dimensionality) is thorough. Binary search for threshold measurement, N=10 between-trial statistics with 95% CI, noise floor measurement, and gate sensitivity analysis with tuning hierarchy collectively exceed field norms. The one remaining gap is the absence of an alternative-paradigm baseline. |
| Evidence Sufficiency (25%) | 72 | Core claims (0.73 mm idle precision, 80 N omnidirectional push survival, anchor component contributions) are well-supported. Statistical power limitations are honestly disclosed. The LQR 100% failure claim is well-evidenced within the classical paradigm. Flight/terrain correctly labeled as qualitative (N=3-10). The main gap: no evidence that ACC outperforms non-LQR approaches. Latency sensitivity is measured but the hardware latency budget is estimated, not validated. |
| Argument Coherence (15%) | 85 | Clear logical flow: problem (trilemma) -> mechanism (proximity-gated anchor) -> architecture (two-channel assembly) -> validation (scenario-to-component mapping) -> limitations. The design narrative ("each gate was added after a measured failure mode was traced to a root cause") is pedagogically effective. Honest calibration of what the anchor achieves vs. what it does not builds reader trust. |
| Writing Quality (15%) | 80 | Technical prose is clear and precise. The mathematical specification (Equations 1-9) is complete. Tables are information-dense but readable. The gate activation figure is exemplary. Remaining issues: some defensive/repetitive prose in the ablation interpretation, a few unframed forward-looking claims (RL precondition), and one material omission in the abstract (push degradation under delay). |
| **Weighted Average** | **78.7** | |

---

## Summary

This manuscript has traversed eight review rounds and emerged as a substantially stronger paper. The core contribution -- a proximity-gated anchor with asymmetric envelope follower enabling sub-millimeter precision without sacrificing push survival -- is genuine, well-validated by factorial ablation, and honestly contextualized within its limitations. All five Round 7 P2 fixes are correctly implemented. Both Round 7 DA CRITICAL findings (scheduled k_p contradiction, bottleneck overstatement) are addressed, with the bottleneck claim partially resolved (improved but still slightly overstated). The one remaining structural gap is the absence of an alternative-paradigm baseline, which is now acceptably scoped by Limitation 10 but not yet reflected in the abstract. The remaining issues (W1-W5) are all text-level framing adjustments achievable within one revision cycle. **Recommendation: Minor Revision.**
