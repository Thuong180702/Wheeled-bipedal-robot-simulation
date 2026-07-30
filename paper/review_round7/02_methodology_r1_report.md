Now I have the complete paper. Let me produce the Methodology Review Report.

---

## Methodology Review Report (Peer Reviewer 1)

### Reviewer Identity
Control systems methodology expert with expertise in simulation-based robotics validation, experimental design for controller comparison, and statistical rigor in robotics experiments. Focus: whether the methods can answer the questions posed, whether the data collection is appropriate, whether the analysis is correct, and whether conclusions are supported by data.

### Overall Recommendation
**Major Revision**

### Confidence Score
4 -- Confident in the assessment of experimental design, statistical power, and reporting gaps; the simulation parameterization and LQR gain-sweep details are partially opaque and would benefit from code-level verification.

### Summary Assessment
This paper presents ACC, a conditionally-gated controller for wheeled bipedal balance, and evaluates it through a well-structured factorial ablation design, binary search for push thresholds, and comparison against four LQR baselines. The experimental architecture (additive + subtractive ablation, noise/delay factorial sweep, compound-disturbance scenarios) is methodologically thoughtful and among the stronger aspects of the work. The authors are commendably transparent about limitations: they explicitly acknowledge N=3 and N=4 as below the threshold for statistical inference, discuss the Bonferroni correction honestly, flag the implicit algebraic loop, and report a minimum detectable effect size. These disclosures raise the credibility of the work substantially.

However, several methodological weaknesses must be addressed before acceptance. The statistical tests producing the reported p-values are never named, and no effect sizes (Cohen's d, eta-squared) accompany them. The 46-configuration LQR gain sweep is mentioned as a footnote but never tabulated, making it impossible to assess whether the sweep was systematic or cherry-picked. The robustness factorial (Table VII in paper / Table 8 here) contains 12 cells with N=5 but reports no interaction statistics for the noise-by-delay design. The binary search methodology, while appropriately described procedurally, lacks verification that the threshold is monotonic with force magnitude or that the 8-iteration, 5N-tolerance protocol converges to the same threshold within measurement noise. The "implicit algebraic loop" limitation (item 7) is acknowledged but the one-sample delay resolution is never validated by varying the control rate to confirm that the 100Hz assumption is adequate. Finally, the paper makes quantitative claims about F_min and F_med from N=5 replications but the binary search outcome per replication is itself a noisy estimate, and it is unclear whether the reported standard deviations capture threshold estimation uncertainty or between-trial variability or both. These issues are individually addressable but collectively require a revision round.

### Strengths

1. **S1: Factorial Ablation Design**: The additive (bottom-up L0-L3) and subtractive (top-down S0-S5) ablation structure is a controlled, single-variable-at-a-time design that cleanly isolates each component's contribution. This is far stronger than the common "ablation" practice of removing one component from the full system. **Evidence Anchor**: `table: Table II (push_ablation), rows L0-L3 and S0-S5`

2. **S2: Transparent Limitations Section**: The Discussion explicitly quantifies statistical power (MDE ~4.5N at N=5), reports Clopper-Pearson CIs for N=3 binary outcomes, discusses the Bonferroni correction honestly, and flags that sub-2N differences are non-significant. This degree of statistical candor is unusually strong for a robotics conference paper. **Evidence Anchor**: `text: "With N=5, the minimum detectable effect is ~1.5x the within-cell std (~4.5 N). Differences <5 N cannot be distinguished from noise."` (Section VI, Discussion)

3. **S3: Multiple Baseline Variants with Controlled Ablation**: The four LQR variants differ by exactly one mechanism each (integral channel, action path, state dimension), making this a controlled comparison rather than an apples-to-oranges benchmark. The AW validation script is referenced. **Evidence Anchor**: `text: "The LQR and LQR+AW differ only by the integral channel; the direct-torque variant differs only in the action path---each a controlled ablation"` (Section V-D)

4. **S4: Pre-Registration of Thresholds**: The authors state that all gate thresholds were frozen prior to the formal ablation sweep, which reduces the risk of post-hoc threshold tuning to achieve desired results. **Evidence Anchor**: `text: "All thresholds were frozen prior to the formal ablation sweep (Table II)"` (Section VI, Discussion)

5. **S5: Reproducibility Infrastructure**: Code is version-controlled with a specific commit hash, seeds are documented, and raw per-trial JSON data are preserved in a named output directory. Controller implementations and configs are in the repository. **Evidence Anchor**: `text: "Code and parameters: https://github.com/thuong180702/Wheeled-bipedal-robot-simulation (commit f92640c)"` (Acknowledgment)

### Weaknesses

1. **W1: Statistical Test Identity Not Disclosed**: The paper reports three p-values (p<0.001, n.s., p<0.05) for pairwise comparisons in the additive ablation path but never names the statistical test used. Given N=5 per group, the choice matters: a paired t-test assumes normality and independence of binary search outcomes across replications, while a Wilcoxon signed-rank test makes fewer assumptions but has lower power at N=5. The reader cannot assess whether test assumptions are satisfied because neither the test nor assumption checks are reported.
   - **Severity**: Major | **Evidence Anchor**: `text: "L0->L1 +5.5 N (p<0.001); L1->L2 +1.1 N (n.s.); L2->L3 +4.1 N (p<0.05)"` (Table II footnote) | **Confidence**: 5 -- competence basis: statistical reporting standards for experimental robotics

2. **W2: No Effect Sizes Reported**: All statistical comparisons report only raw differences and p-values. No standardized effect sizes (Cohen's d, Hedges' g for small samples, or eta-squared) are provided. The 5.5N improvement from L0 to L1 is statistically significant but its practical magnitude relative to within-group variability is unknown without an effect size. APA 7.0 and most engineering journals require effect sizes alongside significance tests.
   - **Severity**: Major | **Evidence Anchor**: `absence: Section V (Experimental Validation) — expected standardized effect size (e.g., Cohen's d) for each pairwise comparison; checked all tables, footnotes, and prose in Sections V-A through V-D` | **Confidence**: 5 -- competence basis: APA 7.0 statistical reporting standards

3. **W3: 46-Configuration LQR Gain Sweep Not Tabulated or Characterized**: The paper states that a 46-configuration gain sweep "confirmed this failure is structural, not parametric" and describes it in a footnote (14 roll gain values x 3 damping values x 5 pitch stiffness values = 210, not 46). The arithmetic is inconsistent (14 x 3 x 5 = 210, not 46), and the results are never shown. Without a table or figure showing survival time vs. gain parameters, the reader cannot verify that the sweep was exhaustive, that no configuration achieved non-trivial survival, or that the claimed "46 configurations" were selected systematically rather than cherry-picked post-hoc. A supplementary table listing each configuration with its survival time is needed.
   - **Severity**: Major | **Evidence Anchor**: `text: "A 46-configuration gain sweep confirmed this failure is structural, not parametric"` (Section V-D) and `text: "Swept roll PD gains 0.1-2.5 (14 values), velocity damping {1.0,1.5,2.0}, pitch stiffness {35,40,45,50,55} (13.8x nominal roll gain range)"` (footnote) | **Confidence**: 4 -- competence basis: experimental design methodology; the footnote arithmetic (14 x 3 x 5 = 210) contradicts "46 configurations"

4. **W4: Binary Search Measurement Uncertainty Unaccounted For**: The push thresholds are measured via binary search with 5N tolerance and 8 iterations per direction. Each of the N=5 replications produces one F_min value (the minimum across 8 directions). The reported standard deviations (e.g., 80.4 +/- 3.0 N) conflate two sources of uncertainty: (a) binary search convergence within +/-5N for each direction, and (b) genuine between-trial variability. The paper does not report whether binary search produces identical thresholds when repeated on the same seed, making it impossible to partition the variance. This matters because sub-2N differences between configurations are declared non-significant based on N=5, but if the binary search itself has 5N tolerance, those differences may be below the measurement resolution.
   - **Severity**: Major | **Evidence Anchor**: `text: "All thresholds measured via binary search over [10 N, 160 N] (8 iterations per direction, 5 N tolerance)"` (Section V-B) | **Confidence**: 4 -- competence basis: measurement uncertainty analysis in experimental robotics

5. **W5: Noise-Delay Factorial Lacks Interaction Analysis**: Table VII reports a 4 (noise) x 3 (delay) factorial design with N=5 per cell, yielding 12 cells. The paper describes main effects qualitatively ("delay bifurcation") but reports no formal two-way ANOVA, no interaction F-test, and no simple main effects. The medium-to-high noise reversal at 0ms delay (34mm -> 3mm RMS) is explained post-hoc as anchor disengagement but was not predicted and is not tested. A factorial design without interaction analysis underuses the data collected.
   - **Severity**: Major | **Evidence Anchor**: `table: Table VII (robustness), 12-cell factorial with no interaction statistics reported` | **Confidence**: 5 -- competence basis: factorial experimental design and ANOVA methodology

6. **W6: Implicit Algebraic Loop Resolution Not Validated**: Limitation (7) states that the bidirectional coupling of the k_p schedule and anchor integral "forms an implicit algebraic loop, resolved at 100 Hz by one-sample computational delay." This resolution mechanism is never validated. Varying the control rate (e.g., 50Hz, 200Hz) would test whether the one-sample delay assumption holds; without this, the stability margin of the algebraic loop resolution is unknown.
   - **Severity**: Minor | **Evidence Anchor**: `text: "Bidirectional coupling of k_p schedule and anchor integral forms an implicit algebraic loop, resolved at 100 Hz by one-sample computational delay"` (Limitation 7, Section VI) | **Confidence**: 4 -- competence basis: discrete-time control systems analysis

7. **W7: No Cross-Validation of MuJoCo Contact Parameters**: The paper uses stiff body contacts (solref={0.002,1}) that "idealize a rigid ground." This is acknowledged, but the sensitivity of the claimed 0.73mm idle precision to contact stiffness, solver iteration count (100 Newton iterations), and integration timestep is not probed. Given that sub-millimeter precision is the headline result, a contact-parameter sensitivity sweep (e.g., solref stiffness varied by +/-50%) would establish whether this result is robust to the simulation's contact model or is an artifact of idealized rigid contact.
   - **Severity**: Minor | **Evidence Anchor**: `text: "The stiff body contact (solref={0.002,1}) idealizes a rigid ground; tire/ground compliance adds 0.1-0.5 mm on real surfaces"` (Section III-A) | **Confidence**: 4 -- competence basis: MuJoCo simulation fidelity analysis

8. **W8: Torque Rate-Limit Invariance Claim Unquantified**: The paper states ACC performance is "completely invariant" across 200-800 Nm/s (4x range) but provides no metrics, no table, and no N value for this claim. "Completely invariant" is an absolute claim that requires quantified support -- at minimum, idle RMS and F_min at each rate limit tested.
   - **Severity**: Minor | **Evidence Anchor**: `text: "ACC performance is completely invariant across 200-800 Nm/s (4x range)---its ceiling is architectural, not actuator-limited"` (Section V-E) | **Confidence**: 5 -- competence basis: quantitative claims require quantitative evidence

### Detailed Comments

#### Research Questions and Hypotheses
The central research question -- can a conditionally-gated controller achieve both sub-millimeter idle precision and omnidirectional push recovery without sacrificing either -- is clear, falsifiable, and well-motivated by the P-only limit cycle problem. The two stated contributions (proximity-gated anchor mechanism, unified two-channel architecture) are appropriately scoped. The implicit hypothesis that the PID servo action path is the bottleneck for classical LQR controllers is tested directly via the direct-torque and coupled 6-state variants. These are well-posed questions.

#### Research Design
The factorial ablation design (additive + subtractive paths) is a strength. The controlled-variable approach for LQR baselines (each variant changes exactly one mechanism) is also strong. The noise/delay factorial sweep and compound-disturbance scenarios add ecological validity. However, the design would be stronger if (a) the order of ablation runs was randomized rather than sequential, (b) the binary search starting point was randomized to avoid anchoring bias, and (c) the experimenter was blinded to configuration identity during data collection (or at minimum, the data collection was automated and batch-processed).

#### Sampling Strategy
The N=10 for idle precision is adequate for estimating a mean with reasonable CI width. The N=5 for push ablation is the minimum for any statistical inference and the authors acknowledge this transparently. The N=3 for terrain and N=4 for ledge are explicitly labeled as qualitative feasibility demonstrations, which is appropriate. However, the LQR baselines use inconsistent N values (N=60 for LQR/LQR+AW, N=20 for LQR-DT and 6-state LQR, N=10 for ACC), and the justification for these different sample sizes is not provided. If the larger N for LQR baselines is because they were cheaper to run (they fall in 0.52s), that should be stated.

#### Data Collection
The binary search protocol (8 iterations, 5N tolerance, 8-direction polar sweep) is procedurally well-specified. The survival criterion (pitch < 46 degrees for 17s post-push) is clear. The settle period (3s before measurement) is specified. However, the paper does not state whether the binary search for each direction is independent or whether F_min is computed from independent per-direction searches, which matters for the interpretation of the aggregate F_min statistic.

#### Analysis Methods
The statistical analysis is incomplete. Specific issues:
- No named statistical test for the three reported p-values.
- No effect sizes.
- No interaction analysis for the 4x3 factorial robustness design.
- No correction for multiple comparisons in the robustness table (12 cells, multiple implicit comparisons).
- The finite-difference gradient analysis for gate parameter sensitivity reports raw sensitivity values without units or normalization, making "220.4" uninterpretable without knowing the perturbation size and output metric.

The authors' explicit discussion of the Bonferroni correction and MDE is commendable and partially mitigates these concerns.

#### Results Presentation
Tables are clear and well-structured. Figures (polar envelope, ringdown time series, gate activation, curb straddle) are informative. The supplementary footnote system for table annotations is helpful. The marginal notes in Table II are unusually detailed for a conference paper and add substantial value.

One concern: Table I (LQR baselines) reports LQR/LQR+AW with identical mean and standard deviation values to three significant figures (0.52 +/- 0.00, 0.04 +/- 0.01, 24.2 +/- 2.1, 371 +/- 45, 92.1 +/- 3.2). The identical values for all five metrics across two different controllers (N=60 each) are statistically improbable. If AW never triggers, the raw numbers should still differ due to sampling variability across 60 episodes with different random seeds. The paper states "The identical LQR/LQR+AW values are expected: AW never triggers" but this explains identical means, not identical standard deviations to three significant figures. This warrants clarification.

#### Reproducibility
Above average for a robotics conference paper. The GitHub repository with commit hash, documented seeds, named output directories, and referenced validation scripts provide a strong foundation. The 50+ parameter table with exact numerical values enables reimplementation. Two gaps: (a) the 46-configuration LQR sweep results are not in the repository as a table (the script is referenced but output is not), (b) the binary search automation script is not explicitly named or referenced, so the exact protocol (starting value, direction order, convergence criterion) cannot be verified from the paper alone.

#### Methodological Fallacies Detected
- **P-hacking risk (moderate):** Seven pairwise comparisons reported without full correction. The authors acknowledge this and note that primary claims do not depend on borderline p-values. Mitigated by transparency.
- **Confirmation bias risk (low):** The post-hoc explanation for the medium-to-high noise reversal (anchor disengagement) is plausible but was not predicted a priori. Labeling it as a post-hoc observation would be more appropriate than presenting it as an expected mechanism.
- **Survivorship bias risk (low):** The paper reports only configurations that survived the ablation sweep. Failed intermediate configurations during tuning are not described. Mitigated by the explicit statement that thresholds were frozen before the formal sweep.

### Questions for Authors
1. What specific statistical test produced the three p-values (p<0.001, n.s., p<0.05) in Table II? Were assumption checks (normality, equal variance) performed?
2. The footnote states the LQR gain sweep used 14 roll gain values x 3 damping values x 5 pitch stiffness values, which multiplies to 210, not 46. What exactly constitutes the "46 configurations"?
3. Were the binary search starting points randomized per direction and per replication? If not, could anchoring to the initial value of 85N (midpoint of 10-160N) bias the threshold estimates?
4. The LQR and LQR+AW rows in Table I report identical means and identical standard deviations across all five metrics. Given N=60 independent episodes per controller with different random seeds, what explains the identical standard deviations?
5. Has the binary search convergence been validated by repeating the search on the same seed and confirming identical thresholds?
6. What is the perturbation size used for the finite-difference gate sensitivity analysis, and what output metric is the sensitivity computed against?

### Minor Issues
- The phrase "46-configuration gain sweep" should be reconciled with the footnote arithmetic (14 x 3 x 5 = 210).
- The "complete invariance" claim for torque rate limit requires at minimum a supporting table or N value.
- The finite-difference sensitivity value "220.4" for alpha_r needs units (e.g., delta-F_min / delta-alpha_r at perturbation epsilon = X).
- Table I: the ACC torque RMS of 3.28 +/- 0.00 Nm has zero reported standard deviation but N=10 trials -- this should show some variability unless torque RMS is deterministic.
- In the abstract, "0.73 +/- 0.07 mm CoM RMS (N=10, 95%CI [0.68,0.78] mm)" and in the table footnote "95%CI [0.50,0.62] mm" appear to refer to different quantities (between-trial vs. within-run) but use the same notation, which could confuse readers.
- The noise level definitions (low = 0.01 rad/s angular velocity, etc.) should specify whether these are added as Gaussian white noise at each timestep or as biases, and whether they are independent across sensors.