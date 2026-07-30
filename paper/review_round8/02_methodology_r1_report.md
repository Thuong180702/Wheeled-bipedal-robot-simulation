# Peer Review Report — R1 Methodology (Round 8)

## Reviewer Identity
Researcher in experimental robotics evaluation methodology, specializing in systematic ablation design and statistical validation for classical and learned controllers. Background in control theory with a focus on reproducible experimental protocols and statistical power analysis for legged robot evaluation.

## Overall Assessment
### Recommendation: Minor Revision
### Confidence Score: 4
### Summary Assessment

This paper presents a well-structured experimental validation with an honest statistical discussion that is rare in robotics submissions. The factorial ablation (L0--L3, S0--S5) correctly isolates component contributions through both additive and subtractive paths, with a notable exception (see Weakness W1). Methodological strengths include: pre-registration of thresholds before the formal sweep, version-controlled scripts and configs tied to a specific commit hash (f92640c), raw data preservation in a documented output directory, explicit minimum-detectable-effect analysis at N=5, and correct application of all four Round 7 P2 fixes. The statistical discussion in Section VI is unusually self-aware for a robotics paper---acknowledging the limitations of N=5, the Bonferroni correction for 7 comparisons, and the wide Clopper-Pearson bounds for N=3 binary outcomes.

However, three issues must be resolved before acceptance. First, a genuine CI inconsistency exists between the abstract and Table IV for the anchor idle precision---the two 95% confidence intervals do not overlap and cannot be derived from the same data. Second, the paper reports "paired t-test" results, but the measurement protocol (independent bisection runs with different random seeds per configuration) yields independent samples; the test type and degrees of freedom must be corrected or justified. Third, several statistical reporting details need tightening: Hedges' g correction for N=5 effect sizes, the identical zero-variance survival times in the LQR baselines, and the unexplained discrepancy between L2 (global-I works) and S4 (global-I windups).

The paper is already above the bar for experimental methodology in robotics venues. Addressing these three issues---two of which are straightforward corrections---will make it exemplary.

## Strengths

**S1. Pre-registered threshold freezing.** All gate thresholds were frozen before the formal ablation sweep (Sec. VI, l. 662). This prevents p-hacking through post-hoc threshold tuning and is a best practice rarely observed in robotics evaluation. *Evidence:* "All thresholds were frozen prior to the formal ablation sweep (Table IV)."

**S2. Honest statistical power analysis.** The Discussion explicitly states the minimum detectable effect (~4.5 N at N=5), notes that sub-2 N differences are indistinguishable from noise, identifies which comparisons survive Bonferroni correction, and recommends N >= 10 for binomial outcomes before journal submission. *Evidence:* Sec. VI, l. 664--667.

**S3. Complete reproducibility infrastructure.** Four scripts (`push_ci.py`, `robustness_sweep.py`, `validate_lqr_aw.py`, `sweep_lqr_roll_gains.py`) are explicitly named and exist in the repository. The commit hash (f92640c) is cited. LQR baseline configs are version-controlled. Raw JSON data is preserved in `outputs/paper_statistics/`. Seed determinism is documented: `seed = 20260727 + trial_id` for idle, `seed = 20260728 * 100 + rep` for push ablation. *Evidence:* l. 412--413, l. 709.

**S4. Correct Round 7 P2 fix application.** All four fixes are properly applied: (i) P2-R9: non-monotonic noise behavior explained via partial-engagement resonance mechanism (l. 641); (ii) P2-R10: contact stiffness sensitivity documented with solref softening yielding ~60% degradation (l. 109--110); (iii) P2-R11: Bonferroni-adjusted alpha=0.007 noted in Table IV footnote (l. 485); (iv) P2-R13: 46-config footnote corrected to 35+5+6 breakdown, no longer 14x3x5=210 (l. 595 footnote). *Evidence:* All verified against current manuscript text.

**S5. Factorial design structure is sound.** The additive (L0->L1->L2->L3) and subtractive (S0->S1->S2->S3->S4->S5) chains correctly implement a forward-selection/backward-elimination approach for causal component attribution. Each level adds or removes exactly one mechanism, and the double-count of L3=S0 serves as a consistency check on the measurement pipeline. *Evidence:* Table IV, l. 462--477.

**S6. Measurement protocol is well-specified.** Binary search parameters (8 iterations, 5 N tolerance, [10, 160] N range) are explicit. After 8 bisections, resolution is 150/2^8 = 0.59 N, well below the 5 N tolerance. Fresh physics and controller state per bisection iteration eliminates carryover effects. *Evidence:* push_ci.py, l. 34, 106--108.

## Weaknesses

### W1. Inconsistent 95% CI for idle precision
**Problem:** The abstract reports idle RMS as 0.73 +/- 0.07 mm, N=10, with 95% CI [0.68, 0.78] mm. Table IV's footnote reports "ACC 95% CI (between-trial): [0.50, 0.62] mm." These two intervals do not overlap and have different centers (0.73 vs. 0.56). The abstract CI [0.68, 0.78] is mathematically consistent with mean=0.73, std=0.07, N=10, t_0.025,9=2.262 (yielding SEM=0.022, CI=0.73 +/- 0.05). The table CI [0.50, 0.62] cannot be derived from the same mean and standard deviation---its center of 0.56 differs from the reported mean of 0.73 by approximately 2.4 sigma.

**Evidence Anchor:** Abstract l. 49 vs. Table IV footnote l. 437. Verified by computation: CI = 0.73 +/- 2.262*(0.07/sqrt(10)) = [0.68, 0.78]; [0.50, 0.62] is off-center by 0.17.

**Why it matters:** This is the paper's headline quantitative claim ("sub-millimeter idle precision"). An internal inconsistency in its confidence interval undermines credibility. Reviewers and readers must know which CI is correct.

**Suggestion:** Reconcile the two CIs. If the table's [0.50, 0.62] uses a different variance estimator (e.g., pooled across all trials rather than trial-wise means), document the method. If it is an error, replace it with the CI consistent with the reported mean and standard deviation. Confirm which CI computation method was used and apply it consistently throughout the paper.

**Severity:** Moderate
**Confidence:** High

### W2. "Paired t-test" claimed for independent samples
**Problem:** The Table IV footnote (l. 485) reports "Marginal (paired t-test, N=5)" for the L0->L1, L1->L2, L2->L3 comparisons. However, the push_ci.py protocol (l. 144--149) runs each configuration independently: each rep uses `seed = 20260728 * 100 + rep`, and since different controller configurations (L0 vs. L1) take different bisection paths, the measurements are not paired in any meaningful sense. A paired t-test would require that the same random perturbation sequence be applied identically to both configurations being compared, which is not the case here. The appropriate test is an independent-samples (Welch's) t-test with df=8.

**Evidence Anchor:** Table IV footnote l. 485 vs. push_ci.py l. 144--149 (independent seed initialization, no pairing mechanism).

**Why it matters:** The test type affects the standard error and p-values. An independent-samples t-test uses SE = sqrt(s1^2/n1 + s2^2/n2) rather than SE_diff/sqrt(n), which changes the t-statistic and p-value. With n1=n2=5 and similar variances, the difference is modest but should be correct. The Bonferroni-adjusted significance threshold (alpha=0.007) and the conclusion that only L0->L1 is significant are unlikely to change, but the precise p-values may shift.

**Suggestion:** Either (a) change to an independent-samples (Welch's) t-test, or (b) if the measurements are truly paired (e.g., same rep uses identical random seed for both configurations), document the pairing mechanism in the measurement protocol and verify that paired seeds were used. In either case, report the test type and degrees of freedom explicitly.

**Severity:** Minor
**Confidence:** High

### W3. Effect sizes reported as Cohen's d without small-sample correction
**Problem:** The paper reports Cohen's d from pooled SD for N=5 per group (l. 666). With small samples, the pooled variance estimator is biased downward, inflating d. The appropriate metric is Hedges' g, which applies a correction factor of approximately 1 - 3/(4*df - 1). For N=5 per group (df=8), the correction factor reduces d by approximately 10%: d=6.44 becomes g=5.81; d=1.9 becomes g=1.71. These remain "very large" and "large" effects respectively, so the substantive conclusions are unchanged, but the reporting should be statistically precise.

**Evidence Anchor:** l. 666. Verified: pooled SD for L0->L1 = sqrt((1.1^2 + 0.5^2)/2) = 0.854, d = 5.5/0.854 = 6.44, g = 6.44 * (1 - 3/(4*8 - 1)) = 5.81.

**Why it matters:** While the conclusions are robust to this correction, reporting uncorrected Cohen's d for N=5 slightly overstates effect sizes. In a paper that prides itself on statistical honesty, this precision matters.

**Suggestion:** Report Hedges' g alongside or instead of Cohen's d, and note the correction factor applied.

**Severity:** Minor
**Confidence:** High

### W4. LQR baseline survival time standard deviation is implausible
**Problem:** Table V reports LQR, LQR+AW, and 6-State LQR survival times as "0.52 +/- 0.00" seconds (N=60) and "24.2 +/- 2.1" degrees roll RMS. The identical across-configuration values (LQR = LQR+AW = 6-St. LQR) are explained by the PID servo bottleneck. However, `+/- 0.00` for survival time across 60 episodes is mechanically implausible. Even with a deterministic simulator, the paper states that seeds are cycled across {0, 42, 123} (l. 412) and that episodes start from randomized initial joint positions (+/- 0.005 rad). These randomized initial conditions should produce non-zero variance in survival time unless the fall detection threshold is reached at a deterministic simulator timestep independent of initial pose.

**Evidence Anchor:** Table V, l. 581--584. Seed cycling: l. 412--413.

**Why it matters:** A standard deviation of exactly zero for a physical measurement with randomized initial conditions raises questions about measurement precision or rounding. If survival time is rounded to centiseconds or the fall detection threshold is reached at the same timestep regardless of initial conditions, this should be stated.

**Suggestion:** Either (a) report the actual standard deviation at higher precision (e.g., 0.52 +/- 0.003 s), (b) explain that survival time is discretized at the controller timestep (0.01 s) and all episodes terminated at the same step, or (c) confirm through independent re-measurement that the zero variance is genuine and add a footnote explaining the determinism of the fall mode.

**Severity:** Minor
**Confidence:** Medium

### W5. L2 (global-I, works) vs. S4 (global-I, windups) discrepancy not explained
**Problem:** In the additive chain, L2 adds a global integral without any gate to L1's P + homing controller, yielding F_min = 76.3 N (Table IV). In the subtractive chain, S4 removes the proximity gate from Full ACC while retaining the boost and EMA, yielding "windup" behavior. Both configurations contain a global (ungated) integral, yet L2 survives pushes while S4 windups. The paper's footnote (l. 482) states "Global I windup prevents standing" for S4 but does not address why L2's global integral does not suffer the same fate.

**Evidence Anchor:** Table IV, L2 row (l. 466) vs. S4 row (l. 475) and S4 footnote (l. 482).

**Why it matters:** If the same mechanism (global integral) produces opposite outcomes in two configurations, the causal attribution is questionable. The likely explanation is that S4's boost mechanism interacts destructively with the ungated integral (the boost amplifies windup), while L2 has no boost and therefore the integral remains bounded. This explanation would strengthen the ablation logic by showing the interaction between components, but it is currently absent.

**Suggestion:** Add a sentence or footnote explaining why L2's global integral survives while S4's windups---specifically, whether the presence of the damping boost in S4 (but not L2) creates a destructive interaction with the ungated integral.

**Severity:** Minor
**Confidence:** Medium

## Detailed Comments

### Section V: Experimental Validation (l. 409--597)

**V.A Anchored Standing (l. 414--445):**
The idle precision measurement protocol is well-specified (20 s measurement after 5 s settle, N=10, randomized init at +/- 0.005 rad). The within-run CV of 14.3% is reported, which is useful. Two items need attention:

1. The EMA initialization transient (~7.5 s) exceeds the 5 s settle period. The paper correctly notes that precision stabilizes by t=5 s (l. 419), and recommends hardware initialize EMA_0=1.0 (l. 239). However, the measurement window (t=5 s to t=25 s) partially overlaps with the EMA decay tail from its zero-initialization. The supplementary material should include a time-series plot of the EMA during the settle+measurement window to confirm stabilization by t=5 s. Minor methodological concern.

2. The 95% CI inconsistency (see W1 above) must be resolved.

**V.B Omnidirectional Push Recovery (l. 447--505):**
The binary search protocol and factorial design are sound (see S5, S6). Specific comments:

- The p-value hierarchy is correctly reported: only L0->L1 survives Bonferroni at p<0.001. This is honest.
- The statement "L2 (global integral) adds only 1.1 N over L1 (n.s.)" is correctly cautious given N=5.
- The "Marginal" note (l. 485) with Bonferroni-adjusted alpha and explicit CI half-widths for non-significant comparisons is exemplary.
- The CI_95 half-width formula (1.24 * std) uses the correct t_0.025,4=2.776 but should note this is half-width of the per-group CI, not the CI of the difference.

**V.D LQR Baselines (l. 558--596):**

- The 46-config sweep is a strong methodological choice. It preempts the objection that LQR failure is due to poor tuning. The breakdown (35 roll PD combos + 5 wheel velocity limits + 6 cross-coupling gains = 46) is correctly documented and the sweep script is referenced.
- The anti-windup validation script (`scripts/validate_lqr_aw.py`) should be cited in the table footnote rather than only in the body text (l. 569), so that a reader examining Table V can immediately find the verification.
- The identical 6-State and 4-state LQR values are the paper's strongest causal claim (PID servo is the bottleneck). The argument is logically sound, but the zero-variance survival times (W4) somewhat weaken the presentation.

**V.E Architectural Analysis (l. 599--654):**

- The robustness sweep (Table VI) is a genuine 4x3 factorial design with two dependent variables (idle RMS, F_max). This is methodologically stronger than the component ablation because it varies independent factors orthogonally. The "sharp delay bifurcation" finding (F_max drops from ~95 N to ~45 N at any non-zero delay) is well-supported by the data.
- The non-monotonic noise behavior explanation (partial-engagement resonance, l. 641) is mechanistically plausible and was correctly added per P2-R9. However, the paper should note that this explanation is a post-hoc hypothesis---the original experiment was not designed to test the resonance mechanism. A brief acknowledgment ("we hypothesize") would be appropriate.
- Gate parameter sensitivity (l. 644--647): The finite-difference gradient analysis showing release coefficient dominance (220.4 sensitivity, 20x the next parameter) is novel and practically useful. The 5-step tuning guide is a genuine contribution for practitioners. However, the gradient magnitudes depend on the perturbation size, which is not reported.

### Section VI: Discussion (l. 657--692)

**Statistical Power (l. 664--667):**
This subsection is a model of honest reporting and should be retained. The minimum detectable effect calculation, Bonferroni discussion, and Clopper-Pearson bounds are all correct. Two minor additions would strengthen it:
- State explicitly which comparisons were pre-specified vs. post-hoc.
- Report the Bonferroni-adjusted p-values for all 7 comparisons, not just the significant one.

**Limitations (l. 669--680):**
Item (9) correctly restates the N=5 power limitation. Item (10) honestly notes that LQR baseline SDs are cross-episode aggregates and per-episode distributions are not logged. This transparency is commendable.

### Reproducibility Checklist

| Criterion | Status | Notes |
|-----------|--------|-------|
| Scripts referenced by name | PASS | push_ci.py, robustness_sweep.py, validate_lqr_aw.py, sweep_lqr_roll_gains.py all exist |
| Configs version-controlled | PASS | baseline_coupled_lqr.yaml, baseline_lqr_anti_windup.yaml, baseline_lqr_torque.yaml staged in git |
| Commit hash cited | PASS | f92640c (current HEAD) |
| Raw data preserved | PASS | outputs/paper_statistics/ contains JSON files for all major experiments |
| Seeds documented | PASS | l. 412-413 with explicit formulas |
| Measurement protocol parameters | PASS | Binary search: 8 iter, 5 N tol, [10, 160] N range |
| Full parameter set accessible | PARTIAL | "50+ gains" said to be in supplementary material but no file path/URL given |
| Gate threshold values reported | PASS | All in Table II and Eq. 9--11 |
| Noise model parameters documented | PASS | Table VI footnote: 1-sigma values for low/med/high noise |

### Round 7 P2 Fix Verification

| Fix | Description | Status | Evidence |
|-----|-------------|--------|----------|
| P2-R9 | Non-monotonic noise explained | PASS | l. 641: "partial-engagement resonance" mechanism described |
| P2-R10 | Contact stiffness sensitivity | PASS | l. 109-110: solref softening to 0.02 -> 60% RMS increase |
| P2-R11 | Bonferroni alpha=0.007 in Table IV | PASS | l. 485: footnote with adjusted alpha and cross-reference to Sec. VI |
| P2-R13 | 46-config footnote corrected | PASS | l. 595: "35+5+6" breakdown, no longer 14x3x5=210 |

All four fixes are correctly and completely applied.

## Questions for Authors

**Q1. CI reconciliation.** The abstract reports 95% CI [0.68, 0.78] mm for the ACC idle RMS, while Table IV's footnote reports [0.50, 0.62] mm. These intervals do not overlap. Which is correct? If the [0.50, 0.62] CI uses a different variance estimator (e.g., between-trial variance of trial-level means rather than the SEM of the grand mean), please document the method and explain why the two CIs differ. If it is an error, please correct it.

**Q2. Test type for pairwise comparisons.** The Table IV footnote states "paired t-test" for the L0->L1, L1->L2, L2->L3 comparisons. However, push_ci.py runs each configuration independently with different random seeds---there is no pairing mechanism (the same rep number does not use the same random perturbation sequence across configurations). Were these comparisons computed as paired or independent-samples tests? If paired, what is the specific pairing mechanism? If independent-samples, please correct the test type and report Welch's t-test results.

**Q3. L2 vs. S4 global integral discrepancy.** L2 (global integral without gate, in the additive chain) achieves F_min=76.3 N and survives pushes, while S4 (proximity gate removed from full ACC, also a global integral) exhibits windup. What explains this difference? Is the damping boost in S4 responsible for amplifying the integral windup? A one-sentence explanation in the ablation discussion would close this apparent contradiction.

**Q4. LQR survival time variance.** Table V reports survival times of 0.52 +/- 0.00 s for three LQR configurations (N=60 each). With randomized initial joint positions (+/- 0.005 rad) and three cycled seeds, is the standard deviation genuinely zero to within the reported precision, or is survival time discretized at the 0.01 s controller timestep with all 60 episodes terminating at the same step? If the latter, please note the discretization.

## Minor Issues

1. **l. 419:** "precision stabilizes by t=5 s" but EMA zero-init transient is ~7.5 s. Add a sentence clarifying how precision stabilizes before the EMA fully decays, or show the settle verification in supplementary material.

2. **l. 485:** "CI_95 half-width approx 1.24 * std" should specify that this is the half-width of the per-group 95% CI (using t_0.025,4 = 2.776, and 2.776/sqrt(5) = 1.24), not the CI of the difference between groups.

3. **l. 569:** The AW validation script (`validate_lqr_aw.py`) is cited in the body text but not in Table V's footnote. Add a cross-reference in the table footnote for readers who examine the table independently.

4. **l. 644:** The finite-difference gradient magnitudes for gate parameter sensitivity do not report the perturbation size used. Add the perturbation step (e.g., +/- 1% or +/- 0.01) to make the sensitivity values interpretable.

5. **l. 709:** "structural ablation configs (L0--L3, S0--S5) are implemented via base-controller parameter modification and documented in the repository supplementary material." No specific file path or URL is given for this supplementary material. Add a path (e.g., `configs/ablations/` or an appendix listing the parameter modifications for each config).

6. **l. 666:** Cohen's d reported without small-sample correction. Either apply Hedges' g or note that the correction is negligible for the conclusions drawn.

7. **l. 641:** The partial-engagement resonance explanation for non-monotonic idle RMS is a post-hoc hypothesis, not a pre-registered prediction. Qualify with "we hypothesize" or "a likely mechanism is..." to distinguish from the pre-registered threshold tests.

8. **l. 640--641:** "The apparent improvement from medium to high noise at 0 ms delay (34 mm -> 3 mm RMS) occurs because high noise holds the EMA permanently above threshold, disengaging the anchor---the robot reverts to P-only behavior." This is a strong mechanistic claim. Consider adding a footnote that this interpretation is consistent with the EMA threshold model but was not independently verified (e.g., by checking whether the EMA signal actually stayed above threshold during high-noise trials).

## Dimension Scores

| Dimension | Score (0-100) | Notes |
|-----------|--------------|-------|
| Methodological Rigor (25%) | 82 | Sound factorial design, pre-registered thresholds, explicit measurement protocols. Deductions for CI inconsistency (W1), test-type ambiguity (W2), and missing small-sample correction (W3). |
| Evidence Sufficiency (25%) | 78 | All claims supported by quantitative data except L2/S4 discrepancy (W5) and zero-variance LQR survival (W4). N=5 power limitation honestly discussed. |
| Statistical Validity (25%) | 75 | Bonferroni correction and power analysis correctly reported. Deductions for CI inconsistency, test-type claim, and uncorrected effect sizes. The statistical discussion in Sec. VI is otherwise exemplary. |
| Reproducibility (25%) | 88 | Scripts, configs, seeds, raw data, and commit hash all documented. Only gap: supplementary material path for full parameter set and ablation configs not specified. |

**Weighted Total: 81/100** (Minor Revision)
