Now I have a thorough understanding of both the paper (in its current, substantially revised form) and the extensive multi-round review history. Let me produce the EIC review.

---

## EIC Review Report

### Reviewer Identity
Editor-in-Chief, IEEE Robotics and Automation Letters (RA-L). Research background: 15+ years in legged and wheeled robotics, whole-body control (WBC), model predictive control (MPC) for floating-base systems, and simulation-to-reality transfer. I evaluate this manuscript from the bird's-eye-view perspective appropriate to an editorial decision: journal fit, originality, significance, structural coherence, and overall readiness for archival publication.

### Overall Recommendation
**Major Revision**

### Confidence Score
**4/5.** I am confident in my assessment of the paper's contribution, structure, and fit. The primary source of residual uncertainty is the projected behavior of the coupled 6-state LQR baseline under fully re-optimized gains for the direct-torque plant -- the paper reports this now (Surv 0.76 s, 100% fall rate), but the Q/R matrices and re-optimization details are only partially described, and I cannot independently verify that further retuning would not close the gap. This does not change my recommendation, but it limits my confidence in declaring the comparative narrative fully settled.

### Summary Assessment
This manuscript presents the Anchored Conditionally-Gated Controller (ACC), a two-channel torque-assembly architecture for a 10-DOF wheeled biped robot. The core mechanism is a proximity-gated position integral enabled by an asymmetric envelope follower (fast-attack ~23 ms, slow-release ~1.5 s) that distinguishes quiet stance from post-push ringdown. The controller achieves 0.73 mm CoM sagittal RMS idle precision (N=10, 95% CI [0.68, 0.78] mm) and omnidirectional push recovery (F_min = 80 N, F_med = 110 N, 8-direction sweep). Four LQR variants all fail (100% fall rate within 0.76 s), and a 100M-step PPO baseline suffers an 85% fall rate on multi-height transitions.

The paper has been through extensive revision and is now in substantially improved form compared to earlier versions. The coupled 6-state 3D LQR baseline has been added (Surv 0.52 s), the PPO baseline (100M steps) provides a learning-based reference, the idle precision claim is supported by N=10 with 95% CI, the factorial ablation (L0-L3 additive, S0-S5 subtractive) isolates per-component contributions, contact parameters are documented, and missing references have been added. The paper is intellectually honest: seven explicit limitations are enumerated, flight/terrain results are labeled as independent extensions with per-condition N=10, and the latency sensitivity is quantified rather than hidden.

However, three issues prevent me from recommending acceptance in current form. First, the paper frames ACC as a "hypothesized precondition for RL on this platform" (Introduction) and a "height-general structured prior that a residual policy can later augment" -- but this hypothesis is entirely untested. The paper's actual contribution is a standalone classical controller; the RL-precondition framing is an untested promissory note that inflates the significance claim without evidence. Second, the statistical power for the ablation study is acknowledged as insufficient: Welch's t-test at N=5 yields a minimum detectable effect of ~4.5 N, meaning sub-2 N differences between configurations (e.g., S1 vs. S3, Delta F_min = 1.1 N, CI_95(S1) ~ +/-1.0 N, CI_95(S3) ~ +/-2.7 N) are non-significant. The ablation design is methodologically sound, but the N=5 replication limits the conclusions that can be drawn about fine-grained component contributions. Third, the paper remains simulation-only (MuJoCo), and while the hardware path discussion is commendably detailed, the headline 0.73 mm precision claim is explicitly conditional on stiff ground contact (solref = {0.002, 1}); softening to tire-on-asphalt increases idle RMS ~60% to ~1.2 mm. The abstract should carry this qualification.

These are bounded, achievable revisions. The underlying contribution -- a proximity-gated anchor mechanism with asymmetric envelope detection that resolves the precision-push trade-off -- is genuine, well-validated, and of interest to the RA-L community.

### Strengths

**S1: The proximity-gated anchor with asymmetric envelope follower is a genuinely novel mechanism in wheeled biped control.**
The specific combination of spatial gating (g_prox, 5-15 cm band), asymmetric envelope detection (tau_attack ~23 ms, tau_release ~1.5 s), pitch stability interlock (g_theta), and gated damping boost has not been previously demonstrated. The symmetric-EMA ablation confirming bistable failure (post-push oscillation holds EMA in mid-band, gate half-open, ringdown never decays) provides convincing evidence that the asymmetry is essential, not incidental. The pitch-gate ablation (S5: removing g_theta collapses F_min from ~80 N to 9.0 N, producing 56 mm parametric oscillation) is a genuinely insightful finding that demonstrates the architecture catches interactions a simpler design would miss.
*Evidence anchor: Section IV-C (Eqs. 4-5, gates defined in Eqs. 6-8), Table IV (Push Ablation, S3 and S5 rows), Figure 3 (gate activation time series).*

**S2: The factorial ablation design (L0-L3 additive, S0-S5 subtractive) is exemplary by field standards.**
The two-directional decomposition -- building up from P-only (L0) to full ACC (L3), then stripping components from full ACC (S0-S5) -- isolates per-component marginal contributions while controlling for order effects. The decomposition of the L2-L3 bundle (proximity gate + asymmetric EMA + boost + pitch gate, added together in L3) into individually attributable sub-components (S2-S5, each removed subtractively from full ACC) addresses the confound of compound intervention. This ablation structure is stronger than what appears in most RA-L control papers.
*Evidence anchor: Table IV (Push Ablation), full additive and subtractive rows with F_min, F_med, Idle RMS, and Ringdown per configuration; footnotes documenting Welch's t-test and Bonferroni correction.*

**S3: Intellectual honesty about limitations is commendable.**
Seven explicit limitations are enumerated in the Discussion (Section VI): backward push weakness, oblique ledge descent failure, simulation-only status, latency sensitivity <10 ms, 84% push degradation during height transitions, coupled 6-state LQR limitations, and acknowledged statistical power constraints (minimum detectable effect ~4.5 N at N=5). The paper reports that scheduled k_p provides no benefit over fixed k_p (S0 vs. S1), that F_med degrades from 123 N (P-only) to 110 N (ACC) -- the anchor trades some push survival for precision -- and that the 0.73 mm claim is conditional on stiff ground contact. This transparency builds significant reader trust.
*Evidence anchor: Section VI, Limitations (1)-(9); Table IV S0 vs. S1 comparison showing scheduled k_p adds zero benefit; Section III, MuJoCo contact parameter disclosure (solref = {0.002, 1}).*

**S4: The baseline comparison is now appropriately broad.**
Four LQR variants (4-state PID-servo, 4-state + AW, direct-torque re-optimized, and coupled 6-state 3D) plus a 100M-step PPO baseline provide coverage across classical and learning-based approaches. The 46-configuration LQR gain sweep confirms that failure is structural, not parametric. The PPO baseline (3.20 s survival at single height, 85% fall rate on multi-height transitions, 54.2 mm height RMSE) demonstrates that unconstrained model-free RL does not trivially solve this problem. The honest reporting that PPO outperforms all LQR variants on single-height standing (3.20 s vs. 0.52-0.76 s) avoids the straw-man trap of earlier versions.
*Evidence anchor: Table V (LQR and PPO baselines vs. ACC), Section V-D discussion, footnote on 46-configuration sweep.*

**S5: The gate sensitivity analysis provides transferable tuning heuristics, not just parameter values.**
The finite-difference gradient analysis revealing that alpha_r (release coefficient, sensitivity 220.4) dominates by 20x over the next parameter and 130x over pitch gates provides actionable guidance for porting ACC to new morphologies. The five-step tuning procedure (identify ringdown period T_r, set tau_release ~3-5 x T_r, set quiet-stance band at 2-3x idle velocity noise floor, set pitch gates conservatively, tune k_p) elevates the paper from "we tuned it for our robot" to "here is how to tune it for yours."
*Evidence anchor: Section V-E, gate sensitivity analysis with finite-difference gradients, five-step tuning procedure.*

**S6: The hardware transfer discussion is among the most detailed in the simulation-only wheeled-biped literature.**
Specific estimation methods (IMU + odometry complementary filter for v_sag, joint-torque-based F_z estimation, forward-kinematics contact-point estimation), latency budgets (end-to-end ~9.5 ms, 0.5 ms margin below 10 ms bifurcation threshold), noise characterization, real-time OS recommendations, and a staged commissioning sequence (static torque -> fixed-support -> free standing -> push -> flight -> terrain) demonstrate genuine deployment thinking. The explicit acknowledgment that the 0.5 ms margin is "insufficient for industrial deployment" and that a dedicated balance-loop microcontroller is needed is the right level of caution.
*Evidence anchor: Section VI "Path to Hardware," latency budget, staged commissioning sequence.*

**S7: Reproducibility infrastructure exceeds field norms.**
Public GitHub repository with specific commit hash (f92640c), complete core parameter table (Table II, 30+ parameters with units), LQR AW verification script, 46-configuration gain sweep script, and deterministic RNG seeding protocols are provided. This level of transparency is uncommon in simulation-based robotics papers.
*Evidence anchor: Acknowledgment section with URL and commit hash; Table II; documented seed protocols in Section V.*

### Weaknesses

**W1: The "precondition for RL" framing is an untested promissory note.**
*Severity: Major. Evidence anchor: Introduction, "ACC is thus hypothesized as a precondition for RL on this platform: a height-general structured prior that a residual policy can later augment; this hypothesis awaits empirical validation via residual RL training." Confidence: 5/5.*
The paper's actual contribution is a standalone classical controller. The Introduction frames ACC as a "precondition" for future residual RL -- but this hypothesis is entirely untested. No residual RL training has been performed. No evidence is presented that ACC's base actions improve RL sample efficiency, asymptotic performance, or sim-to-real transfer. This framing inflates the paper's significance claim (ACC is not just a controller, it is infrastructure for a future learning system) without evidence. The paper would be stronger if it claimed what it has actually demonstrated: a classical controller that achieves sub-millimeter precision and omnidirectional push recovery where LQR and PPO baselines fail. The RL-precondition hypothesis belongs in future work, not in the introduction's motivation.
*Suggestion: Move the RL-precondition hypothesis to Future Work. Reframe the introduction to motivate ACC as a standalone contribution that solves an open problem (the precision-push trade-off in wheeled bipedal balance) and note that it may also serve as a structured prior for future learning-based methods.*

**W2: "Sub-millimeter" precision claim in the abstract lacks noise and contact-model qualification.**
*Severity: Major. Evidence anchor: Abstract, "sub-millimeter idle precision (0.73 +/- 0.07 mm CoM RMS)"; Section III, "The stiff body contact (solref = {0.002,1}) idealizes a rigid ground. Softening solref[0] to 0.02 (tire-on-asphalt) increases idle RMS ~60% to ~1.2 mm, so the 0.73 mm claim is conditional on stiff ground contact." Confidence: 5/5.*
The abstract reports 0.73 mm without qualification. The reader learns only in Section III that this requires idealized stiff ground contact and that realistic tire compliance adds 0.1-0.5 mm. The robustness table (Table VI) shows idle RMS degrades to 2.30 mm under 10 ms delay alone, and to 34.00 mm under medium sensor noise. The abstract should carry the qualification: "under clean sensors and stiff ground contact." The claim is technically correct for the stated conditions, but the conditions are buried.
*Suggestion: Revise abstract to read "...sub-millimeter idle precision (0.73 +/- 0.07 mm CoM RMS, N=10, clean sensors, stiff ground contact; degrades to 2.30 mm under 10 ms delay and F_max bifurcates...)" or equivalent.*

**W3: Statistical power for the ablation study is acknowledged as insufficient for fine-grained comparisons.**
*Severity: Major. Evidence anchor: Table IV footnote, "With N=5, 95% CI half-width ~1.24 x std (t_{0.025,4}=2.776); sub-2 N differences (e.g., S1 vs. S3 Delta F_min = 1.1 N, CI_95(S1) ~ +/-1.0 N, CI_95(S3) ~ +/-2.7 N) remain non-significant." Confidence: 4/5.*
The paper correctly acknowledges this limitation (Limitation 8), which is commendable. However, the ablation study is one of the paper's three claimed contributions (Contribution 3) and serves as the primary evidence for the "each component contributes measurably" claim. At N=5, only the L0-L1 comparison (Delta = +5.5 N, p < 0.001) and the S5 collapse (80 N to 9 N) are statistically distinguishable from noise. The fine-grained claims about L1-L2 (+1.1 N, n.s.), L2-L3 (+4.1 N, p < 0.05 but n.s. after Bonferroni correction at alpha = 0.007), and S1-S4 (all sub-5 N differences) cannot be statistically supported at current sample sizes. The paper should either increase N to achieve adequate power for its comparative claims, or narrow those claims to what the data can support at N=5.
*Suggestion: Either (a) increase N to >=10 for key ablation comparisons, or (b) restructure the ablation narrative around the two unambiguous findings (L0-L1 homing benefit, S5 pitch-gate criticality) and treat the remaining fine-grained comparisons as suggestive trends rather than confirmed effects.*

**W4: Single morphology, single simulator -- architectural generality remains to be demonstrated.**
*Severity: Major. Evidence anchor: Section VI, Limitation (7): "Architectural generality across morphologies and simulators remains to be demonstrated." Confidence: 4/5.*
ACC is validated on one robot model (8.1 kg, 10-DOF, specific mass distribution and joint limits) in one simulator (MuJoCo, specific contact model and integration scheme). The gate sensitivity analysis provides tuning heuristics for porting, but no port has been attempted. The paper cannot claim ACC is a general architecture for wheeled bipedal balance; it can claim ACC works on this specific platform. For RA-L, single-platform validation is acceptable if claims are appropriately scoped. The paper acknowledges this limitation, but the abstract and title ("Anchored Conditionally-Gated Control for Wheeled Bipedal Balance") imply broader applicability than demonstrated.
*Suggestion: Add "on a 10-DOF wheeled biped" or equivalent qualifier to the title or abstract. Retain the general architectural description in the body but scope claims to the validated platform.*

**W5: The asymmetric envelope follower's lineage is under-cited.**
*Severity: Minor. Evidence anchor: Section IV-C Eq. 9 (asymmetric EMA); Related Work, "the concept traces to peak-hold and envelope-detector circuits [astrom2005advanced]." Confidence: 4/5.*
The asymmetric exponentially-weighted moving average with fast-attack/slow-release is a standard signal-processing building block used in audio dynamics processors (compressors, noise gates), battery state-of-charge estimators, and communication receivers since the 1970s. The paper cites only Astrom & Hagglund (2005) for the envelope-detector lineage. Adding 1-2 signal-processing references would strengthen scholarly positioning and acknowledge the technique's history without diminishing the contribution (the novelty is the application to robotic balance gating, not the filter itself).
*Suggestion: Add 1-2 signal-processing citations for the asymmetric envelope detector lineage. One sentence acknowledging the technique's history suffices.*

**W6: No empirical comparison with disturbance observer (DOB) or active disturbance rejection control (ADRC).**
*Severity: Minor. Evidence anchor: Section II-B, "DOB and ADRC provide temporal disturbance estimation but cannot distinguish a push from post-push ringdown -- both occupy the same frequency band." Confidence: 3/5.*
The Related Work section argues that DOB/ADRC cannot solve this problem because push and ringdown occupy the same frequency band. This is a theoretical assertion -- no empirical comparison is provided. The paper already has four LQR baselines and a PPO baseline; adding a DOB or ADRC baseline would strengthen the claim but is not essential given the paper's already-broad baseline coverage. However, the assertion should be softened from a definitive claim ("cannot distinguish") to a hypothesis ("we hypothesize that DOB/ADRC would struggle because...") unless empirical evidence is provided.
*Suggestion: Soften to "We hypothesize that DOB/ADRC would struggle to distinguish push from ringdown because both occupy overlapping frequency bands; empirical comparison is deferred to future work." Alternatively, implement a simple DOB baseline and report results.*

### Detailed Comments

#### Journal Fit
RA-L seeks concise (6-page) letters presenting a single focused contribution with clear significance to the robotics and automation community. The core ACC anchor mechanism -- a proximity-gated integral with asymmetric envelope detection -- is well-suited for a letter-format contribution. The current manuscript, however, attempts to cover balance + push recovery + flight recovery + terrain adaptation + four LQR baselines + PPO baseline + robustness characterization + gate sensitivity analysis + compound disturbances. This scope exceeds what a 6-page RA-L letter can adequately present. I recommend focusing the main letter on the anchor mechanism, balance, and push recovery (with the LQR/PPO baselines), and moving flight/terrain to a clearly marked "Extensions" section or supplementary material. This would also address the concern that flight/terrain (N=10 per condition, but lower task relevance to the core contribution) dilute the paper's focus.

#### Originality
The proximity-gated anchor with asymmetric envelope follower is genuinely novel in the wheeled biped control domain. The specific combination of spatial gating (proximity band), temporal gating (asymmetric EMA), safety interlock (pitch gate), and gated damping boost has not been previously demonstrated. The paper correctly distinguishes ACC from saturation-triggered anti-windup (which triggers on actuator limits, not state-dependent position error during pushes), DOB/ADRC (frequency-domain estimation), and cascade control (nested loops). The S5 parametric oscillation finding (boost without pitch gate creates, not just fails to suppress, oscillation) adds a dimension absent from prior work.

The contribution is incremental in the sense that each individual mechanism (spatial gating, envelope detection, integral control, gain scheduling) is known -- but the specific synthesis, the ablation-validated demonstration that all components are jointly required, and the quantitative evidence that this synthesis resolves the precision-push trade-off constitute a genuine contribution.

#### Significance
If the 0.73 mm idle precision transfers to hardware (even degraded to the projected 1-3 mm), this would be significant: no existing wheeled biped platform reports sub-millimeter standing precision with quantified between-trial statistics and a demonstrated push-recovery envelope. The finding that four LQR variants and a 100M-step PPO policy all fail (100% and 85% fall rates, respectively) establishes that this is a non-trivial control problem. The gate sensitivity analysis providing transferable tuning heuristics (tau_release ~3-5x ringdown period) makes the contribution actionable for other researchers.

However, the significance is moderated by three factors: (1) simulation-only validation, (2) single-platform demonstration, and (3) the practical latency constraint (<10 ms) that requires dedicated real-time hardware for deployment. These are acknowledged limitations, not hidden defects, but they bound the contribution's immediate impact.

#### Structural Coherence
The paper flows logically: problem statement and motivation (Introduction) -> prior work positioning (Related Work) -> platform description (Robot Platform) -> architecture exposition (ACC Formulation, Sections IV-A through IV-F) -> experimental validation (Section V, organized by scenario class with targeted ablations) -> architectural analysis (robustness, sensitivity, compound disturbances) -> discussion and conclusion. The removal of the TWIP pole-migration analysis from earlier versions improves coherence. The gate sensitivity analysis provides a natural bridge from results to practical tuning guidance.

One structural concern: the "ACC is hypothesized as a precondition for RL" framing in the Introduction is disconnected from the rest of the paper, which is entirely about a standalone classical controller. A reader expecting an RL-focused contribution (based on the introduction) will be confused to find none. Either integrate this hypothesis into the narrative (e.g., by providing preliminary evidence that ACC's base actions are useful for RL) or move it to Future Work.

#### Title and Abstract
**Title:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery" is distinctive and informative. "Anchored" captures the spatial confinement of the integral. "Conditionally-Gated" accurately describes the smoothstep activation by physical conditions. The title implies broader architectural generality than the single-platform validation supports; consider adding "on a 10-DOF Wheeled Biped" to scope the claim.

**Abstract:** The abstract is informative and quantitatively specific, which is good. Issues: (a) the 0.73 mm claim lacks noise/contact-model qualification (see W2); (b) the PPO 85% fall rate is presented without the context that PPO outperforms all LQR variants on single-height standing (3.20 s vs. 0.52-0.76 s) -- this selective reporting makes the comparison appear more one-sided than the full data show; (c) "Flight recovery and per-leg terrain adaptation are demonstrated (N=10 per condition)" appears as a co-equal contribution alongside the core balance results -- these are independent extensions that should be scoped as such.

#### Conclusion
The conclusion appropriately summarizes the contribution, acknowledges limitations (simulation-only, latency sensitivity, backward push weakness, push degradation during height transitions), and identifies future work (hardware validation, EMA settle-time characterization, dynamic locomotion). The staged commissioning plan and hardware estimation pathways are a strength. The claim that "ACC tolerates moderate noise but falls under 10x consumer IMU noise" is appropriately cautious.

One missing element: the conclusion should explicitly state that ACC's hypothesized role as a structured prior for residual RL remains untested and is deferred to future work, to close the loop on the introduction's framing.

### Questions for Authors

1. **RL integration roadmap.** The introduction frames ACC as a "hypothesized precondition for RL" and "height-general structured prior that a residual policy can later augment." What is your planned timeline for residual RL training? Have you conducted any preliminary experiments (e.g., using ACC base actions as a fixed prior with a small PPO residual policy)? If not, would you consider moving this hypothesis to Future Work and reframing the paper around ACC's standalone contribution?

2. **Ablation statistical power.** You acknowledge that at N=5, the minimum detectable effect is ~4.5 N and sub-2 N differences are non-significant. Do you plan to increase N for the ablation study? If not, would you restructure the ablation narrative around the two unambiguous findings (L0-L1 homing benefit of +5.5 N, S5 pitch-gate criticality with 80 N to 9 N collapse) and treat the remaining comparisons as suggestive trends?

3. **Coupled 6-state LQR re-optimization.** The coupled 6-state LQR reports Surv = 0.52 s (identical to the 4-state PID-servo variant) with Q = diag(10, 2, 3, 0.5, 3, 0.3), R = diag(0.8, 1.0). Were these Q/R matrices re-optimized for the direct-torque plant, or were they derived for a PID-servo path and ported unchanged? If the latter, what confidence do you have that a fully re-optimized coupled LQR (Q/R tuned specifically for direct torque at 400 Nm/s rate limit) would not achieve significantly better performance?

4. **Hardware timeline.** You provide a detailed hardware transfer discussion with a staged commissioning plan and latency budget. What is your realistic timeline for hardware validation? Do you have access to a physical platform, or is hardware construction part of the future work? This context matters for how reviewers weigh the simulation-only limitation.

### Minor Issues

1. **Section V, Table V (LQR baselines):** The coupled 6-state LQR row reports Surv = 0.52 +/- 0.00 s, roll RMS = 24.2 +/- 2.1 deg, and height RMS = 371 +/- 45 mm -- identical to the 4-state TWIP LQR row. This numerical identity is suspicious. If genuinely identical (which would imply the coupled roll dynamics provide zero benefit over the decoupled PD), this should be explicitly discussed. If a copy-paste error, it should be corrected.

2. **Abstract line count and density:** The abstract is very long (~230 words) and information-dense. Consider a two-tier structure: a 150-word abstract for the paper header, with expanded details in the introduction.

3. **Figure 1 (robot morphology):** The annotated joint layout is clear, but the coordinate frame definition (X forward, Y left, Z up) should be repeated in the caption for standalone readability.

4. **Ringdown definition:** "Ringdown" is first used in Table III's caption but should be defined at first use in the main text (Section V-A).

5. **Gate activation figure (Figure 3):** The grey shading for the 7-step push window is helpful but the y-axis labels on the top panel (|v_sag| and EMA) could use unit clarification (m/s).

6. **Reference formatting:** Reference [20] (Swiss-Mile) lists authors as "M. Bjelonic, P. K. Sankar, C. D. Bellicoso, and M. Hutter" -- verify this author list against the actual publication, as Swiss-Mile typically has additional co-authors.

7. **Table II (parameters):** The "Vel. damp. scale" parameter (1.5) and "Boost scale k_vb" (5.0) are dimensionless multipliers applied to which base gains? Clarify in the caption or add a brief note.

8. **Section V-E gate sensitivity:** The finite-difference perturbation magnitudes for the gradient computation should be specified (e.g., +/- 1% of nominal value). This matters for reproducibility of the sensitivity ranking.

9. **Consistency check:** The abstract reports "N=20-60" for LQR baselines, while Table V reports N=60 for LQR/LQR+AW, N=20 for DT/6-State/PPO, and N=10 for ACC. Ensure consistency between abstract and table.

10. **Platform comparison table (Table I):** Ascento's push recovery is listed as "Omnidir. (WBC)" -- Klemm et al. (2020) demonstrated omnidirectional balancing through WBC with kinematic loops, which is accurately captured. However, the "---" for Ascento idle precision is correct (Ascento does not report standing precision as a metric). Consider adding a footnote clarifying that "---" means "not reported in cited work" (not "tested and failed"), as the paper already does in the table caption.