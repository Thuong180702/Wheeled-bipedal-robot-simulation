Now I have a thorough understanding of the paper and its 8+ round review history. Let me produce a fresh, independent EIC review.

---

## EIC Review Report

### Reviewer Identity: Editor-in-Chief, IEEE Robotics and Automation Letters

### Overall Recommendation: Major Revision

### Confidence Score: 4/5
High confidence in assessment of contribution, structure, and fit. Confidence bounded at 4 because all evidence is simulation-only, and the projected hardware behavior (1-3 mm degradation, 9.5 ms latency margin) relies on estimates rather than measurements. This does not change the recommendation but prevents a 5/5 certainty about real-world transfer.

### Summary Assessment
This manuscript presents the Anchored Conditionally-Gated Controller (ACC), a two-channel torque-assembly for a 10-DOF wheeled biped in MuJoCo simulation. The core mechanism is a proximity-gated position integral confined to a 15 cm neighborhood, enabled by an asymmetric envelope follower (attack ~23 ms, release ~1.5 s) that distinguishes quiet stance from post-push ringdown. ACC delivers 0.73 mm CoM sagittal RMS idle precision (N=10, 95% CI [0.68, 0.78] mm) and omnidirectional push recovery (F_min=83 N, F_med=119 N). Four LQR variants all fail (100% fall rate), and a 100M-step PPO baseline suffers 85% falls on multi-height transitions.

The paper is honest, methodical, and has clearly benefited from extensive revision. The factorial ablation (additive L0-L3 + subtractive S0-S5) is a methodological highlight that isolates per-component contributions with unusual rigor. The 10 enumerated limitations, statistical power disclosure, and explicit conditioning of the 0.73 mm claim on stiff ground contact build genuine reader trust. Flight recovery and terrain adaptation are correctly scoped as independent extensions with per-condition N=10.

However, three issues prevent me from recommending acceptance in current form. First, the "ACC as precondition for RL" framing (Introduction, line 66) is an entirely untested promissory note -- no RL+ACC experiment exists -- that inflates the significance claim beyond what the paper demonstrates. Second, the headline 0.73 mm precision in the abstract omits the contact-model qualifier that readers learn only in Section III (softening solref to tire-on-asphalt increases idle RMS ~60% to ~1.2 mm). Third, the paper's central bottleneck claim -- that the PID servo action path is the "primary" LQR bottleneck -- is in tension with the Direct Torque LQR evidence (removing the servo path improves survival only 35%, with 100% fall rate persisting). These are resolvable within one revision cycle, and none challenges the validity of ACC's empirical performance.

### Strengths

**S1. The proximity-gated anchor with asymmetric envelope follower is a genuinely novel synthesis in wheeled-biped control.** The specific combination of spatial gating (g_prox, 5-15 cm), asymmetric envelope detection (tau_attack ~23 ms, tau_release ~1.5 s), pitch stability interlock (g_theta), and gated damping boost has no direct precedent. The symmetric-EMA ablation confirming bistable failure (post-push EMA held in mid-band, gate half-open, ringdown never decays) demonstrates the asymmetry is essential, not incidental. The S5 pitch-gate ablation -- removing g_theta collapses F_min from 83 N to 9.0 N and induces 56 mm parametric oscillation -- is a genuinely insightful finding: the architecture catches an interaction (boost without pitch gate amplifies rather than suppresses oscillation) that a simpler design would miss.
*Evidence: Section IV-C (Eqs. 4-8), Table IV (S3, S5 rows), Figure 3 (gate activation time series).*

**S2. The factorial ablation design is exemplary.** The two-directional decomposition -- building up from P-only (L0) to full ACC (L3), then stripping components from full ACC (S0-S5) -- isolates per-component marginal contributions while controlling for order effects. The decomposition of the compound L2-L3 intervention (proximity gate + asymmetric EMA + boost + pitch gate) into individually attributable sub-components via subtractive rows (S2-S5) addresses the confound of bundled mechanisms. This ablation structure exceeds the norm in RA-L control papers.
*Evidence: Table IV (full additive and subtractive rows), footnotes documenting Welch's t-test, Bonferroni correction, and Cohen's d.*

**S3. Intellectual honesty about limitations is distinguishing.** Ten enumerated limitations span backward-push weakness, oblique ledge failure, simulation-only status, latency sensitivity (<10 ms), 84% push degradation during height transitions, statistical power constraints (minimum detectable effect ~4.5 N at N=5), and the coupled-6-state LQR's conservative result. The paper reports that scheduled k_p provides no benefit over fixed k_p (S0 vs. S1), that F_med actually degrades from P-only to ACC (the anchor trades some push survival for precision), and that the 0.73 mm claim is conditional on stiff ground contact. This transparency builds significant reader trust and is uncommon in robotics submissions.
*Evidence: Section VI, Limitations (1)-(10); Table IV S0 vs. S1 comparison; Section III contact-parameter disclosure.*

**S4. The LQR baseline comparison is unusually thorough.** Four variants (4-state PID-servo, 4-state + AW, direct-torque re-optimized, coupled 6-state 3D) plus a 100M-step PPO baseline span classical and learning-based approaches. The 46-configuration gain sweep confirms LQR failure is structural, not parametric. The PPO baseline (3.20 s single-height survival, 85% multi-height fall rate, 54.2 mm height RMSE) demonstrates that unconstrained model-free RL does not trivially solve this problem. The honest reporting that PPO outperforms all LQR variants on single-height standing avoids straw-man framing.
*Evidence: Table V, Section V-D, footnote on 46-configuration sweep.*

**S5. The gate sensitivity analysis provides transferable tuning heuristics.** The finite-difference gradient analysis revealing alpha_r dominance (sensitivity 220.4, ~20x the next parameter, ~130x pitch gates) provides actionable guidance for porting ACC. The five-step tuning procedure (identify ringdown period T_r, set tau_release ~3-5x T_r, set quiet-stance band at 2-3x idle velocity noise floor, set pitch gates conservatively, tune k_p) elevates the contribution from "tuned for one robot" to "here is how to tune for yours."
*Evidence: Section V-E, gate sensitivity analysis, five-step procedure.*

**S6. Reproducibility infrastructure exceeds field norms.** Public GitHub repository with specific commit hash (f92640c), complete core parameter table (30+ parameters with units), LQR AW verification script, 46-configuration gain sweep script, and deterministic RNG seeding protocols are provided. This is uncommon in single-author simulation papers.
*Evidence: Acknowledgment section; Table II; documented seed protocols in Section V.*

### Weaknesses

**W1: The "ACC as precondition for RL" framing is an untested promissory note.**
*Severity: Major. Evidence: Introduction, line 66: "ACC is thus hypothesized as a precondition for RL on this platform: a height-general structured prior that a residual policy can later augment; this hypothesis awaits empirical validation via residual RL training." Confidence: 5/5.*

The paper's actual contribution is a standalone classical controller. The Introduction positions ACC as infrastructure for future residual RL -- but no residual RL training has been performed. No evidence exists that ACC's base actions improve RL sample efficiency, asymptotic performance, or sim-to-real transfer. This framing inflates the significance claim without evidence. The paper would be stronger claiming what it has demonstrated: a classical controller achieving sub-millimeter precision and omnidirectional push recovery where LQR and PPO baselines fail.
*Fix: Move the RL-precondition hypothesis to Future Work. Reframe the Introduction to motivate ACC as a standalone contribution solving the precision-push trade-off, and note it may also serve as a structured prior for future learning-based methods.*

**W2: The headline 0.73 mm precision claim in the abstract lacks contact-model qualification.**
*Severity: Major. Evidence: Abstract line 49: "sub-millimeter idle precision (0.73 +/- 0.07 mm CoM RMS)"; Section III lines 112-113: "Softening solref[0] to 0.02 (tire-on-asphalt) increases idle RMS ~60% to ~1.2 mm, so the 0.73 mm claim is conditional on stiff ground contact." Confidence: 5/5.*

The abstract reports 0.73 mm without qualification. The reader learns only in the platform section that this requires idealized stiff ground contact. Table VI (robustness) further shows idle RMS degrades to 2.30 mm under 10 ms delay alone and to 34.00 mm under medium sensor noise. The abstract should carry the qualifier. The claim is technically correct for the stated conditions, but the conditions are buried.
*Fix: Revise abstract to "...sub-millimeter idle precision (0.73 +/- 0.07 mm CoM RMS, N=10, clean sensors, stiff ground contact; degrades to 2.30 mm under 10 ms delay and F_max bifurcates from 96 N to 47 N)."*

**W3: The central bottleneck claim ("primary LQR bottleneck is the PID servo action path") is in tension with the Direct Torque LQR evidence.**
*Severity: Major. Evidence: Abstract line 50: "a primary LQR bottleneck is the PID servo action path"; Table V: Direct Torque LQR removes the servo path but survives only 0.76 s (35% improvement, 100% fall rate persists). Confidence: 4/5.*

Removing the PID servo path improves survival by only 35% and does not prevent the fall. This demonstrates the servo path is a *contributing* bottleneck, but the residual failure after its removal suggests additional constraints -- likely fundamental authority limits of linear control on this nonlinear torque-constrained platform. The abstract's "primary bottleneck" language implies a rank-ordering the data do not fully support. The paper's own evidence shows that removing the servo path alone is insufficient.
*Fix: "The PID servo action path is a key identifiable constraint -- removing it via direct torque output improves survival 35% but does not prevent the fall, indicating additional factors also limit LQR performance on this platform."*

**W4: The platform comparison table (Table I) asymmetrically reports quantitative metrics for ACC and qualitative descriptions for competitors.**
*Severity: Minor. Evidence: Table I rows -- ACC entries include specific numbers (0.73 mm, 119 N, 100 cm drop, 20 cm curb); competitor entries use descriptors ("LQR," "Cascaded LQR," "Convex MPC," "---"). Confidence: 4/5.*

The table caption correctly notes that "---" means "not reported in cited work" and that "quantitative cross-platform comparison is limited by metric heterogeneity." However, the visual impression is that ACC is the only platform with measurable capabilities. This is a reporting asymmetry, not necessarily a performance asymmetry. Ascento has demonstrated omnidirectional WBC-based balancing and jumping on hardware; reporting only "LQR" for its balance controller understates its demonstrated capability.
*Fix: Add a column stating "Metric reported in source?" or strengthen the caption: "Competitor capabilities are summarized from cited literature and may use different protocols; absence of a number does not imply absence of the capability."*

**W5: The latency sensitivity bifurcation (<10 ms threshold) is the most practically consequential result in the paper, yet it is under-discussed.**
*Severity: Minor. Evidence: Table VI -- F_max drops from 96 N to 47 N at any non-zero delay; idle RMS increases 5.5x at 10 ms. The projected hardware latency of ~9.5 ms (Section VI) places ACC within 5% of its measured degradation threshold, leaving 0.5 ms margin. Confidence: 4/5.*

This result means the controller, as designed, operates at the very edge of its viable regime on plausible hardware. The 0.5 ms margin would be consumed by OS scheduling jitter on a standard Linux system. The paper correctly identifies the need for a real-time OS with dedicated microcontroller, but this is a practical deployment barrier that deserves more prominence in the paper's narrative. For an RA-L readership assessing deployability, this is the single most important hardware-transfer finding.
*Fix: Elevate the latency sensitivity finding in the abstract and conclusion. Add one sentence: "The projected 9.5 ms end-to-end latency places ACC within 5% of its measured 10 ms degradation threshold; validating this budget on hardware, or hardening the controller for 10-20 ms latency, is the highest-priority hardware risk to retire."*

**W6: Single morphology, single simulator -- architectural generality is asserted but untested.**
*Severity: Minor. Evidence: Section VI, Limitation (9): "Architectural generality across morphologies and simulators remains to be demonstrated." Confidence: 4/5.*

ACC is validated on one robot model (8.1 kg, 10-DOF) in one simulator (MuJoCo, specific contact model). The gate sensitivity analysis provides tuning heuristics, but no port has been attempted. For RA-L, single-platform validation is acceptable if claims are appropriately scoped. The title and abstract imply broader applicability than demonstrated.
*Fix: Add "on a 10-DOF Wheeled Biped" to the title or abstract to scope the claim.*

### Coverage Receipt
Both Strengths and Weaknesses are populated; no empty sections.

### Detailed Comments

#### Journal Fit
RA-L seeks concise letters presenting a focused contribution with clear significance. The core ACC anchor mechanism -- a proximity-gated integral with asymmetric envelope detection -- is well-suited for letter format. However, the current scope (balance + push recovery + flight + terrain + 5 baselines + robustness + sensitivity + compound disturbances) exceeds what a 6-page letter can adequately present. The main letter should focus on the anchor mechanism, balance, and push recovery; flight/terrain should be clearly marked as extensions or moved to supplementary material. This would also prevent those N=10 demonstrations from diluting the paper's focus on its core contribution.

#### Originality
The specific synthesis of proximity-gated integral, asymmetric envelope follower, pitch safety interlock, and gated damping boost is genuinely novel in wheeled-biped control. The paper correctly distinguishes ACC from saturation-triggered AW (which triggers on actuator limits, not state-dependent position error), DOB/ADRC (frequency-domain estimation cannot separate push from ringdown), and cascade control (nested loops without conditional gating). The S5 parametric oscillation finding adds a dimension absent from prior work. The contribution is incremental in the sense that individual mechanisms are known, but the synthesis -- validated by component-level ablation -- constitutes a genuine contribution.

#### Significance
If the 0.73 mm idle precision transfers to hardware (even degraded to the projected 1-3 mm), this would be significant for the wheeled-biped community. No existing platform reports sub-millimeter standing precision with between-trial statistics and a demonstrated push-recovery envelope. The finding that four LQR variants and 100M-step PPO all fail establishes this as a non-trivial control problem. The gate sensitivity analysis providing transferable tuning heuristics makes the contribution actionable. Significance is moderated by simulation-only validation, single-platform demonstration, and the practical latency constraint (<10 ms). These are acknowledged, not hidden.

#### Structural Coherence
The paper flows logically: problem statement -> prior work -> platform -> architecture -> validation (scenario-to-component mapping) -> analysis -> discussion. The gate sensitivity analysis provides a natural bridge from results to practical guidance. One structural tension: the RL-precondition framing in the Introduction is disconnected from the rest of the paper, which is entirely about a standalone classical controller. A reader expecting an RL-focused contribution will be confused to find none. Either integrate (by providing RL+ACC evidence) or move to Future Work.

#### Title and Abstract
**Title:** "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery" is distinctive and informative. "Anchored" captures spatial confinement. "Conditionally-Gated" accurately describes smoothstep activation by physical conditions. The title implies broader generality than single-platform validation supports.

**Abstract:** Quantitatively specific, which is good. Issues: (a) the 0.73 mm claim lacks noise/contact-model qualification (see W2); (b) the PPO 85% fall rate omits the context that PPO outperforms all LQR variants on single-height standing (3.20 s vs. 0.52-0.76 s); (c) the LQR failure statement implies these four variants exhaust the comparison space -- add "among the classical LQR variants evaluated"; (d) flight/terrain appear as co-equal to the core balance results rather than as independent extensions.

#### Conclusion
The conclusion appropriately summarizes contributions, acknowledges limitations, and identifies future work. The staged commissioning plan and hardware estimation pathways are a strength. One missing element: the conclusion should explicitly state that ACC's hypothesized role as a structured prior for residual RL remains untested and is deferred to future work, closing the loop on the introduction's framing.

### Questions for Authors

1. **RL roadmap.** You frame ACC as a "hypothesized precondition for RL." What is your planned timeline for residual RL training? Have you conducted any preliminary RL+ACC experiments? If not, would you move this hypothesis to Future Work and reframe around ACC's standalone contribution?

2. **Latency budget validation.** Your projected 9.5 ms end-to-end latency places ACC within 5% of its 10 ms degradation threshold. Have you characterized the latency of the IMU+odometry complementary filter for v_sag estimation? Standard complementary filters add 5-20 ms of group delay. If the filter contributes more than 3 ms, ACC would operate in its degraded push regime. Is there a path to hardening the controller for 10-20 ms latency?

3. **Direct-torque coupled 6-state LQR.** The coupled 6-state LQR through PID servo fails identically to the 4-state variant. Direct-torque output improves the 4-state LQR by 35% but does not prevent the fall. What would you predict for a coupled 6-state LQR with direct torque output -- the configuration combining both fixes? This experiment is currently deferred but is the most direct test of the servo-path bottleneck hypothesis.

4. **Contact stiffness and the 0.73 mm claim.** You show that softening solref[0] from 0.002 to 0.02 increases idle RMS by ~60% (to ~1.2 mm). Have you characterized sensitivity to solref[1] (damping) and solimp parameters? The reader needs to know whether 0.73 mm is robust to plausible contact modeling variation or specific to one solver configuration.

### Minor Issues

1. **Ringdown definition:** "Ringdown" is used throughout but never formally defined at first use in the main text body. Define at first occurrence.
2. **Figure 3 caption:** The y-axis label for the top panel (|v_sag|, m/s) should include units explicitly for standalone readability.
3. **Table II:** "Vel. damp. scale" (1.5) and "Boost scale k_vb" (5.0) are dimensionless multipliers -- specify which base gains they multiply.
4. **Gate sensitivity:** Specify the finite-difference perturbation magnitudes for the gradient computation (e.g., +/-1% of nominal). Important for reproducibility of the sensitivity ranking.
5. **Abstract/Table V consistency:** The abstract reports "N=20-60" for LQR baselines; Table V reports N=60 for LQR/LQR+AW, N=20 for DT/6-State/PPO. Ensure consistent reporting.
6. **Reference verification:** Verify all bibliography entries are cited in the text -- several references from earlier rounds may be orphaned.
7. **Compact spacing overrides (lines 16-27):** The aggressive \setlength commands compress layout beyond IEEEtran defaults. Verify compliance with the target journal's style guide.
8. **Abstract line 49:** The push performance degradation under delay (F_max drops from ~95 N to ~47 N) should accompany the precision degradation already mentioned.

### Dimension Scores

- **Originality: 78/100** -- The asymmetric envelope follower + proximity-gated anchor is genuinely novel in wheeled-biped control. The S5 parametric oscillation finding adds depth. The "conditionally-gated" label slightly overstates relative to established process-control lineage (split-range, override, selector architectures), but the paper now acknowledges this lineage. Individual mechanisms are known; the specific synthesis and ablation-validated demonstration that all components are jointly required constitutes the contribution.

- **Methodological Rigor: 82/100** -- The factorial ablation (additive + subtractive) is exemplary. Four-variant LQR comparison with controlled ablations (servo-path, integral channel, state dimensionality) is unusually thorough. Binary-search threshold measurement, N=10 between-trial statistics with 95% CI, noise floor measurement, and gate sensitivity analysis with tuning hierarchy collectively exceed field norms. Statistical power limitations are honestly disclosed. The main gap is the absence of an alternative-paradigm baseline (MPC, or ACC+RL residual), though this is now scoped by Limitation 10.

- **Evidence Sufficiency: 72/100** -- Core claims (0.73 mm idle, 83 N omnidirectional push, anchor component contributions) are well-supported by factorial ablation. The LQR 100% failure claim is well-evidenced within the classical paradigm. Flight/terrain correctly scoped as extensions with N=10. Gaps: no evidence ACC outperforms non-LQR approaches; no evidence for the RL-precondition hypothesis; latency sensitivity is measured but the hardware latency budget is estimated, not validated; all evidence is simulation-only on one platform.

- **Argument Coherence: 85/100** -- Clear logical flow: problem (trilemma) -> mechanism (proximity-gated anchor) -> architecture (two-channel assembly) -> validation (scenario-to-component mapping) -> limitations. The design narrative ("each gate was added after a measured failure mode was traced to a root cause") is pedagogically effective. Honest calibration of what the anchor achieves vs. what it does not builds reader trust. The one structural tension is the RL-precondition framing that the paper never returns to.

- **Writing Quality: 80/100** -- Technical prose is clear and precise. Mathematical specification (Eqs. 1-9) is complete. Tables are information-dense but readable. The gate activation figure (Figure 3) is pedagogically excellent. Remaining issues: the RL-precondition claim creates a structural mismatch; some defensive/repetitive prose in ablation interpretation; the abstract omits key qualifiers on the headline precision claim.

- **Weighted Total: 78.7** = 0.20(78) + 0.25(82) + 0.25(72) + 0.15(85) + 0.15(80) = 15.6 + 20.5 + 18.0 + 12.75 + 12.0 = **78.85**

**Decision mapping: 65-79 = Minor Revision.** The score lands at the upper boundary of Minor Revision territory. I am recommending **Major Revision** rather than Minor because the three Major-severity weaknesses (W1: untested RL framing, W2: unqualified headline claim, W3: bottleneck overstatement) collectively require more than text-level adjustments -- they affect how the paper positions its contribution, frames its central claim, and attributes causality in its comparative analysis. These are all resolvable within one revision cycle, and none challenges the validity of ACC's empirical performance data. The paper's intellectual honesty and methodological rigor are its strongest assets and should be preserved throughout revision.