Now I have a thorough understanding of the paper in its current state after 8 rounds of revision, the entire review history, and the editorial decisions. Let me produce a comprehensive, independent Domain Expert review.

---

# Peer Review Report -- Domain Expert

## Reviewer Identity

Senior researcher in wheeled bipedal robotics and balance control with 15+ years of experience in LQR/WBC/MPC for legged and wheeled-legged robots. Extensive publication and review record at IEEE RA-L, T-RO, ICRA, and IROS. Deep familiarity with the wheeled-biped literature including Ascento (Klemm et al.), Handle (Boston Dynamics), DIABLO (Liu et al.), SKATER (Wang et al.), Whleaper (Zhu et al.), BITeno (Wada et al.), Swiss-Mile (Bjelonic et al.), and coupled pitch-roll LQR formulations (Zamani et al., Xin et al.).

## Review Focus

Domain-specific: literature coverage completeness, theoretical framework appropriateness, positioning within the wheeled-biped field, domain contribution significance, and technical correctness of claims about related work.

## Overall Recommendation

**Minor Revision**

## Confidence Score

**5/5** -- I have direct hands-on experience with wheeled biped platforms using LQR/WBC for balance, have implemented coupled pitch-roll controllers on articulated-leg morphologies, and am intimately familiar with the practical challenges this paper addresses (PID servo lag, integral windup, stiffness-precision trade-off). My assessment of the literature coverage, baseline fairness, and contribution significance is made with high confidence.

## Summary Assessment

This paper presents the Anchored Conditionally-Gated Controller (ACC), a classical control architecture for a 10-DOF wheeled biped whose core mechanism is a proximity-gated position integral enabled by an asymmetric envelope follower. The paper has traversed an unusually thorough revision process, and the current manuscript reflects that investment: the LQR baseline comparison now spans four controlled variants (PID-servo, integral-AW, direct-torque, and coupled 6-state 3D LQR), the factorial ablation decomposes the anchor bundle into individually attributable sub-components via additive (L0-L3) and subtractive (S0-S5) paths, and the limitations section enumerates 10 specific, honest constraints on the claims.

From a domain expertise perspective, the asymmetric envelope follower (tau_attack ~23 ms, tau_release ~1.5 s) is genuinely novel in the wheeled-biped context and elegantly resolves a real engineering tension I have observed in my own platform work: stiff proportional gains that deliver precision collapse under pushes, while integral action that cancels equilibrium bias windups during disturbances. The S5 finding -- that a damping boost without pitch gating creates parametric oscillation instead of merely failing to damp -- is a genuinely insightful negative result that reveals an interaction effect rather than a simple necessity test.

The coupled 6-state 3D LQR baseline, tested through the PID servo path and yielding performance identical to the 4-state baseline (0.52 s survival, 24.2 deg roll RMS), provides the strongest evidence that the PID servo path is a binding constraint on this morphology. The direct-torque LQR variant (35% survival improvement, 10x torque reduction, but still 100% fall rate) cleanly demonstrates that removing servo lag is necessary but not sufficient.

However, I identify issues that prevent acceptance without revision. The platform comparison table (Table I) makes several characterizations that are incomplete or potentially misleading relative to the cited literature. The characterization of the "dominant bottleneck" as being the PID servo path is partially contradicted by the paper's own data: direct-torque LQR still falls 100%, and the deepened Zamani et al. comparison reveals platform-level factors beyond servo architecture that the current narrative does not adequately acknowledge. The flight and terrain demonstrations remain at N=3-4 for key conditions and are characterized as "qualitative feasibility proofs" -- language that overstates what N=3 binary outcomes can support (Clopper-Pearson 95% CI [0.29, 1.0]). The literature review omits proprioceptive state estimation references directly relevant to the hardware transfer claims.

These are all text-level or single-experiment issues, not fundamental threats to the core contribution. The underlying empirical work is rigorous, honestly reported, and makes a genuine contribution to the wheeled-biped control literature.

## Strengths

**S1. The coupled 6-state 3D LQR baseline with controlled ablations is the most rigorous classical-baseline evaluation I have seen in a wheeled-biped control paper.**
Four LQR variants differ by exactly one controlled factor each: integral channel (LQR vs. LQR+AW), action path (PID-servo vs. direct-torque), and state dimensionality (4-state decoupled vs. 6-state coupled). This design enables clean causal attribution that is rare in our field. The 46-configuration gain sweep eliminates parametric-tuning confounds. The identical 6-state and 4-state performance (both 0.52 s survival) is the single strongest piece of evidence for the PID-servo-path bottleneck claim.
*Evidence: Section V-D, Table V, lines 560-596.*

**S2. The asymmetric envelope follower with fast attack (~23 ms) and slow release (~1.5 s) is a genuinely novel mechanism in wheeled-biped control.**
The symmetric-EMA ablation confirming bistable failure (post-push cycle holds EMA in mid-band, boost half-open, oscillation perpetual) is compelling. The time-domain attack-vs-release trade-off as an alternative to the estimation-speed-vs-noise trade-off in DOB/ADRC is a crisp intellectual contribution. The gate activation figure (Fig. 3) pedagogically illustrates the complete push-recovery gating sequence.
*Evidence: Section III-C, Eqs. 7-9, Fig. 3, Table III ablation rows.*

**S3. The S5 finding -- pitch gate removal causes parametric oscillation -- is genuinely insightful.**
This is not merely "removing component X degrades performance Y." It is "removing component X creates a qualitatively different failure mode (parametric oscillation at 56 mm, worse than the P-only limit cycle)." This reveals an interaction between the boost and pitch dynamics that is non-obvious and demonstrates that the ablation design can detect unexpected interactions rather than merely confirming design intent.
*Evidence: Table IV, S5 row: F_min collapses from 80 N to 9 N, 56 mm oscillation.*

**S4. Honest statistical reporting.**
The paper explicitly reports that with N=5, only L0-L1 is significant under Bonferroni correction (alpha=0.007), that sub-2 N differences are indistinguishable from noise, and that N=3 binary outcomes have Clopper-Pearson CI [0.29, 1.0]. This transparency is rare in our field and builds substantial reader trust.
*Evidence: Table IV footnotes, Section VI statistical power discussion, lines 670-672.*

**S5. Practical controller design with hardware-aligned specifications.**
The raw-torque output, 400 Nm/s rate limit, 100 Hz control, standard sensor requirements (IMU + encoders), and staged commissioning plan (static torque -> fixed-support -> free standing -> push recovery -> flight -> terrain) reflect genuine hardware deployment thinking. The gate parameter tuning procedure (tau_release ~ 3-5x ringdown period, envelope band ~2-3x idle velocity noise floor) provides actionable guidance for porting to new morphologies.
*Evidence: Section VI "Path to Hardware," gate sensitivity analysis, Table II.*

**S6. Ten explicit, specific limitations.**
The limitations section enumerates: backward push weakness, oblique ledge limits, simulation-only status, latency sensitivity below 10 ms, push degradation during height transitions (84%), coupled-LQR-through-PID-servo scope, architectural generality untested, statistical power constraints, advanced WBC/MPC baselines deferred, and the scheduled-k_p algebraic loop. Each is specific and falsifiable rather than a generic caveat. This is exemplary practice.
*Evidence: Section VI, lines 674-683.*

## Weaknesses

### W1. Platform comparison table contains incomplete characterizations of cited platforms.

**Severity:** Medium. **Confidence:** 4. **Evidence Anchor:** Table I, lines 295-348.

The table makes several claims that a domain expert familiar with the cited platforms would question:

**(a) Ascento push recovery characterized as "Omnidir. (WBC)."** The cited paper [klemm2020lqrwbc] demonstrates WBC with kinematic loops for balancing but does not report an omnidirectional push recovery envelope or push force magnitudes. The entry creates an impression of a quantitative comparison that does not exist in the cited work. Furthermore, Ascento has demonstrated capabilities that this table does not reflect: stair climbing via RL [chamorro2024stairclimbing], locomotion speeds exceeding 8 km/h, and controlled jumping with landing recovery. These are capabilities ACC does not currently demonstrate and provide important context.

**(b) SKATER height capability as "Contin."** The cited paper [skater2024] demonstrates high-speed turning robustness and terrain adaptability via convex MPC. While MPC can optimize CoM height as a continuous variable, I am not aware of documented continuous height squatting across a 0.40-0.70 m range on SKATER. If this entry is inferred from the MPC formulation rather than a demonstrated capability, that should be stated.

**(c) Missing capability rows.** The table omits dimensions where comparison platforms have documented capabilities that ACC does not: locomotion speed (Ascento >8 km/h, SKATER high-speed turning, Swiss-Mile driving+walking), stair climbing (Ascento via RL), and jumping height (Ascento). Including these would provide honest context for ACC's current limitations.

**(d) "---" convention.** The footnote states "---" means "not reported in cited work." This is a defensible convention, but it lumps together genuinely absent capabilities with capabilities that exist but were measured with different metrics. For example, DIABLO's direct-drive joints likely achieve good idle precision; the absence of a reported number does not mean the capability is absent.

**Suggestion:** (1) Add a "Locomotion speed" row. (2) Add a "Stair climbing" row. (3) Change Ascento push recovery to "WBC (qual.)" to indicate absence of quantitative push envelope data in the cited work. (4) Add an explicit footnote differentiating "not reported in cited work" from "demonstrably absent." (5) Verify SKATER "Contin." height entry against the cited paper or change it.

### W2. The "dominant bottleneck" claim is partially contradicted by the paper's own evidence.

**Severity:** Medium-High. **Confidence:** 5. **Evidence Anchor:** Section V-D, Table V, lines 87-88, 592-596.

The paper argues that the PID servo action path is "the dominant bottleneck" preventing LQR from succeeding on this platform. The evidence supporting this includes: (i) the direct-torque LQR achieves a 35% survival improvement and 10x torque reduction, and (ii) the coupled 6-state LQR through the PID servo performs identically to the 4-state baseline.

However, three data points create an unresolved tension:

| Configuration | Servo Lag? | Outcome |
|--------------|------------|---------|
| Zamani et al. (2020) -- rigid two-wheeled, direct-drive | No | Survives (theta_RMS ~1.5 deg) |
| ACC 6-State LQR -- legged morphology, PID servo | Yes | Falls 0.52 s (100%) |
| ACC Direct Torque LQR -- legged morphology, direct torque | No | Falls 0.70 s (100%) |

If servo lag were the sole distinguishing factor, the Direct Torque LQR should look more like Zamani's result than the PID-servo result. It does not. The 35% improvement and 10x torque reduction are significant but insufficient to prevent falling. This means platform-level factors beyond servo architecture -- rigid vs. articulated body, mass distribution, actuator authority, contact dynamics -- are contributing substantially to the outcome gap. The paper's current narrative at lines 87-88 reads: "...the ~0.25 s servo lag -- absent in Zamani's direct-drive platform -- introduces phase margin loss that the coupled LQR cannot overcome, confirming the servo path as the dominant bottleneck." This sentence implies servo lag is *the* distinguishing factor between Zamani's success and ACC's LQR failure, which the Direct Torque data contradicts.

Additionally, the claim at line 596 -- that "Re-optimizing gains and benchmarking model-free RL confirm that neither servo-lag removal nor unconstrained learning resolves the height-adaptive posture trade-off" -- is not fully supported. The direct-torque LQR used gains optimized for the PID-servo plant, ported unchanged to direct torque. This makes the Direct Torque result a conservative lower bound. Gains re-optimized for the direct-torque plant would likely improve further. The paper acknowledges this implicitly (the coupled 6-state direct-torque experiment is "deferred to future work," line 596) but the headline narrative does not reflect this conservatism.

**Suggestion:** (1) Add 1-2 sentences explicitly acknowledging that platform-level differences (rigid body vs. articulated legs, mass distribution, actuator authority) contribute to Zamani's success beyond servo architecture alone. (2) Acknowledge that the Direct Torque LQR gains were optimized for the servo plant, making the result a conservative lower bound. (3) Narrow the "dominant bottleneck" characterization to "the dominant bottleneck among the factors tested on this platform" or "the primary identifiable bottleneck."

### W3. Flight and terrain demonstrations below community evidence standards for publication claims.

**Severity:** Low-Medium. **Confidence:** 4. **Evidence Anchor:** Section V-C, Table VII, lines 506-549.

The paper characterizes flight recovery and terrain adaptation as "qualitative feasibility proofs." However:
- Ledge 50 cm: N=4
- Terrain 10/15/20 cm curb: N=3 each
- No-adaptation baseline: N=3

With N=3 binary outcomes (3/3 survival), the Clopper-Pearson 95% CI is [0.29, 1.0]. The paper honestly reports this CI, but "feasibility proof" overstates what N=3 supports. A single failure would produce 2/3 survival with CI [0.09, 0.99].

Furthermore, the flight controller (Eq. 13) is a 3-term PD with 2 Nm saturation. While the 0/10 survival with free-spinning wheels establishes necessity, an ablation against a simpler alternative (e.g., wheels locked at constant velocity, or a constant nose-up bias torque) would establish that the PD structure specifically matters rather than any non-zero wheel torque being sufficient given favorable passive dynamics.

**Suggestion:** (1) Increase N to at least 10 for the ledge 50 cm and 20 cm curb conditions. (2) Or downgrade language from "feasibility proof" to "preliminary qualitative demonstration" with an explicit statement that statistical validation is deferred. (3) Add a simple ablation for the flight controller (e.g., constant nose-up torque).

### W4. Literature review omits proprioceptive state estimation references relevant to hardware transfer claims.

**Severity:** Low-Medium. **Confidence:** 3. **Evidence Anchor:** Section II, Section VI.

The "Path to Hardware" section sketches estimation approaches for sagittal velocity, ground reaction forces, and contact-point positions but does not cite the relevant wheeled-biped estimation literature. The SKATER paper [skater2024] discusses proprioceptive state estimation for wheeled bipeds. The Ascento WBC paper [klemm2020lqrwbc] addresses sensor fusion with kinematic loops. Bjelonic et al. [swissmile2021] discuss proprioceptive control with contact estimation for hybrid wheeled-legged locomotion. Bloesch et al. (2018, "State Estimation for Legged Robots -- Kinematics, Inertial, and Contact Sensing," Int. J. Robotics Research) provides foundational methods for contact estimation from joint torque and kinematics that are directly relevant to the paper's proposed hardware estimation approach.

**Suggestion:** Add 2-3 citations to proprioceptive state estimation literature where velocity, contact force, and contact-point estimation are discussed (Section VI).

### W5. Hyon et al. (2007) gated control comparison is underdeveloped.

**Severity:** Low. **Confidence:** 4. **Evidence Anchor:** Line 98.

The paper states: "Gated control for humanoid balancing was explored by Hyon et al. [hyon2007gated]; ACC extends conditional gating to wheeled bipeds with continuous smoothstep transitions." This is a one-sentence dismissal of the closest conceptual predecessor. Hyon's work uses state-dependent switching between balance strategies based on CoM state relative to the support polygon -- the intellectual lineage deserves more development. Specifically: (a) what do ACC's smoothstep gates add beyond discrete switching? (b) do Hyon's gating conditions (support polygon boundaries) have any analogue in ACC's proximity and envelope gates? (c) is the key distinction the specific gating variable (support polygon vs. spatial proximity + velocity envelope) or the transition mechanism (discrete switch vs. continuous smoothstep)?

**Suggestion:** Add 2-3 sentences developing the Hyon comparison, clarifying what specifically ACC adds beyond discrete-switching gated control.

## Detailed Comments

### Literature Review Completeness

The literature review has improved substantially across revision rounds. The coverage of wheeled biped platforms is now comprehensive: Ascento, Handle, DIABLO, Whleaper, SKATER, BITeno, and Swiss-Mile are all cited with accurate descriptions of their contributions. The process control literature (Shinskey 1996, Astrom & Hagglund 2005/2006) is now properly cited for override/selector-based control, and the hybrid/switched systems connection (Liberzon 2003) provides appropriate theoretical context.

Two gaps remain:
1. **Proprioceptive state estimation for wheeled/legged systems** -- See W4 above.
2. **Deeper engagement with Whleaper's LQR+PPO residual architecture** -- The paper notes (line 86) that Whleaper is "closest to our morphology, uses LQR+PPO residual architecture" and states that "ACC targets the same paradigm with a stronger classical prior." This is an accurate characterization, but the paper could strengthen its positioning by noting that ACC's anchor mechanism is specifically designed to provide a more capable classical prior for exactly this hybrid paradigm, connecting the current contribution to the stated future work of residual RL.

### Theoretical Framework

ACC's theoretical framework is sound within its scope. The paper correctly frames its contribution as an engineering control architecture rather than a control-theoretic advance. The smoothstep gates provide C0-continuous transitions (C1 for smoothstep itself, C0 when composed with the C0 asymmetric EMA), which the paper honestly acknowledges via footnote.

The paper's positioning within classical control theory is now appropriate: it cites override/selector-based control, gain scheduling, cascade control, and hybrid/switched systems, correctly distinguishing ACC's continuous blending from discrete switching while acknowledging the lineage.

One theoretical observation: the gate-state bifurcation analysis (Section V-E) identifies a pathological partial-engagement regime where anchor performance is worse than P-only. The paper treats this as a tuning observation, but it is more fundamentally a limitation of threshold-based gating: a controller with N binary-threshold gates has 2^N operating regimes, some of which may be pathological, and the controller has no mechanism to detect or escape pathological regimes once entered. A brief acknowledgment of this as a fundamental limitation (not merely a tuning corner case) would strengthen the theoretical framing.

### Positioning vs. Prior Work

ACC is well-positioned relative to classical LQR-based controllers. The four-variant LQR comparison with controlled ablations makes a compelling case that off-the-shelf LQR through a PID servo layer cannot stabilize this morphology, and that direct-torque LQR extends survival but does not prevent falling.

The positioning relative to optimization-based methods (WBC, MPC) is weaker. The paper acknowledges this honestly: Limitation 9 states "advanced optimization-based WBC/MPC methods are deferred." This is acceptable for a letter-format submission focused on classical control, but the abstract should carry a scoping qualifier (currently the abstract states LQR baselines "all fail" without the "among the classical LQR variants evaluated" qualifier that appears in the limitations).

The positioning relative to learned controllers is partially supported. The pure PPO baseline (100M steps) demonstrates that unconstrained model-free RL does not trivially solve the multi-height balance problem, which supports the paper's motivation. However, the claim that "ACC is hypothesized as a precondition for RL on this platform" (line 66) has zero supporting evidence beyond the PPO-only failure. No ACC+RL experiment has been conducted. This should be clearly marked as a hypothesis (the paper does use "hypothesized" once at line 65) but the word "precondition" implies necessity -- "structured prior" or "candidate base controller for residual RL" would be more accurate.

### Technical Correctness of Related Work Claims

Several claims about related work require qualification:

1. **Line 61:** "no single classical controller integrates all three [millimeter-level precision, omnidirectional push recovery, terrain adaptation]." This is arguably true for wheeled bipeds specifically, but quadrupedal WBC controllers (e.g., Bellicoso et al. 2018) integrate precision, disturbance rejection, and terrain adaptation. The claim should be scoped to "wheeled bipedal classical controllers."

2. **Lines 87-88:** The Zamani comparison and servo-lag bottleneck claim -- See W2 above. The logical chain (Zamani succeeds -> no servo lag -> ACC LQR fails because of servo lag) is contradicted by the Direct Torque data.

3. **Line 63:** "omitting integral action leaves a 1.3 Nm equilibrium bias sustaining a persistent 55 mm p-p limit cycle regardless of k_p." The phrase "regardless of k_p" is too strong. The paper swept k_p from 20-55 during tuning; the claim should be "regardless of k_p within the tested range (20-55 Nm/rad)."

4. **Line 89:** The Whleaper comparison states "ACC targets the same paradigm with a stronger classical prior." This is fair and accurate, but the paper should note that Whleaper's residual PPO has been trained and evaluated, while ACC's residual RL hypothesis has not -- making the comparison aspirational rather than demonstrated.

## Dimension Scores

| Dimension | Score (0-100) | Notes |
|-----------|--------------|-------|
| Originality (20%) | 82 | Asymmetric envelope + proximity-gated anchor is genuinely novel in wheeled-biped context. Smoothstep gating extends process-control ideas meaningfully. Individual components exist separately but their specific combination and the gating-by-physical-conditions philosophy are new. The S5 parametric oscillation finding adds a genuinely non-obvious contribution. |
| Methodological Rigor (25%) | 84 | Exemplary factorial ablation with additive and subtractive paths. Four-variant LQR comparison with single controlled differences. 46-configuration gain sweep. Pre-registered thresholds. Binary search for push thresholds. Honest statistical power analysis with Bonferroni correction. Welch's t-test with reported degrees of freedom. Among the most rigorous experimental designs in the wheeled-biped control literature. |
| Evidence Sufficiency (25%) | 70 | Core claims (anchor precision, push survival, LQR failure) are well-supported with N>=5 and appropriate statistics. Flight/terrain demonstrations remain at N=3-4 and are characterized as "feasibility proofs" -- this language overstates what the data support. The combined-hardware-degradation analysis is missing. The k_p schedule is architectural dead weight -- removing it would simplify the architecture without any performance loss. |
| Argument Coherence (15%) | 80 | Clear narrative arc: problem (precision-push trade-off) -> mechanism (anchor + envelope) -> validation (standing + push + ablation) -> characterization (robustness, sensitivity) -> discussion (limitations, hardware path). The LQR baseline analysis is logically structured. Two remaining logical tensions (Zamani/DT contradiction, k_p schedule architectural inconsistency) are not yet fully resolved. |
| Writing Quality (15%) | 78 | Clear technical prose with honest qualification. Gate definitions are crisp and each has a specific physical interpretation. The parameter table is comprehensive. Minor issues: the CI inconsistency between abstract and Table III needs reconciliation; boldface bias in Table I; statistical power discussion should appear before results tables; "ringdown" should be defined at first use in body text. |
| Literature Integration (optional) | 74 | Covers major wheeled-biped platforms comprehensively. Process control lineage now properly cited. Missing: proprioceptive estimation literature for wheeled bipeds; Hyon et al. gated control lineage underdeveloped; cited reference verification needed (several reference keys flagged in prior rounds). |

**Weighted Average:**
(0.20 x 82) + (0.25 x 84) + (0.25 x 70) + (0.15 x 80) + (0.15 x 78) = 16.4 + 21.0 + 17.5 + 12.0 + 11.7 = **78.6 / 100**

**Weighted Average with Literature Integration at 10% (reducing other weights proportionally):**
(0.18 x 82) + (0.225 x 84) + (0.225 x 70) + (0.135 x 80) + (0.135 x 78) + (0.10 x 74) = 14.76 + 18.90 + 15.75 + 10.80 + 10.53 + 7.40 = **78.1 / 100**

**Decision boundary:** Score of 78.1 corresponds to **Minor Revision**. For RA-L, this paper is in the 75-82 range typical of papers accepted after one additional revision cycle.

## Questions for Authors

**Q1. What is your quantitative prediction for a coupled 6-state LQR with direct torque output?**
Based on the 35% survival improvement and 10x torque reduction from switching the 4-state LQR to direct torque, and the zero improvement from adding roll coupling through the PID servo, what would you predict for a coupled 6-state LQR with direct torque output? If you believe it would still fall quickly, what is the next bottleneck after the PID servo path? If you believe it might survive indefinitely (at some level of imprecision), how would that change the narrative about the anchor's necessity vs. its role as a precision mechanism?

**Q2. What is the physical mechanism behind the 10 ms delay bifurcation?**
The robustness data show that any non-zero delay drops F_max from ~95 N to ~45 N, but 10 ms and 30 ms delays produce similar performance (47 vs. 46 N). This suggests a threshold effect rather than continuous degradation. Have you estimated the critical delay analytically (e.g., from the phase margin of the pitch control loop)? The additional phase lag at the Nyquist frequency (50 Hz at 100 Hz control) from a 10 ms delay is ~18 degrees -- is this sufficient to exhaust the available phase margin, or is the mechanism more complex?

**Q3. Why does the scheduled k_p not help push recovery?**
The intuition that softening gains during pushes aids recovery is widespread in the wheeled-biped community. Your data show the opposite: fixed k_p=50 matches or exceeds scheduled k_p in all metrics. Is the slow-release envelope too slow to modulate stiffness during the ~0.5 s push window? Does the stiffness reduction impair the initial counter-torque that arrests pitch divergence? Do you have mechanistic insight into why this widely-held intuition fails on your platform?

**Q4. How do you see ACC interfacing with the Whleaper-style LQR+PPO residual paradigm?**
You state (line 86) that ACC "targets the same paradigm with a stronger classical prior." Can you elaborate on the specific interface? Would ACC replace Whleaper's LQR base controller entirely, or would the anchor mechanism augment the LQR prior? Would the residual policy observe the gate states (g_prox, g_env, etc.) as inputs, enabling it to learn different corrections in different physical regimes?