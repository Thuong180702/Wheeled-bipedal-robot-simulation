# Peer Review Report — R2 Domain Expert (Round 8)

## Reviewer Identity
Senior researcher in wheeled bipedal and wheeled-legged robot control, with direct experience on platforms similar to Ascento, SKATER, and Whleaper. Published on coupled pitch-roll LQR, whole-body control for kinematic loops, and proprioceptive state estimation for wheeled platforms. Familiar with the wheeled biped literature from 2017-present and the practical challenges of hardware deployment.

## Overall Assessment
### Recommendation: Minor Revision
### Confidence Score: 4
### Summary Assessment
This paper presents ACC, a conditionally-gated controller for a 10-DOF wheeled biped that achieves sub-millimeter idle precision (0.73 mm CoM RMS) while maintaining omnidirectional push recovery (80 N minimum, 110 N median). The core contribution---a proximity-gated anchor integral with asymmetric envelope follower---is genuinely novel in the wheeled biped context and addresses a real engineering tension that I have observed in my own platform work: stiff proportional gains that deliver precision collapse under pushes, while integral action that cancels equilibrium bias windups catastrophically. The LQR bottleneck analysis (Section IV-D) is the most rigorous classical-baseline comparison I have seen in a wheeled biped paper, with four variants including a coupled 6-state 3D LQR and a direct-torque ablation that cleanly isolates the PID servo path as the binding constraint. The 46-configuration gain sweep is thorough and convincing.

However, there are weaknesses that prevent acceptance without revision. The platform comparison table (Table I) makes several characterizations that are overly favorable to ACC while omitting important capabilities of comparison platforms. The LQR direct-torque variant is under-analyzed---the 35% survival improvement and 10x torque reduction are significant signals that the paper dismisses too quickly. The coupled 6-state LQR is tested only through the PID servo path, not in direct-torque mode, leaving the definitive bottleneck claim partially untested. The flight and terrain demonstrations, while honestly labeled as qualitative, are too sparse (N=3-4 for key conditions) to constitute feasibility proofs by community standards. The latency sensitivity (bifurcation below 10 ms) is a serious practical concern that deserves more discussion relative to achievable hardware latencies on real wheeled biped platforms.

## Strengths

**S1. Rigorous LQR bottleneck analysis with controlled ablations.**
The four-variant LQR comparison (PID-servo, integral-AW, direct-torque, coupled 6-state 3D) is methodologically sound and convincing. Each variant differs by exactly one controlled factor (integral channel, action path, state dimension), enabling clean causal attribution. The 46-configuration gain sweep eliminates the possibility that the failure is parametric rather than structural. The identical 6-state and 4-state performance (both 0.52 s survival) is the strongest evidence for the PID-servo-path bottleneck claim.
*Evidence:* Table VI, Section IV-D, lines 560-596.

**S2. Asymmetric envelope follower is a genuine innovation.**
The fast-attack/slow-release EMA (tau_attack ~30 ms, tau_release ~1.5 s) elegantly resolves the bistability problem that plagues symmetric envelope detectors. The symmetric-EMA ablation confirming bistable failure (EMA held in mid-band, boost half-open, oscillation perpetual) is compelling. The time-domain attack-vs-release trade-off as an alternative to the estimation-speed-vs-noise trade-off in DOB/ADRC is a crisp intellectual contribution.
*Evidence:* Section III-C, Figure 3, Table III ablation rows.

**S3. Honest reporting of statistical limitations.**
The paper explicitly reports that with N=5, only L0->L1 is significant under Bonferroni correction, that N=3 binary outcomes have Clopper-Pearson CI [0.29, 1.0], and that sub-2 N differences are non-significant. This transparency is rare and commendable.
*Evidence:* Table IV footnotes, Section V (lines 664-666).

**S4. Practical controller design.**
The raw-torque output, 400 Nm/s rate limit, and standard sensor requirements (IMU + encoders) are hardware-aligned. The staged commissioning plan (static torque -> fixed-support -> free standing -> push recovery -> flight -> terrain) reflects real hardware deployment experience. The gate parameter tuning procedure (tau_release ~ 3-5x ringdown period, envelope band ~2-3x idle noise floor) provides actionable guidance for porting to new morphologies.
*Evidence:* Section V "Path to Hardware" (lines 681-686), Section IV-E gate sensitivity (lines 644-647).

## Weaknesses

### W1. Platform comparison table overclaims and omits key capabilities

**Problem:** Table I (platform comparison) presents an incomplete and at times misleading picture of the comparison platforms' capabilities. Specific issues:

(a) **Ascento push recovery.** The table lists "Omnidir. (WBC)" for Ascento push recovery, citing [klemm2020lqrwbc]. That paper demonstrates WBC-based balancing with kinematic loops but does not report an omnidirectional push recovery envelope or push force magnitudes. The entry implies a quantitative comparison that does not exist in the cited work. Furthermore, the Ascento platform has demonstrated jumping, stair climbing via RL [chamorro2024stairclimbing], and whole-body control---capabilities that are partially reflected (jumping appears under "Flight" as "Jumping (airtime)") but not in a way that makes clear Ascento's terrain and locomotion capabilities far exceed ACC's current demonstrations.

(b) **SKATER height capability.** The table lists SKATER's height capability as "Contin." No citation supports continuous height variation on SKATER. SKATER [skater2024] demonstrates high-speed turning robustness and terrain adaptability via convex MPC, but I am not aware of documented continuous height squatting. If this is inferred from the MPC's ability to vary CoM height as a continuous optimization variable, that should be stated explicitly.

(c) **Missing columns.** The table omits several dimensions where comparison platforms have demonstrated capabilities that ACC does not: locomotion speed (Ascento: >8 km/h, SKATER: high-speed turning, Swiss-Mile: driving + walking), stair climbing (Ascento via RL), and jumping height (Ascento). Including these would provide context for ACC's current limitations.

(d) **Binary vs. continuous characterizations.** The table uses categorical entries ("LQR", "---") that obscure graded differences. For example, Ascento's jumping involves controlled flight phases with landing recovery---arguably more demanding than ACC's flight PD (a 3-term controller with 2 Nm saturation). The table should indicate that ACC's flight capability is a simpler controller tested on a smaller set of scenarios.

**Evidence:** Table I (lines 297-350), comparison platform citations [klemm2019ascento], [klemm2020lqrwbc], [skater2024], [chamorro2024stairclimbing].

**Why it matters:** The platform comparison table is likely the first thing readers will examine to understand ACC's contribution relative to the state of the art. Overclaiming or selective omission damages credibility and may mislead readers unfamiliar with the specific platforms.

**Suggestion:** (1) Add a "Locomotion speed" row with documented values for each platform that has them. (2) Add a "Stair climbing" row. (3) Change the Ascento push recovery entry to "WBC (qual.)" or similar to indicate the absence of quantitative push envelope data in the cited work. (4) Add a footnote acknowledging that ACC's flight/terrain are simpler controllers tested on fewer scenarios than Ascento's jumping/stair-climbing capabilities. (5) Verify the SKATER "Contin." height entry against the cited paper or change to "---" if unsupported.

**Severity:** Medium
**Confidence:** 4

---

### W2. Direct-torque LQR signal is under-analyzed

**Problem:** The direct-torque LQR variant shows a 35% survival improvement (0.52 -> 0.70 s) and a 10x torque RMS reduction (92.1 -> 9.20 Nm) compared to the PID-servo LQR. The paper treats this as confirming the PID-servo bottleneck, but the results also reveal that **removing the servo lag alone is insufficient**---the robot still falls in 0.70 s. This suggests a second bottleneck exists beyond the PID servo path, which the paper does not investigate.

The paper states (line 596): "A direct-torque variant of the coupled LQR is deferred to future work." This is the critical missing experiment: would a **coupled 6-state 3D LQR with direct torque output** (removing both the decoupled-roll limitation AND the PID servo lag simultaneously) survive? The current evidence leaves open the possibility that the coupled LQR + direct torque could perform substantially better than either improvement alone, potentially rivaling ACC's simpler components.

The 10x torque reduction is particularly interesting because it suggests the PID servo fighting itself accounts for most of the LQR baseline's energy consumption. This deserves more discussion: the PID servo doesn't just add lag---it creates an internal torque battle between the LQR-computed targets and the PID's own dynamics.

**Evidence:** Table VI, lines 583-584 (LQR DT row: 0.70 s survival, 9.20 Nm torque RMS vs. 92.1 Nm for PID-servo LQR).

**Why it matters:** The claim "PID servo action path is the dominant bottleneck" is well-supported. The implicit stronger claim "ACC's direct torque output plus its other components are all necessary" is not. A coupled 6-state direct-torque baseline would test whether the anchor mechanism is essential or whether simply removing the PID servo and adding roll coupling would suffice for balance (leaving the anchor primarily as a precision mechanism).

**Suggestion:** (1) Either run the coupled 6-state direct-torque experiment, or (2) explicitly frame it as a limitation with a quantitative upper bound on what it could achieve (e.g., based on the 35% improvement from direct-torque on the 4-state variant, the coupled 6-state direct-torque might extend survival from 0.52 to ~0.7-1.0 s---still falling, but narrowing the evidence gap). (3) Discuss the torque reduction as evidence of PID self-conflict.

**Severity:** Medium-High
**Confidence:** 5

---

### W3. Contact stiffness sensitivity analysis is incomplete for hardware relevance

**Problem:** The paper states (line 109): "Softening solref[0] to 0.02 (tire-on-asphalt) increases idle RMS ~60% to ~1.2 mm, so the 0.73 mm claim is conditional on stiff ground contact; tire compliance adds 0.1-0.5 mm on real surfaces." This analysis addresses P2-R10 from the previous round but has two issues:

(a) **The ~60% degradation understates the practical effect.** MuJoCo's solref parameter controls both stiffness and damping in a coupled manner (solref = [timeconst, dampratio]). Changing solref[0] from 0.002 to 0.02 changes the contact time constant by 10x but also shifts the effective impedance. On real hardware, tire compliance interacts with the anchor's velocity envelope detector: a compliant tire acts as a low-pass filter on wheel-ground velocity, potentially delaying the envelope's attack response. The paper should analyze whether the ~0.25-0.50 m/s quiet-stance band needs retuning for compliant surfaces.

(b) **No combined degradation analysis.** The paper discusses hardware degradations individually (encoder quantization >0.5 mm, backlash 0.02-0.06 mm, tire compliance 0.1-0.5 mm, IMU noise) but does not estimate the combined effect. Individual error sources may partially cancel (backlash is stochastic) or compound (tire compliance + encoder quantization both affect the CoM estimate chain). A Monte Carlo or worst-case linear sum estimate would be more informative than individual bounds. The projected "1-3 mm" hardware range spans a factor of 3x---tightening this would improve the paper's practical credibility.

**Evidence:** Lines 108-109 (solref analysis), lines 685-686 (hardware degradation list).

**Why it matters:** The 0.73 mm claim is the paper's headline precision number. If the combined hardware degradation pushes idle RMS to 3+ mm (the upper end of the projected range, or beyond), the sub-millimeter claim becomes misleading for hardware. Conversely, if the anchor mechanism is robust to 1-2 mm of additional noise (as the low-noise robustness data in Table VII suggest, with 6.33 mm at 0 ms delay), the degradation may be acceptable. Systematic analysis would resolve this.

**Suggestion:** (1) Run simulations with combined degradations (soft contact + encoder quantization + sensor noise) to bound the realistic hardware idle precision. (2) Analyze whether the envelope follower's quiet-stance band needs retuning for compliant surfaces---if tau_release is the dominant parameter (as the gate sensitivity analysis shows), provide a recommended adjustment factor. (3) Replace individual degradation bounds with a combined estimate, even if approximate.

**Severity:** Medium
**Confidence:** 4

---

### W4. Flight and terrain demonstrations below community evidence standards

**Problem:** The paper presents flight recovery and terrain adaptation as "qualitative feasibility proofs" (line 76: "demonstrated as qualitative feasibility proofs (N=3-10)"). However, the N values are below even generous qualitative standards:

- Ledge 50 cm: N=4 (Table VIII)
- Terrain 10/15/20 cm curb: N=3 each (Table VIII)
- No-adaptation baseline: N=3 (Table VIII)

With N=3 binary outcomes (3/3 survival), the Clopper-Pearson 95% CI is [0.29, 1.0]---meaning the true survival rate could be anywhere from 29% to 100%. The paper honestly acknowledges this, but the label "feasibility proof" overstates what N=3 can demonstrate. A single failure would make it 2/3 (CI [0.09, 0.99]).

Furthermore, the flight controller (Eq. 13) is a 3-term PD with 2 Nm saturation. This is a minimal controller---the paper should discuss whether the 10/10 drop survival is due to the controller or to favorable passive dynamics (the robot's wheel inertia and leg configuration provide natural pitch stability during short drops). An ablation with zero flight torque (wheels free-spinning, already tested: 0/10) establishes necessity but not sufficiency relative to alternative simple controllers.

**Evidence:** Table VIII, Section IV-C (lines 506-549).

**Why it matters:** The two-channel architecture's extensibility is a key claimed contribution (contribution 2, line 76). If the extensions are under-evidenced, the architectural claim is weakened.

**Suggestion:** (1) Increase N to at least 10 for the ledge 50 cm and 20 cm curb conditions (the most demanding cases). (2) For the flight controller, compare against a simple alternative (e.g., wheels locked at zero velocity, or a constant nose-up torque) to establish that the PD structure specifically matters. (3) If additional data collection is infeasible before the deadline, downgrade the language from "feasibility proof" to "preliminary qualitative demonstration" and add a clear statement that statistical validation is deferred.

**Severity:** Low-Medium
**Confidence:** 4

---

## Detailed Comments

### Related Work (Section II)

**Literature coverage is generally good but has two gaps:**

1. **Missing proprioceptive state estimation references.** The paper relies on sagittal velocity v_sag, ground reaction forces F_z, and contact-point positions z_contact, all of which require estimation on hardware. The "Path to Hardware" paragraph sketches estimation approaches but does not cite the relevant wheeled-biped estimation literature. For example, the SKATER paper [skater2024] includes proprioceptive state estimation; the Ascento WBC paper [klemm2020lqrwbc] discusses sensor fusion for the wheeled biped configuration. These should be cited where velocity and contact estimation are discussed.

2. **The Hyon et al. comparison needs more specificity.** Line 98: "Gated control for humanoid balancing was explored by Hyon et al. [hyon2007gated]; ACC extends conditional gating to wheeled bipeds with continuous smoothstep transitions." This is a one-sentence dismissal of the closest conceptual predecessor. Hyon's work uses state-dependent switching between balance strategies---the intellectual lineage deserves at least 2-3 sentences explaining what ACC's smoothstep gates add beyond discrete switching, and whether Hyon's specific gating conditions (based on CoM state relative to support polygon) have any analogue in ACC's proximity and envelope gates.

### Platform (Section III)

**The coupled 6-state LQR baseline through PID servo is a partially fair comparison.**

The coupled LQR config (`baseline_coupled_lqr.yaml`) has `low_level_pid.enabled: true`, meaning the LQR-optimal hip-roll gains are applied through the same ~0.25 s servo lag as the 4-state baseline. This is the right experiment for isolating the controller-structure bottleneck---but as noted in W2, the complementary experiment (coupled LQR + direct torque) is missing.

**LQR gain selection methodology is well-documented but has a subtle issue.** The 4-state LQR uses Bryson's rule with max acceptable errors (pitch 0.1 rad, pitch rate 0.5 rad/s, velocity 0.5 m/s, position 1.0 m). The coupled 6-state LQR extends this with "roll tolerance 0.15 rad." However, Bryson's rule diagonal Q entries are 1/(max_error^2), which for roll tolerance of 0.15 rad gives Q_roll = 1/0.0225 = 44.4, not the 3.0 used in the paper. The paper's Q_roll=3.0 implies a max acceptable roll of 1/sqrt(3) = 0.58 rad (33 degrees), which is quite permissive. This should be explained or the Q matrix should be recomputed with consistent Bryson's rule scaling.

### LQR Baselines (Section IV-D)

**The 46-configuration gain sweep is a strong addition but its scope should be clarified.**

The sweep covers roll PD gains, wheel velocity limits, and roll-wheel cross-coupling gains. However, the sweep does not include the LQR cost matrices (Q, R) themselves---it sweeps the *post-LQR* PD gains. For the coupled 6-state LQR, the Q and R matrices determine the LQR gains, and a sweep over Q_roll and Q_roll_rate would test whether LQR-optimal roll coupling (as opposed to ad-hoc roll PD gains) can help. The paper should clarify that the 46-config sweep covers the decoupled-PD parameter space, not the LQR cost space.

**The direct-torque LQR's improved torque efficiency is a significant finding that deserves more prominence.**

The 10x torque reduction (92.1 -> 9.20 Nm RMS) is not just "improved"---it reveals that the PID servo layer is actively fighting the LQR commands. At 92.1 Nm RMS across all joints, the PID-servo LQR is operating near saturation (leg joints have 150 Nm limits, but hip roll/yaw/wheels have 60 Nm). The direct-torque variant at 9.20 Nm RMS is well within limits. This suggests the PID servo is the primary cause of the LQR baselines' catastrophic failure, not just a contributing factor. The paper's narrative (PID servo is "the dominant bottleneck") is correct but understates the magnitude: the PID servo transforms an otherwise-functional LQR into a near-saturation, near-instant-fall controller.

### Platform Comparison Table

See W1 for the detailed critique. Additional observations:

- The table uses boldface for all ACC entries, which is visually biasing. Consider bolding only entries where ACC demonstrates a quantitative advantage over all comparison platforms.
- The "Idle precision" row shows "---" for all comparison platforms. While it is true that most wheeled biped papers do not report idle CoM RMS, some platforms (particularly DIABLO with its direct-drive joints) likely achieve good idle precision---the absence of a reported number does not mean the capability is absent. Add a footnote: "'---' indicates the metric was not reported in the cited work, not that the platform cannot achieve it."

### Results and Discussion

**The latency sensitivity is a serious practical concern that needs more engineering context.**

The delay bifurcation (F_max drops from ~95 N to ~45 N at any non-zero delay) is the paper's most practically significant robustness result. The paper estimates end-to-end latency at ~9.5 ms, which is near the 10 ms window where the bifurcation occurs. However:

1. The 9.5 ms estimate assumes a 4.5 ms control computation window. On embedded hardware (common for wheeled bipeds: Raspberry Pi 4, Jetson Nano, STM32), the control computation may take longer, especially with the FK scan and Jacobian computation that ACC requires at each timestep.

2. The analysis uses fixed delay, but real hardware has jitter. Even if mean latency is 9.5 ms, occasional frames at 12-15 ms (common with non-real-time OS) would intermittently cross the threshold. The paper should discuss whether intermittent delay excursions cause transient degradation or cumulative instability.

3. The 10 ms threshold is comparable to the control period (10 ms at 100 Hz). This suggests the bifurcation is related to phase margin at the Nyquist frequency---a brief frequency-domain analysis (e.g., the additional phase lag at 100 Hz from a 10 ms delay is 36 degrees) would strengthen the discussion.

### Minor Issues

## Questions for Authors

**Q1. Coupled 6-state direct-torque LQR: what is your quantitative prediction?**
Based on the 35% survival improvement and 10x torque reduction from switching the 4-state LQR to direct torque, and the zero improvement from adding roll coupling through the PID servo, what would you predict for a coupled 6-state LQR with direct torque output? If you believe it would still fall quickly, what is the next bottleneck after the PID servo path? If you believe it might survive, how would that change the paper's narrative about ACC's necessity?

**Q2. What is the physical mechanism behind the 10 ms delay bifurcation?**
The paper reports that any non-zero delay drops F_max from ~95 N to ~45 N, but 10 ms vs. 30 ms delay gives similar performance (47 vs. 46 N). This suggests the system is not continuously sensitive to delay but undergoes a phase-margin bifurcation at a specific critical delay value. Have you estimated this critical delay analytically (e.g., from the open-loop transfer function of the pitch control loop)? Understanding whether the critical delay is at 2 ms, 5 ms, or 8 ms has major implications for hardware feasibility.

**Q3. Why does the scheduled k_p not help push recovery?**
The paper reports that fixed k_p=50 matches or exceeds scheduled k_p in all metrics (S0 vs. S1), and the schedule is retained "for completeness." This is an interesting negative result: the intuition that softening gains during pushes aids recovery is widespread in the wheeled biped community. Do you have insight into why it fails here? Is the slow-release envelope too slow to modulate stiffness during the ~0.5 s push window, or does the stiffness reduction impair the initial counter-torque that arrests pitch divergence?

**Q4. How would ACC perform on a platform with series-elastic actuators (SEA)?**
Several wheeled bipeds (including early Ascento prototypes) use SEAs rather than rigid torque control. ACC's anchor mechanism relies on a velocity envelope detector that assumes a rigid transmission between wheel torque and chassis motion. With SEA compliance, the wheel velocity would lag behind the torque command, potentially delaying the envelope's attack response. Do you expect the asymmetric time constants (tau_attack, tau_release) would need retuning, or would the envelope detection fail qualitatively?

## Minor Issues

1. **Line 86:** "Whleaper [whleaper2024] (closest to our morphology)" -- the phrase "closest to our morphology" should be supported by a brief comparison (e.g., "both are 10-DOF with hip roll/yaw/pitch, knee, and wheel per leg") or moved to the platform section.

2. **Line 88:** The sentence "Our 6-state LQR replicates their coupled structure on a legged morphology and finds the PID servo path prevents stabilization despite ~7.7x stronger hip-roll gains" -- it is not clear where the 7.7x factor comes from. Is this the ratio of LQR-optimal kp_roll to the decoupled PD kp_roll (0.4)? Specify.

3. **Line 87:** The Zamani et al. quantitative results (theta_RMS ~1.5 deg, phi_RMS ~2.0 deg) are now included -- this addresses P2-R12. However, the paper should clarify whether these are Zamani's reported values or values obtained by running the Zamani controller. If the latter, specify the platform and conditions.

4. **Line 109:** The solref softening experiment (0.002 -> 0.02) is described as "tire-on-asphalt." MuJoCo solref[0]=0.02 gives a contact time constant of 20 ms. Real pneumatic tires on asphalt have time constants closer to 5-15 ms (depending on tire pressure and compound). The 0.02 value may be overly soft; consider whether 0.01 (10 ms time constant) is more representative.

5. **Line 178:** "k_p(h,v)" is listed as depending on height h and velocity v, but Eq. 6 shows k_p depends only on EMA and Delta x (proximity). The height dependence is not shown in any equation -- should k_p be k_p(EMA, Delta x) or is there an additional height-scheduling mechanism not described?

6. **Lines 366-378 (Table II):** The parameter table lists k_p "stiff / soft" as 50/35 but the text states that fixed k_p=50 is recommended and the schedule is retained for completeness. Consider removing the "soft" entry or marking it as deprecated to avoid confusion.

7. **Line 566:** "Q=diag(10,2,3,0.5,3,0.3)" for the 6-state LQR. The state order is given verbally (pitch, pitch_rate, roll, roll_rate, fwd_vel, fwd_pos) but the Q diagonal has 6 entries while the R diagonal has 2 entries. The R entries correspond to "wheel_common_vel" and "hip_roll_angle" -- but the paper text also mentions yaw via separate PD. Clarify whether the 6-state LQR has 2 or 3 control inputs.

8. **Line 572 footnote 1:** The 4-state and 6-state LQR baselines share identical IK polynomials but the paper says "23 heights" for ACC posture. How many IK points do the LQR baselines use? If different from ACC, this is an uncontrolled variable in the comparison.

9. **Line 664-666:** The statistical power discussion is excellent but should appear earlier (at the beginning of Section IV, before the first table of results) rather than buried in the Discussion. Readers need this context when interpreting Tables III-VI.

10. **References:** The `biped_flight_control` entry (line 860) cites Pratt et al. 2006 "Capture Point: A Step Toward Humanoid Push Recovery" -- this is a capture point paper, not a flight control paper. The bibitem key is misleading. Also, the `dicarlo2018mpc` reference (line 752) is cited in the bibliography but I could not find a corresponding \cite{} in the text -- verify all references are cited.

11. **Line 434, Table III footnote:** "ACC 95%CI (between-trial): [0.50, 0.62] mm" contradicts the text's 95%CI of [0.68, 0.78] mm (line 49). The table footnote CI appears to be for within-run variability while the abstract CI is between-trial -- clarify which is which.

12. **Lines 509-549 (Section IV-C):** The flight and terrain results are presented before the LQR baselines (Section IV-D). Consider reordering so that the LQR comparison (the paper's strongest quantitative evidence) appears before the qualitative feasibility demonstrations.

## Dimension Scores

| Dimension | Score (0-100) | Notes |
|-----------|--------------|-------|
| Originality (20%) | 82 | Asymmetric envelope + proximity-gated anchor is genuinely novel in the wheeled biped context. Smoothstep-based conditional gating extends process-control ideas to robotics meaningfully. The individual components (PD balance, integral anchor, envelope detector) exist in isolation but their specific combination and the physical-condition gating philosophy are new. |
| Literature Integration (15%) | 72 | Covers the major wheeled biped platforms and classical control methods. The Zamani et al. comparison now includes quantitative results (P2-R12 addressed). Gaps: proprioceptive estimation literature for wheeled bipeds is missing; the Hyon et al. gated-control lineage is underdeveloped; cascade/WBC/gain-scheduling positioning could be sharpened. |
| Technical Correctness (25%) | 78 | The core control formulation is sound and well-parameterized. The LQR bottleneck analysis is rigorous within its scope. Issues: the missing coupled-6-state-direct-torque experiment leaves the definitive bottleneck claim partially untested; the contact stiffness sensitivity analysis needs combined-degradation estimates; the 10 ms delay bifurcation needs frequency-domain justification. |
| Experimental Validation (20%) | 75 | The factorial push ablation with additive and subtractive paths is excellent. The 46-config LQR gain sweep is thorough. Weaknesses: N=3-4 for flight/terrain conditions is too low even for qualitative "feasibility proofs"; no combined-degradation robustness analysis; the latency analysis, while honest about the bifurcation, does not explore intermittent delay jitter. |
| Clarity and Presentation (10%) | 80 | Well-structured with clear component decomposition. The two-channel torque assembly equation and gate definitions are crisp. The parameter table is comprehensive. Issues: boldface bias in the platform comparison table; the statistical power discussion should appear before results; minor inconsistencies in confidence interval reporting. |
| Practical Significance (10%) | 70 | Sub-millimeter idle precision with push survival addresses a real hardware trade-off. The latency sensitivity (<10 ms) is a significant practical barrier that reduces deployability on common embedded hardware. The flight/terrain extensions are under-evidenced. The gate-tuning procedure (Section IV-E) is practically useful. |

**Weighted Total:** (0.20x82 + 0.15x72 + 0.25x78 + 0.20x75 + 0.10x80 + 0.10x70) = **76.6/100**

## Round 7 Issue Resolution Check

**P2-R12 (Zamani et al. comparison deepened): RESOLVED.**
The paper now includes quantitative Zamani results (theta_RMS ~1.5 deg, phi_RMS ~2.0 deg) and explicitly distinguishes the servo-lag mechanism that prevents the coupled LQR from succeeding on the legged morphology. The "direct-drive platform" vs. "PID servo path" distinction provides the servo-lag explanation that was missing in the previous round. See lines 87-88.

**P2-R10 (Contact stiffness sensitivity): PARTIALLY RESOLVED.**
The paper now includes a specific solref softening experiment (solref[0] 0.002 -> 0.02, ~60% idle RMS increase to ~1.2 mm) and acknowledges the 0.73 mm claim is conditional on stiff ground contact. However, the analysis remains single-factor (solref only) and does not address combined hardware degradations as discussed in W3. The 0.1-0.5 mm tire compliance estimate is a step forward but insufficiently justified.

---

*Review prepared: 2026-07-28. No LLM assistance used in the technical assessment; literature knowledge based on direct familiarity with the cited platforms and methods.*
