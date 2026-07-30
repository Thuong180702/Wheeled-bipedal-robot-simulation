I now have a thorough understanding of the paper and its 8-round review history. Here is the complete field analysis with all 5 reviewer configuration cards.

---

# Field Analysis Report

## Paper: "Anchored Conditionally-Gated Control for Wheeled Bipedal Balance and Disturbance Recovery"

**Author:** Van Thuong Nguyen (Independent Researcher)
**Date of Analysis:** 2026-07-29
**Prior Review History:** 8 rounds of internal review; Round 8 decision was Minor Revision with all 4 recommending reviewers converging. 23 weakness sub-claims inventoried, with 7 reaching multi-reviewer consensus.

---

## Dimension 1 — Primary Discipline

**Robotics and Control Systems** -- specifically, classical torque-level balance control for underactuated wheeled bipedal platforms.

The paper designs, implements, and empirically evaluates a deterministic controller (ACC) on a 10-DOF wheeled biped in MuJoCo simulation. The core intellectual contribution is a novel conditional-integration mechanism (proximity-gated anchor with asymmetric envelope follower), which is a control-systems innovation applied to a robotics platform. The paper speaks the language of both fields: torque budgets, PID servo paths, actuator rate limits (robotics) alongside anti-windup taxonomy, hybrid/switched systems, and gain scheduling (control theory).

The primary readership would be found in IEEE RA-L, IEEE/ASME TMECH, ICRA, IROS, and IEEE T-RO.

---

## Dimension 2 — Secondary Disciplines (max 3)

1. **Control Theory (Nonlinear and Hybrid Systems)** -- The proximity-gated anchor with smoothstep transitions, asymmetric envelope follower, and conditional activation of control components directly engages the literature on hybrid/switched systems (Liberzon 2003), anti-windup taxonomy, override/selector control (Shinskey 1996, Astrom & Hagglund 2006), and gain scheduling (Rugh & Shamma 2000). The paper positions ACC as extending selector-based architectures from discrete switching to continuous $C^1$ blending.

2. **Mechatronics and Actuator Systems** -- The paper's latency-sensitivity analysis (sub-10 ms bifurcation threshold), torque rate-limit invariance sweep (200-800 Nm/s), detailed sensor noise modeling (3 levels x 3 delay conditions), actuator armature parameters, and hardware degradation projection (encoder quantization, backlash, stiction, tire compliance) engage mechatronic systems engineering. The Path to Hardware section with staged commissioning plan and latency budget breakdown is fundamentally a mechatronics contribution.

3. **Simulation-Based Empirical Robotics** -- The paper's methodology is simulation-only and treats the simulator as an experimental apparatus. It specifies MuJoCo solver parameters (Newton, 100 iterations, pyramidal cone, semi-implicit Euler), contact model stiffness (solref), and acknowledges that the 0.73 mm claim is conditional on stiff ground contact. The paper itself is partly about simulation methodology -- how to design factorial ablation experiments, report statistical power at N=5, and distinguish simulator artifacts from genuine controller properties.

---

## Dimension 3 — Research Paradigm

**Design-Science / Engineering Design Paradigm** (Hevner et al. 2004, Design Science in Information Systems Research).

The paper follows the canonical design-science arc:
1. **Problem identification:** The stiffness-precision vs. push-recovery trade-off in wheeled bipedal balance, rooted in the fundamental tension between integral action (needed for equilibrium bias cancellation) and windup (during disturbances). The problem is shown to be structural -- 4 LQR variants all fail (100% fall rate), and model-free PPO (100M steps) fails at multi-height generalization (85% fall rate).
2. **Artifact design:** ACC, a two-channel torque assembly with proximity-gated anchor, asymmetric envelope follower, gated damping boost, and independent flight/terrain extensions.
3. **Artifact evaluation:** Factorial ablation (additive L0->L3 + subtractive S0->S5), robustness sweep (4x3 noise/delay), parameter sensitivity, compound-disturbance recovery, and comparison against 5 baselines.
4. **Knowledge contribution:** The proximity-gated anchor mechanism as a reusable design pattern -- the paper provides a 5-step porting procedure and argues the critical parameter is $\tau_{\text{release}}$.

This is not hypothesis-testing science (no null hypothesis about a natural phenomenon), nor is it mathematical control theory (no stability proofs, no Lyapunov analysis). It is engineering design: build a controller, demonstrate it works, show which components matter via ablation, and characterize its failure modes.

---

## Dimension 4 — Methodology Type

**Empirical Simulation-Based Evaluation with Factorial Ablation Design.**

The methodology has the following characteristics:

- **Controlled simulation experiments:** All evaluation in MuJoCo at 500 Hz simulation / 100 Hz control, with explicit solver configuration.
- **Factorial ablation:** Additive (L0->L3) and subtractive (S0->S5) paths that decompose the contribution of each ACC component. This is the paper's strongest methodological feature.
- **Statistical testing:** Welch's unpaired t-test with Bonferroni correction across 7 pairwise comparisons; Cohen's d effect sizes; Clopper-Pearson confidence intervals for binary outcomes (flight/terrain survival). Minimum detectable effect characterized (~4.5 N at N=5).
- **Binary search for thresholds:** $F_{\min}$ and $F_{\max}$ determined via binary search over [10 N, 160 N] with 5 N tolerance and 8 iterations per direction.
- **Parameter sensitivity:** Finite-difference gradients revealing $\alpha_r$ dominance (220.4, ~20x the next parameter).
- **Robustness sweep:** 4x3 factorial (noise level x delay) with N=5 per cell.
- **Deterministic seeding:** Explicit seed formulas (seed = 20260727 + trial_id) for reproducibility.
- **Sample sizes:** N=5 (push ablation), N=10 (idle standing, flight, terrain), N=20-60 (LQR baselines), N=20 (PPO baseline).

The methodology is strong for an empirical robotics paper but has a known limitation: all results are simulation-only, so the methodology evaluates ACC-in-MuJoCo, not ACC-on-hardware. The paper acknowledges this transparently.

---

## Dimension 5 — Target Journal Tier

**Q1-Q2 boundary, with Q1 achievable after hardware validation.**

**Rationale:**

The paper's strengths that support Q1 targeting:
- The proximity-gated anchor with asymmetric envelope follower is a genuinely novel control mechanism -- not an incremental tuning of existing methods.
- The factorial ablation design (additive + subtractive, with pre-registered thresholds frozen prior to sweep) is exemplary and exceeds the methodological rigor typical in robotics venues.
- Statistical disclosure is thorough: Welch's t-test, Bonferroni correction, Cohen's d, Clopper-Pearson CIs, minimum detectable effect, and explicit acknowledgment of which comparisons are underpowered.
- The paper engages deeply with both robotics literature (Ascento, DIABLO, Whleaper, SKATER) and control theory literature (anti-windup taxonomy, hybrid systems, override control).
- The 10-item limitations section demonstrates exceptional candor.
- The gate parameter sensitivity analysis with a 5-step porting procedure adds practical value beyond the specific platform.

Factors that constrain to Q2:
- **Simulation-only** -- this is the primary barrier. Q1 robotics journals (IEEE RA-L, IEEE T-RO) strongly prefer hardware validation. The paper's 0.73 mm precision claim is explicitly conditional on stiff ground contact (solref={0.002,1}) and would degrade on real surfaces.
- **Single-platform validation** -- the controller is demonstrated on one robot morphology. Generality claims are untested.
- **Single author, independent researcher** -- while this should not affect scientific evaluation, it may raise procedural questions at some venues about conflict of interest and peer review independence.
- **No theoretical analysis** -- no stability proofs, no Lyapunov analysis, no formal guarantees. The paper is purely empirical.
- **50+ hand-tuned gains** without systematic tuning methodology -- reproducibility concern for independent reimplementation.

**Tier verdict:** The paper, in its current form, is a strong Q2 paper that would be competitive at Q1 venues if accompanied by even minimal hardware validation (e.g., static torque testing + fixed-support standing on a real platform). The simulation-only nature is the decisive factor.

---

## Dimension 6 — Paper Maturity

**Mature -- Ready for External Submission (with minor revisions).**

Evidence of maturity:
- 8 rounds of internal review with 23 weakness sub-claims systematically inventoried and tracked.
- Architecture is stable: the canonical ACC configuration (S1, fixed $k_p=50$) has been identified and all scheduled-gain artifacts removed from the primary architecture description.
- Statistical methodology has been corrected across rounds (paired -> unpaired t-test transition, Bonferroni correction added, Cohen's d with Hedges' g noted).
- Claims have been progressively scoped and caveated: "LQR fails" -> "LQR through PID servo path fails"; abstract now includes the delay bifurcation caveat; 95% CI intervals have been harmonized.
- Figures are complete: architecture diagram, gate activation time series, polar push envelope, ringdown time series, curb straddle rendering.
- Supplementary material is referenced (50+ parameter set, scripts, configs).
- Limitations section is comprehensive (9 items + statistical power discussion).

Remaining concerns before external submission:
- The "ACC as precondition for RL" hypothesis (line 66) remains unevidenced -- zero RL+ACC experiments exist. This is explicitly labeled as a hypothesis awaiting empirical validation, but it sits prominently in the Introduction and could distract reviewers.
- The RL/MPC/DOB baseline gap (consensus finding from Round 8, SC1) has been partially addressed by adding the PPO baseline, but WBC/MPC comparisons remain deferred.
- The 0.5 ms latency margin claim (line 686) remains optimistic for industrial deployment and may draw criticism from mechatronics reviewers.
- The platform comparison table (Table I) has been appropriately caveated as qualitative, but the asymmetry (quantitative for ACC, qualitative for competitors) may still draw criticism.

---

---

# Reviewer Configuration Cards

## EIC -- Editor-in-Chief

**Journal:** IEEE Robotics and Automation Letters (IEEE RA-L)

**Identity Description:**
A senior professor of robotics at a major European university who has served as Editor-in-Chief of IEEE RA-L since 2024. Their own research spans quadrotor control, multi-robot coordination, and control theory for underactuated systems, but their editorial portfolio covers the full breadth of RA-L including legged and wheeled locomotion. They have overseen approximately 2,400 submissions and are known for enforcing RA-L's "must be implementable" standard -- they frequently desk-reject papers where claims depend on idealized simulation conditions without adequate disclosure. They have publicly advocated for raising the bar on simulation-only submissions, arguing that RA-L should require either hardware validation OR a compelling argument for why simulation results are structurally informative.

**Review Focus (3 items):**
1. **Journal fit and contribution significance:** Is the proximity-gated anchor mechanism a sufficient conceptual advance for RA-L, or is it incremental engineering? Does the paper meet RA-L's originality threshold given that it uses only classical control techniques (PD + gated integral)?
2. **Simulation-only acceptability:** Does the paper make a compelling case that its results are structurally informative despite being simulation-only? Are the simulation-to-hardware degradation projections (Sec. VI, Path to Hardware) sufficiently rigorous to justify publication without hardware data?
3. **Architectural clarity and reusability:** Is the ACC architecture described with sufficient precision that an independent group could reimplement it? The 50+ hand-tuned gains without systematic tuning methodology are a potential barrier to replication.

**What they particularly care about:**
- Whether the paper's claims survive the transition from idealized MuJoCo contact (solref={0.002,1}) to realistic hardware. The acknowledgement that 0.73 mm degrades to ~1.2 mm even under softened contact suggests the headline number is brittle.
- The "ACC as precondition for RL" framing: this is a forward-looking claim that RA-L reviewers will scrutinize. Without any RL+ACC experiment, it is speculative and should either be removed or moved to the Discussion as future work.
- Whether the paper's factorial ablation methodology sets a standard that other RA-L submissions should aspire to, or whether the small sample sizes (N=5) and simulation-only nature limit its methodological contribution.

**Possible blind spots:**
- May undervalue the control-theoretic novelty (asymmetric envelope follower, smoothstep gate composition) because they primarily evaluate from a robotics-impact perspective rather than a control-systems perspective.
- May over-weight the hardware validation requirement and under-weight the paper's argument that ACC's failure modes are structurally informative (the gate bifurcation analysis showing partial-engagement is worse than P-only).
- Less attuned to the anti-windup taxonomy nuances and may not fully appreciate why saturation-triggered AW fails vs. ACC's state-gated approach.

---

## Reviewer 1 -- Methodology

**Role:** Peer Reviewer 1 -- Methodology

**Identity Description:**
A mid-carere researcher at a US national laboratory who specializes in **experimental methodology and metrology for robotics systems**. Their PhD was in mechanical engineering with a focus on precision measurement, and they now lead a lab that develops standardized benchmarking protocols for legged robot performance evaluation. They are the lead author of a widely-cited survey on "Reproducibility in Robotics Research" and serve on the ICRA/IROS benchmarking technical committees. They are NOT a control theorist -- they evaluate whether experimental claims are supported by the measurement apparatus and statistical design, not whether the controller is mathematically elegant. Their signature concern is: "You measured 0.73 mm. How do you KNOW it's 0.73 mm, and would another lab get the same number?"

**Review Focus (3 items):**
1. **Measurement validity of the 0.73 mm precision claim:** What is the noise floor of the measurement apparatus? The paper reports SNR 289x (noise floor 0.002 mm CoM X RMS). Is this the MuJoCo solver noise floor, and if so, is 0.73 mm genuinely a controller property or a solver artifact? How would this claim be verified independently -- what instrumentation would a hardware lab need?
2. **Statistical design adequacy at N=5:** The push ablation uses N=5 per cell with Welch's t-test across 7 comparisons. Only L0->L1 survives Bonferroni correction. Are the non-significant comparisons genuinely "no difference" or merely underpowered? The paper reports minimum detectable effect ~4.5 N -- what N would be required to detect the 1.1 N L1->L2 difference, and should the paper acknowledge that its core contribution (anchor over global I) relies on a non-significant push comparison compensated by a qualitative precision difference?
3. **Reproducibility infrastructure audit:** The paper provides seed formulas, commit hash, and supplementary parameters. Can an independent researcher, given only the paper and the GitHub repository, reproduce Table IV exactly? What about the specific MuJoCo build version, OS, and floating-point behavior -- are these specified at a level sufficient for bitwise reproduction?

**What they particularly care about:**
- The distinction between "measured precision" and "simulator precision." In MuJoCo with deterministic seeding and semi-implicit Euler integration at 500 Hz, the solver itself has a noise floor. Is 0.73 mm above or at that floor?
- The gate parameter sensitivity analysis: the hierarchy ($\alpha_r$ dominates, 20x the next) is practically valuable, but was the sensitivity computed via finite differences at a single operating point? How does sensitivity vary across the state space?
- The settle-time issue: EMA$_0=0$ biases $g_{\text{env}}\approx 1$ for ~7.5 s. The paper uses different settle times for different tables (3s for robustness, 5s for standing). Is there a systematic settle-time characterization, and could the 0.73 mm figure include residual EMA transient effects?

**Possible blind spots:**
- May not fully appreciate the control-theoretic significance of the asymmetric envelope follower as distinct from existing anti-windup methods -- they evaluate measurement, not novelty.
- May flag the N=5 sample size as inadequate without acknowledging that in robotics, N=5 per condition with binary search for thresholds is substantially more rigorous than typical practice (which often reports N=1 or N=3).
- May be overly concerned with bitwise reproducibility when conceptual reproducibility (can another lab implement the ACC concept on their own platform?) is arguably more important.

---

## Reviewer 2 -- Domain Expert

**Role:** Peer Reviewer 2 -- Domain

**Identity Description:**
A senior research scientist at a major industrial robotics lab (analogous to Boston Dynamics or the ANYmal team at ETH Zurich) who has **built and deployed torque-controlled legged robots on hardware for over 15 years**. Their expertise is in the mechatronic integration chain: actuator selection, transmission design, torque control bandwidth, sensor fusion, and the thousand practical details that determine whether a controller that works in simulation survives contact with reality. They led the hardware development of a well-known quadruped platform and have published extensively on torque-controlled locomotion, state estimation for legged robots, and sim-to-real transfer. They are deeply skeptical of sub-millimeter precision claims from simulation and will scrutinize every assumption between the MuJoCo model and a real robot.

**Review Focus (3 items):**
1. **Hardware feasibility of the latency budget:** The paper claims end-to-end latency ~9.5 ms with 0.5 ms margin below the 10 ms bifurcation threshold. Is this realistic? A research-grade IMU (e.g., XSens MTi-100) has 2-5 ms group delay; a standard CAN bus at 1 Mbps adds 1-2 ms; the "filter ~3 ms" estimate is vague (what filter? at what cutoff?). The 0.5 ms margin would be consumed by OS scheduling jitter alone on a standard Linux system. Is the paper's "real-time OS with dedicated microcontroller" suggestion a practical solution or does it essentially concede that current hardware cannot run ACC?
2. **Sim-to-reality gap for the anchor mechanism:** The anchor depends on $v_{\text{sag}}$ estimation (IMU + odometry complementary filter) and $g_{\text{env}}$ envelope detection of $|v_{\text{sag}}|$. On hardware, IMU accelerometer bias drift and wheel odometry slip will inject low-frequency noise directly into the envelope follower's input. At the quiet-stance band of 0.25-0.50 m/s, can the anchor distinguish genuine quiet stance from sensor bias? The paper's noise robustness sweep (Table VI) uses additive Gaussian noise on angular velocity and acceleration -- does this adequately model the structured, frequency-dependent noise of real IMUs?
3. **Torque control bandwidth and contact stability:** ACC outputs raw torque commands at 100 Hz with 400 Nm/s rate limit. On a real robot with series-elastic or proprioceptive actuators, the torque control bandwidth is typically 50-200 Hz with nontrivial phase lag at higher frequencies. The stiff MuJoCo contact (solref={0.002,1}) assumes rigid ground; on real surfaces with finite compliance, the coupled actuator-contact dynamics may introduce oscillations not captured in the simulation. Has the paper considered the combined effect of actuator bandwidth limitation + contact compliance on the anchor mechanism?

**What they particularly care about:**
- The staged commissioning plan (static torque -> fixed-support -> free standing -> push -> flight -> terrain) -- this is exactly the right approach, and its inclusion signals hardware awareness. But the paper should acknowledge that each stage may reveal controller limitations not visible in simulation.
- The encoder quantization analysis: the paper projects 0.5 mm degradation from encoder quantization alone. At a 0.06 m wheel radius, 0.5 mm sagittal displacement corresponds to ~0.008 rad wheel rotation, or ~0.48 degrees. A standard 12-bit magnetic encoder gives 0.088 degree resolution -- sufficient. But the paper should show this calculation explicitly.
- The 400 Nm/s rate limit: this is a software limit in the simulation. On hardware, rate limiting emerges from motor inductance, bus voltage, and current control bandwidth -- it cannot be arbitrarily set. Does the 400 Nm/s limit correspond to a real actuator's capability, or is it an arbitrary choice?

**Possible blind spots:**
- May undervalue the control-theoretic contribution because they evaluate everything through the lens of "will this work on hardware?" -- a paper can make a valid scientific contribution without hardware validation.
- May not fully appreciate the factorial ablation design because their evaluation standard is hardware performance, not experimental methodology.
- May over-emphasize the need for actuator-specific characterization when the paper's contribution is the controller architecture, not the actuator model.

---

## Reviewer 3 -- Cross-Disciplinary / Perspective

**Role:** Peer Reviewer 3 -- Cross-Disciplinary Perspective

**Identity Description:**
A researcher in **sensorimotor neuroscience and human postural control** at a leading cognitive science department, with a secondary appointment in bioengineering. They study how the human central nervous system solves balance control -- specifically, the evidence that human standing balance uses intermittent (not continuous) control, with "sensory dead zones" and gated muscle activation triggered by threshold-crossing events. Their key finding (published in *Journal of Physiology* and *PLoS Computational Biology*) is that humans do not continuously integrate sensory error -- instead, the CNS applies an intermittent, event-driven control policy where corrective torque is triggered only when state variables exceed threshold boundaries. They have collaborated with roboticists to implement biologically-inspired intermittent controllers on humanoid robots.

This reviewer reads the ACC paper and immediately sees a striking parallel: ACC's proximity gate ($g_{\text{prox}}$), envelope gate ($g_{\text{env}}$), and pitch stability gate ($g_\theta$) are effectively implementing an *intermittent control policy* with smoothstep transitions -- exactly what the human CNS appears to do. They will evaluate whether the authors recognize this connection and whether ACC's gating strategy aligns with or diverges from biological evidence.

**Review Focus (3 items):**
1. **Connection to intermittent control theory:** The paper frames ACC in terms of anti-windup and selector-based process control, but the gated architecture is fundamentally an intermittent feedback controller: the anchor integral engages only within a state-space region, and the asymmetric envelope follower implements an event-detection mechanism. Does the paper engage with the substantial literature on intermittent control in human postural balance (e.g., Bottaro et al. 2008, Gawthrop et al. 2011, Asai et al. 2009)? This literature has independently arrived at the same insight -- that conditional activation resolves the precision-stability trade-off -- through a completely different path (neuroscience experiments rather than engineering design).
2. **Biological plausibility of the asymmetric EMA:** The human vestibular system has frequency-dependent dynamics that act as a bandpass filter, attenuating both DC (constant tilt) and high-frequency vibrations. The otolith organs have adaptive time constants that provide something functionally similar to ACC's fast-attack/slow-release envelope -- they respond quickly to onset of acceleration but adapt to sustained stimuli. Is ACC's asymmetry ($\tau_{\text{attack}}\approx 23$ ms, $\tau_{\text{release}}\approx 1.5$ s) biologically motivated or purely engineering-driven? The 65:1 ratio between release and attack time constants is striking.
3. **Gate transition dynamics and biological "blending":** ACC uses smoothstep ($C^1$) transitions rather than hard switches, which the paper correctly notes avoids discrete jumps. But in biological motor control, there is evidence for *hysteresis* in threshold crossing -- once a corrective action is triggered, it persists beyond the threshold boundary to prevent chattering. ACC's flight gate uses hysteresis (2-step engage, 5-step release), but the anchor gates do not. Would adding hysteresis to $g_{\text{prox}}$ or $g_{\text{env}}$ prevent the parametric resonance observed in S5 (pitch gate removal)?

**What they particularly care about:**
- Whether ACC provides a *functional model* for human balance control. The paper does not make this claim, but if the parallel is deliberate, it should be explored. If unintentional, the authors should be made aware that their engineering solution has independently converged on a principle used by evolution.
- The temporal dynamics: human postural responses have ~100 ms neural delay, substantially longer than ACC's 10 ms sensitivity threshold. Yet humans achieve sub-degree sway. This suggests that biological intermittent control achieves precision through mechanisms beyond fast feedback -- likely predictive/internal models. Does ACC have any predictive component, or is it purely reactive?
- The "quiet stance" definition: ACC defines quiet stance as $|v_{\text{sag}}|$ envelope < 0.25-0.50 m/s. Human quiet stance is characterized by ~1 cm CoP sway at ~0.1-0.3 Hz. Is ACC's 0.25 m/s threshold equivalent to "quiet" in a biological sense, or is it essentially "not actively being pushed"?

**Possible blind spots:**
- May over-interpret the biological parallel -- ACC was designed as an engineering solution, not a biological model, and the convergence may be coincidental rather than principled.
- May not fully appreciate the practical engineering constraints (torque rate limits, actuator bandwidth, sensor noise) that shaped ACC's design, focusing instead on conceptual parallels.
- The human balance literature spans hundreds of papers with competing theories (continuous vs. intermittent control, PID vs. optimal control models). Expecting deep engagement with this literature from a robotics paper may be unreasonable.

---

## Devil's Advocate

**Role:** Devil's Advocate -- Core Argument Challenger

**Identity Description:**
A senior professor of **control systems engineering** at a major Asian university, known in the field for their combative conference Q&A style and for publishing influential critiques of over-claimed results in robotics. They have spent 25 years working on nonlinear control, adaptive control, and most recently on formal verification of robotic controllers. Their intellectual trademark is demanding that authors answer: "What does your controller guarantee, and under what conditions does that guarantee fail?" They are particularly allergic to empirical-only papers that claim architectural insights without formal analysis, and to papers that compare against weak baselines to appear strong. They have previously published a widely-read commentary arguing that the robotics community's tolerance for simulation-only empirical results without stability analysis has produced a literature of non-reproducible claims.

**Review Focus (3 items):**
1. **No stability guarantees -- what does ACC actually ensure?** The paper demonstrates that ACC works for the specific robot, specific gains, specific MuJoCo configuration. But there is no Lyapunov analysis, no region-of-attraction estimate, no formal guarantee about when the anchor will engage vs. disengage. The gate activation sequence (Fig. 4) shows a single push-recovery cycle -- what about the boundary cases? Are there initial conditions from which ACC will never recover? The parametric resonance in S5 suggests fragile dynamics near the gate boundary. What is the basin of attraction of the quiet-stance equilibrium under ACC, and how does it compare to P-only?
2. **The baseline comparison structure overstates ACC's advantage:** The paper compares ACC against (a) 4 LQR variants that ALL pass through a PID servo layer (except DT LQR), and (b) an untuned PPO baseline. The PID servo path is a known bottleneck -- the paper itself demonstrates this (DT LQR restores roll stability). So the LQR baselines are testing "LQR + bad low-level controller" not "LQR." A fair comparison would be LQR (or MPC) with direct torque output, optimized for the same plant ACC controls. Furthermore, the PPO baseline at 100M steps on a 42-dim observation space is a weak RL baseline -- modern implementations would use domain randomization, privileged information, and RNN policies. Is the paper comparing ACC against the strongest available alternatives, or against strawmen?
3. **Is the "proximity-gated anchor" genuinely novel, or is it state-gated conditional integration with an envelope detector?** The paper positions ACC as distinct from anti-windup, but the anti-windup literature includes conditional integration schemes that gate the integrator based on control error. ACC gates based on physical state (position, velocity envelope, pitch) rather than control error. Is this a distinction with a difference, or a relabeling? The paper acknowledges this in Appendix A but the argument is brief. A rigorous response would benchmark ACC against a well-designed conditional integration AW scheme (not just the simplest variant), with both tuned for the same plant.

**What they particularly care about:**
- Formal statements: every claim about ACC's behavior should be accompanied by a statement of the conditions under which it holds. "ACC achieves sub-millimeter precision" -- under what conditions? (Answer: clean sensors, zero delay, stiff ground contact, specific MuJoCo solver configuration.) The paper is actually good about this, but the abstract's brevity necessarily omits the qualifiers.
- The architectural generality claim: the paper provides a 5-step porting procedure, but this procedure assumes the new platform's dynamics admit a single dominant ringdown mode. What if the new platform has multiple coupled oscillatory modes? Is the procedure still valid?
- The "ACC as precondition for RL" hypothesis: this is a significant claim about the architecture of intelligent control systems. If true, it implies that RL alone cannot solve coupled pitch-height dynamics -- a strong statement about the limits of model-free RL. Where is the evidence? A proper test would be: RL-only vs. ACC-only vs. ACC+RL residual on the same multi-height task. Without this experiment, the claim is speculation dressed as hypothesis.

**Possible blind spots:**
- May demand formal guarantees that are unrealistic for a 10-DOF nonlinear system with discontinuous contact dynamics -- Lyapunov analysis on this system is extremely difficult.
- May underweight the practical engineering value of a controller that demonstrably works (even without formal guarantees) on a challenging platform.
- May not acknowledge that the robotics field's standards for baseline comparisons have historically been low, and ACC's 5 baselines + 46-config gain sweep are actually above typical practice.

---

---

# Top 3 Recommended Target Journals

## 1. IEEE Robotics and Automation Letters (IEEE RA-L) -- Primary Recommendation

**Tier:** Q1
**Rationale:** IEEE RA-L is the strongest fit for this paper in its current form. The 6-8 page format aligns with the paper's length. RA-L publishes rapid-turnaround letters emphasizing novelty and experimental validation -- the proximity-gated anchor mechanism is a clear, scoped contribution well-suited to the letters format. RA-L has recently published several wheeled bipedal papers (Klemm et al. 2020, Xin et al. 2022) establishing a readership for this topic. The journal's review criteria emphasize "technical correctness and novelty" over "completeness" -- the simulation-only limitation, while significant, is not disqualifying if the simulation methodology is rigorous. The paper's factorial ablation design and statistical disclosure would be viewed favorably by RA-L reviewers who increasingly demand methodological rigor. The primary risk is that RA-L reviewers may request hardware validation or demand the RL+ACC residual experiment referenced in the introduction.

**Mitigation strategy:** Before submission, either (a) remove the "ACC as precondition for RL" hypothesis from the Introduction (moving it to Discussion as speculative future work), OR (b) add a minimal RL+ACC residual experiment (even 1M steps at fixed height) to demonstrate the architecture. Option (a) is lower risk.

## 2. IEEE/ASME Transactions on Mechatronics (TMECH) -- Strong Alternative

**Tier:** Q1
**Rationale:** TMECH is an excellent fit for the mechatronic and control engineering dimensions of this paper. The journal's scope explicitly includes "motion control, vibration control, and system identification" and "actuators, sensors, and integrated electro-mechanical systems." ACC's latency-sensitivity characterization, torque rate-limit invariance, sensor noise modeling, hardware degradation projections, and staged commissioning plan all engage core TMECH concerns. The paper's practical, engineering-design orientation aligns with TMECH's emphasis on "solutions that work" over theoretical novelty. TMECH is generally more accepting of simulation-heavy work than T-RO, provided the mechatronic modeling is detailed. The paper's comprehensive parameter table, contact model specification, and actuator armature parameters would be viewed as good mechatronics practice. The journal also has a strong Asia-Pacific readership, where much wheeled bipedal robotics research is concentrated.

**Mitigation strategy:** The paper should strengthen the mechatronic analysis: (a) add a block diagram of the complete signal chain from IMU through state estimation to torque command, with explicit latency annotations per stage; (b) include the encoder quantization calculation explicitly; (c) characterize the torque control bandwidth assumption and justify the 400 Nm/s rate limit against a specific actuator example.

## 3. Robotica (Cambridge University Press) -- Safe Backup

**Tier:** Q2
**Rationale:** Robotica is the pragmatic backup option. It publishes a broad range of robotics papers with less stringent novelty requirements than IEEE T-RO or RA-L. The journal has a long history of accepting simulation-based controller design papers, particularly when the experimental methodology is thorough. The paper's factorial ablation design would be above the typical methodological standard for Robotica. The journal's review process is generally constructive rather than adversarial, and there is less pressure to demonstrate hardware validation. The primary trade-off is impact: Robotica has a lower impact factor (~2.0) and narrower readership than the IEEE venues. However, for an independent researcher's first submission, Robotica offers a higher acceptance probability while still providing rigorous peer review.

**Mitigation strategy:** If submitting to Robotica, the paper should be reformatted to the journal's longer format (typically 12-15 pages), which would allow expanded treatment of the related work (particularly the intermittent control and anti-windup literature connections), more detailed parameter sensitivity analysis, and additional discussion of failure modes. The longer format would also permit inclusion of the full 50+ parameter set in an appendix rather than as supplementary material.

---

**Bottom-line assessment:** Submit to IEEE RA-L first. If desk-rejected or rejected after review, address reviewer concerns (particularly around hardware validation) and submit to TMECH. If further revision is needed, Robotica is the safe landing venue. The paper is strong enough that it should eventually find a home at one of these three venues.