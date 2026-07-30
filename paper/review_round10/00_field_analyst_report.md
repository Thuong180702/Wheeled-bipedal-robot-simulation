Now I have a complete understanding of the paper. Below is the full field analysis and reviewer configuration.

---

# Field Analysis and Reviewer Configuration Cards

## SIX-DIMENSION FIELD ANALYSIS

### 1. Primary Discipline

**Robotics -- Legged/Wheeled Bipedal Locomotion and Balancing Control**

The paper's central concern is the stabilization and disturbance rejection of a 10-DOF articulated wheeled biped robot. It engages with core problems of this subfield: the stiffness-vs-recovery trade-off, integral windup during pushes, and the inadequacy of classical LQR on articulated morphologies. The platform morphology (two legs, five actuated joints per leg, two driven wheels) places it squarely in the wheeled-legged robot lineage stretching from Ascento to SKATER to Whleaper.

### 2. Secondary Disciplines (max 3)

1. **Control Systems Engineering (Classical/Nonlinear)** -- PID tuning, anti-windup architectures, conditional integration, envelope detection (attack/release), hybrid/switched systems, split-range blending, gain scheduling. The paper's intellectual ancestry is process control (Shinskey, Astrom & Hagglund) applied to a robotics domain. The asymmetric EMA envelope follower is a direct descendant of peak-hold circuits and envelope detectors from signal processing/process control.

2. **Mechatronics / Embedded Systems Design** -- The paper is deeply concerned with torque rate limits (400 Nm/s), encoder quantization (contributing 0.5mm+ error), sensor noise models (IMU-grade, 3 levels), latency budgets (9.5ms end-to-end, 0.5ms margin below 10ms bifurcation threshold), bus jitter, RTOS considerations, and staged hardware commissioning. The hardware-path section is unusually detailed for a simulation paper.

3. **Reinforcement Learning (as comparator/motivation)** -- The PPO baseline (100M steps, BalanceEnv) is used to motivate the need for structured priors. The paper explicitly positions ACC as "hypothesized precondition for residual RL" -- a structured prior that a residual policy can augment. This frames the work within the hybrid model-based+learning paradigm (Whleaper's LQR+PPO, residual RL architectures).

### 3. Research Paradigm

**Empirical-quantitative, with physical-intuition-driven design.**

The paper does not derive formal stability guarantees (no Lyapunov analysis, no passivity proofs, no contraction theory). Instead, it follows a "trace-failure-mode-to-root-cause, then add-a-gate" methodology. The five gates were each added after observing a specific measured failure:
- Bistable envelope → asymmetric EMA
- Global-I windup → proximity gate
- Parametric pumping → pitch safety interlock
- Hard-leash oscillation → position clip
- Self-referential stiffness → fixed kp (rejecting gain scheduling)

Validation is exclusively through simulation experiments with statistical rigor: Welch's t-tests, Cohen's d, Bonferroni-corrected alpha, Clopper-Pearson confidence intervals for binary outcomes, and minimum-detectable-effect calculations. This is hypothesis-driven empirical engineering, not mathematical control theory.

### 4. Methodology Type

**Controller design + factorial ablation + comparative benchmarking via simulation.**

Methodological structure:
- **Design phase:** Physical failure-mode analysis → gate mechanism design → parameter tuning (frozen before ablation)
- **Additive ablation (L0→L3):** Build up from P-only baseline through progressive component addition, with statistical testing of each increment
- **Subtractive ablation (S0→S5):** Tear down from full ACC, isolating individual gate contributions
- **Comparative baselines:** 4 LQR variants (TWIP, +AW, direct-torque, coupled 6-state) plus PPO (100M steps)
- **Sensitivity analysis:** Gate parameter gradients (finite-difference), torque rate-limit sweep (200-800 Nm/s), sensor noise + actuator delay factorial grid (4x3)
- **Extension validation:** Flight recovery (4 conditions, N=10) and terrain adaptation (4 conditions, N=10) as independent extensions
- **Compound scenarios:** Push-during-squat (84% degradation), sequential forward-then-backward push (5/5 survival)

The methodology is thorough for simulation-only work but lacks: (a) formal stability analysis, (b) cross-morphology or cross-simulator validation, (c) hardware validation.

### 5. Target Journal Tier (Q1-Q4, with rationale)

**Assessment: Q2 with Q1 potential after hardware validation.**

| Tier | Venue | Rationale |
|------|-------|-----------|
| **Q1** (conditional) | IEEE RA-L, IEEE T-RO | RA-L achievable IF hardware validation is added and N≥5 is upgraded to N≥20. T-RO would require formal stability analysis + hardware + cross-platform generality. Currently simulation-only precludes Q1. |
| **Q2** (current fit) | IEEE/ASME TMECH, ICRA, IROS | TMECH is the strongest fit: values mechatronic system design, tolerant of simulation-only when mechatronic characterization is thorough (which this paper does exceptionally). ICRA/IROS are good conference fits; the 6-page limit matches current length. |
| **Q3** (fallback) | IEEE Access, Robotics and Autonomous Systems, J. Bionic Engineering | Would accept simulation-only with thorough evaluation. Lower bar for novelty, higher tolerance for single-platform validation. |

**Rationale for Q2 (not Q1):** The paper's explicit limitations block Q1 placement:
1. Simulation-only -- the abstract and conclusion both state "hardware validation is future work"
2. N=5 for push ablation -- the author honestly flags this as a limitation and calculates minimum detectable effect
3. Coupled 6-state LQR not re-derived for DT plant -- an honest admission but also a missing analysis
4. Single morphology, single simulator (MuJoCo)
5. Independent researcher without institutional affiliation -- reviewers will scrutinize claims more heavily due to absence of institutional peer review before submission
6. No formal stability guarantees

**Why not Q3-Q4:** The paper is too well-executed for lower tiers. The statistical rigor (Welch's t-test, Bonferroni correction, Cohen's d, Clopper-Pearson CI), the factorial ablation design (additive + subtractive, 9 configurations), the thorough baseline comparison (4 LQR variants + PPO), the honest limitations section, and the hardware-path analysis all exceed typical Q3-Q4 expectations.

### 6. Paper Maturity

**Late-stage draft, submission-ready to mid-tier venues.**

Strengths indicating maturity:
- Clear problem statement and contribution articulation
- Well-structured experimental design with explicit mapping of scenarios to components tested
- Honest, detailed limitations section (8 enumerated points)
- Proper statistical treatment (though underpowered at N=5)
- Hardware-path discussion showing awareness of sim-to-real gap
- Code and parameters publicly available with commit hash
- Reproducible seed conventions documented

Weaknesses capping maturity:
- Simulation-only -- this is the single largest maturity cap
- N=5 for push ablation limits statistical power; author acknowledges this
- No formal control-theoretic analysis (no Lyapunov, no passivity, no contraction)
- Some overclaiming in the abstract ("ACC delivers omnidirectional push recovery" without the "in simulation" qualifier)
- Non-native English phrasing in places (e.g., "the robot cannot survive minimal push", "ACC is fundamentally latency-sensitive" would be better as "ACC exhibits latency-sensitive behavior")
- L2.5 (L2 + proximity gate only) to decompose the additive L2→L3 confound is explicitly deferred -- a gap in the ablation
- The "academic lineage" is thin for the hybrid-switched-systems framing; Liberzon is cited but not deeply engaged
- Platform comparison table (Table I) is qualitative/self-reported rather than standardized benchmarks

---

## REVIEWER CONFIGURATION CARDS

---

### EIC (Editor-in-Chief)

**Journal:** IEEE/ASME Transactions on Mechatronics (TMECH)

**Identity:** Senior Editor for Control Systems and Robotics, 25+ years in mechatronic systems integration. Formerly chaired the IEEE RAS Technical Committee on Mechanisms and Design. Known for favoring papers where mechanical design considerations and control design co-evolve rather than papers that treat one as fixed and optimize the other.

**Areas of expertise:**
- Mechatronic system co-design (actuator selection, transmission design, sensor placement interacting with control architecture)
- Torque-controlled legged systems and series-elastic actuation
- Hardware validation methodology and staged commissioning

**What the EIC focuses on:**
1. **Mechatronic coherence:** Does the control architecture respect the physical plant? The 400 Nm/s torque rate limit, the encoder quantization contributing 0.5mm+ to the precision claim, the armature inertia specifications -- are these coherent with the controller design, or is the controller designed for an idealized plant?
2. **Hardware path credibility:** The paper claims a 0.5ms latency margin below the 10ms bifurcation threshold and acknowledges this is "insufficient for industrial deployment." As EIC, this is simultaneously honest (good) and concerning -- a paper published in TMECH should present a controller that can actually run on hardware, not just one that would work if latency were 0.5ms tighter. Is the staged commissioning plan realistic?
3. **Claim-to-evidence proportionality:** "0.73mm CoM RMS" is the headline precision number. The paper honestly notes this degrades to ~1.2mm with softer ground contact (solref[0]=0.02) and that tire compliance adds 0.1-0.5mm on real surfaces. The EIC will check: does the abstract foreground the idealized number and bury the degradation, or are claims properly scoped?
4. **Comparison fairness:** The platform comparison table (Table I) reports metrics for competitors as "---" (not reported). This creates an asymmetric comparison where ACC appears uniquely capable. The EIC will ask: are these metrics measured under comparable conditions, or is this a "we measured things others didn't report" table masquerading as a superiority table?

**Bottom-line question:** "Does this paper advance the state of mechatronic system design for wheeled bipeds, or does it present a clever controller that works in an idealized simulator but cannot credibly transition to hardware within the next iteration?"

---

### Reviewer 1 -- Methodology (Statistical/Experimental Design Specialist)

**Identity:** Professor of Robotics at a European technical university. PhD in control systems with a postdoc in applied statistics. Known in the community for developing rigorous benchmarking protocols and calling out p-hacking, underpowered studies, and improper statistical tests in robotics papers. Has published methodological critiques in IEEE RAM and Science Robotics. Serves on the ICRA/IROS Best Paper Award committee specifically to audit statistical claims.

**Specific methodological expertise:**
- Experimental design for robotics benchmarking (factorial designs, within-subjects vs between-subjects, blocking, randomization)
- Statistical power analysis and minimum detectable effect calculation
- Multiple comparison correction (Bonferroni, Holm, Benjamini-Hochberg)
- Non-parametric alternatives for small samples
- Reproducibility infrastructure (seeds, containerization, code release)

**What Reviewer 1 particularly scrutinizes:**

1. **N=5 with parametric tests:** The push ablation uses N=5 with Welch's unpaired t-test. Welch's t-test assumes approximate normality of the sampling distribution, which at N=5 cannot be verified. The reviewer will ask: why not a non-parametric test (Mann-Whitney U) for N=5? Why not bootstrap confidence intervals? The paper calculates minimum detectable effect (~4.5N) and acknowledges sub-2N differences are non-significant -- this is honest but doesn't fix the fundamental power problem.

2. **Multiple comparison correction validity:** Bonferroni with 7 pairwise comparisons gives alpha=0.007. But the paper reports both additive (L0-L3, 3 comparisons) and subtractive (S0-S5, 5 comparisons) series. Are the 7 comparisons pre-registered, or selected post hoc? If the subtractive series was a separate family, it should have its own correction. This reviewer understands that Bonferroni is conservative and will check whether the choice of correction family inflates or deflates significance.

3. **Binary outcome CIs for flight/terrain:** Clopper-Pearson 95%CI [0.69, 1.00] for 10/10 survival is correct. But flight and terrain use "fixed initial conditions (N=10 each, identical outcomes)" -- meaning the 10 trials are not independent perturbations around a distribution but identical replays. A Clopper-Pearson CI on identical replays inflates apparent precision. The reviewer will flag this as a misapplication -- the CI implies sampling variability that doesn't exist when initial conditions are fixed.

4. **Between-trial CV reporting:** CV=10% for idle RMS at N=10 is reasonable, but CV=30% for P-only at N=1 cannot be a CV (it requires at least 2 trials). The reviewer will catch this.

5. **The "46-configuration sweep" claim:** The paper states "46-configuration sweep verified no retuning alters outcome" in a footnote. What was swept? What was the criterion? What was the stopping rule? Sweep-then-claim-no-difference is vulnerable to confirmation bias unless the sweep protocol is pre-specified. The reviewer will demand the sweep script be included in supplementary material and the sweep dimensions be fully enumerated.

6. **Seed conventions:** "RNG seeds are deterministic: idle standing uses seed = 20260727 + trial_id; push ablation uses seed = 20260728 x 100 + rep; LQR baselines cycle seeds {0, 42, 123} across episodes." The reviewer will ask: why different seed conventions for different experiments? The push ablation seed formula (date x 100 + rep) generates seeds 2026072800, 2026072801, ... -- is there any risk of correlated RNG sequences? Are these seeds documented in the code, or only in the paper?

**Likely recommendation:** Weak accept with mandatory revision: increase N to 20 minimum for push ablation, report non-parametric tests alongside parametric, and properly randomize flight/terrain trials. Without these, the statistical basis for core claims (push thresholds at N=5) is too thin.

---

### Reviewer 2 -- Domain (Wheeled Biped Control / Legged Robotics Specialist)

**Identity:** Research Scientist at a major robotics lab (ETH Zurich RSL or equivalent). Has personally built and tested wheeled biped controllers on hardware, including LQR, WBC, and MPC. Published on Ascento, ANYmal, or similar platforms. Knows the pain of PID servo lag, the reality of encoder backlash, and exactly how MuJoCo's contact model differs from hardware. Has strong opinions about what constitutes a "fair" baseline comparison.

**Research interests:**
- Whole-body control (WBC) for floating-base systems with kinematic loops
- Model predictive control (MPC) for wheeled-legged locomotion
- State estimation for balancing (IMU+odometry fusion, contact detection)
- Hardware-in-the-loop validation methodology
- The gap between MuJoCo simulation and hardware reality for wheeled systems

**Literature knowledge:**
- Deep familiarity with Ascento lineage (Klemm 2019, Klemm 2020, Chamorro 2024)
- Knows Handle/Swiss-Mile/BITeno/DIABLO/Whleaper/SKATER papers intimately
- Understands the distinction between Zamani's rigid two-wheeler (structurally simpler) and articulated-legged platforms
- Has read Xin et al. (2022) centroidal dynamics paper and understands why pitch-height coupling is hard
- Can evaluate whether the PPO baseline was given a fair chance or was set up to fail

**What Reviewer 2 particularly scrutinizes:**

1. **LQR baseline fairness:** The four LQR baselines all fail within 0.76s (100% fall rate). The reviewer knows that "LQR fails" can mean many things: poorly tuned gains, wrong plant model, wrong state representation, unrealistic servo model, etc. The paper claims a 46-configuration sweep verified structural failure, but the reviewer will ask: was the LQR solved on the exact MuJoCo linearization, or on a simplified model? What was the linearization height? Were the Q/R matrices optimized (Bryson's rule is a starting point, not the optimum)? Was the DT LQR properly discretized including the 100Hz ZOH? The paper admits "coupled 6-state LQR gain matrix was not re-derived for the DT plant" -- the reviewer will seize on this as evidence that the LQR comparison is incomplete.

2. **The PID servo path as bottleneck claim:** The paper claims "a primary LQR bottleneck is the PID servo action path." The direct-torque LQR (which removes the servo path) also fails at 0.76s. So the paper argues: servo-removal alone doesn't fix LQR (confirmed), but ACC's direct-torque architecture IS essential. This reviewer will ask: what is ACC's advantage over direct-torque LQR? The direct-torque LQR achieves 2.23deg roll RMS and 9.27Nm torque RMS (much better than TWIP LQR) but still falls. ACC achieves 0.010deg roll RMS and 3.28Nm torque RMS. What mechanism accounts for this 200x roll improvement? The paper says "the structured prior (proximity-gated anchor, asymmetric envelope, gated damping), not servo-path elimination alone." But the reviewer will push: DT LQR already eliminates the servo path -- what specifically in ACC's torque assembly accounts for the remaining gap?

3. **WBC/MPC baseline absence:** The paper compares against LQR and PPO but not against WBC or MPC. The platform comparison table mentions SKATER uses "Convex MPC" and Ascento uses "LQR + WBC." The reviewer will note that WBC with contact constraints and hierarchical optimization is the state-of-the-practice for legged robots, and its absence as a baseline weakens the claim that "no single classical controller integrates all three" (precision, push recovery, terrain). A task-space WBC with proper contact modeling might achieve comparable results.

4. **Height-dependent dynamics:** ACC uses a posture controller scheduled across 23 heights. The core anchor (wheel-torque channel) uses fixed gains (kp=50, kd=3, etc.) across all heights. The reviewer, who has built height-varying controllers for wheeled bipeds, knows that pitch dynamics change with CoM height (pendulum frequency scales with sqrt(g/h)). Does the fixed-gain anchor work equally well at 0.40m and 0.70m? The paper reports compound-disturbance results (push during squat: 84% degradation) suggesting it does not. The reviewer will ask for fixed-height push sweeps at each height to characterize the height-dependence of ACC's push envelope.

5. **Terrain extension novelty:** The per-leg terrain adaptation (Eq. 32-34) is essentially a 3-line height split based on contact-point estimates. The reviewer will note that proprioceptive terrain adaptation via contact-point estimation has been standard in legged robotics since Bellicoso et al. (2016) and is not novel. The paper's contribution here is extending this to wheeled bipeds, which is incremental. The reviewer will want the terrain section's novelty claim to be properly scoped.

6. **Whleaper comparison specificity:** Whleaper (Zhu et al., 2024) uses LQR+PPO residual architecture and shares the 10-DOF morphology. This is the closest prior work. The reviewer will ask: what is ACC's push recovery performance compared to Whleaper's? The table says "---" (not reported). Given Whleaper is the closest comparator, the reviewer may suggest implementing ACC on Whleaper's published model for a fair comparison, or at minimum discussing the architectural differences in detail (not just one sentence).

**Likely recommendation:** Weak reject or major revision. The LQR comparison is suggestive but not dispositive (missing DT-optimized 6-state LQR). The WBC/MPC baseline absence is a significant gap for a paper claiming to supersede classical methods. The terrain adaptation is incremental. The core anchor mechanism is genuinely novel and well-validated in isolation, but the paper overclaims by positioning it as a comprehensive replacement for LQR/WBC/MPC/RL when it has only been compared against LQR and RL.

---

### Reviewer 3 -- Cross-Disciplinary (Nonlinear Dynamics / Hybrid Systems Theorist)

**Identity:** Associate Professor in Applied Mathematics or Mechanical Engineering with a research program in nonlinear dynamics, hybrid/switched systems, and bifurcation analysis. Has published in Physica D, Nonlinear Dynamics, and IEEE TAC on stability of switched/hybrid dynamical systems. Understands that smoothstep gates create a continuous vector field but that the underlying dynamics are piecewise-defined with distinct regimes. Approaches robotics papers not from "does it work" but from "what is the dynamical system actually doing, and can we prove anything about its stability?"

**Research interests:**
- Hybrid/switched system stability (multiple Lyapunov functions, dwell-time conditions, average dwell-time)
- Bifurcation analysis and parametric resonance in controlled mechanical systems
- Non-smooth dynamics (impact, friction, contact)
- Envelope/peak-hold dynamics as nonlinear filters
- Passivity-based control and energy shaping

**Literature knowledge:**
- Liberzon (2003) "Switching in Systems and Control" -- deeply familiar with the entire framework
- Knows the smoothstep literature and its limitations (C1 but not C2; switching surface geometry)
- Understands why anti-windup is hard -- not just an engineering annoyance but a fundamental nonlinear control problem (saturation nonlinearity + integral state = finite escape time)
- Has analyzed limit cycles in balancing robots and understands why P-only controllers produce persistent oscillations (it's a Hopf bifurcation in the closed-loop dynamics)

**What Reviewer 3 particularly scrutinizes:**

1. **Where is the stability analysis?** The paper claims "ACC achieves all demonstrated behaviors" but provides zero formal stability analysis. ACC has 5 interacting gates, each of which creates a distinct dynamical regime. The state space is partitioned into at least 2^5 = 32 regions by the gates (though practical interactions reduce this). Even if full Lyapunov analysis is intractable, the reviewer expects: (a) an analysis of the dynamics in each limiting regime (all gates 0, all gates 1, and the transitions between them), (b) dwell-time or average-dwell-time analysis proving that the system does not chatter at switching boundaries, and (c) at minimum, a proof that the equilibrium (robot standing at home position) is locally exponentially stable in the all-gates-1 regime.

2. **The gate composition is C0, not C1:** The paper notes that "composite g_env is C0" because the asymmetric EMA is C0 (piecewise definition) even though smoothstep itself is C1. C0 vector fields can exhibit non-unique solutions, sliding modes, and chattering at the switching surface. The reviewer will ask: have you verified that trajectories do not get stuck on or chatter along the C0 switching surface? The smoothstep on top of the C0 EMA may hide this from visual inspection, but it doesn't eliminate the C0 structure underneath.

3. **The "delay bifurcation" is probably a real bifurcation:** The paper reports that "F_max bifurcates from 96N to 47N" at 10ms delay. The reviewer will note that this is extremely unlikely to be a literal bifurcation (where the qualitative behavior changes at exactly 10ms) -- it's more likely a continuous degradation with a sharp knee. But the reviewer will be intrigued and ask: can you characterize this as a function of delay with finer resolution (e.g., 1ms steps from 0-15ms)? If there IS a genuine bifurcation (e.g., a Hopf bifurcation in the delay-differential equation), this would be a publishable dynamical-systems result in its own right.

4. **Asymmetric EMA dynamics:** The asymmetric EMA is a nonlinear dynamical system itself. Fast attack (23ms), slow release (1.5s). The reviewer will view this as a nonlinear filter with hysteresis-like behavior and ask: can you characterize the input-output map of this filter? Under what input signals (sinusoidal, impulsive, stochastic) does it produce a useful quiet-stance detection? The sensitivity analysis (alpha_r dominates, sensitivity 220.4) is empirical. Can you derive the sensitivity analytically from the filter structure?

5. **Parametric resonance from the pitch gate:** S5 (removing pitch gate) causes "56mm parametric oscillation." The term "parametric resonance" is specific: it means the gated boost modulates the effective damping at a frequency near twice the natural frequency, pumping energy into the system. The reviewer will ask: can you compute the natural frequency of the pitch mode and verify that the gated-boost modulation frequency (when half-open due to pitch oscillation) is approximately 2x the natural frequency? If not, this is forced oscillation, not parametric resonance, and the terminology should be corrected.

6. **Comparison to passivity-based control:** The gated damping boost (g_boost * K_boost * (-v_sag)) is essentially a state-dependent damping injection. Passivity-based control provides a framework for proving that energy-shaping + damping-injection stabilizes mechanical systems. The reviewer will ask: can ACC be reformulated as interconnection-and-damping-assignment passivity-based control (IDA-PBC)? This would provide formal stability guarantees and connect ACC to a well-established theoretical framework.

7. **The ringdown time constant:** The paper reports 9.0s ringdown time but with tau_release=1.5s. Why does ringdown take 6x the release time constant? Is this because the boost is only active when all three gates are open, and the gates gradually close as the robot settles? A dynamical-systems analysis of the ringdown phase (showing how gate states evolve during ringdown) would be illuminating.

**Likely recommendation:** Major revision. The mechanism is physically well-motivated and empirically validated, but a paper claiming a control architecture with 5 interacting continuous gates and associated claims about limit-cycle elimination, ringdown, and bifurcation should include formal analysis. Without it, the paper is an empirical demonstration, not a scientific contribution to control theory. Recommend: add regime-by-regime stability analysis, characterize the delay bifurcation with finer resolution, and correctly identify (or abandon) the "parametric resonance" claim.

---

### Devil's Advocate

**Identity:** A skeptical senior colleague who has seen many "clever controller" papers come and go. Not necessarily an expert in wheeled bipeds, but deeply experienced in evaluating whether a proposed method is genuinely general or merely overtuned to one simulation model. Approaches the paper with the question: "If I implemented this on my own robot, would it work, or are there hidden assumptions that only hold in this specific MuJoCo model?"

**Specific angle for challenging core arguments:**

**1. "The five gates were frozen before ablation" -- but were they really independently chosen?**

The paper claims all thresholds were "frozen prior to the ablation sweep." But the five gates were designed by tracing five failure modes that occurred during development on THIS SPECIFIC robot model. This is circular: you designed gates to fix failures you observed on this model, then validated them on this model. The real test is whether the same gate structure and parameter set generalizes to a different wheeled biped morphology. The paper's own porting guide (Section IV-D) uses the same model's ringdown period to set tau_release -- it doesn't demonstrate that the gate *structure* generalizes, only that the tuning procedure can be repeated.

**2. "Four LQR baselines all fail" -- but this is a strawman comparison.**

The strongest classical baseline is not LQR-through-PID-servo (which the paper itself identifies as a bottleneck), but rather: (a) a properly tuned full-state LQR on the linearized 10-DOF plant solving directly for torques (not through a servo layer), or (b) a task-space WBC with contact constraints solving a QP at each timestep, or (c) a DIV-convex-MPC as used by SKATER. The paper compares against crippled LQR variants (admitted: "a primary LQR bottleneck is the PID servo action path") and then claims LQR/classical methods cannot solve the problem. A fair comparison would be against the best available classical method, not the weakest.

**3. "0.73mm CoM RMS" -- what exactly is being measured, and is it meaningful?**

The paper reports sagittal CoM RMS of 0.73mm. But the simulation uses stiff ground contact (solref={0.002,1}) that the paper admits is "idealized rigid ground." Softening to 0.02 (tire-on-asphalt) increases RMS ~60% to ~1.2mm. On real hardware, encoder quantization alone contributes >0.5mm, backlash 0.02-0.06mm, stiction unknown, and tire compliance 0.1-0.5mm. The "sub-millimeter" claim evaporates once these are accounted for. The abstract headlines 0.73mm -- a number that is conditional on idealized simulation contact parameters. This is misleading to a reader who skips the limitations section.

**4. "Sub-millimeter idle precision without sacrificing omnidirectional push survival" -- but the push thresholds are measured at N=5 with wide confidence intervals.**

The push threshold F_min=83N with N=5 has a 95% CI half-width of approximately 1.24 x std (t_0.025,4 = 2.776). For S1, CI_95 is approximately 83.1 +/- 1.0N (using the reported std of 0.8N). But for S3 (symmetric EMA), CI_95 is 82.0 +/- 2.7N. The overlap between S1 and S3 CIs is substantial. At N=5, the paper CANNOT reliably distinguish push performance between ACC and its ablated variants for sub-5N differences. The claim that ACC "preserves omnidirectional push survival" relative to P-only (F_min=69.7N) is plausible; the claim that the anchor specifically contributes to push survival (L2->L3: +6.8N, p<0.001) is statistically supported. But the implicit claim that ACC achieves push performance comparable to the best configurations while adding precision is only partially supported -- at N=5, many of the comparisons are underpowered.

**5. "ACC is hypothesized as a precondition for RL" -- but this is an untested hypothesis stated as a conclusion.**

The paper concludes that "ACC's proximity-gated anchor provides the essential structured prior needed for stable balance across height commands" -- but the RL community's experience with domain randomization and careful reward design suggests that PPO might succeed with better tuning (longer training, different reward, better curriculum). 100M steps sounds like a lot, but for a 10-DOF system with coupled pitch-height dynamics, it may be insufficient. The hypothesis that ACC is a precondition for RL is stated without a single residual-RL experiment. This is speculation dressed as a conclusion.

**6. The flight recovery is a simple PD-on-wheels-as-reaction-flywheels controller.**

The flight controller (2.0*theta + 0.5*dtheta - 0.02*omega_wheel, clipped to +/-2Nm) is a 3-term PD using wheels as reaction flywheels. This is a standard trick in the wheeled robot community -- Ascento does jumping, Handle does aerial maneuvers. The paper's contribution is the discrete load-based hysteresis engage/release logic (2-step engage, 5-step release), which is a minor refinement. Calling this an "independent extension" that demonstrates the modularity of ACC's architecture is generous -- it's a standalone controller that runs when the balance controller is disengaged. Any architecture with a mode-switching capability could host this.

**7. The compound-disturbance degradation (84%) undermines the "omnidirectional" claim.**

The abstract claims "omnidirectional push recovery" (8 directions). But during height transitions, push survival degrades 84% (94N -> 15N). If the robot cannot handle pushes during height transitions, and height transitions are part of normal operation (the paper demonstrates commanded squats), then "omnidirectional push recovery" is conditional on the robot being at a fixed height. This is a serious practical limitation that the abstract should foreground rather than bury in the analysis section.

**Likely recommendation:** Reject with encouragement to resubmit after: (a) hardware validation or at minimum multi-simulator validation, (b) upgrading N to 20+ for push ablation, (c) adding a WBC or MPC baseline, (d) tempering the LQR-failure claim to acknowledge that a properly-tuned full-state DT LQR on the 10-DOF plant was not tested, and (e) removing or clearly labeling as speculation the "precondition for RL" hypothesis.

---

## TOP 3 TARGET JOURNAL RECOMMENDATION

### 1. IEEE Robotics and Automation Letters (RA-L) -- WITH hardware validation

**Rationale:** RA-L is the best fit for the paper's scope and format. The letters format (6-8 pages) accommodates the current length well. RA-L has a track record of publishing well-executed controller design papers with thorough empirical evaluation. The review process is faster than T-RO (typically 8-12 weeks to first decision). Key caveat: **RA-L in its current simulation-only form is borderline.** The paper would be much stronger with even minimal hardware validation (e.g., static standing + push recovery on a benchtop setup). The Editors typically expect "letters" to report completed work, not work-in-progress with deferred hardware validation.

**Acceptance probability:** 40% (simulation-only) / 65% (with hardware validation)

### 2. IEEE/ASME Transactions on Mechatronics (TMECH) -- BEST FIT for current form

**Rationale:** TMECH values the mechatronic systems perspective that this paper excels at: co-design of controller architecture with realistic actuator constraints (torque rate limits, encoder quantization, latency budgets, bus jitter). TMech reviewers are more tolerant of simulation-only work when the mechatronic modeling is thorough (which it is here -- the latency budget analysis, hardware degradation projection, and staged commissioning plan are unusually detailed). The paper's integration of mechanical design constraints (armature inertia, contact stiffness sensitivity) with control design (gate thresholds accounting for encoder noise floor) is exactly what TMech values.

**Acceptance probability:** 55%

### 3. IEEE International Conference on Robotics and Automation (ICRA) -- Good conference option

**Rationale:** ICRA's 6-page format matches the current paper length. The paper has strong video potential (gate activation visualization, push recovery polar envelope, curb straddle, flight recovery). Conference reviewers are slightly more tolerant of simulation-only work than journal reviewers, especially when accompanied by compelling video. However, ICRA is highly competitive (acceptance rate typically 35-45%) and the independent-researcher affiliation will raise eyebrows. The paper's honest limitations section would play well with ICRA reviewers who value reproducibility.

**Acceptance probability:** 35%

---

## SUMMARY TABLE

| Dimension | Assessment |
|-----------|-----------|
| **Primary Discipline** | Robotics -- Legged/Wheeled Bipedal Balance and Disturbance Recovery |
| **Secondary Disciplines** | Control Systems Engineering (classical/nonlinear PID, anti-windup, hybrid systems), Mechatronics/Embedded Systems (latency budgets, sensor noise, staged commissioning), Reinforcement Learning (as comparator/motivation) |
| **Research Paradigm** | Empirical-quantitative, physical-intuition-driven design (failure-mode tracing + gate addition), simulation-only validation |
| **Methodology Type** | Controller design + factorial ablation (additive + subtractive, 9 configurations) + comparative benchmarking (4 LQR + PPO) + sensitivity analysis (noise/delay grid, gate gradients, rate-limit sweep) |
| **Target Journal Tier** | Q2 currently; Q1 potential after hardware validation. Best current fit: TMECH (Q2). RA-L (Q1) conditional on hardware or multi-platform validation. |
| **Paper Maturity** | Late-stage draft, submission-ready for mid-tier venues. Strengths: honest limitations, statistical rigor, thorough ablation, open code. Weaknesses: simulation-only, N=5, no formal stability analysis, some overclaiming in abstract. |