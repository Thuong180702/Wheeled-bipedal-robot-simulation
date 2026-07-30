# Peer Review Report -- R3-Alt: Industrial Process Control Perspective (Round 8 External)

## Reviewer Identity

**Senior Process Control Engineer, 20+ years. ExxonMobil/BASF background. Now robotics research.**

I spent two decades commissioning override controllers, selector blocks, split-range architectures, and anti-windup schemes on refinery distillation columns, exothermic batch reactors, and ethylene crackers before pivoting into robotics research. I have read Shinskey (1979, 1988, 1996) in its original editions, implemented every variant of conditional integration and tracking back-calculation AW on DCS and PLC platforms (Honeywell TDC 3000, Emerson DeltaV, Siemens PCS 7), and debugged more "why did the override hang?" incidents at 3am than I care to count. I now evaluate robotics control papers from the perspective of someone who knows what it takes to keep a controller running 24/7/365 without PhD tuning support.

**My "outsider" status:** I am not a wheeled-biped specialist. I evaluate this paper as an industrial control practitioner who has seen every variant of the techniques ACC claims to extend. My value to this review panel is flagging what the robotics reviewers -- who are domain experts in the platform and problem -- would miss about the control architecture's intellectual lineage, practical deployability, and safety implications.

I have read the Round 8 reviews (EIC, R1 Methodology, R2 Domain, R3 Cross-Disciplinary, DA) and the existing R3 report. My review is complementary: I do not repeat points already well-covered by the existing R3 (e.g., the AW taxonomy detail in Appendix A, the split-range literature suggestion). I focus on what someone who has *operated* -- not just studied -- process control would see.

---

## Overall Assessment

### Recommendation: Minor Revision (convergent with existing Round 8 consensus)

### Confidence Score: 4

### Summary Assessment

This paper describes a controller that demonstrably works in simulation. The empirical results are thorough, the ablation is well-structured, and the intellectual honesty (10 explicit limitations, statistical power caveats, simulation-only qualifier) is exceptional by robotics standards. The proximity-gated anchor with asymmetric envelope follower is a genuinely clever synthesis.

My concerns are from a different angle than the existing R3's. Where the existing R3 flagged that ACC "relabels established process control practice," I want to go deeper: **what specifically is different about ACC from the override controller I would have designed for this problem in 1995, and does the paper help me understand those differences?** The paper has engaged with the process control literature at the citation level (Shinskey, Astrom & Hagglund are now cited), but it has not yet engaged with it at the *conceptual architecture* level. A reader from process control will recognize every pattern in ACC and wonder: what did the robotics-specific synthesis add?

I also bring concerns about practical deployability (the 0.73mm precision claim under industrial conditions, the latency margin, the 50+ parameter count) and safety (stuck-gate failure modes, the absence of any functional safety discussion) that the robotics reviewers, focused on comparative baselines and statistical validity, understandably did not prioritize.

This paper is close to ready. My review provides the industrial practitioner's perspective that rounds out the panel. None of my concerns require new experiments -- they require sharper conceptual framing and acknowledgment of industrial realities.

---

## Strengths

**S1: The asymmetric EMA is a genuinely elegant control design choice.** I have used asymmetric filters for decades in process control -- typically for setpoint tracking (fast response to setpoint changes, slow response to measurement noise). Applying the fast-attack/slow-release pattern to *detecting the boundary between disturbance and recovery* is a creative repurposing. The symmetric-EMA ablation confirming bistable failure is the kind of diagnostic test I would want my junior engineers to run. This is good engineering. *Evidence: Section IV-C, Table IV S3 row.*

**S2: The factorial ablation with additive and subtractive paths is methodologically sound.** The ability to trace each component's marginal contribution in both directions (starting from P-only and building toward ACC, then starting from ACC and removing components) is exactly the right way to validate a multi-component controller. I have used the same approach when commissioning complex override schemes -- start with the base loop, add selectors one at a time, then remove them one at a time from the full configuration. This is industrial best practice, and it is correctly applied here. *Evidence: Table IV.*

**S3: The LQR comparison that identifies the PID servo path as the bottleneck is a valuable negative result.** In process control, we learn early that the actuator path matters as much as the controller structure -- a perfectly tuned PID is useless if the valve stiction is 5%. The paper's demonstration that four LQR variants all fail (100% fall rate) and that the Direct Torque variant improves survival by 35% but still fails is a clean, well-controlled experiment that every control engineer should appreciate. *Evidence: Section V-D, Table VIII.*

**S4: The "path to hardware" discussion is more detailed than most simulation-only robotics papers I review.** The estimation pathways (IMU+odometry for v_sag, joint torque for F_z, FK for z_contact), the latency budget, and the staged commissioning plan demonstrate thinking that goes beyond simulation. This is the kind of pragmatism I value. *Evidence: Section VI.*

**S5: Honest statistical reporting with effect sizes and Bonferroni context.** Stating that only L0 to L1 survives Bonferroni correction, reporting Cohen's d, and explicitly noting the MDE of ~4.5N -- this is rare and commendable. In industrial practice, we often make decisions based on small-N data without this level of statistical self-awareness. *Evidence: Table IV notes, Section VI statistical power discussion.*

---

## Weaknesses

### W1: ACC's "conditionally-gated" architecture is not distinguished from standard industrial override control with sufficient precision. [Major, Confidence: 5]

**What the existing R3 said:** ACC "relabels established process control practice without fully acknowledging the depth of prior art" and suggested adding a paragraph on split-range control.

**What I add:** The existing R3 identified the right problem but suggested the wrong fix. Adding a citation paragraph is necessary but not sufficient. The paper needs to answer a specific question: **If I, as an ExxonMobil control engineer in 1995, had faced this wheeled-biped balancing problem, what would my override controller have looked like, and how specifically does ACC differ?**

Let me sketch that controller. The standard industrial approach would be:

1. **Primary controller (always active):** PD balance with fixed gains (ACC's tau_balance).
2. **Override selector (high-select):** A second controller that takes over when position error exceeds a threshold -- the override controller would have a stronger position term to drive the robot back toward home after a push.
3. **Conditional integrator:** Integral action on position error, with the integrator frozen (not wound up) when the override is active. Standard conditional integration -- freeze when |Delta x| > threshold, resume when |Delta x| < threshold.
4. **Bumpless transfer:** When switching between primary and override controllers, the inactive controller tracks the active controller's output to prevent bumps. Standard in every DCS since the 1980s.
5. **Rate-of-change limiting on the output:** To prevent the torque command from jumping when the selector switches. Also standard.

This is not a straw man. This is exactly what I would have specified in a functional design document in 1995, using off-the-shelf DCS function blocks. No smoothstep, no asymmetric EMA, no envelope follower -- just standard industrial building blocks.

The question the paper should answer: **does this 1995-vintage override controller work, and if not, why not?** My hypothesis, based on the paper's data:

- The hard threshold (|Delta x| > threshold for override activation) creates a switching surface that the post-push ringdown can cross repeatedly, causing chattering between primary and override modes. ACC's smoothstep eliminates this.
- The conditional integrator with a simple dead-zone either over-restricts (frozen too long) or under-restricts (activates during ringdown). ACC's velocity-envelope-gated integral solves this by using a *temporal* indicator of quietness (the asymmetric EMA) rather than a *spatial* indicator (position error).
- The rate-of-change limiter on the DCS output provides smoothing but introduces phase lag that destabilizes the balance loop -- exactly the PID servo lag problem ACC identifies.

If my hypothesis is correct, then ACC's specific innovations relative to industrial override control are: (1) smoothstep blending instead of discrete switching (eliminates chattering on the switching surface), (2) velocity-envelope-gated integral activation instead of position-error-gated activation (distinguishes quiet stance from ringdown in a way that spatial gating cannot), and (3) coupling the integral with a concurrently-gated damping boost (a joint mechanism with no direct industrial analogue).

**This is a stronger and more honest framing than "ACC extends selector-based control from discrete selection to continuous blending."** It acknowledges that continuous blending is already standard (split-range has used it for decades), and instead identifies the *specific problem* that motivated each ACC design choice and the *specific industrial pattern* that would have been the obvious alternative.

**Evidence Anchor:** Lines 88-89, 139. The paper currently claims ACC extends "from discrete selection to continuous blending," which is imprecise -- both discrete selection (override with hysteresis) and continuous blending (split-range, bumpless transfer with tracking) are standard. What ACC adds is the *problem-specific gating condition design*, specifically the velocity envelope as a quietness detector.

**Suggestion:** Replace or supplement the current Related Work paragraph (lines 88-89) with a substantive comparison table or paragraph: "The standard industrial approach to conditional controller activation -- override control with discrete switching and bumpless transfer -- would face three specific failure modes on this platform: (i) chattering across the switching surface during ringdown, (ii) position-error-gated conditional integration that cannot distinguish quiet stance from slow ringdown drift, and (iii) PID-servo-induced phase lag from the rate-of-change limiting needed for bump suppression. ACC addresses each: smoothstep blending (i), velocity-envelope gating (ii), and direct torque output (iii)."

**Severity:** Major (conceptual framing, not empirical). This is a text-level fix.

**Confidence:** 5. I have literally designed and commissioned the override controller described above.

---

### W2: The asymmetric EMA is a standard industrial building block; the paper should source its lineage more precisely. [Minor, Confidence: 5]

**What the existing R3 said:** The existing R3 did not address this specifically. R2 (Domain) noted that "envelope detection for control mode switching" literature is thin.

**What I add:** The asymmetric EMA (fast attack, slow release) is not just a signal-processing concept. It is a standard industrial control building block with specific lineages:

1. **Pneumatic controllers (pre-1970):** Asymmetric reset (integral) action with different up/down rates was implemented mechanically via different orifice sizes for the bellows fill/bleed. This predates digital control.
2. **Anti-surge compressor control (1970s-present):** The recycle valve controller uses an asymmetric response -- fast opening on approach to surge, slow closing after the operating point recovers. The asymmetry prevents cycling.
3. **pH neutralization control (1980s-present):** The highly nonlinear titration curve motivates asymmetric controller gains -- aggressive in one direction (near the equivalence point where gain is enormous), conservative in the other.
4. **Level control in surge tanks (standard practice):** The level controller is tuned to be aggressive on inflow (prevent overflow) and conservative on outflow (allow the tank to absorb disturbances). Asymmetric filter time constants on the level measurement are the standard implementation.

The point is not that the paper fails to cite these applications -- that would be unreasonable. The point is that the paper should acknowledge that *asymmetric temporal filtering for regime-dependent control* is a mature industrial practice with a 50+ year history. The contribution is the *specific application* of this pattern to wheeled-biped quiet-stance detection, not the pattern itself.

**Evidence Anchor:** Section IV-C describes the asymmetric EMA as if its asymmetry is the novel insight. Lines 229-230: "This asymmetry is essential: a symmetric EMA is bistable."

**Suggestion:** Add one sentence: "Asymmetric filtering for regime-dependent control has a long history in industrial practice (e.g., anti-surge compressor control, pH neutralization, surge-tank level control), typically implemented via different up/down filter time constants. ACC's contribution is adapting this pattern to the specific problem of distinguishing post-push ringdown from quiet stance in wheeled-biped balancing, where the fast-attack/slow-release envelope provides a continuous quietness metric."

**Severity:** Minor. Does not affect the empirical contribution.

**Confidence:** 5.

---

### W3: The latency margin is not credible for industrial deployment. [Major, Confidence: 4]

**What the existing R3 said:** Not addressed. The EIC flagged this (SC-W6) as a consensus concern but primarily from the robotics sensor-chain perspective.

**What I add:** The paper reports a latency budget of ~9.5ms (IMU ~1ms + filter ~3ms + control ~4.5ms + bus ~1ms) and a bifurcation threshold at 10ms. That leaves a 0.5ms margin. In 20 years of industrial practice, I have never seen a system operate reliably with a 5% timing margin. Here is why:

1. **IMU jitter:** MEMS IMUs do not deliver data at exactly 1ms intervals. The actual readout time varies by 10-50% depending on the specific chip, SPI bus contention, and interrupt latency.
2. **Filter computation variability:** If the complementary filter for v_sag estimation runs on the same microcontroller as ACC, its execution time varies with compiler optimization, cache state, and interrupt load.
3. **Bus arbitration:** CAN bus at 1Mbps has non-deterministic arbitration. A lower-priority message can be delayed by multiple higher-priority messages in the same transmission window. The worst-case bus latency is not 1ms -- it's the maximum number of higher-priority messages times the frame transmission time.
4. **Jitter accumulation:** The individual components (IMU read, filter compute, ACC compute, CAN transmit, actuator receive, actuator PID compute, torque ramp) each have jitter. The *worst-case end-to-end latency* is the sum of worst-case latencies, not the sum of means. The paper reports means.
5. **Thermal effects:** Microcontroller clock stability varies with temperature. An outdoor robot operating from -10C to +40C will see clock drift.

The paper's conclusion that ACC is "fundamentally latency-sensitive with a threshold below 10ms" (line 669, Limitation 4) is honest. But the hardware latency budget implies deployability with 0.5ms margin, which is not realistic. A more honest statement would be: "The 10ms bifurcation threshold requires end-to-end latency below 8ms for safe operational margin, which exceeds the estimated sensor chain latency of ~9.5ms. Hardware deployment will require either latency-robust estimation (e.g., state prediction to compensate for measurement delay) or a faster sensor chain (e.g., dedicated IMU processor, EtherCAT instead of CAN bus)."

**Evidence Anchor:** Section VI "Path to Hardware" latency budget vs Table VI robustness results.

**Suggestion:** Add a paragraph to the "Path to Hardware" section acknowledging the margin problem and discussing latency mitigation strategies: (a) dedicated IMU with on-chip sensor fusion (e.g., BNO055, ICM-20948 with DMP) to bypass the 3ms filter stage, (b) EtherCAT or SPI-based actuator communication to eliminate CAN bus arbitration jitter, (c) state prediction/forward-simulation to compensate for known measurement delay.

**Severity:** Major (impacts the hardware deployability claim). Text-level fix.

**Confidence:** 4. I have not measured the specific sensor chain; my estimate of the margin problem is based on general industrial experience with real-time control loops.

---

### W4: The 50+ hand-tuned parameter set is an industrial maintainability concern. [Major, Confidence: 5]

**What the existing R3 said:** Not addressed as a weakness. The DA flagged it (SC-W23) as a reproducibility concern carried from Round 7.

**What I add:** The paper's intellectual honesty about statistical power and limitations is commendable. But there is a different kind of honesty that matters for industrial deployment: **maintainability honesty.** A controller with 50+ hand-tuned parameters is a controller that:

1. **Cannot be tuned by anyone except the original designer.** In a chemical plant, if the instrument technician needs to call the PhD control engineer every time the process conditions change, the controller is not deployable. ACC has the same characteristic.
2. **Has hidden parameter interactions.** The paper's gate sensitivity analysis shows alpha_r dominates at 220.4 (20x the next parameter). But this is a *local* sensitivity -- at a different operating point or after retuning another parameter, the sensitivity hierarchy may change. With 50 parameters, exhaustive interaction testing is combinatorially impossible.
3. **Creates a calibration burden that scales with N_robots.** Tuning ACC on one robot required extensive simulation experimentation. Tuning it on 10 robots with manufacturing variance (different link masses, different actuator friction, different gearbox backlash) would require 10x the effort -- or accepting degraded performance.
4. **Is fragile under component aging.** Actuator torque constant degrades with temperature and wear. Gearbox efficiency changes with lubricant aging. The 50 parameters were tuned for a single simulation model -- when the real robot's dynamics drift, which parameters should be retuned, and how?

The paper partially addresses this with the 5-step porting procedure, which focuses tuning effort on the critical parameters (alpha_r, envelope band). But the procedure has been validated on exactly zero additional platforms (Limitation 8). The claim that "A fixed kp and anchor integral gain Ki require minimal adjustment across platforms" (line 647) is unevidenced.

This is not a fatal weakness -- the paper is a first demonstration, not a product. But the "Path to Hardware" section should acknowledge the maintainability challenge and discuss auto-tuning strategies (e.g., system identification for the ringdown period, automated envelope-band calibration from idle-velocity statistics, gain margin measurement for kp validation).

**Evidence Anchor:** Lines 647, Limitation 8 (line 678).

**Suggestion:** Add to the "Path to Hardware" discussion: "The 50+ parameter set was determined through iterative manual tuning on a single simulated platform. For multi-robot deployment with manufacturing variance, an automated calibration procedure would be needed -- for example, system identification of the dominant ringdown mode from a short push-response test, automated quiet-band calibration from idle-velocity statistics, and gain-margin measurement for kp validation. The gate sensitivity hierarchy (alpha_r dominating) focuses such automation on the critical few parameters."

**Severity:** Major (practical deployability). Text-level fix.

**Confidence:** 5. I have maintained multi-loop controllers with hundreds of parameters across refinery units. Parameter count directly correlates with maintenance cost.

---

### W5: Stuck-gate failure modes are not analyzed, and there is no functional safety discussion. [Major, Confidence: 4]

**What the existing R3 said:** Not addressed. This is unique to the industrial safety perspective.

**What I add:** In process control, every conditional activation mechanism must be analyzed for its failure modes. Specifically, every gate must be evaluated for:

1. **Stuck-at-1 (gate fails closed):** The controlled component is permanently active. Consequences:
   - Stuck-at-1 for g_prox + g_env + g_anchor: anchor integral always active, winds up when robot is pushed far from home, prevents recovery. **Fail-dangerous.**
   - Stuck-at-1 for g_theta: boost always active, parametric oscillation if pitch is nonzero. **Fail-dangerous.**
   - Stuck-at-1 for g_flight: flight PD always active, wheels acting as reaction flywheels during normal balancing, destabilizes the balance loop. **Fail-dangerous.**
   - Stuck-at-1 for g_terrain: per-leg height adaptation always active, splits height command during normal two-leg stance, induces unnecessary roll. **Fail-degraded (not immediately dangerous).**

2. **Stuck-at-0 (gate fails open):** The controlled component is permanently inactive. Consequences:
   - Stuck-at-0 for g_prox + g_env + g_anchor: anchor never active, robot operates in P-only mode with 55mm limit cycle. **Fail-degraded (precision lost, balance maintained).**
   - Stuck-at-0 for g_theta: boost never active, ringdown never completes, persistent oscillation after push. **Fail-degraded (recovery impaired, balance maintained).**
   - Stuck-at-0 for g_flight: flight PD never activates during free-fall, wheels free-spin, robot flips. **Fail-dangerous during flight conditions.**
   - Stuck-at-0 for g_terrain: per-leg adaptation never activates, downhill leg collapses on curb. **Fail-dangerous during terrain conditions.**

3. **Stuck-at-intermediate (gate fails half-open):** The controller operates in the partial-engagement regime. The paper's own data shows this is dangerous -- medium noise at 0ms delay produces 34mm idle RMS (Table VI), worse than P-only (55mm RMS but at least predictable). The gate-state bifurcation the DA identified is real: the partial-engagement regime is systematically worse than full disengagement.

**The safety implication:** ACC has no mechanism to detect that a gate is stuck and fail over to a safe degraded mode. In industrial practice, every override controller has: (a) a watchdog timer that detects if the selector has not changed state within an expected interval, (b) a manual override that forces the controller to a known safe state, and (c) an alarm that notifies the operator of the abnormal condition. ACC has none of these.

For a research demonstration, this is acceptable. For a paper claiming a "path to hardware," it is a significant omission. The staged commissioning plan should include a gate-health monitoring stage: before free standing, verify each gate opens and closes correctly under known test conditions.

**Evidence Anchor:** Absence of any failure-mode or safety discussion in the paper. The staged commissioning plan (Section VI) lists stages but does not address failure detection at each stage.

**Suggestion:** Add a paragraph to the "Path to Hardware" section: "Gate failure modes must be characterized before untethered operation. Each gate's stuck-at-1 and stuck-at-0 consequences should be analyzed, and gate-health monitoring (e.g., watchdog timers that verify gate state transitions within expected intervals, pre-flight gate-exercise sequences that confirm correct open/close behavior) should be implemented. For the initial hardware deployment, a manual kill-switch that forces all gates to 0 (reverting to P-only balance) provides a known-safe degraded mode."

**Severity:** Major (safety). Text-level fix -- does not require new experiments but requires the author to think through failure modes.

**Confidence:** 4. I am applying industrial functional safety standards (IEC 61511) to a research prototype. The paper does not claim compliance with these standards, so my concern is about the *trajectory* toward safe deployment, not about current non-compliance.

---

### W6: "ACC as precondition for RL" is a strong claim with zero evidence. [Major, Confidence: 4]

**What the existing R3 said:** Not addressed as a weakness. The EIC flagged it (SC-W3) and the DA flagged it (M2).

**What I add:** From an industrial perspective, the idea that a structured classical controller is a safer base for RL residual learning than training from scratch is **intuitively correct** and **empirically unsupported.** In process control, we have learned the hard way that combining controllers creates coupling modes that neither controller exhibits alone:

1. **The RL policy might learn to "fight" the anchor:** During training, the RL policy receives the anchor's integral term as part of the observation (base_action_abs). It might learn that when the anchor is engaging, it should counteract the anchor's torque to create a "softer" effective controller -- essentially learning to cancel the anchor rather than augment it. This is a well-known problem in residual RL.
2. **The anchor's gate state becomes part of the RL observation space:** A policy that conditions on g_prox and g_env will learn different behaviors in different gate regimes. When the gate is half-open (partial engagement), the policy might exhibit behaviors that were never seen during training because that regime was rare. This is the "distributional shift" problem.
3. **Safety during RL exploration:** During RL training, the policy will explore actions that destabilize the robot. The anchor is designed to recover from pushes, but can it recover from the policy deliberately commanding destabilizing actions? If the policy learns to oscillate at exactly the ringdown frequency, the anchor's envelope follower will stay above threshold, the gates will stay closed, and the robot will fall.

The paper claims ACC is a "precondition" for RL (line 66: "ACC is thus a precondition for RL on this platform: a height-general structured prior that a residual policy can later augment"). This is a prediction, not a finding. Until residual RL training with ACC as the base controller is demonstrated, this claim should be presented as a hypothesis, not a conclusion.

**Evidence Anchor:** Lines 65-66: "ACC is thus a precondition for RL on this platform."

**Suggestion:** Replace with: "We hypothesize that ACC's height-general structured prior makes it suitable as a base controller for residual RL (Section II), where a bounded PPO policy would learn corrective actions while ACC provides the stability baseline. Validating this hypothesis requires residual RL training, which is deferred to future work."

**Severity:** Major (claim-evidence alignment). Text-level fix.

**Confidence:** 4.

---

## Assumption Audit (Cross-Disciplinary Lens)

### Explicit Assumptions

1. **"Physical conditions (proximity, velocity envelope, pitch stability) are the right gating variables."** From process control experience: this is a reasonable design choice but not the only possible one. An alternative would be to gate on *control effort* (is the torque command near saturation?), *estimation quality* (is the velocity estimate reliable?), or *temporal* conditions (has enough time passed since the last push?). The paper's choice of physical conditions is well-motivated by the specific failure modes (integral windup during sustained position error, boost-induced parametric oscillation during pitch excitation), but the paper could strengthen its argument by discussing *why* these particular physical conditions were chosen over alternatives -- specifically, that spatial gating (proximity) directly addresses the integral windup mechanism, while temporal gating (envelope follower) directly addresses the quiet-stance detection problem.

2. **"Smoothstep (3t^2-2t^3) is the right blending function."** From process control: the choice of blending function matters for the behavior in the transition region, but smoothstep is one of many options. Alternatives: (a) linear ramp (simpler, C^0 only, but sufficient if the transition region is narrow), (b) sigmoid/logistic (C-infinity, but computationally more expensive), (c) piecewise-constant with hysteresis (simplest, but creates a switching surface). The paper does not discuss why smoothstep was chosen over alternatives. The gate shape ablation that the DA requested in Round 7 was never performed. I do not consider this a critical omission -- the smoothstep works, and the ablation already demonstrates the gates' necessity -- but it is a blind spot the paper should acknowledge.

3. **"The actuator path (direct torque, 400 Nm/s rate limit) is representative of real hardware."** From process control: this is the single biggest assumption in the paper, and the one most likely to fail on hardware. Real actuators have: (a) torque ripple (spatial harmonics in the motor that cause torque variation with rotor angle -- typically 5-15% of rated torque), (b) cogging torque (detent torque at specific rotor positions -- can be 1-5% of rated torque even in "cogless" motors), (c) friction nonlinearity (Stribeck curve -- static friction > Coulomb friction > viscous friction -- the transition through zero velocity causes torque discontinuities that the 400 Nm/s rate limit cannot smooth), (d) torque constant variation with temperature (NdFeB magnets lose ~0.1%/C, so a 40C temperature rise reduces torque by ~4%). These effects are not modeled in MuJoCo and will degrade the 0.73mm precision figure. The paper acknowledges hardware degradation generically but does not decompose it into specific physical mechanisms.

### Implicit Assumptions

1. **"The simulation captures all dynamics relevant to the precision-vs-recovery trade-off."** From process control: simulation always misses something. The question is whether what it misses matters. For this paper, the most important missing dynamics are: (a) stiction-induced limit cycles (when the integral term builds up enough to overcome static friction, the robot lurches, the integral resets, and the cycle repeats -- this creates a limit cycle that the simulation's perfect viscous friction model cannot reproduce), (b) backlash in the gearbox (the 0.06mm backlash estimate assumes ideal anti-backlash gearing; real harmonic drives have 0.5-2 arcmin of lost motion, which translates to position uncertainty at the wheel), and (c) ground compliance (the paper shows that softening solref from 0.002 to 0.02 increases idle RMS by ~60% -- real ground compliance is likely worse than solref=0.02).

2. **"The asymmetric EMA with fixed time constants works across all operating conditions."** From process control: fixed-parameter filters work until they don't. The tau_release=1.5s was tuned for the specific ringdown dynamics of this 8.1kg robot at 0.54m CoM height. If the robot carries a payload that changes the CoM height, the ringdown period changes, and the optimal tau_release changes. The gate sensitivity analysis shows tau_release dominates, so this parameter is both the most important and the most vulnerable to operating-condition changes. The 5-step tuning procedure (set tau_release = 3-5 x T_r) addresses this for different morphologies, but not for the same morphology under varying payload.

### Paradigmatic Assumptions

1. **"Classical control is the right approach for this problem."** The paper explicitly argues for classical control over RL (lines 65-66), but this is a paradigmatic choice, not an empirical finding. From process control: I am sympathetic to this choice -- classical controllers are more transparent, more analyzable, and safer than learned policies. But the paper should acknowledge that this is a design philosophy, not a proven superiority. An RL policy trained on the same platform might achieve similar or better results with zero hand-tuning (at the cost of opacity and sim-to-real transfer risk). The paper's value proposition is not "ACC is better than RL" -- it is "ACC is a structured, interpretable, and safe baseline that can later serve as a prior for RL." This framing is already present but could be sharpened.

2. **"Component-by-component ablation is the right way to validate a multi-component controller."** From process control: this is exactly right, and the paper does it well. But there is a subtlety: the ablation demonstrates that *in the presence of the other components*, removing any single component degrades performance. It does not demonstrate that each component *individually* would improve a simpler controller. For example, adding the boost to P-only (without the anchor integral) might not help, because the boost and integral interact (the boost damps the oscillation that the integral would otherwise fight). The paper's additive ablation (L0 to L3) partially addresses this by showing the cumulative effect, but the subtractive ablation (S1 to S5) starts from the full configuration and removes components one at a time. This is valid for demonstrating necessity-in-context, but the paper should be careful not to claim that each component is individually beneficial in isolation.

---

## Cross-Disciplinary Connections: Deep Analysis

### Is "conditional gating" genuinely novel?

**Short answer:** No, as an architecture concept. Yes, as a specific synthesis for this specific problem.

**Long answer from process control experience:**

The *architecture* of conditionally activating control components based on process conditions is as old as industrial control itself. Here is a partial taxonomy:

| Industrial Pattern | When Used | How It Works | ACC Analogue |
|---|---|---|---|
| **Override control** (high/low select) | Multiple constraints, one actuator | Multiple controllers compute candidate outputs; a selector (high/low/median) picks the active one | ACC's gates act as continuous selectors rather than discrete ones |
| **Split-range control** | Two actuators, one controller | Controller output is split into two ranges, each controlling one actuator, with overlapping proportional bands | ACC's two-channel assembly (wheel vs leg torque) is structurally similar to split-range |
| **Valve position control (VPC)** | Optimize energy efficiency while meeting constraints | A slow "valve position controller" adjusts a secondary actuator to keep the primary actuator near its optimal operating point | ACC's anchor integral slowly adjusts to cancel equilibrium bias -- functionally similar to VPC's slow loop |
| **Conditional integration** (AW) | Prevent integral windup during actuator saturation or process excursions | Integrator frozen/limited when error exceeds threshold or actuator saturates | ACC's proximity-gated integral is conditional integration gated on spatial proximity |
| **Gain scheduling** | Process dynamics change with operating point | Controller gains are scheduled as a function of a measured scheduling variable | ACC's k_p schedule is gain scheduling on gate state |
| **Bumpless transfer** | Switching between manual/auto or between controllers | The inactive controller tracks the active one; when switched, output is continuous | ACC's smoothstep gates implement continuous (C^0) transfer between active/inactive states |
| **Asymmetric filtering** | Different response needed in different directions | Filter time constants differ for increasing vs decreasing signals | ACC's asymmetric EMA is exactly this pattern |
| **Envelope detection** | Detect signal amplitude for control decisions | Peak-hold or RMS envelope of a signal used to gate control actions | ACC's EMA of `|v_sag|` is an envelope detector |

**Every component of ACC has a direct industrial analogue.** This is not a criticism -- it is an observation that ACC is a well-engineered *synthesis* of mature techniques, not a *discovery* of new ones.

**What is genuinely novel in ACC (from process control perspective):**

1. **The specific gating variables:** Using spatial proximity + velocity envelope + pitch stability as three concurrent gate conditions, where each condition gates a different component and the conditions are derived from fundamentally different physical measurements (position, velocity, attitude). In process control, override selectors typically gate on a single variable (e.g., pressure, temperature, level). Multi-variable gating with derived conditions (EMA of velocity, not raw velocity) is uncommon.

2. **The coupling of gated integral with gated damping boost:** In process control, conditional integration and derivative action are typically independent. ACC couples them: the same gate that enables the integral also enables the boost, and removing the boost (S2) or the pitch gate (S5) causes qualitatively different failure modes (no ringdown vs parametric oscillation). This coupling is not a standard industrial pattern.

3. **The velocity-envelope approach to quiet-stance detection:** Using an asymmetric EMA of |velocity| to distinguish quiet stance from ringdown is a creative adaptation of the envelope-detection pattern. In process control, we use envelope detection for compressor surge detection and vibration monitoring, but not for "quietness" detection in the sense ACC uses it.

4. **The problem-specific synthesis:** The combination of smoothstep blending functions, asymmetric envelope follower, and multi-condition gating, applied to the specific problem of wheeled-biped precision-vs-recovery trade-off, is a genuine engineering contribution even though each individual component has industrial precedent.

**Recommendation:** The paper should include a version of the table above (condensed) in the Related Work or Discussion to transparently situate ACC within the process control lineage. This would preempt the "this is just override control" criticism by showing exactly where ACC differs from standard industrial patterns.

### The smoothstep function (3t^2-2t^3) -- contribution or implementation detail?

**Implementation detail.** The smoothstep function is the Hermite interpolation polynomial between 0 and 1 with zero derivatives at both endpoints. It was popularized in computer graphics by Ken Perlin in the 1980s but has been used in control engineering for decades as a "soft switching function." The specific polynomial form is not novel.

What *would* be a contribution is an analysis of *why* smoothstep is the right choice for this application. Specifically:
- C^1 continuity at the endpoints ensures that the gate derivative is zero at full-open and full-closed, preventing torque-rate transients when the gate transitions into or out of its active region.
- The monotonicity of smoothstep ensures that the gate never decreases when the gating variable moves toward the active region.
- If a linear ramp were used instead (C^0 at endpoints), the torque rate would have a step change at the gate boundaries, which could interact with the 400 Nm/s rate limit to produce torque clipping.

The paper mentions the C^1 property in a footnote (line 135) but does not analyze its practical significance. Adding 2-3 sentences explaining why the shape matters (not just that it is smooth) would elevate this from implementation detail to informed design choice.

---

## Practical Impact: Industrial Reality Check

### The 0.73mm precision claim -- from simulation to hardware

The paper discloses hardware degradation factors: encoder quantization (>0.5mm), backlash (0.02-0.06mm), stiction, tire compliance (0.1-0.5mm), and IMU noise. The projected hardware range is 1-3mm. I would add:

1. **Ground truth problem:** In simulation, the CoM position is known exactly. On hardware, the CoM position must be estimated from joint encoders + IMU. Every estimation error becomes a position control error. The paper's 0.73mm is a *state estimation lower bound* -- hardware precision will be strictly worse.

2. **Actuator torque resolution:** 60Nm actuators with 12-bit current sensing give ~0.015 Nm torque resolution. But the torque *ripple* (spatial harmonics) is typically 5-15% of rated torque, or 3-9 Nm at 60 Nm. This ripple is at the electrical frequency (pole pairs x mechanical speed), which for a slow-rolling wheel is ~10-50 Hz -- within the balance control bandwidth. The ripple acts as an unmeasured torque disturbance.

3. **Thermal expansion:** An aluminum link expands by ~23 ppm/C. For a 0.26m thigh link, a 40C temperature rise changes the length by ~0.24mm. The IK model assumes constant link lengths -- if the links thermally expand, the posture controller commands the wrong joint angles, and the CoM height deviates.

The 1-3mm hardware projection is reasonable as a mean estimate under controlled laboratory conditions. Under outdoor conditions with temperature variation, ground compliance, and sensor degradation, expect 3-10mm.

### The latency sensitivity -- bifurcation at 10ms

The paper's latency sensitivity finding (delay bifurcation at 10ms) is the most important practical result. From process control experience, this is a phase margin problem: at 100Hz control rate with stiff gains (kp=50), the closed-loop bandwidth is likely 3-5 Hz. A 10ms delay corresponds to 18-36 degrees of phase lag at 3-5 Hz, which is enough to consume the available phase margin of a PD controller with moderate derivative gain (kd=3.0, providing ~20-30 degrees of phase lead at 3-5 Hz).

The paper presents the latency sensitivity as a finding. It should also present it as a *design constraint*: any hardware implementation of ACC must have end-to-end latency below 8ms (for safe margin), which means:
- IMU readout: <1ms (requires SPI at 10MHz+)
- Sensor fusion: <2ms (requires optimized complementary filter, possibly on dedicated IMU processor)
- Control computation: <2ms (requires <200 FLOPs at 100Hz -- ACC achieves this)
- Communication: <1ms (requires EtherCAT or SPI, CAN bus insufficient)
- Actuator response: <2ms (requires low-inductance motors, high-bandwidth current loop)

This is achievable with modern hardware but requires careful engineering, not off-the-shelf components.

### The staged commissioning plan -- credible but incomplete

The paper's staged commissioning plan (static torque -> fixed-support -> free standing -> push recovery -> flight -> terrain) is sensible and follows standard robotics practice. From process control, I would add:

1. **Gate-exercise stage (between static torque and fixed-support):** Before any dynamic testing, each gate should be exercised through its full range to verify correct open/close behavior. This catches gate implementation bugs (sign errors, threshold off-by-one) before they cause falls.
2. **Failure-injection stage (between push recovery and flight):** Deliberately inject gate failures (force each gate to 0 and to 1) to verify the robot's degraded-mode behavior and to train the operator on failure recognition. In process control, we do this as part of the "pre-startup safety review" (PSSR).
3. **Long-duration standing test:** Run the controller for 1+ hours to characterize thermal drift, sensor bias drift (IMU gyro bias changes with temperature), and actuator temperature rise. The 0.73mm figure is from a 20-second window -- does it hold for 1 hour?

---

## Broader Implications

### Safety implications of conditionally-gated control

The paper positions conditional gating as a design pattern for robotics control. From process control, I see both opportunity and risk:

**Opportunity:** Conditional gating makes the controller's behavior interpretable. If the robot falls, you can check which gate was active and diagnose the failure mode. This is vastly preferable to a black-box neural network policy where a fall leaves you with no diagnostic information.

**Risk:** Conditional gating creates *mode confusion* -- situations where the controller is in an unexpected mode because a gate is in an unexpected state. This is the single biggest cause of industrial accidents involving advanced control: the operator thinks the controller is in Mode A, but it is actually in Mode B because a gate transitioned silently. The paper should discuss how gate states would be monitored and displayed in a real deployment.

**Risk:** The paper identifies the partial-engagement regime (gates half-open) as a failure mode (medium noise at 0ms delay produces 34mm idle RMS, worse than P-only). But ACC has no mechanism to detect when it is in this regime and recover. A watchdog that monitors gate state durations and forces a safe mode if a gate stays in the transition region too long would be a simple safety addition.

### ACC as precondition for RL -- industrial perspective

From process control: the idea of using a classical controller as a safety baseline for RL is sound and aligns with industrial practice. In chemical plants, we never deploy a purely data-driven controller without a classical fallback. The typical architecture is:

```
Classical PID (always active, safety-guaranteed)
    +
Model-based optimizer (active when model is valid)
    +
Data-driven corrector (active when data is sufficient)
```

This is structurally similar to ACC + residual RL. The classical controller provides the safety floor; the learning-based components provide performance optimization.

However, the industrial approach differs from the paper's proposal in a crucial way: **in industry, the classical controller is the primary controller, and the learning-based components are add-ons that can be disabled without loss of basic functionality.** The paper's proposal (base_action + residual_scale * residual_action) has the residual output added to the base action -- the RL policy can override the classical controller. This is riskier than the industrial approach, where the learning component typically provides a *feedforward* term or a *setpoint adjustment*, not a direct action override.

For safety, I would recommend: **the residual action should be bounded such that the final action can never be more than X% different from the base action.** The paper's residual scale already bounds the residual by 5-40% of the action range, which is a reasonable choice, but the safety implications of this bound should be discussed.

### Generality across morphologies

The paper claims generality (implicitly, through the 5-step porting procedure) but tests on exactly one platform. From process control: I am skeptical of any claim of generality without multi-unit testing. Chemical processes are all "continuous stirred-tank reactors" in the abstract, but the specific dynamics (time constants, gains, nonlinearities) differ by orders of magnitude, and a controller that works on one reactor rarely transfers without retuning.

The 5-step procedure is a tuning guide, not a generality proof. Generalization requires: (a) testing on at least 2-3 morphologically distinct platforms (different mass, CoM height, leg kinematics), (b) demonstrating that the procedure produces stable controllers without per-platform manual tuning beyond what the procedure specifies, and (c) characterizing the procedure's failure modes (when does it produce an unstable controller?).

This is future work, and the paper (Limitation 8) acknowledges it. I flag it to ensure the generality language in Section V-E does not overclaim relative to the evidence.

---

## Stakeholder Blind Spots

### Industrial robotics deployment

The paper is a research contribution, not a product, and it appropriately does not claim industrial readiness. However, as someone who has deployed controllers in industrial settings, I note these blind spots:

1. **Operator interface:** What does a robot operator see when ACC is running? Which gate states are displayed? How does the operator know the robot is in "anchored standing" vs "P-only fallback" vs "flight mode"? In a factory, the operator needs at-a-glance situational awareness.

2. **Alarm management:** What conditions trigger an alarm? Gate stuck? Position error exceeds limit? The paper identifies failure modes but does not map them to alarm conditions.

3. **Preventive maintenance:** How often should the gate parameters be recalibrated? Actuator wear changes the motor torque constant, which changes the effective kp. After 1000 hours of operation, is the robot still sub-millimeter?

4. **Fleet management:** If you deploy 50 of these robots in a warehouse, each with slight manufacturing variance, can you use the same ACC parameters for all 50? Or does each robot need individual calibration? The maintenance burden of 50 individually-tuned robots is vastly higher than 50 identically-tuned robots.

5. **Failure forensics:** When a robot falls and damages itself or its environment, can you reconstruct what happened from the gate state log? The paper's gate activation figure (Fig. 4) suggests yes, but the logging infrastructure for a deployed system needs to capture gate states, sensor data, and control outputs at sufficient bandwidth for post-incident analysis.

### Safety certification

The paper does not address functional safety standards (IEC 61508, ISO 13849, ISO 10218 for industrial robots). This is appropriate for a research paper -- these standards are not expected in a robotics conference/journal submission. However, the "path to hardware" discussion should acknowledge that safety certification imposes additional requirements:

- **IEC 61508 SIL 2** (typical for industrial robots): Requires hazard analysis (HAZOP or FMEA), safety integrity level determination, and proof that the safety function meets the required probability of failure on demand. ACC's gates would need to be analyzed as safety functions.
- **ISO 13849 PL d** (typical for collaborative robots): Requires redundant channels for safety-critical functions, diagnostic coverage >90%, and common-cause failure analysis. ACC's gate-health monitoring (see W5) would need to meet these requirements.

The paper does not need to comply with these standards. It should acknowledge that they exist and that hardware deployment would require engaging with them.

### Manufacturing integration

How does ACC integrate with higher-level task planning? A warehouse robot needs to: receive a pick location, navigate to the location, stop, lower to pick height, pick the item, raise to travel height, navigate to the place location, stop, lower to place height, place the item. ACC provides the "stop and balance" primitive, but:

- How does ACC handle the transition from navigation (rolling) to balancing (standing still)? The paper's height transition results show push survival degrades 84% during squat transitions -- how does this affect task-level reliability?
- How does ACC coordinate with a navigation stack? The navigation controller commands wheel velocities; ACC commands wheel torques. Who wins when they disagree?
- What is the interface? Is ACC a ROS node with standard control topics? The paper does not specify the software architecture.

These are implementation concerns that go beyond the paper's scope, but the paper positions ACC as a foundation for real deployment, and these integration questions would be among the first asked by an industrial user.

---

## Cross-Disciplinary Reading Recommendations

These are references from the process control and industrial automation literature that would enrich the paper's intellectual lineage. I do not claim they are mandatory citations -- they are suggestions for authors who want to engage more deeply with the process control perspective.

1. **[VERIFIED] Shinskey, F.G. (1996).** *Process Control Systems: Application, Design, and Tuning*, 4th ed. McGraw-Hill. **Already cited.** Specifically, Chapter 7 (Split-Range and Valve-Position Control) and Chapter 9 (Override and Constraint Control) are directly relevant. ACC's two-channel torque assembly is structurally similar to split-range, and the anchor is functionally similar to valve-position control (a slow loop that keeps a fast loop near its optimal operating point).

2. **[VERIFIED] Astrom, K.J. and Hagglund, T. (2005).** *Advanced PID Control*. ISA. **Already cited.** Section 10.5 (Selector Control) and Chapter 3 (Controller Design) discuss override architectures and conditional integration. ACC's smoothstep blending can be viewed as a continuous generalization of the discrete selector patterns described here.

3. **[UNVERIFIED] McMillan, G.K. (1994).** *Tuning and Control Loop Performance*, 3rd ed. ISA. Chapter on "Controller Output Characterization" describes asymmetric filter time constants for different operating regimes -- directly relevant to ACC's asymmetric EMA. Search for: "asymmetric filter process control."

4. **[UNVERIFIED] Blevins, T.L. and Nixon, M. (2011).** *Control Loop Foundation -- Batch and Continuous Processes*. ISA. Chapter on "Override Control and Windup Protection" documents industrial best practices for bumpless transfer and tracking initialization -- the techniques ACC's smoothstep gates functionally replace.

5. **[UNVERIFIED] Luyben, W.L. (1990).** *Process Modeling, Simulation, and Control for Chemical Engineers*, 2nd ed. McGraw-Hill. Chapter 13, "Override and Selective Control," describes the standard industrial taxonomy of override architectures (high-select, low-select, median-select) that ACC's continuous blending extends.

6. **[UNVERIFIED] Rhinehart, R.R. (2000).** "The Century's Greatest Contributions to Control Practice." *ISA Transactions*, 39(1), 3-13. A survey of the most impactful control innovations of the 20th century from an industrial practitioner's perspective. Useful for situating ACC within the broader trajectory of control practice.

7. **[UNVERIFIED] Henson, M.A. and Seborg, D.E. (1997).** *Nonlinear Process Control*. Prentice Hall. Chapter on "Input-Output Linearization and Nonlinear Control Strategies for Chemical Reactors" discusses switching control strategies for exothermic reactors -- structurally similar to ACC's regime-based gating. Search for: "switching control exothermic reactor."

---

## Dimension Scores

| Dimension | Score | Descriptor |
|-----------|-------|------------|
| **Control-Theoretic Soundness** | 68 | Components individually sound; synthesis lacks formal analysis (no Lyapunov, describing function, or linearized stability proof). Empirical validation compensates partially. |
| **Anti-Windup Engagement** | 72 | ACC's anchor is correctly positioned relative to conditional integration. But claim that gating on "physical state" is novel vs AW is imprecise (see W2 in existing R3 review). The envelope-detected, boost-coupled form is the distinction, not state-gating per se. |
| **Process Control Literature Integration** | 60 | Shinskey and Astrom & Hagglund now cited -- a major improvement. But engagement remains at citation level, not conceptual level. The paper positions ACC as extending selector-based control without analyzing what a standard industrial override controller would look like for this problem (see my W1). |
| **Embedded/Hardware Feasibility** | 58 | Realistic FLOPs budget. Latency sensitivity well-characterized. But 0.5ms timing margin, 50+ parameter maintainability, and absence of failure-mode analysis are significant concerns for industrial deployment. |
| **Industrial Relevance** | 62 | Proximity-gated anchor + asymmetric EMA are genuinely useful patterns. Ablation methodology is industrial best practice. Safety and maintainability gaps prevent higher score. |
| **Practical Transferability** | 55 | 5-step porting procedure is heuristic and untested on second platform. Hardware degradation analysis is qualitative. Latency budget is optimistic. Tuning burden is high. |
| **Weighted Aggregate** | ~63 | The paper's empirical strengths are clear, but from an industrial practitioner's perspective, the gap between simulation demonstration and deployable controller is larger than the paper acknowledges. |

**Note on scoring:** My scores are lower than the robotics reviewers' because I am evaluating from the perspective of "could I commission this controller in a chemical plant?" rather than "is this a strong simulation-based robotics paper?" The paper is a strong simulation-based robotics paper. It is not yet a deployable industrial controller. The score reflects the latter standard.

---

## Detailed Comments

### On the existing R3 review

The existing R3 (Cross-Disciplinary / Process Control) correctly identified that ACC's architectural framing overclaims relative to established process control practice (SC-W15), that the AW distinctiveness claim needs qualification (SC-W16), and that the 5-step porting procedure generality claims are untested (SC-W17). I endorse all three of these weaknesses.

My review adds:
1. **A specific alternative controller design** (1995-vintage override controller) against which ACC's innovations can be measured (my W1).
2. **The latency margin analysis** showing the 0.5ms margin is unrealistic (my W3).
3. **The parameter maintainability concern** from an industrial operations perspective (my W4).
4. **The stuck-gate failure mode analysis** and functional safety gap (my W5).
5. **The "ACC as precondition for RL" claim-evidence gap** (my W6).
6. **The industrial deployment blind spots** (operator interface, alarm management, preventive maintenance, fleet management, failure forensics).

These concerns are complementary to, not duplicative of, the existing R3's concerns.

### On the DA's CRITICAL issues

I agree with the DA's C1 (k_p schedule inconsistency -- S1 outperforms S0) and note that this is a *configuration management* issue: the paper's "recommended configuration" (S1, fixed k_p=50) is different from its "reference configuration" (S0, scheduled k_p). This is exactly the kind of documentation-software mismatch that causes industrial incidents ("the P&ID shows the override active, but the DCS has it bypassed"). The fix is simple: make S1 the reference, rename "Full ACC" to "ACC + scheduled kp" and S1 to "ACC (recommended)."

I agree with the DA's M1 (gate-state bifurcation -- partial engagement worse than P-only). This is a systematic fragility. The controller has no mechanism to detect or avoid the pathological middle regime. From process control: this is analogous to a split-range controller where the overlap region creates an unintended low-gain zone. The standard fix is to narrow the overlap or add hysteresis to prevent the controller from dwelling in the transition region. ACC could adopt a similar strategy: if the gate remains in [0.2, 0.8] for more than N seconds, assume a fault and force the gate to 0 (safe state).

### Overall assessment of the paper's contribution

From process control, I see this paper as a **well-engineered synthesis of mature control techniques applied to a specific robotics problem.** The individual techniques (conditional integration, envelope detection, gain scheduling, smooth blending) are all standard. The synthesis -- specifically the combination of proximity-gated integral with asymmetric-envelope-detected damping boost, coupled with the smoothstep-based continuous blending -- is novel in the wheeled-biped context and demonstrably effective.

**The paper's most valuable contributions, from an industrial perspective, are:**

1. The demonstration that standard LQR through a PID servo fails (the negative result).
2. The systematic ablation showing which components matter and why.
3. The parameter sensitivity analysis showing alpha_r dominates (focuses tuning effort).
4. The honest acknowledgment of limitations (builds trust).

**The paper's most significant gaps, from an industrial perspective, are:**

1. The failure to engage with the process control literature at the conceptual architecture level (not just citation level).
2. The absence of any failure-mode or safety analysis.
3. The optimistic hardware latency budget.
4. The unevidenced "precondition for RL" claim.

**None of these gaps require new experiments.** All are addressable through text revisions: sharper framing, acknowledgment of industrial realities, and qualification of claims.

---

## Recommendation

**Minor Revision** -- convergent with the existing Round 8 consensus.

The paper's empirical contribution is solid. The framing and positioning need industrial-realist adjustments. The five weaknesses I identify (W1: process control architecture comparison, W2: asymmetric EMA lineage, W3: latency margin, W4: parameter maintainability, W5: stuck-gate failure modes, W6: RL precondition claim) are all text-level fixes.

I would like to see these changes in revision:

### Required (for my accept)

1. **Add a substantive comparison to a hypothetical industrial override controller** (W1). A paragraph or table showing: "Here is what a standard override controller would look like for this problem, here is specifically where ACC differs, and here is why those differences matter." This single addition would address the most significant gap in the paper's intellectual framing.

2. **Qualify the "ACC as precondition for RL" claim as a hypothesis** (W6). Replace the present-tense assertion with a future-work hypothesis.

3. **Add a gate failure-mode analysis** (W5). A short paragraph in the "Path to Hardware" section listing the stuck-at-1 and stuck-at-0 consequences for each gate.

### Recommended (strengthens the paper)

4. **Acknowledge the asymmetric EMA's industrial lineage** (W2). One sentence citing the use of asymmetric filters in anti-surge control, pH control, and surge-tank level control.

5. **Address the latency margin problem** (W3). Acknowledge that the 9.5ms estimate leaves 0.5ms margin at the 10ms bifurcation threshold, and discuss latency mitigation strategies.

6. **Acknowledge the parameter maintainability challenge** (W4). One paragraph on auto-tuning strategies for multi-robot deployment.

7. **Include a process-control lineage table** (see Cross-Disciplinary Connections section). Map each ACC component to its industrial analogue, with specific citations. This transparently situates ACC and preempts "this is just X from the 1970s" criticisms.

### Optional (for the authors' consideration)

8. **Request a second-platform test** for the 5-step porting procedure. This would transform the paper from a single-platform case study to a demonstration of architectural generality. I recognize this is a non-trivial request (requires a second MuJoCo model or porting to a different simulator) and do not make it mandatory.

---

## Questions for Authors

1. **Have you tested what happens when a gate is forced to an intermediate value (e.g., 0.5) for an extended period?** The robustness results (Table VI) show partial-engagement under medium noise at 0ms delay produces worse performance than P-only. Is this a fundamental property of the architecture, or can it be mitigated by narrowing the transition band?

2. **What would change if you used a linear ramp instead of smoothstep for the gate blending functions?** The paper claims C1 continuity matters but does not test the alternative. Is there a measurable difference, or is smoothstep chosen for aesthetic reasons?

3. **How would you auto-tune the critical parameters (alpha_r, envelope band) for a new robot without manual experimentation?** The 5-step procedure assumes manual simulation access. For factory calibration of 50+ robots, an automated procedure is needed.

4. **What is the worst-case end-to-end latency** in your estimated sensor chain (not the mean)? If the worst-case exceeds 10ms even occasionally, does ACC handle occasional deadline misses gracefully, or does a single missed deadline cause a fall?

5. **Have you considered adding a gate-health watchdog** that forces all gates to 0 (P-only fallback) if a gate dwells in the transition region for too long? This would address the partial-engagement failure mode.

---

*Review prepared 2026-07-29 by a reviewer with 20+ years of industrial process control experience (ExxonMobil/BASF), now in robotics research. I have read the Round 8 reviews and this review is intended as a complementary industrial-practitioner perspective. All page/line references are to the manuscript version reviewed (commit f92640c).*
