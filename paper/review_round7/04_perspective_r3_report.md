Now I have read both the reviewer agent definition and the full paper. Let me produce the Perspective Review Report as Peer Reviewer 3, from the perspective of classical control theory and industrial process control.

---

## Perspective Review Report (Peer Reviewer 3)

### Reviewer Identity

I approach this paper from **classical control theory and industrial process control** -- a field where conditional, gated, and selector-based control architectures have been standard practice for decades (override control in chemical plants, anti-windup in aerospace, bumpless transfer in flight control systems). I have seen multiple "novel" robotics controllers that rediscover control-theoretic ideas under new names. My value is spotting where this paper genuinely extends classical ideas versus where it renames them, and assessing whether the claimed contributions would survive contact with real hardware.

### Overall Recommendation

Minor Revision

### Confidence Score

4 -- I am confident in my assessment of the control-theoretic framing and practical feasibility questions, but I acknowledge I am not a wheeled-biped hardware experimentalist, so my judgment of the simulation fidelity claims carries appropriate uncertainty.

### Summary Assessment

This paper presents a carefully engineered, empirically validated controller for a specific robot morphology. The core idea -- conditionally gating control components by physical regime -- is genuinely useful and well-executed. The ablation study is unusually thorough for a robotics paper, and the honesty about latency sensitivity and simulation-only status is commendable. However, the paper systematically understates how much of its architecture is a direct application of established process-control principles (override control, bumpless transfer, anti-windup conditional integration) to a robotics platform. The "asymmetric envelope follower" is the most genuinely novel element and is correctly identified as the critical mechanism. The hardware degradation estimate of 1-3mm is optimistic given that the controller itself is latency-sensitive below 10ms -- a threshold that typical embedded IMU+encoder fusion pipelines approach or exceed. The S5 pitch gate ablation revealing a 9x push-survival collapse raises a safety concern that the paper acknowledges but does not adequately address from a failure-modes perspective. As a single-author paper, the scope is ambitious but the author is transparent about what is qualitative versus quantitative. The paper should be publishable after addressing the control-theory lineage more honestly and strengthening the hardware-path discussion.

### Strengths

1. **S1 -- Honest empirical characterization**: The paper explicitly measures and reports what classical control engineers would call the "control authority envelope" -- latency sensitivity below 10ms, noise thresholds, rate-limit invariance, and the sharp delay bifurcation. This is rare in robotics papers and reflects mature engineering practice. The acknowledgment that "high noise causes immediate fall" and that the EMA cannot distinguish noise from motion at 10x consumer IMU levels is exactly the kind of frankness needed.
   - **Evidence Anchor**: `text: Table 4 (Robustness to sensor noise and actuator delay), §4.5, "ACC is fundamentally latency-sensitive with a threshold below 10ms"`

2. **S2 -- Factorial ablation structure**: The additive (L0-L3) and subtractive (S0-S5) ablation paths independently converge on the same conclusions, and the statistical power limitations are explicitly discussed with Bonferroni correction and minimum detectable effect sizes. This is control-engineering discipline transferred to robotics evaluation.
   - **Evidence Anchor**: `text: Table 3 (Factorial ablation), §4.2, "L0-L3 additive, S0-S5 subtractive"`

3. **S3 -- The asymmetric envelope follower concept**: The fast-attack/slow-release EMA for distinguishing quiet stance from post-push ringdown is genuinely elegant. The symmetric-EMA ablation confirming bistable failure is a clean demonstration that the asymmetry is necessary, not cosmetic. This mechanism has no direct analogue in classical override or selector control and represents a real contribution.
   - **Evidence Anchor**: `text: Eq. 5 -- "asymmetric envelope follower" with tau_attack ~ 30ms, tau_release ~ 1.5s`

4. **S4 -- Explicit acknowledgment of simulation limitations**: The paper states flatly that "all results are simulation-only," lists 10 specific limitations, and provides a staged commissioning plan for hardware. This is far more transparent than typical robotics submissions.
   - **Evidence Anchor**: `text: §6, "All results are simulation-only (MuJoCo); hardware validation remains future work"`

### Weaknesses

1. **W1 -- Understated debt to classical override/selector control**: The paper references Astrom & Hagglund (2006) for selector-based control and acknowledges the connection, but the framing still presents ACC as a novel architecture when its structure -- multiple controllers with condition-based activation gates -- is the standard architecture of override control in process plants. Chemical plants have used maximum/minimum selectors to switch between throughput, pressure, and temperature controllers for 50+ years. The smoothstep gate is a genuine improvement over discrete switching (it provides bumpless transfer, another classical concept the paper does not name), but the architectural pattern is not new. The paper would be stronger if it positioned ACC explicitly as "we apply well-known override/selector control principles to a wheeled biped, with three innovations: (i) smoothstep blending instead of discrete selection, (ii) an asymmetric envelope follower for regime detection, and (iii) joint integral-plus-damping gating." This is a framing issue, not a technical flaw.
   - **Severity**: Major | **Evidence Anchor**: `text: §3.1, "Conditionally-gated denotes continuous smoothstep activation by physical conditions -- a continuous generalization of selector-based and override control from discrete switching to smooth blending"` | **Confidence**: 4 -- control-theory lineage assessment

2. **W2 -- Hardware latency budget is implausibly tight**: The paper reports 4.5ms computation time (45% of the 10ms control window) and a "fundamentally latency-sensitive" threshold below 10ms. On real hardware, the signal chain is: IMU read (1-2ms via SPI/I2C) + encoder read + complementary filter for velocity estimation (~5ms, per the paper's own estimate) + ACC computation (4.5ms) + torque command transmission + actuator response. This already approaches or exceeds 10ms end-to-end. The paper acknowledges the velocity estimation latency but does not factor it into the total loop delay budget. If the complementary filter adds 5ms and ACC adds 4.5ms, the minimum end-to-end latency is ~10ms -- exactly at the bifurcation threshold. This means the hardware robot would operate in the degraded regime (2.30mm idle, 47N push) permanently, not just under "delay" conditions. The paper should present a realistic end-to-end latency budget for the hardware signal chain.
   - **Severity**: Major | **Evidence Anchor**: `text: §6, "Control computation: ~200-300 FLOPs, ~4.5ms (45% of 10ms window)" and "v_sag via IMU+odometry complementary filter (~5ms latency)"` | **Confidence**: 4 -- embedded systems and sensor fusion latency

3. **W3 -- S5 pitch gate as single point of failure**: The S5 ablation shows that removing the pitch stability gate $g_\theta$ collapses push survival from 80N to 9N -- a 9x reduction. The paper calls this "critical safety" but does not discuss what happens if the pitch gate threshold is miscalibrated on hardware (e.g., due to IMU mounting misalignment, gyro bias drift over temperature, or the 2-degree threshold being too tight for the real robot's mechanical compliance). In process control, any single component whose removal causes 9x performance degradation would trigger a failure modes and effects analysis (FMEA). The paper should discuss: (i) what is the safe operating range for the pitch gate thresholds, (ii) how would a miscalibrated pitch gate manifest, and (iii) is there a graceful degradation path if the pitch gyro fails?
   - **Severity**: Major | **Evidence Anchor**: `text: Table 3, S5 row -- "F_min = 9.0±0.7N" vs Full ACC "F_min = 80.4±3.0N"` | **Confidence**: 3 -- safety analysis from process-control perspective; I am not a functional-safety expert for legged robots

4. **W4 -- Hardware degradation estimate of 1-3mm is optimistic**: The paper projects 1-3mm hardware precision from 0.73mm simulation, enumerating five degradation mechanisms. However, it omits two critical factors: (i) the anchor mechanism depends on accurate sagittal velocity for the envelope follower -- if the velocity estimate has 5ms latency and is derived from a complementary filter with accelerometer bias, the quiet-stance detection degrades, not just the position precision; (ii) the 0.73mm figure is achieved at 0ms delay and clean sensors -- with even 10ms delay, idle RMS is already 2.30mm. Since the hardware latency budget likely exceeds 10ms (W2 above), the effective hardware idle precision would be 2-3mm from latency alone, before adding the five degradation mechanisms. The projected 1-3mm range should be revised to 2-5mm or the latency budget should be demonstrated to stay below 10ms end-to-end.
   - **Severity**: Major | **Evidence Anchor**: `text: Table 4 -- "Clean, 10ms delay: idle RMS 2.30±0.16mm" vs §6 "projected hardware 1-3mm"` | **Confidence**: 4 -- the latency data is the paper's own; the inconsistency is arithmetic

5. **W5 -- "So what?" for broader engineering**: The paper claims ACC is a "precondition for RL on this platform" and a "height-general structured prior." This is a reasonable vision, but the paper does not actually demonstrate that ACC's architecture transfers to other morphologies or even other simulators. The gate-tuning procedure (5 steps in §4.5) is described qualitatively but not validated on a second platform. From a process-control perspective, the question is: is conditionally-gated control a generalizable design principle, or is it a one-off tuning exercise for one specific robot? The paper needs to either (i) demonstrate the architecture on at least one variant morphology (different leg lengths, different mass distribution) or (ii) more modestly claim that conditionally-gated control is a promising approach whose generality remains to be established. The current framing implies generality without evidence.
   - **Severity**: Minor | **Evidence Anchor**: `text: §6, "Architectural generality across morphologies and simulators remains to be demonstrated"` | **Confidence**: 3 -- this is a framing scope question

6. **W6 -- Single-author scope concerns**: The paper ambitiously covers: balance core, anchor mechanism, posture control, flight recovery, terrain adaptation, four LQR baselines, factorial ablation, noise/delay robustness, gate sensitivity analysis, rate-limit invariance, and compound disturbance recovery. As a single-author submission with simulation-only results, the breadth raises quality-control questions: were all measurements independently reproduced? The paper states "means verified via independent re-measurement" for LQR baselines but does not describe who performed the verification or what independence means. For journal publication, independent verification by a second person would strengthen credibility. This is not a rejection criterion -- single-author work is valid -- but readers should know whether any co-authors contributed to measurement verification.
   - **Severity**: Minor | **Evidence Anchor**: `absence: §4.4 (LQR baseline results) -- expected statement about who performed independent verification; checked author list, acknowledgments, and methods section` | **Confidence**: 3 -- this is a procedural concern, not a technical finding

### Detailed Comments

#### Assumption Audit

**Explicit assumptions:**
- MuJoCo contact model adequately represents rigid-ground wheel-terrain interaction. The paper acknowledges idealization ("stiff body contact idealizes a rigid ground") and estimates tire/ground compliance at 0.1-0.5mm on real surfaces. This is reasonable for initial validation.
- The 400Nm/s torque rate limit is a realistic representation of actuator dynamics. This is plausible for direct-drive or low-reduction actuators but would not hold for high-reduction harmonic drives with significant friction.
- Sensor noise can be modeled as additive Gaussian at specified levels. The paper tests this explicitly and shows where it breaks down (high noise causes EMA misclassification). This is a strength.

**Implicit assumptions:**
- The hardware-deployment path is incremental and straightforward (staged commissioning). The paper does not account for the fact that several control signals ($v_{\text{sag}}$, $F_z$, $z_{\text{contact}}$) require estimation from noisy sensors, and that estimation errors compound with the controller's demonstrated latency sensitivity.
- The MuJoCo solver configuration (Newton, 100 iterations, pyramidal cone, semi-implicit Euler) is assumed to provide sufficient fidelity for contact-rich maneuvers. While MuJoCo is widely used, the stiff body contact (`solref={0.002,1}`) may mask instabilities that would appear on compliant real surfaces, particularly for the terrain adaptation module.
- The 23-point height grid for posture PD gains implies that interpolation between grid points is adequate. The paper does not discuss whether the grid density was validated (e.g., testing at intermediate heights to confirm no interpolation artifacts).

**Paradigmatic assumptions:**
- The "controller as assembly of independently-tunable components with condition gates" paradigm assumes that components interact only through the gating mechanism, not through unmodeled dynamics. The paper acknowledges one such interaction (bidirectional coupling of $k_p$ schedule and anchor integral forms an implicit algebraic loop) but does not systematically analyze whether other component interactions could emerge on hardware with different compliance characteristics.
- The paradigm assumes that the regime classification (quiet stance vs. push recovery vs. flight) is exhaustive and mutually exclusive for the intended operating envelope. The compound-disturbance test partially addresses this but does not explore regime-boundary pathologies (e.g., what happens when the robot hovers near the gate transition band?).

#### Cross-Disciplinary Connections

**Parallel research:**
The paper correctly cites Astrom & Hagglund (2006) for selector-based and override control. However, it misses several deeper connections:

1. **Bumpless transfer** (Hanus et al., 1987; Peng et al., 1996): The smoothstep gate is functionally a bumpless transfer mechanism -- it ensures continuous transition between controller modes without transients. This is a well-studied problem in flight control (switching between manual and autopilot) and process control (switching between manual and automatic modes). The paper's smoothstep approach is a specific instance of "conditioned transfer" with a fixed blending function. This connection would strengthen the paper's theoretical grounding.

2. **Split-range control**: In process control, split-range controllers activate different actuators (e.g., heating vs. cooling valves) based on the controller output range. ACC's gating architecture is structurally similar -- different control components are activated in different "ranges" of the physical state space. The key difference is that split-range control activates by controller output magnitude, while ACC activates by physical state conditions. This distinction is worth making explicit.

3. **Gain-scheduled control with multiple scheduling variables**: The paper contrasts ACC with gain scheduling ("varies parameters continuously with one variable"), but modern gain scheduling (Rugh & Shamma, 2000, which is cited) includes multi-variable scheduling. ACC's $k_p$ schedule uses two scheduling variables (EMA and proximity) -- this is a standard gain-scheduling pattern, not a departure from it.

**Borrowing opportunities:**
- **Hysteresis design from industrial control**: The flight gate uses discrete hysteresis (2-step engage, 5-step release). Industrial process control has well-developed theory for hysteresis band design to prevent chatter (chattering in relay control). The paper could reference this for the flight gate design rationale.
- **Integral windup taxonomy**: Appendix A partially covers anti-windup connections but focuses on conditional integration. The broader taxonomy (Peng et al., 1996; Hippe, 2006) includes "realizable reference" and "observer-based" approaches that may offer alternatives to the proximity gate for platforms where position measurement is less reliable.

**Methodological borrowing:**
- **Describing function analysis**: The P-only limit cycle (55mm p-p) could be quantitatively analyzed using describing function methods (a standard nonlinear control technique) to predict the limit cycle amplitude from the dead-zone and integral bias. This would replace the qualitative statement that a "1.3 Nm equilibrium bias sustains the limit cycle" with a quantitative prediction.
- **Lyapunov-based region-of-attraction estimation**: The paper reports push survival thresholds empirically via binary search. Sum-of-squares or Lyapunov-based methods could provide theoretical bounds on the capture region, strengthening the contribution beyond empirical demonstration.

#### Practical Impact

**Real-world application:**
If ACC works on hardware as simulated, it provides a computationally cheap (no GPU, no neural network) controller for wheeled bipedal standing with push recovery. This has value for: (i) service robots that need to maintain precise position while tolerating occasional human contact, (ii) inspection robots on factory floors where sub-millimeter positioning is needed, and (iii) as a safety-controller layer beneath more complex locomotion planners.

**Implementation feasibility:**
The primary barrier is the latency sensitivity. As discussed in W2, the end-to-end latency budget is extremely tight. On the positive side, ACC's computational cost (~4.5ms on unspecified hardware) could be reduced further with optimized C/C++ implementation (the paper reports 200-300 FLOPs, which should execute in microseconds, not milliseconds, on any modern microcontroller). The 4.5ms figure suggests the measurement includes Python overhead or was measured on a non-real-time OS. A bare-metal implementation on a Teensy 4.0 or STM32H7 would likely achieve <100us computation time, which would recover most of the latency budget for sensor fusion.

**Stakeholders:**
The paper does not discuss who would deploy this controller or under what conditions. For a robot operating near humans, the S5 pitch gate's single-point-of-failure nature becomes a safety concern. If this controller were deployed on a service robot in a hospital or warehouse, what is the failure response when the pitch gyro drifts? Does the robot have a mechanical fall arrest or an independent safety controller? These considerations are out of scope for a methods paper but should be mentioned in limitations.

#### Broader Implications

**Ethical dimensions:**
The primary ethical concern is the S5 pitch gate as a safety-critical single point of failure. In process control, a 9x performance degradation from removing one component would trigger a mandatory safety integrity level (SIL) analysis. For a robot that could tip over near humans, the failure mode of the pitch gate (what happens when $g_\theta$ is stuck at 1 or 0?) should be analyzed. The paper mentions that the pitch gate thresholds are "conservative" but does not define what conservative means in a safety context.

A secondary concern: the paper positions ACC as a "precondition for RL" and a "height-general structured prior." If future RL policies are trained on top of ACC, the RL policy will inherit ACC's latency sensitivity and noise vulnerability. The paper does not discuss how this coupling might affect safety of learned policies.

**Social impact:**
Not applicable -- this is a technical methods paper without direct social implications beyond general robotics safety.

**Future directions:**
From a cross-disciplinary perspective, the most valuable follow-up would be:
1. A formal describing-function or Lyapunov analysis of the anchor mechanism to provide theoretical guarantees on the limit cycle elimination.
2. A comparison of ACC's smoothstep gates against classical bumpless transfer schemes (e.g., Hanus conditioning) to quantify whether the smoothstep provides any advantage beyond what standard bumpless transfer achieves.
3. Testing the gating architecture on a second platform (different morphology) to assess whether the design principles transfer or whether each platform requires de novo gate tuning.
4. A failure modes and effects analysis (FMEA) of the gating architecture, particularly the pitch gate, to inform hardware safety design.

### Cross-Disciplinary Reading Recommendations

- Hanus, R., Kinnaert, M., & Henrotte, J. L. (1987). "Conditioning technique, a general anti-windup and bumpless transfer method." *Automatica*, 23(6), 729-739. -- The foundational paper on bumpless transfer; directly relevant to ACC's smoothstep gate transitions.

- Peng, Y., Vrancic, D., & Hanus, R. (1996). "Anti-windup, bumpless, and conditioned transfer techniques for PID controllers." *IEEE Control Systems Magazine*, 16(4), 48-57. -- A survey covering the full taxonomy of anti-windup and bumpless transfer; would contextualize ACC's gate mechanism within established theory.

- Hippe, P. (2006). *Windup in Control: Its Effects and Their Prevention*. Springer. -- Comprehensive treatment of windup prevention; the observer-based anti-windup chapter is relevant to ACC's state-gated integral.

- Gelb, A., & Vander Velde, W. E. (1968). *Multiple-Input Describing Functions and Nonlinear System Design*. McGraw-Hill. -- Describing function analysis for nonlinear control systems; would enable quantitative prediction of the P-only limit cycle amplitude. `[UNVERIFIED]` -- search lead for describing function analysis of dead-zone + integral systems.

- Morari, M., & Zafiriou, E. (1989). *Robust Process Control*. Prentice Hall. -- The chapter on constraint control and selector blocks is the process-control foundation that ACC's architecture parallels; would provide theoretical language for the gating structure.

### Questions for Authors

1. Have you measured the end-to-end latency budget for your target hardware platform (IMU read + encoder read + complementary filter + ACC computation + torque command transmission + actuator electrical time constant)? If this exceeds 10ms, your own data (Table 4) suggests the robot would operate in the degraded 2.30mm / 47N regime permanently. How do you plan to close this gap?

2. You report 4.5ms computation time for 200-300 FLOPs. On a modern microcontroller (Cortex-M7 at 480MHz), 300 FLOPs should execute in <1us. Is the 4.5ms figure dominated by Python interpreter overhead or MuJoCo step time? What is the expected computation time on your target embedded platform?

3. The S5 ablation shows that removing the pitch gate reduces push survival from 80N to 9N. In process control, a component with 9x leverage would trigger a failure modes analysis. Have you analyzed: (a) what happens if the IMU gyro has a bias drift of 1-2 deg/s over temperature, causing the pitch rate threshold to be permanently exceeded? (b) What is the failure response -- does the robot fall, or is there a graceful degradation to P-only behavior?

4. Your gate-tuning procedure (§4.5) recommends setting $\tau_{\text{release}} \approx 3-5 \times T_r$. But $T_r$ is identified from "post-push free decay" -- which requires the robot to already be operating. How would you bootstrap the gate parameters for a new platform where you cannot yet perform push-recovery experiments? Is there a first-principles method based on the linearized pendulum dynamics?

5. You position ACC as a "precondition for RL." Given ACC's demonstrated latency sensitivity below 10ms, would an RL residual policy trained on top of ACC also inherit this sensitivity? Have you considered whether the RL policy could learn to compensate for latency by anticipating gate transitions?

### Minor Issues

- The paper uses "conditionally-gated" throughout but the abbreviation "ACC" loses the key term. Consider "CGC" (Conditionally-Gated Controller) for clarity.
- Table 3 footnote reports between-trial 95%CI as [0.50, 0.62]mm but the abstract and conclusion report [0.68, 0.78]mm. These appear to reference different quantities (within-trial vs. between-trial) but the distinction is not clearly explained.
- The paper states "RNG seeds are deterministic" but the seed formula `20260728 x 100 + rep` (with `x` as multiplication) uses a forward-date seed that suggests post-hoc selection. Use a fixed historical date or explain the choice.
- The LQR baseline standard deviations are "cross-episode aggregates; per-episode distributions are not logged" -- this is a data transparency limitation that should be elevated from limitation (10) to a more prominent position, as it prevents readers from assessing episode-to-episode variability.
- The phrase "bumpless transfer" never appears in the paper despite the smoothstep gate being precisely a bumpless transfer mechanism. Adding this term would connect ACC to a well-established control literature.
- The paper's "ACC tolerates moderate sensor noise" claim at 6.33mm idle is misleading -- 6.33mm is 9x worse than the headline 0.73mm figure and comparable to the P-only limit cycle. "Tolerates" overstates the case; "degrades to P-only performance levels" would be more accurate.
- The flight controller uses wheel-spin to pitch the nose upward during ledge descent, but the sign convention for forward wheel spin relative to pitch direction is not explicitly stated. A sign error here would cause the nose to pitch downward, worsening the fall. Clarify the sign relationship.
- The paper reports that ACC achieves 0.060 +/- 0.006 deg pitch RMS in Table 5 but 0.73mm CoM RMS in Table 2. These appear to be different runs (10 trials vs. 10 trials with different initialization). Clarify whether these figures come from the same dataset.