Here is the cross-disciplinary review.

---

## CROSS-DISCIPLINARY RESEARCHER REVIEW

**Reviewer role:** Control theory / ML / hardware deployment bridge, previously at a wheeled-robot startup, now in an academic sim-to-real lab. **Focus:** practical impact, hardware readiness, cross-disciplinary connections, broader implications.

---

### DIMENSION SCORES

| Dimension | Score (1-10) | Confidence (1-5) |
|-----------|-------------|-------------------|
| Technical soundness | 7 | 4 |
| Novelty | 6 | 4 |
| Experimental rigor | 8 | 5 |
| Clarity / framing | 7 | 5 |
| Hardware readiness | 4 | 4 |
| Sim-to-real credibility | 3 | 4 |
| Practical utility | 5 | 4 |
| Cross-disciplinary engagement | 5 | 4 |
| Generalizability | 4 | 3 |
| Overall recommendation | **Weak Accept (RA-L) / Major Revision (T-RO/TMECH)** | 4 |

---

### STRENGTHS

**S1. The asymmetric envelope follower is genuinely elegant.** Fast-attack (23ms) / slow-release (1.5s) as a time-domain solution to the estimation-speed-vs-noise trade-off is the right instinct. It sidesteps the frequency-domain ambiguity that plagues DOB/ADRC approaches (push and post-push ringdown occupy overlapping spectra) by attacking in the time domain where quiet stance and post-push oscillation are trivially separable by signal envelope. This is the kind of idea that makes control practitioners nod: cheap, intuitive, effective. The sensitivity analysis confirming $\tau_{\text{release}}$ dominates (sensitivity 220.4, 20x the next parameter) is the right way to validate a tuning claim.

**S2. Factorial ablation is the correct experimental design.** Additive (L0-L3) and subtractive (S0-S5) paths provide two-directional causal evidence. The Welch's t-test with Bonferroni correction is appropriate for $N=5$ independent trials. The $F_{\min}$ metric (worst direction) is a more honest push-recovery metric than $F_{\text{med}}$ or $F_{\max}$ alone. The author correctly notes which comparisons achieve significance and which don't (sub-2N differences non-significant at $N=5$).

**S3. Torque rate-limit invariance (200-800 Nm/s) is a strong result.** A controller that degrades gracefully across a 4x actuator bandwidth range has a real deployment advantage: you don't need premium actuators for acceptable performance.

**S4. Honest limitations section.** Eight self-identified limitations, explicit $N$ disclosure, the 10ms delay bifurcation acknowledged as fundamental rather than worked around, the EMA$_0=0$ initialization bias acknowledged and characterized. This is above-average transparency for a single-author paper.

**S5. The LQR failure analysis is more nuanced than it first appears.** Four LQR variants tested, including a direct-torque re-optimization that removes the PID servo path. The finding that direct-torque LQR also fails (100% fall within 0.76s) despite restored roll stability ($\phi_{\text{RMS}}=2.23^\circ$) is the paper's strongest comparative claim: it shows the structured prior matters, not just the direct-torque architecture. However, the 6-state coupled LQR gains were not re-derived for the DT plant (limitation 6 acknowledges this).

---

### WEAKNESSES

**W1. The "Path to Hardware" section is dangerously optimistic (Hardware Readiness).**

The end-to-end latency budget ($\approx 9.5$ms) with $0.5$ms margin below the 10ms bifurcation is a **red flag** for anyone who has deployed real-time control on physical hardware. The author acknowledges this margin is insufficient and suggests an RTOS + dedicated microcontroller can recover 2-4ms. But the real concerns are deeper:

- **IMU latency:** 1ms is optimistic for a consumer IMU. The ICM-20948 (common on research platforms) has a 1.125ms gyro group delay at 900Hz ODR, but bus readout (I2C at 400kHz or SPI) adds 0.5-1ms, and most software stacks poll at 2-5ms intervals. A realistic IMU pipeline is 3-5ms, not 1ms. The 10ms bifurcation means you are one SPI bus collision or Linux scheduler preemption away from degraded recovery. This is not a theoretical concern -- it is the dominant failure mode of every real-time balancing platform I've worked on.

- **$v_{\text{sag}}$ estimation:** The claim that a complementary filter (IMU + odometry) suffices is standard but glosses over the key failure mode: odometry slip during aggressive push recovery. When wheels lose traction, odometry diverges from true velocity, the complementary filter blends corrupted odometry with clean IMU, and the resulting $v_{\text{sag}}$ estimate is wrong precisely when the envelope follower most needs to be accurate. This is a classic "works in sim, breaks on hardware" pattern. A covariance-gated approach (increase IMU weight when slip is detected) would address this but is not proposed.

- **$F_z$ estimation via $\tau_{\text{wheel}} = I\dot{\omega} + rF_z$:** Standard and works in principle, but $\dot{\omega}$ estimation on real encoders (finite differences amplified at low speeds) is noisy. The 0.5mg flight threshold corresponds to $\sim$40N on this 8.1kg robot. A noisy $\dot{\omega}$ estimate with 0.02 kg-m$^2$ wheel armature means $\pm$0.5 rad/s$^2$ angular acceleration noise translates to $\pm$0.01Nm torque noise, or $\pm$0.17N force noise -- fine in principle. But the real issue is **contact detection hysteresis on compliant surfaces**: on carpet, foam, or grass, the "contact" transition is gradual (not the step function MuJoCo's stiff contact provides), and the binary 0.5mg threshold with hysteresis may chatter or misclassify. The author acknowledges contact-height degradation on compliant surfaces; the same applies to contact detection itself.

- **EMA$_0=0$ initialization:** The settle-time characterization acknowledges $\sim$7.5s of bias. For hardware, EMA$_0=1.0$ is recommended. But the $\sim$7.5s "warm start" during which precision is 0.42mm (not 0.73mm) is the headline settling-time claim, and a settle-time sweep with EMA$_0=1.0$ is **deferred** -- this is a gap. The hardware operator will experience worse precision during the first 5-10 seconds after startup; whether this matters depends on whether 0.42mm vs 0.73mm is distinguishable in hardware noise (probably not at 1-3mm projected degradation, but unmeasured).

- **Staged commissioning plan:** Static torque --> fixed-support standing --> free standing --> push recovery --> flight (tethered) --> terrain. This is the right sequence. But the paper provides no intermediate pass/fail criteria for each stage, no instrumentation plan (how do you measure 0.73mm CoM RMS on hardware?), and no fallback strategies. A documented commissioning plan needs more than a one-sentence enumeration.

**Scoring the Path to Hardware: 3/10.** It names the right issues but underestimates their severity. The latency budget in particular reads like a simulation paper's afterthought rather than a deployment-tested claim. If the author is serious about hardware, the 10ms bifurcation needs to be characterized with realistic timing jitter (not just constant delay), and the latency budget needs to be re-derived from the IMU datasheet of a specific target platform.

**W2. The 0.73mm precision claim is contingent on idealized contact (Sim-to-Real).**

The author honestly reports that softening `solref[0]` from 0.002 to 0.02 (tire-on-asphalt) increases idle RMS ~60% to ~1.2mm (line 112-113). This is good -- but it means the **sub-millimeter** headline is conditional on stiff ground contact, and the paper's central quantitative claim (the number in the abstract: $0.73\pm0.07$mm) degrades 60% under realistic tire compliance alone, before we even add encoder quantization, backlash, stiction, and IMU noise. The projected 1-3mm hardware range is credible, but then the claim is **millimeter**, not sub-millimeter. The 53x improvement over P-only would be $\sim$30x under realistic tire compliance -- still meaningful, but the headline precision metric is simulation-specific.

The stiff solref is not a methodological flaw (every simulator makes idealized contact choices), but it means the quantitative precision claim should carry a caveat earlier than Section V. The abstract's "$0.73\pm0.07$mm CoM RMS" should ideally read "$0.73\pm0.07$mm (simulation, stiff ground)" or the tire-degraded figure should be the primary claim.

**W3. The cross-disciplinary connections are hinted at but underdeveloped.**

The author mentions envelope-detector circuits (Astrom & Hagglund, 2006), split-range/override control (Shinskey, 1996), and hybrid/switched systems (Liberzon, 2003). These are the right touchpoints. But deeper connections are unexplored:

- **Audio engineering:** The asymmetric envelope follower with fast-attack/slow-release is functionally identical to an **audio compressor's envelope detector**. In fact, the $\tau_{\text{attack}}=23$ms / $\tau_{\text{release}}=1.5$s ratio of ~65x is within the range of broadcast audio compressors (where attack is 1-50ms and release is 100ms-3s). The "bistable symmetric EMA" failure mode the author discovered through ablation is the exact same problem that drove audio engineers to asymmetric envelopes decades ago: a symmetric smoother oscillates at half the signal frequency, creating a "pumping" artifact. The author independently rediscovered a principle well-known in audio DSP. Citing this connection would strengthen the paper significantly: the asymmetric envelope is not an ad-hoc trick, it's a principle with decades of engineering validation in a different domain. Key references: Giannoulis et al., "Digital Dynamic Range Compressor Design -- A Tutorial and Analysis," JAES 2012; Zolzer, "DAFX: Digital Audio Effects," Wiley 2011.

- **Proximity-gated integral as describing function analysis:** The author asks whether this is a dead-zone nonlinearity with hysteresis. It is more precisely a **state-dependent switching surface with smoothstep-softened boundary** -- which falls under hybrid/switched systems (Liberzon, already cited). The proximity gate alone is a dead-zone in $\Delta x$ (integral active for $|\Delta x| < 0.15m$, zero outside). But the envelope gate makes it **input-history-dependent**: the integral is active only when $|\Delta x|$ has been small AND velocity has been quiet for $\sim$1.5s. This is a dynamic constraint, not a static dead-zone. Describing function analysis could characterize the equivalent gain of the composite gate $g_{\text{prox}} \cdot g_{\text{env}}$ as a function of oscillation amplitude and frequency, providing a pseudo-linear stability prediction for the limit cycle. The author's empirical finding that symmetric EMA is bistable (a limit cycle sustained by half-open gate) is exactly the kind of phenomenon describing functions predict.

- **Smoothstep blending as gain scheduling:** The author explicitly distinguishes smoothstep from gain scheduling (Section III-A). The distinction is defensible: gain scheduling varies parameters continuously with a single scheduling variable; ACC's gates respond to qualitatively distinct physical regimes (near/far, quiet/oscillating, contact/airborne). But the boundary is blurry. Smoothstep IS a continuous blending function, and when it modulates integral gain as a function of $\Delta x$ and EMA($|v|$), it is functionally gain scheduling the integral term by two scheduling variables. The author could strengthen the argument by noting that the scheduling variables are **regime-detecting** rather than continuous state-tracking, which is a qualitative difference from standard gain scheduling -- but needs to acknowledge the functional similarity.

- **Neuroscience:** The fast-attack/slow-release envelope detector has a loose analog in **short-term synaptic plasticity**: fast neurotransmitter depletion (attack) followed by slow vesicle replenishment (release). Similarly, the proximity gate resembles **spatial attention mechanisms**. These are tenuous connections and I would not recommend forcing them, but the broader point stands: asymmetric envelope detection is a ubiquity across engineered and biological systems that deal with transient-vs-steady-state discrimination.

**W4. Practical utility question: Who uses this and why? (Practical Utility)**

The paper positions ACC as a structured prior for residual RL (line 66: "ACC is thus hypothesized as a precondition for RL on this platform"). This is the most honest framing in the paper and the strongest practical argument. But it creates a tension:

- **As a standalone controller:** ACC solves quiet standing + push recovery on a specific morphology with a 50+ parameter hand-tuned controller. The tuning guide (identify $T_r$, set $\tau_{\text{release}} \approx 3-5 \times T_r$, etc.) claims portability but has been tested on exactly one robot. A control engineer wanting to deploy this on a different wheeled biped would need to re-tune all 50+ parameters, trace failure modes, and characterize the delay bifurcation for their specific hardware. This is standard for classical control but limits standalone impact.

- **As an RL prior:** The value proposition is clearer: ACC provides a height-general structured prior that covers the "easy" regime (quiet standing, moderate pushes), letting a residual policy focus on the "hard" regime (transitions, extreme pushes, terrain). The P-only baseline doing surprisingly well on push ($F_{\min}=69.7$N vs ACC's $83.1$N, only 16% gap) supports this: the PD core handles push recovery competently; the anchor adds precision, not push performance. A residual policy trained on top of the P-only core might close the precision gap through learning, potentially making ACC's anchor unnecessary. The ablation shows the anchor's main contribution is **precision** (53x), not push survival. If residual RL can learn the precision correction, ACC's value as a prior may be limited to providing a good initialization -- which the P-only controller already provides for push recovery.

- **Versus a well-tuned cascaded PID:** The author frames this as the P-only limit cycle (55mm) being the PID baseline. But a cascaded PID with a properly tuned integral term (not global, but gated by the same proximity condition) would likely achieve similar precision. ACC's innovation is the **envelope gate** (temporal discrimination), not the proximity gate (spatial discrimination) -- proximity-gated integration is standard conditional integration. The envelope gate is genuinely novel for this application, but whether it outperforms a simpler timer-based gate ("integral active only when $|v| < \epsilon$ for $> T_{\text{settle}}$") is an ablation the paper doesn't test. The symmetric EMA ablation confirms the envelope asymmetry matters, but comparing against a timer-based gate would determine whether the envelope mechanism is necessary or merely one sufficient implementation.

**W5. Energy efficiency and computational budget are absent (Missing Perspectives).**

- **Energy:** Torque RMS of $3.28$Nm (Table IV) is reported but not contextualized. This is impressively low -- for an 8.1kg robot with 30Nm actuators (knee/pitch: 150Nm), operating at ~2-11% of torque limits at idle is efficient. But standing energy compared to what? A locked-actuator standing posture would consume near-zero torque. The 3.28Nm represents the continuous cost of balance, but the paper doesn't report power consumption (torque $\times$ velocity) or compare against the P-only idle power consumption (which should be higher due to the limit cycle). For a hardware deployment, battery life during standing is a practical concern, and the anchor's energy cost vs. its precision benefit should be quantified.

- **Compute:** The paper doesn't report computational cost. ACC is essentially 50+ floating-point operations (a few PD terms, one EMA update, three smoothstep evaluations, a capped integral, and Jacobian-based posture PD). This should run in **microseconds** on any embedded MCU (STM32 at 168MHz would take maybe 50-100$\mu$s). For comparison, the PPO policy requires a neural network forward pass (potentially 10-100x more compute). The "without neural network training" claim implicitly gestures at this advantage but doesn't quantify it. An explicit compute budget ($<$0.1% of 10ms control cycle) would strengthen the embedded deployment case.

- **Scalability to locomotion:** The paper mentions "extending conditional activation to dynamic locomotion" as future work (line 697). But the anchor mechanism fundamentally depends on a **home position** ($\Delta x$ relative to latched home). For dynamic locomotion (walking, running), there is no static home -- position error is intended, not an error to correct. Extending ACC would require redefining the anchor to gate on **step-cycle phase** rather than spatial proximity, which is a fundamentally different mechanism. The paper should be explicit that the current ACC formulation does not extend to locomotion without architectural changes, and that the conditional-activation principle (regime-dependent gate structure) is the transferable concept, not the specific gates.

**W6. Safety interlock failure mode analysis is incomplete (Broader Implications).**

The pitch gate $g_\theta$ (2-12$^\circ$) is described as a "safety interlock" and S5 shows removing it collapses $F_{\min}$ to 9N. But the failure mode analysis only considers gate-stuck-at-1 and gate-stuck-at-0. Real hardware failures are more varied:

- **Sensor dropout:** If the IMU pitch reading freezes (common failure mode: I2C bus lockup with last-value hold), $g_\theta$ stays at its last computed value while the robot may be tilting. The gate doesn't close because it thinks $\theta$ hasn't changed. This is a **silent failure** -- the boost remains engaged while the robot falls, potentially amplifying oscillations (per S5's 56mm parametric oscillation mode).

- **Gate threshold invariance to robot state:** The pitch gate thresholds (2-12$^\circ$) are fixed. During a squat transition with commanded height change, the robot's pitch may naturally increase due to CoM shift. The fixed threshold doesn't account for commanded state, potentially triggering gate closure during normal operation.

- **Deployment near humans:** The paper doesn't discuss safety for human-proximal deployment, which is responsible given the pitch-safety-interlock framing. The robot's wheels at 20 rad/s (~2 m/s at 0.06m radius) can cause injury on contact. The fall-back safety behavior when $g_\theta$ closes (disengaging the boost) is better than the alternative (S5: 56mm parametric oscillation), but the robot still falls. A fall on an 8.1kg wheeled platform with spinning wheels is a hazard. The paper should note that ACC assumes a controlled environment and doesn't provide operational safety guarantees.

**W7. The PPO baseline comparison is methodologically problematic but improving.**

The current PPO baseline (100M steps, 85% fall rate over multi-height transitions) is a learning baseline. Prior versions of the paper were criticized for claiming this as a comprehensive RL comparison. The current framing is more careful: "a model-free PPO baseline (100M steps) suffers an 85% fall rate" and ACC is "hypothesized as a precondition for RL." This is honest. However, two concerns remain:

- **Domain randomization gap:** The PPO policy was trained in `BalanceEnv` (single-height standing) and evaluated on multi-height transitions. This is an **out-of-distribution generalization test**, not a fair capability comparison. A PPO policy trained with height curriculum or domain randomization would likely perform better. The 85% fall rate characterizes the generalization gap, not RL's ceiling -- and this distinction should be explicit in the main text, not just implied.

- **The residual RL hypothesis is untested:** The paper hypothesizes ACC as a residual RL prior but doesn't test it. This is explicitly future work, but it means the paper's strongest practical framing ("ACC enables residual RL") is unsupported by data, making the contribution scope narrower than the framing suggests.

---

### CROSS-DISCIPLINARY CONNECTIONS: DETAILED ANALYSIS

**1. Asymmetric envelope follower -- audio engineering parallel (strong connection)**

The ACC envelope follower is structurally identical to a **peak-detecting audio compressor sidechain**:
- Fast attack (23ms) captures transient onset (push)
- Slow release (1.5s) holds the envelope through ringdown decay
- The smoothstep gate maps the envelope to a gain reduction (audio) or integral enable (ACC)
- The bistable symmetric-EMA failure mode is the "pumping" artifact well-known in audio compression

This is not merely an analogy -- the signal processing structure is identical. The difference is the controlled variable (audio amplitude vs. robot velocity) and the actuated variable (gain vs. integral gate). The audio engineering literature on compressor design provides established theory for envelope parameter selection, distortion analysis, and stereo linking (analogous to multi-directional envelope coupling for ACC's 8-direction push). Citing this connection would give ACC's envelope design a principled foundation rather than appearing empirical.

Suggested references:
- D. Giannoulis, M. Massberg, J. D. Reiss, "Digital Dynamic Range Compressor Design -- A Tutorial and Analysis," JAES, 60(6):399-408, 2012.
- U. Zolzer, "DAFX: Digital Audio Effects," 2nd ed., Wiley, 2011, Ch. 4 (Dynamics Processing).

**2. Proximity-gated integral -- describing function analysis (moderate connection)**

The composite gate $g = g_{\text{prox}}(\Delta x) \cdot g_{\text{env}}(\text{EMA}(|v|))$ is a nonlinear, state-and-history-dependent gain on the integral term. Describing function analysis could:
- Predict the amplitude and frequency of the limit cycle when the gate is partially open (the "bistable" regime)
- Determine whether the smoothstep softening (from hard switch to $C^0/C^1$ transition) eliminates or merely reduces the equivalent describing function gain discontinuity
- Characterize the stability boundary in $(k_p, K_i, \tau_{\text{release}})$ space

The author's empirical findings (bistable symmetric EMA, the 56mm parametric oscillation when $g_\theta$ is removed) are exactly the phenomena describing function analysis predicts for nonlinear feedback with amplitude-dependent gain. Even a single-sinusoid describing function of the composite gate would provide theoretical grounding for the empirical gate sensitivity results.

**3. Smoothstep blending -- relationship to gain scheduling (weak connection, already partially addressed)**

The author's distinction is: gain scheduling varies parameters continuously with a continuous scheduling variable; ACC's gates respond to qualitatively distinct regimes. This is a defensible boundary but functionally fuzzy. A more precise characterization: ACC is **hybrid gain scheduling with soft boundary**, where the scheduling variables are regime-detecting (proximity to home for the integral, quietness for the boost) rather than operating-point-tracking. The soft boundary (smoothstep) makes the hybrid system practically continuous, but the underlying logic is discrete -- the gate is "on" near home and quiet, "off" otherwise, with a transition band.

The connection to Shinskey's split-range and override control is appropriate but could be strengthened: ACC's gates are **AND-composed** ($g_{\text{boost}} = g_{\text{prox}} \cdot g_{\text{env}} \cdot g_\theta$), whereas industrial override typically uses **MAX** or **MIN** selection among parallel controllers. The AND-composition makes the boost more conservative (requires all conditions), which is appropriate for a stability-critical term on an unstable platform.

---

### SUMMARY ASSESSMENT

This paper has a genuinely good idea at its core: asymmetric time-domain gating for conditional integral action on a wheeled biped. The envelope follower is elegant, the ablation is thorough, and the limitations are honest. The LQR and PPO comparisons, while imperfect, provide useful context.

The weaknesses are concentrated in the transition from simulation to reality. The 10ms latency bifurcation, characterized in simulation but not with realistic timing jitter, is the paper's most vulnerable claim. The 0.73mm precision, while rigorously measured in sim, is contingent on stiff contact and would benefit from being qualified earlier. The cross-disciplinary connections (audio compression, describing functions, hybrid systems) are underexploited -- the paper reinvents some wheels that other fields have already refined, and misses opportunities to ground its empirical findings in existing theory.

**Bottom line:** ACC is a clean, well-ablated classical controller that would benefit from (a) a hardware demonstration even on a benchtop, (b) testing the residual RL hypothesis it proposes, and (c) engaging the audio/process-control/hybrid-systems literatures to ground its envelope follower in established theory. As a simulation paper with a credible but risky path to hardware, it is suitable for RA-L or ICRA with minor revisions. For T-RO or TMECH, the absence of hardware validation and theoretical stability analysis are significant gaps.