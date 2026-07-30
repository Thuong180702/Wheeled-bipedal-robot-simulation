---

# Methodology Reviewer Report (R1)

## Reviewer Identity
Methodology / Statistical Validation reviewer. Research background: simulation-based controller validation, classical control theory (LQR, PID, AW), reinforcement learning (PPO), experimental design for robotics. I evaluate simulation fidelity, experimental design, statistical validity, baseline fairness, and reproducibility.

## Overall Recommendation: **MAJOR REVISION**

## Final Score: **68/100**

| Dimension | Score | Weight |
|-----------|-------|--------|
| Simulation Fidelity & Setup | 45/100 | 25% |
| Experimental Design & Statistics | 70/100 | 25% |
| Baseline Fairness | 55/100 | 25% |
| Reproducibility | 75/100 | 15% |
| Controller Validation & Claim Bounding | 85/100 | 10% |

## Confidence Score: **5/5**
I have inspected the paper and the corresponding codebase (commit f92640c) at the file level. My findings are anchored to specific lines in both the paper and the source code.

---

## CRITICAL FINDINGS (Blocking Acceptance)

### C1: Simulator configuration claims materially misrepresent the actual MuJoCo setup.

**Severity: CRITICAL. Confidence: 5/5.**

The paper (Section III, line 111) states:

> "MuJoCo solver: Newton, 100 iterations; cone: pyramidal (elliptic for wheels, μs=1.2, μk=0.8 body); armature: 0.008 kg·m² (wheels), 0.02 kg·m² (leg joints); contact stiffness/damping: solref={0.002,1} (body), {0.005,0.7} (wheels); integration: semi-implicit Euler."

The actual robot XML (`assets/robot/wheeled_biped_real.xml`, lines 38-41) specifies:

```xml
<option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"
        iterations="4" ls_iterations="8" cone="pyramidal">
```

Five discrepancies between paper claims and actual code:

| Parameter | Paper Claim | Actual XML | Impact |
|-----------|-------------|------------|--------|
| Integrator | Newton | implicitfast | implicitfast uses semi-implicit Euler per constraint, not Newton iterations. Paper also says "semi-implicit Euler" which is contradictory with "Newton." |
| Iterations | 100 | 4 | 25× overstatement. This is the solver iteration count, not physics substeps. |
| ls_iterations | Not mentioned | 8 | Linesearch iterations. |
| Cone | pyramidal (elliptic for wheels) | pyramidal only | No elliptic cone is configured for wheels or any body. |
| Friction | μs=1.2, μk=0.8 | tangential=1.2, torsional=0.005, rolling=0.0003 | MuJoCo uses a 3-parameter friction model (tangential, torsional, rolling). The paper incorrectly maps this to Coulomb μs/μk. |

The remaining claims (armature, solref/solimp values, timestep) are accurate against the XML.

**Fix:** Rewrite solver description to match actual XML. Either configure the claimed Newton/100/elliptic settings and re-run all experiments, or report the actual implicitfast/pyramidal/4-iteration configuration. The current text presents a higher-fidelity simulation than was actually used, which is a significant research integrity concern. If the implicitfast integrator with 4 iterations produces artifacts, the experimental results may not replicate at higher accuracy settings.

---

### C2: Control frequency claimed as 100Hz throughout the paper; code uses 50Hz.

**Severity: CRITICAL. Confidence: 5/5.**

The paper states "controller at 100Hz (Δ_ctrl = 0.01s)" in the abstract, Section III (line 111), and throughout all results. The claimed latency threshold of 10ms is exactly one control step at 100Hz -- the paper's central sensitivity finding hinges on this.

Code evidence:

- **`wheeled_biped/envs/base_env.py:61`**: `CONTROL_DT: float = 0.02  # 50Hz → 10 substeps`
- **`wheeled_biped/controllers/coupled_lqr_3d.py:58`**: `_CONTROL_DT = 0.02`
- **`wheeled_biped/controllers/fair_lqr_torque.py:68`**: `_CONTROL_DT = 0.02`
- **`wheeled_biped/sim/low_level_control.py:79`**: `control_dt: float = 0.02`
- **`configs/training/balance.yaml:108`**: "1 = ~20 ms delay (typical real hardware)"

However:

- **`scripts/push_ci.py:27`**: `DT = 0.01; SUBSTEPS = 5` (100Hz)
- **`scripts/robustness_sweep.py:30`**: `DT = 0.01` (100Hz)

The evaluation scripts run at 100Hz but the RL training environment and LQR baselines run at 50Hz. This means:
1. The delay "bifurcation" at 10ms is actually **half a control step** at 50Hz, not a full step at 100Hz as the paper implies.
2. The latency budget analysis (Section 6, "Path to Hardware") claiming 0.5ms margin below 10ms is based on wrong control period. At 50Hz, the budget is 20ms, making the claimed "critical latency sensitivity" far less constraining.
3. The EMA time constants (τ_attack≈23ms, τ_release≈1.5s) are computed at 100Hz but the training controller actually runs at 50Hz, potentially changing effective α.

**Fix:** Either (a) correct the paper to state 50Hz control throughout and re-derive all timing-dependent claims (delay thresholds, EMA α, latency budget), OR (b) change base_env.py to 100Hz, re-train PPO baseline, and re-run all experiments at 100Hz. Option (a) is more honest; option (b) would be required to preserve the 10ms narrative.

---

### C3: PPO baseline "100M steps" claim is unsubstantiated.

**Severity: CRITICAL. Confidence: 4/5.**

The paper claims a PPO baseline trained for "100M environment steps" (Table V, Section V-D). The only checkpoints in the repository are at `backup_checkpoints/Day1/` and `backup_checkpoints/Day2/`, with the highest being `step_5390336` (~5.4M steps). No 100M-step checkpoint exists. The balance.yaml config does not specify a `total_steps` parameter.

Additionally, the PPO baseline config (`balance.yaml`) specifies:
- `lin_vel_mode: "clean"` (simulator-exact velocity)
- `action_delay_steps: 0` (no latency)
- `push_magnitude: 0` (no push disturbances during training)
- `domain_randomization` with mass_range [0.9, 1.1] only

This means the PPO policy:
1. May not have been trained to the claimed 100M steps
2. Was trained in "sim prototyping" mode with clean sensors and zero delay
3. Was evaluated on a task (multi-height transitions, Table V) that it never experienced during training since push_magnitude=0 and the height curriculum was eval-gated

**Evidence anchor:** balance.yaml lines 176, 198-213; backup_checkpoints directory showing max step 5,390,336.

**Fix:** Provide the actual 100M-step checkpoint with training log/wandb curve, or reduce the claim to the maximum trained steps. Document what "100M steps" means: total environment steps (num_envs × rollout_length × num_updates) or just a training target. If the PPO baseline was not actually trained to completion, this must be stated explicitly.

---

## MAJOR FINDINGS (Require Substantial Revision)

### M1: Control rate discrepancy affects all timing-dependent results.

**Severity: MAJOR. Confidence: 5/5.**

Beyond C2, the control rate inconsistency cascades:

- The EMA coefficients in the paper (α_a≈0.35, α_r≈0.007) are derived at Δt=0.01s (100Hz). At the actual 50Hz used in base_env.py, the correct values would be different.
- The latency sensitivity claim of "<10ms threshold" is fundamentally tied to the control rate. At 50Hz, one step of delay is 20ms, not 10ms.
- The robustness table (Table VI) reports 10ms and 30ms delay conditions. These correspond to 0.5 and 1.5 control steps at 50Hz, or 1 and 3 steps at 100Hz. The eval scripts use 100Hz, so the delay_steps=1 means 10ms is correct for the eval runs -- but the baseline training runs at 50Hz.

**Evidence anchor:** base_env.py:60-61 vs scripts/robustness_sweep.py:30; paper line 111.

**Fix:** Same as C2 -- reconcile the control rate globally.

---

### M2: LQR baseline comparison mixes control architectures.

**Severity: MAJOR. Confidence: 4/5.**

The paper's central competitive claim is that "four classical LQR baselines all fail (100% fall rate)" vs ACC's success. However:

1. **4-state LQR** operates through a PID servo layer (~0.25s lag, per the paper's own measurement). ACC operates through direct torque. These are different control architectures competing under different effective bandwidth constraints.

2. **Direct Torque LQR** removes the servo path -- addressing the architecture unfairness -- but the paper acknowledges the gain matrix was "not re-derived for the DT plant" and reports "100% fall rate within 0.76s." A re-derived Q/R for the direct-torque plant (no servo lag, different effective plant dynamics) would produce different gains. The paper uses the same Q/R matrices from the velocity-commanded plant.

3. **Coupled 6-state LQR** adds roll to the state but retains the PID servo path. The paper states it performs "identically to the 4-state baseline" (0.52s) -- yet the code (`coupled_lqr_3d.py`) estimates "Indefinite survival at nominal height" in its docstring, contradicting the paper's reported result.

4. The **46-configuration sweep** (`sweep_lqr_roll_gains.py`) only sweeps kp_roll (7 values) and kd_roll (5 values) = 35 configurations, plus wheel_vel_limit variants. The sweep does not explore Q/R matrices, LQR re-derivation, or servo gain tuning. All LQR variants use the same pre-computed gains from the TWIP model.

5. The paper's claim that the "PID servo action path" is the "primary LQR bottleneck" (Abstract, line 51) is partially undercut by Direct Torque LQR's continued failure at 0.76s, but the paper does not clarify whether re-deriving Q/R for the direct-torque plant was attempted.

**Fix:** Either (a) re-derive LQR gains for the direct-torque plant with properly tuned Q/R, report the best possible LQR result, and compare fairly, OR (b) explicitly state that the LQR comparison is conservative (under-tuned), acknowledge that retuned LQR might survive longer, and bound claims accordingly. The current framing presents LQR failure as a definitive finding when the comparison is architecturally asymmetric.

---

### M3: N=5 statistical power limits the ablation conclusions.

**Severity: MAJOR. Confidence: 4/5.**

The paper honestly acknowledges the power limitation (Limitation 8). However, the ablation study is Contribution 3 and serves as primary evidence for component contributions. At N=5:

- Only L0→L1 (+5.5N, p<0.001) and L2→L3 (+6.8N, p<0.001) survive Bonferroni correction at α=0.007
- L1→L2 (+1.1N, n.s.) cannot distinguish the integral from noise
- S1→S2, S1→S3, S1→S4 all have sub-2N differences that are not statistically distinguishable
- S5 (pitch gate removal) collapse is unambiguous (83N→9N) and requires no formal test

The fine-grained subtractive ablation conclusions (e.g., "S3+S2 required for ringdown," "boost is the dominant precision mechanism") rely on binary outcomes (ringdown decays/never-decays) which are partly qualitative rather than quantitative. This is acceptable for the ringdown claims, but the push-survival delta claims for S2-S4 should be qualified.

**Statistical methodology check:** Welch's t-test is appropriate for unequal variances at small N. Bonferroni correction with α=0.007 for 7 comparisons is correct. However, the paper computes CI half-width as "~1.24 × std" using t_{0.025,4} for single-group CI, while between-group comparison CIs (S1 vs S3) would use different df via Welch-Satterthwaite. This is acknowledged in the footnotes and is correct.

**Fix:** Increase N to at least 10 for the subtractive ablation (S2-S5), or restructure the narrative to focus on the two unambiguous findings (L0→L1, L2→L3, S5 collapse) and present S2-S4 as exploratory trends.

---

## MODERATE FINDINGS (Should Be Addressed)

### Md1: The implicitfast integrator with 4 iterations may affect contact dynamics.

**Severity: MODERATE. Confidence: 3/5.**

MuJoCo's `implicitfast` integrator with 4 iterations relaxes the contact complementarity conditions. For stiff contacts (solref time-constant = 0.002s), 4 iterations may not fully resolve contact forces during impacts. This could affect:
- Push impulse transmission fidelity
- Flight landing dynamics
- Contact-point estimation accuracy for terrain adaptation

Newton solver with 100 iterations (as claimed) would produce more accurate contact resolution. The actual 4-iteration implicitfast may over- or under-estimate contact stiffness, potentially inflating idle precision and altering push-recovery dynamics. This is a known MuJoCo artifact: low iteration counts trade accuracy for speed.

**Evidence anchor:** wheeled_biped_real.xml lines 38-39; MuJoCo documentation on solver iterations.

**Fix:** Either increase iterations and re-run key benchmarks (idle precision, push force thresholds) to confirm invariance, or document the 4-iteration limit as a known limitation with estimated error bounds.

---

### Md2: EMA initialization transient is incompletely characterized.

**Severity: MODERATE. Confidence: 4/5.**

The paper documents that EMA₀=0 biases g_env≈1 for ~7.5s (Section IV-C). However:
- Table IV (Standing) uses 5s settle time, which is within the 7.5s transient
- Table VI (Robustness) uses 3s settle, producing 0.42mm (vs 0.73mm in Table IV)
- The paper attributes this difference to EMA₀=0 bias but does not run a settle-time sweep
- The "noise floor" claim of 0.002mm CoM X RMS (SNR 289×) may be contaminated by the EMA transient

The idle precision measurement is therefore sensitive to the settle duration in a way that interacts with the gate dynamics. The conservative figure (0.73mm at 5s settle) is appropriate, but without a full settle-time characterization, readers cannot determine the true steady-state precision.

**Evidence anchor:** Table IV caption, Table VI caption, paper lines 236-238.

**Fix:** Run a settle-time sweep (1s, 3s, 5s, 10s, 15s) at EMA₀=0 and EMA₀=1.0, report the asymptotic value. Confirm that 0.73mm is conservative.

---

### Md3: The "53× improvement" claim requires context.

**Severity: MODERATE. Confidence: 5/5.**

The paper reports "~53× idle precision improvement" (0.73mm ACC vs 39mm P-only RMS). This is numerically correct (39/0.73 ≈ 53.4) but compares qualitatively different operating modes:
- P-only: perpetual limit cycle oscillation (55mm p-p, RMS~39mm) -- fundamentally unstable equilibrium
- ACC: actively stabilized equilibrium with anchor integral

The 53× ratio conflates "elimination of a limit cycle" with "precision improvement." A limit cycle is not a precision benchmark; it is a failure mode. The improvement should be framed as "elimination of the 55mm limit cycle, achieving 0.73mm RMS" rather than a single multiplicative ratio.

**Evidence anchor:** Table II, Abstract.

**Fix:** Reframe as "ACC eliminates the 55mm p-p limit cycle observed in P-only control and achieves 0.73mm CoM RMS idle precision." Retain the 53× figure only as a parenthetical or supplementary metric with appropriate qualification.

---

### Md4: "Flight recovery and per-leg terrain adaptation as independent extensions" -- contribution framing issue.

**Severity: MODERATE. Confidence: 4/5.**

Contribution 2 claims flight recovery and terrain adaptation as "independent extensions." The code implementation confirms they ARE independent in the sense that they do not modify the core anchor. However:

1. Flight recovery (Section IV-E) is a 3-term attitude PD with discrete hysteresis -- a standard building block, not a novel contribution.
2. Terrain adaptation (Section IV-F) is a 3-line height-split with contact-point estimation -- again, standard.
3. Both are presented at N=10 with 10/10 survival, but the flight test is a drop/ledge (not in-situ push recovery during flight) and terrain is a fixed curb straddle (not dynamic terrain traversal).

Presenting these as a standalone contribution (#2 of 3) inflates their significance relative to the core anchor mechanism.

**Fix:** Either (a) demote flight/terrain to an "Extensions" section and reduce contribution count to 2, or (b) demonstrate more demanding flight/terrain scenarios (push recovery during flight, dynamic terrain with varying curb heights).

---

### Md5: The robot's world-frame orientation is not stated clearly enough for sign-convention audit.

**Severity: MODERATE. Confidence: 3/5.**

The XML (line 26-29) documents: "robot faces -Y direction." The lqr_balance.py header documents: "pitch_x_rad: Body pitch angle (rad), cp_error_y_m: Capture point error in forward direction (m)." However, the paper does not state the world-frame convention in the main text (only in code). The sign convention for pitch (nose-up positive or negative?) affects the interpretation of the balance core PD signs and LQR gain signs. A sign error in the LQR gain derivation could explain some of the baseline failure.

This is not currently a claim-invalidating issue, but it makes independent verification of controller correctness difficult.

**Fix:** Add one sentence to Section III stating the world-frame convention (robot faces -Y, X left, Z up) and all sign conventions for pitch, roll, yaw, forward velocity, and wheel torques.

---

## MINOR FINDINGS

### m1: Armature default value discrepancy.

Paper claims "armature: 0.008 kg·m² (wheels), 0.02 kg·m² (leg joints)." The XML shows:
- Default armature: 0.01 (line 49)
- Leg joints: 0.02 (matches paper, lines 157, 170, 182, 194, 228, 241, 253, 265)
- Wheels: 0.008 (matches paper, lines 206, 277)

The paper omits the default armature of 0.01 and only reports the overridden values. Minor, but the default value could affect any joints not explicitly listed.

### m2: Reference [7] (roscia2023flight) is cited as "Sensors, vol. 23, no. 3, 1234, 2023" -- the journal name is inconsistent with its official IEEE citation format.

### m3: The "non-monotonic noise behavior" is explained (EMA disengagement) but the explanation for high noise improving to 3mm from 34mm medium noise is incomplete. The paper says "high noise holds EMA above threshold, disengaging the anchor into P-only mode" -- but at 3mm RMS, P-only mode would exhibit the 39mm limit cycle. The 3mm figure requires additional explanation.

### m4: The "between-trial CV 10.0%" reported in Table IV applies to idle RMS across N=10 trials. This is acceptably low for a deterministic controller with randomized initial conditions, but the paper should clarify whether this is the coefficient of variation of the CoM RMS metric itself or of some underlying per-trial statistic.

---

## SUMMARY OF REQUIRED CHANGES

### Blocking (must fix before resubmission):
1. **Fix simulation configuration description** (C1): Report actual solver settings (implicitfast, 4 iterations, pyramidal cone) or configure to match claims and re-run experiments.
2. **Reconcile control frequency** (C2): Either correct paper to 50Hz throughout or change code to 100Hz and re-run. This affects latency claims, EMA parameters, and the hardware transfer discussion.
3. **Substantiate PPO baseline training** (C3): Provide evidence of 100M-step training or reduce the claim.

### Major (substantial revision required):
4. Fairer LQR comparison with re-derived DT-plant gains (M2).
5. Increase N for subtractive ablation or narrow claims (M3).
6. Document integrator/iteration sensitivity (Md1).
7. Characterize EMA settle-time asymptotic (Md2).
8. Reframe 53× claim (Md3).
9. Reconsider flight/terrain contribution framing (Md4).
10. Add sign convention documentation in paper (Md5).

### Minor (at authors' discretion):
11. Report default armature 0.01 (m1).
12. Fix reference formatting (m2).
13. Clarify non-monotonic noise behavior (m3).
14. Clarify between-trial CV computation (m4).

---

## STRENGTHS (Acknowledged)

1. **Factorial ablation design** (L0→L3 additive + S0→S5 subtractive) is methodologically strong and exceeds field norms for simulation-based robotics papers.
2. **Statistical transparency** (Welch's t-test, Bonferroni, Cohen's d, Clopper-Pearson CIs, explicit power limitations) is commendable and builds trust.
3. **Gate sensitivity analysis** providing transferable tuning heuristics (τ_release ~3-5× T_r) makes the contribution actionable -- this is rare and valuable.
4. **S5 pitch-gate finding** (removing g_θ collapses F_min from 83N to 9N, producing parametric oscillation) is a genuinely insightful result demonstrating the architecture catches interactions a simpler design would miss.
5. **Hardware path discussion** is among the most detailed in the simulation-only wheeled-biped literature, with specific estimation methods, latency budgets, and staged commissioning.
6. **Acknowledgment of 10 explicit limitations** demonstrates intellectual honesty that is uncommon and builds significant reader trust.

---

## Recommended Decision Justification

The paper presents a genuinely novel mechanism (proximity-gated anchor with asymmetric envelope follower) with strong ablation evidence. The S5 pitch-gate finding alone is a contribution that advances understanding of gated control for wheeled bipeds.

However, three critical issues prevent acceptance: (C1) the simulation setup description misrepresents the actual configuration in ways that could affect result validity; (C2) the control frequency discrepancy cascades into multiple claims (latency sensitivity, EMA parameters, hardware transfer); and (C3) the PPO baseline training depth is unsubstantiated.

These are bounded, technically fixable issues. None requires new theoretical development. The core contribution -- the anchor mechanism -- survives the audit. With the critical fixes applied (primarily honest documentation of the actual simulation configuration and control rate), the paper would meet RA-L standards for methodology. The score of 68/100 reflects the penalty for these critical documentation gaps relative to an otherwise well-executed study.