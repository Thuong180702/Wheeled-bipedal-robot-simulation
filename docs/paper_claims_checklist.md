# Paper Claims Checklist

**Date:** 2026-05-06  
**Purpose:** Distinguish safe claims (backed by code/tests/data) from unsafe claims (unvalidated/aspirational)  
**Status:** Use this checklist during paper rewrite to avoid fabricated results

---

## ✅ SAFE CLAIMS (Backed by Evidence)

### Architecture and Implementation

- [x] **10-actuated wheeled biped robot** — model exists in `scene.xml`
- [x] **5 joints per leg: hip roll, hip yaw, hip pitch, knee, wheel** — verified in model
- [x] **MuJoCo MJX simulation** — codebase uses MJX throughout
- [x] **JAX/PPO training framework** — implemented in `wheeled_biped/training/ppo.py`
- [x] **Bounded residual RL architecture** — implemented in Phase C
- [x] **Height-dependent LQR/IK prior** — implemented in `wheeled_biped/controllers/lqr_ik_prior.py`
- [x] **Residual action composition: `final = clip(base + scale * residual, -1, 1)`** — implemented in `action_codec.py`
- [x] **52-dimensional residual observation (42 base + 10 base_action)** — implemented in `ResidualBalanceEnv`
- [x] **Per-joint residual scale vector** — implemented and configurable
- [x] **Low-level PID control** — implemented in `wheeled_biped/sim/low_level_control.py`
- [x] **Action smoothing and delay** — implemented in base env
- [x] **Curriculum learning** — implemented in `wheeled_biped/training/curriculum.py`
- [x] **Eval-gated curriculum promotion** — implemented and tested

### Testing and Validation

- [x] **Action codec tests pass** — 31/31 tests passing
- [x] **LQR/IK prior tests pass** — 14/14 tests passing
- [x] **Residual env tests pass** — 9/9 tests passing
- [x] **Checkpoint validation tests pass** — 7/7 tests passing
- [x] **Base env tests pass** — 9/9 tests passing
- [x] **Total: 56/56 tests passing** — verified 2026-05-06
- [x] **LQR sign convention validated** — pitch/velocity tests pass
- [x] **Height IK monotonicity validated** — tested in Phase B.2
- [x] **Joint limit compliance validated** — tested in Phase B.2
- [x] **Residual composition correctness validated** — zero residual returns base action
- [x] **Action clipping validated** — tested in `test_action_codec.py`

### Prior Baseline Results (Phase B.5)

- [x] **LQR/IK prior evaluated on fixed-height sweep** — see `docs/phase_b5_classical_prior_variants_report.md`
- [x] **Geometric LQR/IK variant tested** — 85-100% fall rates across heights
- [x] **Prior is structured but insufficient standalone** — quantified in Phase B.5
- [x] **Prior provides nominal balance behavior** — validated but limited
- [x] **Fixed-height baseline table exists** — Table IV data available

### Code Infrastructure

- [x] **Residual-aware checkpoint metadata** — implemented in Phase C
- [x] **Checkpoint type detection** — residual vs pure PPO
- [x] **Action breakdown logging** — base/residual/final components
- [x] **Residual metrics logging** — norm, saturation, rate
- [x] **Training config for balance_residual** — `configs/training/balance_residual.yaml` exists
- [x] **Experiment plan documented** — `docs/experiment_plan.md` created
- [x] **Results checklist documented** — `docs/results_todo_checklist.md` created

---

## ⚠️ UNSAFE CLAIMS (Not Yet Validated)

### Training Results (Require Experiments)

- [ ] **Residual PPO achieves >80% survival rate** — NO DATA YET
- [ ] **Height RMSE <0.03m** — NO DATA YET
- [ ] **Push recovery >100N** — NO DATA YET
- [ ] **Training converges in 50M steps** — NO DATA YET
- [ ] **3-seed reproducibility** — NO DATA YET
- [ ] **Curriculum advances through all stages** — NO DATA YET
- [ ] **Residual magnitude stays bounded** — NO DATA YET
- [ ] **Residual saturation rate <X%** — NO DATA YET

### Evaluation Results (Require Experiments)

- [ ] **Random-height balance performance** — NO DATA YET
- [ ] **Fixed-height sweep results** — NO DATA YET (except LQR/IK baseline)
- [ ] **Height transition performance** — NO DATA YET
- [ ] **Push recovery performance** — NO DATA YET
- [ ] **Robustness to model uncertainty** — NO DATA YET
- [ ] **Residual action analysis** — NO DATA YET
- [ ] **Ablation study results** — NO DATA YET
- [ ] **Pure PPO baseline comparison** — NO DATA YET

### Comparative Claims (Require Fair Baselines)

- [ ] **Outperforms pure PPO** — NO DATA YET
- [ ] **Outperforms LQR/IK alone** — PARTIAL (Phase B.5 shows LQR/IK limitations, but no residual PPO data)
- [ ] **Residual improves over base prior** — NO DATA YET
- [ ] **Better sample efficiency than pure PPO** — NO DATA YET
- [ ] **More robust than pure PPO** — NO DATA YET

### Generalization Claims (Require Evaluation)

- [ ] **Generalizes across height range** — NO DATA YET
- [ ] **Recovers from large pushes** — NO DATA YET
- [ ] **Robust to friction variation** — NO DATA YET
- [ ] **Robust to mass perturbations** — NO DATA YET
- [ ] **Robust to sensor noise** — NO DATA YET
- [ ] **Robust to action delay** — NO DATA YET

### Hardware Claims (Not Validated)

- [ ] **Sim-to-real transfer** — NOT VALIDATED
- [ ] **Hardware deployment** — NOT VALIDATED
- [ ] **Real-world push recovery** — NOT VALIDATED
- [ ] **Real-world height transitions** — NOT VALIDATED
- [ ] **Hardware state estimation** — NOT VALIDATED
- [ ] **Hardware safety** — NOT VALIDATED

### Novelty Claims (Require Literature Review)

- [ ] **First residual RL for wheeled bipeds** — NOT VERIFIED
- [ ] **First LQR/RL hybrid for wheeled robots** — NOT VERIFIED
- [ ] **Novel bounded residual architecture** — NOT VERIFIED
- [ ] **Novel height-dependent prior** — NOT VERIFIED

### Future Work (Not Implemented)

- [ ] **Stand-up recovery** — NOT IMPLEMENTED
- [ ] **Wheeled locomotion** — NOT IMPLEMENTED
- [ ] **Walking** — NOT IMPLEMENTED
- [ ] **Stair climbing** — NOT IMPLEMENTED
- [ ] **Rough terrain** — NOT IMPLEMENTED
- [ ] **Dynamic transitions** — NOT IMPLEMENTED

---

## 🔴 FORBIDDEN CLAIMS (Never Make These)

### Fabricated Results

- **NEVER** claim specific numerical results without data
- **NEVER** claim "achieves X%" without running experiments
- **NEVER** claim "outperforms Y" without fair comparison
- **NEVER** claim "robust to Z" without robustness evaluation
- **NEVER** claim "generalizes to W" without generalization tests

### Overstated Capabilities

- **NEVER** claim hardware validation without hardware tests
- **NEVER** claim sim-to-real success without real robot
- **NEVER** claim safety without safety analysis
- **NEVER** claim real-time performance without timing benchmarks
- **NEVER** claim energy efficiency without energy measurements

### Unsupported Novelty

- **NEVER** claim "first" without literature review
- **NEVER** claim "novel" without prior art search
- **NEVER** claim "state-of-the-art" without benchmarks
- **NEVER** claim "breakthrough" without justification

### Unimplemented Features

- **NEVER** claim stand-up recovery works without training/eval
- **NEVER** claim locomotion works without training/eval
- **NEVER** claim walking works without training/eval
- **NEVER** claim stair climbing works without training/eval
- **NEVER** claim rough terrain works without training/eval

### Misleading Comparisons

- **NEVER** compare against strawman baselines
- **NEVER** compare against unfairly tuned baselines
- **NEVER** compare against different training budgets
- **NEVER** compare against different evaluation scenarios
- **NEVER** cherry-pick favorable comparisons

---

## 📝 SAFE PHRASING GUIDELINES

### For Unvalidated Results

Instead of:
- ❌ "Our method achieves 95% success rate"
- ❌ "The robot recovers from 150N pushes"
- ❌ "Training converges in 50M steps"

Use:
- ✅ "We evaluate success rate across 3 seeds (results in Section V)"
- ✅ "We test push recovery up to 200N (results in Section V-D)"
- ✅ "We train for 50M steps (convergence analysis in Section V-A)"

### For Future Work

Instead of:
- ❌ "Our method enables stand-up recovery"
- ❌ "The framework supports locomotion"
- ❌ "We demonstrate stair climbing"

Use:
- ✅ "Stand-up recovery is a natural extension (future work)"
- ✅ "The framework is designed to support locomotion tasks"
- ✅ "Stair climbing remains an open challenge"

### For Hardware

Instead of:
- ❌ "We validate on hardware"
- ❌ "Sim-to-real transfer succeeds"
- ❌ "The robot operates safely"

Use:
- ✅ "Hardware validation is planned future work"
- ✅ "Sim-to-real transfer is an important next step"
- ✅ "Safety analysis is beyond the scope of this work"

### For Comparisons

Instead of:
- ❌ "Our method outperforms pure PPO"
- ❌ "We achieve better sample efficiency"
- ❌ "Our approach is more robust"

Use:
- ✅ "We compare against pure PPO baseline (Section V-H)"
- ✅ "We analyze sample efficiency (Section V-A)"
- ✅ "We evaluate robustness (Section V-F)"

### For Novelty

Instead of:
- ❌ "We propose the first residual RL method for wheeled bipeds"
- ❌ "Our approach is novel"
- ❌ "We achieve state-of-the-art performance"

Use:
- ✅ "We propose a bounded residual RL approach for wheeled bipeds"
- ✅ "Our approach combines LQR/IK priors with residual RL"
- ✅ "We evaluate performance on challenging scenarios"

---

## 🎯 PAPER STRUCTURE GUIDANCE

### Abstract

**SAFE:**
- Describe the problem (height-adaptive balance for wheeled bipeds)
- Describe the approach (bounded residual RL over LQR/IK prior)
- Describe the evaluation (scenarios tested, metrics used)
- Use "we evaluate", "we test", "we analyze" instead of "we achieve"

**UNSAFE:**
- Specific numerical results without data
- Claims of superiority without comparison
- Claims of novelty without verification
- Claims of hardware success without validation

### Introduction

**SAFE:**
- Motivate the problem
- Describe the challenges
- Outline the proposed approach
- Preview the evaluation plan
- State contributions clearly

**UNSAFE:**
- Claim results before presenting them
- Overstate novelty
- Promise unimplemented features
- Claim hardware readiness

### Method

**SAFE:**
- Describe architecture in detail
- Explain design choices
- Show equations and algorithms
- Reference implementation details
- Discuss limitations

**UNSAFE:**
- Claim performance without data
- Claim optimality without proof
- Claim generality without testing
- Hide important assumptions

### Results

**SAFE:**
- Present data with error bars (mean ± std over 3 seeds)
- Show training curves
- Show evaluation metrics
- Compare against fair baselines
- Discuss failure cases
- Use TODO placeholders for missing data

**UNSAFE:**
- Fabricate numbers
- Cherry-pick results
- Hide negative results
- Compare against unfair baselines
- Claim success without metrics

### Discussion

**SAFE:**
- Analyze results honestly
- Discuss limitations
- Compare to related work
- Suggest future directions
- Acknowledge assumptions

**UNSAFE:**
- Overstate contributions
- Ignore limitations
- Dismiss related work
- Promise unvalidated extensions
- Claim hardware readiness

### Conclusion

**SAFE:**
- Summarize contributions
- Restate key findings (with data)
- Acknowledge limitations
- Suggest future work

**UNSAFE:**
- Claim results without data
- Overstate impact
- Promise unvalidated features
- Claim hardware success

---

## ✅ PRE-SUBMISSION CHECKLIST

Before submitting the paper, verify:

- [ ] Every numerical claim has corresponding data in results section
- [ ] Every "achieves X%" claim has experimental evidence
- [ ] Every "outperforms Y" claim has fair comparison
- [ ] Every "robust to Z" claim has robustness evaluation
- [ ] No hardware claims without hardware validation
- [ ] No sim-to-real claims without real robot
- [ ] No stand-up/locomotion claims without training/eval
- [ ] No "first" or "novel" claims without literature review
- [ ] All TODO placeholders filled with real data
- [ ] All tables have real numbers, not placeholders
- [ ] All figures show real data, not mockups
- [ ] Abstract matches results section
- [ ] Contributions match what was actually done
- [ ] Limitations section is honest and complete
- [ ] Future work section clearly separates from completed work

---

## 📋 REVIEW QUESTIONS

Ask yourself before each claim:

1. **Do I have data for this?** If no → don't claim it
2. **Did I run the experiment?** If no → don't claim it
3. **Is the comparison fair?** If no → don't make it
4. **Is this validated on hardware?** If no → don't claim it
5. **Is this implemented and tested?** If no → don't claim it
6. **Would a reviewer ask for evidence?** If yes → provide it or remove claim
7. **Could this be seen as fabrication?** If yes → remove it immediately
8. **Is this aspirational or actual?** If aspirational → move to future work

---

## 🚨 RED FLAGS

Watch for these warning signs in your writing:

- "Our method achieves..." without data
- "We demonstrate..." without experiments
- "Results show..." without results section
- "Outperforms..." without comparison
- "Robust to..." without robustness tests
- "Generalizes to..." without generalization tests
- "Enables..." without implementation
- "Validates..." without validation
- "First..." without literature review
- "Novel..." without prior art search
- "State-of-the-art..." without benchmarks
- "Real-time..." without timing data
- "Energy-efficient..." without energy measurements
- "Safe..." without safety analysis
- "Hardware..." without hardware tests

If you see any of these, stop and verify you have evidence.

---

## 📖 EXAMPLE SAFE ABSTRACT

> **Abstract** — Wheeled bipedal robots combine the stability of wheeled platforms with the versatility of legged systems, but maintaining balance across varying body heights remains challenging. We propose a bounded residual reinforcement learning approach that combines a height-dependent LQR/IK prior with a learned PPO residual policy. The prior provides structured nominal balance behavior, while the bounded residual learns corrective actions for height-adaptive stabilization and push-disturbance recovery. We evaluate our approach on random-height balance, commanded height transitions, and push recovery scenarios in simulation. We compare against the LQR/IK prior alone and a pure PPO baseline. Our evaluation includes robustness tests under model uncertainty and ablation studies of key design choices. We analyze residual action characteristics and discuss limitations and future directions including hardware validation.

**Why this is safe:**
- Describes problem and approach clearly
- Uses "we propose", "we evaluate", "we compare" instead of "we achieve"
- No specific numerical claims
- No hardware claims
- No novelty overclaims
- Honest about simulation-only scope
- Mentions limitations and future work

---

## 📖 EXAMPLE UNSAFE ABSTRACT (DO NOT USE)

> **Abstract** — We present the first residual RL method for wheeled bipedal robots, achieving 95% success rate on height-adaptive balance tasks. Our novel bounded residual architecture outperforms pure PPO by 40% and enables robust recovery from 150N pushes. We demonstrate state-of-the-art performance on challenging scenarios including stand-up recovery, locomotion, and stair climbing. Our method generalizes to hardware with minimal sim-to-real gap and operates safely in real-world environments. Results show superior sample efficiency and energy efficiency compared to existing methods.

**Why this is unsafe:**
- Claims "first" without verification
- Claims specific numbers without data
- Claims "outperforms" without comparison
- Claims stand-up/locomotion without implementation
- Claims hardware success without validation
- Claims safety without analysis
- Claims energy efficiency without measurements
- Overstates novelty and contributions

---

## 🎓 REMEMBER

**The goal is not to write the most impressive paper.**  
**The goal is to write an honest, reproducible, and useful paper.**

- Be honest about what you did and didn't do
- Be clear about what works and what doesn't
- Be transparent about limitations
- Be fair in comparisons
- Be reproducible in methods
- Be useful to future researchers

**When in doubt, err on the side of caution.**  
**A conservative claim is better than a retracted paper.**
