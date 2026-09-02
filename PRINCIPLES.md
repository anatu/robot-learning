# Principles, in order

One line per principle, taken verbatim from each lesson's **Carry forward** block. This is the course in ~100 sentences: if a line here is not obvious to you, that lesson is not done. Regenerate after editing any README's Carry-forward block.

## Phase 1 — Data

### [Lesson 01 — LeRobotDataset Anatomy](lessons/01-dataset-anatomy/)

- Episode boundaries are in `meta/episodes/`, never in filenames.
- Windowing = validate (multiple of 1/fps) → clamp to episode → mask the clamped positions.
- Padding is a mask because a filled frame would be a lie the model learns.
- Video frames are located by (shard, timestamp) from metadata; frame index alone is not enough once episodes share files.

### [Lesson 02 — Write Your Own Dataset](lessons/02-dataset-writer/)

- `finalize()` is part of the lifecycle, not cleanup: without it the parquet footers never get written.
- `action` is the command, `observation.state` the measurement; the gap between them is the training signal, not noise.
- Lock recording to the physics step count, never to wall-clock; assert the ratio is an integer.
- `add_frame` takes HWC uint8; `dataset[i]` returns CHW float32 in [0,1]. The library owns the conversion; you own the input convention.

## Phase 2 — Classical core

### [Lesson 03 — Kinematics](lessons/03-kinematics/)

- FK = compose body offsets and joint rotations down the MJCF chain; validate against `mj_forward` to 1e-10, and an error of 1e-3 is a convention bug, not numerics.
- Jacobian column $i$ = $[\hat\omega_i \times (p_e - p_i);\ \hat\omega_i]$; validate against *both* `mj_jacSite` and finite differences, because a shared frame mistake cancels in one check but not the other.
- DLS caps the velocity gain at $1/2\lambda$ and buys it with a tracking bias $\propto \lambda$; the $\lambda$-tradeoff plot is how you pick it.
- A 5-DOF arm reaches a 5-dim slice of SE(3): pose IK is generically infeasible, position IK is not.

### [Lesson 04 — Differential IK as Optimization + Feedback](lessons/04-diff-ik-control/)

- Open loop integrates error; feedback makes the error dynamics $\dot e = -K_p e$ and rejects anything the model got wrong in *magnitude* — but not in *direction* (25% mismatch shows you the limit).
- The gain ceiling is set by the control period ($\sim 2/\Delta t$) and eroded by how much $J$ changes along the trace.
- Clipping a solution and constraining an optimization are different operations; only the second keeps the task direction when a limit binds.
- $\epsilon \lVert \dot q \rVert^2$ in a QP is DLS; velocity bounds in a QP do structurally what $\lambda$ did heuristically near singularities.

### [Lesson 05 — Optimal Control Sprint: LQR → iLQR → TVLQR](lessons/05-optimal-control/)

- Optimality with a quadratic cost and linear dynamics *produces* linear feedback; $P_k$ is the cost-to-go Hessian, and it is computed backward because cost-to-go is defined backward.
- iLQR = repeated LQR on linearizations, made safe by two guards: $\mu$ (keeps $Q_{uu} \succ 0$) and line search (keeps the second-order model honest). Remove either and the iteration diverges in a characteristic way.
- The forward pass is closed-loop ($K_k$ inside the rollout), which is why iLQR tolerates long horizons where shooting cannot.
- TVLQR gains stabilize a tube, not a point — they cover perturbations the optimizer never visited.
- $V_{xx} \leftrightarrow$ value-function curvature, $Q_u = 0 \leftrightarrow$ the policy-gradient stationarity condition: RL is this problem without $A_k, B_k$.

### [Lesson 06 — Grasp Mechanics *(optional)*](lessons/06-grasp-mechanics/)

- The grasp map turns contact forces into an object wrench; closure is a statement about whether the image of the friction cones under $G$ fills wrench space.
- Form closure is geometry (7 contacts in 3D); force closure is friction (2 antipodal contacts suffice iff the line between them is inside both cones).
- Linearized cones are conservative: an $m$-edge pyramid is inscribed in the true cone.
- Holding a load is a feasibility question; the SOCP's infeasibility is a physical verdict, and the three ways out are more friction, more squeeze, or different contacts.
- None of this sees deformables, uncertainty, or dynamics — the reason Phase 4 learns instead.

### [Lesson 07 — Motion Planning *(optional)*](lessons/07-motion-planning/)

- Planning lives in $\mathcal{C}$; the collision oracle is the only window into $\mathcal{C}_{\text{obs}}$, and its speed sets the planning budget.
- RRT: complete but not optimal; shortcut afterwards. RRT*'s one-line rewiring change buys asymptotic optimality at a per-iteration cost.
- Trajopt: locally optimal and smooth, blind to topology; seed it with a sample-based path.
- "Converged" ≠ "collision-free" — always re-verify with the oracle.
- The whole pipeline assumes a perfect scene model; a 2 cm pose error breaks it in three places, and Phase 4 exists because of that.

## Phase 3 — Reinforcement learning

### [Lesson 08 — The RL Ladder: REINFORCE → DQN → SAC](lessons/08-rl-ladder/)

- Policy gradient: reward-to-go and a state-only baseline cut variance and add no bias; a baseline that depends on $a_t$ does add bias.
- Deadly triad = function approximation + bootstrapping + off-policy data; the target network amputates the bootstrapping leg's feedback loop.
- SAC = twin-min critics + reparameterized tanh-Gaussian actor (with the $\log(1-\tanh^2)$ correction) + auto-temperature to a target entropy of $-\dim(\mathcal{A})$.
- Sample-efficiency ordering follows data reuse: on-policy (burn after one update) ≪ off-policy replay.

### [Lesson 09 — Sample-Efficient, Data-Driven RL: RLPD](lessons/09-rlpd/)

- Preloading dilutes as $|\mathcal{D}_{\text{online}}|$ grows; symmetric sampling keeps the demo share fixed at 50% of every batch.
- Effective oversampling of a demo transition = $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$, growing through training.
- Critic LayerNorm bounds Q-extrapolation on off-distribution actions; demos trigger that failure earliest.
- Demo-to-transition conversion (reward placement, action frame) is where the silent bugs live, not the algorithm.

### [Lesson 10 — HIL-SERL in Simulation](lessons/10-hil-serl-sim/)

- Classifier precision is the binding constraint: a false positive is a reward the policy will learn to farm.
- Actor and learner are decoupled by two queues; staleness is bounded by the parameter-push rate and tolerated because SAC is off-policy.
- Interventions are on-policy corrections at the policy's own failure states, and they land in both buffers.
- The intervention-rate curve decaying toward zero is the system working; a rising reward with flat true success is the classifier being exploited.

### [Lesson 11 — Domain Randomization & the Reality Gap](lessons/11-domain-randomization/)

- DR optimizes an average over $\Xi$; the width $w$ trades nominal performance for coverage.
- Randomize from pristine values, per episode, by name (`mj_name2id`), and log $\xi$.
- The heatmap's two failure modes: a bright island at $w{=}1$ that dies off-nominal; a flat map at large $w$ whose nominal cell has dropped.
- DR is blind to parameters you didn't randomize; real data isn't.

## Phase 4 — Generative imitation policies

### [Lesson 12 — Why Generative Policies: The Multimodality Lab](lessons/12-multimodality-lab/)

- MSE regression is Gaussian maximum likelihood; on multimodal data it returns the conditional mean, which may be an action nobody took.
- Mode-averaging is a modeling-class failure; compounding error is a distribution-shift failure. Different diseases, different cures.
- Sampling from a generative head fixes averaging and leaves shift untouched; chunking mitigates shift and leaves averaging untouched.
- The four losses in the table are the whole Phase 4 vocabulary. ACT is CVAE + chunking; Diffusion Policy is DDPM + chunking; π0 is CFM + chunking + a VLM.

### [Lesson 13 — Derivation Dossier](lessons/13-derivations/)

- The forward posterior is tractable only because the chain is Markov (I1); everything downstream is Gaussian algebra (I2, I3).
- "Simplified loss" means a reweighting across timesteps, not an algebraic simplification; it changes the objective.
- The CFM cross-term vanishes in the *gradient* via the tower rule, which is why you may regress on conditional targets you can sample.
- $\epsilon$, score, and velocity are interconvertible per path family; do not mix variance-preserving formulas into OT-path formulas.
- Straight (OT) paths are why few-step sampling works; Lesson 15 measures this.

### [Lesson 14 — ACT: Action Chunking with Transformers](lessons/14-act/)

- Chunking divides the decision horizon; that is why $H_a{=}1$ collapses and $H_a{=}100$ works on the same data.
- The CVAE encoder exists to absorb style so the decoder need not average it; $z{=}0$ at inference asks for the mean style.
- Ensembling is a convex combination of overlapping predictions with exponential weights; it trades reactivity for smoothness and reappears as Lesson 16's aggregation function.
- Success rates without seeds and intervals are noise. The harness built here is the yardstick for the rest of the course.

### [Lesson 15 — Diffusion Policy + Sampler Study](lessons/15-diffusion-policy/)

- $T_o / T_p / T_a$: condition on a little, predict a chunk, execute a fraction. Each horizon is a knob with a failure at both ends.
- DDIM reuses DDPM weights because it shares the marginals; it changes the sampler, not the objective. Flow matching changes the objective.
- Latency is only real when the device is synchronized before both timestamps.
- Few-step tolerance is a property of path straightness, and straightness is measurable.

### [Lesson 16 — Async Inference](lessons/16-async-inference/)

- Idle fraction under sync $= l_S/(H_a\Delta t + l_S)$; async removes it iff $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$.
- $g$ trades starvation against wasted chunks; the sweet spot sits just above $g^\*$ computed from the *right* latency statistic.
- Overlap aggregation is temporal ensembling across processes; RTC moves the smoothing into the sampler.
- Latency numbers are only comparable with synchronized devices and same-clock timestamps.

## Phase 5 — Generalist policies

### [Lesson 17 — The VLA Landscape + π0 Dissection](lessons/17-vla-landscape-pi0/)

- Five axes place any VLA: backbone, action interface (discrete / FAST / FM / diffusion), action head, data mix, inference scheme. Latency and generalization follow from the placement.
- Continuous chunk heads won the 2024–25 round on rate and precision; FAST made autoregressive tokens competitive again by compressing chunks.
- Prefix KV caching across denoising steps is exact iff the prefix cannot attend to the actions. The mask is the invariant.
- "VLM understands" is separated from "expert acts" in every current design; KI is the training-time version of the same separation.

### [Lesson 18 — Fine-Tune SmolVLA](lessons/18-smolvla-finetune/)

- A fine-tune's loss curve has a shape (steep for ~2k steps, then a grind); a curve without the shape means the inputs are wrong, not the model.
- "Full fine-tune" is a parameter-group list, not a word; read it out of the trainer before you compare anything.
- LoRA's cost is a slightly higher floor; its benefit is 10–100× fewer trainable parameters. Whether the floor matters is an empirical question per task.
- Mid-depth VLM features suffice for control when they wouldn't for VQA; that says the action expert reads geometry, not semantics.
- The benchmark's docs page is the API of record; a README is a snapshot.

### [Lesson 19 — Comparative VLA Lab](lessons/19-vla-comparative/)

- A comparison is only as good as the budget rule and the pre-registration; pick both before training.
- Equal GPU-hours penalizes large models' step counts; that is a feature when the question is "what do I fine-tune with my money".
- Report per-task, not just aggregate; report CIs everywhere; report cost.
- A surprise without a follow-up test is an anecdote.
- Threats to validity are stated per conclusion: which claims survive which threat.

## Phase 6 — Frontier

### [Lesson 20 — Beyond Imitation: RL-from-Experience & World Models](lessons/20-beyond-imitation/)

- Sort a post-imitation method by *who grades the experience*; that column predicts its data cost and where its gains show up.
- Signal source and task specification are orthogonal axes; place a method on both.
- A world model can pay off entirely at training time (VLA-JEPA) or be kept at test time (FastWAM); the choice is a bet about whether imagination is needed to act.
- A learned evaluator is usable at three bars — data filter, eval replacement, RL reward — and they are ordered by the precision they demand.
- A frontier claim without weights, paper, or baseline spec is a hypothesis with a date; write down what would change your mind.

### [Lesson 21 — Embodied Reasoning as a Planning Layer](lessons/21-embodied-reasoning/)

- Hierarchy trades one hard problem (reactive compositional control) for three measurable ones: grounding, planning, verification. Measure each before wiring them together.
- Ground at dispatch time; the frame you planned from is stale by the second subtask.
- Verification FPs are worse than FNs in a sequential loop; threshold accordingly.
- Schemas and evals survive a model roll; prompts don't. Put the model string in one file.

## Hardware track

### [H1 — Bring-Up the SO-101](hardware/H1-build-bringup/)

- Leader 1/191 (base, elbow) · 1/345 (shoulder lift) · 1/147 (wrist flex, wrist roll, gripper); follower 6× 1/345.
- Motor IDs and baudrate live in EEPROM, written once; calibration lives in a file named by `--robot.id`, and that id never changes.
- A calibration is only as good as the sweep: check the file, not the feel.
- Lag follows velocity, offset is constant, sag is configuration-dependent.
- E-stop within reach whenever torque is on. No exceptions.

### [H2 — Real Teleop Data Collection](hardware/H2-data-collection/)

- Do the task from the camera images alone, or the policy can't either.
- One strategy, no pauses, demonstrated start distribution = evaluated start distribution.
- The failed-demo policy is written before recording; silent mixing is the one indefensible choice.
- Rig drift is a distribution shift; witness marks and a preflight checklist are the fix, not vigilance.
- `--resume=true` counts *additional* episodes and needs `--dataset.root`.

### [H3 — ACT & Diffusion Policy on Real Hardware](hardware/H3-act-dp-real/)

- Pre-register conditions, start sequence, N, success sentence, and taxonomy; the commit timestamp is the receipt.
- N=20 means a 95% CI ~±20 points; report the interval, never the point alone.
- Parity is checked by diffing features against config, not by watching the arm.
- One primary failure label per failed trial, from video; `hardware` is a label, not an exclusion.
- Interleave policies within a session so drift and fatigue don't load onto one arm of the comparison.

### [H4 — VLAs on Your Arm](hardware/H4-vla-real/)

- Zero-shot success is a measurement of pretraining coverage, not of generalization in the abstract.
- Pretraining is expected to show up as robustness across conditions before it shows up as peak ID success.
- Intervene at *incipient* failure so the dataset contains the states the policy actually reaches, then a clean recovery from them.
- At N=20 per cell, claim improvement only when the targeted failure class shrank *and* nothing else regressed.

### [H5 — Real-Robot RL: HIL-SERL (Stretch)](hardware/H5-hil-serl-real/)

- The bounds box is the safety mechanism; the human is the *learning* mechanism.
- Classifier precision gates everything downstream: a false positive is a reward the policy will learn to farm.
- Short episodes and EE-space actions are what make the 1–2 h claim even plausible; both are sample-efficiency levers, not conveniences.
- The Lesson 10 curves are predictions; reality's deviations are the lesson.

### [H6 — Mobile Manipulation: LeKiwi → XLeRobot (Stretch)](hardware/H6-mobile/)

- On a distributed robot, network latency is a data-quality variable; budget it and gate episodes on it.
- Holonomy is a rank condition on the wheel-to-body map; dead reckoning drifts because that map integrates slip it cannot see.
- Phase consistency in mobile demos plays the role grasp consistency played in H2.
- A new embodiment costs a config, not a pipeline.
