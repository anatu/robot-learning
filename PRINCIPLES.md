# Principles, in Order

This file collects the **Carry forward** block from every lesson, in course order. Each bullet is a principle stated as a complete sentence with its reason, and together they are a compact statement of what the course teaches. If a bullet here does not read as obvious to you, the lesson it comes from is not finished. The file is generated from the lesson READMEs; edit a lesson's Carry-forward block and regenerate rather than editing here.

## Phase 1 — Data

### [Lesson 01 — LeRobotDataset Anatomy](lessons/01-dataset-anatomy/)

- Episode boundaries are recorded in `meta/episodes/` rather than implied by filenames, because a format that must scale to millions of episodes cannot afford one file per episode.
- Windowing applies three rules in order: each offset must be a multiple of the frame period within `tolerance_s`; the target index is clamped to the episode; and positions whose unclamped index fell outside the episode are flagged in a padding mask.
- Padding is reported as a mask rather than filled with synthetic frames, so that training code can exclude padded positions from the loss instead of learning from invented observations.
- A video frame is located by shard and timestamp from the metadata; once episodes share files, the frame index alone is not enough to find it.

### [Lesson 02 — Write Your Own Dataset](lessons/02-dataset-writer/)

- `finalize()` is part of the write lifecycle rather than an optional cleanup step, because it flushes the buffered episode metadata and writes the parquet footers; without it the files cannot be read.
- `action` records the command and `observation.state` records the measurement. The gap between them is the controller's tracking error, and it is the signal that behaviour cloning learns from, not noise to be removed.
- Recording must be locked to the physics step count, never to wall-clock time, and the number of physics steps per frame must be asserted to be an integer, because a non-integer ratio puts frames off the frame grid and invalidates every window over them.
- `add_frame` takes images as HWC uint8 arrays, and `dataset[i]` returns them as CHW float32 tensors in [0, 1]. The library owns that conversion; you are responsible for supplying the input in the expected convention.

## Phase 2 — Classical core

### [Lesson 03 — Kinematics](lessons/03-kinematics/)

- Forward kinematics is the composition of body offsets and joint rotations down the MJCF chain. Validate it against `mj_forward` to 1e-10; an error of order 1e-3 indicates a frame-convention bug rather than numerical error, because composing exact transforms has no other source of error at that magnitude.
- Column $i$ of the geometric Jacobian is $[\hat\omega_i \times (p_e - p_i);\ \hat\omega_i]$. Validate it against both `mj_jacSite` and finite differences of your own forward kinematics, because a frame mistake shared by your FK and your Jacobian cancels in one of those checks but not in the other.
- Damped least squares caps the velocity gain at $1/2\lambda$ and pays for that cap with a tracking bias proportional to $\lambda$. The plot of bias against velocity as $\lambda$ varies is the basis for choosing $\lambda$.
- A five-joint arm reaches a five-dimensional slice of SE(3), so inverse kinematics for a full pose is generically infeasible while inverse kinematics for a position is not.

### [Lesson 04 — Differential IK as Optimization + Feedback](lessons/04-diff-ik-control/)

- Open-loop tracking integrates its own error, whereas proportional feedback gives the error dynamics $\dot e = -K_p e$ and corrects anything the model got wrong in magnitude. It cannot correct errors in direction, which is why tracking degrades again at large model mismatch.
- The proportional-gain ceiling is set by the control period, at roughly $2/\Delta t$, and is lowered in practice by how much the Jacobian changes along the trajectory.
- Clipping a solution and constraining an optimisation are different operations. Only the constrained optimisation keeps the task direction when a limit binds, because it re-solves for the remaining freedom rather than truncating one component.
- The $\epsilon \lVert \dot q \rVert^2$ regulariser in the QP is damped least squares, and the QP's velocity bounds do explicitly near singularities what the damping parameter $\lambda$ did heuristically.

### [Lesson 05 — Optimal Control Sprint: LQR → iLQR → TVLQR](lessons/05-optimal-control/)

- With a quadratic cost and linear dynamics, optimality produces a linear feedback law; $P_k$ is the Hessian of the cost-to-go, and the recursion runs backward because the cost-to-go at one step is defined in terms of the cost-to-go at the next.
- iLQR is repeated LQR on local linearizations, made safe by two guards: regularization $\mu$ keeps $Q_{uu}$ positive definite, and the line search keeps the second-order model honest. Removing either produces a characteristic divergence.
- The forward pass is closed-loop because the gain $K_k$ is applied inside the rollout, which is why iLQR tolerates long horizons where plain shooting cannot.
- TVLQR gains stabilize a tube around a trajectory rather than a single point, so they handle perturbations the optimizer never visited.
- The value curvature $V_{xx}$ and the stationarity condition $Q_u = 0$ have direct counterparts in reinforcement learning, which is the same problem solved without access to $A_k$ and $B_k$.

### [Lesson 06 — Grasp Mechanics *(optional)*](lessons/06-grasp-mechanics/)

- The grasp map turns contact forces into an object wrench, and closure is a statement about whether the image of the friction cones under $G$ fills wrench space.
- Form closure is a geometric property that needs at least seven contacts in 3-D, whereas force closure relies on friction and is achieved by two antipodal contacts whenever the line between them lies inside both cones.
- A linearized friction cone is an inscribed pyramid, so every test built on it is conservative.
- Holding a load is a feasibility question; the SOCP's infeasibility is a physical verdict, and the three ways to restore feasibility are more friction, more normal force, or different contact locations.
- None of this analysis sees deformable objects, uncertainty in contact location, or dynamics, which is the reason Phase 4 turns to learned policies.

### [Lesson 07 — Motion Planning *(optional)*](lessons/07-motion-planning/)

- Planning lives in configuration space, and the collision oracle is the only window into $\mathcal{C}_{\text{obs}}$; its speed sets the planning budget.
- RRT is probabilistically complete but not optimal, so its paths are shortcut afterwards; the rewiring step of RRT* buys asymptotic optimality at a per-iteration cost.
- Trajectory optimization is locally optimal and smooth but blind to topology, so it should be seeded with a sample-based path.
- An optimizer's convergence flag does not establish that a path is collision-free, because the penalty may be too weak or the constraint may hold only at waypoints; re-verify with the oracle.
- The whole pipeline assumes a perfect scene model, and a 2 cm error in an object's pose breaks it in three places; the learned methods of Phase 4 exist largely because of that fragility.

## Phase 3 — Reinforcement learning

### [Lesson 08 — The RL Ladder: REINFORCE → DQN → SAC](lessons/08-rl-ladder/)

- In the policy gradient, reward-to-go and a state-only baseline reduce variance without adding bias; a baseline that depends on the action does add bias, because the cancellation argument requires $b$ to be constant with respect to $a_t$.
- The deadly triad is function approximation plus bootstrapping plus off-policy data. The target network breaks the feedback loop in the bootstrapping leg by holding the regression target fixed between updates.
- SAC consists of twin critics with a minimum in the target, a reparameterized tanh-Gaussian actor with the $\log(1-\tanh^2)$ correction, and an automatically tuned temperature that drives entropy toward $-\dim(\mathcal{A})$.
- Sample efficiency follows data reuse: an on-policy method discards each trajectory after one update, whereas an off-policy method with a replay buffer reuses each transition many times.

### [Lesson 09 — Sample-Efficient, Data-Driven RL: RLPD](lessons/09-rlpd/)

- Preloading demonstrations into the online buffer dilutes them as the buffer grows, because uniform sampling gives each transition a share proportional to its count; symmetric sampling fixes the demonstration share at half of every batch regardless of buffer sizes.
- The effective oversampling factor of a demonstration transition under symmetric sampling is $|\mathcal{D}_{\text{online}}|/|\mathcal{D}_{\text{demo}}|$, and it grows throughout training.
- LayerNorm in the critic bounds Q-value extrapolation on off-distribution actions, and demonstrations are the earliest source of such actions, so they trigger the failure first.
- Most failures of RLPD in practice are in the conversion of demonstrations to transitions, in particular the placement of the reward and the frame of the actions, rather than in the algorithm.

### [Lesson 10 — HIL-SERL in Simulation](lessons/10-hil-serl-sim/)

- The classifier's precision is the binding constraint on the whole system, because a false positive is a reward the policy will learn to farm.
- The actor and learner are decoupled by two queues; the staleness of the actor's parameters is bounded by the parameter-push rate and is tolerated because SAC is off-policy.
- Interventions are on-policy corrections at the policy's own failure states, and they enter both buffers, which raises their sampling weight under symmetric sampling.
- An intervention-rate curve that decays toward zero is the sign of a working system, whereas a rising reward alongside a flat true success rate is the sign of a classifier being exploited.

### [Lesson 11 — Domain Randomization and the Reality Gap](lessons/11-domain-randomization/)

- Domain randomization optimizes an average over the parameter distribution $\Xi$, so the width $w$ trades nominal performance for coverage of off-nominal dynamics.
- A randomization wrapper must draw its multipliers per episode, apply them to the pristine model values, resolve bodies and geoms by name through `mj_name2id`, and log the drawn $\xi$; applying multipliers to current values makes the dynamics random-walk.
- The transfer heatmap has two failure modes: at $w{=}1$ a bright island at nominal that dies off-nominal, and at large $w$ a flat map whose nominal cell has dropped.
- Domain randomization is blind to parameters that were not randomized, whereas real data varies along every axis, which is why real data is preferred when it can be afforded.

## Phase 4 — Generative imitation policies

### [Lesson 12 — Why Generative Policies: The Multimodality Lab](lessons/12-multimodality-lab/)

- Mean-squared-error regression is maximum likelihood under a unimodal Gaussian, so on multimodal data it returns the conditional mean, which may be an action no expert ever took.
- Mode-averaging is a failure of the model class and compounding error is a failure of distribution shift; because their causes differ, their remedies differ too.
- Sampling from a generative head fixes averaging and leaves shift untouched, whereas chunking mitigates shift and leaves averaging untouched, so a practical policy needs both.
- The four losses in the table are the vocabulary of Phase 4: ACT is a CVAE with chunking, Diffusion Policy is a DDPM with chunking, and π0 is conditional flow matching with chunking on top of a vision-language model.

### [Lesson 13 — Derivation Dossier](lessons/13-derivations/)

- The forward posterior $q(x_{t-1} \mid x_t, x_0)$ is tractable only because the forward chain is Markov, which is what licenses identity (I1); everything after that step is Gaussian algebra via (I2) and (I3).
- The "simplified" DDPM loss is a reweighting across timesteps rather than an algebraic simplification, and it changes the objective being optimized, not merely its form.
- In the conditional flow-matching proof, the cross-term vanishes in the gradient by the tower rule, which is why it is legitimate to regress onto conditional targets that you can sample even though the marginal target cannot be computed.
- The conversions among $\epsilon$, score and velocity hold within a path family, so formulas for the variance-preserving path must not be mixed with formulas for the optimal-transport path.
- Straight optimal-transport paths are the reason few-step sampling works, and Lesson 15 measures this directly.

### [Lesson 14 — ACT: Action Chunking with Transformers](lessons/14-act/)

- Chunking divides the number of decisions in an episode by $H_a$, which is why $H_a{=}1$ collapses and $H_a{=}100$ succeeds on exactly the same data.
- The CVAE encoder exists so that demonstration style is absorbed into $z$ rather than averaged by the decoder, and setting $z{=}0$ at inference requests the mean style rather than the mean action.
- Temporal ensembling is a convex combination of overlapping predictions with exponential weights; it trades reactivity for smoothness, and the same operation reappears in Lesson 16 as the aggregation function between processes.
- A success rate without a seed list and a confidence interval is not a measurement, and the harness built in this lesson is the yardstick used by every later lesson.

### [Lesson 15 — Diffusion Policy and the Sampler Study](lessons/15-diffusion-policy/)

- Diffusion Policy conditions on $T_o$ observations, predicts a chunk of $T_p$ actions, and executes $T_a$ of them; each horizon has a characteristic failure at both extremes, so the settings are design decisions rather than defaults to accept.
- DDIM can reuse DDPM's trained weights because the two processes share the same marginals; it changes only the sampler. Flow matching changes the training objective, so it requires retraining.
- A latency measurement is only meaningful if the accelerator is synchronized before both timestamps, because GPU and MPS dispatch return before the computation has finished.
- The number of steps a sampler can skip is governed by how straight its denoising paths are, and straightness is a quantity you can measure directly.

### [Lesson 16 — Async Inference](lessons/16-async-inference/)

- Under synchronous inference the idle fraction is $l_S/(H_a\Delta t + l_S)$; asynchronous inference removes the idle time if and only if $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$, because the queue must hold enough actions to cover one round trip of consumption.
- The threshold $g$ trades queue starvation against wasted chunks, so the useful setting sits just above $g^\*$, and $g^\*$ must be computed from the latency statistic that actually governs starvation, which is the tail rather than the mean when latency has a spread.
- Aggregation on chunk overlap is temporal ensembling carried out across processes; real-time chunking moves the same smoothing into the sampler itself.
- Latency numbers are comparable only when the device was synchronized before both timestamps and both timestamps were taken on the same clock.

## Phase 5 — Generalist policies

### [Lesson 17 — The VLA Landscape and a π0 Dissection](lessons/17-vla-landscape-pi0/)

- Any VLA can be placed on five axes, namely backbone, action interface (discrete, FAST, flow matching, or diffusion), action head, data mix, and inference scheme, and its latency and generalization profile follow from that placement because each axis constrains the others.
- Continuous chunk-producing heads won the 2024–25 round on control rate and precision, and FAST made autoregressive tokens competitive again by compressing action chunks so that far fewer tokens are needed per chunk.
- Caching the prefix's keys and values across denoising steps is exact if and only if the prefix cannot attend to the action tokens, because a token's keys and values depend only on what it attends to; the mask is the invariant that the cache relies on.
- Every current design separates a component that understands (the VLM) from a component that acts (the expert), and knowledge insulation is the training-time form of that same separation.

### [Lesson 18 — Fine-Tune SmolVLA](lessons/18-smolvla-finetune/)

- A fine-tune's loss curve has a characteristic shape, a steep drop over roughly the first two thousand steps followed by a slow decline. A curve that is flat from the start indicates that the inputs do not match what the policy expects, not that the model cannot learn.
- "Full fine-tune" names a list of parameter groups, and that list differs between LeRobot versions, so you must read it out of the trainer before you compare adaptation methods.
- LoRA trains ten to a hundred times fewer parameters at the cost of a slightly higher loss floor; whether that floor matters for success rate is an empirical question that has to be answered per task.
- Mid-depth VLM features are sufficient for control even though they would not be sufficient for visual question answering, which tells you that the action expert is reading geometric rather than semantic content.
- The benchmark's documentation page is the interface of record, because it tracks the installed version and a README is a snapshot.

### [Lesson 19 — Comparative VLA Lab](lessons/19-vla-comparative/)

- A comparison is only as good as its budget rule and its pre-registration, because those two decisions are the ones most easily bent after the results are in; choose both before training.
- Equal GPU-hours gives large models fewer steps, and that is appropriate when the question is which model to fine-tune with a fixed amount of money rather than which model is best in principle.
- Report per-task results and confidence intervals alongside the aggregate, and report cost, because aggregates hide the task-level disagreements that explain differences.
- A surprising result that contradicts your prediction is only evidence once you have run a test that could have falsified your explanation for it.
- A threats-to-validity section is stated per conclusion, so that a reader knows which claims survive which objection.

## Phase 6 — Frontier

### [Lesson 20 — Beyond Imitation: RL-from-Experience and World Models](lessons/20-beyond-imitation/)

- A post-imitation method is best classified by who grades the experience, because that choice predicts both what the method costs in data and where its gains appear.
- Signal source and task specification are orthogonal axes, and a method should be placed on both before it is compared with anything.
- A world model can pay for itself entirely at training time, as in VLA-JEPA, or be retained at test time, as in FastWAM; the difference is a bet about whether the policy needs to imagine in order to act.
- A learned evaluator can be used at three levels of trust, as a data filter, as a replacement for ground-truth evaluation, and as a reward for RL, and the three demand increasing precision because the cost of a false positive rises at each level.
- A frontier claim without weights, a paper, or a baseline specification is a hypothesis with a date attached, and the useful response is to write down what evidence would change your mind.

### [Lesson 21 — Embodied Reasoning as a Planning Layer](lessons/21-embodied-reasoning/)

- A hierarchy replaces one hard problem, reactive compositional control, with three problems that can each be measured on their own: grounding, planning, and verification. Measure each before wiring them together, because the component measurements predict how the assembled system will fail.
- Ground each subtask's target at dispatch time rather than at planning time, because the scene changes after every subtask and a point computed from the planning frame is stale by the second subtask.
- In a sequential loop a verification false positive compounds silently across later subtasks, whereas a false negative costs one bounded replan; set the verification threshold to favour precision accordingly.
- Schemas and evaluation sets survive a model roll and prompts do not, so keep the model string in exactly one file and invest in the parts that survive.

## Hardware track

### [H1 — Bring-Up the SO-101](hardware/H1-build-bringup/)

- The leader uses three gear ratios (1/191 on base and elbow, 1/345 on shoulder lift, 1/147 on wrist flex, wrist roll and gripper) and the follower uses 1/345 throughout, because the leader must be back-drivable by hand while the follower must hold its pose.
- Motor IDs and baudrate are written once into each motor's EEPROM, while the calibration lives in a file named by `--robot.id`; that id must never change, because every later command looks the calibration up by it.
- A calibration is only as good as the sweep that produced it, and a bad sweep is visible in the file as a narrow or inverted joint range before it is visible in motion.
- The three sources of tracking error have distinct signatures: lag scales with velocity, a calibration offset is constant, and gravity sag depends on configuration.
- The e-stop stays within reach whenever torque is on, without exception.

### [H2 — Real Teleop Data Collection](hardware/H2-data-collection/)

- A demonstration is usable only if you could perform the task from the camera images alone, because the policy has no other input.
- Demonstrations should use one grasp strategy, contain no pauses, and cover the same start distribution that will be evaluated, because a policy imitates hesitation, averages over strategies, and cannot generalize to positions it never saw.
- The policy for failed demonstrations is written before recording begins; mixing unlabeled failures into the dataset is the one choice that cannot be defended, because the trainer cannot tell them apart from successes.
- Rig drift between sessions is a distribution shift that cannot be undone, so it is prevented physically with witness marks and a preflight checklist rather than by attention.
- `--resume=true` counts additional episodes rather than the total, and it requires `--dataset.root`.

### [H3 — ACT & Diffusion Policy on Real Hardware](hardware/H3-act-dp-real/)

- Pre-register the conditions, start sequence, trial count, success sentence and failure taxonomy in a commit before the first trial, because at N=20 any adjustment made after seeing results can move the number by more than the effect being measured.
- Twenty trials gives a 95% confidence interval roughly ±20 points wide, so report the interval and never the point estimate alone.
- Observation parity is checked by diffing the dataset features against the rollout configuration, not by watching the arm, because a parity failure produces confident motion rather than an error.
- Every failed trial receives exactly one primary label from the frozen taxonomy, decided from video; a hardware fault is labeled `hardware` and counted, never excluded.
- Policies are interleaved within a session so that rig drift and operator fatigue affect both arms of the comparison equally.

### [H4 — VLAs on Your Arm](hardware/H4-vla-real/)

- Zero-shot success on your rig measures how well the model's pretraining data covers your embodiment, cameras, and task; it is not evidence of generalization in the abstract.
- Pretraining is expected to show up as robustness across conditions before it shows up as higher success on the demonstrated conditions, so a comparison must include out-of-distribution cells to be informative.
- Interventions should begin at incipient failure rather than after full failure, because the purpose of DAgger data is to cover the states the policy actually reaches, and then to show a clean recovery from them.
- At twenty trials per cell, an improvement claim is only defensible when the targeted failure class shrank and no other class grew, since the interval on any single cell is too wide to support more.

### [H5 — Real-Robot RL: HIL-SERL (Stretch)](hardware/H5-hil-serl-real/)

- The bounds box is the safety mechanism and the human is the learning mechanism; a human reaction is too slow to stop an exploring arm, but a human intervention is exactly the corrective data the learner needs.
- Classifier precision gates everything downstream, because a false positive is a reward the policy will learn to farm, and a session spent farming it is wasted.
- Short episodes and end-effector-space actions are what make the one-to-two-hour claim plausible at all; both are sample-efficiency levers rather than conveniences.
- The Lesson 10 curves are predictions for this lesson, and the places where reality deviates from them are what the lesson teaches.

### [H6 — Mobile Manipulation: LeKiwi → XLeRobot (Stretch)](hardware/H6-mobile/)

- On a distributed robot, network latency is a data-quality variable, because a late command is indistinguishable from an operator's hesitation in the recorded data; budget the latency and gate episodes on it.
- Holonomy is a rank condition on the wheel-to-body velocity map, and dead reckoning drifts because that map integrates ideal rolling while the real base slips.
- Phase consistency in mobile demonstrations plays the role that grasp consistency played in H2: a policy trained on demonstrations with varying phase order has to learn a decision the demonstrator never made deliberately.
- A new embodiment costs a configuration, not a new pipeline, and that is the property of the software stack worth verifying.
