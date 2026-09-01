# Robot Learning: A Self-Study Course

A custom curriculum for re-developing robot learning fundamentals through to 2026-frontier generalist policies. Built around Hugging Face's [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) (LeRobot team, Oct 2025), with theory depth drawn from Berkeley CS 285, Stanford CS 224R/CS 237B, CMU 16-745, MIT Robotic Manipulation, and ETH's Robot Learning course, plus everything that has happened since the tutorial shipped (LeRobot v0.5/v0.6, π*0.6/RECAP, GR00T N1.6+, MolmoAct 2, world-model policies).

**Who this is for:** me — MS Robotics background, strong math/ML, refreshing and modernizing. Every lesson assumes graduate-level comfort with linear algebra, probability, optimization, and PyTorch.

**Format:** discrete lessons, built in public. One lesson = one directory = one merged PR with code, results, and a short writeup. Every lesson README follows [TEMPLATE.md](TEMPLATE.md) — backward-designed from a binary "done when" bar: measurable learning objectives, a Background briefing with the actual math, numbered build parts with verbatim commands and ✅ checkpoints, a deliverables manifest with acceptance criteria, self-check questions, and a researched pitfalls table. Each lesson ships a `RESULTS.md` with numbers, plots, and interpretation.

## The arc

Classical robotics → why it hits a wall → RL on real robots → generative imitation policies (ACT, Diffusion Policy) → generalist VLAs (π0, SmolVLA) → what's past imitation (RL-from-experience, world models). This mirrors the tutorial's argument, augmented where it's thin: real math for kinematics/control, the RL algorithm core it skips, rigorous evaluation, and the post-Oct-2025 frontier.

## Tracks

- **Core track (lessons 00–22):** theory + simulation. Runs on an Apple Silicon Mac (MuJoCo, gym-pusht/aloha/hil, Meta-World all run natively); heavy training rents a cloud GPU (~$1–8/run on Vast/RunPod — datasets and checkpoints round-trip through the HF Hub).
- **Hardware track (H1–H6):** a real SO-101 leader/follower pair (~$375 all-in). Starts any time after Lesson 02; interleaves with the core track. H5/H6 are stretch.

## Syllabus

### Phase 0 — Bootstrap
| # | Lesson | Deliverable |
|---|--------|-------------|
| 00 | [Setup & repo bootstrap](lessons/00-setup/) | Working env (lerobot 0.6.x, MuJoCo/mjpython), hello-dataset script. **Order the SO-101 kit today** — lead time overlaps Phase 1–2. |

### Phase 1 — Data (tutorial §1)
| # | Lesson | Deliverable |
|---|--------|-------------|
| 01 | [LeRobotDataset anatomy](lessons/01-dataset-anatomy/) | Format parser that reproduces `LeRobotDataset` outputs, with parity tests |
| 02 | [Write your own dataset](lessons/02-dataset-writer/) | Scripted MuJoCo SO-101 trajectories serialized to a valid v3 dataset on the Hub |

### Phase 2 — Classical core (tutorial §2, deepened via MIT/CS223A/16-745)
| # | Lesson | Deliverable |
|---|--------|-------------|
| 03 | [Kinematics from scratch](lessons/03-kinematics/) | FK/IK/Jacobians for the SO-101, singularity study, animated notebook |
| 04 | [Differential IK + feedback](lessons/04-diff-ik-control/) | Constrained diff-IK tracker, disturbance/model-mismatch experiments |
| 05 | [Optimal control sprint](lessons/05-optimal-control/) | LQR + iLQR cartpole swing-up, from-scratch solvers |
| 06 | [Grasp mechanics](lessons/06-grasp-mechanics/) *(optional)* | Force closure + grasp force optimization as convex programs |
| 07 | [Motion planning](lessons/07-motion-planning/) *(optional)* | RRT + trajectory optimization for the SO-101 in clutter |

### Phase 3 — Reinforcement learning (tutorial §3 + CS 285 core)
| # | Lesson | Deliverable |
|---|--------|-------------|
| 08 | [The RL ladder](lessons/08-rl-ladder/) | REINFORCE → DQN → SAC from scratch, seeded training curves |
| 09 | [Sample-efficient RL: RLPD](lessons/09-rlpd/) | Offline/online buffer ablation in gym-hil with efficiency curves |
| 10 | [HIL-SERL in sim](lessons/10-hil-serl-sim/) | Reward classifier + actor/learner + human interventions, trained pick policy |
| 11 | [Domain randomization](lessons/11-domain-randomization/) | Transfer heatmap across dynamics distributions |

### Phase 4 — Generative imitation policies (tutorial §4)
| # | Lesson | Deliverable |
|---|--------|-------------|
| 12 | [Why generative policies](lessons/12-multimodality-lab/) | MSE vs CVAE vs DDPM vs flow matching on multimodal toy data |
| 13 | [Derivation dossier](lessons/13-derivations/) | Typed note: ELBO → DDPM simplified loss; FM marginal/conditional equivalence |
| 14 | [ACT](lessons/14-act/) | Trained ACT on gym-aloha + chunking/ensembling ablation |
| 15 | [Diffusion Policy](lessons/15-diffusion-policy/) | PushT policy; DDPM vs DDIM vs FM sampler study |
| 16 | [Async inference](lessons/16-async-inference/) | PolicyServer/RobotClient deployment, queue-threshold sweep |

### Phase 5 — Generalist policies (tutorial §5 + 2026 landscape)
| # | Lesson | Deliverable |
|---|--------|-------------|
| 17 | [VLA landscape + π0 dissection](lessons/17-vla-landscape-pi0/) | Comparative note + reimplemented blockwise-causal attention w/ KV cache |
| 18 | [Fine-tune SmolVLA](lessons/18-smolvla-finetune/) | LoRA fine-tune in cloud, `lerobot-eval` on Meta-World/LIBERO |
| 19 | [Comparative VLA lab](lessons/19-vla-comparative/) | 2–3 open VLAs fine-tuned on one dataset, benchmarked head-to-head |

### Phase 6 — Frontier + capstone
| # | Lesson | Deliverable |
|---|--------|-------------|
| 20 | [Beyond imitation](lessons/20-beyond-imitation/) | Survey note on RECAP/π*0.6 + world-model policies + task-specification spectrum; run one in LeRobot v0.6; Skild S1 claims audit |
| 21 | [Embodied reasoning](lessons/21-embodied-reasoning/) | Gemini Robotics-ER 2 as planner over a local policy |
| 22 | [Capstone](lessons/22-capstone/) | Open-ended project: proposal, code, video, report |

### Hardware track (SO-101; start after Lesson 02)
| # | Lesson | Deliverable |
|---|--------|-------------|
| H1 | [Build & bring-up](hardware/H1-build-bringup/) | Assembled, calibrated leader/follower pair; teleop video |
| H2 | [Data collection](hardware/H2-data-collection/) | 50-episode pick-place dataset on the Hub with card + failure log |
| H3 | [ACT & Diffusion Policy on real hardware](hardware/H3-act-dp-real/) | Two deployed policies, 20-trial ID/OOD evaluation |
| H4 | [VLAs on your arm](hardware/H4-vla-real/) | MolmoAct 2 zero-shot + fine-tuned SmolVLA, async deployment |
| H5 | [Real-robot RL](hardware/H5-hil-serl-real/) *(stretch)* | HIL-SERL policy trained live on the arm |
| H6 | [Mobile manipulation](hardware/H6-mobile/) *(stretch)* | LeKiwi/XLeRobot extension |

## Budget

| Item | Cost |
|------|------|
| Seeed SO-ARM101 Pro motor/electronics kit (US warehouse) | $289 |
| Official 3D-printed parts set (or ~$20 PLA+ if self-printing) | $35 |
| 2× USB webcams (InnoMaker 1080p wrist + any 1080p overhead — use two different models) | ~$50 |
| Cloud GPU credits (Vast.ai/RunPod 4090/A100; ~$1–3 per ACT/DP run, ~$3–8 per SmolVLA fine-tune) | ~$50 |
| **Total** | **~$425** |

Stretch: LeKiwi base kit $179 + Raspberry Pi 5 ~$80 (mobile), then ~$250 more for the XLeRobot dual-arm upgrade.

## Toolchain and version policy

- **LeRobot ≥ 0.6.1** (Python ≥ 3.12, Transformers v5, PyTorch 2.7–2.11). The tutorial targets v0.4.0 and no longer runs verbatim; each lesson notes API deltas (e.g. `pip install "lerobot[training]"` extras split, `sac` → `gaussian_actor`, `lerobot.types` → `lerobot.lerobot_types`). The LeRobotDataset v3 format is unchanged, so all data work is stable.
- **Mac-local:** MuJoCo (`mjpython`), gym-pusht/gym-aloha/gym-xarm/gym-hil, Meta-World, Drake — all native on Apple Silicon. Inference and small training runs on `mps`.
- **Cloud (Linux + NVIDIA):** big ACT/DP runs, SmolVLA/VLA fine-tunes, HIL-SERL learners, LIBERO/RoboCasa/ManiSkill evals (`MUJOCO_GL=egl`). Record locally → push to Hub → train in cloud → pull checkpoint.

## Primary sources

- **Backbone:** [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) ([interactive](https://huggingface.co/spaces/lerobot/robot-learning-tutorial), [code](https://github.com/fracapuano/robot-learning-tutorial)) + [HF Robotics Course](https://huggingface.co/learn/robotics-course) (maintained against current LeRobot)
- **RL theory:** [Berkeley CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/) (Fall 2023 videos, Sp26 homeworks), [Stanford CS 224R](https://cs224r.stanford.edu/)
- **Control/classical:** [CMU 16-745](https://optimalcontrol.ri.cmu.edu/), [MIT Robotic Manipulation](https://manipulation.csail.mit.edu/), [Underactuated Robotics](https://underactuated.csail.mit.edu/), [CS223A via SEE](https://see.stanford.edu/Course/CS223A)
- **Peer syllabus:** [ETH Robot Learning: From Fundamentals to Foundation Models](https://cvg.ethz.ch/lectures/Robot-Learning/) (Mees, Sp26 — open videos + homeworks)
- **Frontier reading spine:** [Physical Intelligence blog](https://www.pi.website/blog) (π0 → FAST → π0.5 → knowledge insulation → real-time chunking → π*0.6 → π0.7) + [Skild S1](https://skild.ai/blogs/s1) (in-context task specification — audited in Lesson 20)

## Progress

- [ ] 00 · Setup — [ ] 01 — [ ] 02 — [ ] 03 — [ ] 04 — [ ] 05 — [ ] 06* — [ ] 07*
- [ ] 08 — [ ] 09 — [ ] 10 — [ ] 11 — [ ] 12 — [ ] 13 — [ ] 14 — [ ] 15 — [ ] 16
- [ ] 17 — [ ] 18 — [ ] 19 — [ ] 20 — [ ] 21 — [ ] 22 · Capstone
- [ ] H1 — [ ] H2 — [ ] H3 — [ ] H4 — [ ] H5* — [ ] H6*

\* optional/stretch
