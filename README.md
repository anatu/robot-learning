# Robot Learning: A Self-Study Course

A custom curriculum for re-developing robot learning fundamentals through to 2026-frontier generalist policies. Built around Hugging Face's [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) (LeRobot team, Oct 2025), with theory depth drawn from Berkeley CS 285, Stanford CS 224R/CS 237B, CMU 16-745, MIT Robotic Manipulation, and ETH's Robot Learning course, plus everything that has happened since the tutorial shipped (LeRobot v0.5/v0.6, π*0.6/RECAP, GR00T N1.6+, MolmoAct 2, world-model policies).

**Who this is for:** me — MS Robotics background, strong math/ML, refreshing and modernizing after several years away from robotics code. Every lesson assumes graduate-level comfort with linear algebra, probability, optimization, and PyTorch.

**What a lesson is.** Principles first, then practical exercises that prove you have them. Each lesson README follows [TEMPLATE.md](TEMPLATE.md): a **Principles** section with the mental model and the math, a **Carry forward** block, then numbered exercises typed as [Predict → Run], [Diagnose], [Decide], [Derive], [Read the kernel], [Build], [Read], or [Write], each ending in an observable checkpoint. Code is instrumental and AI-assisted: you write the spec and the check, an AI tool drafts the code, you read the draft and run the check. What you cannot delegate is the prediction, the interpretation, the derivation, and the decision. From-scratch reimplementation lives under **Going deeper** and never gates progression. Every lesson ships a `RESULTS.md` with predictions vs outcomes, figures, and interpretation. The course's one-line principles, in order, are collected in [PRINCIPLES.md](PRINCIPLES.md).

## Status

*Updated 2026-09-02. Detail: [JOURNAL.md](JOURNAL.md); per-lesson checkboxes below under [Progress](#progress).*

| | |
|---|---|
| Core track | **1 / 23** — Lesson 00 (setup) done |
| Hardware track | **0 / 6** — all hardware ordered (Partabot SO-101 assembled + 2 cameras) |
| Now / next | Lesson 01 — dataset anatomy; H1 unblocks when the arm lands |
| Course format | restructured 2026-09-02 to principles-first, AI-assisted coding (see [JOURNAL.md](JOURNAL.md)) |
| Environment | lerobot 0.6.1 · torch 2.11.0 (MPS) · mujoco 3.12.0 · Python 3.12.12 — pinned in [requirements.lock](requirements.lock) |

## How to work a lesson

1. Read **Principles** and the **Carry forward** block until you could reproduce the equations and the design argument without the page.
2. Work the exercises in order. Before any [Predict → Run], write the prediction in `RESULTS.md` first. For a [Build], write the spec (interface, behavior, the check), have Claude or another AI tool draft it, read the draft, run the check.
3. Every [Diagnose] plants a real bug or reproduces a known failure. Name the mechanism, not just the fix.
4. Every lesson ends in a [Decide] or [Write]: a config you would ship, a protocol you commit before running, a note that survives a skeptical reader.
5. Answer the **Self-check** without notes. If one stumps you, Principles names where to re-read.
6. Journal per [CLAUDE.md](CLAUDE.md): tooling fixes, spend, deviations.

## The arc

Classical robotics → why it hits a wall → RL on real robots → generative imitation policies (ACT, Diffusion Policy) → generalist VLAs (π0, SmolVLA) → what's past imitation (RL-from-experience, world models). This mirrors the tutorial's argument, augmented where it's thin: real math for kinematics/control, the RL algorithm core it skips, rigorous evaluation, and the post-Oct-2025 frontier.

## Tracks

- **Core track (lessons 00–22):** theory + simulation. Runs on an Apple Silicon Mac (MuJoCo, gym-pusht/aloha/hil, Meta-World all run natively); heavy training rents a cloud GPU (~$1–8/run on Vast/RunPod — datasets and checkpoints round-trip through the HF Hub).
- **Hardware track (H1–H6):** a real SO-101 leader/follower pair (~$610 all-in as built — assembled kit; see Budget). Starts any time after Lesson 02; interleaves with the core track. H5/H6 are stretch.

## Syllabus

### Phase 0 — Bootstrap
| # | Lesson | What you leave with |
|---|--------|-------------|
| 00 | [Setup & repo bootstrap](lessons/00-setup/) | Working env (lerobot 0.6.x, MuJoCo), hello-dataset script; hardware ordered ✅ |

### Phase 1 — Data (tutorial §1)
| # | Lesson | What you leave with |
|---|--------|-------------|
| 01 | [LeRobotDataset anatomy](lessons/01-dataset-anatomy/) | The v3 windowing rules predicted and verified at episode boundaries; `window()` + a one-page field guide |
| 02 | [Write your own dataset](lessons/02-dataset-writer/) | A scripted SO-101 dataset on the Hub; state≠action, rate-locking, and `finalize()` each broken on purpose and explained |

### Phase 2 — Classical core (tutorial §2, deepened via MIT/CS223A/16-745)
| # | Lesson | What you leave with |
|---|--------|-------------|
| 03 | [Kinematics](lessons/03-kinematics/) | `fk`/`jacobian` validated against MuJoCo; a planted frame-convention bug diagnosed from its signature; singularity atlas and a λ decision |
| 04 | [Differential IK + feedback](lessons/04-diff-ik-control/) | Open-loop drift, gain ceiling, mismatch, and QP-vs-clipping each predicted then measured; the controller H1 runs |
| 05 | [Optimal control sprint](lessons/05-optimal-control/) | LQR → iLQR swing-up → TVLQR; the backward pass annotated line-by-line; ablations predicted; the RL-mapping table |
| 06 | [Grasp mechanics](lessons/06-grasp-mechanics/) *(optional)* | Antipodal μ threshold derived and confirmed; force optimization as an SOCP; the polar disturbance plot |
| 07 | [Motion planning](lessons/07-motion-planning/) *(optional)* | RRT vs trajopt vs hybrid on 20 problems; the local-minimum failure demonstrated and cured |

### Phase 3 — Reinforcement learning (tutorial §3 + CS 285 core)
| # | Lesson | What you leave with |
|---|--------|-------------|
| 08 | [The RL ladder](lessons/08-rl-ladder/) | CleanRL SAC annotated against tutorial Eqs. 14–17; variance and target-net ablations predicted then run; the sample-efficiency ranking |
| 09 | [Sample-efficient RL: RLPD](lessons/09-rlpd/) | The oversampling factor derived first, then measured; an 8-run composition × LayerNorm study |
| 10 | [HIL-SERL in sim](lessons/10-hil-serl-sim/) | A calibrated reward classifier, the actor/learner system diagram from source, your own intervention-decay curve |
| 11 | [Domain randomization](lessons/11-domain-randomization/) | A pre-registered transfer heatmap showing both failure modes; a shipped width |

### Phase 4 — Generative imitation policies (tutorial §4)
| # | Lesson | What you leave with |
|---|--------|-------------|
| 12 | [Why generative policies](lessons/12-multimodality-lab/) | MSE vs CVAE vs DDPM vs flow matching with C/I metrics predicted first; mode-averaging and covariate shift separated experimentally |
| 13 | [Derivation dossier](lessons/13-derivations/) | ELBO → DDPM simplified loss; FM marginal/conditional equivalence; ε/score/velocity conversions — each numerically checked |
| 14 | [ACT](lessons/14-act/) | Trained ACT on gym-aloha; the chunking ablation; LeRobot's ensembler annotated; the reusable `evaluate()` harness |
| 15 | [Diffusion Policy](lessons/15-diffusion-policy/) | PushT policy; DDPM vs DDIM on one set of weights; FM few-step advantage measured on real action data; the H3 deployment choice |
| 16 | [Async inference](lessons/16-async-inference/) | The idle-avoidance bound derived, g* predicted from measured latency, then validated; the H3 serving config |

### Phase 5 — Generalist policies (tutorial §5 + 2026 landscape)
| # | Lesson | What you leave with |
|---|--------|-------------|
| 17 | [VLA landscape + π0 dissection](lessons/17-vla-landscape-pi0/) | A 2-page comparative note; why prefix KV caching is exact, proved then tested; the speedup curve |
| 18 | [Fine-tune SmolVLA](lessons/18-smolvla-finetune/) | Zero-shot vs LoRA vs full FT with CIs; the layer-skip Pareto; when PEFT is enough |
| 19 | [Comparative VLA lab](lessons/19-vla-comparative/) | Two VLAs, one dataset, one pre-registered protocol; a leaderboard with a threats-to-validity section |

### Phase 6 — Frontier + capstone
| # | Lesson | What you leave with |
|---|--------|-------------|
| 20 | [Beyond imitation](lessons/20-beyond-imitation/) | A taxonomy note; a reward model calibrated and tiered; the Skild S1 claims audit with a dated prediction |
| 21 | [Embodied reasoning](lessons/21-embodied-reasoning/) | Gemini Robotics-ER 2 as planner over a local policy; every failure attributed to a layer |
| 22 | [Capstone](lessons/22-capstone/) | Open-ended project: proposal, protocol, code, video, report |

### Hardware track (SO-101; start after Lesson 02)
| # | Lesson | What you leave with |
|---|--------|-------------|
| H1 | [Bring-up](hardware/H1-build-bringup/) | Calibrated leader/follower pair, cameras measured, teleop video, your controller's circle trace scored through your FK |
| H2 | [Data collection](hardware/H2-data-collection/) | 50-episode pick-place dataset on the Hub with a protocol, card, and failure log |
| H3 | [ACT & Diffusion Policy on real hardware](hardware/H3-act-dp-real/) | Two deployed policies under a pre-registered 20-trial ID/OOD protocol |
| H4 | [VLAs on your arm](hardware/H4-vla-real/) | MolmoAct 2 zero-shot + fine-tuned SmolVLA vs the specialists; one DAgger loop closed |
| H5 | [Real-robot RL](hardware/H5-hil-serl-real/) *(stretch)* | HIL-SERL policy trained live on the arm behind a classifier gate |
| H6 | [Mobile manipulation](hardware/H6-mobile/) *(stretch)* | LeKiwi fetch-and-carry, uncut |

## Budget

| Item | Cost |
|------|------|
| Partabot SO-ARM101 Full Kit, assembled — actual, incl. shipping + tax ([ORDER.md](hardware/ORDER.md)) | $556.79 |
| 2× USB webcams — actual: InnoMaker 1080p wrist + EMEET C960 overhead, incl. tax | $51.70 |
| Cloud GPU credits (Vast.ai/RunPod 4090/A100; ~$1–3 per ACT/DP run, ~$3–8 per SmolVLA fine-tune) | ~$50 |
| **Total** | **~$658** |

Stretch: LeKiwi base kit $179 + Raspberry Pi 5 ~$80 (mobile), then ~$250 more for the XLeRobot dual-arm upgrade.

## Toolchain and version policy

- **LeRobot ≥ 0.6.1** (Python ≥ 3.12, Transformers v5, PyTorch 2.7–2.11). The tutorial targets v0.4.0 and no longer runs verbatim; each lesson notes API deltas (e.g. `pip install "lerobot[training]"` extras split, `sac` → `gaussian_actor`, `lerobot.types` → `lerobot.lerobot_types`). The LeRobotDataset v3 format is unchanged, so all data work is stable.
- **Mac-local:** MuJoCo, gym-pusht/gym-aloha/gym-xarm/gym-hil, Meta-World, Drake — all native on Apple Silicon. Inference and small training runs on `mps`. (`mjpython` is broken on this machine; the managed viewer runs under plain `python`, see Lesson 00 Pitfalls.)
- **Cloud (Linux + NVIDIA):** big ACT/DP runs, SmolVLA/VLA fine-tunes, HIL-SERL learners, LIBERO/RoboCasa/ManiSkill evals (`MUJOCO_GL=egl`). Record locally → push to Hub → train in cloud → pull checkpoint.
- **RL code base:** CleanRL single-file scripts, vendored and annotated in Lesson 08, patched in 09, reused in 11.

## Primary sources

- **Backbone:** [Robot Learning: A Tutorial](https://arxiv.org/abs/2510.12403) ([interactive](https://huggingface.co/spaces/lerobot/robot-learning-tutorial), [code](https://github.com/fracapuano/robot-learning-tutorial)) + [HF Robotics Course](https://huggingface.co/learn/robotics-course) (maintained against current LeRobot)
- **RL theory:** [Berkeley CS 285](https://rail.eecs.berkeley.edu/deeprlcourse/) (Fall 2023 videos, Sp26 homeworks), [Stanford CS 224R](https://cs224r.stanford.edu/)
- **Control/classical:** [CMU 16-745](https://optimalcontrol.ri.cmu.edu/), [MIT Robotic Manipulation](https://manipulation.csail.mit.edu/), [Underactuated Robotics](https://underactuated.csail.mit.edu/), [CS223A via SEE](https://see.stanford.edu/Course/CS223A)
- **Peer syllabus:** [ETH Robot Learning: From Fundamentals to Foundation Models](https://cvg.ethz.ch/lectures/Robot-Learning/) (Mees, Sp26 — open videos + homeworks)
- **Frontier reading spine:** [Physical Intelligence blog](https://www.pi.website/blog) (π0 → FAST → π0.5 → knowledge insulation → real-time chunking → π*0.6 → π0.7) + [Skild S1](https://skild.ai/blogs/s1) (in-context task specification — audited in Lesson 20)
- **RL reference code:** [CleanRL](https://github.com/vwxyzjn/cleanrl) (single-file `dqn.py`, `sac_continuous_action.py`)

## Progress

- [x] 00 · Setup — [ ] 01 — [ ] 02 — [ ] 03 — [ ] 04 — [ ] 05 — [ ] 06* — [ ] 07*
- [ ] 08 — [ ] 09 — [ ] 10 — [ ] 11 — [ ] 12 — [ ] 13 — [ ] 14 — [ ] 15 — [ ] 16
- [ ] 17 — [ ] 18 — [ ] 19 — [ ] 20 — [ ] 21 — [ ] 22 · Capstone
- [ ] H1 — [ ] H2 — [ ] H3 — [ ] H4 — [ ] H5* — [ ] H6*

\* optional/stretch
