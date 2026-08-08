# Lesson 08 — The RL Ladder: REINFORCE → DQN → SAC

**Goal:** the tutorial states Eqs. 11–17 (TD loss, DPG, max-entropy objective, SAC) without derivation or implementation. Build the ladder from scratch, CS 285 style, ending at the exact algorithm LeRobot's HIL-SERL uses.

## Read/Watch
- Tutorial §3.1–3.2 (MDPs through SAC).
- CS 285 lectures 4–9 (policy gradients, actor-critic, Q-learning, SAC): Fall 2023 playlist via https://rail.eecs.berkeley.edu/deeprlcourse/
- Haarnoja et al. 2018 (SAC).

## Build
1. REINFORCE with reward-to-go and a baseline on CartPole; variance-reduction ablation.
2. DQN with replay buffer + target network on a pixel task (or LunarLander); double-DQN comparison.
3. SAC on a MuJoCo continuous-control task (Pendulum → HalfCheetah). Twin critics, entropy temperature auto-tuning.
4. Cross-reference each loss to the tutorial's equation numbers in docstrings.

## Deliverables
- One repo module per algorithm; seeded training curves; a test suite asserting each agent exceeds a reward threshold with fixed seeds.
- Writeup: what each rung adds and why SAC is the real-world workhorse (sample efficiency, off-policy reuse).

## Done when
All three agents pass their thresholded tests reproducibly. HalfCheetah SAC can train overnight on `mps` or in ~1 hr on a rented 4090.
