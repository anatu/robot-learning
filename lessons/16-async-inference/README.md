# Lesson 16 — Async Inference

This lesson separates policy inference from robot control by running LeRobot's PolicyServer and RobotClient as two processes, with a simulated robot standing in for the physical one. You will derive the condition under which the client's action queue never runs empty, measure the latencies that enter that condition, predict where the queue begins to starve as the trigger threshold is varied, and then check the prediction against a sweep. Up to this point inference latency has been a column in a table; from here on it is a constraint on the control loop, and the configuration you settle on is the one H3 deploys on the arm.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 4–5 h desk; experiments run on the Mac (optionally one cloud GPU hour for the remote variant under Going deeper) |
| **Cost** | $0–2 |
| **Prerequisites** | 14 (the ACT checkpoint and the chunk and ensembling concepts), 15 (the DP checkpoint and its synchronized per-decision latencies, which give you two latency regimes), 01 (reading observation specifications) |
| **Feeds into** | H3/H4 (the identical stack, with the policy server on a rented GPU and the real arm as client), 18 (SmolVLA is served the same way) |

## Learning objectives

After this lesson you can:

1. **Derive** the idle-time account for synchronous inference and the asynchronous idle-avoidance bound $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$ from a queue model, and say what every symbol means operationally.
2. **Run** LeRobot's `PolicyServer` and `RobotClient` pair against a simulated robot with your own checkpoints.
3. **Predict** the queue-starvation crossover from measured latency before sweeping the threshold, and reconcile the prediction against the sweep.
4. **Diagnose** when a bound computed from mean latency fails and a tail quantile is needed instead.
5. **Decide** on the (policy, sampler, chunk size, threshold) configuration to ship to H3, from your own measurements.

## Principles

### What synchronous inference wastes

A synchronous control loop executes a chunk of $H_a$ actions, which takes $H_a \Delta t$ seconds at a control period of $\Delta t$, and then stops while the policy server computes the next chunk. That pause lasts one round trip $l_S$, which includes inference, serialization, and any network transfer. The fraction of time spent idle is therefore $l_S / (H_a \Delta t + l_S)$. With a DDPM sampler at 100 steps the idle time is long enough to see as a jerk-and-freeze pattern in the robot's motion, and as policies grow to the size of SmolVLA or π0 the idle fraction becomes the dominant failure rather than a nuisance.

### How the asynchronous stack removes the pause

The asynchronous design decouples the two activities. The client keeps a queue of actions that have been predicted but not yet executed, and it consumes one action per control tick. When the queue drops to a fraction $g$ of the chunk size (the `chunk_size_threshold` parameter), the client sends a fresh observation to the server. The server replies with a new chunk of $H_a$ actions whose early timesteps overlap the actions still sitting in the queue, and the overlapping timesteps are merged by an aggregation function. The default aggregation, `weighted_average`, is the temporal ensembling of Lesson 14, now performed between two processes rather than inside one.

### The idle-avoidance bound

Whether the queue ever empties is a matter of arithmetic. While the server computes, the client continues to consume actions at one per tick, so it uses about $\mathbb{E}[l_S]/\Delta t$ actions during a round trip. The queue never runs dry if the trigger level covers that consumption, which is the condition $g \cdot H_a \ge \mathbb{E}[l_S]/\Delta t$, or equivalently $g \ge g^\* = (\mathbb{E}[l_S]/\Delta t)/H_a$. Setting $g$ below the bound buys idle frames. Setting it far above the bound wastes bandwidth and compute, because a new observation is sent while most of the previous chunk remains unused and the returned chunk is mostly discarded. At $g = 1$ an observation is sent every tick, and at $g = 0$ the loop is synchronous in all but name.

The bound rests on two assumptions: that $l_S$ is stationary, and that the frame rate is fixed. When the latency distribution has a heavy tail, a prediction based on the mean will be optimistic, because a single slow request can drain a queue that was sized for the average. Exercise 5 measures how much that matters.

### Real-time chunking

Aggregation smooths the join between chunks after both chunks have been produced. Real-Time Chunking (RTC), introduced by Physical Intelligence and available in LeRobot since v0.5, intervenes earlier: it adds a guidance term to the denoising process so that the new chunk's overlapping prefix is drawn close to the actions already being executed. RTC is an inference-time technique and applies only to diffusion and flow-matching policies, since it needs a sampler to guide. It is enabled with `--policy.rtc_config.enabled=true`; consult `huggingface.co/docs/lerobot/rtc` for the policy types your installed version supports.

### Clock discipline

Every latency in this lesson is a difference of two timestamps, and the difference is only meaningful if both timestamps come from the same clock. Client send time minus client receive time is a round trip. Server-side compute time is a separate quantity, measured entirely on the server. Client send time minus server receive time compares two machines' clocks and is meaningless unless both are NTP-disciplined and you state as much. On a single machine the distinction is easy to overlook, and it becomes the first bug when the server moves to a rented GPU.

**Carry forward**

- Under synchronous inference the idle fraction is $l_S/(H_a\Delta t + l_S)$; asynchronous inference removes the idle time if and only if $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$, because the queue must hold enough actions to cover one round trip of consumption.
- The threshold $g$ trades queue starvation against wasted chunks, so the useful setting sits just above $g^\*$, and $g^\*$ must be computed from the latency statistic that actually governs starvation, which is the tail rather than the mean when latency has a spread.
- Aggregation on chunk overlap is temporal ensembling carried out across processes; real-time chunking moves the same smoothing into the sampler itself.
- Latency numbers are comparable only when the device was synchronized before both timestamps and both timestamps were taken on the same clock.

| Source | Read for |
|---|---|
| Tutorial §4.4 (Algorithm 1, Fig. 33) | the queue-evolution plots you reproduce, and the exact statement of the bound with its assumptions (stationary $l_S$, fixed fps) |
| LeRobot async docs (`huggingface.co/docs/lerobot/async`) | the verbatim server and client invocations, and the `actions_per_chunk` / `chunk_size_threshold` tuning guidance that you test rather than trust |
| `lerobot/async_inference/robot_client.py` (installed source) | the control loop, and the aggregation-function registry (`aggregate_fn_name`) |
| LeRobot RTC docs | how the guidance term enters the denoiser, and which policies are supported |

## Exercise 1 — Derive the bound [Derive]

The bound is short enough to derive in three lines, and deriving it yourself is what makes the symbols meaningful when you later measure them. In `RESULTS.md`, starting from the queue model in the Principles section, write out (a) the synchronous idle fraction; (b) the asynchronous condition for a never-empty queue; and (c) one sentence per symbol ($l_S$, $\Delta t$, $H_a$, $g$) saying where in the running system each is measured or set. Then state the two assumptions the bound makes and name the exercise below that stresses each of them.

**✅ Checkpoint:** the derivation and the assumption-to-exercise map are written before any code runs.

## Exercise 2 — Stand up the server and a simulated client [Build]

The stock `RobotClient` wraps a physical robot, and you do not have one yet. In this exercise you serve a real policy to a simulated robot instead, which lets you exercise the full asynchronous stack, including its queue and its logging, on the Mac.

1. Install the extra and start the server:
   ```bash
   pip install "lerobot[async]"
   python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
   ```
   The server starts empty; the policy type, checkpoint, and device arrive in the client's first handshake.
2. Write the specification for `sim_client.py` and have an AI tool draft it. The recommended route (route (a), a robot adapter) is a class implementing the `Robot` interface (`connect`, `get_observation`, `send_action`) around `gym_pusht/PushT-v0` (or the Lesson 14 aloha environment), stepped at the dataset's control rate ($\Delta t = 1/\text{fps}$). The stock `RobotClient` is then reused unchanged through its Python API: `RobotClientConfig(robot=..., server_address="localhost:8080", policy_type="act", pretrained_name_or_path="<you>/act_transfercube_base", actions_per_chunk=..., chunk_size_threshold=...)`, a `receive_actions` thread, and `control_loop()`, mirroring the docs example. The client must log natively, per tick, the pair `(tick, queue_len)` (the client already tracks `client.action_queue_size`), idle ticks (the queue is empty and no action can be issued), the client round trip per request, and the server-side compute time per request. Episodes are seeded, and the environment target and policy come from a config file. An alternative route (b) is a minimal client that copies the loop from `robot_client.py` against the environment directly; it teaches the gRPC surface at the cost of more code. Say in `RESULTS.md` which route you chose.
3. Smoke test: serve the Lesson 14 ACT checkpoint for 5 episodes at $g = 0.5$ with `actions_per_chunk = 50`. Success should be within the confidence interval of Lesson 14's synchronous evaluation, because asynchronous execution must not change the task outcome when the queue never starves.

**✅ Checkpoint:** 5 clean episodes with success consistent with Lesson 14, and `debug_visualize_queue_size=True` (or `visualize_action_queue_size(client.action_queue_size)` from `lerobot.async_inference.helpers`) produces a sawtooth queue plot.

## Exercise 3 — Measure latency and predict the crossover [Predict → Run]

The bound says that $g^\*$ can be computed from the round-trip latency alone. In this exercise you measure that latency for three policies and then commit to a predicted crossover before running the sweep, so that Exercise 4 tests the bound rather than fits it.

1. Build the latency table from measurements taken inside the processes: server-side compute $l_S$ per request and client-side round trip, recorded separately, for ACT, DP-DDIM-10, and DP-DDPM-100. Lesson 15's two DP regimes give you a large latency range without involving a network. Report medians and interquartile ranges over at least 100 requests, with the device synchronized inside the server's timing block. On localhost the two columns differ only by serialization.
2. Before running the sweep, write down for ACT and for DP-DDPM-100, at $H_a = 50$ and the environment's $\Delta t$: the value of $g^\*$ computed from the median $\mathbb{E}[l_S]$; which of $g \in \{0.1, 0.3, 0.5, 0.8\}$ you expect to starve the queue; and the queue-trace shape you expect at the lowest and highest $g$.

**✅ Checkpoint:** a 3-row latency table with medians and interquartile ranges, and two predicted $g^\*$ values with starvation predictions, dated before Exercise 4 runs.

## Exercise 4 — Sweep the threshold and test the bound [Predict → Run]

Here the bound meets data. You sweep the threshold across two latency regimes, reproduce the tutorial's queue-evolution figure, and compare the measured crossover against the value predicted in Exercise 3.

1. Sweep $g \in \{0.1, 0.3, 0.5, 0.8\}$ for each regime in {ACT, DP-DDPM-100}, with 10 episodes per cell and fixed seeds. Record the idle fraction, the observation-send rate in observations per second, task success, and the queue trace.
2. Reproduce the tutorial's Figure 33: queue-size evolution per $g$, one panel per regime. You should see a sawtooth that reaches zero when $g$ is too low (starvation) and a shallow sawtooth with frequent sends when $g$ is high.
3. Plot idle fraction against $g$ for each regime, with a vertical line at the predicted $g^\*$. Report the measured crossover against the predicted one for each regime and reconcile against Exercise 3.

**✅ Checkpoint:** the idle fraction is approximately zero for $g > g^\*$ and grows below it, for at least one regime, and both reconciliations are written. Ten episodes per cell is enough for the idle-fraction curve but too few for a claim about success; state that caveat in `RESULTS.md`.

## Exercise 5 — Mean latency or tail latency [Diagnose]

The bound uses the mean of $l_S$, but a queue starves on individual slow requests, not on the average. This exercise finds out, for each regime, which statistic actually predicted the crossover.

1. Predict which regime's measured crossover will miss the mean-based $g^\*$, using the interquartile ranges from Exercise 3: the wider the spread, the worse the mean should predict.
2. Plot the $l_S$ histogram for both regimes. Recompute $g^\*$ from the 90th-percentile latency and mark it on the idle-versus-$g$ plot alongside the mean-based line.
3. Write a one-sentence verdict per regime in `RESULTS.md`: which statistic predicted the crossover, and why. The mechanism is that a single slow request empties a queue that was sized for the mean.

**✅ Checkpoint:** the histograms exist, and for the wide-spread regime the p90-based $g^\*$ sits closer to the measured crossover than the mean-based one, or the reconciliation explains why not.

## Exercise 6 — Compare aggregation functions [Predict → Run]

What happens where two chunks overlap is a design decision rather than an implementation detail, because it is the same operation as Lesson 14's temporal ensembling. This exercise compares the default weighted average against a function that simply takes the newest prediction.

1. Before running, write down the direction of the jerk difference you expect between `weighted_average` and `latest_wins`, and whether you expect success to move.
2. Register a `latest_wins` function in the client's aggregation registry (see `robot_client.py`). Fix $g = 0.7$ and use DP-DDIM-10. Compare mean squared jerk (Lesson 14's metric, computed on the executed actions) and success over 20 episodes.

**✅ Checkpoint:** an aggregation table with jerk and success for both functions, and the reconciliation written.

## Exercise 7 — Real-time chunking [Read]

The question for this exercise is what RTC constrains that aggregation cannot, and whether your installed version supports it for Diffusion Policy. Read the support matrix in the RTC docs. If RTC is supported, run one arm at $g = 0.7$ with `--policy.rtc_config.enabled=true` and compare jerk at the chunk boundaries, success, and queue behaviour against the `weighted_average` row from Exercise 6. If it is not supported, write the deferral in `RESULTS.md` with a forward pointer to the flow-matching policy of Lesson 18, rather than reporting a result you did not obtain.

**✅ Checkpoint:** the RTC row is filled in, or the deferral is documented against the installed version's support matrix.

## Exercise 8 — Choose the H3 configuration [Decide]

The measurements above exist so that the deployment configuration for the real arm can be chosen from evidence. For the SO-101 at 30 fps, choose the policy, sampler, $H_a$, $g$, and aggregation or RTC setting, and cite the measurement that justifies each choice, one row of your tables per knob. Then state what you would re-measure once the server sits on a rented GPU across a network, which is the remote variant under Going deeper.

**✅ Checkpoint:** the configuration and its supporting rows are in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `sim_client.py` (plus the adapter) | runs both environment targets from a config; seeded; logs latency, queue, and idle ticks natively |
| `run_sweep.py` | reproduces the sweep, the Figure 33 panels, and the idle-versus-$g$ plot from one command |
| `plots/` | queue-evolution panels per $g$ per regime; idle-versus-$g$ with mean- and p90-based $g^\*$ lines; latency histograms |
| `bound_validation.md` | per regime: $\mathbb{E}[l_S]$ (median and p90), $\Delta t$, $H_a$, predicted $g^\*$ (both), measured crossover |
| `RESULTS.md` | the Exercise 1 derivation; the Exercise 3–6 predictions with reconciliations; the aggregation and RTC table; the H3 configuration with its rows; at most 12 sentences of interpretation |

## Done when

- [ ] Asynchronous inference at a healthy $g$ matches synchronous success while cutting the idle fraction to approximately zero, with both numbers in the report.
- [ ] $g^\*$ was predicted from measured latency before the sweep for both regimes, and reconciled after it.
- [ ] The mean-versus-tail diagnosis is written with the histogram that supports it.
- [ ] The aggregation comparison is quantified with jerk, and RTC is compared or its deferral documented.
- [ ] A stranger could rerun the whole sweep with `python run_sweep.py`.

## Self-check

1. Rederive $g^\* = (\mathbb{E}[l_S]/\Delta t)/H_a$ from the queue model in three lines. Which assumption breaks first in practice, and which exercise showed it?
2. $g = 1.0$ sends an observation every tick. What failure mode does the tutorial's analysis predict for compute, and what benefit does the SmolVLA paper's framing claim in exchange?
3. Why is aggregation on overlap the same mathematical object as Lesson 14's temporal ensembling, and what is the one operational difference?
4. RTC modifies denoising rather than averaging outputs. Why can it not apply to ACT?
5. In a remote deployment, which timestamps may legitimately be subtracted, and why is client-send minus server-receive not one of them?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Client hangs at startup | server not up yet, port blocked, or gRPC max-message-size exceeded by camera frames | start the server first; check `lsof -i :8080`; downscale images to the policy's training resolution |
| Queue plot flatlines at maximum | client fps effectively slower than intended (env rendering in the loop) | step the env headless; render only saved episodes |
| Success drops only at high $g$ | aggregation thrash: constant replanning with `latest_wins`-like behaviour | confirm `weighted_average`; inspect executed-action jerk at chunk boundaries |
| Idle frames despite $g > g^\*$ | latency spread: a slow-tail request starved the queue | use tail-quantile $l_S$ in the bound; report both predictions (Exercise 5) |
| Remote: enormous $l_S$ variance | camera frames over TCP through the tunnel | JPEG-compress observations, or measure with state-only observations to isolate transport |
| RTC flag rejected | installed version lacks RTC for this policy type | check the RTC doc's support matrix; defer to the Lesson 18 arm as specified |
| Latencies look too good | measuring dispatch rather than compute (asynchronous CUDA or MPS) | synchronize the device inside the server timing block |

## Going deeper

- **The remote variant.** Do this before H3 if you can: run the policy server on a rented GPU box and the client on the Mac, over an SSH tunnel or Tailscale, with `--server_address=<host>:8080`. Re-measure $\mathbb{E}[l_S]$ with the network included and confirm that the bound still predicts the crossover. This is a rehearsal of the deployment topology H3 and H4 use.
- **The latency cliff.** Serve DP-DDPM-100 over the remote link and find the (fps, $H_a$, $g$) frontier at which the task still succeeds. Then write the two-paragraph design memo that H3 will use.

## References

- LeRobot team. *Robot Learning: A Tutorial*, §4.4 (Algorithm 1, Fig. 33). arXiv:2510.12403.
- LeRobot async-inference docs: huggingface.co/docs/lerobot/async (the commands above are mirrored from there) and blog: huggingface.co/blog/async-robot-inference.
- Shukor et al. *SmolVLA*, 2025, the section on async inference (the source of the $g$ notation). arXiv:2506.01844.
- Black et al. / Physical Intelligence, *Real-Time Chunking*, via the LeRobot RTC docs: huggingface.co/docs/lerobot/rtc.
