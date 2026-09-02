# Lesson 16 — Async Inference

Decouple inference from control — PolicyServer/RobotClient against a simulated robot — and validate the tutorial's analytic idle-avoidance bound with your own latency measurements. This is where "inference latency" stops being a table column and becomes a control-loop constraint.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 4–5 h desk; experiments run on the Mac (optionally one cloud GPU hour for the remote variant under Going deeper) |
| **Cost** | $0–2 |
| **Prerequisites** | 14 (ACT checkpoint + the chunk/ensembling mental model), 15 (DP checkpoint + synchronized per-decision latencies — your two latency regimes), 01 (reading observation specs) |
| **Feeds into** | H3/H4 (identical stack, policy server on a rented GPU, real arm as client), 18 (SmolVLA is served the same way) |

## Learning objectives

After this lesson you can:

1. **Derive** the sync-inference idle-time account and the async idle-avoidance bound $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$ from a queue model, and say what every symbol means operationally.
2. **Run** LeRobot's `PolicyServer`/`RobotClient` pair against a simulated robot with your own checkpoints.
3. **Predict** the queue-starvation crossover from measured latency *before* sweeping the threshold, and reconcile against the sweep.
4. **Diagnose** when the mean-latency bound fails and a tail quantile is needed.
5. **Decide** the (policy, sampler, chunk size, threshold) configuration to ship to H3, from your measurements.

## Principles

**Sync inference wastes exactly $l_S$ every $H_a$ steps.** A synchronous loop executes a chunk of $H_a$ actions in $H_a \Delta t$ seconds, then *stops* for one server round-trip $l_S$ (inference + serialization + network) before the next chunk exists. Idle fraction $= l_S / (H_a \Delta t + l_S)$. At DP-DDPM latencies this is visible jerk-and-freeze; as policies grow (SmolVLA, π0), it becomes the dominant failure.

**Async decouples planning from acting.** The client holds a queue of not-yet-executed actions and consumes one per control tick. When the queue drops to a fraction $g$ of the chunk size (`chunk_size_threshold`), the client ships a fresh observation to the server; the server returns a new $H_a$-chunk that *overlaps* the actions still queued; overlapping timesteps are merged by an aggregation function (default `weighted_average` — Lesson 14's temporal ensembling, now living between processes).

**The bound.** While the server computes, the client keeps consuming ≈ $\mathbb{E}[l_S]/\Delta t$ actions. The queue never empties iff the trigger level covers that consumption: $g \cdot H_a \ge \mathbb{E}[l_S]/\Delta t$, i.e. $g \ge g^\* = (\mathbb{E}[l_S]/\Delta t)/H_a$. Below the bound you buy idle frames; far above it you pay bandwidth and compute for chunks you mostly discard ($g \to 1$ = send every tick; $g \to 0$ = sync in disguise). The bound assumes a stationary $l_S$ and a fixed fps; a heavy latency tail breaks the mean-based prediction, and Exercise 5 finds out where.

**RTC is the other half.** Aggregation smooths *between* chunks after the fact; Real-Time Chunking (Physical Intelligence; in LeRobot since v0.5) instead constrains the *denoising process itself* — a guidance term forces the new chunk's overlapping prefix to stay close to the actions already being executed. Inference-time only, for diffusion/flow-based policies (`--policy.rtc_config.enabled=true`; check `huggingface.co/docs/lerobot/rtc` for which policy types your installed version supports).

**Clock discipline.** Only subtract timestamps taken on the same machine. Client send→receive is a round-trip; server-side compute is a different number; client-send minus server-receive is meaningless unless NTP-disciplined and stated.

**Carry forward**

- Idle fraction under sync $= l_S/(H_a\Delta t + l_S)$; async removes it iff $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$.
- $g$ trades starvation against wasted chunks; the sweet spot sits just above $g^\*$ computed from the *right* latency statistic.
- Overlap aggregation is temporal ensembling across processes; RTC moves the smoothing into the sampler.
- Latency numbers are only comparable with synchronized devices and same-clock timestamps.

| Source | Read for |
|---|---|
| Tutorial §4.4 (Algorithm 1, Fig. 33) | the queue-evolution plots you reproduce; the exact statement of the bound and its assumptions (stationary $l_S$, fixed fps) |
| LeRobot async docs (`huggingface.co/docs/lerobot/async`) | verbatim server/client invocations; the `actions_per_chunk` / `chunk_size_threshold` tuning guidance you test rather than trust |
| `lerobot/async_inference/robot_client.py` (installed source) | the control loop; the aggregation-function registry (`aggregate_fn_name`) |
| LeRobot RTC docs | how the guidance term enters the denoiser; supported policies |

## Exercise 1 — The bound, on paper [Derive]

Tests objective 1. In `RESULTS.md`, from the queue model: (a) the sync idle fraction; (b) the async condition for a never-empty queue, in three lines; (c) one sentence per symbol ($l_S$, $\Delta t$, $H_a$, $g$) saying where in the running system each is measured or set. State the two assumptions the bound makes and which exercise below stresses each.

**✅ Checkpoint:** the derivation and the assumption→exercise map are written before any code runs.

## Exercise 2 — Stand the stack up [Build]

Tests objective 2. The stock `RobotClient` wraps a *physical* robot; you serve a real policy to a simulated one.

1. Install and start the server:
   ```bash
   pip install "lerobot[async]"
   python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
   ```
   The server boots *empty* — policy type, checkpoint, and device arrive in the client's first handshake.
2. Spec for `sim_client.py` (route (a), robot-adapter): a class implementing the `Robot` interface (`connect`, `get_observation`, `send_action`) around `gym_pusht/PushT-v0` (or the Lesson 14 aloha env), stepped at the dataset's control rate ($\Delta t = 1/\text{fps}$); reuse `RobotClient` unchanged via its Python API — `RobotClientConfig(robot=..., server_address="localhost:8080", policy_type="act", pretrained_name_or_path="<you>/act_transfercube_base", actions_per_chunk=..., chunk_size_threshold=...)`, a `receive_actions` thread + `control_loop()`, mirroring the docs example. Native logging of per-tick `(tick, queue_len)` (the client tracks `client.action_queue_size`), idle ticks (queue empty, no action issued), client round-trip per request, and server-side compute per request. Seeded episodes; env target and policy from a config.
   (Route (b), a minimal client copying `robot_client.py`'s loop against the env directly, teaches the gRPC surface; say in `RESULTS.md` which you chose.)
3. Smoke test: serve the Lesson 14 ACT checkpoint, 5 episodes at $g = 0.5$, `actions_per_chunk = 50`. Success should be within CI of Lesson 14's *synchronous* eval — async must not change task outcome when the queue never starves.

**✅ Checkpoint:** 5 clean episodes; success consistent with L14; `debug_visualize_queue_size=True` (or `visualize_action_queue_size(client.action_queue_size)` from `lerobot.async_inference.helpers`) produces a sawtooth queue plot.

## Exercise 3 — Measure, then predict the crossover [Predict → Run]

Tests objective 3: $g^\*$ is computable before the sweep.

1. Latency table, from *inside* the processes: server-side compute $l_S$ per request and client-side round-trip, separately, for ACT, DP-DDIM-10, and DP-DDPM-100 (Lesson 15's two DP regimes give you a large latency handle without touching the network). Medians + IQRs over ≥ 100 requests; device synchronized inside the server timing block. On localhost the two columns differ by serialization only.
2. **Write first**, for ACT and DP-DDPM-100 at $H_a = 50$ and the env's $\Delta t$: $g^\*$ from the median $\mathbb{E}[l_S]$; which of $g \in \{0.1, 0.3, 0.5, 0.8\}$ you expect to starve; the queue-trace shape you expect at the lowest and highest $g$.

**✅ Checkpoint:** a 3-row latency table with medians + IQRs; two predicted $g^\*$ values and the starvation predictions, dated before Exercise 4 runs.

## Exercise 4 — The sweep and the bound [Predict → Run]

Tests objectives 1 and 3 together: the bound against data.

1. Sweep $g \in \{0.1, 0.3, 0.5, 0.8\}$ × regime ∈ {ACT, DP-DDPM-100}, 10 episodes each, fixed seeds. Record idle fraction, observation-send rate (obs/s), task success, and the queue trace.
2. Reproduce the tutorial's Fig. 33: queue-size evolution per $g$, one panel per regime. Expected: $g$ too low → sawtooth hitting zero (starvation); $g$ high → shallow sawtooth, frequent sends.
3. Plot idle fraction vs $g$ per regime with a vertical line at the predicted $g^\*$. Report measured-vs-predicted crossover per regime and reconcile against Exercise 3.

**✅ Checkpoint:** idle fraction ≈ 0 for $g > g^\*$ and growing below it, for at least one regime; both reconciliations written. (10 episodes per cell suffices for the idle-fraction curve; it is too few for a success claim — state that caveat.)

## Exercise 5 — Mean or tail? [Diagnose]

Tests objective 4: the bound uses the *mean* $l_S$; the queue starves on the *tail*.

1. **Predict** which regime's measured crossover misses the mean-based $g^\*$, from the IQRs in Exercise 3 (the wider the spread, the worse the mean predicts).
2. Plot the $l_S$ histogram for both regimes. Recompute $g^\*$ from the p90 latency and mark it on the idle-vs-$g$ plot.
3. One-sentence verdict per regime in `RESULTS.md`: which statistic predicted the crossover, and the mechanism (a single slow request empties a queue sized for the mean).

**✅ Checkpoint:** histograms exist; for the wide-spread regime the p90-based $g^\*$ sits closer to the measured crossover than the mean-based one, or the reconciliation says why not.

## Exercise 6 — Aggregation on overlap [Predict → Run]

Tests the ensembling-across-processes principle: what happens on chunk overlap is a policy decision, not a detail.

1. **Write first:** the direction of the jerk difference between `weighted_average` and `latest_wins`, and whether you expect success to move.
2. Register a `latest_wins` function in the client's aggregation registry (see `robot_client.py`). Fix $g = 0.7$, DP-DDIM-10. Compare mean squared jerk (Lesson 14's metric, on executed actions) and success over 20 episodes.

**✅ Checkpoint:** aggregation table (jerk + success, 2 functions); reconciliation written.

## Exercise 7 — RTC [Read]

Question: what does RTC constrain that aggregation cannot, and does your installed version support it for DP? Read the RTC docs' support matrix. If supported, run one arm at $g = 0.7$ with `--policy.rtc_config.enabled=true` and compare jerk at chunk boundaries, success, and queue behavior against Exercise 6's `weighted_average` row. If not, write the deferral and a forward pointer to the flow-matching policy in Lesson 18 — don't fake it.

**✅ Checkpoint:** RTC row filled, or the documented deferral against the installed version's support matrix.

## Exercise 8 — The H3 configuration [Decide]

Tests objective 5. For the SO-101 at 30 fps: choose policy, sampler, $H_a$, $g$, and aggregation/RTC, with the measurement that justifies each — one row of your tables per knob. State what you would re-measure once the server sits on a rented GPU across a network (Going deeper).

**✅ Checkpoint:** the configuration and its supporting rows are in `RESULTS.md`.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `sim_client.py` (+ adapter) | runs both env targets from a config; seeded; logs latency/queue/idle natively |
| `run_sweep.py` | reproduces the sweep, the Fig. 33 panels, and the idle-vs-$g$ plot from one command |
| `plots/` | queue-evolution panels per $g$ per regime; idle-vs-$g$ with mean- and p90-based $g^\*$ lines; latency histograms |
| `bound_validation.md` | per regime: $\mathbb{E}[l_S]$ (median, p90), $\Delta t$, $H_a$, predicted $g^\*$ (both), measured crossover |
| `RESULTS.md` | Exercise 1 derivation; Exercise 3–6 predictions with reconciliations; aggregation + RTC table; the H3 configuration with its rows; ≤ 12 sentences of interpretation |

## Done when

- [ ] Async at healthy $g$ matches sync success while cutting idle fraction to ≈ 0 — both numbers in the report.
- [ ] $g^\*$ predicted from measured latency *before* the sweep, for both regimes, and reconciled after.
- [ ] Mean-vs-tail diagnosis written with the histogram that supports it.
- [ ] Aggregation A/B quantified with jerk, not adjectives; RTC compared or its deferral documented.
- [ ] A stranger could rerun the whole sweep from `python run_sweep.py`.

## Self-check

1. Rederive $g^\* = (\mathbb{E}[l_S]/\Delta t)/H_a$ from the queue model in three lines. Which assumption breaks first in practice, and which exercise showed it?
2. $g = 1.0$ sends an observation every tick. What failure mode does the tutorial's analysis predict for compute, and what *benefit* does the SmolVLA-paper framing claim in exchange?
3. Why is aggregation-on-overlap the same mathematical object as Lesson 14's temporal ensembling, and what is the one operational difference?
4. RTC modifies denoising rather than averaging outputs. Why can't it apply to ACT?
5. In a remote deployment, which timestamps are legal to subtract, and why is client-send minus server-receive not one of them?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Client hangs at startup | server not up yet, port blocked, or gRPC max-message-size exceeded by camera frames | start server first; check `lsof -i :8080`; downscale images to the policy's training resolution |
| Queue plot flatlines at max | client fps effectively slower than intended (env render in the loop) | step env headless; render only saved episodes |
| Success drops only at high $g$ | aggregation thrash — constant replanning with `latest_wins`-like behavior | confirm `weighted_average`; inspect executed-action jerk at chunk boundaries |
| Idle frames despite $g > g^\*$ | latency spread — a slow-tail request starved the queue | use tail-quantile $l_S$ in the bound; report both predictions (Exercise 5) |
| Remote: enormous $l_S$ variance | camera frames over TCP through the tunnel | JPEG-compress observations or measure with state-only obs to isolate transport |
| RTC flag rejected | installed version lacks RTC for this policy type | check the RTC doc's support matrix; defer to the L18 arm as specced |
| Latencies look too good | measuring dispatch, not compute (async CUDA/MPS) | synchronize the device inside the server timing block |

## Going deeper

- **Remote variant** (do it before H3 if you can): policy server on a rented GPU box, client on the Mac (SSH tunnel or Tailscale; plain `--server_address=<host>:8080`). Re-measure $\mathbb{E}[l_S]$ with the network inside it and confirm the bound still predicts the crossover — a dress rehearsal for H3/H4's deployment topology.
- **The cliff.** Serve DP-DDPM-100 over the remote link and find the (fps, $H_a$, $g$) frontier where the task still succeeds. Then write the two-paragraph design memo H3 will actually use.

## References

- LeRobot team. *Robot Learning: A Tutorial*, §4.4 (Algorithm 1, Fig. 33). arXiv:2510.12403.
- LeRobot async-inference docs: huggingface.co/docs/lerobot/async (verbatim commands mirrored above) and blog: huggingface.co/blog/async-robot-inference.
- Shukor et al. *SmolVLA*, 2025 — §on async inference (the $g$ notation source). arXiv:2506.01844.
- Black et al. / Physical Intelligence, *Real-Time Chunking* — via LeRobot RTC docs: huggingface.co/docs/lerobot/rtc.
