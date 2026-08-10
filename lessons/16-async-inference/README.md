# Lesson 16 — Async Inference

Deploy the tutorial's own engineering contribution — a decoupled PolicyServer/RobotClient — against a simulated robot, then validate its analytic idle-avoidance bound with your own latency measurements. This is the lesson where "inference latency" stops being a table column and starts being a control-loop constraint.

| | |
|---|---|
| **Phase** | 4 — Generative imitation policies |
| **Time** | 5–7 h desk; experiments run on the Mac (optionally one cloud GPU hour for the remote-server variant) |
| **Cost** | $0–2 |
| **Prerequisites** | 14 (ACT checkpoint + the chunk/ensembling mental model), 15 (DP checkpoint + measured per-decision latencies — your two latency regimes), 01 (reading observation specs) |
| **Feeds into** | H3/H4 (identical stack, policy server on a rented GPU, real arm as client), 18 (SmolVLA is served the same way) |

## Learning objectives

After this lesson you can:

1. **Derive** the sync-inference idle-time account and the async idle-avoidance bound $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$ from a queue model, and say what every symbol means operationally.
2. **Run** LeRobot's `PolicyServer`/`RobotClient` pair, locally and across machines, with your own checkpoints.
3. **Instrument** the loop: queue-depth traces, observation-send rate, idle frames, and server latency, with sound clock discipline.
4. **Validate** the bound empirically across a sweep of `chunk_size_threshold` and two policy-latency regimes.
5. **Compare** chunk-overlap aggregation strategies and Real-Time Chunking (RTC) against the plain async stack.

## Background

**Sync inference wastes exactly $l_S$ every $H_a$ steps.** A synchronous loop executes a chunk of $H_a$ actions in $H_a \Delta t$ seconds, then *stops* for one server round-trip $l_S$ (inference + serialization + network) before the next chunk exists. Idle fraction $= l_S / (H_a \Delta t + l_S)$. At DP-DDPM latencies this is visible jerk-and-freeze; as policies grow (SmolVLA, π0), it becomes the dominant failure.

**Async decouples planning from acting.** The client holds a queue of not-yet-executed actions and consumes one per control tick. When the queue drops to a fraction $g$ of the chunk size (`chunk_size_threshold`), the client ships a fresh observation to the server; the server returns a new $H_a$-chunk that *overlaps* the actions still queued; overlapping timesteps are merged by an aggregation function (default `weighted_average` — Lesson 14's ensembling, now living between processes).

**The bound.** While the server computes, the client keeps consuming ≈ $\mathbb{E}[l_S]/\Delta t$ actions. The queue never empties iff the trigger level covers that consumption: $g \cdot H_a \ge \mathbb{E}[l_S]/\Delta t$, i.e. $g \ge (\mathbb{E}[l_S]/\Delta t)/H_a$. Below the bound you buy idle frames; far above it you pay bandwidth and compute for chunks you mostly discard ($g \to 1$ = send every tick; $g \to 0$ = sync in disguise). The sweep in Part 3 maps this whole curve.

**RTC is the other half of the story.** Aggregation smooths *between* chunks after the fact; Real-Time Chunking (Physical Intelligence; in LeRobot since v0.5) instead constrains the *denoising process itself* — a guidance term forces the new chunk's overlapping prefix to stay close to the actions already being executed. Inference-time only, for diffusion/flow-based policies (`--policy.rtc_config.enabled=true`; check `huggingface.co/docs/lerobot/rtc` for which policy types your installed version supports).

| Source | Read for |
|---|---|
| Tutorial §4.4 (Algorithm 1, Fig. 33) | the queue-evolution plots you'll reproduce; the exact statement of the bound and its assumptions (stationary $l_S$, fixed fps) |
| LeRobot async docs (`huggingface.co/docs/lerobot/async`) | verbatim server/client invocations; the `actions_per_chunk` / `chunk_size_threshold` tuning guidance you'll test rather than trust |
| `lerobot/async_inference/robot_client.py` (installed source) | the control loop you're about to imitate for sim; the aggregation-function registry (`aggregate_fn_name`) |
| LeRobot RTC docs | how the guidance term enters the denoiser; supported policies |

## Part 1 — Stand the stack up (≈1–2 h)

The stock `RobotClient` wraps a *physical* robot. You don't have one yet — so you'll serve a real policy to a simulated one.

1. Install the extra: `pip install "lerobot[async]"` (gRPC dependencies). Start the server:
   ```bash
   python -m lerobot.async_inference.policy_server --host=127.0.0.1 --port=8080
   ```
   The server boots *empty* — policy type, checkpoint, and device arrive in the client's first handshake.
2. Build `sim_client.py`: a client that drives `gym_pusht/PushT-v0` (or the L14 aloha env) instead of motors. Two routes, pick one and say which in `RESULTS.md`:
   - **(a) Robot-adapter:** implement the `Robot` interface (`connect`, `get_observation`, `send_action`) around the gym env, then reuse `RobotClient` unchanged via its Python API (`RobotClientConfig(robot=..., server_address="localhost:8080", policy_type="act", pretrained_name_or_path="<you>/act_transfercube_base", actions_per_chunk=..., chunk_size_threshold=...)`, a `receive_actions` thread + `control_loop()` — mirror the docs example).
   - **(b) Minimal client:** copy `robot_client.py`'s loop (obs → send when queue ≤ $g H_a$ → pop action each tick at fixed fps) against the env directly.
   Route (a) is less code and keeps you on the maintained path; (b) teaches you the gRPC surface. Either way, keep the env stepped at the dataset's control rate (Δt = 1/fps).
3. Smoke test: serve the Lesson 14 ACT checkpoint, run 5 episodes at $g = 0.5$, `actions_per_chunk = 50`. Success should be within CI of Lesson 14's *synchronous* eval — async must not change task outcome when the queue never starves.

**✅ Checkpoint:** 5 clean episodes; success consistent with L14; `debug_visualize_queue_size=True` (or `visualize_action_queue_size(client.action_queue_size)` from `lerobot.async_inference.helpers`) produces a sawtooth queue plot.

## Part 2 — Instrument honestly (≈1 h)

Numbers first, sweeps second. All measurements from *inside* the processes, not wall-clock guesswork.

1. **Server latency $l_S$:** log per-request compute time server-side; log client-side round-trip separately. On localhost these differ by serialization only; over a network the gap is your transport cost. Never mix timestamps from two unsynchronized clocks — always difference timestamps taken on the *same* machine (send→receive on the client is fine; client-send vs server-receive is not, unless NTP-disciplined and stated).
2. **Queue trace:** record `(tick, queue_len)` every control tick (the client already tracks this — `client.action_queue_size`).
3. **Idle frames:** a tick where the queue is empty and no action can be issued. Count them; idle fraction = idle ticks / total ticks.
4. Baseline table: $\mathbb{E}[l_S]$ and its spread for ACT and for DP at two regimes from Lesson 15 — DDPM-100 (slow) and DDIM-10 (fast) — on the serving device. These two regimes are your experimental handle on $l_S$ without touching the network.

**✅ Checkpoint:** a 3-row latency table (ACT, DP-DDIM-10, DP-DDPM-100) with medians + IQRs; queue traces attach cleanly to episodes.

## Part 3 — The sweep, and the bound (the core; ≈2 h)

1. Sweep $g \in \{0, 0.3, 0.5, 0.7, 1.0\}$ × policy-regime ∈ {ACT, DP-DDIM-10, DP-DDPM-100}, ≥ 20 episodes each, fixed seeds. Record: idle fraction, observation-send rate (obs/s), task success, and the queue trace.
2. Reproduce the tutorial's Fig. 33: queue-size evolution per $g$, one panel per regime. The shapes to expect: $g$ too low → sawtooth hitting zero (starvation); $g$ high → shallow sawtooth, frequent sends.
3. **Validate the bound.** For each regime compute the predicted threshold $g^\* = (\mathbb{E}[l_S]/\Delta t)/H_a$ from Part 2's measured $\mathbb{E}[l_S]$. Plot idle fraction vs $g$ with a vertical line at $g^\*$: idle fraction should fall to ≈ 0 for $g > g^\*$ and grow below it. Report measured-vs-predicted crossover per regime.
4. Stress the assumption: the bound uses the *mean* $l_S$. With DP-DDPM-100 (largest latency spread), check whether the crossover sits at the mean-based prediction or needs a tail quantile — one sentence of verdict in `RESULTS.md`.

**✅ Checkpoint:** measured crossover within noise of $g^\*$ for at least two regimes; where it isn't, the latency-spread explanation is written down with the histogram that supports it.

## Part 4 — Aggregation and RTC (≈1–2 h)

What happens on chunk overlap is a policy decision, not a detail.

1. **Aggregation A/B:** `weighted_average` vs a `latest_wins` function you register in the client's aggregation registry (see `robot_client.py`). Fix $g = 0.7$, DP-DDIM-10. Compare action jerk (L14's metric, computed on executed actions) and success over ≥ 20 episodes.
2. **RTC arm:** enable RTC for the diffusion-family policy per the RTC docs (`--policy.rtc_config.enabled=true`); if your installed version doesn't support RTC for DP, run this arm with the flow-matching policy in Lesson 18 instead and leave a forward-pointer here — don't fake it. Compare plain-async vs RTC at the same $g$: jerk at chunk boundaries, success, queue behavior.
3. **Remote variant (optional but recommended before H3):** policy server on a rented GPU box, client on the Mac (SSH tunnel or Tailscale; plain `--server_address=<host>:8080`). Re-measure $\mathbb{E}[l_S]$ — network now inside it — and confirm the bound still predicts the crossover. This is a dress rehearsal for H3/H4's deployment topology.

**✅ Checkpoint:** aggregation table (jerk + success, 2 functions); RTC comparison (or the documented deferral); if run remote: a second bound-validation row with network-inclusive $l_S$.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `sim_client.py` (+ adapter if route (a)) | runs both env targets from a config; seeded; logs latency/queue/idle natively |
| `plots/` | queue-evolution panels per $g$ per regime (Fig. 33 reproduction); idle-vs-$g$ with $g^\*$ lines; latency histograms |
| `bound_validation.md` table | per regime: $\mathbb{E}[l_S]$, $\Delta t$, $H_a$, predicted $g^\*$, measured crossover |
| aggregation + RTC table | jerk + success per strategy, ≥ 20 episodes each |
| `RESULTS.md` | the sync-vs-async-vs-RTC verdict in ≤ 12 sentences, including the one config you'd ship for H3 and why |

## Done when

- [ ] Async at healthy $g$ matches sync success while cutting idle fraction to ≈ 0 — both numbers in the report.
- [ ] Bound validated: measured crossover vs $g^\*$ agree within noise for ≥ 2 latency regimes, deviations explained via the latency histogram.
- [ ] Aggregation A/B quantified with jerk, not adjectives.
- [ ] RTC compared or its deferral documented against the installed version's support matrix.
- [ ] A stranger could rerun the whole sweep from `python run_sweep.py`.

## Self-check

1. Rederive $g^\* = (\mathbb{E}[l_S]/\Delta t)/H_a$ from the queue model in three lines. Which assumption breaks first in practice, and which experiment in this lesson showed it?
2. $g = 1.0$ sends an observation every tick. What failure mode does the tutorial's analysis predict for compute, and what *benefit* does the SmolVLA-paper framing claim in exchange?
3. Why is aggregation-on-overlap the same mathematical object as Lesson 14's temporal ensembling, and what's the one operational difference?
4. RTC modifies denoising rather than averaging outputs. Why can't it apply to ACT?
5. In the remote variant, which timestamps are legal to subtract, and why is client-send minus server-receive not one of them?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Client hangs at startup | server not up yet, port blocked, or gRPC max-message-size exceeded by camera frames | start server first; check `lsof -i :8080`; downscale images to the policy's training resolution |
| Queue plot flatlines at max | client fps effectively slower than intended (env render in the loop) | step env headless; render only saved episodes |
| Success drops only at high $g$ | aggregation thrash — constant replanning with `latest_wins`-like behavior | confirm `weighted_average`; inspect executed-action jerk at chunk boundaries |
| Idle frames despite $g > g^\*$ | latency spread — a slow-tail request starved the queue | use tail-quantile $l_S$ in the bound; report both predictions |
| Remote: enormous $l_S$ variance | camera frames over TCP through the tunnel | JPEG-compress observations or measure with state-only obs to isolate transport |
| RTC flag rejected | installed version lacks RTC for this policy type | check the RTC doc's support matrix; defer to L18 arm as specced |
| Latencies look too good | measuring dispatch, not compute (async CUDA/MPS) | synchronize the device inside the server timing block |

## Stretch

Push one latency regime to the cliff: serve DP-DDPM-100 over the remote link and find the (fps, $H_a$, $g$) frontier where the task still succeeds. Then write the two-paragraph design memo H3 will actually use: for the SO-101 at 30 fps, which policy, sampler, chunk size, and $g$ — with the measurements that justify each choice.

## References

- LeRobot team. *Robot Learning: A Tutorial*, §4.4 (Algorithm 1, Fig. 33). arXiv:2510.12403.
- LeRobot async-inference docs: huggingface.co/docs/lerobot/async (verbatim commands mirrored above) and blog: huggingface.co/blog/async-robot-inference.
- Shukor et al. *SmolVLA*, 2025 — §on async inference (the $g$ notation source). arXiv:2506.01844.
- Black et al. / Physical Intelligence, *Real-Time Chunking* — via LeRobot RTC docs: huggingface.co/docs/lerobot/rtc.
