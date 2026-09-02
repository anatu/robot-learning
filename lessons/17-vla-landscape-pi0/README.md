# Lesson 17 — The VLA Landscape and a π0 Dissection

This lesson has two halves. In the first you map the design space of vision-language-action models as it stands in mid-2026, producing a short comparative note that Lessons 18 and 19 use when choosing what to fine-tune. In the second you take one architectural idea from π0, the blockwise-causal attention mask, and show first on paper and then with a passing test why it makes caching the prefix's keys and values across denoising steps exact rather than approximate. The second half is small enough to run on a Mac, and the caching argument recurs in every flow-matching VLA you will meet from here on.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~2 sessions: 3–4 h for the comparative note, 3–4 h for the mask and cache module plus the benchmark |
| **Cost** | $0; everything runs on the Mac, since the module is small enough for `mps` or `cpu` |
| **Prerequisites** | 13 (flow matching, which is exactly the machinery of π0's action head), 14 (you have trained a transformer policy and know what a chunk is), 12 (why generative heads are used at all) |
| **Feeds into** | 18/19 (you fine-tune and compare the models mapped here), 20 (world-action models extend this taxonomy), H4 (MolmoAct2 and SmolVLA on your arm) |

## Learning objectives

After this lesson you can:

1. **Place** any VLA on the design axes of backbone, action interface, action head, data mix, and inference scheme, and predict its latency and generalization profile from that placement.
2. **Explain** why discrete action tokens lost to continuous generative heads for high-rate control, and how FAST partially reversed that verdict.
3. **Derive** why caching the prefix's keys and values across flow-matching denoising steps is exact under the blockwise-causal mask, and identify the mask entries on which the exactness depends.
4. **Predict** the latency gain from caching as a function of denoising steps and prefix length, and then measure it.
5. **Diagnose** a broken caching invariant from the failure signature of the equivalence test.

## Principles

### From action tokens to action experts

The generalist-policy lineage begins with the idea of treating actions as tokens. RT-1 (2022) discretized each action dimension into 256 bins and ran a 35M-parameter transformer at 3 Hz, which demonstrated that a language-model-style architecture could emit robot actions at all. RT-2 (2023) went further by co-fine-tuning a full vision-language model so that the action tokens lived in the same vocabulary as text, which bought the policy web-scale semantic knowledge. OpenVLA (2024), a 7B model built from Llama-2 with fused DINOv2 and SigLIP vision, open-sourced that recipe at scale on roughly 970k Open X-Embodiment episodes.

The token interface has a ceiling, and it is set by control rate and precision. Autoregressive detokenization of per-step binned actions is far too slow for 50 Hz bimanual control, and binning each dimension into 256 values destroys the precision that contact-rich manipulation needs. Two escapes from that ceiling emerged. FAST (Physical Intelligence, 2025) applies a DCT-based compression to action chunks so that an autoregressive VLA emits roughly 15 times fewer, information-dense tokens, which keeps the token interface but makes it fast enough. Continuous generative heads take the other route: π0 attaches a flow-matching action expert to the VLM and regresses whole action chunks in one pass, abandoning discretization entirely.

### π0's recipe

π0 (Black et al. 2024) consists of a PaliGemma 3B vision-language model (SigLIP vision and a Gemma language model) together with an action expert of roughly 300M parameters. The expert shares the transformer with the VLM but uses its own weights for the action-token stream. It emits chunks of $H = 50$ actions at up to 50 Hz by conditional flow matching: a network $v_\theta(x_\tau, \tau, \text{ctx})$ is trained to regress the conditional velocity, and at inference the action chunk is produced by integrating it for about 10 Euler steps. The flow-matching timesteps used in training are drawn from a shifted Beta(1.5, 1) distribution, which overweights the noisier end of the schedule; one of the self-check questions asks you to explain why that choice suits action data.

### The blockwise-causal mask

π0 arranges its tokens in three blocks and applies a blockwise-causal attention mask. Attention is fully bidirectional within a block, and blocks are ordered causally, so that a block may attend to itself and to the blocks before it but not to the blocks after it.

| attends to → | images+language | proprio state | action tokens |
|---|---|---|---|
| **images+language** | ✅ full | ❌ | ❌ |
| **proprio state** | ✅ | ✅ full | ❌ |
| **action tokens** | ✅ | ✅ | ✅ full |

### Why prefix caching is exact

During flow-matching integration only the action block changes, because the noised actions $x_\tau$ are updated at every Euler step while the images, language, and proprioceptive state stay fixed. A token's keys and values at a given layer are functions of its hidden state entering that layer, and that hidden state is in turn a function only of the tokens it attended to in the layers below. Under the mask above, the prefix (the images-and-language block together with the state block) cannot attend to the action block at any layer. By induction over layers, every prefix hidden state, and therefore every prefix key and value at every layer, is independent of $x_\tau$. The prefix keys and values can therefore be computed once and reused at every denoising step, and the cached computation is identical to the uncached one rather than an approximation of it.

The exactness hinges on the two ❌ entries in the action column. If either were flipped so that a prefix token could attend to actions, that token's hidden state would depend on $x_\tau$ from the first layer at which it attended, and its keys and values at every later layer would change from one denoising step to the next. The model would still run, but a cached path would now compute something different from the uncached path. π0.5 and knowledge insulation (Driess et al. 2025) keep this skeleton. Knowledge insulation additionally stops gradients from flowing from the action expert into the VLM and co-trains the VLM on FAST tokens and web data, so that learning to act stops eroding the backbone's competence at visual question answering and language.

### What came after

The comparative note covers the designs that followed. GR00T uses a dual-system split, with a reasoning VLM feeding a DiT action expert, and its N1.7 version is available in LeRobot v0.6. X-VLA (arXiv 2510.10274, ICLR 2026) is a 0.9B unified transformer that adapts to a new embodiment by training only about 9M soft-prompt parameters against a frozen backbone. MolmoAct2 (arXiv 2605.02881) introduces action-reasoning models that produce depth-aware tokens, then a visual trace, then actions. π0.7 (April 2026) distills RECAP-trained specialists into a single steerable generalist. World-action models, which Lesson 20 treats in depth, extend the taxonomy again.

**Carry forward**

- Any VLA can be placed on five axes, namely backbone, action interface (discrete, FAST, flow matching, or diffusion), action head, data mix, and inference scheme, and its latency and generalization profile follow from that placement because each axis constrains the others.
- Continuous chunk-producing heads won the 2024–25 round on control rate and precision, and FAST made autoregressive tokens competitive again by compressing action chunks so that far fewer tokens are needed per chunk.
- Caching the prefix's keys and values across denoising steps is exact if and only if the prefix cannot attend to the action tokens, because a token's keys and values depend only on what it attends to; the mask is the invariant that the cache relies on.
- Every current design separates a component that understands (the VLM) from a component that acts (the expert), and knowledge insulation is the training-time form of that same separation.

| Source | Read for |
|---|---|
| π0 paper (Black et al. 2024), §III and appendix | the exact block structure and the flow-matching sampling loop, which Exercises 3–5 implement |
| OpenVLA (Kim et al. 2024), §3 | the discrete-token interface at its best, and where its latency actually goes |
| PI FAST blog post and paper | which property of action chunks makes DCT compression work |
| Knowledge insulation (Driess et al. 2025) | what exactly degrades without the stop-gradient, and how it is measured |
| LeRobot v0.6.0 release blog (huggingface.co/blog/lerobot-release-v060) | the current model roster your note must cover |
| π0.7 post (pi.website/blog/pi07) | what "steerable" means mechanically, and what was distilled from RECAP |

## Exercise 1 — Write the comparative note [Write]

The comparative note is the reference you will consult in Lessons 18 and 19 when deciding which models to fine-tune and compare. It is limited to two pages so that it stays a reference rather than a survey, and it is organized around a table and five questions.

1. Build the table first, with one row per model (RT-2, OpenVLA, π0, π0.5/KI, SmolVLA, GR00T N1.7, X-VLA, MolmoAct2, π0.7) and columns for backbone, parameter count, action interface (discrete, FAST, flow matching, or diffusion), chunk size and control rate, adaptation mechanism (full fine-tune, LoRA, or soft prompts), whether the weights are open, and one distinguishing idea.
2. For the post-cutoff entries (GR00T N1.7, MolmoAct2, π0.7), verify each cell against the model card or release post before filling it in, starting from the LeRobot v0.6 release blog and the allenai/molmoact2 GitHub repository. Any cell you cannot verify gets a "?"; an honest question mark is more useful than a plausible guess.
3. Then write the prose, with at most two paragraphs per question: (a) why did continuous heads win the 2024–25 round; (b) what did FAST change about that argument; (c) why do all current designs separate the component that understands from the component that acts, and what does knowledge insulation add; (d) what do soft prompts (X-VLA) and action reasoning (MolmoAct2) each claim the other is missing; (e) what is already replacing plain flow-matching experts, as a preview of Lesson 20.

**✅ Checkpoint:** the table has at most 5 "?" cells; each question has a committed answer with a citation rather than a survey of positions; and the note fits in two pages.

## Exercise 2 — Prove exactness on paper [Derive]

The caching argument in the Principles section is stated informally. In this exercise you write it out as a proof, so that Exercise 6 can test a specific prediction about what breaks when the mask is changed.

In `RESULTS.md`, in about half a page: write the per-layer dependency of a token's keys and values on the hidden states of the tokens it attends to; show by induction over layers that under the mask above every prefix and state token's hidden state is independent of the action block; and conclude that the prefix keys and values computed at denoising step 1 are exactly equal to those at step 10. Then state precisely what changes if the state row's action entry becomes ✅: which token's hidden state first depends on $x_\tau$, at which layer, and what an equivalence test between the cached and uncached paths would then measure.

**✅ Checkpoint:** the argument names the induction step and the mask column on which it depends, and the "what if" paragraph predicts the outcome of Exercise 6.

## Exercise 3 — Implement the blockwise-causal mask [Build]

The mask is a function of about thirty lines that most descriptions of π0 wave at rather than specify. In this exercise you specify it exactly by writing its checks before its code, which is the discipline that makes the later equivalence test trustworthy.

1. Write the checks first, as the specification. The mask has shape $(T, T)$ with $T = \text{prefix} + \text{state} + \text{action}$; prefix rows have `False` on all state and action columns; state rows have `False` on action columns; every block's diagonal sub-square is entirely `True`, since attention is bidirectional within a block; and the last action row attends to all $T$ positions. Add one property check: the mask equals the block-lower-triangular matrix built independently from `torch.block_diag` and ones, so that two constructions must agree.
2. Then have the function drafted from the specification:
   ```python
   def make_blockwise_causal_mask(prefix_len: int, state_len: int, action_len: int,
                                  device=None) -> torch.BoolTensor:
       """(T, T) with T = prefix+state+action. True = query may attend to key.
       Full attention within each block; block i attends to blocks j <= i."""
   ```
   The convention is that `True` means the query may attend to the key, which is the boolean-mask convention of `F.scaled_dot_product_attention`; `nn.Transformer` uses the opposite convention. Assert your convention with a hand-built three-token example.

**✅ Checkpoint:** all six checks pass, and you can explain why the proprioceptive state must not attend to the actions in terms of the caching invariant rather than as a modeling preference.

## Exercise 4 — Build the toy denoiser with two paths [Build]

To test the caching argument you need a model small enough to run on a Mac that has π0's structure: a long fixed prefix, a state token, and an action block that changes at every denoising step. This exercise specifies that model together with an uncached and a cached sampling path.

Write the specification and have an AI tool draft it. The model is a 4-layer transformer with $d = 256$ and 8 heads, using the mask from Exercise 3 through SDPA; a random-projected "image and language" prefix of 300 tokens, 1 state token, and 50 action tokens; absolute positional encodings; and a flow-matching head that predicts the velocity for the action block. There are two sampling paths, each taking 10 Euler steps from $x_0 \sim \mathcal{N}(0, I)$:

- The uncached path re-encodes the full sequence at every step.
- The cached path runs one forward pass over the prefix and state, storing the per-layer keys and values, and then at each denoising step forwards only the 50 action tokens, attending to the cached keys and values concatenated with the action block's own. The suffix positions must be offset by `prefix_len + state_len`.

The check for this exercise is the equivalence test of Exercise 5. The module should be under 300 lines and import nothing from LeRobot.

**✅ Checkpoint:** both paths produce a `(50, d_a)` action chunk from the same seed.

## Exercise 5 — Test equivalence and measure the speedup [Predict → Run]

This exercise turns exactness into a number and the caching benefit into a curve. Before measuring, you predict both, so that the measurement tests your understanding of where the compute goes rather than simply reporting it.

1. Write down first: the expected value of `max |cached − uncached|` in fp32, as an order of magnitude; the expected speedup at 10 steps for prefix lengths of 100, 300, and 1000, reasoning from the fact that the uncached cost is proportional to steps × (prefix + 51) while the cached cost is proportional to prefix + steps × 51, before accounting for attention's quadratic term; and how the slope of the uncached cost against steps should differ from the cached one.
2. Run the equivalence test: over 10 random seeds, `max |cached − uncached| < 1e-5` in fp32 on the final action chunk. If the test fails, the positional encodings or the mask offsets differ between the two paths, and finding which is the substance of the exercise (see Pitfalls).
3. Benchmark the wall-clock time per action chunk against denoising steps $\{1, 2, 5, 10, 20\}$, cached and uncached, for prefix lengths $\{100, 300, 1000\}$. On `mps`, call `torch.mps.synchronize()` around the timers, and time steady-state chunks only. Plot both curves, report the measured speedup at 10 steps for each prefix length, and reconcile against your predictions.

**✅ Checkpoint:** the equivalence test passes; the uncached cost grows roughly linearly in steps at full-sequence price while the cached cost flattens; and the measured speedup at 10 steps exceeds 2× and grows with prefix length.

## Exercise 6 — Break the caching invariant [Diagnose]

Exercise 2 predicted what happens when a prefix token is allowed to attend to the actions. This exercise makes that change and checks the prediction, so that you can recognize a violated invariant from the equivalence test's failure signature rather than from reading the mask.

1. Predict first. In a copy of the mask, let the state token attend to the action tokens by flipping the state row's action entries to `True`. Write down whether the equivalence test will pass or fail; if it fails, the magnitude you expect relative to $10^{-5}$ and where the divergence first appears, which should be the state token's hidden state after layer 1, hence its layer-2 keys and values, and then everything downstream; and whether the uncached path is still correct on its own.
2. Run the equivalence test on the flipped mask. Confirm that the divergence appears at the layer and token your derivation named by comparing per-layer keys and values between the two paths. Quantify what exact inference would now cost per step, given that the state token must be re-encoded at every step alongside the actions.

**✅ Checkpoint:** the failure signature matches the "what if" paragraph from Exercise 2, and the per-step cost of the broken invariant is stated.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | at most 2 pages; the table and the five answered questions; every post-cutoff cell cited or marked "?" |
| `vla_attention.py` | the mask and the cached denoiser; no LeRobot imports; under 300 lines |
| `check_attention.py` | the six mask checks, the 10-seed equivalence test, and the flipped-mask diagnosis, run from one command |
| `plots/latency.png` | latency against steps, cached and uncached, for three prefix lengths |
| `RESULTS.md` | the Exercise 2 derivation; the Exercise 5 predictions with reconciliations; the measured speedups; the Exercise 6 failure signature; three sentences on when caching stops mattering (a small prefix, or single-step flow matching) |

## Done when

- [ ] The mask checks and the $10^{-5}$ equivalence test pass on `cpu` and on `mps`.
- [ ] The latency table exists with at least a 2× speedup at 10 steps and a 300-token prefix, predicted before it was measured.
- [ ] The exactness derivation names the induction step and the mask column on which it depends.
- [ ] The flipped-mask diagnosis matches the derivation's prediction.
- [ ] `NOTE.md` answers all five questions with citations in at most two pages.

## Self-check

1. Why is prefix KV caching exact here, when KV caching in an autoregressive language model is simply ordinary decoding? Which masking property do both rely on?
2. π0 samples flow-matching timesteps from a shifted Beta(1.5, 1) rather than uniformly. Which failure mode of uniform sampling does overweighting the noisier timesteps address for action data?
3. If the state token were allowed to attend to the action tokens, inference would still run. What specifically breaks, and by how much per denoising step?
4. Why does knowledge insulation stop gradients into the VLM but still co-train the VLM on FAST tokens? What would each half alone fail at?
5. Name one concrete reason chunk-level flow matching beats per-step autoregressive decoding at 50 Hz, and one reason it is worse.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Equivalence test off by about $10^{-3}$ | fp16 or bf16 accumulation differences between the two paths | run the test in fp32; benchmark separately in half precision |
| Everything attends everywhere despite your mask | PyTorch convention mismatch: `F.scaled_dot_product_attention`'s boolean mask uses True = attend, while `nn.Transformer` uses True = masked | use SDPA and assert the convention in a test with a hand-built three-token example |
| Cached path diverges only for tokens beyond position 300 | absolute positional encodings restarted at 0 for the suffix forward pass | offset suffix positions by `prefix_len + state_len` |
| Benchmark shows no speedup | timing includes the prefix encode in both paths, or `mps` asynchrony hides work | time steady-state chunks only; synchronize before and after |
| Note keeps growing past two pages | surveying instead of answering | each of the five questions gets at most two paragraphs; cut anything that answers none of them |

## Going deeper

Port your cached denoiser to load real π0 weights through `openpi` or LeRobot's π0 port, and reproduce the equivalence and latency study at full scale on a rented A100. Compare your measured speedup with what the paper implies.

## References

- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. arXiv:2406.09246.
- Driess et al., *Knowledge Insulating Vision-Language-Action Models*, 2025. arXiv:2505.23705.
- Wang et al., *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA*, ICLR 2026. arXiv:2510.10274.
- Fang et al., *MolmoAct2: Action Reasoning Models for Real-World Deployment*, 2026. arXiv:2605.02881.
- Physical Intelligence: FAST, π0.5, *π0.7: a Steerable Model with Emergent Capabilities* (pi.website/blog/pi07).
- LeRobot v0.6.0 release: huggingface.co/blog/lerobot-release-v060.
