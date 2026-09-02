# Lesson 17 — The VLA Landscape + π0 Dissection

Map the generalist-policy design space as it stands in mid-2026, then prove — on paper and with a passing test — why π0's blockwise-causal attention makes prefix KV caching across denoising steps exact rather than approximate.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~2 sessions: 3–4 h for the comparative note, 3–4 h for the mask/cache module + benchmark |
| **Cost** | $0 — all Mac-local (the module is small enough for `mps`/`cpu`) |
| **Prerequisites** | 13 (flow matching — π0's action head is exactly that machinery), 14 (you have trained a transformer policy and know what a chunk is), 12 (why generative heads at all) |
| **Feeds into** | 18/19 (you fine-tune and compare the models mapped here), 20 (world-action models extend this taxonomy), H4 (MolmoAct2 + SmolVLA on your arm) |

## Learning objectives

After this lesson you can:

1. **Place** any VLA on the design axes — backbone, action interface, action head, data mix, inference scheme — and predict its latency and generalization profile from the placement.
2. **Explain** why discrete action tokens lost to continuous generative heads for high-rate control, and how FAST partially reversed that verdict.
3. **Derive** why prefix KV caching across flow-matching denoising steps is *exact* under the blockwise-causal mask, and which mask entries the exactness depends on.
4. **Predict** the latency win from caching as a function of denoising steps and prefix length, then measure it.
5. **Diagnose** a broken caching invariant from the equivalence test's failure signature.

## Principles

**The lineage.** RT-1 (2022) discretized each action dimension into 256 bins and ran a 35M-param transformer at 3 Hz — proof that "actions as tokens" works. RT-2 (2023) co-fine-tuned a full VLM so that action tokens live in the same vocabulary as text, buying web-scale semantics. OpenVLA (2024, 7B: Llama-2 + fused DINOv2/SigLIP vision) open-sourced the recipe at scale on ~970k Open X-Embodiment episodes. The ceiling: autoregressive detokenization of per-step binned actions is far too slow for 50 Hz bimanual control, and binning destroys precision. Two escapes emerged: **FAST** (Physical Intelligence, 2025) applies a DCT-based compression to action chunks so an autoregressive VLA emits ~15× fewer, information-dense tokens; and **continuous generative heads** — π0 bolts a flow-matching "action expert" onto the VLM and regresses whole chunks in one shot.

**π0's recipe** (Black et al. 2024). A PaliGemma 3B VLM (SigLIP vision + Gemma LM) plus a ~300M-param action expert that shares the transformer but uses its own weights for the action-token stream. The expert emits chunks of $H=50$ actions at up to 50 Hz via conditional flow matching: train $v_\theta(x_\tau, \tau, \text{ctx})$ to regress the conditional velocity, integrate ~10 Euler steps at inference. Timesteps are sampled from a shifted Beta(1.5, 1) that overweights the noisier end of the schedule — a self-check question asks you why.

**The attention structure.** π0 arranges tokens in three blocks and applies a *blockwise-causal* mask — full bidirectional attention within a block, causal ordering across blocks:

| attends to → | images+language | proprio state | action tokens |
|---|---|---|---|
| **images+language** | ✅ full | ❌ | ❌ |
| **proprio state** | ✅ | ✅ full | ❌ |
| **action tokens** | ✅ | ✅ | ✅ full |

**Why caching is exact.** During flow-matching integration only the action block changes (the noised actions $x_\tau$ update every Euler step). A token's keys and values at a layer are functions of its hidden state entering that layer, and that hidden state is a function only of the tokens it attended to in the layers below. Because the prefix (images+language, and state) *cannot attend to* the action block, every prefix hidden state — and hence every prefix K/V at every layer — is independent of $x_\tau$. Compute the prefix KV once; reuse it for every denoising step. This is exactness by masking, not an approximation, and it hinges on the two ❌ entries in the action column. Flip either and the prefix K/V become functions of $x_\tau$: still runnable, but a cached path now computes something different from the uncached one. π0.5 and knowledge insulation (Driess et al. 2025) keep this skeleton; KI additionally *stops gradients* from the action expert into the VLM and co-trains the VLM on FAST tokens + web data, so action learning stops eroding the backbone's VQA/language competence.

**What came after** (the note's subject matter): GR00T's dual-system split (reasoning VLM + DiT action expert, now N1.7 in LeRobot v0.6), X-VLA's soft prompts (0.9B unified transformer; adapts to a new embodiment by training only ~9M prompt params against a frozen backbone — arXiv 2510.10274, ICLR 2026), MolmoAct2's action-reasoning models (depth-aware tokens → visual trace → actions; arXiv 2605.02881), π0.7's distillation of RECAP-trained specialists into one steerable generalist (Apr 2026), and world-action models (Lesson 20's territory).

**Carry forward**

- Five axes place any VLA: backbone, action interface (discrete / FAST / FM / diffusion), action head, data mix, inference scheme. Latency and generalization follow from the placement.
- Continuous chunk heads won the 2024–25 round on rate and precision; FAST made autoregressive tokens competitive again by compressing chunks.
- Prefix KV caching across denoising steps is exact iff the prefix cannot attend to the actions. The mask is the invariant.
- "VLM understands" is separated from "expert acts" in every current design; KI is the training-time version of the same separation.

| Source | Read for |
|---|---|
| π0 paper (Black et al. 2024), §III + appendix | the exact block structure and FM sampling loop — everything Exercises 3–5 implement |
| OpenVLA (Kim et al. 2024), §3 | the discrete-token interface at its best; where does its latency actually go? |
| PI FAST blog post / paper | what property of action chunks makes DCT compression work? |
| Knowledge insulation (Driess et al. 2025) | what exactly degrades without the stop-gradient, and how is it measured? |
| LeRobot v0.6.0 release blog (huggingface.co/blog/lerobot-release-v060) | the current model roster your note must cover |
| π0.7 post (pi.website/blog/pi07) | what "steerable" means mechanically; what got distilled from RECAP |

## Exercise 1 — The comparative note [Write]

Tests objectives 1–2. Produces `NOTE.md`, ≤ 2 pages, the reference you reach for in Lessons 18–19 when choosing what to fine-tune.

1. Build the spine first — a table with one row per model (RT-2, OpenVLA, π0, π0.5/KI, SmolVLA, GR00T N1.7, X-VLA, MolmoAct2, π0.7) and columns: backbone, params, action interface (discrete/FAST/FM/diffusion), chunk size + control rate, adaptation mechanism (full FT / LoRA / soft prompts), open weights (Y/N), one distinguishing idea.
2. For post-cutoff entries (GR00T N1.7, MolmoAct2, π0.7), verify each cell against the model card or release post before filling it in — start from the LeRobot v0.6 release blog and the allenai/molmoact2 GitHub. Cells you cannot verify get "?" — an honest "?" beats a plausible guess.
3. Then the prose, ≤ 2 paragraphs per question: (a) why did continuous heads win the 2024–25 round? (b) what did FAST change about that argument? (c) why do all current designs separate "VLM understands" from "expert acts", and what does KI add? (d) what do soft prompts (X-VLA) and action reasoning (MolmoAct2) each claim the others are missing? (e) what is already replacing plain flow-matching experts (your Lesson 20 preview)?

**✅ Checkpoint:** the table has ≤ 5 "?" cells; each question has a committed answer with a citation, not a survey shrug; the note fits in 2 pages.

## Exercise 2 — Exactness, on paper [Derive]

Tests objective 3. In `RESULTS.md`, half a page: write the per-layer dependency of a token's K/V on the hidden states of the tokens it attends to; show by induction over layers that under the mask above every prefix and state token's hidden state is independent of the action block; conclude that the prefix K/V computed at denoising step 1 equal those at step 10 exactly. Then state precisely what changes if the state row's action entry becomes ✅: which token's hidden state first depends on $x_\tau$, at which layer, and what an equivalence test between cached and uncached paths would then measure.

**✅ Checkpoint:** the argument names the induction step and the mask column it depends on; the "what if" paragraph predicts Exercise 6's outcome.

## Exercise 3 — The blockwise-causal mask [Build]

Tests objective 3 as code. A ~30-line function everyone hand-waves; you spec it exactly.

1. **Write the checks first**, as the spec: exact shape $(T, T)$ with $T = \text{prefix} + \text{state} + \text{action}$; prefix rows have `False` on all state/action columns; state rows have `False` on action columns; every block's diagonal sub-square is all-`True` (bidirectional within block); the last action row attends to all $T$ positions. Plus a property: the mask equals the block-lower-triangular matrix built independently from `torch.block_diag` + ones — two constructions, one truth.
2. Then the function, drafted from the spec:
   ```python
   def make_blockwise_causal_mask(prefix_len: int, state_len: int, action_len: int,
                                  device=None) -> torch.BoolTensor:
       """(T, T) with T = prefix+state+action. True = query may attend to key.
       Full attention within each block; block i attends to blocks j <= i."""
   ```
   Convention: `True` = *attend* (the `F.scaled_dot_product_attention` bool-mask convention; `nn.Transformer` is the opposite). Assert your convention with a hand-built 3-token example.

**✅ Checkpoint:** all six checks green; you can answer *why* proprio must not attend to actions (the caching invariant, not a modeling preference).

## Exercise 4 — A π0-shaped toy denoiser, two paths [Build]

Tests objective 3's mechanism in a model small enough for `mps`.

Spec: a 4-layer, $d{=}256$, 8-head transformer using your Exercise 3 mask via SDPA; a random-projected "image+language" prefix of 300 tokens, 1 state token, 50 action tokens; absolute positional encodings; a flow-matching head predicting the velocity for the action block. Two sampling paths, 10 Euler steps from $x_0 \sim \mathcal{N}(0, I)$:

- **Uncached:** re-encode the *full* sequence each step.
- **Cached:** one prefix forward pass storing per-layer K/V for prefix+state; each denoising step forwards *only* the 50 action tokens, attending to cached KV concatenated with the action block's own KV. Suffix positions offset by `prefix_len + state_len`.

The check is Exercise 5's equivalence test. Under 300 lines, no LeRobot imports.

**✅ Checkpoint:** both paths produce a `(50, d_a)` chunk from the same seed.

## Exercise 5 — Equivalence and the latency curve [Predict → Run]

Tests objectives 3–4: exactness as a number, and the win as a curve.

1. **Write first:** the expected `max |cached − uncached|` in fp32 (order of magnitude); the expected speedup at 10 steps for prefix ∈ {100, 300, 1000} (reason from the uncached cost ∝ steps × (prefix + 51) vs cached ∝ prefix + steps × 51, before attention's quadratic term); and how the uncached-vs-steps curve should differ in slope from the cached one.
2. Equivalence: over 10 random seeds, `max |cached − uncached| < 1e-5` in fp32 on the final action chunk. If it fails, your positional encodings or mask offsets differ between paths — that *is* the lesson (see Pitfalls).
3. Benchmark: wall-clock per action chunk vs denoising steps $\{1, 2, 5, 10, 20\}$, cached vs uncached, prefix lengths $\{100, 300, 1000\}$. On `mps`, call `torch.mps.synchronize()` around timers; time steady-state chunks only. Plot both curves; report the measured speedup at 10 steps per prefix length and reconcile against your predictions.

**✅ Checkpoint:** equivalence green; uncached cost grows ~linearly in steps at full-sequence price while cached flattens; measured speedup at 10 steps exceeds 2× and grows with prefix length.

## Exercise 6 — Break the invariant [Diagnose]

Tests objective 5: what a violated mask does to the cache.

1. **Predict:** in a copy of the mask, let the state token attend to the action tokens (flip the state row's action entries to `True`). Write down what the equivalence test will show (pass/fail; if fail, the magnitude you expect relative to 1e-5, and *where* divergence first appears — the state token's hidden state after layer 1, hence its layer-2 K/V, then everything downstream), and whether the uncached path is still correct on its own.
2. Run the equivalence test on the flipped mask. Confirm the divergence appears at the layer and token your derivation named (compare per-layer K/V between paths). Quantify: how much per-step compute exact inference would now cost (the state token must be re-encoded every step alongside the actions).

**✅ Checkpoint:** the failure signature matches Exercise 2's "what if" paragraph; the per-step cost of the broken invariant is stated.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | ≤ 2 pages; the table + five answered questions; every post-cutoff cell cited or marked "?" |
| `vla_attention.py` | mask + cached denoiser, no LeRobot imports, < 300 lines |
| `check_attention.py` | the six mask checks + the 10-seed equivalence test + the flipped-mask diagnosis, one command |
| `plots/latency.png` | latency vs steps, cached/uncached × 3 prefix lengths |
| `RESULTS.md` | Exercise 2 derivation; Exercise 5 predictions with reconciliations; measured speedups; Exercise 6 signature; 3 sentences on when caching stops mattering (small prefix, single-step FM) |

## Done when

- [ ] Mask checks and the 1e-5 equivalence test pass on `cpu` and `mps`.
- [ ] The latency table exists with ≥ 2× speedup demonstrated at 10 steps, 300-token prefix, predicted before measured.
- [ ] The exactness derivation names the induction step and the mask column it depends on.
- [ ] The flipped-mask diagnosis matches the derivation's prediction.
- [ ] `NOTE.md` answers all five questions with citations in ≤ 2 pages.

## Self-check

1. Why is prefix KV caching *exact* here, when KV caching in autoregressive LMs is just... normal decoding? What masking property do both rely on?
2. π0 samples FM timesteps from a shifted Beta(1.5, 1) rather than uniform. What failure mode of uniform sampling does overweighting noisier timesteps address for *action* data?
3. If the state token were allowed to attend to action tokens, inference would still run — what specifically breaks, and by how much per denoising step?
4. Why does knowledge insulation stop gradients into the VLM but still co-train the VLM on FAST tokens? What would each half alone fail at?
5. Name one concrete reason chunk-level flow matching beats per-step autoregressive decoding at 50 Hz, and one reason it is worse.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Equivalence test off by ~1e-3 | fp16/bf16 accumulation differences between paths | run the test in fp32; benchmark separately in half precision |
| Everything attends everywhere despite your mask | PyTorch convention mismatch: `F.scaled_dot_product_attention` bool mask uses True = *attend*, `nn.Transformer` uses True = *masked* | pick SDPA, assert convention in a test with a hand-built 3-token example |
| Cached path diverges only for tokens > position 300 | absolute positional encodings restarted at 0 for the suffix forward | offset suffix positions by `prefix_len + state_len` |
| Benchmark shows no speedup | timing includes prefix encode in both paths, or `mps` async hides work | time steady-state chunks only; synchronize before/after |
| Note keeps growing past 2 pages | surveying instead of answering | each of the five questions gets ≤ 2 paragraphs; cut anything that answers none of them |

## Going deeper

Port your cached denoiser to load real π0 weights via `openpi` or LeRobot's π0 port and reproduce the equivalence + latency study at full scale on a rented A100. Compare your measured speedup to what the paper implies.

## References

- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. arXiv:2406.09246.
- Driess et al., *Knowledge Insulating Vision-Language-Action Models*, 2025. arXiv:2505.23705.
- Wang et al., *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA*, ICLR 2026. arXiv:2510.10274.
- Fang et al., *MolmoAct2: Action Reasoning Models for Real-World Deployment*, 2026. arXiv:2605.02881.
- Physical Intelligence: FAST, π0.5, *π0.7: a Steerable Model with Emergent Capabilities* (pi.website/blog/pi07).
- LeRobot v0.6.0 release: huggingface.co/blog/lerobot-release-v060.
