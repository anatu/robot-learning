# Lesson 17 — The VLA Landscape + π0 Dissection

Map the generalist-policy design space as it stands in mid-2026, then rebuild π0's two architectural signatures — blockwise-causal attention and prefix KV caching across denoising steps — as tested PyTorch modules.

| | |
|---|---|
| **Phase** | 5 — Generalist policies |
| **Time** | ~3 sessions: 4–6 h for the comparative note, 4–6 h for the attention/cache module + benchmark |
| **Cost** | $0 — all Mac-local (the module is small enough for `mps`/`cpu`) |
| **Prerequisites** | 13 (flow matching — π0's action head is exactly that machinery), 14 (you've trained a transformer policy and know what a chunk is), 12 (why generative heads at all) |
| **Feeds into** | 18/19 (you'll fine-tune and compare the models mapped here), 20 (world-action models extend this taxonomy), H4 (MolmoAct2 + SmolVLA on your arm) |

## Learning objectives

After this lesson you can:

1. **Place** any VLA on the design axes — backbone, action interface, action head, data mix, inference scheme — and predict its latency and generalization profile from the placement.
2. **Explain** why discrete action tokens lost to continuous generative heads for high-rate control, and how FAST partially reversed that verdict.
3. **Implement** π0's blockwise-causal attention mask and defend each block-pair decision.
4. **Prove** that prefix KV caching across flow-matching denoising steps is *exact*, not approximate — first on paper, then with a passing equivalence test.
5. **Quantify** the latency win from caching as a function of denoising steps and prefix length.

## Background

**The lineage.** RT-1 (2022) discretized each action dimension into 256 bins and ran a 35M-param transformer at 3 Hz — proof that "actions as tokens" works. RT-2 (2023) co-fine-tuned a full VLM so that action tokens live in the same vocabulary as text, buying web-scale semantics. OpenVLA (2024, 7B: Llama-2 + fused DINOv2/SigLIP vision) open-sourced the recipe at scale on ~970k Open X-Embodiment episodes. The ceiling: autoregressive detokenization of per-step binned actions is far too slow for 50 Hz bimanual control, and binning destroys precision. Two escapes emerged: **FAST** (Physical Intelligence, 2025) applies a DCT-based compression to action chunks so an autoregressive VLA emits ~15× fewer, information-dense tokens; and **continuous generative heads** — π0 bolts a flow-matching "action expert" onto the VLM and regresses whole chunks in one shot.

**π0's recipe** (Black et al. 2024). A PaliGemma 3B VLM (SigLIP vision + Gemma LM) plus a ~300M-param action expert that shares the transformer but uses its own weights for the action-token stream. The expert emits chunks of $H=50$ actions at up to 50 Hz via conditional flow matching: train $v_\theta(x_\tau, \tau, \text{ctx})$ to regress the conditional velocity, integrate ~10 Euler steps at inference. Timesteps are sampled from a shifted Beta(1.5, 1) that overweights the noisier end of the schedule — a self-check question asks you why.

**The attention structure.** π0 arranges tokens in three blocks and applies a *blockwise-causal* mask — full bidirectional attention within a block, causal ordering across blocks:

| attends to → | images+language | proprio state | action tokens |
|---|---|---|---|
| **images+language** | ✅ full | ❌ | ❌ |
| **proprio state** | ✅ | ✅ full | ❌ |
| **action tokens** | ✅ | ✅ | ✅ full |

The payoff: during flow-matching integration only the action block changes (the noised actions $x_\tau$ update every Euler step). Because the prefix (images+language, and state) *cannot attend to* the action block, its keys and values are mathematically independent of $x_\tau$ — so you compute the prefix KV once and reuse it for every denoising step. This is exactness by masking, not an approximation. π0.5 and knowledge insulation (Driess et al. 2025) keep this skeleton; KI additionally *stops gradients* from the action expert into the VLM and co-trains the VLM on FAST tokens + web data, so action learning stops eroding the backbone's VQA/language competence.

**What came after** (the note's subject matter): GR00T's dual-system split (reasoning VLM + DiT action expert, now N1.7 in LeRobot v0.6), X-VLA's soft prompts (0.9B unified transformer; adapts to a new embodiment by training only ~9M prompt params against a frozen backbone — arXiv 2510.10274, ICLR 2026), MolmoAct2's action-reasoning models (depth-aware tokens → visual trace → actions; arXiv 2605.02881), π0.7's distillation of RECAP-trained specialists into one steerable generalist (Apr 2026), and world-action models (Lesson 20's territory).

| Source | Read for |
|---|---|
| π0 paper (Black et al. 2024), §III + appendix | the exact block structure and FM sampling loop — everything Part 2/3 implements |
| OpenVLA (Kim et al. 2024), §3 | the discrete-token interface at its best; where does its latency actually go? |
| PI FAST blog post / paper | what property of action chunks makes DCT compression work? |
| Knowledge insulation (Driess et al. 2025) | what exactly degrades without the stop-gradient, and how is it measured? |
| LeRobot v0.6.0 release blog (huggingface.co/blog/lerobot-release-v060) | the current model roster your note must cover |
| π0.7 post (pi.website/blog/pi07) | what "steerable" means mechanically; what got distilled from RECAP |

## Part 1 — The comparative note (~4–6 h)

Produces `NOTE.md` (2–3 pages), the reference you'll reach for in Lessons 18–19 when choosing what to fine-tune.

1. Build the spine first — a table with one row per model (RT-2, OpenVLA, π0, π0.5/KI, SmolVLA, GR00T N1.7, X-VLA, MolmoAct2, π0.7) and columns: backbone, params, action interface (discrete/FAST/FM/diffusion), chunk size + control rate, adaptation mechanism (full FT / LoRA / soft prompts), open weights (Y/N), one distinguishing idea.
2. For post-cutoff entries (GR00T N1.7, MolmoAct2, π0.7), verify each cell against the model card or release post before filling it in — start from the LeRobot v0.6 release blog and the allenai/molmoact2 GitHub. Cells you cannot verify get "?" — an honest "?" beats a plausible guess.
3. Then write the prose around five questions: (a) why did continuous heads win the 2024–25 round? (b) what did FAST change about that argument? (c) why do all current designs separate "VLM understands" from "expert acts", and what does KI add? (d) what do soft prompts (X-VLA) and action reasoning (MolmoAct2) each claim the others are missing? (e) what's already replacing plain flow-matching experts (your Lesson 20 preview)?

**✅ Checkpoint:** the table has ≤ 5 "?" cells; each of the five questions has a committed answer with a citation, not a survey shrug.

## Part 2 — The blockwise-causal mask (~1 h)

A ~30-line function everyone hand-waves. You'll spec it exactly.

1. Implement:
   ```python
   def make_blockwise_causal_mask(prefix_len: int, state_len: int, action_len: int,
                                  device=None) -> torch.BoolTensor:
       """(T, T) with T = prefix+state+action. True = query may attend to key.
       Full attention within each block; block i attends to blocks j <= i."""
   ```
2. `pytest` cases: exact shape; prefix rows have `False` on all state/action columns; state rows have `False` on action columns; every block's diagonal sub-square is all-`True` (bidirectional within block); the last action row attends to all $T$ positions.
3. Property test: the mask equals the block-lower-triangular matrix built independently from `torch.block_diag` + ones — two constructions, one truth.

**✅ Checkpoint:** tests green; you can answer *why* proprio must not attend to actions (hint: it's the caching invariant, not a modeling preference).

## Part 3 — Prefix KV cache across denoising steps (~3–4 h)

Build a toy π0-shaped denoiser and make caching provably exact.

1. Model: a 4-layer, $d{=}256$, 8-head transformer using your Part 2 mask; random-projected "image+language" prefix of 300 tokens, 1 state token, 50 action tokens; a flow-matching head that predicts the velocity for the action block.
2. Sampling loop: 10 Euler steps from $x_0 \sim \mathcal{N}(0, I)$, re-encoding the *full* sequence each step. This is the uncached baseline.
3. Cached version: one prefix forward pass storing per-layer K/V for prefix+state; each denoising step forwards *only* the 50 action tokens, attending to cached KV concatenated with the action block's own KV.
4. Equivalence test: over 10 random seeds, `max |cached − uncached| < 1e-5` in fp32 on the final action chunk. If it fails, your positional encodings or mask offsets differ between paths — that *is* the lesson.
5. Benchmark: wall-clock per action chunk vs denoising steps $\{1, 2, 5, 10, 20\}$, cached vs uncached, prefix lengths $\{100, 300, 1000\}$. On `mps`, call `torch.mps.synchronize()` around timers. Plot both curves; report the speedup at 10 steps per prefix length.

**✅ Checkpoint:** equivalence test green; uncached cost grows ~linearly in steps at full-sequence price while cached flattens; measured speedup at 10 steps exceeds 2× and grows with prefix length.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `NOTE.md` | the table + five answered questions; every post-cutoff cell cited or marked "?" |
| `vla_attention.py` | mask + cached denoiser, no LeRobot imports, < 300 lines |
| `tests/` | mask properties + KV equivalence, green in CI |
| `plots/latency.png` | latency vs steps, cached/uncached × 3 prefix lengths |
| `RESULTS.md` | measured speedups; 3 sentences on when caching stops mattering (small prefix, single-step FM) |

## Done when

- [ ] Mask tests and the 1e-5 equivalence test pass on `cpu` and `mps`.
- [ ] The latency table exists with ≥ 2× speedup demonstrated at 10 steps, 300-token prefix.
- [ ] `NOTE.md` answers all five questions with citations.
- [ ] You can whiteboard the 3×3 attention table from memory and derive which KV entries are cacheable.

## Self-check

1. Why is prefix KV caching *exact* here, when KV caching in autoregressive LMs is just... normal decoding? What masking property do both rely on?
2. π0 samples FM timesteps from a shifted Beta(1.5, 1) rather than uniform. What failure mode of uniform sampling does overweighting noisier timesteps address for *action* data?
3. If the state token were allowed to attend to action tokens, inference would still run — what specifically breaks, and by how much per denoising step?
4. Why does knowledge insulation stop gradients into the VLM but still co-train the VLM on FAST tokens? What would each half alone fail at?
5. Name one concrete reason chunk-level flow matching beats per-step autoregressive decoding at 50 Hz, and one reason it's worse.

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Equivalence test off by ~1e-3 | fp16/bf16 accumulation differences between paths | run the test in fp32; benchmark separately in half precision |
| Everything attends everywhere despite your mask | PyTorch convention mismatch: `F.scaled_dot_product_attention` bool mask uses True = *attend*, `nn.Transformer` uses True = *masked* | pick SDPA, assert convention in a test with a hand-built 3-token example |
| Cached path diverges only for tokens > position 300 | absolute positional encodings restarted at 0 for the suffix forward | offset suffix positions by `prefix_len + state_len` |
| Benchmark shows no speedup | timing includes prefix encode in both paths, or `mps` async hides work | time steady-state chunks only; synchronize before/after |
| Note keeps growing past 3 pages | surveying instead of answering | each of the five questions gets ≤ 2 paragraphs; cut anything that answers none of them |

## Stretch

Port your cached denoiser to load real π0 weights via `openpi` or LeRobot's π0 port and reproduce the equivalence + latency study at full scale on a rented A100. Compare your measured speedup to what the paper implies.

## References

- Black et al., *π0: A Vision-Language-Action Flow Model for General Robot Control*, 2024. arXiv:2410.24164.
- Kim et al., *OpenVLA: An Open-Source Vision-Language-Action Model*, 2024. arXiv:2406.09246.
- Driess et al., *Knowledge Insulating Vision-Language-Action Models*, 2025. arXiv:2505.23705.
- Wang et al., *X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment VLA*, ICLR 2026. arXiv:2510.10274.
- Fang et al., *MolmoAct2: Action Reasoning Models for Real-World Deployment*, 2026. arXiv:2605.02881.
- Physical Intelligence: FAST, π0.5, *π0.7: a Steerable Model with Emergent Capabilities* (pi.website/blog/pi07).
- LeRobot v0.6.0 release: huggingface.co/blog/lerobot-release-v060.
