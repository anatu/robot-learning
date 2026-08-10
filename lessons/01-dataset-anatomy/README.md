# Lesson 01 — LeRobotDataset Anatomy

Reimplement the v3 dataset format from bytes up — a parser that reproduces `LeRobotDataset`'s outputs exactly, proven by parity tests. The format is the substrate every later lesson sits on; after this week it holds no surprises.

| | |
|---|---|
| **Phase** | 1 — Data |
| **Time** | 1–2 sessions (6–10 h desk time) |
| **Cost** | $0 (Mac-local) |
| **Prerequisites** | 00 (env + `hello_dataset.py`, which this lesson grows into a parser) |
| **Feeds into** | 02 (your writer is validated by this parser), 14/15 (`delta_timestamps` builds the action chunks), H2 (you'll debug real recordings at the byte level) |

## Learning objectives

After this lesson you can:

1. **Document** the complete v3 on-disk layout — every file, every schema field — well enough that a stranger could implement a reader from your `FORMAT.md` alone.
2. **Resolve** any (episode, frame) pair to bytes: parquet row via metadata offsets, video frame via timestamp seek.
3. **Reimplement** `delta_timestamps` windowing — nearest-frame validation, tolerance, episode-boundary clamping, padding masks — as a standalone function with the exact library semantics.
4. **Prove** parity against `LeRobotDataset` with deterministic and property-based tests.
5. **Explain** why v3 moved from episode-per-file to many-episodes-per-file, and what that costs.

## Background

**The three pillars.** v3 decouples storage from API. (1) *Tabular*: low-dim high-rate signals (states, actions, timestamps) in Apache Parquet, many episodes concatenated per file. (2) *Visual*: camera frames concatenated into MP4 shards per camera. (3) *Metadata*: JSON/Parquet records holding the schema, fps, normalization stats, and — the load-bearing part — **episode segmentation**: per-episode lengths and offsets into the shared parquet/MP4 files. Episode boundaries are resolved through metadata, never filenames. That inversion (v2.1 had `episode-0000.parquet`, one file per episode) is what lets the format scale to millions of episodes without filesystem pressure.

**The layout you'll document:**

| Path | Contents |
|---|---|
| `meta/info.json` | canonical schema (feature names/dtypes/shapes), fps, `codebase_version`, path templates for data/video shards |
| `meta/stats.json` | per-feature mean/std/min/max (downstream normalization reads this) |
| `meta/tasks.jsonl` | natural-language task → integer id |
| `meta/episodes/` | chunked parquet: per-episode lengths, tasks, offsets into shared files |
| `data/` | `file-XXXX.parquet` shards, many episodes each |
| `videos/` | per-camera MP4 shards, many episodes each |

**Windowing semantics (the heart of the lesson).** `delta_timestamps={"key": [d_0, ..., d_{T-1}]}` requests, for query index $i$, frames at times $t_i + d_j$. Every $d_j$ must be a multiple of $1/\text{fps}$ within `tolerance_s` (default `1e-4`); deltas convert to index offsets, offsets are **clamped to the episode** — conceptually `max(ep_start, min(ep_end - 1, i + delta))` — and a boolean mask `{key}_is_pad` marks positions where the unclamped index fell outside `[ep_start, ep_end)`. So a 3-frame history at episode start returns the first frame twice, flagged as padding — policies mask their loss with exactly these tensors. Derive this behavior empirically first; the library's `datasets/dataset_reader.py` (`check_delta_timestamps`, `_get_query_indices`) is your answer key *after* you've written your version.

| Source | Read for |
|---|---|
| Tutorial §1.1–1.2 | why the format exists; the streaming/batching design goals your stretch benchmark will test |
| [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | the layout table above, from the source |
| `src/lerobot/datasets/dataset_reader.py` (installed version) | the reference semantics — read only after Part 3's first draft |

## Part 0 — Get the bytes (30 min)

1. Snapshot the raw repo (not through `LeRobotDataset` — you want files):
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("lerobot/svla_so101_pickplace", repo_type="dataset",
                     local_dir="data/svla_so101_pickplace")
   ```
2. `tree -h data/svla_so101_pickplace` and check `meta/info.json → codebase_version`. The Hub card for this dataset has lagged at v2.1; if that's what you got, convert (converter module path per Lesson 00 Part 5) and parse the **v3 output**. Keep the v2.1 tree around — documenting the diff is FORMAT.md material.
3. Record ground truth from metadata: 50 episodes, 11,939 total frames, 30 fps, features `action`/`observation.state` (6-D: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`, all `.pos`) and cameras `observation.images.up`, `observation.images.side` (480×640×3, AV1).

**✅ Checkpoint:** the tree matches the layout table; you can state which parquet file holds episode 30 and at what row offset, from metadata alone.

## Part 1 — Metadata parsers (1 h)

Module: `parser/meta.py`. No `lerobot` imports anywhere in `parser/` — that's the whole point. (`pyarrow`, `av`/`pyav`, `numpy`, `torch` are fine.)

1. `load_info(root) -> Info`: features dict (name → dtype/shape/names), fps, codebase_version, path templates.
2. `load_episodes(root) -> list[EpisodeMeta]`: per-episode length, task id, and data/video offsets, from `meta/episodes/` chunked parquet.
3. `load_stats(root)`, `load_tasks(root)`.

**✅ Checkpoint:** `sum(ep.length for ep in episodes) == 11_939`; every feature in `info.json` has a stats entry; task table has exactly 1 task.

## Part 2 — Byte-level readers (2 h)

Modules: `parser/tabular.py`, `parser/video.py`.

1. `read_frame(root, ep_idx, frame_idx) -> dict`: resolve via episode offsets into the right `file-XXXX.parquet`, return state/action/timestamp for that row. Use `pyarrow` memory-mapping; do not load whole files.
2. `read_image(root, camera_key, ep_idx, frame_idx) -> np.ndarray`: locate the MP4 shard + in-file timestamp from metadata, then decode with `av`: seek to the nearest keyframe **at or before** the target, decode forward to the exact frame. AV1 note from Lesson 00 applies.
3. Convert to the tensor conventions `LeRobotDataset` uses (CHW, float32 in [0,1] for images).

**✅ Checkpoint:** for 20 random (episode, frame) pairs, your state/action match `dataset[global_idx]` via `torch.equal`, and images match with max abs diff ≤ 1/255 (identical decoder → expect exact; the tolerance only absorbs backend differences).

## Part 3 — Windowing (2 h)

Module: `parser/windowing.py`. This is the function the course reuses mentally forever.

1. Signature:
   ```python
   def window(query_idx: int, ep_start: int, ep_end: int,
              deltas: list[float], fps: float, tolerance_s: float = 1e-4
              ) -> tuple[list[int], list[bool]]:
       """Absolute frame indices to gather, and is_pad per position."""
   ```
   Validate deltas (multiples of 1/fps within tolerance — mind float error: `round(d * fps)`, then check `abs(d - k/fps) <= tolerance_s`), clamp, emit pad mask.
2. Compose into `parser/dataset.py: ParsedDataset.__getitem__` returning the same dict `LeRobotDataset` returns, including `{key}_is_pad` tensors.
3. Parity target: sample #100 with `{"observation.images.up": [-2/30, -1/30, 0.0]}` — `torch.equal` on every non-image key, image tolerance as Part 2, `torch.equal` on the pad masks. Then move the window to an episode's first and last frames and assert the clamp+pad behavior.
4. Now open `dataset_reader.py` and diff semantics against yours. Any divergence: fix, and write the discovered rule into FORMAT.md.

**✅ Checkpoint:** parity on sample #100 and on both episode boundaries of ≥ 3 episodes.

## Part 4 — Property tests (2 h)

`tests/test_windowing_properties.py`, with `hypothesis`. Fuzz the *windowing logic* on synthetic indices — not through video decode (slow, flaky-by-timeout).

1. Strategies: episode lengths 1–500, query index within episode, delta lists drawn from `k/fps` for `k ∈ [-200, 200]` (plus a strategy that perturbs deltas by ±`tolerance_s/2` to test validation).
2. Invariants: (a) output length == len(deltas); (b) `is_pad[j]` ⟺ unclamped index out of `[ep_start, ep_end)`; (c) all returned indices within the episode — never cross-episode leakage; (d) with all-zero deltas, output is `[query_idx]`, no padding; (e) monotone deltas ⇒ monotone indices.
3. Oracle test: same invariants checked against `LeRobotDataset` outputs on the real dataset for a seeded sample of 200 (query, delta-set) pairs over non-image keys.
4. Commit at least one `@example(...)` regression pin for any bug hypothesis found.

**✅ Checkpoint:** `pytest` green, ≥ 200 hypothesis examples per property, at least one pinned regression example exists.

## Part 5 — FORMAT.md (1–2 h)

The deliverable a stranger implements from. Required contents, in order: directory tree; `info.json` field-by-field; episodes-metadata schema and the offset-resolution algorithm (worked example: episode 30 → file, row, video timestamp); stats/tasks schemas; windowing pseudocode with the clamp and pad rules stated precisely; boundary semantics with the episode-start worked example; a "v2.1 vs v3" diff section; a gotchas section (float tolerance, AV1, keyframe seeking).

**✅ Checkpoint:** someone with FORMAT.md and no lerobot source could answer: "sample #100, 3-frame history, episode boundary two frames back — what exactly comes back?"

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `parser/` (`meta.py`, `tabular.py`, `video.py`, `windowing.py`, `dataset.py`) | zero `lerobot` imports; typed; each module independently importable |
| `tests/` | parity tests + property tests, all green in CI; runtime < 5 min |
| `FORMAT.md` | contents list above, including the two worked examples |
| `RESULTS.md` | parity summary table; any semantics you got wrong on the first draft and the rule you'd misassumed; stretch numbers if run |

## Done when

- [ ] Parity: `torch.equal` on all non-image keys and pad masks for sample #100 + boundary cases; images within 1/255.
- [ ] Property suite green with the five invariants.
- [ ] `FORMAT.md` passes the stranger test above.
- [ ] All of it reproducible via `pytest` from a fresh clone + snapshot.

## Self-check

1. Why does v3 concatenate episodes into shared files, and name the two costs it accepts to get that (hint: random access; partial download granularity).
2. `tolerance_s` defaults to `1e-4`. What breaks if you set it to `1/(2·fps)`? To `1/fps`?
3. Why must padding produce a *mask* rather than, say, zero-filled frames? Which consumer breaks otherwise (think Lesson 14's action chunks at episode end)?
4. Where does the v3 format store the mapping from a frame to its video timestamp, and why can't you derive it from the frame index alone in general?
5. Your parser decodes frame $k$ by seeking a keyframe then rolling forward. Why is seeking directly to $t = k/\text{fps}$ wrong for AV1/H.264?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Off-by-one pad masks at episode end | using `> ep_end` instead of `>= ep_end` (end-exclusive) | boundary convention: valid indices are `[ep_start, ep_end)` |
| Delta validation rejects `-1/30` | `int()` truncation instead of `round()` in delta→index | `k = round(d * fps)`, then verify `abs(d - k/fps) ≤ tolerance_s` |
| Images differ by 1–2/255 sporadically | different decode backend or colorspace conversion than the library's | match the library's backend (pyav) and BT.601/BT.709 handling; assert which one in FORMAT.md |
| Decoded frame is a few frames early | stopped at keyframe after `av` seek | decode forward until `frame.pts × time_base` reaches the target |
| Hypothesis suite times out | property tests routed through video decode | fuzz windowing on synthetic indices only (Part 4 note) |
| Everything off by whole episodes | assumed episode boundaries from file boundaries | boundaries live in `meta/episodes/` only — that's the v3 point |

## Stretch

Benchmark `StreamingLeRobotDataset` vs local `LeRobotDataset` vs naive `datasets` streaming: grid over batch size × window size × dataloader workers; verify or refute the tutorial's 80–100 it/s streaming claim; attribute the bottleneck (video decode vs network) with a profile, not a guess. Table + one paragraph in `RESULTS.md`.

## References

- LeRobot team, *Robot Learning: A Tutorial*, §1. arXiv:2510.12403.
- LeRobot dataset v3 docs + `src/lerobot/datasets/dataset_reader.py` (installed version).
- Dataset: `lerobot/svla_so101_pickplace` (50 eps / 11,939 frames / 30 fps / AV1).
