# Lesson 01 — LeRobotDataset Anatomy

Know the v3 format well enough to predict, from metadata alone, exactly what `LeRobotDataset` returns for any index and window — and prove it at the episode boundaries, where the rules bite.

| | |
|---|---|
| **Phase** | 1 — Data |
| **Time** | 1 session (3–5 h desk time) |
| **Cost** | $0 (Mac-local) |
| **Prerequisites** | 00 (env + `hello_dataset.py`) |
| **Feeds into** | 02 (your own dataset gets checked with this lesson's `window()`), 14/15 (`delta_timestamps` builds the action chunks), H2 (debugging real recordings) |

## Learning objectives

After this lesson you can:

1. **Explain** why v3 resolves episode boundaries through metadata rather than filenames, and name the two costs it accepts.
2. **Resolve** any (episode, frame) pair to a parquet file, row, and video timestamp from metadata alone.
3. **Predict** the exact indices and padding mask `delta_timestamps` produces at an episode's first and last frames, then verify it.
4. **Diagnose** a `tolerance_s` failure from its symptom and state the rule that was violated.

## Principles

**Three pillars, decoupled from the API.** v3 stores (1) *tabular* signals (states, actions, timestamps) in Parquet with many episodes per file; (2) *visual* streams as MP4 shards per camera, many episodes per file; (3) *metadata* holding the schema, fps, normalization stats, and the load-bearing part: **episode segmentation**, the per-episode lengths and offsets into the shared files. v2.1 had one parquet per episode, so boundaries were filenames. v3 inverted that so a million episodes don't mean a million files. The costs: random access needs an offset lookup first, and you can no longer download "just episode 30" at file granularity.

| Path | Contents |
|---|---|
| `meta/info.json` | schema (feature names/dtypes/shapes), fps, `codebase_version`, path templates for data/video shards |
| `meta/stats.json` | per-feature mean/std/min/max (normalization reads this) |
| `meta/tasks.jsonl` | natural-language task → integer id |
| `meta/episodes/` | chunked parquet: per-episode lengths, tasks, offsets into shared files |
| `data/` | `file-XXXX.parquet` shards, many episodes each |
| `videos/` | per-camera MP4 shards, many episodes each |

**Windowing is the rule you will use forever.** `delta_timestamps={"key": [d_0, …, d_{T-1}]}` requests, for query index $i$, frames at times $t_i + d_j$. Three rules:

1. *Validation:* every $d_j$ must be a multiple of $1/\text{fps}$ within `tolerance_s` (default `1e-4`). The delta becomes an index offset $k_j = \text{round}(d_j \cdot \text{fps})$.
2. *Clamping:* the target index $i + k_j$ is clamped to the episode: $\max(\text{ep\_start}, \min(\text{ep\_end} - 1, i + k_j))$. Windows never cross episodes.
3. *Padding mask:* `{key}_is_pad[j]` is `True` exactly when the *unclamped* index fell outside $[\text{ep\_start}, \text{ep\_end})$.

So a 3-frame history at an episode's first frame returns that frame three times with mask `[True, True, False]`. Policies mask their loss with exactly these tensors (Lesson 14's action chunks at episode end are the same rule, forward in time). A mask, not zero-filled frames, because a zero frame is a *valid-looking* observation and the model would learn from it.

**State ≠ action, decided by the format.** `observation.state` is what was measured; `action` is what was commanded. They differ by tracking error. Lesson 02 makes this concrete.

**Carry forward**

- Episode boundaries are in `meta/episodes/`, never in filenames.
- Windowing = validate (multiple of 1/fps) → clamp to episode → mask the clamped positions.
- Padding is a mask because a filled frame would be a lie the model learns.
- Video frames are located by (shard, timestamp) from metadata; frame index alone is not enough once episodes share files.

| Source | Read for |
|---|---|
| Tutorial §1.1–1.2 | why the format exists; the streaming/batching goals it optimizes for |
| [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | the layout table above, from the source |
| Installed source: `grep -rn "_get_query_indices\|check_delta_timestamps" $(python -c 'import lerobot,os;print(os.path.dirname(lerobot.__file__))')/datasets` | the reference windowing semantics — read only *after* Exercise 3 |

## Exercise 1 — Walk the bytes [Read]

Tests objective 1: the tree is the argument for metadata-resolved boundaries.

1. Snapshot the raw repo (files, not the `LeRobotDataset` API):
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("lerobot/svla_so101_pickplace", repo_type="dataset",
                     local_dir="data/svla_so101_pickplace")
   ```
2. `tree -h data/svla_so101_pickplace`; check `meta/info.json → codebase_version`. If it reports v2.1, convert (converter module path per Lesson 00 Part 5) and walk the **v3** output; keep the v2.1 tree for the diff in Exercise 6.
3. Write in `RESULTS.md`: the number of parquet shards and MP4 shards per camera versus the number of episodes (50). That ratio is the whole design.

**✅ Checkpoint:** the tree matches the layout table; you have the shard-count vs episode-count line.

## Exercise 2 — Resolve (episode 30, frame 17) [Build]

Tests objective 2. Spec for a small script `resolve.py` (an AI tool drafts it; ~20 lines with `pyarrow`, no `lerobot` imports):

- `resolve(root, ep_idx, frame_idx) -> dict` returns the parquet file path, row index within that file, the global frame index, and for each camera the MP4 shard path and in-file timestamp, all read from `meta/info.json` + `meta/episodes/`.
- The check: for 5 (episode, frame) pairs including (30, 17), the state/action read from that parquet row via `pyarrow` equals `LeRobotDataset(...)[global_idx]` under `torch.equal`.

Before running, write your answer for (30, 17) by reading the episode metadata yourself. Then run.

**✅ Checkpoint:** 5/5 parity; your hand-resolved answer for (30, 17) matches the script.

## Exercise 3 — Windowing at the boundaries [Predict → Run]

Tests objective 3: the clamp and pad rules, at the places they matter.

1. **Write first**, in `RESULTS.md`, for `delta_timestamps={"observation.state": [-2/30, -1/30, 0.0, 1/30]}`: the four gathered indices and the four mask values at (a) episode 3's first frame, (b) episode 3's last frame, (c) a mid-episode frame. State the rule you used for each position.
2. Run:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset("lerobot/svla_so101_pickplace",
                       delta_timestamps={"observation.state": [-2/30, -1/30, 0.0, 1/30]})
   ep = ds.meta.episodes[3]                      # inspect: episode start/end frame indices
   for i in (ep_start, ep_end - 1, (ep_start + ep_end) // 2):
       item = ds[i]
       print(i, item["observation.state"].shape, item["observation.state_is_pad"])
   ```
   (`ds.meta.episodes` is a table in v3; pull the start/end from it however your version exposes it. Verify with `ds.meta.episodes` and the docs page.)
3. Reconcile. Any mismatch is a rule you had wrong; fix the rule in your notes, not the number.
4. Now open the installed `_get_query_indices` and confirm the three rules from Principles against the code.

**✅ Checkpoint:** predictions match at all three positions, or the reconciliation names the misread rule.

## Exercise 4 — `window()` [Build]

Tests objective 3 as a reusable check. This is the one kernel from this lesson that later lessons reuse (Lesson 02 runs it against your own dataset). Spec:

```python
def window(query_idx: int, ep_start: int, ep_end: int,
           deltas: list[float], fps: float, tolerance_s: float = 1e-4
           ) -> tuple[list[int], list[bool]]:
    """Absolute frame indices to gather, and is_pad per position, under the
    three rules: validate (k = round(d*fps), require |d - k/fps| <= tolerance_s),
    clamp to [ep_start, ep_end), pad where the unclamped index was outside."""
```

The check (`check_window.py`): for 200 seeded (episode, frame, delta-list) draws over non-image keys of the real dataset, `window()`'s indices and mask reproduce `LeRobotDataset`'s returned values and `_is_pad` tensors exactly. Add the five invariants as asserts on synthetic inputs: output length equals `len(deltas)`; pad iff unclamped index out of range; all indices inside the episode; all-zero deltas return `[query_idx]` unpadded; monotone deltas give monotone indices.

**✅ Checkpoint:** `python check_window.py` prints `200/200 parity` and the five invariants pass.

## Exercise 5 — Break the tolerance [Diagnose]

Tests objective 4: what `tolerance_s` protects.

1. Predict, in writing: with `tolerance_s = 1/(2·fps)` what changes? With `tolerance_s = 1/fps`? With a delta of `-0.0334` (not a multiple of 1/30) at the default tolerance?
2. Run the three cases through `window()` and through `LeRobotDataset` (pass `tolerance_s=` to the constructor).
3. Explain the mechanism in `RESULTS.md`: what a too-loose tolerance lets through and which downstream consumer (Lesson 02's recorder, Lesson 14's chunks) it silently corrupts.

**✅ Checkpoint:** the three outcomes are recorded with the rule that produced each; the default-tolerance case raises.

## Exercise 6 — Field guide [Write]

`FORMAT.md`, one page. Bar: a stranger with this page and no source could answer "sample #100, 3-frame history, episode boundary two frames back — what comes back?" Required: the layout table; the offset-resolution worked example from Exercise 2 (episode 30 → file, row, video timestamp); the three windowing rules with the boundary worked example from Exercise 3; a v2.1 vs v3 diff paragraph; gotchas (float tolerance, AV1 decode, keyframe seeking).

**✅ Checkpoint:** the stranger question is answered on the page.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `resolve.py`, `window.py`, `check_window.py` | no `lerobot` imports in `resolve.py`/`window.py`; `check_window.py` reports 200/200 parity |
| `FORMAT.md` | one page; passes the stranger question |
| `RESULTS.md` | Exercise 3 and 5 predictions with reconciliations; shard-count line; the misread rule, if any |

## Done when

- [ ] `check_window.py` green: 200/200 parity + five invariants.
- [ ] Exercise 3's three boundary predictions are written *before* the run and reconciled after.
- [ ] `FORMAT.md` answers the stranger question.
- [ ] Exercise 5's three tolerance cases are explained by mechanism.

## Self-check

1. Why does v3 concatenate episodes into shared files, and what are the two costs?
2. `tolerance_s` defaults to `1e-4`. What breaks at `1/(2·fps)`? At `1/fps`?
3. Why must padding produce a *mask* rather than zero-filled frames? Which consumer breaks otherwise?
4. Where does v3 store the mapping from a frame to its video timestamp, and why can't you derive it from the frame index alone?
5. Decoding frame $k$ by seeking directly to $t = k/\text{fps}$ is wrong for AV1/H.264. Why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Off-by-one pad masks at episode end | `> ep_end` instead of `>= ep_end` | valid indices are `[ep_start, ep_end)`, end-exclusive |
| Delta validation rejects `-1/30` | `int()` truncation instead of `round()` | `k = round(d * fps)`, then `abs(d - k/fps) <= tolerance_s` |
| Everything off by whole episodes | assumed episode boundaries from file boundaries | boundaries live in `meta/episodes/` only |
| `ds.meta.episodes[3]` isn't subscriptable like that | v3 stores episodes as a table, accessor differs by minor version | inspect `type(ds.meta.episodes)`; the v3 docs page shows the current accessor |
| Video decode errors / green frames | AV1 MP4s vs old ffmpeg/pyav | `brew install ffmpeg` (≥ 6), reinstall `av` |

## Going deeper

- **Byte-level parser.** Reimplement `LeRobotDataset.__getitem__` with zero `lerobot` imports: `pyarrow` memory-mapped tabular reads and `av` video decoding (seek to the keyframe at or before the target, decode forward to the exact `pts`). Parity: `torch.equal` on non-image keys, images within 1/255. Property-test `window()` with `hypothesis`.
- **Streaming benchmark.** `StreamingLeRobotDataset` vs local `LeRobotDataset` over batch size × window size × workers; verify or refute the tutorial's 80–100 it/s streaming claim and attribute the bottleneck with a profile.

## References

- LeRobot team, *Robot Learning: A Tutorial*, §1. arXiv:2510.12403.
- LeRobot dataset v3 docs + the installed `datasets/` source.
- Dataset: `lerobot/svla_so101_pickplace` (50 eps / 11,939 frames / 30 fps / AV1).
