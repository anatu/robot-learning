# Lesson 01 — LeRobotDataset Anatomy

This lesson is about the on-disk format that every later lesson reads from and writes to: version 3 of the LeRobotDataset format. You will download a real dataset as raw files, work out from its metadata how a given (episode, frame) pair is located inside the shared parquet and video files, and then study the one behaviour that policies depend on most directly, the `delta_timestamps` windowing rule that assembles observation histories and action chunks. By the end you should be able to predict what the library returns for any index, including at the first and last frames of an episode where the rules do the most work, and to verify those predictions against the library itself.

| | |
|---|---|
| **Phase** | 1 — Data |
| **Time** | 1 session (3–5 h desk time) |
| **Cost** | $0 (Mac-local) |
| **Prerequisites** | 00 (working environment and `hello_dataset.py`) |
| **Feeds into** | 02 (you check your own dataset with this lesson's `window()`), 14 and 15 (`delta_timestamps` builds the action chunks those policies train on), H2 (debugging real recordings) |

## Learning objectives

After this lesson you can:

1. **Explain** why version 3 records episode boundaries in metadata rather than expressing them through filenames, and name the two costs of that choice.
2. **Resolve** any (episode, frame) pair to a parquet file, a row within it, and a video timestamp, using the metadata alone.
3. **Predict** the exact indices and padding mask that `delta_timestamps` produces at the first and last frames of an episode, and verify the prediction against the library.
4. **Diagnose** a `tolerance_s` failure from its symptom and state which rule was violated.

## Principles

### Three kinds of data, stored separately

A robot dataset contains three kinds of data with very different access patterns. Low-dimensional, high-rate signals such as joint positions, commanded actions and timestamps are small per frame and are usually read in bulk. Camera frames are large, expensive to decode, and read a few at a time. Metadata, meaning the schema, the frame rate, the normalization statistics and the episode structure, is tiny and is read once at load time. Version 3 stores each kind where it is cheapest to access: tabular signals go into Apache Parquet files, each camera's frames go into MP4 files as a video stream, and metadata goes into JSON and Parquet files under `meta/`. The table below lists the layout.

| Path | Contents |
|---|---|
| `meta/info.json` | the schema (feature names, dtypes, shapes), fps, `codebase_version`, and the path templates for data and video shards |
| `meta/stats.json` | per-feature mean, std, min and max, which the training code reads for normalization |
| `meta/tasks.jsonl` | the mapping from natural-language task string to integer id |
| `meta/episodes/` | chunked parquet with, per episode, its length, its task, and its offsets into the shared data and video files |
| `data/` | `file-XXXX.parquet` shards, each holding many episodes |
| `videos/` | per-camera MP4 shards, each holding many episodes |

The important design choice, and the one this lesson is built around, is that many episodes are concatenated into each parquet file and each MP4 file.

### Episode boundaries live in metadata

In the previous version of the format (v2.1), each episode was its own parquet file, so the filesystem itself recorded where episodes began and ended. That arrangement is simple to reason about, but a dataset with a million episodes then needs a million files, and both local filesystems and object stores handle very large numbers of small files badly. Version 3 inverts the arrangement. Episodes share files, and the per-episode lengths and offsets into those files are stored in `meta/episodes/`. To find frame 17 of episode 30, you read the episode table, look up which parquet shard holds episode 30 and at which row the episode starts, and add 17. The same lookup gives you the video shard and the timestamp within it for each camera.

Two costs come with this design, and you should be able to name both. Random access now requires a metadata lookup before the data read, and you can no longer download a single episode at file granularity, because its neighbours share the file.

### The windowing rule

Policies rarely consume a single frame. ACT (Lesson 14) predicts a chunk of future actions, Diffusion Policy (Lesson 15) conditions on a short history of observations, and in both cases the dataset class assembles the window rather than the policy. The mechanism is the `delta_timestamps` argument. For a feature key you supply a list of time offsets $[d_0, \dots, d_{T-1}]$ in seconds, and for a query at frame index $i$ the dataset returns that feature at times $t_i + d_j$, stacked along a new leading dimension of length $T$. Three rules determine exactly which frames come back.

The first rule is validation. Each offset must be a multiple of the frame period $1/\text{fps}$, to within a tolerance `tolerance_s` whose default is $10^{-4}$ seconds. The offset is converted to an integer frame offset $k_j = \text{round}(d_j \cdot \text{fps})$, and the dataset checks that $|d_j - k_j/\text{fps}| \le$ `tolerance_s`. The tolerance exists because offsets are usually written as floating-point fractions such as `-1/30`, which are not exactly representable; the check rejects offsets that are genuinely off the frame grid while allowing ordinary floating-point error.

The second rule is clamping. The target index $i + k_j$ is clamped to the episode that contains $i$. If the episode occupies indices $[\text{ep\_start}, \text{ep\_end})$, the returned index is $\max(\text{ep\_start}, \min(\text{ep\_end} - 1, i + k_j))$. A window therefore never reaches into a neighbouring episode, even though neighbouring episodes sit next to each other in the same file.

The third rule is padding. For each position $j$ the dataset also returns a boolean tensor named `{key}_is_pad`, which is `True` exactly when the unclamped index $i + k_j$ fell outside the episode. Consider a three-frame history requested at the first frame of an episode, with offsets $[-2/\text{fps}, -1/\text{fps}, 0]$. The first two targets fall before the episode starts, so both are clamped to the first frame, and the mask reads `[True, True, False]`. The returned tensor contains the first frame three times, and the mask tells the consumer that two of those copies are not real history.

It is worth asking why the format reports a mask rather than filling the padded positions with, say, zeros. A zero frame is a perfectly valid-looking observation, and a model trained on it would learn that the robot sometimes sees black. With a mask, the training code can exclude padded positions from the loss instead. The action chunks of Lesson 14 are the same situation in the forward direction: near the end of an episode the chunk extends past the last frame, the extra positions are padded, and the loss is masked accordingly.

### State and action are different quantities

The format has two low-dimensional features that look alike and are not. `observation.state` is what was measured, usually the joint positions read back from the motors. `action` is what was commanded, usually the target positions sent to the motors. The two differ by the tracking error of the controller, and that difference is what makes behaviour cloning on actions meaningful: the policy learns to reproduce the commands that produced the motion, not the motion itself. Lesson 02 makes this concrete by logging both quantities in simulation and plotting the gap.

**Carry forward**

- Episode boundaries are recorded in `meta/episodes/` rather than implied by filenames, because a format that must scale to millions of episodes cannot afford one file per episode.
- Windowing applies three rules in order: each offset must be a multiple of the frame period within `tolerance_s`; the target index is clamped to the episode; and positions whose unclamped index fell outside the episode are flagged in a padding mask.
- Padding is reported as a mask rather than filled with synthetic frames, so that training code can exclude padded positions from the loss instead of learning from invented observations.
- A video frame is located by shard and timestamp from the metadata; once episodes share files, the frame index alone is not enough to find it.

| Source | Read for |
|---|---|
| Tutorial §1.1–1.2 | the reasons the format exists, and the streaming and batching goals it was designed around |
| [LeRobot dataset v3 docs](https://huggingface.co/docs/lerobot/lerobot-dataset-v3) | the layout table above, from the source |
| Installed source: `grep -rn "_get_query_indices\|check_delta_timestamps" $(python -c 'import lerobot,os;print(os.path.dirname(lerobot.__file__))')/datasets` | the reference windowing implementation; read it only after Exercise 3, so that your prediction is your own |

## Exercise 1 — Inspect the on-disk layout [Read]

In this exercise you download the dataset as raw files and look at how it is laid out. The purpose is to see the design decision from the Principles section directly: fifty episodes stored in only a handful of files, which means the episode structure has to be recorded somewhere other than the filesystem.

1. Snapshot the raw repository, bypassing the `LeRobotDataset` API so that you get files rather than tensors:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download("lerobot/svla_so101_pickplace", repo_type="dataset",
                     local_dir="data/svla_so101_pickplace")
   ```
2. Run `tree -h data/svla_so101_pickplace` and check `codebase_version` in `meta/info.json`. If the Hub copy reports v2.1, convert it (the converter module path is in Lesson 00, Part 5) and inspect the v3 output; keep the v2.1 tree, because the difference between the two is material for Exercise 6.
3. Record in `RESULTS.md` the number of parquet shards and the number of MP4 shards per camera, alongside the episode count (50). The ratio between those numbers is the design decision in one line.

**✅ Checkpoint:** the tree matches the layout table, and `RESULTS.md` has the shard-count line.

## Exercise 2 — Locate a frame from metadata alone [Build]

Here you write a small script that resolves an (episode, frame) pair to its storage location using only the metadata files, without importing `lerobot`. Doing the lookup once by hand, and then checking a script that does the same thing against the library, is the most direct way to learn what the episode table contains and how the offsets compose.

Write the specification for `resolve.py` and have an AI tool draft it. The script is about twenty lines and uses `pyarrow`:

- `resolve(root, ep_idx, frame_idx) -> dict` returns the parquet file path, the row index within that file, the global frame index, and for each camera the MP4 shard path and the timestamp within it, all derived from `meta/info.json` and `meta/episodes/`.
- The check: for five (episode, frame) pairs including (30, 17), the state and action read from the resolved parquet row via `pyarrow` must equal `LeRobotDataset(...)[global_idx]` under `torch.equal`.

Before running the script, work out the answer for (30, 17) yourself by reading the episode metadata, and write it in `RESULTS.md`. Then run the script and compare.

**✅ Checkpoint:** all five pairs match the library, and your hand-resolved answer for (30, 17) agrees with the script.

## Exercise 3 — Predict the window at the episode boundaries [Predict → Run]

The three windowing rules are simple to state and easy to apply slightly wrong. In this exercise you apply them by hand to three query positions within one episode, write down what the library should return, and then run the library to check. The first and last frames are where the clamping and padding rules act, so those are the positions worth predicting.

1. Before running anything, write in `RESULTS.md`, for `delta_timestamps={"observation.state": [-2/30, -1/30, 0.0, 1/30]}`, the four gathered indices and the four mask values at (a) the first frame of episode 3, (b) the last frame of episode 3, and (c) a frame in the middle of episode 3. Next to each position, state which rule determined each entry.
2. Run the library at the same three positions:
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset("lerobot/svla_so101_pickplace",
                       delta_timestamps={"observation.state": [-2/30, -1/30, 0.0, 1/30]})
   ep = ds.meta.episodes[3]                      # read the episode's start and end frame indices from here
   for i in (ep_start, ep_end - 1, (ep_start + ep_end) // 2):
       item = ds[i]
       print(i, item["observation.state"].shape, item["observation.state_is_pad"])
   ```
   In v3 the episode table is a tabular object, and the exact accessor for start and end indices varies between minor versions; inspect `ds.meta.episodes` and the docs page to find the fields you need.
3. Reconcile your prediction with the output. If they differ, the difference points to a rule you had misread; correct the rule in your notes rather than adjusting the numbers.
4. Only now open the installed `_get_query_indices` and confirm that the three rules in the Principles section describe what the code does.

**✅ Checkpoint:** the predictions match at all three positions, or `RESULTS.md` names the rule that was misread and why.

## Exercise 4 — Implement `window()` [Build]

The windowing rule is worth having as a standalone function, both because Lesson 02 will use it to check a dataset you write yourself and because writing its specification forces you to state the three rules precisely. Write the specification below and have an AI tool draft the implementation; then read the draft against the rules.

```python
def window(query_idx: int, ep_start: int, ep_end: int,
           deltas: list[float], fps: float, tolerance_s: float = 1e-4
           ) -> tuple[list[int], list[bool]]:
    """Absolute frame indices to gather, and is_pad per position, under the
    three rules: validate (k = round(d*fps), require |d - k/fps| <= tolerance_s),
    clamp to [ep_start, ep_end), pad where the unclamped index was outside."""
```

The check, in `check_window.py`, has two parts. First, for 200 seeded draws of (episode, frame, delta list) over the non-image keys of the real dataset, the indices and mask from `window()` must reproduce the values and `_is_pad` tensors that `LeRobotDataset` returns. Second, five invariants must hold on synthetic inputs: the output length equals `len(deltas)`; a position is padded if and only if its unclamped index is out of range; every returned index lies inside the episode; an all-zero delta list returns `[query_idx]` with no padding; and monotone deltas produce monotone indices.

**✅ Checkpoint:** `python check_window.py` prints `200/200 parity` and reports all five invariants as passing.

## Exercise 5 — Vary the tolerance [Diagnose]

`tolerance_s` looks like an implementation detail, and it is easy to loosen it to make an error go away. This exercise shows what the tolerance protects by changing it and observing what gets through.

1. Before running, write down what you expect to change with `tolerance_s = 1/(2·fps)`, with `tolerance_s = 1/fps`, and with a delta of `-0.0334` (which is not a multiple of 1/30) at the default tolerance.
2. Run the three cases through your `window()` and through `LeRobotDataset`, passing `tolerance_s=` to the constructor.
3. In `RESULTS.md`, explain the mechanism: what a too-loose tolerance lets through, and which downstream consumer it silently corrupts. Lesson 02's recorder and Lesson 14's action chunks are the two to think about.

**✅ Checkpoint:** the three outcomes are recorded along with the rule that produced each, and the default-tolerance case raises an error.

## Exercise 6 — Write the field guide [Write]

`FORMAT.md` is a one-page reference for the format, written so that someone with the page and no source code could answer the question "for sample #100, with a three-frame history, when the episode boundary is two frames back, what exactly comes back?" It should contain the layout table; the offset-resolution procedure with the worked example from Exercise 2 (episode 30 to file, row, and video timestamp); the three windowing rules with the boundary example from Exercise 3; a paragraph on what changed between v2.1 and v3 and why; and a short list of gotchas covering floating-point tolerance, AV1 decoding, and keyframe seeking.

**✅ Checkpoint:** the page answers the question above without reference to anything else.

## Deliverables

| Artifact | Acceptance criteria |
|---|---|
| `resolve.py`, `window.py`, `check_window.py` | no `lerobot` imports in `resolve.py` or `window.py`; `check_window.py` reports 200/200 parity and five passing invariants |
| `FORMAT.md` | one page; answers the Exercise 6 question |
| `RESULTS.md` | the Exercise 3 and Exercise 5 predictions with their reconciliations; the shard-count line; the misread rule, if there was one |

## Done when

- [ ] `check_window.py` reports 200/200 parity and five passing invariants.
- [ ] The three boundary predictions in Exercise 3 were written before the run and reconciled after it.
- [ ] `FORMAT.md` answers the Exercise 6 question.
- [ ] The three tolerance cases in Exercise 5 are explained by mechanism.

## Self-check

1. Why does v3 concatenate episodes into shared files, and what are the two costs of doing so?
2. `tolerance_s` defaults to `1e-4`. What breaks at `1/(2·fps)`? At `1/fps`?
3. Why must padding produce a mask rather than zero-filled frames? Which consumer breaks otherwise?
4. Where does v3 store the mapping from a frame to its video timestamp, and why can't you derive it from the frame index alone?
5. Decoding frame $k$ by seeking directly to time $k/\text{fps}$ is wrong for AV1 and H.264 streams. Why?

## Pitfalls

| Symptom | Cause | Fix |
|---|---|---|
| Off-by-one pad masks at the episode end | using `> ep_end` instead of `>= ep_end` | valid indices are `[ep_start, ep_end)`; the end is exclusive |
| Delta validation rejects `-1/30` | `int()` truncation instead of `round()` when converting the delta | `k = round(d * fps)`, then check `abs(d - k/fps) <= tolerance_s` |
| Everything is off by whole episodes | episode boundaries assumed from file boundaries | boundaries are only in `meta/episodes/` |
| `ds.meta.episodes[3]` is not subscriptable as written | v3 stores episodes as a table, and the accessor differs by minor version | inspect `type(ds.meta.episodes)`; the v3 docs page shows the current accessor |
| Video decode errors or green frames | AV1-encoded MP4s with an old ffmpeg or pyav | `brew install ffmpeg` (version 6 or newer), then reinstall `av` |

## Going deeper

- **A byte-level parser.** Reimplement `LeRobotDataset.__getitem__` with no `lerobot` imports at all: memory-mapped tabular reads with `pyarrow`, and video decoding with `av` that seeks to the keyframe at or before the target and decodes forward to the exact presentation timestamp. The parity bar is `torch.equal` on non-image keys and agreement within 1/255 on images. Property-test `window()` with `hypothesis`.
- **A streaming benchmark.** Compare `StreamingLeRobotDataset` against a local `LeRobotDataset` over a grid of batch size, window size, and dataloader workers. Verify or refute the tutorial's claim of 80–100 iterations per second when streaming, and attribute the bottleneck with a profiler rather than a guess.

## References

- LeRobot team, *Robot Learning: A Tutorial*, §1. arXiv:2510.12403.
- LeRobot dataset v3 documentation, and the installed `datasets/` source.
- Dataset card: `lerobot/svla_so101_pickplace` (50 episodes, 11,939 frames, 30 fps, AV1 video).
