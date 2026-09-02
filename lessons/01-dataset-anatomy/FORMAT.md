# LeRobotDataset v3 — Field Guide

<!-- One page. Bar: a stranger with this page and no source could answer "sample #100, 3-frame
     history, episode boundary two frames back — what exactly comes back?" (Lesson 01, Exercise 6). -->

## Layout

<!-- The directory tree and the layout table: meta/info.json, meta/stats.json, meta/tasks.jsonl,
     meta/episodes/, data/file-XXXX.parquet, videos/<camera>/file-XXXX.mp4 — one line each on contents. -->

## Resolving (episode, frame) → bytes

<!-- The offset-resolution algorithm, then the worked example from Exercise 2:
     episode 30, frame 17 → parquet file, row, global index, per-camera MP4 shard + timestamp. -->

## Windowing: the three rules

<!-- 1. validate: k = round(d*fps), require |d - k/fps| <= tolerance_s
     2. clamp: max(ep_start, min(ep_end - 1, i + k))
     3. pad: is_pad[j] iff the unclamped index was outside [ep_start, ep_end)
     Then the boundary worked example from Exercise 3 (first frame, last frame, mid-episode). -->

## v2.1 vs v3

<!-- One paragraph: what moved from filenames into metadata, and the two costs. -->

## Gotchas

<!-- float tolerance (round, not int); AV1 decode; keyframe seeking (seek to the keyframe at or
     before the target, decode forward to the exact pts). -->
