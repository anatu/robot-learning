# Lesson 01 — LeRobotDataset Anatomy

**Goal:** understand the v3 dataset format at the byte level — the substrate everything else in this course sits on.

## Read
- Tutorial §1.1–1.2 (LeRobotDataset design, streaming/batching): https://arxiv.org/abs/2510.12403
- LeRobot dataset docs for the current release.

## Build
1. Download `lerobot/svla_so101_pickplace`. Write a parser that reads `meta/info.json`, `meta/stats.json`, `meta/tasks.jsonl`, the chunked parquet files, and the per-episode/camera MP4s **without importing `LeRobotDataset`**.
2. Reimplement `delta_timestamps` windowing (nearest-frame matching, tolerance, episode-boundary semantics) as a standalone function.
3. Reconstruct sample #100 (with a 3-frame wrist-camera history) and assert tensor-equality against what `LeRobotDataset` returns.
4. Fuzz the windowing logic against library behavior across episode lengths (`hypothesis` property tests).

## Deliverables
- `parser/` + passing `pytest` proving parity with the library.
- `FORMAT.md`: your own documentation of the v3 on-disk format, including the boundary semantics you discovered.

## Done when
Parity tests pass; a reader could implement the format from your `FORMAT.md` alone.

## Stretch
Benchmark `StreamingLeRobotDataset` vs local vs naive HF-datasets streaming (batch size × window size × workers); verify or refute the paper's 80–100 it/s claim and identify the bottleneck (video decode vs network).
