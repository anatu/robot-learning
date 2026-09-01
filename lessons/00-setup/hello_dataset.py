from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset(
    "lerobot/svla_so101_pickplace",
    delta_timestamps={"observation.images.up": [-2 / 30, -1 / 30, 0.0]},
)
print(ds.meta.fps, ds.meta.total_episodes, ds.meta.total_frames)
item = ds[100]
for k, v in item.items():
    print(k, getattr(v, "shape", None), getattr(v, "dtype", type(v)))
