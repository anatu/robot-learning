import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("lerobot/svla_so101_pickplace", episodes=[0])
idxs = torch.linspace(0, len(ds) - 1, 9).long()

fig, axes = plt.subplots(3, 3, figsize=(12, 9))
for ax, i in zip(axes.flat, idxs):
    item = ds[int(i)]
    ax.imshow(item["observation.images.up"].permute(1, 2, 0))
    ax.set_title(f"frame {int(i)}  t={float(item['timestamp']):.2f}s", fontsize=9)
    ax.axis("off")
fig.suptitle(f"svla_so101_pickplace ep0 — {item['task']}", fontsize=11)
fig.tight_layout()
fig.savefig("hello_grid.png", dpi=150)
print(f"saved hello_grid.png ({len(ds)} frames in episode, sampled {list(map(int, idxs))})")
