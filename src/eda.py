import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import load_elpv, get_splits

images, labels, df = load_elpv()
print("Total images:", len(df))
print("\nRaw defect probabilities:\n", df["proba"].value_counts().sort_index())
print("\nCell types:\n", df["type"].value_counts())
print("\nBinary labels (0=healthy, 1=defective):", np.bincount(labels))
print("Image size example:", images[0].size)

tr, va, te = get_splits(labels)
for name, s in zip(["train", "val", "test"], [tr, va, te]):
    print(f"{name}: {len(s)} images, defective share {labels[s].mean():.2f}")

fig, axes = plt.subplots(2, 6, figsize=(14, 5))
for row, cls in enumerate([0, 1]):
    idx = np.where(labels == cls)[0][:6]
    for ax, i in zip(axes[row], idx):
        ax.imshow(images[i], cmap="gray"); ax.axis("off")
    axes[row][0].set_title("healthy" if cls == 0 else "defective", loc="left")
plt.savefig("outputs/samples.png", dpi=120, bbox_inches="tight")