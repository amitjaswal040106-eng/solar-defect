import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from data import get_loaders, MEAN, STD
from models import get_model, cam_layer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet18", choices=["cnn", "resnet18"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n", type=int, default=4)
args = ap.parse_args()

model = get_model(args.model)  # CPU on purpose, small n
model.load_state_dict(torch.load(f"outputs/{args.model}_seed{args.seed}.pt", map_location="cpu"))
model.eval()

loaders, _, _, _ = get_loaders(batch_size=64)
xs, ys = next(iter(loaders["test"]))
with torch.no_grad():
    preds = model(xs).argmax(1)

chosen = [(i, "correct") for i in (preds == ys).nonzero().flatten()[:args.n].tolist()] + \
         [(i, "wrong") for i in (preds != ys).nonzero().flatten()[:args.n].tolist()]

cam = GradCAM(model=model, target_layers=[cam_layer(model, args.model)])
mean, std = np.array(MEAN), np.array(STD)

fig, axes = plt.subplots(2, len(chosen), figsize=(3 * len(chosen), 6), squeeze=False)
for col, (i, tag) in enumerate(chosen):
    heat = cam(input_tensor=xs[i:i+1], targets=None)[0]
    rgb = np.clip(xs[i].permute(1, 2, 0).numpy() * std + mean, 0, 1).astype(np.float32)
    axes[0, col].imshow(rgb)
    axes[0, col].set_title(f"{tag}\ntrue={int(ys[i])} pred={int(preds[i])}", fontsize=9)
    axes[1, col].imshow(show_cam_on_image(rgb, heat, use_rgb=True))
    axes[0, col].axis("off"); axes[1, col].axis("off")
plt.tight_layout()
plt.savefig(f"outputs/{args.model}_seed{args.seed}_gradcam.png", dpi=150)