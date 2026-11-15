import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, f1_score)
from data import get_device, get_loaders
from models import get_model

NAMES = ["healthy", "defective"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="resnet18", choices=["cnn", "resnet18"])
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

device = get_device()
loaders, labels, (tr, va, te), df = get_loaders()
y_true = labels[te]

maj = np.bincount(labels[tr]).argmax()
y_maj = np.full_like(y_true, maj)
print("=== Majority baseline ===")
print(f"accuracy {accuracy_score(y_true, y_maj):.3f} | "
      f"macro-F1 {f1_score(y_true, y_maj, average='macro', zero_division=0):.3f}\n")

model = get_model(args.model)
model.load_state_dict(torch.load(f"outputs/{args.model}_seed{args.seed}.pt", map_location="cpu"))
model.to(device).eval()

preds = []
with torch.no_grad():
    for x, _ in loaders["test"]:
        preds += model(x.to(device)).argmax(1).cpu().tolist()
preds = np.array(preds)  # test loader is unshuffled, so order matches `te`

print(f"=== {args.model} (seed {args.seed}) ===")
print(classification_report(y_true, preds, target_names=NAMES, digits=3))

ConfusionMatrixDisplay(confusion_matrix(y_true, preds), display_labels=NAMES).plot(cmap="Blues")
plt.savefig(f"outputs/{args.model}_seed{args.seed}_cm.png", dpi=150, bbox_inches="tight")

wrong = te[preds != y_true]
out = df.iloc[wrong].copy()
out["true"], out["pred"] = labels[wrong], preds[preds != y_true]
out.to_csv(f"outputs/{args.model}_seed{args.seed}_misclassified.csv", index=False)
print(len(wrong), "misclassified saved")