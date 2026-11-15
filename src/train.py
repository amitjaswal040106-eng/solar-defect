import argparse, copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from data import get_device, set_seed, get_loaders
from models import get_model


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train(train)
    losses, preds, ys = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
            preds += out.argmax(1).cpu().tolist()
            ys += y.cpu().tolist()
    return float(np.mean(losses)), f1_score(ys, preds, average="macro")


def fit(model, loaders, criterion, optimizer, device, epochs, patience, state):
    bad = 0
    for ep in range(1, epochs + 1):
        trl, trf = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        vl, vf = run_epoch(model, loaders["val"], criterion, None, device, False)
        print(f"  ep {ep:02d} | train loss {trl:.3f} F1 {trf:.3f} | val loss {vl:.3f} F1 {vf:.3f}")
        if vf > state["best_f1"]:
            state["best_f1"], state["best_w"], bad = vf, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("  early stop"); break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18", choices=["cnn", "resnet18"])
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30, help="cnn only")
    ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_ft", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print("device:", device)
    loaders, labels, (tr, va, te), _ = get_loaders(args.bs)

    counts = np.bincount(labels[tr], minlength=2)
    w = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model = get_model(args.model).to(device)
    state = {"best_f1": -1.0, "best_w": None}

    if args.model == "cnn":
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        fit(model, loaders, criterion, opt, device, args.epochs, args.patience, state)
    else:
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith("fc")
        opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
        print("Stage A: head only")
        fit(model, loaders, criterion, opt, device, args.epochs_head, args.patience, state)

        for p in model.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        print("Stage B: full fine-tuning")
        fit(model, loaders, criterion, opt, device, args.epochs_ft, args.patience, state)

    path = f"outputs/{args.model}_seed{args.seed}.pt"
    torch.save(state["best_w"], path)
    print(f"best val macro-F1 {state['best_f1']:.3f} -> saved {path}")


if __name__ == "__main__":
    main()