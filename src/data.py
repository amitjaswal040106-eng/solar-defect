import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

ROOT = Path("data/elpv-dataset/src/elpv_dataset/data")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
SPLIT_SEED = 42  # fixed, so every model sees the same split


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_elpv(threshold=0.5):
    df = pd.read_csv(ROOT / "labels.csv", sep=r"\s+", header=None,
                     names=["path", "proba", "type"])
    images = [Image.open(ROOT / p).convert("L") for p in df["path"]]
    labels = (df["proba"].values >= threshold).astype(int)
    return images, labels, df


def get_splits(labels, seed=SPLIT_SEED):
    idx = np.arange(len(labels))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=labels, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=labels[tmp], random_state=seed)
    return tr, va, te


def train_tf(size=224):
    return T.Compose([
        T.Grayscale(num_output_channels=3), T.Resize((size, size)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(10),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_tf(size=224):
    return T.Compose([
        T.Grayscale(num_output_channels=3), T.Resize((size, size)),
        T.ToTensor(), T.Normalize(MEAN, STD)])


class ELPV(Dataset):
    def __init__(self, images, labels, indices, transform):
        self.images, self.labels = images, labels
        self.indices, self.transform = indices, transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.transform(self.images[j]), int(self.labels[j])


def get_loaders(batch_size=32, threshold=0.5):
    images, labels, df = load_elpv(threshold)
    tr, va, te = get_splits(labels)
    loaders = {
        "train": DataLoader(ELPV(images, labels, tr, train_tf()),
                            batch_size=batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(ELPV(images, labels, va, eval_tf()),
                          batch_size=batch_size, num_workers=0),
        "test": DataLoader(ELPV(images, labels, te, eval_tf()),
                           batch_size=batch_size, num_workers=0),
    }
    return loaders, labels, (tr, va, te), df