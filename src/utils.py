"""Shared utilities: device selection, reproducibility, and data loading."""

import os
import random
import numpy as np
import torch

# ─── Dataset config ──────────────────────────────────────────────────────────
DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "OfficeHomeDataset_10072016")
CLASSES = ["Calculator", "Keyboard", "Laptop", "Monitor", "Mouse", "Printer"]

# ImageNet stats (apply to both domains — see challenge pitfalls section)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE    = 224

# ─── Device ──────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEEDS = [42, 123, 2024]  # Three seeds required by the challenge


# ─── Transforms ──────────────────────────────────────────────────────────────

import torchvision.transforms as T

def get_train_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_eval_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ─── Dataset builder ─────────────────────────────────────────────────────────

import torch.utils.data as data
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from PIL import Image
from typing import Tuple, List, Optional


class SubsetImageDataset(data.Dataset):
    """Thin wrapper over a list of (path, label) pairs with a transform."""

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[T.Compose] = None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_domain_splits(
    domain: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List, List, List]:
    """
    Load all (path, label) pairs for the given domain and split into
    train / val / test with stratification.

    Args:
        domain: Subdirectory name, e.g. "Art" or "Clipart".
        train_ratio: Fraction for train (remaining is split equally val/test).
        val_ratio: Fraction for val out of the total.
        seed: Random seed for the split.

    Returns:
        (train_samples, val_samples, test_samples)
    """
    domain_path = os.path.join(DATA_ROOT, domain)
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}

    all_samples: List[Tuple[str, int]] = []
    for cls in CLASSES:
        cls_dir = os.path.join(domain_path, cls)
        for fname in sorted(os.listdir(cls_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))

    paths  = [s[0] for s in all_samples]
    labels = [s[1] for s in all_samples]

    # First split: train vs. (val + test)
    test_ratio = 1.0 - train_ratio - val_ratio
    paths_train, paths_tmp, labels_train, labels_tmp = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs. test
    val_frac = val_ratio / (val_ratio + test_ratio)
    paths_val, paths_test, labels_val, labels_test = train_test_split(
        paths_tmp, labels_tmp,
        test_size=(1 - val_frac),
        stratify=labels_tmp,
        random_state=seed,
    )

    train = list(zip(paths_train, labels_train))
    val   = list(zip(paths_val,   labels_val))
    test  = list(zip(paths_test,  labels_test))
    return train, val, test


def make_loaders(
    train_samples: List,
    val_samples: List,
    test_samples: List,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """Build DataLoaders for train / val / test splits."""
    train_ds = SubsetImageDataset(train_samples, transform=get_train_transform())
    val_ds   = SubsetImageDataset(val_samples,   transform=get_eval_transform())
    test_ds  = SubsetImageDataset(test_samples,  transform=get_eval_transform())

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ─── Evaluation helper ───────────────────────────────────────────────────────

def evaluate(
    model: torch.nn.Module,
    loader: data.DataLoader,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate model accuracy.

    Returns:
        (overall_accuracy, per_class_accuracy array of shape [num_classes])
    """
    model.eval()
    correct = np.zeros(len(CLASSES))
    total   = np.zeros(len(CLASSES))

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for c in range(len(CLASSES)):
                mask = labels == c
                correct[c] += (preds[mask] == labels[mask]).sum().item()
                total[c]   += mask.sum().item()

    per_class = np.where(total > 0, correct / total, 0.0)
    overall   = correct.sum() / total.sum()
    return float(overall), per_class
