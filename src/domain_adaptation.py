"""
Part C — Domain Shift Measurement and Adaptation.

Three strategies compared on the Art → Clipart shift:
  1. no_adaptation        — best Part A model evaluated on Clipart test set
  2. target_finetuning    — fine-tune last block + head on 50 labelled Clipart images/class
  3. style_augmentation   — retrain with Art + synthetic NST images (from Part B)

Optional (bonus): DANN — Domain-Adversarial Neural Network.
"""

import copy
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from tqdm import tqdm

from src.classifier import (
    build_resnet_finetuned,
    count_trainable_params,
    make_optimizer,
    train_epoch,
)
from src.utils import (
    CLASSES, DATA_ROOT, SEEDS, SubsetImageDataset, evaluate, get_device,
    get_eval_transform, get_train_transform, make_loaders, set_seed,
)

import torch.utils.data as data

NUM_CLASSES = len(CLASSES)
SYNTHETIC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_target")


# ─── Strategy 2: Target domain fine-tuning ───────────────────────────────────

def load_target_samples(
    domain: str = "Clipart",
    n_per_class: int = 50,
    seed: int = 42,
) -> Tuple[List, List]:
    """
    Sample n_per_class labelled images from the target domain for fine-tuning.
    Returns (train_samples, val_samples) — a small hold-out val set is kept.
    """
    import random
    random.seed(seed)

    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    train_samples: List = []
    val_samples:   List = []

    for cls in CLASSES:
        cls_dir = os.path.join(DATA_ROOT, domain, cls)
        imgs = sorted([
            os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        random.shuffle(imgs)
        # 40 train, 10 val (out of 50)
        n_train = min(40, len(imgs))
        n_val   = min(10, max(0, len(imgs) - n_train))
        for p in imgs[:n_train]:
            train_samples.append((p, class_to_idx[cls]))
        for p in imgs[n_train: n_train + n_val]:
            val_samples.append((p, class_to_idx[cls]))

    return train_samples, val_samples


def target_finetuning(
    pretrained_state: dict,
    test_target_loader: data.DataLoader,
    num_epochs: int = 20,
    patience: int = 5,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Fine-tune the last convolutional block + head of a pretrained model
    using a small labelled set from the Clipart domain.
    """
    if device is None:
        device = get_device()

    set_seed(seed)
    model = build_resnet_finetuned(NUM_CLASSES)
    model.load_state_dict(pretrained_state)
    model = model.to(device)

    train_samples, val_samples = load_target_samples(seed=seed)
    train_ds = SubsetImageDataset(train_samples, transform=get_train_transform())
    val_ds   = SubsetImageDataset(val_samples,   transform=get_eval_transform())
    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = data.DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, "finetuned", lr_head=1e-3, lr_backbone=1e-5)

    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())
    no_improve   = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="[target_finetuning]"):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    tgt_acc, per_class = evaluate(model, test_target_loader, device)

    if checkpoint_path:
        torch.save(best_state, checkpoint_path)

    return {"strategy": "target_finetuning", "tgt_acc": tgt_acc, "per_class": per_class}


# ─── Strategy 3: Style-transfer augmentation ─────────────────────────────────

def load_synthetic_samples() -> List:
    """Load all synthetic NST images from data/synthetic_target/."""
    class_to_idx = {cls: i for i, cls in enumerate(CLASSES)}
    samples: List = []
    for cls in CLASSES:
        cls_dir = os.path.join(SYNTHETIC_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        for f in sorted(os.listdir(cls_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((os.path.join(cls_dir, f), class_to_idx[cls]))
    return samples


def style_augmentation_training(
    source_train_samples: List,
    source_val_samples:   List,
    test_source_loader:   data.DataLoader,
    test_target_loader:   data.DataLoader,
    num_epochs: int = 30,
    patience:   int = 7,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Retrain a model from scratch on the Art domain images + synthetic NST images.
    No target-domain labels are used — the synthetic images are unsupervised augmentation.
    """
    if device is None:
        device = get_device()

    set_seed(seed)

    synthetic = load_synthetic_samples()
    augmented_train = source_train_samples + synthetic

    train_ds = SubsetImageDataset(augmented_train, transform=get_train_transform())
    val_ds   = SubsetImageDataset(source_val_samples, transform=get_eval_transform())
    train_loader = data.DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=2)
    val_loader   = data.DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=2)

    model = build_resnet_finetuned(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, "finetuned")

    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())
    no_improve   = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="[style_augmentation]"):
        train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _ = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    src_acc, _ = evaluate(model, test_source_loader, device)
    tgt_acc, per_class = evaluate(model, test_target_loader, device)

    if checkpoint_path:
        torch.save(best_state, checkpoint_path)

    return {
        "strategy": "style_augmentation",
        "src_acc": src_acc, "tgt_acc": tgt_acc, "per_class": per_class,
    }


# ─── Optional: DANN ──────────────────────────────────────────────────────────

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class DANNClassifier(nn.Module):
    """Domain-Adversarial Neural Network (Ganin et al., 2016)."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone    = backbone
        self.class_head  = nn.Linear(feat_dim, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),   # 0 = source, 1 = target
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        feat    = self.backbone(x)
        cls_out = self.class_head(feat)
        rev     = GradientReversal.apply(feat, alpha)
        dom_out = self.domain_head(rev)
        return cls_out, dom_out

    def predict(self, x: torch.Tensor):
        feat = self.backbone(x)
        return self.class_head(feat)


def train_dann(
    source_train_samples: List,
    source_val_samples:   List,
    test_source_loader:   data.DataLoader,
    test_target_loader:   data.DataLoader,
    target_unlabelled_samples: List,
    num_epochs:  int   = 30,
    lambda_d:    float = 1.0,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Train DANN with source labelled data and unlabelled target images.
    Uses a linear alpha schedule: alpha goes from 0 to lambda_d over training.
    """
    if device is None:
        device = get_device()

    set_seed(seed)
    model = DANNClassifier(NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    src_ds = SubsetImageDataset(source_train_samples, transform=get_train_transform())
    tgt_ds = SubsetImageDataset(
        [(p, 0) for p, _ in target_unlabelled_samples],  # labels ignored
        transform=get_train_transform(),
    )

    src_loader = data.DataLoader(src_ds, batch_size=32, shuffle=True,  num_workers=2, drop_last=True)
    tgt_loader = data.DataLoader(tgt_ds, batch_size=32, shuffle=True,  num_workers=2, drop_last=True)

    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(1, num_epochs + 1), desc="[DANN]"):
        model.train()
        alpha = lambda_d * (epoch / num_epochs)  # linear schedule 0 → lambda_d

        for (src_imgs, src_labels), (tgt_imgs, _) in zip(src_loader, tgt_loader):
            src_imgs   = src_imgs.to(device)
            src_labels = src_labels.to(device)
            tgt_imgs   = tgt_imgs.to(device)

            dom_labels = torch.cat([
                torch.zeros(src_imgs.size(0), dtype=torch.long, device=device),
                torch.ones( tgt_imgs.size(0), dtype=torch.long, device=device),
            ])

            all_imgs = torch.cat([src_imgs, tgt_imgs])
            cls_out, dom_out = model(all_imgs, alpha=alpha)

            cls_loss = criterion(cls_out[:src_imgs.size(0)], src_labels)
            dom_loss = criterion(dom_out, dom_labels)
            loss     = cls_loss + lambda_d * dom_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate on source val
        val_ds = SubsetImageDataset(source_val_samples, transform=get_eval_transform())
        val_loader = data.DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model.predict(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    # Evaluate using predict (class head only)
    def dann_evaluate(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                preds = model.predict(imgs).argmax(dim=1)
                correct += (preds == lbls).sum().item()
                total   += lbls.size(0)
        return correct / total

    src_acc = dann_evaluate(test_source_loader)
    tgt_acc = dann_evaluate(test_target_loader)

    if checkpoint_path:
        torch.save(best_state, checkpoint_path)

    return {
        "strategy": "dann",
        "src_acc": src_acc, "tgt_acc": tgt_acc,
        "lambda_d": lambda_d,
    }


# ─── Summary table builder ────────────────────────────────────────────────────

def build_summary_table(results: List[Dict]) -> "pd.DataFrame":
    """Convert a list of result dicts into a Pandas summary table."""
    import pandas as pd

    rows = []
    for r in results:
        src  = r.get("src_mean", r.get("src_acc", float("nan")))
        src_s = r.get("src_std", 0.0)
        tgt  = r.get("tgt_mean", r.get("tgt_acc", float("nan")))
        tgt_s = r.get("tgt_std", 0.0)
        rows.append({
            "Strategy":          r["strategy"],
            "Source Acc (mean)": f"{src:.4f} ± {src_s:.4f}",
            "Target Acc (mean)": f"{tgt:.4f} ± {tgt_s:.4f}",
            "Δ_shift":           f"{src - tgt:.4f}",
        })
    return pd.DataFrame(rows)
