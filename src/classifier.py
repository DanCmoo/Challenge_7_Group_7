"""
Part A — Few-Shot Classification with Transfer Learning.

Three strategies:
  1. from_scratch  — ResNet-50 with random weights (baseline)
  2. frozen        — ResNet-50 pretrained, backbone frozen, only head trained
  3. finetuned     — ResNet-50 pretrained, layer3+layer4+head unfrozen
"""

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

from src.utils import CLASSES, SEEDS, evaluate, get_device, set_seed

NUM_CLASSES = len(CLASSES)


# ─── Model builders ──────────────────────────────────────────────────────────

def build_resnet_scratch(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with random initialization (no pretrained weights)."""
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet_frozen(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 with frozen backbone — only the classification head is trained."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head (trainable by default)
    return model


def build_resnet_finetuned(num_classes: int = NUM_CLASSES) -> nn.Module:
    """ResNet-50 pretrained, with layer3 + layer4 + head unfrozen for fine-tuning."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze last two residual blocks and the head
    for name, param in model.named_parameters():
        if any(layer in name for layer in ("layer3", "layer4", "fc")):
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Optimizer factory ───────────────────────────────────────────────────────

def make_optimizer(
    model: nn.Module,
    strategy: str,
    lr_head: float = 1e-3,
    lr_backbone: float = 1e-4,
) -> torch.optim.Optimizer:
    """
    Return an Adam optimizer.

    - 'scratch' / 'frozen': single lr for all trainable params.
    - 'finetuned': lower lr for backbone layers, higher for head.
    """
    if strategy == "finetuned":
        head_params     = [p for n, p in model.named_parameters() if "fc" in n and p.requires_grad]
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n and p.requires_grad]
        return torch.optim.Adam(
            [
                {"params": backbone_params, "lr": lr_backbone},
                {"params": head_params,     "lr": lr_head},
            ]
        )
    trainable = filter(lambda p: p.requires_grad, model.parameters())
    lr = lr_head if strategy == "frozen" else 1e-3
    return torch.optim.Adam(trainable, lr=lr)


# ─── Training loop ───────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    strategy: str,
    num_epochs: int = 30,
    patience: int = 7,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict:
    """
    Train with early stopping on validation accuracy.

    Returns a dict with keys:
        model        — best model state_dict
        history      — dict with train_loss, train_acc, val_loss, val_acc per epoch
        best_val_acc — float
        best_epoch   — int
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, strategy)

    best_val_acc  = 0.0
    best_epoch    = 0
    best_state    = copy.deepcopy(model.state_dict())
    no_improve    = 0

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []
    }

    epochs_iter = tqdm(range(1, num_epochs + 1), desc=f"[{strategy}]") if verbose else range(1, num_epochs + 1)

    for epoch in epochs_iter:
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, _ = evaluate(model, val_loader, device)
        # val_loss (lightweight — reuse criterion on val set)
        val_loss = _compute_loss(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose and hasattr(epochs_iter, "set_postfix"):
            epochs_iter.set_postfix(
                train_acc=f"{train_acc:.3f}",
                val_acc=f"{val_acc:.3f}",
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    return {
        "model":        best_state,
        "history":      history,
        "best_val_acc": best_val_acc,
        "best_epoch":   best_epoch,
    }


def _compute_loss(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            total      += images.size(0)
    return total_loss / total


# ─── Multi-seed evaluation ────────────────────────────────────────────────────

def run_multi_seed_experiment(
    strategy: str,
    train_loader_fn,
    val_loader_fn,
    test_source_loader: torch.utils.data.DataLoader,
    test_target_loader: torch.utils.data.DataLoader,
    num_epochs: int = 30,
    checkpoint_dir: Optional[str] = None,
) -> Dict:
    """
    Run train+evaluate over SEEDS (3 seeds).
    Returns summary with mean ± std for source and target accuracy.
    """
    device = get_device()
    src_accs, tgt_accs = [], []

    builder = {
        "scratch":   build_resnet_scratch,
        "frozen":    build_resnet_frozen,
        "finetuned": build_resnet_finetuned,
    }[strategy]

    for seed in SEEDS:
        set_seed(seed)
        model  = builder()
        train_loader = train_loader_fn(seed)
        val_loader   = val_loader_fn(seed)

        result = train_model(
            model, train_loader, val_loader,
            strategy=strategy,
            num_epochs=num_epochs,
            device=device,
        )

        model.load_state_dict(result["model"])
        src_acc, _ = evaluate(model, test_source_loader, device)
        tgt_acc, _ = evaluate(model, test_target_loader, device)
        src_accs.append(src_acc)
        tgt_accs.append(tgt_acc)

        if checkpoint_dir:
            path = Path(checkpoint_dir) / f"{strategy}_seed{seed}.pt"
            torch.save(result["model"], path)
            print(f"  Saved checkpoint: {path}")

        print(
            f"  Seed {seed}: src_acc={src_acc:.4f}  tgt_acc={tgt_acc:.4f}  "
            f"Δshift={src_acc - tgt_acc:.4f}"
        )

    summary = {
        "strategy":         strategy,
        "src_mean":         float(np.mean(src_accs)),
        "src_std":          float(np.std(src_accs)),
        "tgt_mean":         float(np.mean(tgt_accs)),
        "tgt_std":          float(np.std(tgt_accs)),
        "delta_shift_mean": float(np.mean(np.array(src_accs) - np.array(tgt_accs))),
        "per_seed": {"src": src_accs, "tgt": tgt_accs},
    }
    print(
        f"\n[{strategy}] src={summary['src_mean']:.4f}±{summary['src_std']:.4f}  "
        f"tgt={summary['tgt_mean']:.4f}±{summary['tgt_std']:.4f}  "
        f"Δshift={summary['delta_shift_mean']:.4f}"
    )
    return summary
