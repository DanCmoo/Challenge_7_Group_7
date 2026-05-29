"""
Part B — Neural Style Transfer (Gatys et al., 2016).

Content: Art domain images (source).
Style:   Clipart domain images (target).
Output:  Synthetic images saved to data/synthetic_target/.

Suggested layers (from challenge spec):
  Content: relu4_2  → VGG-19 index '22'
  Style:   relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
"""

import os
import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from src.utils import (
    CLASSES, DATA_ROOT, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, get_device, set_seed
)

# ─── VGG-19 layer name → index mapping ──────────────────────────────────────
# These are indices into vgg.features children (0-based).
# VGG-19 structure: conv blocks separated by MaxPool.
# relu1_1=1, relu2_1=6, relu3_1=11, relu4_1=20, relu4_2=22, relu5_1=29
CONTENT_LAYERS: List[str] = ["22"]   # relu4_2
STYLE_LAYERS:   List[str] = ["1", "6", "11", "20", "29"]  # one per block


# ─── Gram matrix ─────────────────────────────────────────────────────────────

def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute normalised Gram matrix G_l.

    G_l_{ij} = (1 / C*H*W) * sum_k F_{ik} * F_{jk}
    """
    b, c, h, w = feat.size()
    feat = feat.view(b, c, h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)


# ─── Feature extractor ───────────────────────────────────────────────────────

class VGGFeatureExtractor(nn.Module):
    """Extract intermediate feature maps from frozen VGG-19."""

    def __init__(self, content_layers: List[str], style_layers: List[str]):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        for param in self.features.parameters():
            param.requires_grad = False

        self.content_layers = set(content_layers)
        self.style_layers   = set(style_layers)
        self.all_layers     = self.content_layers | self.style_layers

        # Only keep layers up to the last required one
        max_layer = max(int(l) for l in self.all_layers)
        self.features = nn.Sequential(*list(self.features.children())[: max_layer + 1])

    def forward(self, x: torch.Tensor) -> Tuple[Dict, Dict]:
        content_feats: Dict[str, torch.Tensor] = {}
        style_feats:   Dict[str, torch.Tensor] = {}

        for name, layer in self.features.named_children():
            x = layer(x)
            if name in self.content_layers:
                content_feats[name] = x
            if name in self.style_layers:
                style_feats[name] = x

        return content_feats, style_feats


# ─── Loss computation ────────────────────────────────────────────────────────

def content_loss(
    gen_feats: Dict[str, torch.Tensor],
    content_feats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """L_content = 0.5 * sum_l ||F_l - P_l||^2_F"""
    loss = torch.tensor(0.0, device=next(iter(gen_feats.values())).device)
    for layer in gen_feats:
        loss = loss + 0.5 * F.mse_loss(gen_feats[layer], content_feats[layer].detach())
    return loss


def style_loss(
    gen_feats: Dict[str, torch.Tensor],
    style_grams: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """L_style = sum_l ||G_l - G^hat_l||^2_F"""
    loss = torch.tensor(0.0, device=next(iter(gen_feats.values())).device)
    for layer in gen_feats:
        loss = loss + F.mse_loss(gram_matrix(gen_feats[layer]), style_grams[layer].detach())
    return loss


# ─── Image I/O helpers ───────────────────────────────────────────────────────

def load_image(
    path: str,
    size: int = IMAGE_SIZE,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Load and preprocess a single image to a 4-D tensor (1, C, H, W)."""
    transform = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    if device:
        tensor = tensor.to(device)
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised (1, C, H, W) tensor back to a PIL Image."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(tensor.device)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(tensor.device)
    img  = tensor.squeeze(0) * std + mean
    img  = img.clamp(0, 1).cpu()
    return T.ToPILImage()(img)


# ─── Core style transfer ─────────────────────────────────────────────────────

def style_transfer(
    content_path: str,
    style_path:   str,
    device:       torch.device,
    alpha:        float = 1.0,
    beta:         float = 1e4,
    n_steps:      int   = 300,
    image_size:   int   = IMAGE_SIZE,
    seed:         int   = 42,
) -> Image.Image:
    """
    Perform Gatys-style Neural Style Transfer.

    Args:
        content_path: Path to the content image (Art domain).
        style_path:   Path to the style image (Clipart domain).
        alpha:        Weight for content loss.
        beta:         Weight for style loss.
        n_steps:      Number of L-BFGS optimisation steps.

    Returns:
        Generated PIL image.
    """
    set_seed(seed)

    extractor = VGGFeatureExtractor(CONTENT_LAYERS, STYLE_LAYERS).to(device)

    content_img = load_image(content_path, size=image_size, device=device)
    style_img   = load_image(style_path,   size=image_size, device=device)

    # Pre-compute reference feature maps
    with torch.no_grad():
        content_feats_ref, _              = extractor(content_img)
        _,                 style_feats_ref = extractor(style_img)
        style_grams_ref = {l: gram_matrix(f) for l, f in style_feats_ref.items()}

    # Generated image starts as a copy of the content image
    generated = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([generated], lr=1.0, max_iter=20)

    step = [0]

    def closure():
        with torch.no_grad():
            generated.clamp_(
                min=-2.5, max=2.5
            )  # prevent extreme pixel values
        optimizer.zero_grad()
        gen_content, gen_style = extractor(generated)
        c_loss = content_loss(gen_content, content_feats_ref)
        s_loss = style_loss(gen_style, style_grams_ref)
        total  = alpha * c_loss + beta * s_loss
        total.backward()
        step[0] += 1
        return total

    for _ in range(n_steps // 20):  # LBFGS runs max_iter=20 per .step()
        optimizer.step(closure)

    return tensor_to_pil(generated.detach())


# ─── Batch generation ────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    output_dir:  str,
    n_per_class: int   = 30,
    alpha:       float = 1.0,
    beta:        float = 1e4,
    n_steps:     int   = 300,
    image_size:  int   = IMAGE_SIZE,
) -> None:
    """
    Generate n_per_class style-transferred images for each of the 6 classes.

    Content images come from the Art domain; style images from Clipart.
    Results are saved to: output_dir/<class_name>/<idx>.jpg

    At least 30 per class (180 total) are required by the challenge.
    """
    device = get_device()
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for cls in CLASSES:
        art_dir     = os.path.join(DATA_ROOT, "Art",     cls)
        clipart_dir = os.path.join(DATA_ROOT, "Clipart", cls)
        out_cls_dir = os.path.join(output_dir, cls)
        os.makedirs(out_cls_dir, exist_ok=True)

        art_images     = sorted([
            os.path.join(art_dir, f) for f in os.listdir(art_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        clipart_images = sorted([
            os.path.join(clipart_dir, f) for f in os.listdir(clipart_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        # Pick one fixed style image per class (first Clipart image)
        style_path = clipart_images[0]

        pbar = tqdm(range(n_per_class), desc=f"NST [{cls}]", leave=False)
        for i in pbar:
            content_path = art_images[i % len(art_images)]
            out_path     = os.path.join(out_cls_dir, f"{i:04d}.jpg")

            if os.path.exists(out_path):
                continue  # skip already generated

            try:
                img = style_transfer(
                    content_path, style_path,
                    device=device,
                    alpha=alpha, beta=beta,
                    n_steps=n_steps,
                    image_size=image_size,
                    seed=i,
                )
                img.save(out_path)
            except Exception as e:
                print(f"  Warning: failed for {content_path}: {e}")

        print(f"  [{cls}] generated {n_per_class} images → {out_cls_dir}")
