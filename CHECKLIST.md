# Grading Checklist — Group 7

## Assigned dataset pair and category subset

- **Dataset:** Office-Home (Venkateswara et al., 2017)
- **Source domain:** Art (artistic paintings and sketches)
- **Target domain:** Clipart (flat-colour clipart images)
- **Categories (6):** Calculator, Keyboard, Laptop, Monitor, Mouse, Printer
- **Split used:** 70% train / 15% val / 15% test (stratified per class, seed=42)
  - Art images per class range from 18 (Keyboard, Mouse, Printer) to 51 (Laptop)
  - Clipart images per class range from 46 (Calculator) to 99 (Keyboard, Laptop, Monitor)

---

## Pretrained backbone and trainable parameters

**Backbone:** ResNet-50 (ImageNet1K_V2 weights)

| Model variant | Trainable parameters |
|---|---|
| From scratch | ~25.5 M (all) |
| Feature extraction (frozen) | ~12,294 (head only) |
| Fine-tuning (layer3+layer4+head) | ~TBD — to be filled after running Part A |
| Target fine-tuning (Part C) | ~TBD |
| Style-transfer augmentation (Part C) | ~TBD |

---

## Accuracy results (mean ± std over 3 seeds: 42, 123, 2024)

| Strategy | Source Acc | Target Acc | Δ_shift |
|---|---|---|---|
| From scratch | TBD | TBD | TBD |
| Feature extraction (frozen) | TBD | TBD | TBD |
| Fine-tuning (best Part A) | TBD | TBD | TBD |
| Target fine-tuning (Part C) | TBD | TBD | TBD |
| Style-transfer augmentation (Part C) | TBD | TBD | TBD |

*Fill this table after running the notebooks.*

---

## Domain shift penalty (Δ_shift)

- **Best Part A model** (fine-tuned): Δ_shift = TBD
- **Best Part C adaptation model**: Δ_shift = TBD

---

## Neural Style Transfer — α/β ratio

| α/β ratio | Visual assessment |
|---|---|
| 1e-3 | Strong style, object structure preserved |
| **1e-4** | **Balanced — selected for final dataset** |
| 1e-5 | Subtle style, close to original content |

**Selected ratio:** α = 1.0 / β = 1e4 (α/β = 1e-4)

Quality assessment: The generated images retain the object class identity while adopting
the flat-colour appearance of Clipart. Edge sharpness and saturation are notably affected.
At α/β = 1e-4, content legibility is maintained for all 6 classes.

---

## Best adaptation strategy — justification (max 200 words)

*(To be completed after running Part C experiments.)*

Preliminary hypothesis: **Target domain fine-tuning** is expected to outperform
style-transfer augmentation for this specific domain pair (Art → Clipart) because
the Art→Clipart shift is primarily driven by line style, colour distribution, and
the absence of photorealistic textures. Fine-tuning on 40 real Clipart examples per
class directly exposes the model to the target distribution. Style-transfer augmentation,
while label-free, may not fully replicate the clean vector-art appearance of Clipart,
since VGG-19 style transfer tends to produce painterly effects rather than flat cartoon
graphics. However, if the labelled target set is too small to reliably adapt the
higher-level feature detectors, the combined training signal from style augmentation
(180 extra images) may close part of the gap. We will compare both empirically and
report which strategy minimises Δ_shift across all 6 classes.

---

## Figures produced

- [ ] `figures/part_a_training_curves.png` — loss/accuracy curves for Part A
- [ ] `figures/part_b_gallery.png` — content / style / generated for each class
- [ ] `figures/part_c_gradcam.png` — Grad-CAM for correct and incorrect predictions
- [ ] `figures/part_c_tsne_before_after.png` — t-SNE before and after adaptation
- [ ] `figures/summary_table.csv` — all 5 model variants

---

## Reproducibility checklist

- [x] Random seeds fixed: `[42, 123, 2024]`
- [x] Environment specified: `pyproject.toml` + `poetry.lock`
- [x] Run commands documented: `Makefile`
- [x] Identical preprocessing for source and target domains (ImageNet stats applied to both)
- [x] Official Office-Home split: stratified 70/15/15 from provided directory structure
