# Challenge 7 — Group 7

**Transfer Learning: Few-Shot Classification, Neural Style Transfer, and Domain Shift Adaptation**

**Dataset:** Office-Home | **Source:** Art → **Target:** Clipart
**Classes:** Calculator, Keyboard, Laptop, Monitor, Mouse, Printer

---

## Quickstart

### Requirements
- Python 3.10 or 3.12
- [Poetry](https://python-poetry.org/docs/#installation) 2.x

### Install (Mac / Windows CPU)

```bash
# 1. Install all dependencies
make install

# 2. Register the Jupyter kernel
make kernel

# 3. Launch JupyterLab
make notebook
```

### Install with CUDA (Windows/Linux GPU users only)

```bash
make install           # installs CPU build first
make install-cuda      # overrides torch/torchvision with CUDA 12.1 wheels
make kernel
make notebook
```

### Verify environment

```bash
make env-info
```

---

## Repository structure

```
.
├── pyproject.toml              # Poetry dependency spec
├── Makefile                    # Reproducible commands
├── CHECKLIST.md                # Required submission checklist
├── src/
│   ├── utils.py                # Device, seeds, transforms, data loaders
│   ├── classifier.py           # Part A — few-shot classifier
│   ├── style_transfer.py       # Part B — Neural Style Transfer (VGG-19)
│   └── domain_adaptation.py    # Part C — adaptation strategies + DANN
├── notebooks/
│   ├── part_a_classification.ipynb
│   ├── part_b_style_transfer.ipynb
│   └── part_c_domain_adaptation.ipynb
├── data/
│   └── synthetic_target/       # Generated NST images (Part B output)
├── checkpoints/                # Saved .pt model weights
├── figures/                    # Required figures and summary CSV
└── OfficeHomeDataset_10072016/ # Dataset (Art and Clipart domains used)
```

---

## Reproducibility

All experiments fix three random seeds: `[42, 123, 2024]`.

```python
from src.utils import set_seed, SEEDS
set_seed(SEEDS[0])   # 42
```

To reproduce all results from scratch:

```bash
make all    # runs part-a → part-b → part-c sequentially
```

**Note on NST runtime:** generating 180 style-transferred images takes 3–9 h on CPU.
Running Part B on MPS (Mac Apple Silicon) or CUDA is strongly recommended.

---

## References

- Gatys, Ecker & Bethge (2016) — Neural Style Transfer
- He et al. (2016) — ResNet
- Venkateswara et al. (2017) — Office-Home dataset
- Ganin et al. (2016) — DANN
- Tan et al. (2018); Zhuang et al. (2021) — Transfer learning surveys
