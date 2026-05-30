"""Evaluate all Part A checkpoints and print results."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import csv

from src.utils import get_device, build_domain_splits, make_loaders, SEEDS, evaluate
from src.classifier import build_resnet_scratch, build_resnet_frozen, build_resnet_finetuned

device = get_device()

art_train, art_val, art_test    = build_domain_splits('Art',     seed=42)
clip_train, clip_val, clip_test = build_domain_splits('Clipart', seed=42)
_, _, src_test = make_loaders(art_train,  art_val,  art_test,  batch_size=32, num_workers=0)
_, _, tgt_test = make_loaders(clip_train, clip_val, clip_test, batch_size=32, num_workers=0)

builders = {
    'scratch':   build_resnet_scratch,
    'frozen':    build_resnet_frozen,
    'finetuned': build_resnet_finetuned,
}

ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
results = {}

for strategy, builder in builders.items():
    src_accs, tgt_accs = [], []
    for seed in SEEDS:
        path = os.path.join(ckpt_dir, f'{strategy}_seed{seed}.pt')
        model = builder()
        model.load_state_dict(torch.load(path, map_location=device))
        model = model.to(device)
        src_acc, _ = evaluate(model, src_test, device)
        tgt_acc, _ = evaluate(model, tgt_test, device)
        src_accs.append(src_acc)
        tgt_accs.append(tgt_acc)
        print(f"  {strategy} seed={seed}: src={src_acc:.4f}  tgt={tgt_acc:.4f}")
    results[strategy] = {
        'src_mean': float(np.mean(src_accs)), 'src_std': float(np.std(src_accs)),
        'tgt_mean': float(np.mean(tgt_accs)), 'tgt_std': float(np.std(tgt_accs)),
    }

# Part C results
partc = {
    'target_finetuning':  {'src_acc': 0.7857, 'tgt_acc': 0.8831},
    'style_augmentation': {'src_acc': 0.9286, 'tgt_acc': 0.7143},
}

print("\n=== PART A ===")
for s, r in results.items():
    delta = r['src_mean'] - r['tgt_mean']
    print(f"{s:12s}  src={r['src_mean']:.4f}±{r['src_std']:.4f}  tgt={r['tgt_mean']:.4f}±{r['tgt_std']:.4f}  Δ={delta:.4f}")

print("\n=== PART C ===")
for s, r in partc.items():
    print(f"{s:22s}  tgt={r['tgt_acc']:.4f}  Δ={r['src_acc']-r['tgt_acc']:.4f}")

# Write updated CSV
rows = []
for s, r in results.items():
    rows.append({
        'Strategy': s,
        'Source Acc (mean)': f"{r['src_mean']:.4f} ± {r['src_std']:.4f}",
        'Target Acc (mean)': f"{r['tgt_mean']:.4f} ± {r['tgt_std']:.4f}",
        'Δ_shift': f"{r['src_mean'] - r['tgt_mean']:.4f}",
    })
for s, r in partc.items():
    rows.append({
        'Strategy': s,
        'Source Acc (mean)': f"{r['src_acc']:.4f} ± 0.0000",
        'Target Acc (mean)': f"{r['tgt_acc']:.4f} ± 0.0000",
        'Δ_shift': f"{r['src_acc'] - r['tgt_acc']:.4f}",
    })

csv_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'summary_table.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['Strategy', 'Source Acc (mean)', 'Target Acc (mean)', 'Δ_shift'])
    writer.writeheader()
    writer.writerows(rows)
print(f"\nUpdated → {csv_path}")
