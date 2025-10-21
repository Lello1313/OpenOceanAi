# scripts/train_from_module.py
import os
import torch
from shipsear_core import (
    set_seed, device_select, ensure_dir,
    make_splits, build_loaders,
    MultiResLogMel, MelCache, MiniResNet,
    SupConLoss, train_epoch, evaluate_epoch
)
import pandas as pd
from torch.optim import AdamW

# --- CONFIGURAZIONE BASE ---
data_dir = "."
meta_csv = "meta/ships_segments.csv"
out_dir = "outputs/v3_from_module"
cache_dir = "cache_mels_v3_from_module"
epochs = 10
batch_size = 64
lr = 3e-4

# --- PREPARAZIONE ---
set_seed(42)
device = device_select("cuda")
ensure_dir(out_dir)

meta = pd.read_csv(meta_csv)
splits = make_splits(meta, n_unseen=3)
class_names = sorted(meta["category"].unique())

# --- FEATURE EXTRACTOR ---
featurizer = MultiResLogMel(
    target_sr=22050,
    n_mels=128,
    configs=((1024, 256), (2048, 512)),
    target_secs=5.0,
    use_deltas=True,
    use_attention=True
)
cache = MelCache(cache_dir=cache_dir)

# --- DATA LOADERS ---
args = type("Args", (), {
    "data_dir": data_dir,
    "wav_dir": "shipsear_segments",
    "alt_wav_dir": "shipsear_raw",
    "batch_size": batch_size,
    "workers": 0,
    "steps_per_epoch": -1
})()
dl_train, dl_calib, dl_unseen = build_loaders(args, featurizer, cache, class_names, meta, splits)

# --- MODELLO ---
model = MiniResNet(in_ch=6, emb_dim=256, proj_dim=256, num_classes=len(class_names)).to(device)

optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
supcon = SupConLoss(temperature=0.07)

best_auroc = -1.0

# --- TRAINING LOOP ---
for ep in range(1, epochs + 1):
    print(f"\nEpoch {ep}/{epochs}")
    stats = train_epoch(model, dl_train, optimizer, None, args, device, queue=None, supcon=supcon, temp_sched=None)
    metrics = evaluate_epoch(model, dl_train, dl_calib, dl_unseen, device, args)
    print(f"Results -> AUROC={metrics['auroc']:.3f}, AUPRC={metrics['auprc']:.3f}, Silhouette={metrics['silhouette']:.3f}")

    if not torch.isnan(torch.tensor(metrics['auroc'])) and metrics['auroc'] > best_auroc:
        best_auroc = metrics['auroc']
        ckpt = {
            "model": model.state_dict(),
            "class_names": class_names,
            "args": {"sr": 22050, "n_mels": 128, "in_ch": 6, "emb_dim": 256, "proj_dim": 256},
            "epoch": ep,
            "best_auroc": best_auroc,
        }
        torch.save(ckpt, os.path.join(out_dir, "best_from_module.pt"))
        print(f"✅ Saved new best checkpoint with AUROC={best_auroc:.3f}")
