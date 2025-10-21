# scripts/train_closedset.py
# Closed-set classification 80/10/10 con class weights + barra di avanzamento e metriche per epoca
# Include stampa conteggi per split, classification report e confusion matrix (numeric + heatmap)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # importa shipsear_core

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from shipsear_core import (
    MultiResLogMel, MelCache, ShipsearDataset, MiniResNet,
    set_seed, device_select, ensure_dir, normpath
)

# tqdm per progress bar (se non presente: pip install tqdm)
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# =========================
# CONFIG
# =========================
DATA_DIR   = "."
META_CSV   = "meta/ships_segments.csv"
OUT_DIR    = "outputs/closedset_v1"
CACHE_DIR  = "cache_mels_closedset_v1noleakprv"
WAV_DIR    = "shipsear_segments"
ALT_WAVDIR = "shipsear_raw"

SR=22050; N_MELS=128; NFFT1=1024; HOP1=256; NFFT2=2048; HOP2=512; TARGET_SECS=5.0
IN_CH=6; EMB_DIM=256; PROJ_DIM=256; NUM_HEADS=3

EPOCHS=40; BATCH_SIZE=64; LR=3e-4; WD=1e-4; WORKERS=0; SEED=42; DEVICE_PREF="cuda"
GRAD_CLIP = 5.0
EPS = 1e-6

# =========================
# UTILS
# =========================
def stratified_split_80_10_10(df, label_col="category", seed=42):
    """Crea split 80/10/10 stratificato per classe."""
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby(label_col, sort=False):
        idx = np.arange(len(g))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(0.7*n))
        n_va = int(round(0.1*n))
        tr = g.iloc[idx[:n_tr]].assign(split="train")
        va = g.iloc[idx[n_tr:n_tr+n_va]].assign(split="val")
        te = g.iloc[idx[n_tr+n_va:]].assign(split="test")
        parts += [tr, va, te]
    return pd.concat(parts, ignore_index=True)

def to_records(df):
    return df[["filename","category"]].to_dict(orient="records")

def make_loader(recs, mode, featurizer, cache, class_names):
    ds = ShipsearDataset(
        recs, DATA_DIR, class_names, featurizer, cache,
        mode=("train" if mode=="train" else "eval"),
        aug_spec=(mode=="train"), aug_wav=False,
        wav_dir=WAV_DIR, alt_wav_dir=ALT_WAVDIR
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=(mode=="train"),
        num_workers=WORKERS,
        pin_memory=True,
        collate_fn=lambda batch: (
            torch.stack([x for x,_ in batch], 0),
            torch.tensor([y for _,y in batch], dtype=torch.long),
        ),
    )

@torch.no_grad()
def eval_accuracy(model, loader, device, desc=None):
    model.eval()
    correct = total = 0
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc=(desc or "eval"), leave=False)
    for xb, yb in iterator:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        _, _, logits = model(xb, return_proj=True)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        total   += yb.numel()
    return correct / max(1, total)

@torch.no_grad()
def eval_macro_f1(model, loader, device):
    try:
        from sklearn.metrics import f1_score
    except Exception:
        return None
    model.eval()
    preds, trues = [], []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="eval F1 (macro)", leave=False)
    for xb, yb in iterator:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        _, _, logits = model(xb, return_proj=True)
        preds += logits.argmax(1).cpu().tolist()
        trues += yb.cpu().tolist()
    return f1_score(trues, preds, average="macro")

def print_split_counts(meta_s, class_names):
    """Stampa tabella per split con conteggio clip per classe."""
    for split in ["train", "val", "test"]:
        sub = meta_s[meta_s.split == split]
        counts = sub["category"].value_counts().reindex(class_names, fill_value=0)
        print(f"\n=== {split.upper()} counts ===")
        print(counts.to_string())
        print(f"TOTAL {split}: {int(counts.sum())}")
    # riepilogo unico
    pivot = pd.crosstab(meta_s["category"], meta_s["split"]).reindex(class_names).fillna(0).astype(int)
    print("\n=== SUMMARY (rows=class, cols=split) ===")
    print(pivot.to_string())
    print(f"\nTOTAL clips: {int(pivot.values.sum())}")

# =========================
# MAIN
# =========================
def main():
    set_seed(SEED)
    device = device_select(DEVICE_PREF)
    ensure_dir(OUT_DIR); ensure_dir(CACHE_DIR)

    # Carica meta e classi
    meta = pd.read_csv("meta/ships_segments.csv")
    class_names = sorted(meta["category"].unique())
    meta_s = meta.copy()


    # Stampa conteggi per split e classe
    print_split_counts(meta_s, class_names)

    tr = meta_s[meta_s.split=="train"]
    va = meta_s[meta_s.split=="val"]
    te = meta_s[meta_s.split=="test"]

    # Featurizer + cache
    featurizer = MultiResLogMel(
        target_sr=SR, n_mels=N_MELS,
        configs=((NFFT1,HOP1),(NFFT2,HOP2)),
        target_secs=TARGET_SECS,
        use_deltas=(IN_CH==6),
        use_attention=True
    )
    cache = MelCache(cache_dir=CACHE_DIR, fe_cfg={
        "sr":SR, "n_mels":N_MELS, "configs":[(NFFT1,HOP1),(NFFT2,HOP2)],
        "target_secs":TARGET_SECS, "use_deltas":(IN_CH==6), "use_attention":True
    })

    # DataLoader
    dl_tr = make_loader(to_records(tr), "train", featurizer, cache, class_names)
    dl_va = make_loader(to_records(va), "val",   featurizer, cache, class_names)
    dl_te = make_loader(to_records(te), "test",  featurizer, cache, class_names)

    # Modello
    model = MiniResNet(
        in_ch=IN_CH, emb_dim=EMB_DIM, proj_dim=PROJ_DIM,
        num_classes=len(class_names), num_heads=NUM_HEADS
    ).to(device)

    # ====== Class Weights (opzione A) ======
    # pesi calcolati sulla distribuzione del TRAIN
    train_counts = tr["category"].value_counts().reindex(class_names, fill_value=0).values.astype(np.float32)
    class_weights = 1.0 / (train_counts + EPS)
    # normalizza per avere media ~1 (stabile)
    class_weights = class_weights * (len(class_names) / class_weights.sum())
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Ottimizzatore + CE pesata
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss(weight=weights_tensor)

    best_val = -1.0
    for ep in range(1, EPOCHS+1):
        model.train()
        running = 0.0; n = 0

        iterator = dl_tr
        if tqdm is not None:
            iterator = tqdm(dl_tr, total=len(dl_tr), desc=f"Epoch {ep}/{EPOCHS} [train]", leave=False)

        for xb, yb in iterator:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()
            _, _, logits = model(xb, return_proj=True)
            loss = ce(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            running += float(loss.detach().cpu()); n += 1
            if tqdm is not None:
                iterator.set_postfix({"loss_avg": f"{running/max(1,n):.3f}"})

        train_ce = running / max(1, n)

        # Val/Test metrics
        val_acc  = eval_accuracy(model, dl_va, device, desc="val acc")
        test_acc = eval_accuracy(model, dl_te, device, desc="test acc")
        val_f1   = eval_macro_f1(model, dl_va, device)
        test_f1  = eval_macro_f1(model, dl_te, device)

        if val_f1 is None or test_f1 is None:
            print(f"[Epoch {ep:02d}] train_ce={train_ce:.3f}  val_acc={val_acc:.3f}  test_acc={test_acc:.3f}")
        else:
            print(f"[Epoch {ep:02d}] train_ce={train_ce:.3f}  val_acc={val_acc:.3f}  val_f1={val_f1:.3f}  "
                  f"test_acc={test_acc:.3f}  test_f1={test_f1:.3f}")

        # checkpoint sul best val_acc
        if val_acc > best_val:
            best_val = val_acc
            ckpt = {
                "model": model.state_dict(),
                "class_names": class_names,
                "args": {
                    "sr":SR,"n_mels":N_MELS,"nfft1":NFFT1,"hop1":HOP1,"nfft2":NFFT2,"hop2":HOP2,
                    "target_secs":TARGET_SECS,"in_ch":IN_CH,"emb_dim":EMB_DIM,"proj_dim":PROJ_DIM,
                    "num_heads":NUM_HEADS, "wav_dir":WAV_DIR, "alt_wav_dir":ALT_WAVDIR
                },
                "epoch": ep, "val_acc": best_val,
                "class_weights": class_weights.tolist(),
            }
            path = normpath(OUT_DIR, "closedset_bestnoleakprv.pt")
            torch.save(ckpt, path)
            print(f"✅ Saved BEST checkpoint to {path} (val_acc={best_val:.3f})")

    # Salva anche l’ultimo
    path_last = normpath(OUT_DIR, "closedset_lastnoleakprv.pt")
    torch.save({"model": model.state_dict(), "class_names": class_names}, path_last)
    print(f"💾 Saved LAST checkpoint to {path_last}")

    # === FINAL EVALUATION (TEST): classification report + confusion matrix ===
    @torch.no_grad()
    def collect_preds(model, loader, device):
        model.eval()
        preds, trues = [], []
        for xb, yb in loader:
            xb = xb.to(device); yb = yb.to(device)
            _, _, logits = model(xb, return_proj=True)
            preds += logits.argmax(1).cpu().tolist()
            trues += yb.cpu().tolist()
        return trues, preds

    y_true, y_pred = collect_preds(model, dl_te, device)

    # --- Classification report
    try:
        from sklearn.metrics import classification_report
        print("\n=== CLASSIFICATION REPORT (TEST) ===")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    except Exception as e:
        print(f"(classification_report non disponibile: {e})")

    # --- Confusion matrix: numerica + heatmap salvata
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

        print("\n=== CONFUSION MATRIX (numeric, rows=True, cols=Pred) ===")
        print(cm_df.to_string())

        # Heatmap (se seaborn/matplotlib disponibili)
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(max(8, len(class_names)*0.8), max(6, len(class_names)*0.6)))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.title("Confusion Matrix (Test Set)")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.tight_layout()
            out_png = os.path.join(OUT_DIR, "confusion_matrix_testnoleak.png")
            plt.savefig(out_png, dpi=200)
            plt.close()
            print(f"💾 Confusion matrix plot saved to {out_png}")
        except Exception as e:
            print(f"(Impossibile generare la heatmap: {e})")
    except Exception as e:
        print(f"(Impossibile calcolare la confusion matrix: {e})")

if __name__ == "__main__":
    main()
