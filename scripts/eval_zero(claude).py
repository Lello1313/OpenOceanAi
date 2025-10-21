# scripts/eval_zero.py
"""Zero-shot open-set evaluation script"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import numpy as np

from shipsear_core import (
    parse_args, set_seed, device_select, normpath,
    MultiResLogMel, MelCache, ShipsearDataset, MiniResNet,
    eval_openset_ensemble, compute_silhouette, eval_individual_scores
)
from torch.utils.data import DataLoader


def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def extract_embeddings(model, loader, device):
    """Extract embeddings, logits, and labels"""
    model.eval()
    E, L, Y = [], [], []
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            e, p, logits = model(xb, return_proj=True)
            E.append(e.cpu().numpy())
            L.append(logits.cpu().numpy())
            Y.append(yb.cpu().numpy())
    
    E = np.concatenate(E, 0) if len(E) > 0 else np.zeros((0, model.norm.normalized_shape[0]))
    L = np.concatenate(L, 0) if len(L) > 0 else np.zeros((0, model.cls.out_features))
    Y = np.concatenate(Y, 0) if len(Y) > 0 else np.zeros((0,), dtype=np.int64)
    
    return E, L, Y


def main():
    args = parse_args(mode='eval_zero')
    set_seed(args.seed)
    device = device_select(args.device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    
    class_names = ckpt['class_names']
    n_classes = len(class_names)
    saved_args = ckpt['args']
    
    # Reconstruct featurizer
    fe_cfg = {
        "sr": saved_args['sr'],
        "n_mels": saved_args['n_mels'],
        "configs": [(saved_args['nfft1'], saved_args['hop1']), (saved_args['nfft2'], saved_args['hop2'])],
        "target_secs": saved_args['target_secs'],
        "use_deltas": (saved_args['in_ch'] == 6),
        "use_attention": True
    }
    
    featurizer = MultiResLogMel(
        target_sr=saved_args['sr'],
        n_mels=saved_args['n_mels'],
        configs=((saved_args['nfft1'], saved_args['hop1']), (saved_args['nfft2'], saved_args['hop2'])),
        target_secs=saved_args['target_secs'],
        use_deltas=(saved_args['in_ch'] == 6),
        use_attention=True
    )
    
    cache = MelCache(cache_dir=args.cache_dir, fe_cfg=fe_cfg)
    
    # Load test data
    print(f"Loading test data: {args.test_csv}")
    test_meta = pd.read_csv(args.test_csv)
    
    def to_records(df):
        return df[["filename", "category"]].to_dict(orient="records")
    
    ds_test = ShipsearDataset(
        to_records(test_meta),
        args.data_dir,
        class_names,
        featurizer,
        cache,
        mode="eval",
        aug_spec=False,
        aug_wav=False,
        wav_dir=saved_args.get('wav_dir', 'shipsear_segments'),
        alt_wav_dir=saved_args.get('alt_wav_dir', 'shipsear_raw')
    )
    
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=True
    )
    
    # Load model
    model = MiniResNet(
        in_ch=saved_args['in_ch'],
        emb_dim=saved_args['emb_dim'],
        proj_dim=saved_args['proj_dim'],
        num_classes=n_classes,
        num_heads=saved_args.get('num_heads', 3)
    ).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # Extract embeddings
    print("Extracting embeddings...")
    E_test, L_test, y_test = extract_embeddings(model, dl_test, device)
    
    # Compute silhouette
    sil = compute_silhouette(E_test, y_test)
    print(f"Silhouette Score: {sil:.3f}")
    
    # Note: For true zero-shot, you would need separate seen/unseen data
    # This is a simplified version
    print("\nZero-shot evaluation complete!")
    print(f"Extracted {E_test.shape[0]} embeddings")
    print(f"Embedding dimension: {E_test.shape[1]}")


if __name__ == "__main__":
    main()