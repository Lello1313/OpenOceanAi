# scripts/eval_zero.py
"""Zero-shot open-set evaluation script (compat con checkpoint che includono 'quantum.*')"""

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

    # fallback safe shapes if empty
    emb_dim = getattr(getattr(model, "norm", None), "normalized_shape", [0])[0] if hasattr(model, "norm") else 0
    out_dim = getattr(getattr(model, "cls", None), "out_features", 0) if hasattr(model, "cls") else 0

    E = np.concatenate(E, 0) if len(E) > 0 else np.zeros((0, emb_dim))
    L = np.concatenate(L, 0) if len(L) > 0 else np.zeros((0, out_dim))
    Y = np.concatenate(Y, 0) if len(Y) > 0 else np.zeros((0,), dtype=np.int64)

    return E, L, Y


def safe_torch_load(path, map_location):
    """
    Carica il checkpoint in modo compatibile con le versioni attuali e future di torch.load,
    evitando warning/flip del default su weights_only.
    """
    try:
        # torch >= 2.4 accetta weights_only (default False). Lo specifichiamo esplicitamente.
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 2.4 non ha il parametro, ricarichiamo senza
        return torch.load(path, map_location=map_location)


def main():
    args = parse_args(mode='eval_zero')
    set_seed(args.seed)
    device = device_select(args.device)

    # Load checkpoint
    print(f"[eval_zero] Loading checkpoint: {args.checkpoint}")
    ckpt = safe_torch_load(args.checkpoint, map_location=device)

    # Recupero classi / argomenti usati nel training
    class_names = ckpt.get('class_names', None)
    if class_names is None:
        raise ValueError("Checkpoint privo di 'class_names'.")

    n_classes = len(class_names)
    saved_args = ckpt.get('args', None)
    if saved_args is None:
        raise ValueError("Checkpoint privo di 'args' (iperparametri del training).")

    # Reconstruct featurizer
    featurizer = MultiResLogMel(
        target_sr=saved_args['sr'],
        n_mels=saved_args['n_mels'],
        configs=((saved_args['nfft1'], saved_args['hop1']), (saved_args['nfft2'], saved_args['hop2'])),
        target_secs=saved_args['target_secs'],
        use_deltas=(saved_args['in_ch'] == 6),
        use_attention=True
    )

    fe_cfg = {
        "sr": saved_args['sr'],
        "n_mels": saved_args['n_mels'],
        "configs": [(saved_args['nfft1'], saved_args['hop1']), (saved_args['nfft2'], saved_args['hop2'])],
        "target_secs": saved_args['target_secs'],
        "use_deltas": (saved_args['in_ch'] == 6),
        "use_attention": True
    }
    cache = MelCache(cache_dir=args.cache_dir, fe_cfg=fe_cfg)

    # Load test data
    print(f"[eval_zero] Loading test CSV: {args.test_csv}")
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

    # Build model (stessa architettura di eval, senza blocchi 'quantum')
    model = MiniResNet(
        in_ch=saved_args['in_ch'],
        emb_dim=saved_args['emb_dim'],
        proj_dim=saved_args['proj_dim'],
        num_classes=n_classes,
        num_heads=saved_args.get('num_heads', 3)
    ).to(device)

    # --- CARICAMENTO ROBUSTO DELLO STATO ---
    raw_state = ckpt.get("model", None)
    if raw_state is None:
        raise ValueError("Checkpoint privo di 'model' (state_dict).")

    # Filtra qualsiasi chiave che provenga da moduli non presenti (es. 'quantum.*')
    filtered_state = {k: v for k, v in raw_state.items() if not k.startswith("quantum.")}

    # Carica con strict=False per ignorare eventuali residue mismatch non critici
    missing_unexpected = model.load_state_dict(filtered_state, strict=False)
    print("[eval_zero] load_state_dict report:", missing_unexpected)
    model.eval()

    # Extract embeddings
    print("[eval_zero] Extracting embeddings...")
    E_test, L_test, y_test = extract_embeddings(model, dl_test, device)

    # Compute silhouette
    sil = compute_silhouette(E_test, y_test)
    print(f"[eval_zero] Silhouette Score: {sil:.3f}")

    # --- Salvataggi per QSCO / analisi successive ---
    import os, torch
    os.makedirs(args.out_dir, exist_ok=True)

    E_t = torch.tensor(E_test)     # (N, D)
    L_t = torch.tensor(L_test)     # (N, C)
    y_t = torch.tensor(y_test)     # (N,)

    torch.save(E_t, os.path.join(args.out_dir, "E_test.pt"))
    torch.save(L_t, os.path.join(args.out_dir, "L_test.pt"))
    torch.save(y_t, os.path.join(args.out_dir, "y_test.pt"))

# alias "calib" se vuoi riusare subito come set di calibrazione QSCO
    torch.save(E_t, os.path.join(args.out_dir, "E_calib.pt"))
    torch.save(y_t, os.path.join(args.out_dir, "y_calib.pt"))


    print("\nZero-shot evaluation complete!")
    print(f"Extracted {E_test.shape[0]} embeddings")
    print(f"Embedding dimension: {E_test.shape[1]}")


if __name__ == "__main__":
    main()
