# scripts/eval_few.py
"""
Few-shot evaluation on ShipSEAR from a single meta CSV.

Features:
- Auto-selects N unseen classes from meta CSV (strategy: 'max' or 'min' by frequency),
  with optional exclusion list.
- Builds SUPPORT (k-shot per unseen class) and QUERY sets avoiding same-recording leakage.
- Loads your trained checkpoint, extracts embeddings for support/query.
- Evaluates Nearest-Prototype with cosine and robust Mahalanobis (Ledoit–Wolf, diag, empiric).
- Diagnostics saved to --out_dir:
  * fewshot_results.json (metrics + silhouette, abstain stats if used)
  * cm_cosine.png / cm_mahal.png
  * support.csv / query.csv
  * proto_l2_matrix.csv, proto_stats.json, proto_proto_cosine.csv
  * cosine_margins.json
  * knn_support_diagnostics.json

Example (Windows one-liner):
python scripts/eval_few.py --checkpoint outputs/v3/shipsear_supcon_v3_best.pt --meta_csv meta/ships_segments.csv --out_dir outputs/fewshot --k_shot 10 --n_unseen 3 --strategy max --exclude_classes "Natural ambient noise" --norm zscore --cov_mode support_lw --cosine_abstain_margin 0.05 --seed 42 --batch_size 64
"""

import sys, os, re, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import pandas as pd
import re 
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# ---- your modular imports (as already used in your repo) ----
from shipsear_core import (
    MultiResLogMel, MelCache, ShipsearDataset, MiniResNet,
    device_select, set_seed, ensure_dir
)


# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    # Required
    p.add_argument("--checkpoint", type=str, required=True)

    # Option A: single meta CSV (auto-build few-shot)
    p.add_argument("--meta_csv", type=str, default="")

    # Option B: prebuilt CSVs
    p.add_argument("--support_csv", type=str, default="")
    p.add_argument("--query_csv", type=str, default="")

    # Few-shot design
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--n_unseen", type=int, default=3)
    p.add_argument("--strategy", type=str, choices=["max", "min"], default="max",
                   help="Pick unseen classes by frequency: 'max' => richest, 'min' => scarcest.")
    p.add_argument("--exclude_classes", type=str, default="",
                   help="Comma-separated class names to exclude from unseen selection.")

    # Preprocessing / normalization
    p.add_argument("--norm", type=str, choices=["none", "l2", "zscore", "whiten"], default="none",
                   help="Embedding normalization before prototypes/Mahalanobis.")

    # Mahalanobis covariance mode
    p.add_argument("--cov_mode", type=str,
                   choices=["lw", "diag", "emp", "support_lw", "support_diag"],
                   default="lw",
                   help="Covariance estimator: lw=LedoitWolf(query), diag=variance diagonal(query), emp=empirical(query)+reg; support_* uses SUPPORT only.")

    # Abstention on cosine margin
    p.add_argument("--cosine_abstain_margin", type=float, default=0.0,
                   help="If >0, abstain when (top1-top2) < margin; report accuracy on covered subset and coverage.")

    # I/O and runtime
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--out_dir", type=str, default="outputs/fewshot")
    p.add_argument("--cache_dir", type=str, default="cache_mels_few")
    p.add_argument("--wav_dir", type=str, default="shipsear_segments")
    p.add_argument("--alt_wav_dir", type=str, default="shipsear_raw")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# =========================
# Utilities
# =========================

def load_checkpoint(ckpt_path, device):
    # explicit weights_only=False to avoid future default flip warning context
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_args = ckpt.get('args', {})
    if not isinstance(saved_args, dict):
        try:
            saved_args = vars(saved_args)
        except Exception:
            saved_args = dict(saved_args.__dict__) if hasattr(saved_args, "__dict__") else {}

    class_names = ckpt.get('class_names', ckpt.get('classes', None))
    if class_names is None:
        raise RuntimeError("Checkpoint missing 'class_names'/'classes'.")

    # Featurizer coherent with training
    featurizer = MultiResLogMel(
        target_sr=saved_args['sr'],
        n_mels=saved_args['n_mels'],
        configs=((saved_args['nfft1'], saved_args['hop1']),
                 (saved_args['nfft2'], saved_args['hop2'])),
        target_secs=saved_args['target_secs'],
        use_deltas=(saved_args['in_ch'] == 6),
        use_attention=True
    )

    # Backbone + projection heads as per training
    model = MiniResNet(
        in_ch=saved_args['in_ch'],
        emb_dim=saved_args['emb_dim'],
        proj_dim=saved_args['proj_dim'],
        num_classes=len(class_names),
        num_heads=saved_args.get('num_heads', 3)
    ).to(device)

    # Tolerant load: ignore unexpected (e.g., quantum.*)
    sd = ckpt['model']
    incompat = model.load_state_dict(sd, strict=False)
    if getattr(incompat, "unexpected_keys", None):
        uq = [k for k in incompat.unexpected_keys if k.startswith("quantum.")]
        if uq:
            print(f"[WARN] Ignoro {len(uq)} pesi del ramo quantistico: primo -> {uq[0]} ...")
        else:
            print(f"[WARN] Chiavi inattese nello state_dict: {incompat.unexpected_keys[:3]} ...")
    if getattr(incompat, "missing_keys", None) and len(incompat.missing_keys) > 0:
        print(f"[WARN] Pesi mancanti ri-inizializzati: {incompat.missing_keys[:3]} ...")

    model.eval()

    fe_cfg = {
        "sr": saved_args['sr'],
        "n_mels": saved_args['n_mels'],
        "configs": [(saved_args['nfft1'], saved_args['hop1']),
                    (saved_args['nfft2'], saved_args['hop2'])],
        "target_secs": saved_args['target_secs'],
        "use_deltas": (saved_args['in_ch'] == 6),
        "use_attention": True
    }

    io_dirs = {
        "wav_dir": saved_args.get('wav_dir', 'shipsear_segments'),
        "alt_wav_dir": saved_args.get('alt_wav_dir', 'shipsear_raw')
    }

    return class_names, featurizer, model, fe_cfg, io_dirs


@torch.no_grad()
def extract_embeddings(model, loader, device):
    E, L, Y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        e, p, logits = model(xb, return_proj=True)
        E.append(e.cpu().numpy())
        L.append(logits.cpu().numpy())
        Y.append(yb.cpu().numpy())
    if len(E) == 0:
        # empty guards
        return (np.zeros((0, model.norm.normalized_shape[0])), 
                np.zeros((0, model.cls.out_features)), 
                np.zeros((0,), dtype=np.int64))
    return np.concatenate(E, 0), np.concatenate(L, 0), np.concatenate(Y, 0)


def build_loader(df, data_dir, class_names, featurizer, cache, batch_size, workers, wav_dir, alt_wav_dir):
    def to_records(d):
        return d[["filename", "category"]].to_dict(orient="records")
    ds = ShipsearDataset(
        to_records(df), data_dir, class_names, featurizer, cache,
        mode="eval", aug_spec=False, aug_wav=False,
        wav_dir=wav_dir, alt_wav_dir=alt_wav_dir
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True)


def l2_normalize(a, axis=1, eps=1e-8):
    n = np.linalg.norm(a, axis=axis, keepdims=True)
    return a / np.maximum(n, eps)


def apply_norm(E, mode):
    if mode == "none":
        return E
    if mode == "l2":
        return l2_normalize(E, axis=1)
    if mode == "zscore":
        mu = E.mean(0, keepdims=True); sd = E.std(0, keepdims=True) + 1e-8
        return (E - mu) / sd
    if mode == "whiten":
        X = E - E.mean(0, keepdims=True)
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        W = Vt.T @ np.diag(1.0 / (S + 1e-6)) @ Vt
        return X @ W
    return E


def nearest_prototype_cosine(E, protos):
    Pn = l2_normalize(protos, axis=1)
    En = l2_normalize(E, axis=1)
    sims = En @ Pn.T                # [N, C]
    ypred = sims.argmax(1)
    return ypred, sims


def nearest_prototype_mahal(E, protos, cov_mode="lw", reg=1e-3, E_support=None):
    """
    cov_mode:
      - 'lw'           : LedoitWolf fit on E (query)
      - 'diag'         : variance diagonal on E
      - 'emp'          : empirical covariance on E (+reg)
      - 'support_lw'   : LedoitWolf fit on E_support (no transductive info)
      - 'support_diag' : variance diagonal on E_support
    """
    def cov_inv_from(X, mode):
        X = np.asarray(X)
        if X.shape[0] < 3:
            v = np.var(X, axis=0) + reg
            return np.diag(1.0 / v)
        if mode == "lw":
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf(store_precision=True).fit(X)
            return lw.precision_
        if mode == "diag":
            v = np.var(X, axis=0) + reg
            return np.diag(1.0 / v)
        if mode == "emp":
            C = np.cov(X.T) + reg * np.eye(X.shape[1])
            return np.linalg.pinv(C)
        raise ValueError(mode)

    if cov_mode.startswith("support"):
        if E_support is None or len(E_support) == 0:
            VI = cov_inv_from(E, "lw")  # fallback
        else:
            VI = cov_inv_from(E_support, "lw" if cov_mode == "support_lw" else "diag")
    else:
        VI = cov_inv_from(E, cov_mode)

    diffs = E[:, None, :] - protos[None, :, :]
    d2 = np.einsum("nik,kl,nil->ni", diffs, VI, diffs, optimize=True)
    ypred = d2.argmin(1)
    return ypred, -np.sqrt(np.maximum(d2, 0.0))


def save_confusion_png(cm, labels, out_png):
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(max(6, 0.5*len(labels)), max(5, 0.5*len(labels))))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation='nearest')
        ax.set_title("Confusion Matrix")
        fig.colorbar(im)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] Could not save confusion matrix image: {e}")


def choose_unseen(meta, n_unseen, strategy="max", seed=42, exclude=None):
    exclude = set([e.strip() for e in (exclude or []) if e.strip()])
    if strategy == "max":
        counts = meta['category'].value_counts().sort_values(ascending=False)
    else:
        counts = meta['category'].value_counts().sort_values(ascending=True)
    ordered = [c for c in counts.index if c not in exclude]
    return ordered[:n_unseen]


def make_support_query_from_meta(meta, unseen, k_shot, seed=42):
    rng = np.random.RandomState(seed)

    import re, pandas as pd
    rec_re = re.compile(r'^(.*?)(?:[-_]?seg\d+|_\d+|\d+)\.wav$', re.IGNORECASE)
    def rec_id(fn):
        # usa SOLO il basename, evita dipendenze dal path
        base = str(fn).split('/')[-1].split('\\')[-1]
        m = rec_re.match(base)
        return m.group(1) if m else base.rsplit('.', 1)[0]

    # ---------- PASSO 1: campiona tutti i SUPPORT ----------
    sup_rows = []
    for c in unseen:
        dfc = meta[meta['category'] == c].copy()
        dfc['recording'] = dfc['filename'].map(rec_id)

        dfc_14 = dfc[dfc['fold'].isin([1,2,3,4])]
        dfc_5  = dfc[dfc['fold'] == 5]

        # preferisci fold 1-4, poi riempi da 5 se serve
        if len(dfc_14) >= k_shot:
            sup = dfc_14.sample(n=k_shot, random_state=seed)
        else:
            take_14 = dfc_14.sample(n=len(dfc_14), random_state=seed) if len(dfc_14) > 0 else dfc_14
            need = k_shot - len(take_14)
            take_5 = dfc_5.sample(n=min(need, len(dfc_5)), random_state=seed) if need > 0 else dfc_5.iloc[0:0]
            sup = pd.concat([take_14, take_5], ignore_index=True)
        sup_rows.append(sup)

    support = pd.concat(sup_rows).reset_index(drop=True) if sup_rows else pd.DataFrame(columns=meta.columns)

    # insieme GLOBALE finale dei recording usati nel SUPPORT
    support['__rec__'] = support['filename'].map(rec_id)
    support_rec_global = set(support['__rec__'].tolist())

    # ---------- PASSO 2: costruisci tutti i QUERY filtrando il globale ----------
    qry_rows = []
    for c in unseen:
        dfc = meta[meta['category'] == c].copy()
        dfc['recording'] = dfc['filename'].map(rec_id)

        # escludi QUALSIASI recording già usato nel support (globale)
        dfc = dfc[~dfc['recording'].isin(support_rec_global)]

        dfc_14 = dfc[dfc['fold'].isin([1,2,3,4])]
        dfc_5  = dfc[dfc['fold'] == 5]
        qry = pd.concat([dfc_5, dfc_14], ignore_index=True)  # priorità al fold 5
        qry_rows.append(qry)

    query = pd.concat(qry_rows).reset_index(drop=True) if qry_rows else pd.DataFrame(columns=meta.columns)

    # pulizia colonne tecniche
    support = support.drop(columns=['__rec__'], errors='ignore')

    return support, query



def encode_labels(df, class_names):
    cls2id = {c:i for i,c in enumerate(class_names)}
    return df['category'].map(cls2id).values.astype(np.int64)


def proto_diagnostics(E_sup, Y_sup, unseen_ids, protos, out_dir, class_names):
    # Intra-class mean L2 to proto
    stats = {}
    for j, cid in enumerate(unseen_ids):
        m = (Y_sup == cid)
        Ej = E_sup[m]
        if len(Ej) > 0:
            intra = np.linalg.norm(Ej - protos[j], axis=1).mean()
        else:
            intra = float("nan")
        stats[class_names[cid]] = {"intra_mean_l2_to_proto": float(intra)}

    # Inter-prototype distances (L2)
    P = protos
    proto_l2 = np.sqrt(((P[:, None, :] - P[None, :, :])**2).sum(-1))
    labels = [class_names[c] for c in unseen_ids]

    pd.DataFrame(proto_l2, index=labels, columns=labels)\
      .to_csv(os.path.join(out_dir, "proto_l2_matrix.csv"))

    with open(os.path.join(out_dir, "proto_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Cosine between prototypes
    Pn = l2_normalize(protos, axis=1)
    cosPP = (Pn @ Pn.T)
    pd.DataFrame(cosPP, index=labels, columns=labels)\
      .to_csv(os.path.join(out_dir, "proto_proto_cosine.csv"))


# =========================
# Main
# =========================

def main():
    args = parse_args()
    set_seed(args.seed)
    device = device_select(args.device)
    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    # Load checkpoint + reconstruct featurizer/model
    class_names, featurizer, model, fe_cfg, io_dirs = load_checkpoint(args.checkpoint, device)
    cache = MelCache(cache_dir=args.cache_dir, fe_cfg=fe_cfg)

    # ----------------- Build SUPPORT/QUERY -----------------
    if args.support_csv and args.query_csv:
        support = pd.read_csv(args.support_csv)
        query   = pd.read_csv(args.query_csv)
        unseen = sorted(support['category'].unique().tolist())
        print(f"[INFO] Loaded provided SUPPORT/QUERY. Unseen={unseen}")

           # --- NEW: verifica e rimozione recording sovrapposti ---
        
        rec_re = re.compile(r'^(.*?)(?:[-_]?seg\d+|_\d+|\d+)\.wav$', re.IGNORECASE)
        def rec_id(fn: str):
          base = str(fn).split('/')[-1].split('\\')[-1]
          m = rec_re.match(base)
          return m.group(1) if m else base.rsplit('.', 1)[0]

        support["__rec__"] = support["filename"].map(rec_id)
        query["__rec__"]   = query["filename"].map(rec_id)


        overlap = set(support["__rec__"]).intersection(set(query["__rec__"]))
        if overlap:
          before = len(query)
          query = query[~query["__rec__"].isin(overlap)].copy()
          print(f"[FIX] Rimosse {before - len(query)} clip dal QUERY per disgiunzione recording "
              f"(overlap={len(overlap)}; es. {list(sorted(overlap))[:5]})")
        else:
          print("[OK] Nessuna registrazione in comune tra SUPPORT e QUERY.")

    # pulizia colonne tecniche
        support.drop(columns=["__rec__"], inplace=True, errors="ignore")
        query.drop(columns=["__rec__"], inplace=True, errors="ignore")
    else:
        if not args.meta_csv:
            raise ValueError("Provide either --meta_csv (auto few-shot) OR both --support_csv and --query_csv.")
        meta = pd.read_csv(args.meta_csv)

        exclude = [s for s in args.exclude_classes.split(",")] if args.exclude_classes else []
        if exclude:
            print(f"[INFO] Excluding classes from selection: {exclude}")
        unseen = choose_unseen(meta, n_unseen=args.n_unseen, strategy=args.strategy,
                               seed=args.seed, exclude=exclude)
        print(f"[INFO] Selected unseen classes ({args.strategy}): {unseen}")

        support, query = make_support_query_from_meta(meta, unseen, k_shot=args.k_shot, seed=args.seed)
        
        # Sanity check: nessun recording in comune tra SUPPORT e QUERY
        rec_re = re.compile(r'^(.*?)(?:[-_]?seg\d+|_\d+|\d+)\.wav$', re.IGNORECASE)
        def rec_id(fn):
           base = str(fn).split('/')[-1].split('\\')[-1]
           m = rec_re.match(base)
           return m.group(1) if m else base.rsplit('.', 1)[0]

        sup_rec = set(pd.Series(support['filename']).map(rec_id).tolist())
        qry_rec = set(pd.Series(query['filename']).map(rec_id).tolist())

        _overlap = sup_rec.intersection(qry_rec)
        if len(_overlap) > 0:
         raise RuntimeError(f"[LEAK] {len(_overlap)} recording_id compaiono in entrambi i set (es. {list(sorted(_overlap))[:5]})")


        # persist for traceability
        sup_path = os.path.join(args.out_dir, "support.csv")
        qry_path = os.path.join(args.out_dir, "query.csv")
        support.to_csv(sup_path, index=False)
        query.to_csv(qry_path, index=False)
        rec_re = re.compile(r'^(.*?)(?:[-_]?seg\d+|_\d+|\d+)\.wav$', re.IGNORECASE)

        def rec_id(fn):
          base = str(fn).split('/')[-1].split('\\')[-1]
          m = rec_re.match(base)
          return m.group(1) if m else base.rsplit('.', 1)[0]


        _sup = support['filename'].map(rec_id)
        _qry = query['filename'].map(rec_id)
        _overlap = set(_sup).intersection(set(_qry))
        if _overlap:
    # Fermati: c’è leak. Salva un report e alza errore.
          leak_path = os.path.join(args.out_dir, "leak_recordings.txt")
          with open(leak_path, "w", encoding="utf-8") as f:
            for r in sorted(_overlap):
              f.write(r + "\n")
          raise RuntimeError(f"[LEAK] {len(_overlap)} recording in comune tra SUPPORT e QUERY. Vedi {leak_path}")
        else:
          print("[OK] Nessuna registrazione condivisa tra SUPPORT e QUERY.")
        print(f"[INFO] Saved SUPPORT to {sup_path} | QUERY to {qry_path}")

    # Map labels to the checkpoint's class order
    y_sup = encode_labels(support, class_names)
    y_qry = encode_labels(query, class_names)

    # unseen check wrt checkpoint class names
    unseen_ids = [class_names.index(c) for c in unseen if c in class_names]
    if len(unseen_ids) != len(unseen):
        missing = [c for c in unseen if c not in class_names]
        raise RuntimeError(f"Classes not present in checkpoint class_names: {missing}")

    # ----------------- Build loaders & extract embeddings -----------------
    dl_sup = build_loader(support, args.data_dir, class_names, featurizer, cache,
                          args.batch_size, args.workers, io_dirs['wav_dir'], io_dirs['alt_wav_dir'])
    dl_qry = build_loader(query, args.data_dir, class_names, featurizer, cache,
                          args.batch_size, args.workers, io_dirs['wav_dir'], io_dirs['alt_wav_dir'])

    print("[INFO] Extracting embeddings for SUPPORT...")
    E_sup, L_sup, Y_sup = extract_embeddings(model, dl_sup, device)
    print("[INFO] Extracting embeddings for QUERY...")
    E_qry, L_qry, Y_qry = extract_embeddings(model, dl_qry, device)

    # Normalization (optional)
    E_sup = apply_norm(E_sup, args.norm)
    E_qry = apply_norm(E_qry, args.norm)

    # ----------------- Build unseen prototypes from SUPPORT -----------------
    protos = []
    for cid in unseen_ids:
        mask = (Y_sup == cid)
        if mask.sum() == 0:
            protos.append(np.zeros((E_sup.shape[1],), dtype=np.float32))
        else:
            protos.append(E_sup[mask].mean(0))
    protos = np.stack(protos, 0)  # [U, D]

    # Reindex QUERY labels into 0..U-1 for reports; keep only unseen
    cid2u = {cid: i for i, cid in enumerate(unseen_ids)}
    keep_mask = np.array([c in cid2u for c in Y_qry], dtype=bool)
    E_qry_u = E_qry[keep_mask]
    u_true  = np.array([cid2u[c] for c in Y_qry[keep_mask]], dtype=np.int64)

    # ----------------- Inference: cosine & mahal -----------------
    y_pred_cos, sims = nearest_prototype_cosine(E_qry_u, protos)
    y_pred_mah, _    = nearest_prototype_mahal(E_qry_u, protos,
                                               cov_mode=args.cov_mode, reg=1e-3, E_support=E_sup)

    # ----------------- Diagnostics -----------------
    # 1) Cosine margins per query
    top2   = np.partition(sims, -2, axis=1)[:, -2:]
    top1   = top2[:, 1]
    second = top2[:, 0]
    margins = top1 - second
    cos_diag = []
    kept_idx = np.where(keep_mask)[0]
    for i in range(len(u_true)):
        cos_diag.append({
            "qry_idx": int(kept_idx[i]),
            "true_u": int(u_true[i]),
            "pred_u": int(y_pred_cos[i]),
            "top1": float(top1[i]),
            "second": float(second[i]),
            "margin": float(margins[i]),
        })
    with open(os.path.join(args.out_dir, "cosine_margins.json"), "w") as f:
        json.dump(cos_diag, f, indent=2)

    # 2) Prototype diagnostics (intra/inter + proto-proto cosine)
    proto_diagnostics(E_sup, Y_sup, unseen_ids, protos, args.out_dir, class_names)

    # 3) Silhouette (cosine) on unseen query
    try:
        sil = float(silhouette_score(E_qry_u, u_true, metric='cosine')) if len(np.unique(u_true)) > 1 and len(E_qry_u) > len(np.unique(u_true)) else float("nan")
    except Exception:
        sil = float("nan")

    # 4) kNN in support (cosine) per query: count how many neighbors share the same recording/class
    #    Rebuild recording IDs to check contextual overlap
    rec_re = re.compile(r'^(.*?)(?:[-_]?seg\d+|_\d+|\d+)\.wav$', re.IGNORECASE)

    def rec_id(fn):
        m = rec_re.match(str(fn)); return m.group(1) if m else str(fn)

    support_recs = np.array([rec_id(fn) for fn in support['filename'].values])
    query_recs   = np.array([rec_id(fn) for fn in query['filename'].values])[kept_idx]
    S = cosine_similarity(E_qry_u, E_sup) if (len(E_qry_u) and len(E_sup)) else np.zeros((0,0))
    topk = min(5, S.shape[1]) if S.size else 0
    nn_rows = []
    if topk > 0:
        nn_idx = np.argsort(-S, axis=1)[:, :topk]
        for i in range(nn_idx.shape[0]):
            same_rec = [(support_recs[j] == query_recs[i]) for j in nn_idx[i]]
            same_cls = [(y_sup[j] == unseen_ids[u_true[i]]) for j in nn_idx[i]]
            nn_rows.append({
                "qry_idx": int(kept_idx[i]),
                "qry_true_u": int(u_true[i]),
                "same_recording_in_topk": int(sum(same_rec)),
                "same_class_in_topk": int(sum(same_cls))
            })
    with open(os.path.join(args.out_dir, "knn_support_diagnostics.json"), "w") as f:
        json.dump(nn_rows, f, indent=2)

    # ----------------- Reports -----------------
    u_labels = [class_names[cid] for cid in unseen_ids]
    rep_cos = classification_report(u_true, y_pred_cos, target_names=u_labels, digits=3, zero_division=0, output_dict=True)
    rep_mah = classification_report(u_true, y_pred_mah, target_names=u_labels, digits=3, zero_division=0, output_dict=True)

    cm_cos = confusion_matrix(u_true, y_pred_cos, labels=list(range(len(u_labels))))
    cm_mah = confusion_matrix(u_true, y_pred_mah, labels=list(range(len(u_labels))))
    save_confusion_png(cm_cos, u_labels, os.path.join(args.out_dir, "cm_cosine3.png"))
    save_confusion_png(cm_mah, u_labels, os.path.join(args.out_dir, "cm_mahal3.png"))

    # Macro-F1 and accuracy summary
    def summarize(rep):
        acc = rep.get("accuracy", float('nan'))
        macro = rep.get("macro avg", {}).get("f1-score", float('nan'))
        weighted = rep.get("weighted avg", {}).get("f1-score", float('nan'))
        return acc, macro, weighted

    acc_c, f1m_c, f1w_c = summarize(rep_cos)
    acc_m, f1m_m, f1w_m = summarize(rep_mah)

    results = {
        "unseen_classes": u_labels,
        "k_shot": args.k_shot,
        "strategy": args.strategy,
        "n_unseen": len(u_labels),
        "support_size": int(E_sup.shape[0]),
        "query_size": int(E_qry_u.shape[0]),
        "norm": args.norm,
        "cov_mode": args.cov_mode,
        "cosine": {
            "accuracy": acc_c, "macro_f1": f1m_c, "weighted_f1": f1w_c,
            "per_class": {lbl: rep_cos[lbl] for lbl in u_labels},
            "silhouette_on_unseen": sil
        },
        "mahalanobis": {
            "accuracy": acc_m, "macro_f1": f1m_m, "weighted_f1": f1w_m,
            "per_class": {lbl: rep_mah[lbl] for lbl in u_labels}
        }
    }

    # Abstention on cosine margin (optional)
    if args.cosine_abstain_margin > 0:
        keep_cov = margins >= args.cosine_abstain_margin
        if keep_cov.sum() > 0:
            acc_on_cov = float((y_pred_cos[keep_cov] == u_true[keep_cov]).mean())
            results["cosine"]["abstain_margin"] = args.cosine_abstain_margin
            results["cosine"]["coverage"] = float(keep_cov.mean())
            results["cosine"]["acc_on_covered"] = acc_on_cov

    out_json = os.path.join(args.out_dir, "fewshot_results3.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[RESULTS] Saved JSON to {out_json}")
    print(f"[RESULTS] COS  acc={acc_c:.3f}  macroF1={f1m_c:.3f}  (sil={sil if isinstance(sil, float) else float('nan'):.3f})")
    print(f"[RESULTS] MAH  acc={acc_m:.3f}  macroF1={f1m_m:.3f}")
    print(f"[INFO] Confusion matrices saved as cm_cosine.png / cm_mahal.png")

if __name__ == "__main__":
    main()
