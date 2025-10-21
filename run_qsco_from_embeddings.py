#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_qsco_from_embeddings.py
Versione PATCH-2: normalizzazione, ensemble unsupervised (Mahalanobis + LOF),
soglie data-driven (FPR/GMM/OTSU), salvataggi CSV/JSON, retro-compatibile.

Richiede:
- torch, numpy
- scikit-learn (per metriche, LedoitWolf, LOF, GMM)
- head_qsco.py (PATCH-1) nella PYTHONPATH (QscoHead, train_qsco, infer_qsco_scores)
"""

import os
import json
import argparse
from typing import Optional, Tuple

import numpy as np
import torch

# Metriche (se disponibili)
try:
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        f1_score, precision_score, recall_score,
        confusion_matrix, accuracy_score, balanced_accuracy_score
    )
    _SK_OK = True
except Exception:
    _SK_OK = False

# Detector extra
try:
    from sklearn.covariance import LedoitWolf
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.mixture import GaussianMixture
    _SK_DET_OK = True
except Exception:
    _SK_DET_OK = False

# QSCO head (PATCH-1)
from head_qsco import QscoHead, train_qsco, infer_qsco_scores


# -----------------------------
# Utility: normalizzazione
# -----------------------------
def _normalize(E: torch.Tensor, mode: str, ref: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
    """
    mode:
      - 'none'
      - 'l2'
      - 'zscore'  => usa (mu, sd) della ref se passata
      - 'whiten'  => usa (mu, W) della ref se passata (W = matrice whitening)
    ref:
      - per zscore: (mu, sd)
      - per whiten: (mu, W)
    """
    if mode == "none":
        return E, None

    if mode == "l2":
        En = torch.nn.functional.normalize(E, p=2, dim=1)
        return En, None

    if mode == "zscore":
        if ref is None:
            mu = E.mean(0, keepdim=True)
            sd = E.std(0, keepdim=True).clamp_min(1e-6)
            return (E - mu) / sd, (mu, sd)
        else:
            mu, sd = ref
            return (E - mu) / sd, (mu, sd)

    if mode == "whiten":
        if ref is None:
            mu = E.mean(0, keepdim=True)
            Xc = E - mu
            C = (Xc.t() @ Xc) / max(1, Xc.size(0) - 1)
            eigvals, eigvecs = torch.linalg.eigh(C + 1e-5 * torch.eye(C.size(0), device=C.device))

            W = eigvecs @ torch.diag(1.0 / eigvals.clamp_min(1e-6).sqrt()) @ eigvecs.t()
            Ew = (E - mu) @ W
            return Ew, (mu, W)
        else:
            mu, W = ref
            return (E - mu) @ W, (mu, W)

    raise ValueError(f"Unknown norm mode: {mode}")


def _standardize_np(a: np.ndarray) -> np.ndarray:
    m = a.mean()
    s = a.std()
    if s < 1e-12:
        return a * 0.0
    return (a - m) / s


# -----------------------------
# Caricamento embeddings
# -----------------------------
def _load_tensor(path: str, key_fallback: Optional[str] = None) -> Optional[torch.Tensor]:
    if not os.path.isfile(path):
        return None
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, dict) and key_fallback is not None and key_fallback in obj:
        t = obj[key_fallback]
        if isinstance(t, torch.Tensor):
            return t
    # prova a forzare
    try:
        return torch.as_tensor(obj)
    except Exception:
        return None


def _find_first_existing(base_dir: str, candidates: list) -> Optional[str]:
    for name in candidates:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return p
    return None


# -----------------------------
# Soglia data-driven
# -----------------------------
def _choose_thr(nov_cal: np.ndarray, mode: str, target_fpr: float = 0.05) -> float:
    nov = np.asarray(nov_cal).astype(np.float64)

    if mode == "fixed":
        # La soglia fissa viene gestita fuori (args.thr)
        return float("nan")

    if mode == "fpr":
        # quantile 1 - FPR dei known
        q = np.quantile(nov, 1.0 - target_fpr)
        return float(q)

    if mode == "gmm":
        if not _SK_DET_OK:
            # fallback: usa media
            return float(nov.mean())
        gm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(nov.reshape(-1, 1))
        means = gm.means_.ravel()
        # soglia alla media tra i due centroidi
        return float(np.mean(means))

    if mode == "otsu":
        # Istogramma 256 bin
        hist, bin_edges = np.histogram(nov, bins=256)
        total = nov.size
        sum_total = (hist * ((bin_edges[:-1] + bin_edges[1:]) * 0.5)).sum()
        sumB = 0.0
        wB = 0.0
        max_var, thr = 0.0, (bin_edges[0] + bin_edges[1]) * 0.5
        for i in range(256):
            wB += hist[i]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += hist[i] * ((bin_edges[i] + bin_edges[i + 1]) * 0.5)
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                thr = (bin_edges[i] + bin_edges[i + 1]) * 0.5
        return float(thr)

    raise ValueError(f"Unknown thr_mode: {mode}")


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("QSCO from pre-computed embeddings (PATCH-2)")
    p.add_argument("--emb_dir", type=str, required=True, help="Directory con i file *.pt degli embeddings")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)

    # compatibilità
    p.add_argument("--thr", type=float, default=0.5, help="Soglia di default (solo se --thr_mode=fixed)")
    p.add_argument("--unsupervised", action="store_true", help="Usa training unsupervised del QSCO")
    p.add_argument("--save_csv", action="store_true", help="Salva CSV di risultati")

    # PATCH-2
    p.add_argument("--norm", type=str, default="zscore",
                   choices=["none", "l2", "zscore", "whiten"],
                   help="Normalizzazione embeddings prima dei detector")
    p.add_argument("--unsup_mode", type=str, default="hard_eig",
                   choices=["hard_eig", "gauss", "hard_iso"],
                   help="Strategia outlier sintetici per QSCO unsupervised")
    p.add_argument("--synth_ratio", type=float, default=1.0,
                   help="Rapporto outlier sintetici / reali")
    p.add_argument("--inflate", type=float, default=2.0,
                   help="Raggio ellissoide per hard negatives")
    p.add_argument("--thr_mode", type=str, default="fpr",
                   choices=["fixed", "fpr", "gmm", "otsu"],
                   help="Strategia di soglia per novelty")
    p.add_argument("--target_fpr", type=float, default=0.05,
                   help="FPR target per la soglia (solo calib, thr_mode=fpr)")
    p.add_argument("--extra_detectors", type=str, default="all",
                   choices=["none", "maha", "lof", "all"],
                   help="Aggiunge detector unsupervised a QSCO e fa ensemble")
    return p.parse_args()


def main():
    args = parse_args()
    d = args.emb_dir
    os.makedirs(d, exist_ok=True)

    dev = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    # ---- Caricamento base (nomi comuni) ----
    path_Ec = _find_first_existing(d, ["E_calib.pt", "E_calib_open.pt", "E_train_calib.pt"])
    path_Et = _find_first_existing(d, ["E_test.pt", "E_eval.pt", "E_query.pt"])
    path_yc = _find_first_existing(d, ["y_calib.pt", "y_calib_open.pt"])
    path_yo = _find_first_existing(d, ["y_test_open.pt", "y_open_test.pt", "y_open.pt"])  # 1=unseen, 0=known (atteso)
    # closed-set (facoltativo, se disponibili)
    path_y_cls_true = _find_first_existing(d, ["y_test_cls_true.pt"])
    path_y_cls_pred = _find_first_existing(d, ["y_test_cls_pred.pt"])

    if path_Ec is None or path_Et is None:
        raise FileNotFoundError("Non ho trovato E_calib.pt/E_test.pt nella cartella emb_dir.")

    E_calib = _load_tensor(path_Ec)
    E_test = _load_tensor(path_Et)
    if E_calib is None or E_test is None:
        raise RuntimeError("Impossibile caricare gli embeddings (formato non riconosciuto).")

    y_calib = _load_tensor(path_yc) if (path_yc is not None and not args.unsupervised) else None
    y_open = _load_tensor(path_yo) if path_yo is not None else None

    # ---- Normalizzazione (fit su calib) ----
    E_calib = E_calib.to(dev).float()
    E_test = E_test.to(dev).float()

    E_calib, norm_stat = _normalize(E_calib, args.norm, ref=None)
    if args.norm == "zscore" and norm_stat is not None:
        mu, sd = norm_stat
        E_test, _ = _normalize(E_test, "zscore", ref=(mu, sd))
    elif args.norm == "whiten" and norm_stat is not None:
        mu, W = norm_stat
        E_test, _ = _normalize(E_test, "whiten", ref=(mu, W))
    elif args.norm == "l2":
        E_test, _ = _normalize(E_test, "l2", ref=None)

    # ---- QSCO head (PATCH-1) ----
    print("[QSCO] Training...")
    head = QscoHead(in_dim=E_calib.size(1), n_qubits=None, depth=2, backend="default.qubit")
    best_loss, last_loss = train_qsco(
        head, E_calib, y_calib if not args.unsupervised else None,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        synth_ratio=args.synth_ratio, inflate=args.inflate, unsup_mode=args.unsup_mode,
        device=str(dev), verbose=True
    )

    # Salvataggio head
    torch.save(head.state_dict(), os.path.join(d, "qsco_head.pt"))
    print(f"[QSCO] Head saved to {os.path.join(d, 'qsco_head.pt')}")

    # ---- Inference QSCO ----
    print("[QSCO] Inference on E_test...")
    known_qsco = infer_qsco_scores(head, E_test, device=str(dev)).numpy()  # ∈[0,1]
    nov_qsco = 1.0 - known_qsco                                         # novelty

    # ---- Extra detectors (unsupervised) ----
    Ecal_np = E_calib.detach().cpu().numpy()
    Etest_np = E_test.detach().cpu().numpy()
    scores_list = [nov_qsco]  # sempre presente

    maha_np = None
    lof_np = None

    if args.extra_detectors in ("maha", "all"):
        if not _SK_DET_OK:
            print("[WARN] scikit-learn non disponibile: Mahalanobis disabilitato.")
        else:
            lw = LedoitWolf().fit(Ecal_np)
            mu = lw.location_
            cov = lw.covariance_
            # Mahalanobis distance
            from numpy.linalg import inv
            iC = inv(cov + 1e-6 * np.eye(cov.shape[0]))
            dif = Etest_np - mu
            maha_np = np.einsum("...i,ij,...j->...", dif, iC, dif)
            scores_list.append(_standardize_np(maha_np))

    if args.extra_detectors in ("lof", "all"):
        if not _SK_DET_OK:
            print("[WARN] scikit-learn non disponibile: LOF disabilitato.")
        else:
            lof_model = LocalOutlierFactor(n_neighbors=35, novelty=True, metric="euclidean")
            lof_model.fit(Ecal_np)
            lof_np = -lof_model.score_samples(Etest_np)  # invert: alto = novelty
            scores_list.append(_standardize_np(lof_np))

    # Ensemble (media dei punteggi standardizzati)
    if len(scores_list) == 1:
        novelty_np = nov_qsco
    else:
        novelty_np = np.mean(np.stack([_standardize_np(s) for s in scores_list], axis=1), axis=1)

    # ---- Soglia data-driven ----
    # usiamo la dist dei known stimata da QSCO su calibrazione
    known_qsco_cal = infer_qsco_scores(head, E_calib, device=str(dev)).numpy()
    nov_cal = 1.0 - known_qsco_cal

    if args.thr_mode == "fixed":
        thr_eff = float(args.thr)
    else:
        thr_eff = _choose_thr(nov_cal, mode=args.thr_mode, target_fpr=args.target_fpr)

    preds_open_np = (novelty_np >= thr_eff).astype(np.int64)
    print(f"[THR] mode={args.thr_mode}  thr={thr_eff:.6f}  (target_fpr={args.target_fpr:.3f})")

    # ---- Salvataggi base ----
    torch.save(torch.tensor(novelty_np, dtype=torch.float32), os.path.join(d, "qsco_scores.pt"))
    torch.save(torch.tensor(preds_open_np, dtype=torch.int64), os.path.join(d, "qsco_preds_open.pt"))
    print(f"[QSCO] Saved: qsco_scores.pt, qsco_preds_open.pt in {d}")

    # ---- Valutazione OPEN-SET (se abbiamo y_open e sklearn) ----
    results = {}
    if y_open is not None and _SK_OK:
        y_open_np = y_open.detach().cpu().numpy().astype(int)  # 1=unseen (positivi), 0=known
        try:
            auroc = roc_auc_score(y_open_np, novelty_np)
        except Exception:
            auroc = float("nan")
        try:
            auprc = average_precision_score(y_open_np, novelty_np)
        except Exception:
            auprc = float("nan")

        f1 = f1_score(y_open_np, preds_open_np, zero_division=0)
        prec = precision_score(y_open_np, preds_open_np, zero_division=0)
        rec = recall_score(y_open_np, preds_open_np, zero_division=0)
        cm = confusion_matrix(y_open_np, preds_open_np, labels=[0,1])

        print("\n================ OPEN-SET DETECTION (QSCO) ================")
        print(f"AUROC (unseen vs known, novelty as positive): {auroc:.4f}")
        print(f"AUPRC (unseen as positive):                {auprc:.4f}")
        print(f"F1 (thr={thr_eff:.2f}):                        {f1:.4f}")
        print(f"Precision (thr={thr_eff:.2f}):                 {prec:.4f}")
        print(f"Recall (thr={thr_eff:.2f}):                    {rec:.4f}")
        print("Confusion matrix (rows=true [known, unseen], cols=pred):")
        print(cm)

        results.update({
            "open_set": {
                "auroc": float(auroc),
                "auprc": float(auprc),
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "thr_eff": float(thr_eff),
                "thr_mode": args.thr_mode,
                "target_fpr": float(args.target_fpr),
                "confusion_matrix": cm.tolist(),
                "class_ratio": {
                    "known": int((y_open_np == 0).sum()),
                    "unseen": int((y_open_np == 1).sum())
                }
            }
        })
    else:
        print("\n[INFO] Valutazione OPEN-SET non eseguita (manca y_open o scikit-learn).")

    # ---- Valutazione CLOSED-SET (opzionale) ----
    if path_y_cls_true is not None and path_y_cls_pred is not None and _SK_OK:
        y_true = _load_tensor(path_y_cls_true)
        y_pred = _load_tensor(path_y_cls_pred)
        if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
            y_true_np = y_true.detach().cpu().numpy()
            y_pred_np = y_pred.detach().cpu().numpy()
            acc = accuracy_score(y_true_np, y_pred_np)
            bacc = balanced_accuracy_score(y_true_np, y_pred_np)
            mf1 = f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)
            cmc = confusion_matrix(y_true_np, y_pred_np)
            print("\n================ CLOSED-SET CLASSIFICATION (optional) ================")
            print(f"Closed-set Accuracy:         {acc:.4f}")
            print(f"Closed-set Balanced Accuracy:{bacc:.4f}")
            print(f"Closed-set Macro-F1:         {mf1:.4f}")
            print("Closed-set Confusion matrix (rows=true, cols=pred):")
            print(cmc)
            results.update({
                "closed_set": {
                    "accuracy": float(acc),
                    "balanced_accuracy": float(bacc),
                    "macro_f1": float(mf1),
                    "confusion_matrix": cmc.tolist()
                }
            })
    else:
        print("[INFO] Valutazione CLOSED-SET non eseguita (mancano y_test_cls_true/y_test_cls_pred o scikit-learn).")

    # ---- CSV/JSON ----
    if args.save_csv:
        csv_path = os.path.join(d, "qsco_results.csv")
        # salvatore minimale compatibile con assenza di pandas
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("novelty_score,pred_open\n")
            for s, p in zip(novelty_np.tolist(), preds_open_np.tolist()):
                f.write(f"{s:.8f},{int(p)}\n")
        print(f"\n[CSV] Salvato: {csv_path}")

    # JSON riassuntivo
    meta = {
        "params": {
            "device": str(dev),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "unsupervised": bool(args.unsupervised),
            "norm": args.norm,
            "unsup_mode": args.unsup_mode,
            "synth_ratio": float(args.synth_ratio),
            "inflate": float(args.inflate),
            "thr_mode": args.thr_mode,
            "target_fpr": float(args.target_fpr),
            "extra_detectors": args.extra_detectors,
        },
        "files": {
            "E_calib": path_Ec,
            "E_test": path_Et,
            "y_calib": path_yc,
            "y_open": path_yo
        },
        "loss": {"best": float(best_loss), "last": float(last_loss)},
        "threshold": {"effective": float(thr_eff)},
        "counts": {
            "n_calib": int(E_calib.size(0)),
            "n_test": int(E_test.size(0))
        }
    }
    meta.update(results)

    js_path = os.path.join(d, "qsco_metrics_summary.json")
    with open(js_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\n[JSON] Riassunto metriche salvato in {js_path}")


if __name__ == "__main__":
    main()
