#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, math, warnings
from typing import Dict, Tuple
import numpy as np
import torch

from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

# ---------- Utils ----------

def tload(path: str) -> torch.Tensor:
    return torch.load(path, map_location="cpu")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def zscore_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    return mu, sd

def zscore_transform(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd

def whiten_fit(X: np.ndarray, shrinkage: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0]-1)
    if shrinkage > 0:
        cov = (1 - shrinkage) * cov + shrinkage * np.eye(cov.shape[0], dtype=cov.dtype)
    w, V = np.linalg.eigh(cov)
    w[w <= 0] = 1e-8
    W = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    return mu, W

def whiten_transform(X: np.ndarray, mu: np.ndarray, W: np.ndarray) -> np.ndarray:
    return (X - mu) @ W

def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def energy_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    # -T * logsumexp(logits/T); usiamo il segno per avere "più alto = più novel"
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=1, keepdims=True))
    return (-T * lse).ravel() * (-1.0)

def mahalanobis2(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    xc = x - mean
    return np.einsum("nd,dd,nd->n", xc, inv_cov, xc, optimize=True)

def fit_inv_cov(X: np.ndarray, mode: str = "ledoit") -> Tuple[np.ndarray, np.ndarray]:
    if mode == "ledoit":
        est = LedoitWolf().fit(X)
        return est.location_.reshape(1, -1), np.linalg.inv(est.covariance_)
    elif mode == "empirical":
        est = EmpiricalCovariance().fit(X)
        return est.location_.reshape(1, -1), np.linalg.pinv(est.covariance_)
    elif mode == "diag":
        mu = X.mean(0, keepdims=True)
        var = X.var(0, keepdims=True)
        var[var <= 1e-8] = 1e-8
        inv_cov = np.diagflat(1.0 / var.ravel())
        return mu, inv_cov
    else:
        raise ValueError(f"Unknown cov_mode: {mode}")

def log_gaussian_ll(x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    w, _ = np.linalg.eigh(inv_cov)
    w[w <= 1e-12] = 1e-12
    logdet_inv = np.log(w).sum()
    logdet = -logdet_inv
    m2 = mahalanobis2(x, mean, inv_cov)
    return -0.5 * (m2 + d * np.log(2 * math.pi) + logdet)

def quantile_threshold_known(scores_known: np.ndarray, target_fpr: float) -> float:
    # punteggi più alti => più novel; FPR = P(known >= thr)
    return float(np.quantile(scores_known, 1.0 - target_fpr, method="higher"))

def eval_open(y_open_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict:
    y_pred_open = (scores >= thr).astype(np.int64)
    auroc = roc_auc_score(y_open_true, scores)
    auprc = average_precision_score(y_open_true, scores)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_open_true, y_pred_open, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_open_true, y_pred_open, labels=[0, 1])
    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        "F1": float(f1),
        "Precision": float(prec),
        "Recall": float(rec),
        "ConfusionMatrix": cm.tolist(),
        "thr": float(thr),
    }, y_pred_open

# ---------- Main classical pipeline ----------

def main():
    ap = argparse.ArgumentParser("CLASSIC Open-Set from pre-computed embeddings")

    ap.add_argument("--emb_dir", type=str, required=True,
                    help="Directory con E_calib.pt, E_test.pt, y_open.pt, (opz) y_calib.pt e L_test.pt")
    ap.add_argument("--norm", type=str, default="whiten", choices=["none", "zscore", "whiten"],
                    help="Normalizzazione fit su CALIB e applicata a TEST")
    ap.add_argument("--whiten_shrink", type=float, default=0.0,
                    help="Shrinkage [0..1] per la covarianza nel whitening")
    ap.add_argument("--cov_mode", type=str, default="ledoit", choices=["ledoit","empirical","diag"],
                    help="Stimatore di covarianza per Mahalanobis/Gauss")
    ap.add_argument("--detectors", type=str, default="all",
                    help="Comma-list: maha_class,maha_global,gauss_ll,lof,ocsvm,msp,energy | 'all'")
    ap.add_argument("--lof_k", type=int, default=20, help="n_neighbors per LOF (novelty=True)")
    ap.add_argument("--ocsvm_nu", type=float, default=0.05, help="nu per OneClassSVM")
    ap.add_argument("--ocsvm_gamma", type=str, default="scale", help="gamma per OneClassSVM")
    ap.add_argument("--temp", type=float, default=1.0, help="Temperatura per MSP/Energy (se L_test presente)")
    ap.add_argument("--thr_mode", type=str, default="fpr", choices=["fpr"], help="Selezione soglia")
    ap.add_argument("--target_fpr", type=float, default=0.05, help="FPR target per definire la soglia")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--save_all_scores", action="store_true", help="Salva vettori di punteggio e pred per detector")
    # nuovi flag
    ap.add_argument("--invert_yopen", action="store_true",
                    help="Inverti etichette open/known (1<->0)")
    ap.add_argument("--invert_scores", action="store_true",
                    help="Inverti la polarità di TUTTI i punteggi (score := -score)")

    args = ap.parse_args()
    d = args.emb_dir
    ensure_dir(d)

    # ---------- Caricamento tensori obbligatori ----------
    E_calib = tload(os.path.join(d, "E_calib.pt")).float().cpu().numpy()
    E_test  = tload(os.path.join(d, "E_test.pt")).float().cpu().numpy()
    y_open  = tload(os.path.join(d, "y_open.pt")).long().cpu().numpy().ravel()

    if args.invert_yopen:
        y_open = 1 - y_open

    # opzionali
    y_calib = None
    ycalib_path = os.path.join(d, "y_calib.pt")
    if os.path.exists(ycalib_path):
        y_calib = tload(ycalib_path).long().cpu().numpy().ravel()

    L_test = None
    ltest_path = os.path.join(d, "L_test.pt")
    if os.path.exists(ltest_path):
        L_test = tload(ltest_path).float().cpu().numpy()

    # ---------- Normalizzazione ----------
    if args.norm == "zscore":
        mu, sd = zscore_fit(E_calib)
        E_calib_n = zscore_transform(E_calib, mu, sd)
        E_test_n  = zscore_transform(E_test,  mu, sd)
    elif args.norm == "whiten":
        mu, W = whiten_fit(E_calib, shrinkage=args.whiten_shrink)
        E_calib_n = whiten_transform(E_calib, mu, W)
        E_test_n  = whiten_transform(E_test,  mu, W)
    else:
        E_calib_n = E_calib
        E_test_n  = E_test

    # ---------- Scelta detector ----------
    if args.detectors == "all":
        det_list = ["maha_class", "maha_global", "gauss_ll", "lof", "ocsvm"]
        if L_test is not None:
            det_list += ["msp", "energy"]
    else:
        det_list = [x.strip() for x in args.detectors.split(",") if x.strip()]

    all_scores: Dict[str, np.ndarray] = {}

    # 1) Mahalanobis per-classe
    if "maha_class" in det_list:
        if y_calib is None:
            warnings.warn("[maha_class] y_calib mancante: salto il detector.")
        else:
            classes = sorted(list(set(y_calib)))
            means, invcovs = [], []
            for c in classes:
                Xc = E_calib_n[y_calib == c]
                mu_c, inv_cov_c = fit_inv_cov(Xc, mode=args.cov_mode)
                means.append(mu_c.reshape(1, -1))
                invcovs.append(inv_cov_c)
            dists = [mahalanobis2(E_test_n, m, ic).reshape(-1,1) for m, ic in zip(means, invcovs)]
            D = np.concatenate(dists, axis=1)
            all_scores["maha_class"] = D.min(axis=1)  # più alto => più lontano da tutte le classi viste

    # 2) Mahalanobis globale
    if "maha_global" in det_list:
        mu_g, invcov_g = fit_inv_cov(E_calib_n, mode=args.cov_mode)
        all_scores["maha_global"] = mahalanobis2(E_test_n, mu_g, invcov_g)

    # 3) Gauss log-likelihood (novelty = -LL)
    if "gauss_ll" in det_list:
        mu_g, invcov_g = fit_inv_cov(E_calib_n, mode=args.cov_mode)
        ll = log_gaussian_ll(E_test_n, mu_g, invcov_g)
        all_scores["gauss_ll"] = -ll

    # 4) LOF
    if "lof" in det_list:
        lof = LocalOutlierFactor(n_neighbors=args.lof_k, novelty=True)
        lof.fit(E_calib_n)
        all_scores["lof"] = -lof.score_samples(E_test_n)  # invertiamo perché score_samples alto = inlier

    # 5) OCSVM
    if "ocsvm" in det_list:
        oc = OneClassSVM(nu=args.ocsvm_nu, kernel="rbf", gamma=args.ocsvm_gamma)
        oc.fit(E_calib_n)
        all_scores["ocsvm"] = -oc.score_samples(E_test_n)  # alto = inlier -> inverti

    # 6) MSP (se disponibili logits)
    if "msp" in det_list and L_test is not None:
        prob = softmax_np(L_test, T=args.temp)
        all_scores["msp"] = 1.0 - prob.max(axis=1)

    # 7) Energy (se disponibili logits)
    if "energy" in det_list and L_test is not None:
        all_scores["energy"] = energy_np(L_test, T=args.temp)

    # ---------- (opzionale) inversione polarità punteggi ----------
    if args.invert_scores:
        for k in list(all_scores.keys()):
            all_scores[k] = -all_scores[k]

    # ---------- Valutazione + salvataggi ----------
    y_open_true = y_open.astype(int)
    known_mask = (y_open_true == 0)

    # ensemble su z-score
    Zscores = {}
    for k, s in all_scores.items():
        s = s.astype(np.float64)
        mu_s, sd_s = s.mean(), s.std()
        sd_s = sd_s if sd_s > 0 else 1.0
        Zscores[k] = (s - mu_s) / sd_s

    if len(Zscores) >= 2:
        zmat = np.vstack([Zscores[k] for k in sorted(Zscores.keys())]).T
        all_scores["ensemble_meanZ"] = zmat.mean(axis=1)

    out_csv = os.path.join(d, "classic_results.csv")
    out_json = os.path.join(d, "classic_metrics_summary.json")

    rows = []
    summary: Dict[str, Dict] = {}

    for name, scores in all_scores.items():
        scores = scores.astype(np.float64)
        thr = quantile_threshold_known(scores[known_mask], args.target_fpr)
        res, ypred = eval_open(y_open_true, scores, thr)
        res["TargetFPR"] = args.target_fpr
        res["N_known"]   = int(known_mask.sum())
        res["N_open"]    = int((~known_mask).sum())
        summary[name] = res

        rows.append({
            "detector": name,
            "thr": res["thr"],
            "AUROC": res["AUROC"],
            "AUPRC": res["AUPRC"],
            "F1": res["F1"],
            "Precision": res["Precision"],
            "Recall": res["Recall"],
            "TN": res["ConfusionMatrix"][0][0],
            "FP": res["ConfusionMatrix"][0][1],
            "FN": res["ConfusionMatrix"][1][0],
            "TP": res["ConfusionMatrix"][1][1],
        })

        if args.save_all_scores:
            torch.save(torch.tensor(scores), os.path.join(d, f"classic_scores_{name}.pt"))
            torch.save(torch.tensor(ypred),  os.path.join(d, f"classic_preds_open_{name}.pt"))

    if args.save_csv and rows:
        import csv
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[CSV] Salvato: {out_csv}")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[JSON] Riassunto metriche salvato in {out_json}")

    print("\n================ CLASSICAL OPEN-SET (from embeddings) ================")
    keys_order = ["AUROC","AUPRC","F1","Precision","Recall","thr"]
    for name in sorted(summary.keys()):
        s = summary[name]
        line = " | ".join([f"{k}={s[k]:.4f}" if isinstance(s[k], float) else f"{k}={s[k]}" for k in keys_order])
        print(f"{name:>14s}: {line}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
