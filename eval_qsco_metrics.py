#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, csv
import numpy as np
import torch

def load_scores(emb_dir):
    csv_path = os.path.join(emb_dir, "qsco_results.csv")
    pt_path  = os.path.join(emb_dir, "qsco_scores.pt")
    if os.path.isfile(csv_path):
        xs, preds = [], []
        with open(csv_path, newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                xs.append(float(row["novelty_score"]))
                preds.append(int(row["pred_open"]))
        return np.array(xs, float), np.array(preds, int), "csv"
    elif os.path.isfile(pt_path):
        t = torch.load(pt_path, map_location="cpu")
        if isinstance(t, torch.Tensor):
            return t.numpy().astype(np.float64), None, "pt"
        return np.asarray(t, dtype=float), None, "pt"
    else:
        raise FileNotFoundError("Non trovo qsco_results.csv né qsco_scores.pt")

def load_labels(emb_dir):
    for name in ["y_test_open.pt", "y_open_test.pt", "y_open.pt"]:
        p = os.path.join(emb_dir, name)
        if os.path.isfile(p):
            t = torch.load(p, map_location="cpu")
            if isinstance(t, torch.Tensor):
                return t.numpy().astype(int)
            return np.asarray(t, dtype=int)
    return None

def load_thr_from_json(emb_dir):
    p = os.path.join(emb_dir, "qsco_metrics_summary.json")
    if os.path.isfile(p):
        with open(p, "r", encoding="utf-8") as f:
            js = json.load(f)
        thr = js.get("threshold", {}).get("effective", None)
        return float(thr) if thr is not None else None
    return None

def compute_curves(y, scores):
    # sort by score ascending
    o = np.argsort(scores)
    s, y = scores[o], y[o]
    # thresholds = unique scores
    thr = np.r_[[-np.inf], (s[1:] + s[:-1]) / 2.0, [np.inf]]
    # cumulative positives/negatives
    P = y.sum()
    N = y.size - P
    tp = (y[::-1].cumsum())[::-1]  # tp when threshold is just below each score
    fp = ((1 - y)[::-1].cumsum())[::-1]
    # align to threshold midpoints
    tp = np.r_[tp[0], tp[:-1]]
    fp = np.r_[fp[0], fp[:-1]]

    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tpr.copy()
    return thr, tpr, fpr, precision, recall

def auc(x, y):
    # assumes x is monotonic increasing
    return float(np.trapz(y, x))

def pick_thr_by_f1(y, scores):
    thr, tpr, fpr, prec, rec = compute_curves(y, scores)
    f1 = (2 * prec * rec) / np.maximum(prec + rec, 1e-12)
    k = np.nanargmax(f1)
    return float(thr[k]), float(f1[k]), float(prec[k]), float(rec[k])

def pick_thr_by_target_fpr(y_known_scores, target_fpr=0.05):
    # target_fpr on KNOWN (y=0 → negatives), we approximate by using novelty of known only
    q = np.quantile(y_known_scores, 1.0 - target_fpr)
    return float(q)

def metrics_at_thr(y, scores, thr):
    pred = (scores >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    P  = max(1, int((y == 1).sum()))
    N  = max(1, int((y == 0).sum()))
    prec = tp / max(tp + fp, 1)
    rec  = tp / P
    spec = tn / N
    f1   = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
    bacc = 0.5 * (rec + spec)
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=prec, recall=rec, specificity=spec, f1=f1, bacc=bacc)

def main():
    ap = argparse.ArgumentParser("Eval QSCO metrics")
    ap.add_argument("--emb_dir", required=True, type=str)
    ap.add_argument("--thr_mode", default="json", choices=["json","fixed","bestf1","fpr"])
    ap.add_argument("--thr", type=float, default=0.5, help="Usata se thr_mode=fixed")
    ap.add_argument("--target_fpr", type=float, default=0.05, help="Usata se thr_mode=fpr")
    args = ap.parse_args()

    scores, preds_file, src = load_scores(args.emb_dir)
    y = load_labels(args.emb_dir)
    thr_json = load_thr_from_json(args.emb_dir)

    print(f"[INFO] Loaded scores from {src}. N={scores.size}")
    print(f"[INFO] y_open present? {'YES' if y is not None else 'NO'}")
    if y is None:
        # stampiamo solo statistiche di distribuzione e % pred positive se abbiamo preds da csv
        print(f"  novelty: min={scores.min():.6f}  max={scores.max():.6f}  mean={scores.mean():.6f}  std={scores.std():.6f}")
        if preds_file is not None:
            rate = preds_file.mean()
            print(f"  predicted unseen rate (from file preds): {rate*100:.2f}%")
        if args.thr_mode == "json" and thr_json is not None:
            rate = (scores >= thr_json).mean()
            print(f"  predicted unseen rate at thr(JSON={thr_json:.6f}): {rate*100:.2f}%")
        return

    # Con etichette → metriche complete
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = roc_auc_score(y, scores)
        auprc = average_precision_score(y, scores)
    except Exception:
        # fallback rapido se sklearn non c'è
        thr, tpr, fpr, prec, rec = compute_curves(y, scores)
        # AUROC via trapz(fpr, tpr) ma servirebbe ordinare per fpr crescente
        order = np.argsort(fpr)
        auroc = auc(fpr[order], tpr[order])
        # AUPRC: trapz(recall, precision) con recall crescente
        order = np.argsort(rec)
        auprc = auc(rec[order], prec[order])

    # Threshold selection
    if args.thr_mode == "json":
        if thr_json is None:
            print("[WARN] thr_mode=json ma non trovo soglia nel JSON; uso fixed=0.5")
            thr_eff = 0.5
        else:
            thr_eff = thr_json
    elif args.thr_mode == "fixed":
        thr_eff = float(args.thr)
    elif args.thr_mode == "bestf1":
        thr_eff, bestf1, p_at, r_at = pick_thr_by_f1(y, scores)
        print(f"[BESTF1] thr={thr_eff:.6f}  F1={bestf1:.4f}  P={p_at:.4f}  R={r_at:.4f}")
    elif args.thr_mode == "fpr":
        # stima sulla coda dei KNOWN (y==0)
        thr_eff = pick_thr_by_target_fpr(scores[y==0], target_fpr=args.target_fpr)
    else:
        raise ValueError(args.thr_mode)

    # Metriche al thr scelto
    M = metrics_at_thr(y, scores, thr_eff)

    # Curve per TPR@FPR e FPR@TPR
    thrC, tprC, fprC, precC, recC = compute_curves(y, scores)
    # TPR @ FPR<=0.05 (massimo tpr sotto vincolo)
    mask = fprC <= 0.05 + 1e-12
    tpr_at_fpr5 = float(tprC[mask].max()) if mask.any() else float("nan")
    # FPR @ TPR>=0.95 (min fpr che raggiunge almeno 95% tpr)
    mask = tprC >= 0.95 - 1e-12
    fprs = fprC[mask]
    fpr_at_tpr95 = float(fprs.min()) if mask.any() else float("nan")

    print("\n================ OPEN-SET DETECTION (QSCO) ================")
    print(f"AUROC (novelty as positive): {auroc:.4f}")
    print(f"AUPRC (novelty as positive): {auprc:.4f}")
    print(f"F1 (thr={thr_eff:.6f}):      {M['f1']:.4f}")
    print(f"Precision:                   {M['precision']:.4f}")
    print(f"Recall (TPR):                {M['recall']:.4f}")
    print(f"Specificity (TNR):           {M['specificity']:.4f}")
    print(f"Balanced Accuracy:           {M['bacc']:.4f}")
    print(f"TPR @ FPR<=0.05:             {tpr_at_fpr5:.4f}")
    print(f"FPR @ TPR>=0.95:             {fpr_at_tpr95:.4f}")
    print("Confusion matrix [ [tn, fp], [fn, tp] ]:")
    print(f"[[{M['tn']:5d} {M['fp']:5d}], [{M['fn']:5d} {M['tp']:5d}]]")

if __name__ == "__main__":
    main()
