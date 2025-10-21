import re, os, argparse
import pandas as pd
import numpy as np

def rec_id(name):
    m = re.match(r"^(.*?)(?:[-_]?seg\d+|_\d+|\d+)$", str(name))
    return m.group(1) if m else str(name)

ap = argparse.ArgumentParser()
ap.add_argument("--pool_csv", required=True)
ap.add_argument("--out_dir", required=True)
ap.add_argument("--k_shot", type=int, required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
meta = pd.read_csv(args.pool_csv)

if "filename" not in meta.columns or "category" not in meta.columns:
    raise SystemExit("unseen_pool.csv deve contenere almeno le colonne: filename, category")

rng = np.random.RandomState(args.seed)
meta = meta.copy()
meta["recording"] = meta["filename"].map(rec_id)

sup_rows, qry_rows = [], []
classes = sorted(meta["category"].unique())

for c in classes:
    dfc = meta[meta["category"] == c].copy()

    # Preferisci SUPPORT da fold 1-4 se la colonna 'fold' esiste
    if "fold" in dfc.columns:
        dfc_14 = dfc[dfc["fold"].isin([1,2,3,4])]
        dfc_5  = dfc[dfc["fold"] == 5]
        if len(dfc_14) >= args.k_shot:
            sup = dfc_14.sample(n=args.k_shot, random_state=args.seed)
        else:
            part = dfc_14.sample(n=min(args.k_shot, len(dfc_14)), random_state=args.seed) if len(dfc_14)>0 else dfc_14.head(0)
            need = args.k_shot - len(part)
            if need > 0 and len(dfc_5)>0:
                part = pd.concat([part, dfc_5.sample(n=min(need, len(dfc_5)), random_state=args.seed)])
            sup = part
        sup_rec = set(sup["recording"].tolist())
        q5  = dfc_5[~dfc_5["recording"].isin(sup_rec)]
        q14 = dfc_14[~dfc_14["recording"].isin(sup_rec)]
        qry = pd.concat([q5, q14], ignore_index=False)
    else:
        # fallback: nessun 'fold' -> prendi k a caso, QUERY = resto, evitando stessi recording del support
        sup = dfc.sample(n=min(args.k_shot, len(dfc)), random_state=args.seed) if len(dfc)>0 else dfc.head(0)
        sup_rec = set(sup["recording"].tolist())
        qry = dfc[~dfc["recording"].isin(sup_rec)]

    sup_rows.append(sup.drop(columns=["recording"]))
    qry_rows.append(qry.drop(columns=["recording"]))

support = pd.concat(sup_rows).reset_index(drop=True) if len(sup_rows) else pd.DataFrame(columns=meta.columns)
query   = pd.concat(qry_rows).reset_index(drop=True) if len(qry_rows) else pd.DataFrame(columns=meta.columns)

sup_path = os.path.join(args.out_dir, "support.csv")
qry_path = os.path.join(args.out_dir, "query.csv")
support.to_csv(sup_path, index=False)
query.to_csv(qry_path, index=False)
print(f"[OK] Saved SUPPORT -> {sup_path}  |  QUERY -> {qry_path}")
