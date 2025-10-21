#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Costruisce calib_seen.csv e test_mixed.csv a partire dagli output di eval_few.py
(support.csv, query.csv) e dal meta_csv completo.

Vincoli:
- Le classi UNSEEN sono dedotte da support/query di eval_few.py.
- calib_seen.csv = solo classi SEEN (cioè non unseen).
- test_mixed.csv = UNSEEN (da query.csv) + SEEN campionati che NON siano usati in training né in calibrazione.
- Evita leakage per registrazione (recording_id) tra calib e test_mixed e tra train e test_mixed.
- Funziona sia con meta che hanno una colonna 'split' (train/val/test) sia senza (allora usa una regex per ricavare recording_id e partiziona).
"""

import argparse
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Optional, Set

def infer_recording_id(name: str, pattern: Optional[re.Pattern], fallback_col_ok: bool) -> str:
    if fallback_col_ok:
        return name
    if pattern is None:
        # fallback: intero nome senza estensione
        base = Path(name).stem
        return base
    m = pattern.match(Path(name).stem)
    if m:
        return m.group(1)
    # se non matcha, usa lo stem
    return Path(name).stem

def ensure_cols(df: pd.DataFrame, filename_col: str, label_col: str):
    missing = [c for c in [filename_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne richieste nel meta: {missing} (presenti: {list(df.columns)[:10]}...)")

def load_unseen_classes(support_csv: Path, query_csv: Path, label_col: str) -> List[str]:
    sup = pd.read_csv(support_csv)
    qry = pd.read_csv(query_csv)
    if label_col not in sup.columns or label_col not in qry.columns:
        raise ValueError(f"Colonna label '{label_col}' non trovata in support/query.")
    unseen = sorted(set(sup[label_col].unique()).union(set(qry[label_col].unique())))
    if len(unseen) == 0:
        raise ValueError("Nessuna classe trovata in support/query: impossibile dedurre UNSEEN.")
    return unseen

def stratified_sample_by_class(df: pd.DataFrame, label_col: str, per_class: Optional[int]=None, frac: Optional[float]=None, seed: int=42):
    rng = np.random.default_rng(seed)
    parts = []
    classes = df[label_col].unique()
    for c in classes:
        sub = df[df[label_col]==c]
        if len(sub)==0:
            continue
        if per_class is not None:
            k = min(per_class, len(sub))
            idx = rng.choice(sub.index.values, size=k, replace=False)
            parts.append(sub.loc[idx])
        elif frac is not None:
            k = max(1, int(np.floor(len(sub)*frac)))
            idx = rng.choice(sub.index.values, size=k, replace=False)
            parts.append(sub.loc[idx])
        else:
            parts.append(sub)
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", type=str, required=True, help="CSV completo (tutte le clip)")
    ap.add_argument("--support_csv", type=str, required=True, help="output di eval_few.py")
    ap.add_argument("--query_csv", type=str, required=True, help="output di eval_few.py")
    ap.add_argument("--out_dir", type=str, required=True, help="cartella per salvare calib_seen.csv e test_mixed.csv")

    ap.add_argument("--filename_col", type=str, default="filename")
    ap.add_argument("--label_col", type=str, default="category")
    ap.add_argument("--split_col", type=str, default="split",
                    help="Se presente, usata per separare train/val/test; altrimenti si ignora.")
    ap.add_argument("--train_values", type=str, default="train",
                    help="Valori del train nella colonna split, separati da virgola (es. 'train,training').")
    ap.add_argument("--eval_values", type=str, default="val,test",
                    help="Valori per valutazione nella colonna split, separati da virgola (es. 'val,test').")

    ap.add_argument("--recording_regex", type=str,
                    default=r"^(.*?)(?:[-_]?seg\d+|_\d+|\d+)$",
                    help="Regex per ricavare recording_id da filename (stem). group(1) = id.")
    ap.add_argument("--known_per_class", type=int, default=100,
                    help="Numero massimo di clip SEEN da includere in test_mixed per classe (campionati).")
    ap.add_argument("--unseen_limit", type=int, default=0,
                    help="Limite massimo di clip UNSEEN in test_mixed (0 = nessun limite; usa tutte le query unseen).")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--calib_frac_per_class", type=float, default=1.0,
                    help="Quota delle clip SEEN (in train) da usare in calibrazione (0<frac<=1).")
    ap.add_argument("--dedupe_by_recording", action="store_true",
                    help="Se non c'è split, impone separazione hard per recording tra calib e test_mixed.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta_csv)
    ensure_cols(meta, args.filename_col, args.label_col)

    # deduce unseen classes
    unseen_classes = load_unseen_classes(Path(args.support_csv), Path(args.query_csv), args.label_col)
    unseen_set: Set[str] = set(unseen_classes)

    # SEEN = tutte le altre
    all_classes = set(meta[args.label_col].unique())
    seen_classes = sorted(list(all_classes - unseen_set))
    if len(seen_classes) == 0:
        raise ValueError("Non restano classi SEEN dopo aver rimosso le UNSEEN.")

    # prepara recording_id
    has_split = args.split_col in meta.columns
    rec_pat = re.compile(args.recording_regex) if args.recording_regex else None
    meta = meta.copy()
    meta["recording_id"] = meta[args.filename_col].apply(lambda x: infer_recording_id(str(x), rec_pat, False))

    # filtri base
    meta_seen = meta[meta[args.label_col].isin(seen_classes)].copy()
    meta_unseen = meta[meta[args.label_col].isin(unseen_classes)].copy()

    rng = np.random.default_rng(args.seed)

    if has_split:
        # ---- Modalità con split ----
        train_vals = {s.strip().lower() for s in args.train_values.split(",") if s.strip()}
        eval_vals  = {s.strip().lower() for s in args.eval_values.split(",") if s.strip()}

        split_norm = meta[args.split_col].astype(str).str.lower()
        meta_seen_train = meta_seen[split_norm.isin(train_vals)].copy()
        meta_seen_eval  = meta_seen[split_norm.isin(eval_vals)].copy()

        # Calibrazione: solo SEEN e solo train
        calib_seen = stratified_sample_by_class(
            meta_seen_train, args.label_col,
            per_class=None,
            frac=args.calib_frac_per_class,
            seed=args.seed
        )

        # Test: unseen da query.csv (unione con meta_unseen per eventuale superset intersezione) + seen SOLO da eval split
        qry = pd.read_csv(args.query_csv)
        if args.label_col not in qry.columns or args.filename_col not in qry.columns:
            raise ValueError("query.csv deve avere le colonne label e filename.")
        qry["recording_id"] = qry[args.filename_col].apply(lambda x: infer_recording_id(str(x), rec_pat, False))
        # Unseen test = intersezione tra meta_unseen e i filename presenti in query.csv (robusto a path diversi)
        test_unseen = meta_unseen.merge(
            qry[[args.filename_col, args.label_col, "recording_id"]],
            on=[args.filename_col, args.label_col, "recording_id"],
            how="inner"
        )

        if args.unseen_limit and len(test_unseen) > args.unseen_limit:
            # campiona test_unseen in modo bilanciato per classe
            test_unseen = stratified_sample_by_class(
                test_unseen, args.label_col, per_class=None,
                frac=args.unseen_limit/len(test_unseen), seed=args.seed
            )

        # Seen per test: SOLO da split di valutazione e con recording_id che non compaiono in calib
        calib_rec = set(calib_seen["recording_id"].unique())
        cand_seen_test = meta_seen_eval[~meta_seen_eval["recording_id"].isin(calib_rec)].copy()

        test_seen = stratified_sample_by_class(
            cand_seen_test, args.label_col, per_class=args.known_per_class, frac=None, seed=args.seed
        )

        test_mixed = pd.concat([test_unseen, test_seen], axis=0).reset_index(drop=True)

    else:
        # ---- Modalità senza split ----
        # Strategia: partizione per recording_id delle SEEN in due insiemi disgiunti: CALIB e TEST_SEEN
        # 1) per ogni classe SEEN, raccogli recordings e splitta 70/30 (random) in calib/test (parametrizzabile se vuoi)
        CALIB_PCT = 0.7

        parts_calib = []
        parts_seen_test = []

        for c in seen_classes:
            sub = meta_seen[meta_seen[args.label_col]==c]
            recs = sub["recording_id"].unique()
            rng.shuffle(recs)
            k = int(np.floor(len(recs)*CALIB_PCT))
            rec_calib = set(recs[:k])
            rec_test  = set(recs[k:])

            parts_calib.append(sub[sub["recording_id"].isin(rec_calib)])
            parts_seen_test.append(sub[sub["recording_id"].isin(rec_test)])

        calib_seen = pd.concat(parts_calib, axis=0).reset_index(drop=True)
        cand_seen_test = pd.concat(parts_seen_test, axis=0).reset_index(drop=True)

        # Unseen test = intersezione con query.csv
        qry = pd.read_csv(args.query_csv)
        if args.label_col not in qry.columns or args.filename_col not in qry.columns:
            raise ValueError("query.csv deve avere le colonne label e filename.")
        qry["recording_id"] = qry[args.filename_col].apply(lambda x: infer_recording_id(str(x), rec_pat, False))
        test_unseen = meta_unseen.merge(
            qry[[args.filename_col, args.label_col, "recording_id"]],
            on=[args.filename_col, args.label_col, "recording_id"],
            how="inner"
        )

        if args.unseen_limit and len(test_unseen) > args.unseen_limit:
            test_unseen = stratified_sample_by_class(
                test_unseen, args.label_col, per_class=None,
                frac=args.unseen_limit/len(test_unseen), seed=args.seed
            )

        # Enforce dedupe_by_recording se richiesto
        if args.dedupe_by_recording:
            calib_rec = set(calib_seen["recording_id"].unique())
            cand_seen_test = cand_seen_test[~cand_seen_test["recording_id"].isin(calib_rec)]

        test_seen = stratified_sample_by_class(
            cand_seen_test, args.label_col, per_class=args.known_per_class, frac=None, seed=args.seed
        )
        test_mixed = pd.concat([test_unseen, test_seen], axis=0).reset_index(drop=True)

    # Salvataggi
    calib_out = out_dir / "calib_seen.csv"
    test_out  = out_dir / "test_mixed.csv"
    calib_seen.to_csv(calib_out, index=False)
    test_mixed.to_csv(test_out, index=False)

    # Log essenziale
    def by_class_counts(df: pd.DataFrame, name: str):
        cnt = df[args.label_col].value_counts().sort_index()
        print(f"\n[{name}] #clip per classe:")
        print(cnt.to_string())
        print(f"Totale {name}: {len(df)}")

    print(f"UNSEEN = {unseen_classes}")
    print(f"SEEN   = {seen_classes}")

    by_class_counts(calib_seen, "CALIB_SEEN")
    by_class_counts(test_mixed, "TEST_MIXED")

    # check recording leakage
    leak_rec = set(calib_seen["recording_id"]).intersection(set(test_mixed["recording_id"]))
    if len(leak_rec) > 0:
        print(f"\n[WARN] Trovate {len(leak_rec)} recording_id in comune tra calibrazione e test_mixed.")
        print("Esempi:", list(sorted(leak_rec))[:10])
    else:
        print("\n[OK] Nessuna recording_id in comune tra calibrazione e test_mixed.")

    # piccola etichetta open/known per ispezione rapida (non usata qui, ma utile a vista)
    tm = test_mixed.copy()
    tm["open_gt"] = np.where(tm[args.label_col].isin(unseen_set), 1, 0)
    tm_sample = tm[[args.filename_col, args.label_col, "recording_id", "open_gt"]].head(20)
    print("\n[Esempio TEST_MIXED (prime 20 righe)]:")
    print(tm_sample.to_string(index=False))


if __name__ == "__main__":
    main()
