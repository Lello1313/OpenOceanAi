#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split_openset_preprocess.py
Crea gli split per una pipeline OSAD coerente:
- train_seen.csv: 9 classi viste (senza 1 recording per classe)
- holdout_seen.csv: 1 recording per classe tenuto fuori dal training
- unseen_pool.csv: classi NON in seen o etichette unknown/-1
- test_mixed.csv: holdout_seen + subset di unseen_pool (parametrizzabile)

Assunzioni di default:
- meta ha colonne: filename, category
- 9 classi seen prese da seen_classes.json (se fornito) oppure da --seen_names,
  altrimenti dai 9 label più frequenti esclusi gli unknown.
"""

import argparse
import json
import os
import re
import sys
from typing import List, Set

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Preprocessing split per open-set anomaly detection (ShipSEAR).")
    p.add_argument("--meta_csv", type=str, required=True, help="Meta CSV di partenza (es. meta/ships_segments.csv)")
    p.add_argument("--out_dir", type=str, default="outputs/openset_splits", help="Cartella output")
    p.add_argument("--filename_col", type=str, default="filename", help="Colonna filename nel meta")
    p.add_argument("--label_col", type=str, default="category", help="Colonna label/classe nel meta")
    p.add_argument("--recording_regex", type=str,
                   default=r'^(.*?)(?:[-_]?seg\d+|_\d+|\d+)$',
                   help="Regex per derivare recording_id dal filename (usa un gruppo di cattura per l'id)")
    p.add_argument("--seed", type=int, default=42, help="Seed per scelte random")
    p.add_argument("--seen_json", type=str, default="", help="seen_classes.json (con chiave 'seen_classes')")
    p.add_argument("--seen_names", type=str, default="",
                   help="Lista di classi seen separate da virgola (override). Es: \"Motorboat,Mussel boat,...\"")
    p.add_argument("--num_seen", type=int, default=9, help="Numero di classi seen da usare se non fornite")
    p.add_argument("--unknown_aliases", type=str, default="-1,unknown,UNKNOWN,Unknown",
                   help="Valori che indicano unknown; separati da virgola")
    p.add_argument("--min_recordings_per_class", type=int, default=2,
                   help="Min # di registrazioni richieste per poter trattenere 1 holdout. Se <2, nessun holdout.")
    p.add_argument("--unseen_sample_per_class", type=int, default=0,
                   help="Se >0, limita in test_mixed il # max di clip unseen per classe a questo valore (0=nessun limite)")
    p.add_argument("--disable_test_mixed", action="store_true", help="Se passato, non crea test_mixed.csv")
    return p.parse_args()


def load_seen_names(args, meta_labels: List[str]) -> List[str]:
    # Priority: --seen_names > --seen_json > top-N frequent (excluding unknown)
    if args.seen_names.strip():
        seen = [s.strip() for s in args.seen_names.split(",") if s.strip()]
        return seen

    if args.seen_json and os.path.isfile(args.seen_json):
        with open(args.seen_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "seen_classes" in obj:
            return list(obj["seen_classes"])

    # fallback: top-N frequenti esclusi unknown
    return meta_labels[: args.num_seen]


def main():
    args = parse_args()
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    # Carica meta
    if not os.path.isfile(args.meta_csv):
        print(f"[ERR] meta_csv non trovato: {args.meta_csv}", file=sys.stderr)
        sys.exit(2)
    df = pd.read_csv(args.meta_csv)
    if args.filename_col not in df.columns or args.label_col not in df.columns:
        print(f"[ERR] Colonne mancanti. Servono: {args.filename_col} e {args.label_col}", file=sys.stderr)
        sys.exit(2)

    # Casting label a string per robustezza
    df[args.label_col] = df[args.label_col].astype(str)

    # Deriva recording_id dal filename
    pat = re.compile(args.recording_regex)
    def get_rec(x: str) -> str:
        m = pat.match(str(x))
        return m.group(1) if m else str(x)

    df["recording_id"] = df[args.filename_col].astype(str).map(get_rec)

    # Determina i label unknown
    unk_set: Set[str] = set([u.strip() for u in args.unknown_aliases.split(",") if u.strip()])
    # Ordina le classi per frequenza (escludendo unknown)
    freq = (
        df[~df[args.label_col].isin(unk_set)][args.label_col]
        .value_counts()
        .rename_axis("label").reset_index(name="count")
    )
    meta_labels_sorted = freq["label"].tolist()

    # Determina le SEEN
    seen_names = load_seen_names(args, meta_labels_sorted)
    if len(seen_names) < 1:
        print("[ERR] Nessuna classe seen determinata. Specifica --seen_names o --seen_json.", file=sys.stderr)
        sys.exit(2)

    # Report classi
    print(f"[INFO] Seen ({len(seen_names)}): {seen_names}")
    unseen_names = sorted(list(set(df[args.label_col].unique()) - set(seen_names)))
    # unseen pool includerà anche gli alias unknown (già in df)
    print(f"[INFO] Potenziali unseen (ex meta): {unseen_names[:20]}{' ...' if len(unseen_names)>20 else ''}")

    # Per ogni classe seen, scegli 1 recording_id da tenere fuori
    holdout_recs = []
    per_class_info = []
    for c in seen_names:
        sub = df[df[args.label_col] == c]
        n_clips = len(sub)
        recs = sub["recording_id"].drop_duplicates().tolist()
        n_recs = len(recs)

        chosen = None
        if n_recs >= args.min_recordings_per_class:
            chosen = np.random.choice(recs, 1)[0]
            holdout_recs.append((c, chosen))
        else:
            # non abbastanza registrazioni per trattenere 1 holdout
            chosen = None

        per_class_info.append({
            "class": c,
            "clips_total": int(n_clips),
            "recs_total": int(n_recs),
            "holdout_rec": chosen
        })

    # Costruisci i dataframe finali
    holdout_rec_ids = set([r for (_, r) in holdout_recs])

    seen_mask = df[args.label_col].isin(seen_names)
    # train_seen: seen e recording NON in holdout_rec_ids
    train_seen = df[seen_mask & ~df["recording_id"].isin(holdout_rec_ids)].copy()
    # holdout_seen: seen e recording in holdout_rec_ids
    holdout_seen = df[seen_mask & df["recording_id"].isin(holdout_rec_ids)].copy()

    # unseen_pool: label non in seen OR label in alias unknown
    unseen_mask = (~df[args.label_col].isin(seen_names)) | (df[args.label_col].isin(unk_set))
    unseen_pool = df[unseen_mask].copy()

    # test_mixed: holdout_seen + subset unseen_pool (facoltativo)
    if args.unseen_sample_per_class > 0:
        # Limita # clip per classe nella parte unseen
        sampled_unseen = (
            unseen_pool
            .groupby(args.label_col, group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), args.unseen_sample_per_class), random_state=args.seed))
        )
    else:
        sampled_unseen = unseen_pool

    test_mixed = None
    if not args.disable_test_mixed:
        test_mixed = pd.concat([holdout_seen, sampled_unseen], ignore_index=True)

    # Salva file
    train_seen_path = os.path.join(args.out_dir, "train_seen.csv")
    holdout_seen_path = os.path.join(args.out_dir, "holdout_seen.csv")
    unseen_pool_path = os.path.join(args.out_dir, "unseen_pool.csv")

    train_seen.to_csv(train_seen_path, index=False)
    holdout_seen.to_csv(holdout_seen_path, index=False)
    unseen_pool.to_csv(unseen_pool_path, index=False)

    saved = {
        "train_seen_csv": train_seen_path,
        "holdout_seen_csv": holdout_seen_path,
        "unseen_pool_csv": unseen_pool_path,
    }

    if test_mixed is not None:
        test_mixed_path = os.path.join(args.out_dir, "test_mixed.csv")
        test_mixed.to_csv(test_mixed_path, index=False)
        saved["test_mixed_csv"] = test_mixed_path

    # Riassunto e contatori
    summary = {
        "args": vars(args),
        "seen_names": seen_names,
        "unknown_aliases": sorted(list(unk_set)),
        "n_train_seen": int(len(train_seen)),
        "n_holdout_seen": int(len(holdout_seen)),
        "n_unseen_pool": int(len(unseen_pool)),
        "per_class_info": per_class_info,
        "holdout_recordings": holdout_recs,
        "saved": saved
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Stampa breve
    print("[DONE] Split creati in:", args.out_dir)
    print(f"  train_seen:   {len(train_seen)}  | recs: {train_seen['recording_id'].nunique()}")
    print(f"  holdout_seen: {len(holdout_seen)} | recs: {holdout_seen['recording_id'].nunique()} (1 per classe se possibile)")
    print(f"  unseen_pool:  {len(unseen_pool)}  | classi: {unseen_pool[args.label_col].nunique()}")
    if test_mixed is not None:
        print(f"  test_mixed:   {len(test_mixed)}")


if __name__ == "__main__":
    main()
