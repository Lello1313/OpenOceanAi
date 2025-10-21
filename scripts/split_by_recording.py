# scripts/split_by_groups_stratified.py
# Split 70/15/15 "group-aware" e globalmente stratificato per classi.
# Nessun leakage: ciascun group (fold) finisce in un solo split.
# Con pochi gruppi (<=12) prova un'assegnazione quasi esaustiva; altrimenti usa greedy.

import os, sys, argparse, itertools, numpy as np, pandas as pd
from collections import Counter, defaultdict

def parse_args():
    ap = argparse.ArgumentParser("Group-aware stratified split 70/15/15 (no leakage)")
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--group_col", default="fold")
    ap.add_argument("--label_col", default="category")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--train_pct", type=float, default=0.70)
    ap.add_argument("--val_pct",   type=float, default=0.15)
    ap.add_argument("--test_pct",  type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_bruteforce_groups", type=int, default=12,
                    help="sopra questa soglia usa greedy (combinazioni = 3^G)")
    return ap.parse_args()

def ensure_dir(p):
    d = os.path.dirname(os.path.abspath(p))
    os.makedirs(d, exist_ok=True)

def score_assignment(assign, group_class_counts, target_totals_per_split, class_list):
    """Valuta quanto l'assegnazione (group->split) rispetta i target 70/15/15 per clip e distribuzioni di classe."""
    # conteggi clip per split
    totals = {sp:0 for sp in ["train","val","test"]}
    class_totals_split = {sp: Counter() for sp in ["train","val","test"]}
    for g, sp in assign.items():
        nclips = sum(group_class_counts[g].values())
        totals[sp] += nclips
        class_totals_split[sp].update(group_class_counts[g])

    # L1 diff sui totali clip per split
    diff_tot = sum(abs(totals[sp] - target_totals_per_split[sp]) for sp in ["train","val","test"])

    # L1 diff sulla distribuzione di classe per split (normalizzata per il totale split)
    diff_cls = 0.0
    for sp in ["train","val","test"]:
        ts = totals[sp]
        if ts == 0:
            diff_cls += 1e6  # penalità forte per split vuoti
            continue
        for c in class_list:
            p_obs = class_totals_split[sp][c] / ts
            p_tgt = target_totals_per_split[sp] / sum(target_totals_per_split.values()) * (1.0/len(class_list))  # target "flat" per classe
            # nota: se vuoi pesare per la freq globale, si può cambiare p_tgt
            diff_cls += abs(p_obs - p_tgt)

    # Penalità per classi assenti completamente in qualche split (se possibile evitarlo)
    absent_pen = 0.0
    global_cls_tot = Counter()
    for sp in ["train","val","test"]:
        global_cls_tot.update(class_totals_split[sp])
    for c in class_list:
        present = {sp: (class_totals_split[sp][c] > 0) for sp in ["train","val","test"]}
        missing_splits = 3 - sum(present.values())
        # se la classe è rarissima è possibile che non stia in tutti gli split; penalizza leggero
        absent_pen += 100.0 * max(0, missing_splits - 1)  # tollera mancanza in 1 split senza penalità enorme

    return diff_tot + diff_cls + absent_pen, totals, class_totals_split

def greedy_assign(groups, group_sizes, group_class_counts, targets, class_list, rng):
    # Ordina per grandezza decrescente
    order = sorted(groups, key=lambda g: group_sizes[g], reverse=True)
    assign = {}
    split_tot = {sp: 0 for sp in ["train","val","test"]}

    for g in order:
        best_sp, best_sc = None, None
        for sp in ["train","val","test"]:
            trial = assign.copy()
            trial[g] = sp
            sc, _, _ = score_assignment(trial, group_class_counts, targets, class_list)
            if best_sc is None or sc < best_sc:
                best_sc, best_sp = sc, sp
        # tie-break random
        if best_sp is None:
            best_sp = rng.choice(["train","val","test"])
        assign[g] = best_sp
        split_tot[best_sp] += group_sizes[g]
    return assign

def brute_force_assign(groups, group_class_counts, targets, class_list):
    best = None
    for choices in itertools.product(["train","val","test"], repeat=len(groups)):
        assign = {g: sp for g, sp in zip(groups, choices)}
        sc, totals, _ = score_assignment(assign, group_class_counts, targets, class_list)
        if best is None or sc < best[0]:
            best = (sc, assign, totals)
    return best[1], best[2]

def main():
    args = parse_args()
    assert abs(args.train_pct + args.val_pct + args.test_pct - 1.0) < 1e-6
    rng = np.random.RandomState(args.seed)

    df = pd.read_csv(args.input_csv)
    for col in [args.group_col, args.label_col, args.filename_col]:
        if col not in df.columns:
            raise ValueError(f"Colonna '{col}' mancante. Colonne: {list(df.columns)}")

    # Stat iniziali
    print(f"[INFO] Tot clip={len(df)} | classi={df[args.label_col].nunique()} | gruppi={df[args.group_col].nunique()}")

    # Costruisci conteggi per gruppo
    groups = sorted(df[args.group_col].unique())
    class_list = sorted(df[args.label_col].unique())
    group_class_counts = {}
    group_sizes = {}
    for g, sub in df.groupby(args.group_col):
        cnt = Counter(sub[args.label_col].values.tolist())
        group_class_counts[g] = cnt
        group_sizes[g] = int(cnt.total())

    total_clips = len(df)
    targets = {
        "train": int(round(args.train_pct * total_clips)),
        "val":   int(round(args.val_pct   * total_clips)),
        "test":  int(round(args.test_pct  * total_clips)),
    }
    # correzione piccole discrepanze
    diff = total_clips - sum(targets.values())
    if diff != 0:
        # aggiusta il maggiore
        k = max(targets, key=lambda s: targets[s])
        targets[k] += diff

    print(f"[INFO] Target clip per split: {targets}")

    # Assegnazione
    if len(groups) <= args.max_bruteforce_groups:
        print(f"[INFO] Uso ricerca quasi esaustiva (3^{len(groups)} combinazioni)")
        assign, reached = brute_force_assign(groups, group_class_counts, targets, class_list)
    else:
        print(f"[INFO] Uso greedy (gruppi={len(groups)} > {args.max_bruteforce_groups})")
        assign = greedy_assign(groups, group_sizes, group_class_counts, targets, class_list, rng)
        # ricalcola totali raggiunti
        _, reached, _ = score_assignment(assign, group_class_counts, targets, class_list)

    # Applica split
    df = df.copy()
    df["split"] = df[args.group_col].map(assign)

    # Report finale
    print("\n=== DISTRIBUZIONE PER CLASSE (clip) ===")
    summary = pd.crosstab(df[args.label_col], df["split"]).reindex(class_list).fillna(0).astype(int)
    print(summary.to_string())

    # Salva
    ensure_dir(args.output_csv)
    df.to_csv(args.output_csv, index=False)
    print(f"\n[OK] CSV salvato in: {args.output_csv}")

    base = os.path.splitext(args.output_csv)[0]
    for sp in ["train","val","test"]:
        p = f"{base}_{sp}.csv"
        df[df["split"]==sp].to_csv(p, index=False)
        print(f"[OK] {sp}.csv -> {p}  (#clip={len(df[df['split']==sp])})")

    # Statistiche clip finali per split
    print("\n=== TOTALE CLIP PER SPLIT ===")
    print({sp:int(v) for sp,v in df["split"].value_counts().to_dict().items()})
    print(f"Raggiunti (clip): {reached}")

if __name__ == "__main__":
    main()
