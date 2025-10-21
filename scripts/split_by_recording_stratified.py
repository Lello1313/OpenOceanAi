#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Group-by-Recording Stratified Split (70/15/15) with No Leakage

- Ensures that ALL clips belonging to the same recording end up in the SAME split.
- Drops classes that do not have enough distinct recordings to support a 3-way split.
- Two assignment strategies:
    * quasi-exhaustive search when #groups <= max_bruteforce_groups (3^G combinations)
    * greedy assignment otherwise

INPUT CSV requirements (minimal):
    - filename column (default: "filename"), e.g. "Dredger/80__04_10_12_adricristuy_seg1.wav"
    - label column (default: "category")

USAGE (example):
    python split_by_recording_stratified.py \
        --input_csv meta/ships_segments.csv \
        --output_csv outputs/ships_segments_grouped.csv \
        --filename_col filename \
        --label_col category \
        --train_pct 0.70 --val_pct 0.15 --test_pct 0.15 \
        --min_groups_per_class 3 \
        --seed 42

Notes
-----
- The script infers `recording_id` from `filename` by removing the trailing segment
  suffix like "_seg12.wav". You may customize the regex via --recording_regex.
- Classes with fewer than `min_groups_per_class` distinct recordings are REMOVED
  to avoid leakage and ensure a meaningful split.
- Outputs:
    * a consolidated CSV with an added 'split' column
    * three CSVs with suffixes _train/_val/_test
"""
import os, re, sys, argparse, itertools, numpy as np, pandas as pd
from collections import Counter

def parse_args():
    ap = argparse.ArgumentParser("Group-by-Recording stratified split 70/15/15 (no leakage)")
    ap.add_argument("--input_csv", required=True, help="Path to input metadata CSV")
    ap.add_argument("--output_csv", required=True, help="Path to output CSV (with split column)")
    ap.add_argument("--filename_col", default="filename")
    ap.add_argument("--label_col", default="category")
    ap.add_argument("--train_pct", type=float, default=0.70)
    ap.add_argument("--val_pct",   type=float, default=0.15)
    ap.add_argument("--test_pct",  type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_bruteforce_groups", type=int, default=12,
                    help="Above this threshold use greedy (combinations = 3^G)")
    ap.add_argument("--min_groups_per_class", type=int, default=3,
                    help="Minimum number of distinct recordings (groups) required per class; otherwise class is dropped")
    ap.add_argument("--recording_regex", default=r'^(.+?)_seg\d+\.(?:wav|flac|mp3)$',
                    help=r"Regex with one capturing group to extract recording_id from filename stem. Default strips '_segN' suffix.")
    return ap.parse_args()

def ensure_dir_for_file(p):
    d = os.path.dirname(os.path.abspath(p))
    if d:
        os.makedirs(d, exist_ok=True)

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def extract_recording_id(fname, pattern):
    st = stem(fname)
    m = re.match(pattern, st)
    if m:
        return m.group(1)
    # fallback: use stem as recording id (will still group all identical stems)
    return st

def score_assignment(assign, group_class_counts, target_totals_per_split, class_list):
    """Evaluate how well the (group->split) assignment matches target totals and class balance."""
    totals = {sp:0 for sp in ["train","val","test"]}
    class_totals_split = {sp: Counter() for sp in ["train","val","test"]}
    for g, sp in assign.items():
        nclips = sum(group_class_counts[g].values())
        totals[sp] += nclips
        class_totals_split[sp].update(group_class_counts[g])

    # L1 diff on total clips per split
    diff_tot = sum(abs(totals[sp] - target_totals_per_split[sp]) for sp in ["train","val","test"])

    # L1 diff on per-class distributions per split (target: proportional to split size, uniform over classes)
    diff_cls = 0.0
    total_target = sum(target_totals_per_split.values())
    for sp in ["train","val","test"]:
        ts = totals[sp]
        if ts == 0:
            diff_cls += 1e6  # heavy penalty for empty split
            continue
        for c in class_list:
            p_obs = class_totals_split[sp][c] / ts
            p_tgt = (target_totals_per_split[sp] / total_target) * (1.0 / max(1, len(class_list)))
            diff_cls += abs(p_obs - p_tgt)

    # penalty for classes totally absent in a split (soft)
    absent_pen = 0.0
    for c in class_list:
        miss = 0
        for sp in ["train","val","test"]:
            if class_totals_split[sp][c] == 0:
                miss += 1
        # tolerate absence in 1 split; penalize beyond that
        if miss >= 2:
            absent_pen += 100.0 * (miss - 1)
    return diff_tot + diff_cls + absent_pen, totals, class_totals_split

def greedy_assign(groups, group_sizes, group_class_counts, targets, class_list, rng):
    order = sorted(groups, key=lambda g: group_sizes[g], reverse=True)
    assign = {}
    for g in order:
        best_sp, best_sc = None, None
        for sp in ["train","val","test"]:
            trial = assign.copy()
            trial[g] = sp
            sc, _, _ = score_assignment(trial, group_class_counts, targets, class_list)
            if best_sc is None or sc < best_sc:
                best_sc, best_sp = sc, sp
        if best_sp is None:
            best_sp = rng.choice(["train","val","test"])
        assign[g] = best_sp
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
    assert abs(args.train_pct + args.val_pct + args.test_pct - 1.0) < 1e-6, "train/val/test pct must sum to 1.0"
    rng = np.random.RandomState(args.seed)

    df = pd.read_csv(args.input_csv)
    for col in [args.filename_col, args.label_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}'. Available: {list(df.columns)}")

    # Build recording_id
    rec_ids = df[args.filename_col].apply(lambda x: extract_recording_id(str(x), args.recording_regex))
    df = df.copy()
    df["recording_id"] = rec_ids

    # Filter classes with insufficient number of distinct recordings
    keep_classes = []
    dropped = []
    for c, sub in df.groupby(args.label_col):
        n_rec = sub["recording_id"].nunique()
        if n_rec >= args.min_groups_per_class:
            keep_classes.append(c)
        else:
            dropped.append((c, n_rec))
    if dropped:
        print("[WARN] Dropping classes with too few distinct recordings (class, #recordings):")
        for c, n in dropped:
            print(f"  - {c}: {n}")
    df = df[df[args.label_col].isin(keep_classes)].reset_index(drop=True)

    if df.empty:
        raise RuntimeError("All classes were dropped due to insufficient recordings. Adjust --min_groups_per_class or provide more data.")

    # Summary
    print(f"[INFO] Total clips after filtering = {len(df)} | classes kept = {len(keep_classes)} | recordings = {df['recording_id'].nunique()}")

    # Build group-level (recording-level) stats
    groups = sorted(df["recording_id"].unique())
    class_list = sorted(df[args.label_col].unique())
    group_class_counts = {}
    group_sizes = {}
    for g, sub in df.groupby("recording_id"):
        cnt = Counter(sub[args.label_col].values.tolist())
        group_class_counts[g] = cnt
        group_sizes[g] = int(cnt.total())

    total_clips = len(df)
    targets = {
        "train": int(round(args.train_pct * total_clips)),
        "val":   int(round(args.val_pct   * total_clips)),
        "test":  int(round(args.test_pct  * total_clips)),
    }
    # Fix rounding residual
    diff = total_clips - sum(targets.values())
    if diff != 0:
        biggest = max(targets, key=targets.get)
        targets[biggest] += diff

    print(f"[INFO] Target #clips per split: {targets}")

    # Assign groups to splits
    if len(groups) <= args.max_bruteforce_groups:
        print(f"[INFO] Using quasi-exhaustive search (3^{len(groups)} combinations)")
        assign, reached = brute_force_assign(groups, group_class_counts, targets, class_list)
    else:
        print(f"[INFO] Using greedy heuristic (groups={len(groups)} > {args.max_bruteforce_groups})")
        assign = greedy_assign(groups, group_sizes, group_class_counts, targets, class_list, rng)
        sc, reached, _ = score_assignment(assign, group_class_counts, targets, class_list)
        print(f"[INFO] Greedy score = {sc:.3f} | reached totals = {reached}")

    # Apply split back to clip-level rows
    df["split"] = df["recording_id"].map(assign)

    # Final report
    print("\n=== CLIP DISTRIBUTION BY CLASS × SPLIT ===")
    summary = pd.crosstab(df[args.label_col], df["split"]).reindex(class_list).fillna(0).astype(int)
    print(summary.to_string())

    # Save outputs
    ensure_dir_for_file(args.output_csv)
    df.to_csv(args.output_csv, index=False)
    print(f"\n[OK] Saved master CSV with splits: {args.output_csv}")

    base = os.path.splitext(args.output_csv)[0]
    for sp in ["train","val","test"]:
        p = f"{base}_{sp}.csv"
        df[df["split"]==sp].to_csv(p, index=False)
        print(f"[OK] {sp}.csv -> {p}  (#clips={len(df[df['split']==sp])})")

    # Totals per split
    print("\n=== TOTAL CLIPS PER SPLIT ===")
    print({sp:int(v) for sp,v in df['split'].value_counts().to_dict().items()})

if __name__ == "__main__":
    main()
