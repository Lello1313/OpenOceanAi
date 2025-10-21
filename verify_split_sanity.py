#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verifica integrità e coerenza dei file .pt usati nei run QSCO / Classico.
"""

import os, torch

emb_dir = "outputs/embeddings_open_rec_u-1_8_7_5_0_3"

def check_exists(fname):
    p = os.path.join(emb_dir, fname)
    ok = os.path.exists(p)
    print(f"[{'OK' if ok else 'MISS'}] {fname}")
    return ok

def load_safe(fname):
    return torch.load(os.path.join(emb_dir, fname), map_location="cpu")

print(f"\n=== Sanity check on {emb_dir} ===")

# --- presenza file ---
files = ["E_calib.pt","E_test.pt","y_calib.pt","y_test.pt","y_open.pt"]
all_present = all(check_exists(f) for f in files)

if not all_present:
    print("❌ File mancanti — impossibile proseguire.\n")
    raise SystemExit(1)

# --- caricamento ---
E_calib = load_safe("E_calib.pt").float()
E_test  = load_safe("E_test.pt").float()
y_calib = load_safe("y_calib.pt").long()
y_test  = load_safe("y_test.pt").long()
y_open  = load_safe("y_open.pt").long()

print(f"\nShapes:")
print(f"E_calib {tuple(E_calib.shape)} | E_test {tuple(E_test.shape)}")
print(f"y_calib {tuple(y_calib.shape)} | y_test {tuple(y_test.shape)} | y_open {tuple(y_open.shape)}")

# --- 1) calib vs test distinti ---
# --- 1) calib vs test distinti ---
if E_calib.shape == E_test.shape:
    identical = torch.allclose(E_calib, E_test)
else:
    identical = False  # shape diverse ⇒ sicuramente non identici
print("✅ E_calib e E_test distinti" if not identical else "❌ Leakage: E_calib == E_test")


# --- 2) y_calib non deve contenere -1 ---
if (y_calib == -1).any():
    print("❌ y_calib contiene -1 (unseen in calibrazione)")
else:
    print("✅ y_calib contiene solo classi viste")

# --- 3) coerenza open vs test ---
open_from_ytest = (y_test == -1).long()
eq = torch.equal(y_open, open_from_ytest)
ratio_open = float(y_open.sum()) / len(y_open)
print(f"y_open positivi = {y_open.sum()} / {len(y_open)} ({ratio_open*100:.2f}%)")

if eq:
    print("✅ y_open = 1{y_test==-1} (strictly-unseen semantics)")
else:
    # se differisce, verifica se open = classi non in calib
    calib_classes = set(y_calib.unique().tolist())
    open_from_calib = (~torch.isin(y_test, torch.tensor(list(calib_classes)))).long()
    if torch.equal(y_open, open_from_calib):
        print("✅ y_open = 1{y_test ∉ classi_calib} (open-wrt-calib semantics)")
    else:
        print("⚠️  y_open non coincide né con strictly-unseen né con open-wrt-calib")

# --- 4) riepilogo classi ---
print(f"\nClassi viste in calib: {sorted(set(y_calib.tolist()))}")
print(f"Classi in test:        {sorted(set(y_test.tolist()))}")
print(f"Totale open positives: {int(y_open.sum())}")

print("\n=== Sanity check completed ===")
