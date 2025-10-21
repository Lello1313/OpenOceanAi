# make_open_split_from_emb.py
import os, argparse, torch, shutil

def load_t(path):
    t = torch.load(path, map_location="cpu")
    return t if isinstance(t, torch.Tensor) else torch.as_tensor(t)

ap = argparse.ArgumentParser()
ap.add_argument("--src_dir", required=True, help="emb_dir sorgente (contiene E_calib.pt, y_calib.pt, E_test.pt, y_test.pt)")
ap.add_argument("--dst_dir", required=True, help="nuova cartella di output")
ap.add_argument("--unseen_ids", required=True, help="lista classi unseen separata da virgole, es: -1,8,7,5")
args = ap.parse_args()

U = set(int(x) for x in args.unseen_ids.split(",") if x.strip()!="")
os.makedirs(args.dst_dir, exist_ok=True)

# carica tensori base
E_cal = load_t(os.path.join(args.src_dir,"E_calib.pt")).float()
y_cal = load_t(os.path.join(args.src_dir,"y_calib.pt")).long().view(-1)
E_tst = load_t(os.path.join(args.src_dir,"E_test.pt")).float()
y_tst = load_t(os.path.join(args.src_dir,"y_test.pt")).long().view(-1)

# 1) filtra calibrazione: rimuovi esempi con label in U
keep = ~torch.isin(y_cal, torch.tensor(list(U), dtype=torch.long))
E_cal_new = E_cal[keep]
y_cal_new = y_cal[keep]

# 2) costruisci y_open sul test: 1 se la classe appartiene a U
y_open = torch.tensor([1 if int(c) in U else 0 for c in y_tst.tolist()], dtype=torch.int64)

# 3) salva
torch.save(E_cal_new, os.path.join(args.dst_dir,"E_calib.pt"))
torch.save(y_cal_new, os.path.join(args.dst_dir,"y_calib.pt"))
torch.save(E_tst,     os.path.join(args.dst_dir,"E_test.pt"))
torch.save(y_tst,     os.path.join(args.dst_dir,"y_test.pt"))
torch.save(y_open,    os.path.join(args.dst_dir,"y_open.pt"))

# 4) copia (se esistono) eventuali file utili
for name in ["support.csv","query.csv","test_mixed.csv"]:
    p = os.path.join(args.src_dir, name)
    if os.path.isfile(p):
        shutil.copy2(p, os.path.join(args.dst_dir, name))

print(f"[DONE] New emb_dir: {args.dst_dir}")
print(f"  calib: {E_cal.shape[0]} -> {E_cal_new.shape[0]} (rimossi {int((~keep).sum())})")
print(f"  test:  {E_tst.shape[0]}")
print(f"  unseen test = {int((y_open==1).sum())} / {y_open.numel()}  ({(y_open.float().mean()*100):.2f}%)")
print(f"  unseen_ids = {sorted(U)}")
