import torch, pandas as pd
ckpt_in  = r"outputs/v3_seen_retrain/models/shipsear_supcon_v3_seen.pt"
ckpt_out = r"outputs/v3_seen_retrain/models/shipsear_supcon_v3_seen+unseen.pt"
sup_csv  = r"outputs/fewshot_from_pool_k10/support.csv"
qry_csv  = r"outputs/fewshot_from_pool_k10/query.csv"

ckpt = torch.load(ckpt_in, map_location="cpu", weights_only=False)
class_names = ckpt.get("class_names", ckpt.get("classes", None))
if class_names is None:
    raise SystemExit("Checkpoint missing 'class_names' or 'classes'.")

sup = pd.read_csv(sup_csv); qry = pd.read_csv(qry_csv)
unseen = sorted(set(sup["category"].unique()).union(set(qry["category"].unique())))
missing = [c for c in unseen if c not in class_names]
print("[INFO] Current classes:", class_names)
print("[INFO] Unseen from CSV:", unseen)
print("[INFO] Missing to add  :", missing)

if missing:
    new_class_names = list(class_names) + missing
    ckpt["class_names"] = new_class_names
    ckpt["classes"] = new_class_names
    print("[OK] Patched class_names len:", len(new_class_names))
    torch.save(ckpt, ckpt_out)
    print("[OK] Saved:", ckpt_out)
else:
    print("[INFO] Nothing to patch; all unseen already in checkpoint.")
