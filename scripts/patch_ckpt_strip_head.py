import torch

ckpt_in = r"outputs/v3_seen_retrain/models/shipsear_supcon_v3_seen+unseen.pt"
ckpt_out = r"outputs/v3_seen_retrain/models/shipsear_supcon_v3_seen+unseen_headless.pt"

ckpt = torch.load(ckpt_in, map_location="cpu", weights_only=False)
sd = ckpt.get("model", {})
removed = []
for k in ["cls.weight", "cls.bias"]:
    if k in sd:
        sd.pop(k)
        removed.append(k)
ckpt["model"] = sd
torch.save(ckpt, ckpt_out)
print("[OK] stripped:", removed, "->", ckpt_out)
