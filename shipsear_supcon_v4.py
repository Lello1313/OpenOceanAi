# shipsear_supcon_v3.py
# Patch richieste: (1) niente AUROC/AUPRC su unseen in-epoch; (2) assert disgiunti per recording_id; (3) steps_per_epoch realistici.

import os, json, random, argparse, math, csv, warnings, pathlib, hashlib
from typing import Dict, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor

# =========================
# Import dal tuo core
# =========================
# Nota: questi moduli devono esistere come nel resto del repo (usati anche da eval_few.py / eval_zero.py)
from shipsear_core import (
    MultiResLogMel, MelCache, ShipsearDataset, MiniResNet,
    set_seed, ensure_dir, device_select,
    SupConLoss, prototype_repulsion_loss
)

print("[PATCHED] shipsear_supcon_v3.py v4 — AUROC in-epoch rimosso")
warnings.filterwarnings("ignore")

# =========================
# CLI
# =========================

def parse_args():
    p = argparse.ArgumentParser()
    # Dati / split
    p.add_argument("--meta_csv", type=str, required=True, help="CSV train_seen completo (fold 1-4).")
    p.add_argument("--calib_csv", type=str, default="", help="CSV holdout_seen (fold 5). Se vuoto, si ricava da split_json.")
    p.add_argument("--unseen_csv", type=str, default="", help="CSV unseen_pool. Se vuoto, si ricava da split_json.")
    p.add_argument("--splits_json", type=str, default="", help="JSON con split precomputati (train/calib/unseen).")

    # IO e runtime
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--wav_dir", type=str, default="shipsear_segments")
    p.add_argument("--alt_wav_dir", type=str, default="shipsear_raw")
    p.add_argument("--out_dir", type=str, default="outputs/v3_seen_retrain")
    p.add_argument("--cache_dir", type=str, default="cache_mels_v3_seen")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Modello/featurizer (coerenti col checkpoint precedente)
    p.add_argument("--sr", type=int, default=32000)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--nfft1", type=int, default=1024)
    p.add_argument("--hop1", type=int, default=320)
    p.add_argument("--nfft2", type=int, default=2048)
    p.add_argument("--hop2", type=int, default=640)
    p.add_argument("--target_secs", type=float, default=4.0)
    p.add_argument("--in_ch", type=int, default=6)      # log-mel + delta + delta-delta
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--num_heads", type=int, default=3)

    # Ottimizzazione
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, choices=["none", "onecycle", "step"], default="onecycle")
    p.add_argument("--steps_per_epoch", type=int, default=-1, help="<=0 = calcolo automatico realistico in base ai dati")

    # Loss weights
    p.add_argument("--lambda_supcon", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.5)
    p.add_argument("--lambda_triplet", type=float, default=0.2)
    p.add_argument("--lambda_proto", type=float, default=0.1)
    p.add_argument("--triplet_margin", type=float, default=0.2)
    p.add_argument("--supcon_temp", type=float, default=0.07)

    return p.parse_args()


# =========================
# Utility split & checks
# =========================

def _recording_id_from_filename(fn: str) -> str:
    # Adatta al tuo naming: es. "21__18_07_13_lanchaMotora_seg5.wav" -> "21__18_07_13_lanchaMotora"
    stem = pathlib.Path(fn).stem
    return stem.split("_seg")[0]

def _ensure_recording_id(df: pd.DataFrame) -> pd.DataFrame:
    if "recording_id" not in df.columns:
        df = df.copy()
        df["recording_id"] = df["filename"].apply(_recording_id_from_filename)
    return df

def assert_disjoint_recordings(sp: Dict[str, pd.DataFrame]):
    tr = _ensure_recording_id(sp["train"])
    ca = _ensure_recording_id(sp["calib"])
    un = _ensure_recording_id(sp["unseen"])
    def _check(a, b, name):
        inter = set(a["recording_id"]) & set(b["recording_id"])
        assert len(inter) == 0, f"LEAKAGE: {name} share recording_id: {list(sorted(inter))[:10]}"
    _check(tr, ca, "train-calib")
    _check(tr, un, "train-unseen")
    _check(ca, un, "calib-unseen")
    print(f"[SPLIT-OK] Disjoint per recording_id: "
          f"{tr['recording_id'].nunique()}/{ca['recording_id'].nunique()}/{un['recording_id'].nunique()}")

def load_splits(args) -> Dict[str, pd.DataFrame]:
    # train sempre da meta_csv (seen fold 1-4)
    train = pd.read_csv(args.meta_csv)

    # calib/unseen da CSV o da JSON
    if args.calib_csv:
        calib = pd.read_csv(args.calib_csv)
    else:
        calib = None
    if args.unseen_csv:
        unseen = pd.read_csv(args.unseen_csv)
    else:
        unseen = None

    if (calib is None or unseen is None) and args.splits_json:
        with open(args.splits_json, "r") as f:
            js = json.load(f)
        if calib is None and "holdout_seen" in js:
            calib = pd.DataFrame(js["holdout_seen"])
        if unseen is None and "unseen_pool" in js:
            unseen = pd.DataFrame(js["unseen_pool"])

    if calib is None or unseen is None:
        raise ValueError("Necessari calib_csv e unseen_csv o uno splits_json che li contenga.")

    sp = {"train": train, "calib": calib, "unseen": unseen}
    # Enforce disgiunzione per recording_id (Fix #2)
    assert_disjoint_recordings(sp)
    return sp


# =========================
# Sampler bilanciato (Fix #3)
# =========================

class BalancedBatchSampler(Sampler):
    """Sampler bilanciato per classi (C×K) con steps_per_epoch realistici."""
    def __init__(self, labels, n_classes, n_samples, steps_per_epoch=None):
        self.labels = np.array(labels)
        self.labels_set = list(sorted(set(self.labels)))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set: np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = int(n_classes)
        self.n_samples = int(n_samples)
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_classes * self.n_samples
        self.steps_per_epoch = steps_per_epoch if (steps_per_epoch and steps_per_epoch > 0) else None

    def __iter__(self):
        self.count = 0
        while True:
            if self.steps_per_epoch is not None and self.count >= self.steps_per_epoch:
                break
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                start = self.used_label_indices_count[class_]
                end = start + self.n_samples
                if end > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    start = 0
                    end = self.n_samples
                sel = self.label_to_indices[class_][start:end]
                indices.extend(sel)
                self.used_label_indices_count[class_] = end
            yield from indices
            self.count += 1
            # Se non imposto steps, fermo quando ho "coperto" grossolanamente il dataset
            if self.steps_per_epoch is None and (self.count * self.batch_size >= self.n_dataset):
                break

    def __len__(self):
        if self.steps_per_epoch is not None:
            return self.steps_per_epoch * self.batch_size
        return (self.n_dataset // self.batch_size) * self.batch_size


# =========================
# Dataloaders
# =========================

def collate_batch(batch):
    xs, ys = zip(*batch)  # xs: list di [1, C, M, T]
    # padding sul last-dim (T) al max T del batch
    Tm = max(x.shape[-1] for x in xs)
    xs_pad = [F.pad(x, (0, Tm - x.shape[-1])) if x.shape[-1] < Tm else x for x in xs]
    x = torch.stack(xs_pad, dim=0)  # [B, 1, C, M, Tm]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def build_loaders(args, featurizer, cache, class_names, splits):
    def to_records(df):
        return df[["filename", "category"]].to_dict(orient="records")

    ds_train  = ShipsearDataset(to_records(splits["train"]),  args.data_dir, class_names, featurizer, cache,
                                mode="train", aug_spec=True,  aug_wav=False,
                                wav_dir=args.wav_dir, alt_wav_dir=args.alt_wav_dir)
    ds_calib  = ShipsearDataset(to_records(splits["calib"]),  args.data_dir, class_names, featurizer, cache,
                                mode="eval",  aug_spec=False, aug_wav=False,
                                wav_dir=args.wav_dir, alt_wav_dir=args.alt_wav_dir)
    ds_unseen = ShipsearDataset(to_records(splits["unseen"]), args.data_dir, class_names, featurizer, cache,
                                mode="eval",  aug_spec=False, aug_wav=False,
                                wav_dir=args.wav_dir, alt_wav_dir=args.alt_wav_dir)

    # Balanced sampler (C×K); calcolo realistico degli steps se args.steps_per_epoch<=0
    train_labels = [r["category"] for r in splits["train"].to_dict(orient="records")]
    unique_classes = list(sorted(set(train_labels)))
    n_classes_in_batch  = min(len(unique_classes), 8)
    n_samples_per_class = max(1, args.batch_size // n_classes_in_batch)

    if args.steps_per_epoch and args.steps_per_epoch > 0:
        steps_for_sampler = args.steps_per_epoch
    else:
        # Fix #3: auto in base ai dati
        samples_per_batch = n_classes_in_batch * n_samples_per_class
        steps_for_sampler = max(1, len(train_labels) // samples_per_batch)

    train_sampler = BalancedBatchSampler(
        labels=train_labels,
        n_classes=n_classes_in_batch,
        n_samples=n_samples_per_class,
        steps_per_epoch=steps_for_sampler
    )

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_batch,
        pin_memory=True,
        drop_last=True
    )
    dl_calib  = DataLoader(ds_calib,  batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=collate_batch, pin_memory=True)
    dl_unseen = DataLoader(ds_unseen, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.workers, collate_fn=collate_batch, pin_memory=True)

    return dl_train, dl_calib, dl_unseen


# =========================
# Embedding & metriche
# =========================

@torch.no_grad()
def extract_embeddings(model, loader, device):
    E, L, Y = [], [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        e, p, logits = model(xb, return_proj=True)
        E.append(e.cpu().numpy())
        L.append(logits.cpu().numpy())
        Y.append(yb.cpu().numpy())
    if len(E) == 0:
        # guardie per vuoti
        return (np.zeros((0, model.norm.normalized_shape[0])), 
                np.zeros((0, model.cls.out_features)), 
                np.zeros((0,), dtype=np.int64))
    return np.concatenate(E, 0), np.concatenate(L, 0), np.concatenate(Y, 0)

def compute_silhouette(E, y):
    try:
        if len(np.unique(y)) < 2 or E.shape[0] < 10: return float("nan")
        return float(silhouette_score(E, y, metric="euclidean"))
    except Exception:
        return float("nan")

def class_prototypes(E, y, ncls):
    protos = []
    for c in range(ncls):
        m = (y == c)
        if m.sum() == 0:
            # se manca, prototipo fittizio zero
            protos.append(np.zeros((E.shape[1],), dtype=np.float32))
        else:
            protos.append(E[m].mean(0))
    return np.stack(protos, 0)

def mahal_scores(E, protos, VI):
    # min Mahalanobis alla classe più vicina
    diffs = E[:, None, :] - protos[None, :, :]  # [N,C,D]
    # d^2 = (x-μ)^T VI (x-μ)
    left = np.einsum("ncd,dd->ncd", diffs, VI)
    d2 = np.einsum("ncd,ncd->nc", left, diffs)
    return d2.min(axis=1)

def cosine_scores(E, protos):
    # 1 - max cosine sim
    Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
    sims = En @ Pn.T
    return 1.0 - sims.max(axis=1)

def energy_scores(L):
    # LogSumExp nei logits: higher = più in-distribution → per novelty usare -energy
    e = np.log(np.exp(L).sum(axis=1) + 1e-6)
    return -e  # maggiore → più outlier

def lof_scores(E, fitted_lof):
    return -fitted_lof.score_samples(E)  # maggiore → più outlier

def _normalize_pair(cal_vec, un_vec):
    s_all = np.concatenate([cal_vec, un_vec], axis=0)
    mn, mx = s_all.min(), s_all.max()
    denom = (mx - mn) + 1e-12
    return (cal_vec - mn)/denom, (un_vec - mn)/denom

def compute_ensemble_scores(E_tr, y_tr, E_cal, L_cal, E_un, L_un, ncls):
    if E_tr.shape[0] < 2:
        E_tr = E_cal; y_tr = y_tr[:E_cal.shape[0]]
    protos = class_prototypes(E_tr, y_tr, ncls)
    cov = np.cov(E_tr.T) + 1e-4*np.eye(E_tr.shape[1])
    VI = np.linalg.pinv(cov)
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05).fit(E_tr)

    def pack(E, L):
        return {
            "mahal": mahal_scores(E, protos, VI),
            "cosine": cosine_scores(E, protos),
            "energy": energy_scores(L),
            "lof":    lof_scores(E, lof),
        }

    cal = pack(E_cal, L_cal)
    un  = pack(E_un,  L_un)

    # Normalizzazione congiunta canale per canale
    for k in ["mahal","cosine","energy","lof"]:
        cal[k], un[k] = _normalize_pair(cal[k], un[k])

    # Pesi (esempio)
    w = {"mahal":0.4, "cosine":0.3, "energy":0.2, "lof":0.1}
    S_cal = sum(w[k]*cal[k] for k in w)
    S_un  = sum(w[k]*un[k]  for k in w)
    return np.array(S_cal), np.array(S_un)

def eval_openset_ensemble(E_tr, y_tr, E_cal, L_cal, E_un, L_un):
    ncls = int(y_tr.max()) + 1 if y_tr.size > 0 else 1
    S_cal, S_un = compute_ensemble_scores(E_tr, y_tr, E_cal, L_cal, E_un, L_un, ncls)
    y_true  = np.concatenate([np.zeros_like(S_cal, dtype=int), np.ones_like(S_un, dtype=int)])
    y_score = np.concatenate([S_cal, S_un])
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return auroc, auprc, (S_cal, S_un), None


# =========================
# Training
# =========================

def build_featurizer_and_cache(args):
    featurizer = MultiResLogMel(
        target_sr=args.sr,
        n_mels=args.n_mels,
        configs=((args.nfft1, args.hop1), (args.nfft2, args.hop2)),
        target_secs=args.target_secs,
        use_deltas=(args.in_ch == 6),
        use_attention=True
    )
    fe_cfg = {
        "sr": args.sr,
        "n_mels": args.n_mels,
        "configs": [(args.nfft1, args.hop1), (args.nfft2, args.hop2)],
        "target_secs": args.target_secs,
        "use_deltas": (args.in_ch == 6),
        "use_attention": True
    }
    cache = MelCache(cache_dir=args.cache_dir, fe_cfg=fe_cfg)
    return featurizer, cache

def build_model(args, n_classes, device):
    model = MiniResNet(
        in_ch=args.in_ch,
        emb_dim=args.emb_dim,
        proj_dim=args.proj_dim,
        num_classes=n_classes,
        num_heads=args.num_heads
    ).to(device)
    return model

def prepare_labels_encoder(class_names):
    cls2id = {c:i for i,c in enumerate(class_names)}
    id2cls = {i:c for c,i in cls2id.items()}
    return cls2id, id2cls

def training_loop(args):
    set_seed(args.seed)
    device = device_select(args.device)

    ensure_dir(args.out_dir)
    models_dir = os.path.join(args.out_dir, "models")
    ensure_dir(models_dir)
    logs_path = os.path.join(args.out_dir, "train_log.csv")

    # Splits (con assert disgiunti per recording_id) — Fix #2
    splits = load_splits(args)

    # Classi
    class_names = sorted(splits["train"]["category"].unique().tolist())
    n_classes = len(class_names)

    # Featurizer, cache, loader
    featurizer, cache = build_featurizer_and_cache(args)
    dl_train, dl_calib, dl_unseen = build_loaders(args, featurizer, cache, class_names, splits)

    # Modello + ottimizzatori
    model = build_model(args, n_classes, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "onecycle":
        lr_sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=args.lr, epochs=args.epochs,
            steps_per_epoch=max(1, len(dl_train))
        )
    elif args.scheduler == "step":
        lr_sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs//3), gamma=0.5)
    else:
        lr_sched = None

    # Losses
    supcon = SupConLoss(temperature=args.supcon_temp)
    triplet = nn.TripletMarginLoss(margin=args.triplet_margin, p=2.0)
    ce_loss = nn.CrossEntropyLoss()

    # Log CSV (aggiungo ClosedAcc al posto delle metriche open-set in-epoch) — Fix #1
    if not os.path.exists(logs_path):
        with open(logs_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch","lr","supcon","ce","triplet","proto_rep","ClosedAcc","AUROC","AUPRC","Silhouette"])

    best_closed = -1.0
    best_path = os.path.join(models_dir, "shipsear_supcon_v3_best_closed.pt")

    for ep in range(1, args.epochs+1):
        model.train()
        run_supcon = run_ce = run_triplet = run_proto = 0.0
        n_batches = 0

        pbar = tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            # forward
            e, proj, logits = model(xb, return_proj=True)

            # SupCon (stesso batch come views augmentate già gestite dal dataset)
            loss_sup = supcon(proj, yb)

            # CE
            loss_ce = ce_loss(logits, yb)

            # Triplet: sampling semplice ancorato alle classi nel batch
            # Costruisco ancore/pos/neg al volo (robusto anche se qualche classe è scarsa)
            with torch.no_grad():
                y_np = yb.detach().cpu().numpy()
                anchors, positives, negatives = [], [], []
                for c in np.unique(y_np):
                    idx = np.where(y_np == c)[0]
                    idx_neg = np.where(y_np != c)[0]
                    if len(idx) >= 2 and len(idx_neg) >= 1:
                        a, p = np.random.choice(idx, size=2, replace=False)
                        n = np.random.choice(idx_neg, size=1, replace=False)[0]
                        anchors.append(a); positives.append(p); negatives.append(n)
                if len(anchors) == 0:
                    anchors = [0]; positives = [0]; negatives = [0]
                anc = e[anchors]; pos = e[positives]; neg = e[negatives]
            loss_trip = triplet(anc, pos, neg)

            # Prototype repulsion
            with torch.no_grad():
                # prototipi grezzi dal batch (fallback se classe non presente)
                ncls = n_classes
                protos_list = []
                for c in range(ncls):
                    m = (yb == c)
                    if m.sum() == 0:
                        protos_list.append(torch.zeros(args.emb_dim, device=yb.device))
                    else:
                        protos_list.append(e[m].mean(0))
                prototypes = torch.stack(protos_list, 0)  # [C, D]
            loss_proto = prototype_repulsion_loss(prototypes, margin=2.0)

            # backward
            total = args.lambda_supcon*loss_sup + args.lambda_ce*loss_ce + args.lambda_triplet*loss_trip + args.lambda_proto*loss_proto
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if lr_sched is not None and args.scheduler == "onecycle":
                lr_sched.step()

            run_supcon += float(loss_sup.detach().cpu())
            run_ce     += float(loss_ce.detach().cpu())
            run_triplet+= float(loss_trip.detach().cpu())
            run_proto  += float(loss_proto.detach().cpu())
            n_batches  += 1

        if lr_sched is not None and args.scheduler != "onecycle":
            lr_sched.step()

        # --------------------
        # EVAL IN-EPOCH (Fix #1):
        #   * Solo closed-set su calib (accuracy) + silhouette su train_eval
        #   * NO AUROC/AUPRC su unseen qui
        # --------------------
        # Ricostruisco train_eval senza augment
        ds_train_eval = ShipsearDataset(
            splits["train"][["filename","category"]].to_dict(orient="records"),
            args.data_dir, class_names, featurizer, cache,
            mode="eval", aug_spec=False, aug_wav=False,
            wav_dir=args.wav_dir, alt_wav_dir=args.alt_wav_dir
        )
        dl_train_eval = DataLoader(
            ds_train_eval, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_batch, pin_memory=True
        )

        model.eval()
        with torch.no_grad():
            E_tr_eval, L_tr_eval, y_tr_eval = extract_embeddings(model, dl_train_eval, device)
            E_ca, L_ca, y_ca = extract_embeddings(model, dl_calib, device)
            # Metrics in-epoch
            closed_acc = float((L_ca.argmax(1) == y_ca).mean()) if L_ca.shape[0] > 0 else float("nan")
            sil = compute_silhouette(E_tr_eval, y_tr_eval)
            # Segnaposto per colonne AUROC/AUPRC (non stimati in-epoch)
            auroc, auprc = float("nan"), float("nan")

        lr_now = opt.param_groups[0]["lr"]
        with open(logs_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                ep, f"{lr_now:.6f}",
                f"{run_supcon/max(1,n_batches):.4f}",
                f"{run_ce/max(1,n_batches):.4f}",
                f"{run_triplet/max(1,n_batches):.4f}",
                f"{run_proto/max(1,n_batches):.4f}",
                f"{closed_acc:.3f}" if not np.isnan(closed_acc) else "nan",
                "nan", "nan",
                f"{sil:.3f}" if not np.isnan(sil) else "nan",
            ])

        # Salvataggi
        torch.save({
            "epoch": ep,
            "model": model.state_dict(),
            "class_names": class_names,
            "args": vars(args)
        }, os.path.join(models_dir, "shipsear_supcon_v3_last.pt"))

        if not np.isnan(closed_acc) and closed_acc > best_closed:
            best_closed = closed_acc
            torch.save({
                "epoch": ep,
                "model": model.state_dict(),
                "class_names": class_names,
                "args": vars(args),
                "best_closed": float(best_closed),
            }, best_path)

    # =========================
    # FINAL OPEN-SET EVAL (una sola volta, al termine) — Fix #1
    # =========================
    print("\n[FINAL-OPENSET] Estrazione embedding per valutazione open-set...")
    # Ricostruisco i loader eval (se necessario)
    ds_train_eval = ShipsearDataset(
        splits["train"][["filename","category"]].to_dict(orient="records"),
        args.data_dir, class_names, featurizer, cache,
        mode="eval", aug_spec=False, aug_wav=False,
        wav_dir=args.wav_dir, alt_wav_dir=args.alt_wav_dir
    )
    dl_train_eval = DataLoader(
        ds_train_eval, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=collate_batch, pin_memory=True
    )

    E_tr_eval, L_tr_eval, y_tr_eval = extract_embeddings(model, dl_train_eval, device)
    E_ca, L_ca, y_ca = extract_embeddings(model, dl_calib,  device)
    E_un, L_un, _    = extract_embeddings(model, dl_unseen, device)

    if E_ca.shape[0] > 0 and E_un.shape[0] > 0:
        final_auroc, final_auprc, _, _ = eval_openset_ensemble(E_tr_eval, y_tr_eval, E_ca, L_ca, E_un, L_un)
        print(f"[FINAL-OPENSET] AUROC={final_auroc:.3f}  AUPRC={final_auprc:.3f}")
    else:
        final_auroc = final_auprc = float("nan")
        print("[FINAL-OPENSET] Dati insufficienti per AUROC/AUPRC (calib o unseen vuoti).")

    # Checkpoint finale
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "class_names": class_names,
        "args": vars(args),
        "final_auroc": None if np.isnan(final_auroc) else float(final_auroc),
        "final_auprc": None if np.isnan(final_auprc) else float(final_auprc),
        "best_closed": None if best_closed < 0 else float(best_closed),
    }, os.path.join(models_dir, "shipsear_supcon_v3_final.pt"))

    print(f"[DONE] Modelli salvati in: {models_dir}")
    print(f"[DONE] Log: {logs_path}")


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)
    training_loop(args)


if __name__ == "__main__":
    print("[PATCHED] shipsear_supcon_v3.py v4 — AUROC in-epoch rimosso")

    main()
