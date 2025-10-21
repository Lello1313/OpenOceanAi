# shipsear_supcon_v3.py
# Tutte le patch richieste integrate.

import os, json, random, argparse, math, csv, warnings, hashlib, pathlib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Sampler


from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import LocalOutlierFactor
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

# -------------------------
# Utils
# -------------------------

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p): os.makedirs(p, exist_ok=True)
def normpath(*x): return os.path.normpath(os.path.join(*x))

def device_select(pref: str = "auto"):
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda": return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Robust audio I/O
# -------------------------

def safe_load_wav(path: str):
    try:
        wav, sr = torchaudio.load(path)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav, sr
    except Exception:
        try:
            y, sr = sf.read(path, dtype='float32', always_2d=False)
            if y.ndim == 1:
                y = y[None, :]
            else:
                if y.shape[0] < y.shape[1]:
                    y = y.T
                y = y[:, 0][None, :]
            return torch.from_numpy(y), sr
        except Exception:
            y, sr = librosa.load(path, sr=None, mono=True)
            return torch.from_numpy(y[None, :]), sr

def trim_silence_librosa(wav_np: np.ndarray, top_db=30):
    y_trim, _ = librosa.effects.trim(wav_np, top_db=top_db)
    if y_trim.size == 0: return wav_np.astype(np.float32)
    return y_trim.astype(np.float32)

def highpass_biquad_ta(wav: torch.Tensor, sr: int, cutoff=50.0):
    return torchaudio.functional.highpass_biquad(wav, sr, cutoff)

# -------------------------
# Advanced Audio Augmentation
# -------------------------

class WavAugment:
    @staticmethod
    def time_stretch(wav_np, rate_range=(0.85, 1.15)):
        if random.random() < 0.5:
            rate = random.uniform(*rate_range)
            try: return librosa.effects.time_stretch(wav_np, rate=rate)
            except: return wav_np
        return wav_np
    
    @staticmethod
    def pitch_shift(wav_np, sr, n_steps_range=(-2, 2)):
        if random.random() < 0.5:
            n_steps = random.randint(*n_steps_range)
            try: return librosa.effects.pitch_shift(wav_np, sr=sr, n_steps=n_steps)
            except: return wav_np
        return wav_np
    
    @staticmethod
    def add_noise(wav_np, snr_db_range=(15, 30)):
        if random.random() < 0.5:
            snr_db = random.uniform(*snr_db_range)
            noise = np.random.randn(*wav_np.shape)
            sp = np.mean(wav_np ** 2)
            if sp < 1e-10: return wav_np
            npow = sp / (10 ** (snr_db / 10))
            noise = noise * np.sqrt(npow)
            return (wav_np + noise).astype(np.float32)
        return wav_np

# -------------------------
# Multi-Resolution Log-Mel with Attention Fusion
# -------------------------

class MultiScaleAttention(nn.Module):
    def __init__(self, num_scales=2, channels_per_scale=3):
        super().__init__()
        self.num_scales = num_scales
        self.cps = channels_per_scale
        total_ch = num_scales * channels_per_scale
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_ch, num_scales, 1),
            nn.Softmax(dim=1)
        )
    def forward(self, feats_list):
        x = torch.cat(feats_list, dim=0).unsqueeze(0)  # [1, C_tot, M, T]
        attn = self.attention(x)                       # [1, S, 1, 1]
        weighted = []
        for i, feat in enumerate(feats_list):
            w = attn[0, i:i+1].squeeze()               # scalar/broadcast
            weighted.append(feat * w)
        return torch.cat(weighted, dim=0)

class MultiResLogMel:
    def __init__(self,
                 target_sr=22050,
                 n_mels=128,
                 configs=((1024,256),(2048,512)),
                 fmin=30, fmax=None,
                 target_secs=5.0,
                 use_deltas=True,
                 use_attention=True):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.target_secs = target_secs
        self.target_len = int(target_sr * target_secs)
        self.use_deltas = use_deltas
        self.use_attention = use_attention
        self.mels = []
        for n_fft, hop in configs:
            self.mels.append(
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=target_sr, n_fft=n_fft, hop_length=hop,
                    n_mels=n_mels, f_min=fmin, f_max=fmax or target_sr // 2, power=2.0
                )
            )
        self.configs = configs
        if use_attention:
            num_scales = len(configs)
            cps = 3 if use_deltas else 1
            self.attention_fusion = MultiScaleAttention(num_scales, cps)

    def _compute_deltas_np(self, mel_np: np.ndarray):
        d1 = librosa.feature.delta(mel_np, order=1)
        d2 = librosa.feature.delta(mel_np, order=2)
        return d1, d2

    def _align_temporal_dims(self, feats):
        max_T = max(f.size(-1) for f in feats)
        aligned = []
        for f in feats:
            if f.size(-1) != max_T:
                f_4d = f.unsqueeze(0)  # [1, C, M, T]
                f_al = F.interpolate(f_4d, size=(f.size(-2), max_T),
                                     mode='bilinear', align_corners=False).squeeze(0)
                aligned.append(f_al)
            else:
                aligned.append(f)
        return aligned

    def waveform_to_features(self, wav: torch.Tensor, sr: int, 
                             mode: str = "eval", apply_crop=True,
                             apply_wav_aug=False):
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
            sr = self.target_sr
        wav = highpass_biquad_ta(wav, sr, cutoff=50.0)
        y = wav.squeeze(0).cpu().numpy().astype(np.float32)
        y = trim_silence_librosa(y, top_db=30)
        if y.size == 0:
            y = wav.squeeze(0).cpu().numpy().astype(np.float32)
        if apply_wav_aug and mode == "train":
            y = WavAugment.time_stretch(y)
            y = WavAugment.pitch_shift(y, sr)
            y = WavAugment.add_noise(y)
        if apply_crop:
            y = self._fix_length(y, mode)
        else:
            if y.shape[0] < self.target_len:
                pad = self.target_len - y.shape[0]
                left = pad // 2; right = pad - left
                y = np.pad(y, (left, right), mode="constant")
        wav = torch.from_numpy(y[None, :])

        feats = []
        for mel_transform in self.mels:
            m = mel_transform(wav)                     # [1, M, T]
            logm = torch.clamp(m, min=1e-10).log().squeeze(0)  # [M, T]
            mu, sigma = logm.mean(), logm.std().clamp(min=1e-8)
            logm = (logm - mu) / sigma
            if self.use_deltas:
                lm_np = logm.cpu().numpy()
                d1, d2 = self._compute_deltas_np(lm_np)
                stacked = np.stack([lm_np, d1, d2], axis=0)
                feats.append(torch.from_numpy(stacked))
            else:
                feats.append(logm.unsqueeze(0))
        feats = self._align_temporal_dims(feats)
        if self.use_attention and hasattr(self, 'attention_fusion'):
            x = self.attention_fusion(feats)
        else:
            x = torch.cat(feats, dim=0)
        return x  # [C, M, T]
    
    def _fix_length(self, y: np.ndarray, mode: str) -> np.ndarray:
        T = y.shape[0]; L = self.target_len
        if T == L: return y
        if T < L:
            pad = L - T; left = pad // 2; right = pad - left
            return np.pad(y, (left, right), mode="constant")
        if mode == "train":
            start = random.randint(0, T - L)
        else:
            start = (T - L) // 2
        return y[start:start+L]

# -------------------------
# Cache NPZ (uncropped)
# -------------------------

class MelCache:
    def __init__(self, cache_dir="cache_mels_v3_noleak", fe_cfg: dict = None):
        self.cache_dir = cache_dir
        ensure_dir(cache_dir)
        self.fe_cfg = fe_cfg or {}
        self.cfg_hash = hashlib.md5(json.dumps(self.fe_cfg, sort_keys=True).encode()).hexdigest()[:8]
    def key(self, wav_relpath: str):
        stem = pathlib.Path(wav_relpath).stem
        return f"{stem}__{self.cfg_hash}.npz"
    def has(self, wav_relpath: str):
        return os.path.exists(normpath(self.cache_dir, self.key(wav_relpath)))
    def load(self, wav_relpath: str):
        p = normpath(self.cache_dir, self.key(wav_relpath))
        with np.load(p) as data:
            arr = data["logmel"]
        return torch.from_numpy(arr)
    def save(self, wav_relpath: str, logmel: torch.Tensor):
        p = normpath(self.cache_dir, self.key(wav_relpath))
        np.savez_compressed(p, logmel=logmel.detach().cpu().numpy())  # FIX: detach

# -------------------------
# Dataset (with dynamic augmentation)
# -------------------------

time_mask = torchaudio.transforms.TimeMasking(time_mask_param=32)
freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=12)

def resolve_audio_path(root: str, rel_filename: str, wav_dir: str, alt_wav_dir: str):
    candidates = [
        normpath(root, wav_dir, rel_filename),
        normpath(root, alt_wav_dir, rel_filename),
        normpath(root, "audio", rel_filename),
        normpath(root, rel_filename),
    ]
    for p in candidates:
        if os.path.exists(p): return p
    raise FileNotFoundError(f"Audio not found: {rel_filename}")

class ShipsearDataset(torch.utils.data.Dataset):
    def __init__(self, records, root, class_names, featurizer: MultiResLogMel,
                 cache: MelCache, mode="train", aug_spec=True, aug_wav=False,
                 wav_dir="shipsear_segments", alt_wav_dir="shipsear_raw"):
        self.records = records
        self.root = root
        self.class_names = list(class_names)
        self.cls2id = {c:i for i,c in enumerate(self.class_names)}
        self.featurizer = featurizer
        self.cache = cache
        self.mode = mode
        self.aug_spec = aug_spec
        self.aug_wav = aug_wav
        self.wav_dir = wav_dir
        self.alt_wav_dir = alt_wav_dir
        hop_fine = self.featurizer.configs[0][1]
        self.target_frames = int(self.featurizer.target_len / hop_fine)

    def __len__(self): return len(self.records)

    def _random_temporal_crop(self, x):
        T = x.size(-1); target = self.target_frames
        if T <= target: return x
        start = random.randint(0, T - target)
        return x[..., start:start+target]

    def __getitem__(self, i):
        r = self.records[i]
        rel = r["filename"]
        abs_path = resolve_audio_path(self.root, rel, self.wav_dir, self.alt_wav_dir)
        key_rel = pathlib.Path(rel).name

        if self.cache.has(key_rel):
            x = self.cache.load(key_rel)  # [C, M, T_full]
        else:
            wav, sr = safe_load_wav(abs_path)
            x = self.featurizer.waveform_to_features(
                wav, sr, mode="eval", apply_crop=False, apply_wav_aug=False
            )
            self.cache.save(key_rel, x)

        if self.mode == "train":
            x = self._random_temporal_crop(x)
            if self.aug_spec:
                for c in range(x.size(0)):
                    if random.random() < 0.5: x[c:c+1] = time_mask(x[c:c+1])
                    if random.random() < 0.5: x[c:c+1] = freq_mask(x[c:c+1])

        y = self.cls2id.get(r["category"], -1)
        return x.unsqueeze(0), y  # [1, C, M, T]

# -------------------------
# Model Components
# -------------------------

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1):
        super().__init__()
        pad = k//2
        self.conv1 = nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, k, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        idn = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None: idn = self.down(idn)
        out = self.act(out + idn)
        return out

class MultiHeadProjection(nn.Module):
    def __init__(self, emb_dim=256, proj_dim=256, num_heads=3):
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, proj_dim)
            ) for _ in range(num_heads)
        ])
    def forward(self, x):
        projs = [head(x) for head in self.heads]
        projs_norm = [F.normalize(p, p=2, dim=1) for p in projs]
        return torch.stack(projs_norm, dim=1)  # [B, H, D]

class MiniResNet(nn.Module):
    def __init__(self, in_ch=6, emb_dim=256, proj_dim=256, num_classes=35, num_heads=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(BasicBlock(32, 64, stride=2), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64,128, stride=2), BasicBlock(128,128))
        self.layer3 = nn.Sequential(BasicBlock(128,192, stride=2), BasicBlock(192,192))
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.head   = nn.Linear(192, emb_dim)
        self.norm   = nn.LayerNorm(emb_dim)
        self.proj   = MultiHeadProjection(emb_dim, proj_dim, num_heads)
        self.cls    = nn.Linear(emb_dim, num_classes)
    def forward(self, x, return_proj=False):
        if x.dim() == 5:
            B, _, C, M, T = x.shape
            x = x.view(B, C, M, T)
        z = self.stem(x)
        z = self.layer1(z); z = self.layer2(z); z = self.layer3(z)
        z = self.pool(z).flatten(1)
        e = self.norm(self.head(z))       # NO L2 QUI
        if return_proj:
            p = self.proj(e)              # L2 per-head nella proiezione
            return e, p, self.cls(e)
        return e

# -------------------------
# Hard Negative Mining with Memory Bank
# -------------------------

class HardNegativeQueue:
    def __init__(self, size=2048, dim=256, num_classes=35, device='cuda'):
        from collections import deque
        self.size = size; self.dim = dim
        self.num_classes = num_classes; self.device = device
        self.queues = {c: deque(maxlen=size) for c in range(num_classes)}
    @torch.no_grad()
    def update(self, embeddings, labels):
        embeddings = embeddings.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        for c in np.unique(labels):
            embs = embeddings[labels==c]
            for e in embs: self.queues[int(c)].append(e)
    def get_hard_negatives(self, anchor, anchor_class, k=10):
        neg_embs = []
        for c in range(self.num_classes):
            if c != anchor_class and len(self.queues[c])>0:
                neg_embs.extend(list(self.queues[c]))
        if len(neg_embs) == 0: return None
        neg = torch.from_numpy(np.array(neg_embs)).to(self.device)
        sim = torch.matmul(anchor.unsqueeze(0), neg.T).squeeze(0)
        k = min(k, len(neg))
        idx = torch.topk(sim, k).indices
        return neg[idx]

# -------------------------
# Losses
# -------------------------

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07): super().__init__(); self.t = temperature
    def forward(self, z, y):
        if z.dim() == 3:
            losses = [self._compute_loss(z[:,h], y) for h in range(z.size(1))]
            return torch.stack(losses).mean()
        return self._compute_loss(z, y)
    def _compute_loss(self, z, y):
        B = z.size(0)
        sim = (z @ z.t()) / self.t
        logits_mask = torch.ones_like(sim) - torch.eye(B, device=sim.device)
        sim = sim - 1e9 * (1 - logits_mask)
        y = y.contiguous().view(-1,1)
        mask_pos = torch.eq(y, y.t()).float() * logits_mask
        logsumexp_all = torch.logsumexp(sim, dim=1)
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * mask_pos).sum(dim=1).clamp(min=1e-12)
        loss = - torch.log(pos_sum) + logsumexp_all
        valid = (mask_pos.sum(dim=1) > 0).float()
        loss = (loss * valid).sum() / valid.clamp(min=1.0).sum()
        return loss

def supcon_mixup(z, y, alpha=0.2):
    if random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(z.size(0), device=z.device)
        z_mix = lam*z + (1-lam)*z[idx]
        same = (y == y[idx]); y_mix = torch.where(same, y, torch.full_like(y, -1))
        valid = y_mix >= 0
        if valid.sum() > 0: return z_mix[valid], y_mix[valid]
    return z, y

def batch_hard_triplet(emb, labels, margin=0.3, queue=None):
    dist = torch.cdist(emb, emb, p=2)
    same = labels.unsqueeze(1).eq(labels.unsqueeze(0))
    diff = ~same
    terms = []
    for i in range(emb.size(0)):
        pos_idx = torch.where(same[i] & (torch.arange(emb.size(0), device=same.device) != i))[0]
        neg_idx = torch.where(diff[i])[0]
        if len(pos_idx)==0 or len(neg_idx)==0: continue
        p = pos_idx[torch.argmax(dist[i, pos_idx])]
        d_ap = dist[i, p]
        n = neg_idx[torch.argmin(dist[i, neg_idx])]
        d_an = dist[i, n]
        if queue is not None:
            hn = queue.get_hard_negatives(emb[i], int(labels[i]), k=5)
            if hn is not None:
                qd = torch.norm(emb[i].unsqueeze(0) - hn, p=2, dim=1)
                d_an = torch.minimum(d_an, qd.min())
        terms.append(F.relu(d_ap - d_an + margin))
    if len(terms)==0: return emb.new_tensor(0.)
    return torch.stack(terms).mean()

def prototype_repulsion_loss(prototypes, margin=2.0):
    if len(prototypes) < 2: return torch.tensor(0., device=prototypes.device)
    D = torch.cdist(prototypes, prototypes, p=2)
    mask = 1 - torch.eye(len(prototypes), device=prototypes.device)
    loss = F.relu(margin - D) * mask
    return loss.sum() / (mask.sum() + 1e-9)

# -------------------------
# Adaptive Temperature
# -------------------------

class AdaptiveTemperature:
    def __init__(self, init_temp=0.07, min_temp=0.02, max_temp=0.10):
        self.temp = init_temp; self.min_temp=min_temp; self.max_temp=max_temp
        self.best_silhouette = -1.0; self.patience = 0
    def update(self, sil):
        if sil > self.best_silhouette + 0.01:
            self.temp = max(self.min_temp, self.temp*0.95)
            self.best_silhouette = sil; self.patience = 0
        else:
            self.patience += 1
            if self.patience >= 3:
                self.temp = min(self.max_temp, self.temp*1.05); self.patience = 0
        return self.temp

# -------------------------
# Splits
# -------------------------

def assign_folds_if_missing(meta: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    if "fold" in meta.columns: return meta
    def fold_from_name(name: str):
        h = int(hashlib.md5((str(name)+str(seed)).encode()).hexdigest(), 16)
        return (h % 5) + 1
    meta = meta.copy(); meta["fold"] = meta["filename"].map(fold_from_name)
    return meta

def make_splits(meta, n_unseen=3, seed=42, fixed_unseen=None):
    classes = sorted(meta["category"].unique()); random.seed(seed)
    if fixed_unseen:
        unseen = sorted([c for c in fixed_unseen if c in classes])
    else:
        unseen = sorted(random.sample(classes, n_unseen))
    seen = [c for c in classes if c not in unseen]
    train = meta[(meta["category"].isin(seen)) & (meta["fold"].isin([1,2,3,4]))]
    calib = meta[(meta["category"].isin(seen)) & (meta["fold"]==5)]
    test_unseen = meta[meta["category"].isin(unseen)]
    return {
        "seen_classes": seen, "unseen_classes": unseen,
        "train": train.reset_index(drop=True),
        "calib": calib.reset_index(drop=True),
        "unseen": test_unseen.reset_index(drop=True),
    }

# -------------------------
# Open-set scoring (ensemble)
# -------------------------

def normalize_01(a):
    a = np.asarray(a); return (a - a.min()) / (a.max() - a.min() + 1e-12)

def class_prototypes(E, y, ncls):
    protos = []
    for c in range(ncls):
        if (y == c).sum() == 0:
            protos.append(np.zeros(E.shape[1], dtype=np.float32))
        else:
            protos.append(E[y==c].mean(0))
    return np.stack(protos, 0)

def mahal_scores(E, protos, VI):
    d = []
    for e in E:
        diffs = protos - e[None, :]
        m = np.sqrt((diffs @ VI * diffs).sum(1))
        d.append(m.min())
    return np.asarray(d)

def cosine_scores(E, protos):
    d = []
    Pn = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
    for e in E:
        en = e / (np.linalg.norm(e) + 1e-8)
        sim = (Pn @ en); d.append(1.0 - sim.max())
    return np.asarray(d)

def energy_scores(logits):
    logits = np.asarray(logits)
    eng = -np.log(np.exp(logits).sum(1) + 1e-12)
    return normalize_01(eng)  # alto = OOD

def lof_scores(E, lof_model):
    s = -lof_model.score_samples(E)
    return normalize_01(s)

def _normalize_pair(cal_vec, un_vec):
    s_all = np.concatenate([cal_vec, un_vec])
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
        "energy": energy_scores(L),   # qui puoi lasciare la sua normalizzazione interna, oppure tornare ai logits grezzi
        "lof":    lof_scores(E, lof)  # meglio grezzo: togli normalize_01 anche qui e normalizza congiunto dopo
    }


    cal = pack(E_cal, L_cal)
    un  = pack(E_un,  L_un)

# Joint normalize per ciascun canale
    for k in ["mahal","cosine","energy","lof"]:
     cal[k], un[k] = _normalize_pair(cal[k], un[k])

    w = {"mahal":0.4, "cosine":0.3, "energy":0.2, "lof":0.1}
    S_cal = sum(w[k]*cal[k] for k in w)
    S_un  = sum(w[k]*un[k]  for k in w)

    return np.array(S_cal), np.array(S_un)

def eval_openset_ensemble(E_tr, y_tr, E_cal, L_cal, E_un, L_un):
    # 1) punteggi ensemble per cal (seen=0) e un (unseen=1)
    ncls = int(y_tr.max()) + 1 if y_tr.size > 0 else 1
    S_cal, S_un = compute_ensemble_scores(E_tr, y_tr, E_cal, L_cal, E_un, L_un, ncls)

    # 2) etichette oneste: cal=0, un=1
    

    y_true  = np.concatenate([
        np.zeros_like(np.ravel(S_cal), dtype=int),
        np.ones_like(np.ravel(S_un),  dtype=int)
    ])
    y_score = np.concatenate([np.ravel(S_cal), np.ravel(S_un)])

    # 3) metriche senza alcun fit su questi dati
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    # 4) niente calibratore in modalità 'none'
    iso = None
    return auroc, auprc, (S_cal, S_un), iso


def compute_silhouette(E, y):
    try:
        if len(np.unique(y)) < 2 or E.shape[0] < 10: return float("nan")
        return float(silhouette_score(E, y, metric="euclidean"))
    except Exception:
        return float("nan")

# -------------------------
# Dataloaders (con steps_per_epoch)
# -------------------------

def collate_batch(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)  # [B, 1, C, M, T]
    y = torch.tensor(ys, dtype=torch.long)
    return x, y

from torch.utils.data import DataLoader, Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    """Sampler bilanciato per classi (C×K) con rispetto di steps_per_epoch."""
    def __init__(self, labels, n_classes, n_samples, steps_per_epoch=None):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_classes * self.n_samples
        self.steps_per_epoch = steps_per_epoch

    def __iter__(self):
        self.count = 0
        steps_limit = self.steps_per_epoch if (self.steps_per_epoch and self.steps_per_epoch > 0) else None

        while True:
            if steps_limit is not None and self.count >= steps_limit:
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
            self.count += 1  # conta un batch

            if steps_limit is None and (self.count * self.batch_size >= self.n_dataset):
                break

    def __len__(self):
        if self.steps_per_epoch and self.steps_per_epoch > 0:
            return self.steps_per_epoch * self.batch_size
        return (self.n_dataset // self.batch_size) * self.batch_size


def build_loaders(args, featurizer, cache, class_names, meta, splits):
    """Costruisce i DataLoader train/calib/unseen con BalancedBatchSampler (C×K)."""

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

    # --- Balanced Sampler (C×K) che rispetta steps_per_epoch ---
    train_labels = [r["category"] for r in splits["train"].to_dict(orient="records")]
    unique_classes = list(sorted(set(train_labels)))
    n_classes_in_batch = min(len(unique_classes), 8)   # regola a piacere (es. 8)
    n_samples_per_class = max(1, args.batch_size // n_classes_in_batch)

    train_sampler = BalancedBatchSampler(
        labels=train_labels,
        n_classes=n_classes_in_batch,
        n_samples=n_samples_per_class,
        steps_per_epoch=args.steps_per_epoch
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



# -------------------------
# Extract
# -------------------------

def extract_embeddings(model, loader, device):
    model.eval()
    E, L, Y = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            e, p, logits = model(xb, return_proj=True)
            E.append(e.cpu().numpy())
            L.append(logits.cpu().numpy())
            Y.append(yb.cpu().numpy())
    E = np.concatenate(E, 0) if len(E)>0 else np.zeros((0, model.norm.normalized_shape[0]))
    L = np.concatenate(L, 0) if len(L)>0 else np.zeros((0, model.cls.out_features))
    Y = np.concatenate(Y, 0) if len(Y)>0 else np.zeros((0,), dtype=np.int64)
    return E, L, Y

# -------------------------
# Training
# -------------------------

def train(args):
    set_seed(args.seed)
    device = device_select(args.device)
    ensure_dir(args.out_dir); ensure_dir(normpath(args.out_dir, "plots")); ensure_dir(normpath(args.out_dir, "models"))

    meta = pd.read_csv(args.meta_csv)
    meta = assign_folds_if_missing(meta, seed=args.seed)

    if args.splits_json and os.path.exists(args.splits_json):
        with open(args.splits_json, "r") as f: sp = json.load(f)
        seen_classes   = sp.get("seen", sp.get("seen_classes", []))
        unseen_classes = sp.get("unseen", sp.get("unseen_classes", []))
        if not seen_classes or not unseen_classes:
            sp = make_splits(meta, n_unseen=args.n_unseen, seed=args.seed)
        else:
            train_df  = meta[(meta["category"].isin(seen_classes)) & (meta["fold"].isin([1,2,3,4]))]
            calib_df  = meta[(meta["category"].isin(seen_classes)) & (meta["fold"]==5)]
            unseen_df = meta[meta["category"].isin(unseen_classes)]
            sp = {
                "seen_classes": seen_classes,
                "unseen_classes": unseen_classes,
                "train": train_df.reset_index(drop=True),
                "calib": calib_df.reset_index(drop=True),
                "unseen": unseen_df.reset_index(drop=True),
            }
    else:
        sp = make_splits(meta, n_unseen=args.n_unseen, seed=args.seed)

        # --- Override opzionali da CSV esterni ---
    if getattr(args, "calib_csv", ""):
        print(f"[INFO] Overriding calib split from: {args.calib_csv}")
        sp["calib"] = pd.read_csv(args.calib_csv).reset_index(drop=True)

    if getattr(args, "unseen_csv", ""):
        print(f"[INFO] Overriding unseen split from: {args.unseen_csv}")
        sp["unseen"] = pd.read_csv(args.unseen_csv).reset_index(drop=True)


    class_names = sp["seen_classes"]; n_classes = len(class_names)

    fe_cfg = {
        "sr": args.sr, "n_mels": args.n_mels, "configs": [(args.nfft1,args.hop1),(args.nfft2,args.hop2)],
        "target_secs": args.target_secs, "use_deltas": (args.in_ch==6), "use_attention": True
    }
    featurizer = MultiResLogMel(
        target_sr=args.sr, n_mels=args.n_mels,
        configs=((args.nfft1,args.hop1),(args.nfft2,args.hop2)),
        target_secs=args.target_secs, use_deltas=(args.in_ch==6), use_attention=True
    )
    cache = MelCache(cache_dir=args.cache_dir, fe_cfg=fe_cfg)

    dl_train, dl_calib, dl_unseen = build_loaders(args, featurizer, cache, class_names, meta, sp)

    model = MiniResNet(in_ch=args.in_ch, emb_dim=args.emb_dim, proj_dim=args.proj_dim, num_classes=n_classes, num_heads=args.num_heads).to(device)
    supcon = SupConLoss(temperature=args.temperature)
    temp_sched = AdaptiveTemperature(init_temp=args.temperature, min_temp=0.02, max_temp=0.10)
    queue = HardNegativeQueue(size=2048, dim=args.emb_dim, num_classes=n_classes, device=str(device))

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    # OneCycle sincronizzato agli step_per_epoch (se fornito)
    if args.scheduler == "onecycle":
        steps_per_epoch = args.steps_per_epoch if (args.steps_per_epoch and args.steps_per_epoch > 0) else max(1, len(dl_train))
        lr_sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    else:
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    ce_loss = nn.CrossEntropyLoss()

    csv_path = normpath(args.out_dir, f"metrics_epoch_v3_seen.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","lr","supcon","ce","triplet","proto_rep","AUROC","AUPRC","Silhouette"])

    best_auroc = -1.0
    for ep in range(1, args.epochs+1):
        model.train()
        run_supcon=run_ce=run_triplet=run_proto=0.0; n_batches=0

        pbar = tqdm(dl_train, desc=f"Epoch {ep}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            e, p, logits = model(xb, return_proj=True)

            # SupCon (multi-head) + mixup opzionale
            if args.mixup:
                p_use, y_sup = supcon_mixup(p, yb, alpha=0.2)
            else:
                p_use, y_sup = p, yb
            loss_sup = supcon(p_use, y_sup)

            loss_ce = ce_loss(logits, yb)
            loss_trip = batch_hard_triplet(e, yb, margin=args.triplet_margin, queue=queue)

            with torch.no_grad():
                E_np = e.detach().cpu().numpy(); Y_np = yb.detach().cpu().numpy()
            protos_list = []
            for c in range(n_classes):
                if (Y_np == c).sum() > 0:
                    protos_list.append(torch.from_numpy(E_np[Y_np==c].mean(0)))
                else:
                    protos_list.append(torch.zeros(args.emb_dim))
            prototypes = torch.stack(protos_list, 0).to(device)
            loss_proto = prototype_repulsion_loss(prototypes, margin=2.0)

            total = args.lambda_supcon*loss_sup + args.lambda_ce*loss_ce + args.lambda_triplet*loss_trip + args.lambda_proto*loss_proto
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if lr_sched is not None and args.scheduler == "onecycle":
                lr_sched.step()

            with torch.no_grad():
                queue.update(e, yb)

            run_supcon += float(loss_sup.detach().cpu())
            run_ce     += float(loss_ce.detach().cpu())
            run_triplet+= float(loss_trip.detach().cpu())
            run_proto  += float(loss_proto.detach().cpu())
            n_batches  += 1

        if lr_sched is not None and args.scheduler != "onecycle":
            lr_sched.step()

        # ---- EVAL (estrazione embedding/logits)
        # 1) Train in modalità "eval" (no crop/aug) -> fit set stabile (niente double-dip su calib)
        ds_train_eval = ShipsearDataset(
            records=sp["train"][["filename","category"]].to_dict(orient="records"),
            root=args.data_dir, class_names=class_names, featurizer=featurizer, cache=cache,
            mode="eval", aug_spec=False, aug_wav=False,
            wav_dir=args.wav_dir, alt_wav_dir=args.alt_wav_dir
        )
        dl_train_eval = DataLoader(
            ds_train_eval, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, collate_fn=collate_batch, pin_memory=True
        )

        # 2) Estrai embedding
        E_tr_eval, L_tr_eval, y_tr_eval = extract_embeddings(model, dl_train_eval, device)  # FIT SET
        E_ca, L_ca, y_ca = extract_embeddings(model, dl_calib,  device)   # KNOWN (val)
        E_un, L_un, _    = extract_embeddings(model, dl_unseen, device)   # UNKNOWN (val)

        # silhouette su TRAIN (puoi lasciarla sul dl_train "train" o usare train_eval; cambia poco)
        sil = compute_silhouette(E_tr_eval, y_tr_eval)

        # Open-set ensemble + DEBUG per singolo score
        if E_ca.shape[0] > 0 and E_un.shape[0] > 0:
            # FIT su train_eval (disgiunto da calib) -> niente double-dipping
            auroc, auprc, (_, _), iso = eval_openset_ensemble(E_tr_eval, y_tr_eval, E_ca, L_ca, E_un, L_un)


            # ===== DEBUG: AUROC/AUPRC per score singoli =====
            def _single_scores_debug(E_tr, y_tr, E_cal, L_cal, E_un, L_un, ncls):
                protos = class_prototypes(E_tr, y_tr, ncls)
                VI = np.linalg.pinv(np.cov(E_tr.T) + 1e-4*np.eye(E_tr.shape[1]))
                lof = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.05).fit(E_tr)
                def pack(E, L):
                    return {
                        "mahal": normalize_01(mahal_scores(E, protos, VI)),
                        "cosine": normalize_01(cosine_scores(E, protos)),
                        "energy": normalize_01(energy_scores(L)),
                        "lof":    normalize_01(lof_scores(E, lof)),
                    }
                cal = pack(E_ca, L_ca); un = pack(E_un, L_un)
                y_true = np.concatenate([np.zeros_like(cal["mahal"]), np.ones_like(un["mahal"])])
                out = {}
                for k in ["mahal","cosine","energy","lof"]:
                    y_score = np.concatenate([cal[k], un[k]])
                    out[k] = (roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score))
                return out
            
            ncls = int(y_tr_eval.max())+1 if y_tr_eval.size>0 else 1
            dbg = _single_scores_debug(E_tr_eval, y_tr_eval, E_ca, L_ca, E_un, L_un, ncls)

            print(f"[DEBUG] AUROC/AUPRC per score -> "
                  f"mahal={dbg['mahal'][0]:.3f}/{dbg['mahal'][1]:.3f}, "
                  f"cos={dbg['cosine'][0]:.3f}/{dbg['cosine'][1]:.3f}, "
                  f"energy={dbg['energy'][0]:.3f}/{dbg['energy'][1]:.3f}, "
                  f"lof={dbg['lof'][0]:.3f}/{dbg['lof'][1]:.3f}")
            # ================================================

        else:
            auroc, auprc = float("nan"), float("nan")

        # Aggiorna temperature SupCon
        supcon.t = temp_sched.update(0.0 if np.isnan(sil) else sil)
        lr_now = lr_sched.get_last_lr()[0] if lr_sched is not None else args.lr

        # Log CSV
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                ep, f"{lr_now:.6f}",
                f"{run_supcon/max(1,n_batches):.4f}",
                f"{run_ce/max(1,n_batches):.4f}",
                f"{run_triplet/max(1,n_batches):.4f}",
                f"{run_proto/max(1,n_batches):.4f}",
                f"{auroc:.3f}" if not np.isnan(auroc) else "nan",
                f"{auprc:.3f}" if not np.isnan(auprc) else "nan",
                f"{sil:.3f}" if not np.isnan(sil) else "nan",
            ])

        print(f"\n--- Epoch {ep} ---  SupCon: {run_supcon/max(1,n_batches):.4f}  CE: {run_ce/max(1,n_batches):.4f}  Triplet: {run_triplet/max(1,n_batches):.4f}  Proto: {run_proto/max(1,n_batches):.4f}")
        print(f"AUROC: {auroc if not np.isnan(auroc) else 'nan'} | AUPRC: {auprc if not np.isnan(auprc) else 'nan'} | Silhouette: {sil if not np.isnan(sil) else 'nan'} | T: {supcon.t:.4f}")

        # Save best
        if not np.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            ckpt = {"model": model.state_dict(), "args": vars(args), "class_names": class_names, "epoch": ep, "best_auroc": best_auroc}
            torch.save(ckpt, normpath(args.out_dir, "models", "shipsear_supcon_v3_seen.pt"))

    # Save last
    ckpt = {"model": model.state_dict(), "args": vars(args), "class_names": class_names, "epoch": args.epochs, "best_auroc": best_auroc}
    torch.save(ckpt, normpath(args.out_dir, "models", "shipsear_supcon_v3_seenbest.pt"))

# -------------------------
# Argparse
# -------------------------

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default=".")
    p.add_argument("--meta_csv", type=str, required=True)
    p.add_argument("--splits_json", type=str, default="")
    p.add_argument("--out_dir", type=str, default="outputs")
    p.add_argument("--cache_dir", type=str, default="cache_mels_v3_noleak")
    p.add_argument("--wav_dir", type=str, default="shipsear_segments")
    p.add_argument("--alt_wav_dir", type=str, default="shipsear_raw")
    p.add_argument("--classes_per_batch", type=int, default=8,
               help="Numero di classi distinte per batch nel BalancedBatchSampler.")
    p.add_argument("--samples_per_class", type=int, default=8,
               help="Numero di esempi per classe per batch nel BalancedBatchSampler.")


    # Featurizer
    p.add_argument("--sr", type=int, default=22050)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--nfft1", type=int, default=1024)
    p.add_argument("--hop1", type=int, default=256)
    p.add_argument("--nfft2", type=int, default=2048)
    p.add_argument("--hop2", type=int, default=512)
    p.add_argument("--target_secs", type=float, default=5.0)
    p.add_argument("--in_ch", type=int, default=6, help="3=solo log-mel per scala; 6=log+Δ+ΔΔ")

    # Model
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--num_heads", type=int, default=3)

    # Training
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle","cosine"])
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_unseen", type=int, default=3)

    #calib e calcolo metriche

    p.add_argument("--calib_csv", type=str, default="",
               help="CSV di calibrazione (escluso dal training). Se fornito, sostituisce la calibrazione fold==5.")
    p.add_argument("--unseen_csv", type=str, default="",
               help="CSV degli unseen usato per AUROC durante il training. Se non fornito, usa meta/splits_json.")


    # Steps per epoch forzati
    p.add_argument("--steps_per_epoch", type=int, default=-1,
                   help="Numero di batch per epoca nel train (-1 usa tutta la epoch classica)")

    # Loss weights
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--lambda_supcon", type=float, default=1.0)
    p.add_argument("--lambda_ce", type=float, default=0.2)
    p.add_argument("--lambda_triplet", type=float, default=0.2)
    p.add_argument("--lambda_proto", type=float, default=0.1)
    p.add_argument("--triplet_margin", type=float, default=0.3)
    p.add_argument("--mixup", action="store_true")

    p.add_argument(
    "--isotonic_mode",
    choices=["none"],
    default="none",
    help="Calibrazione disattivata (scelta rigorosa per valutazione open-set senza leakage)."
)
    
    return p

def main():
    args = build_argparser().parse_args()
    train(args)

if __name__ == "__main__":
    
    main()
