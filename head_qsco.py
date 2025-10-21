"""
head_qsco.py — Quantum Scoring Module (QSCO)
Versione migliorata per training unsupervised stabile e robusto.

Novità:
- Output 'known_score' (alto = campione visto / known)
- Generazione hard-negatives in unsupervised basata sugli autovettori
  della covarianza (oltre l’ellissoide principale)
- Supporto per allenamento supervisionato o non supervisionato
- Funzioni di normalizzazione integrate e riflessione dei negativi
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    import pennylane as qml
except Exception:
    qml = None


# ===============================================================
# UTILITY: generazione hard-negatives unsupervised
# ===============================================================

@torch.no_grad()
def _sample_hard_outliers(E: torch.Tensor, n_samples: int, inflate: float = 2.0, mode: str = "hard_eig") -> torch.Tensor:
    """
    Genera outlier sintetici 'difficili' a partire dagli embeddings E.
    Args:
        E: tensore (N, D)
        n_samples: numero di outlier da generare
        inflate: fattore di espansione rispetto all’ellissoide
        mode: 'hard_eig' (autovettori) o 'gauss' (rumore isotropo)
    """
    assert E.dim() == 2, "E deve avere forma (N, D)"
    N, D = E.shape
    mu = E.mean(0, keepdim=True)

    if mode == "gauss":
        sigma = E.std(0, keepdim=True).clamp_min(1e-6)
        out = mu + torch.randn(n_samples, D, device=E.device) * sigma * inflate
        return out

    # Covarianza e autovettori principali
    X = E - mu
    cov = (X.t() @ X) / max(1, N - 1)
    eigvals, eigvecs = torch.linalg.eigh(cov + 1e-6 * torch.eye(D, device=E.device))
    eigvals = eigvals.clamp(min=1e-8)

    k = min(D, 64)
    top_idx = torch.argsort(eigvals, descending=True)[:k]
    U = eigvecs[:, top_idx]               # (D,k)
    s = eigvals[top_idx].sqrt()           # (k,)

    z = torch.randn(n_samples, k, device=E.device)
    r = (z * s) * inflate
    out = mu + r @ U.t()

    # Riflettiamo alcuni punti per ottenere negativi "hard"
    idx = torch.randperm(n_samples, device=E.device)[: n_samples // 3]
    out[idx] = mu - (out[idx] - mu)
    return out


# ===============================================================
# QSCO HEAD — quantum-inspired scoring head
# ===============================================================

class QscoHead(nn.Module):
    """
    Quantum-inspired scoring head (alto score = 'known').
    Si può usare sia in modalità supervisionata che unsupervised.
    """

    def __init__(
        self,
        in_dim: int,
        n_qubits: Optional[int] = None,
        depth: int = 2,
        backend: str = "default.qubit",
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        if qml is None:
            raise ImportError("PennyLane è richiesto per QscoHead (pip install pennylane)")

        torch.manual_seed(seed)
        self.in_dim = int(in_dim)
        self.n_qubits = int(n_qubits or min(8, max(2, int(math.log2(in_dim)))))
        self.depth = int(depth)
        self.backend = backend
        self.seed = int(seed)

        # Preprocessing: normalizza e proietta in angoli per i qubit
        self.pre = nn.Sequential(
            nn.LayerNorm(self.in_dim),
            nn.Linear(self.in_dim, self.n_qubits),
        )
        # Proiezione finale per ottenere un logit scalare 'known'
        self.post = nn.Linear(self.n_qubits, 1)

        # Setup quantum device
        self.dev = qml.device(self.backend, wires=self.n_qubits, shots=None)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def circuit(angles, weights):
            # encoding RY + entanglement ring (depth L)
            for i in range(self.n_qubits):
                qml.RY(angles[i], wires=i)
            for _ in range(self.depth):
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
                for i in range(self.n_qubits):
                    qml.RY(weights[i], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit
        self.theta = nn.Parameter(torch.zeros(self.n_qubits, dtype=torch.float32))

    def forward(self, E: torch.Tensor) -> torch.Tensor:
        """Ritorna i logit del punteggio 'known'."""
        angles = self.pre(E)
        Z = []
        for b in range(E.size(0)):
            z = self.circuit(angles[b], self.theta)
            Z.append(torch.stack(z))
        Z = torch.stack(Z).to(dtype=angles.dtype)   
        logit_known = self.post(Z).squeeze(-1)
        return logit_known

    @torch.no_grad()
    def score(self, E: torch.Tensor) -> torch.Tensor:
        """Restituisce lo score 'known' in [0,1]."""
        logit = self.forward(E)
        return torch.sigmoid(logit)


# ===============================================================
# TRAINING QSCO (supervised / unsupervised)
# ===============================================================

def train_qsco(
    head: QscoHead,
    E_calib: torch.Tensor,
    y_calib: Optional[torch.Tensor] = None,
    *,
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    synth_ratio: float = 1.0,
    inflate: float = 2.0,
    unsup_mode: str = "hard_eig",
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Addestra il QSCO head.
    - supervised: usa y_calib (1=known, 0=novelty)
    - unsupervised: genera outlier sintetici e li tratta come 0
    """
    if device is not None:
        head.to(device)
        E_calib = E_calib.to(device)
        if y_calib is not None:
            y_calib = y_calib.to(device)

    head.train()
    opt = torch.optim.Adam([p for p in head.parameters() if p.requires_grad], lr=lr)
    bce = nn.BCEWithLogitsLoss()

    N = E_calib.size(0)
    best_loss, last_loss = float("inf"), float("inf")

    for ep in range(1, epochs + 1):
        if y_calib is None:
            # UNSUPERVISED
            n_syn = int(N * synth_ratio)
            E_syn = _sample_hard_outliers(E_calib, n_syn, inflate=inflate, mode=unsup_mode)
            X = torch.cat([E_calib, E_syn], dim=0)
            y = torch.cat([
                torch.ones(N, device=E_calib.device),
                torch.zeros(n_syn, device=E_calib.device)
            ], dim=0)
        else:
            # SUPERVISED
            X = E_calib
            y = (y_calib >= 0).float()

        # Mescoliamo i batch
        perm = torch.randperm(X.size(0), device=X.device)
        epoch_loss = 0.0
        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X[idx], y[idx]
            logit_known = head.forward(xb)
            loss = bce(logit_known, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item()) * idx.numel()

        epoch_loss /= X.size(0)
        best_loss = min(best_loss, epoch_loss)
        last_loss = epoch_loss

        if verbose:
            print(f"[QSCO] epoch {ep:03d}/{epochs:02d}  loss={epoch_loss:.4f}")

    return best_loss, last_loss


# ===============================================================
# INFERENZA: restituisce punteggi "known" [0,1]
# ===============================================================

@torch.no_grad()
def infer_qsco_scores(head: QscoHead, E: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """Restituisce il vettore di 'known_score' in [0,1]."""
    head.eval()
    if device is not None:
        head.to(device)
        E = E.to(device)
    scores = head.score(E)
    return scores.detach().cpu().view(-1)
