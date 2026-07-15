"""
Modelo LSTM sobre la secuencia temporal de ventanas SleepFM.

Cada ventana: 4 modalidades × 128 dims = 512 dims de entrada.
La secuencia completa (n_ventanas, 512) se pasa al LSTM; el último
estado oculto se concatena con los demográficos (age, sex) y se
clasifica con una cabeza lineal.

Protocolo requerido por run_model.py / evaluate.py:
  load_data(df)                          → (data, y)
  suggest_params(trial)                  → dict
  train_fold(data, y, tr_idx, val_idx, params) → (auc, probs)
  index_data(data, indices)              → data
  concat_data(data_a, data_b)            → data
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..config import DEVICE
from ..metrics import age_conditioned_auroc

_WARMUP   = 10   # épocas antes de activar el early stopping
_PATIENCE = 20   # épocas sin mejora en rolling mean antes de parar
_ROLL     = 3    # ventana del rolling mean

SEQ_INPUT_DIM = 512   # 4 modalidades × 128
DEMO_DIM      = 2     # age, sex

DEFAULTS = dict(
    hidden_size=128,
    num_layers=1,
    bidirectional=True,
    pooling="last",
    dropout=0.2,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=16,
    epochs=100,
    max_seq_len=300,   # ventanas máx por paciente; None = sin límite (5min≈73, 5s=4380)
)


class LSTMClassifier(nn.Module):
    def __init__(self, hidden_size, num_layers, dropout, bidirectional, pooling="last"):
        super().__init__()
        self.lstm = nn.LSTM(
            SEQ_INPUT_DIM, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_out + DEMO_DIM, lstm_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out, 1),
        )
        self.bidirectional = bidirectional
        self.pooling = pooling

    def forward(self, x, lengths, demo):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, (h_n, _) = self.lstm(packed)

        if self.pooling == "mean":
            # out: (batch, T_max, hidden*dirs) — media sobre pasos válidos
            out, _ = pad_packed_sequence(out_packed, batch_first=True)
            mask = (torch.arange(out.size(1), device=out.device)[None, :]
                    < lengths.to(out.device)[:, None])          # (batch, T_max)
            h = (out * mask.unsqueeze(-1)).sum(1) / lengths.to(out.device, dtype=out.dtype).unsqueeze(1)
        else:  # "last"
            # h_n: (num_layers * num_directions, batch, hidden_size)
            h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]

        return self.head(torch.cat([h, demo], dim=1)).squeeze(1)


class _SeqDataset(Dataset):
    def __init__(self, seqs, demos, labels):
        self.seqs   = seqs    # list de tensores (n_ventanas, 512)
        self.demos  = demos   # tensor (n, 2)
        self.labels = labels  # tensor (n,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.seqs[i], self.demos[i], self.labels[i]


def _collate(batch):
    seqs, demos, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)
    return padded, lengths, torch.stack(list(demos)), torch.stack(list(labels))


# ── Protocolo ────────────────────────────────────────────────────────────────

def load_data(df, window_size=None, dataset=None):
    from ..data import load_sequences
    return load_sequences(df, window_size=window_size, dataset=dataset)


def suggest_params(trial):
    return {
        "hidden_size":   trial.suggest_categorical("hidden_size",   [64, 128, 256]),
        "num_layers":    trial.suggest_int("num_layers", 1, 2),
        "bidirectional": trial.suggest_categorical("bidirectional",  [True, False]),
        "pooling":       trial.suggest_categorical("pooling", ["last", "mean"]),
        "dropout":       trial.suggest_float("dropout",      0.0, 0.4),
        "lr":            trial.suggest_float("lr",           1e-4, 1e-2, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size":    trial.suggest_categorical("batch_size", [8, 16, 32]),
    }


def train_fold(data, y, tr_idx, val_idx, params=None):
    seqs, demos = data
    ages_val = demos[val_idx, 0]   # age antes de StandardScaler (primer demográfico)
    p = {**DEFAULTS, **(params or {})}
    y_tr, y_val = y[tr_idx], y[val_idx]

    # Truncado de secuencias (crítico para 5s: 4380 ventanas → max_seq_len)
    msl = p["max_seq_len"]
    _clip = (lambda s: s[:msl]) if msl else (lambda s: s)

    # Normalización: ajuste sobre ventanas de entrenamiento
    seq_scaler  = StandardScaler().fit(np.vstack([_clip(seqs[i]) for i in tr_idx]))
    demo_scaler = StandardScaler().fit(demos[tr_idx])

    seqs_tr  = [torch.tensor(seq_scaler.transform(_clip(seqs[i])), dtype=torch.float32) for i in tr_idx]
    seqs_val = [torch.tensor(seq_scaler.transform(_clip(seqs[i])), dtype=torch.float32) for i in val_idx]
    demos_tr  = torch.tensor(demo_scaler.transform(demos[tr_idx]),  dtype=torch.float32)
    demos_val = torch.tensor(demo_scaler.transform(demos[val_idx]), dtype=torch.float32)

    tr_dl = DataLoader(
        _SeqDataset(seqs_tr, demos_tr, torch.tensor(y_tr)),
        batch_size=p["batch_size"], shuffle=True, collate_fn=_collate,
    )
    val_dl = DataLoader(
        _SeqDataset(seqs_val, demos_val, torch.tensor(y_val)),
        batch_size=p["batch_size"], shuffle=False, collate_fn=_collate,
    )

    model = LSTMClassifier(
        p["hidden_size"], p["num_layers"], p["dropout"], p["bidirectional"], p["pooling"]
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"], eta_min=1e-6)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    auc_history, best_roll, best_state, no_improve = [], -1.0, None, 0
    n_tr_batches = len(tr_dl)

    pbar = tqdm(range(p["epochs"]), desc="  epoch", leave=False, dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        running_loss, n_batches = 0.0, 0
        batch_bar = tqdm(tr_dl, desc=f"    e{epoch+1:03d}", leave=False,
                         total=n_tr_batches, dynamic_ncols=True)
        for seqs_b, lengths_b, demos_b, labels_b in batch_bar:
            seqs_b    = seqs_b.to(DEVICE)
            lengths_b = lengths_b.to(DEVICE)
            demos_b   = demos_b.to(DEVICE)
            labels_b  = labels_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seqs_b, lengths_b, demos_b), labels_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1
            batch_bar.set_postfix(loss=f"{running_loss/n_batches:.4f}")
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs_list = []
            for seqs_b, lengths_b, demos_b, _ in val_dl:
                logits = model(seqs_b.to(DEVICE), lengths_b.to(DEVICE), demos_b.to(DEVICE))
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        probs = np.concatenate(probs_list)
        auc = age_conditioned_auroc(y_val, probs, ages_val)
        if np.isnan(auc):
            auc = 0.5
        auc_history.append(auc)

        if (epoch + 1) >= _WARMUP and len(auc_history) >= _ROLL:
            roll = float(np.mean(auc_history[-_ROLL:]))
            if roll > best_roll:
                best_roll  = roll
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            pbar.set_postfix(
                loss=f"{running_loss/n_batches:.4f}",
                roll3=f"{roll:.4f}",
                pat=f"{no_improve}/{_PATIENCE}",
            )
            if no_improve >= _PATIENCE:
                break
        else:
            pbar.set_postfix(loss=f"{running_loss/n_batches:.4f}")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_probs_list = []
        for seqs_b, lengths_b, demos_b, _ in val_dl:
            logits = model(seqs_b.to(DEVICE), lengths_b.to(DEVICE), demos_b.to(DEVICE))
            best_probs_list.append(torch.sigmoid(logits).cpu().numpy())
    best_probs = np.concatenate(best_probs_list)
    best_auc = age_conditioned_auroc(y_val, best_probs, ages_val)
    if np.isnan(best_auc):
        best_auc = 0.5

    return best_auc, best_probs


def extract_ages(data):
    """Extrae edades del array de demográficos (primera columna, antes de scaling)."""
    _, demos = data
    return demos[:, 0]


def index_data(data, indices):
    seqs, demos = data
    return [seqs[i] for i in indices], demos[indices]


def concat_data(data_a, data_b):
    seqs_a, demos_a = data_a
    seqs_b, demos_b = data_b
    return seqs_a + seqs_b, np.vstack([demos_a, demos_b])
