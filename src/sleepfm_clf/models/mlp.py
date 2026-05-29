"""
Modelo MLP sobre embeddings SleepFM agregados (mean/std/p25/p75).

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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from ..config import DEVICE

_WARMUP   = 10   # épocas antes de activar el early stopping
_PATIENCE = 20   # épocas sin mejora en rolling mean antes de parar
_ROLL     = 3    # ventana del rolling mean

DEFAULTS = dict(
    hidden_dims=[256, 128],
    dropout=0.3,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=32,
    epochs=100,
)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Protocolo ────────────────────────────────────────────────────────────────

def load_data(df, window_size=None):
    from ..data import load_aggregated
    return load_aggregated(df, window_size=window_size)


def suggest_params(trial):
    return {
        "hidden_dims":  trial.suggest_categorical("hidden_dims", [
            [128], [256], [512], [1024],
            [256, 128], [512, 256], [1024, 512], [512, 256, 128],
        ]),
        "dropout":      trial.suggest_float("dropout",      0.1, 0.5),
        "lr":           trial.suggest_float("lr",           1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def train_fold(data, y, tr_idx, val_idx, params=None):
    X = data
    p = {**DEFAULTS, **(params or {})}
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    model = MLP(X_tr.shape[1], p["hidden_dims"], p["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"], eta_min=1e-6)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tr_dl = DataLoader(
        TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        batch_size=p["batch_size"], shuffle=True,
    )

    auc_history, best_roll, best_state, no_improve = [], -1.0, None, 0
    X_val_t = torch.tensor(X_val).to(DEVICE)

    pbar = tqdm(range(p["epochs"]), desc="  epoch", leave=False, dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        running_loss, n_batches = 0.0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
        try:
            auc = roc_auc_score(y_val, probs)
        except ValueError:
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
        best_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    try:
        best_auc = roc_auc_score(y_val, best_probs)
    except ValueError:
        best_auc = 0.5

    return best_auc, best_probs


def index_data(data, indices):
    return data[indices]


def concat_data(data_a, data_b):
    return np.vstack([data_a, data_b])
