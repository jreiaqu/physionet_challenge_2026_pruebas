"""
coxph.py
========
Familia E del brief de ablaciones: cabeza CoxPH sobre embeddings SleepFM
CONGELADOS (representación agregada mean/std/p25/p75, igual que A0/mlp.py),
usando `Time_to_Event` (E1: cabeza CoxPH) — hoy ignorado por el resto del estudio.

Target de supervivencia (ver report.md, Fase 0):
  event    = Cognitive_Impairment (1 = diagnóstico observado)
  duration = Time_to_Event   si event==1  (censurado en el diagnóstico)
             Time_to_Last_Visit si event==0  (censurado en el último seguimiento)

La red produce un risk score (sin sigmoid) entrenado con verosimilitud parcial de
Cox (aproximación de Breslow para empates, igual que
`sleepFM/sleepfm-clinical/sleepfm/pipeline/finetune_diagnosis_coxph.py::cox_ph_loss`,
adaptada aquí a una única salida en vez de ~1065 fenotipos simultáneos). Para
poder reutilizar `age_conditioned_auroc` (que solo necesita un ranking, no
probabilidades calibradas) el risk score se pasa por sigmoid antes de devolverlo —
transformación monótona, no cambia el ranking ni el AUROC.
"""
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..config import DEVICE
from ..metrics import age_conditioned_auroc
from ..data import load_aggregated

_WARMUP, _PATIENCE, _ROLL = 10, 20, 3
_MAX_EPOCHS = 60

DEFAULTS = dict(hidden_dims=[256, 128], dropout=0.3, lr=1e-3, weight_decay=1e-4,
                batch_size=32, epochs=_MAX_EPOCHS)


class _RiskMLP(nn.Module):
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


def _cox_ph_loss(risk, duration, event):
    """Verosimilitud parcial de Cox, aproximación de Breslow. event: 0/1 float tensor."""
    order = torch.argsort(duration, descending=True)
    risk_o, event_o = risk[order], event[order]
    log_cumsum = torch.logcumsumexp(risk_o, dim=0)
    denom = event_o.sum().clamp(min=1.0)
    return -((risk_o - log_cumsum) * event_o).sum() / denom


def build(config=None):
    """Factory opcional (Familia E2): config={"aux_bce_weight": float en [0,1]} añade un
    término BCE(sigmoid(risk), y) ponderado junto a la verosimilitud de Cox — Time_to_Event
    como auxiliar en vez de único objetivo. aux_bce_weight=0 (default) = E1 puro."""
    aux_w = (config or {}).get("aux_bce_weight", 0.0)

    class _Module:
        pass
    m = _Module()
    m.load_data = load_data
    m.suggest_params = suggest_params
    m.resolve_params = resolve_params
    m.index_data = index_data
    m.concat_data = concat_data
    m.extract_ages = extract_ages
    m.train_fold = lambda data, y, tr, val, params=None: train_fold(data, y, tr, val, params, aux_bce_weight=aux_w)
    return m


def load_data(df, window_size=None, dataset=None):
    X, y = load_aggregated(df, window_size=window_size, dataset=dataset)
    # Recomputar duration/event alineados a las filas que sí tienen embeddings
    # (load_aggregated ya filtra silenciosamente los pacientes sin HDF5).
    df = df.reset_index(drop=True)
    durations, events = [], []
    j = 0
    from ..config import EMB_DIRS, DEFAULT_WINDOW_SIZE, DEFAULT_DATASET
    import os
    emb_dir = EMB_DIRS[dataset or DEFAULT_DATASET][window_size or DEFAULT_WINDOW_SIZE]
    for _, row in df.iterrows():
        path = os.path.join(emb_dir, f"{row['BidsFolder']}_ses-{row['SessionID']}.hdf5")
        if not os.path.exists(path):
            continue
        event = 1.0 if str(row["Cognitive_Impairment"]).upper() == "TRUE" else 0.0
        duration = row["Time_to_Event"] if event == 1.0 else row["Time_to_Last_Visit"]
        duration = float(duration) if pd.notna(duration) else 0.0
        durations.append(duration)
        events.append(event)
    data = {"X": X, "duration": np.array(durations, dtype=np.float32),
            "event": np.array(events, dtype=np.float32)}
    return data, y


def index_data(data, indices):
    idx = list(indices)
    return {"X": data["X"][idx], "duration": data["duration"][idx], "event": data["event"][idx]}


def concat_data(data_a, data_b):
    return {"X": np.vstack([data_a["X"], data_b["X"]]),
            "duration": np.concatenate([data_a["duration"], data_b["duration"]]),
            "event": np.concatenate([data_a["event"], data_b["event"]])}


def extract_ages(data):
    return data["X"][:, -2]  # penúltima columna, igual que mlp.py (age antes de escalar)


_HIDDEN_DIMS = {"128": [128], "256": [256], "256-128": [256, 128], "512-256": [512, 256]}


def resolve_params(params):
    params = dict(params)
    if "hidden_dims" in params and isinstance(params["hidden_dims"], str):
        params["hidden_dims"] = _HIDDEN_DIMS[params["hidden_dims"]]
    return params


def suggest_params(trial):
    return {
        "hidden_dims": _HIDDEN_DIMS[trial.suggest_categorical("hidden_dims", list(_HIDDEN_DIMS.keys()))],
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),  # lotes grandes: menos ruido en el ranking de Cox
    }


def train_fold(data, y, tr_idx, val_idx, params=None, aux_bce_weight=0.0):
    X = data["X"]
    ages_val = X[val_idx, -2]
    p = {**DEFAULTS, **(params or {})}
    p["epochs"] = min(p.get("epochs", _MAX_EPOCHS), _MAX_EPOCHS)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_val = scaler.transform(X[val_idx])
    dur_tr = torch.tensor(data["duration"][tr_idx], dtype=torch.float32).to(DEVICE)
    ev_tr = torch.tensor(data["event"][tr_idx], dtype=torch.float32).to(DEVICE)
    y_tr_t = torch.tensor(y[tr_idx], dtype=torch.float32).to(DEVICE)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val = y[val_idx]

    model = _RiskMLP(X_tr.shape[1], p["hidden_dims"], p["dropout"]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"], eta_min=1e-6)
    # Mismo criterio que mlp.py/lstm.py/ablation.py: pos_weight=n_neg/n_pos del propio
    # fold de train, para que el término auxiliar BCE (E2) no quede sin ponderar por
    # clase como el resto de pérdidas del estudio.
    y_tr = y[tr_idx]
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], dtype=torch.float32).to(DEVICE)

    n_tr = len(tr_idx)
    batch_size = min(p["batch_size"], n_tr)  # Cox necesita >=2 eventos por lote: lotes grandes

    auc_history, best_roll, best_state, no_improve = [], -1.0, None, 0
    pbar = tqdm(range(p["epochs"]), desc="  epoch", leave=False, dynamic_ncols=True)
    for epoch in pbar:
        model.train()
        perm = torch.randperm(n_tr, device=DEVICE)
        running_loss, n_batches = 0.0, 0
        n_full = (n_tr // batch_size) * batch_size or n_tr
        for s in range(0, n_full, batch_size):
            idx = perm[s:s + batch_size]
            if ev_tr[idx].sum() < 1:   # sin eventos en el lote: verosimilitud de Cox indefinida
                continue
            optimizer.zero_grad()
            risk = model(X_tr_t[idx])
            loss = _cox_ph_loss(risk, dur_tr[idx], ev_tr[idx])
            if aux_bce_weight > 0:
                bce = nn.functional.binary_cross_entropy_with_logits(risk, y_tr_t[idx], pos_weight=pos_weight)
                loss = (1 - aux_bce_weight) * loss + aux_bce_weight * bce
            loss.backward()
            optimizer.step()
            running_loss += loss.item(); n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
        auc = age_conditioned_auroc(y_val, probs, ages_val)
        if np.isnan(auc):
            auc = 0.5
        auc_history.append(auc)

        if (epoch + 1) >= _WARMUP and len(auc_history) >= _ROLL:
            roll = float(np.mean(auc_history[-_ROLL:]))
            if roll > best_roll:
                best_roll, best_state, no_improve = roll, copy.deepcopy(model.state_dict()), 0
            else:
                no_improve += 1
            pbar.set_postfix(loss=f"{running_loss/max(n_batches,1):.4f}", roll3=f"{roll:.4f}", pat=f"{no_improve}/{_PATIENCE}")
            if no_improve >= _PATIENCE:
                break
        else:
            pbar.set_postfix(loss=f"{running_loss/max(n_batches,1):.4f}")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_probs = torch.sigmoid(model(X_val_t)).cpu().numpy()
    best_auc = age_conditioned_auroc(y_val, best_probs, ages_val)
    if np.isnan(best_auc):
        best_auc = 0.5
    return best_auc, best_probs
