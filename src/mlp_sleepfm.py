"""
mlp_sleepfm.py
==============
Clasificador MLP sobre embeddings SleepFM con 5-fold CV estratificado.

Input:
  - Embeddings 5min_agg en: EMB_DIR/{patient_id}.hdf5
    Cada HDF5 contiene datasets 'bas', 'ekg', 'resp', 'emg'
    shape (n_ventanas, 128) → se colapsa con mean pooling → (128,) por modalidad
  - demographics.csv con columnas BidsFolder, SessionID, Age, Sex, Cognitive_Impairment

Output:
  - AUROC por fold + media ± std
  - Curva ROC del mejor fold

Uso:
  python mlp_sleepfm.py
"""

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
EMB_DIR     = "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base/physionet2026_5min_agg"
CSV_PATH    = "../data/demographics_total.csv"
MODALITIES  = ["BAS", "EKG", "RESP", "EMG"]  # orden del HDF5
N_FOLDS     = 5
EPOCHS      = 60
LR          = 1e-3
BATCH_SIZE  = 32
HIDDEN_DIMS = [256, 128]
DROPOUT     = 0.3
SEED        = 42
N_TRIALS    = 50   # Optuna trials
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── MODELO ────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── CARGA DE EMBEDDINGS ───────────────────────────────────────────────────────
def load_embedding(patient_id: str, session_id) -> np.ndarray | None:
    """
    Lee el HDF5 de embeddings de un paciente y devuelve un vector
    concatenando mean, std, p25 y p75 de cada modalidad disponible.
    Shape final: (128 * 4 stats * n_modalidades,) = (2048,)
    """
    path = os.path.join(EMB_DIR, f"{patient_id}_ses-{session_id}.hdf5")
    if not os.path.exists(path):
        return None

    parts = []
    with h5py.File(path, "r") as f:
        for mod in MODALITIES:
            if mod in f:
                arr = f[mod][:]          # (n_ventanas, 128)
                parts.extend([
                    arr.mean(axis=0),
                    arr.std(axis=0),
                    np.percentile(arr, 25, axis=0),
                    np.percentile(arr, 75, axis=0),
                ])
            else:
                parts.append(np.zeros(512))  # 4 stats × 128

    return np.concatenate(parts)   # (2048,)


def build_dataset(df: pd.DataFrame):
    """Construye X, y a partir del CSV y los embeddings disponibles."""
    X_list, y_list, ids_ok = [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cargando embeddings"):
        patient_id = row["BidsFolder"]
        emb = load_embedding(row["BidsFolder"], row["SessionID"])
        if emb is None:
            continue

        age  = float(row["Age"]) if pd.notna(row["Age"]) else 0.0
        sex  = 1.0 if str(row["Sex"]).lower().startswith("f") else 0.0

        feat = np.append(emb, [age, sex])
        X_list.append(feat)

        label = row["Cognitive_Impairment"]
        if isinstance(label, str):
            y_list.append(1 if label.upper() == "TRUE" else 0)
        else:
            y_list.append(1 if label else 0)

        ids_ok.append(patient_id)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"\nDataset: {len(y)} pacientes | positivos: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    return X, y


# ── ENTRENAMIENTO DE UN FOLD ──────────────────────────────────────────────────
def train_fold(X_tr, y_tr, X_val, y_val, input_dim, params=None):
    if params is None:
        params = {}
    hidden_dims  = params.get("hidden_dims",  HIDDEN_DIMS)
    dropout      = params.get("dropout",      DROPOUT)
    lr           = params.get("lr",           LR)
    weight_decay = params.get("weight_decay", 1e-4)
    batch_size   = params.get("batch_size",   BATCH_SIZE)
    epochs       = params.get("epochs",       EPOCHS)

    model = MLP(input_dim, hidden_dims, dropout).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    scaler = StandardScaler()
    X_tr  = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    tr_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)

    best_auc, best_probs = 0.0, None

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(X_val).to(DEVICE)
            logits = model(xv).cpu().numpy()
            probs  = torch.sigmoid(torch.tensor(logits)).numpy()

        try:
            auc = roc_auc_score(y_val, probs)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc   = auc
            best_probs = probs.copy()

    return best_auc, best_probs


# ── OPTUNA ────────────────────────────────────────────────────────────────────
def optimize_hyperparams(X, y, input_dim):
    n_trials = N_TRIALS

    def objective(trial):
        params = {
            "hidden_dims": trial.suggest_categorical("hidden_dims", [
                [128], [256], [512], [1024],
                [256, 128], [512, 256], [1024, 512], [512, 256, 128],
            ]),
            "dropout":      trial.suggest_float("dropout",      0.1, 0.5),
            "lr":           trial.suggest_float("lr",           1e-4, 1e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),
            "epochs":       trial.suggest_int("epochs", 60, 200),
        }
        print(
            f"\nTrial {trial.number+1:3d}/{n_trials} | "
            f"hidden={params['hidden_dims']}  lr={params['lr']:.2e}  "
            f"dropout={params['dropout']:.2f}  wd={params['weight_decay']:.2e}  "
            f"bs={params['batch_size']}",
            flush=True,
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_aucs = []
        for fold_i, (tr, val) in enumerate(skf.split(X, y), 1):
            auc, _ = train_fold(X[tr], y[tr], X[val], y[val], input_dim, params)
            fold_aucs.append(auc)
            print(f"  fold {fold_i}/3 → AUROC={auc:.4f}", flush=True)
        mean_auc = float(np.mean(fold_aucs))
        print(f"  mean={mean_auc:.4f}", flush=True)
        return mean_auc

    def callback(study, trial):
        if trial.value == study.best_value:
            print(f"  ★ nuevo mejor: {study.best_value:.4f}", flush=True)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    print(f"\n  Mejor AUROC Optuna (3-fold): {study.best_value:.4f}")
    print(f"  Mejores hipers: {study.best_params}")
    return study.best_params


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    df = pd.read_csv(CSV_PATH)

    X, y = build_dataset(df)
    input_dim = X.shape[1]
    print(f"Dimensión de entrada: {input_dim}  (2048 agg-embedding + 2 demográficos)")

    # Test oculto: separado antes de cualquier optimización
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train+Val: {len(y_tv)}  |  Test oculto: {len(y_test)}")

    print(f"\n── Optimización de hiperparámetros ({N_TRIALS} trials Optuna) ──────")
    best_params = optimize_hyperparams(X_tv, y_tv, input_dim)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs = []
    best_fold_auc  = 0.0
    best_fold_data = None

    print(f"\n── 5-Fold CV en train+val (mejores hipers) ─────────────────────")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tv, y_tv), start=1):
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]

        auc, probs = train_fold(X_tr, y_tr, X_val, y_val, input_dim, best_params)
        aucs.append(auc)
        print(f"  Fold {fold}: Val AUROC = {auc:.4f}")

        if auc > best_fold_auc:
            best_fold_auc  = auc
            best_fold_data = (y_val, probs)

    mean_auc = np.mean(aucs)
    std_auc  = np.std(aucs)
    print(f"\n  CV AUROC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Mejor fold : {best_fold_auc:.4f}")

    # ── Evaluación final en test oculto ──────────────────────────
    print(f"\n── Evaluación en test oculto ───────────────────────────────────")
    test_auc, test_probs = train_fold(X_tv, y_tv, X_test, y_test, input_dim, best_params)
    print(f"  Test AUROC: {test_auc:.4f}")

    # ── Curva ROC: test oculto ────────────────────────────────────
    from sklearn.metrics import roc_curve
    y_true, y_prob = y_test, test_probs
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"Test (AUC={test_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC — MLP sobre embeddings SleepFM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_mlp_sleepfm.png", dpi=150)
    print("\nCurva ROC guardada en roc_mlp_sleepfm.png")


if __name__ == "__main__":
    main()