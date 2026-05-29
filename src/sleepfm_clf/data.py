"""
data.py
=======
Carga embeddings SleepFM desde archivos HDF5 en dos formatos:
  load_aggregated() → vector 2050-dim por paciente (para MLP)
  load_sequences()  → secuencia (n_ventanas, 512) por paciente (para LSTM)
"""
import os
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from .config import EMB_DIRS, DEFAULT_WINDOW_SIZE, MODALITIES, MOD_DIM


def _hdf5_path(row, emb_dir):
    return os.path.join(emb_dir, f"{row['BidsFolder']}_ses-{row['SessionID']}.hdf5")


def _read_hdf5(path):
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        return {mod: f[mod][:] for mod in MODALITIES if mod in f}


def _parse_label(val):
    return 1 if str(val).upper() == "TRUE" or val is True else 0


def _demographics(row):
    age = float(row["Age"]) if pd.notna(row["Age"]) else 0.0
    sex = 1.0 if str(row["Sex"]).lower().startswith("f") else 0.0
    return np.array([age, sex], dtype=np.float32)


def load_aggregated(df, window_size=None):
    """
    Para MLP: cada paciente → vector (2050,).
    Concatena [mean, std, p25, p75] por modalidad (512 × 4 = 2048) + [age, sex].
    Devuelve (X: np.ndarray (n, 2050), y: np.ndarray (n,)).

    window_size: "5s" | "5min" | None (usa DEFAULT_WINDOW_SIZE)
    """
    emb_dir = EMB_DIRS[window_size or DEFAULT_WINDOW_SIZE]
    X_list, y_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando embeddings ({window_size or DEFAULT_WINDOW_SIZE})"):
        mods = _read_hdf5(_hdf5_path(row, emb_dir))
        if mods is None:
            continue
        parts = []
        for mod in MODALITIES:
            arr = mods.get(mod, np.zeros((1, MOD_DIM)))
            parts.extend([
                arr.mean(0),
                arr.std(0),
                np.percentile(arr, 25, 0),
                np.percentile(arr, 75, 0),
            ])
        X_list.append(np.concatenate(parts + [_demographics(row)]))
        y_list.append(_parse_label(row["Cognitive_Impairment"]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"\nDataset: {len(y)} pacientes | positivos: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    return X, y


def load_sequences(df, window_size=None):
    """
    Para LSTM: cada paciente → array (n_ventanas, 512).
    Las 4 modalidades se concatenan por ventana: [BAS_t, EKG_t, RESP_t, EMG_t].
    Devuelve ((seqs: list[np.ndarray], demos: np.ndarray (n, 2)), y: np.ndarray (n,)).

    window_size: "5s" | "5min" | None (usa DEFAULT_WINDOW_SIZE)
    """
    emb_dir = EMB_DIRS[window_size or DEFAULT_WINDOW_SIZE]
    seqs, demos, y_list = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando embeddings ({window_size or DEFAULT_WINDOW_SIZE} seq)"):
        mods = _read_hdf5(_hdf5_path(row, emb_dir))
        if mods is None:
            continue
        arrays = [mods.get(mod, np.zeros((1, MOD_DIM))) for mod in MODALITIES]
        n_win = min(a.shape[0] for a in arrays)
        seq = np.concatenate([a[:n_win] for a in arrays], axis=1).astype(np.float32)
        seqs.append(seq)
        demos.append(_demographics(row))
        y_list.append(_parse_label(row["Cognitive_Impairment"]))

    y = np.array(y_list, dtype=np.float32)
    print(f"\nDataset: {len(y)} pacientes | positivos: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    return (seqs, np.array(demos, dtype=np.float32)), y
