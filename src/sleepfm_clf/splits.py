"""
splits.py
=========
Carga los splits persistidos en splits/ (generados una vez por build_splits.py) y
los traduce a índices posicionales sobre un DataFrame dado, para que TODOS los
experimentos del estudio de ablaciones se evalúen sobre exactamente los mismos
pacientes.
"""
import json
import os
import numpy as np

_SPLITS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "splits")


def _key(row):
    return f"{row['BidsFolder']}_ses-{row['SessionID']}"


def _key_to_pos(df):
    return {_key(row): i for i, row in df.reset_index(drop=True).iterrows()}


def load_kfold_indices(df, path=None):
    """Devuelve lista de (train_idx, val_idx) np.ndarray alineados a df (orden actual)."""
    path = path or os.path.join(_SPLITS_DIR, "kfold5_large.json")
    with open(path) as f:
        payload = json.load(f)
    key2pos = _key_to_pos(df)
    folds = []
    for fold in payload["folds"]:
        tr = np.array([key2pos[k] for k in fold["train"] if k in key2pos])
        val = np.array([key2pos[k] for k in fold["val"] if k in key2pos])
        folds.append((tr, val))
    return folds


def load_loso_indices(df, path=None):
    """Devuelve lista de (site, train_idx, val_idx) np.ndarray alineados a df."""
    path = path or os.path.join(_SPLITS_DIR, "loso_large.json")
    with open(path) as f:
        payload = json.load(f)
    key2pos = _key_to_pos(df)
    out = []
    for split in payload["splits"]:
        tr = np.array([key2pos[k] for k in split["train"] if k in key2pos])
        val = np.array([key2pos[k] for k in split["val"] if k in key2pos])
        out.append((split["site"], tr, val))
    return out


def load_small_eval_keys(path=None):
    """Claves de dataset_small sin solape con dataset_large_balanced (evita fuga)."""
    path = path or os.path.join(_SPLITS_DIR, "small_eval_keys.json")
    with open(path) as f:
        payload = json.load(f)
    return set(payload["keys"])


def small_eval_mask(df):
    """Máscara booleana sobre df (dataset_small) que selecciona solo pacientes sin solape."""
    keys = load_small_eval_keys()
    return df.apply(lambda r: _key(r) in keys, axis=1).values


def load_usable_df(dataset="large"):
    """
    Carga demographics.csv filtrado a los pacientes con embeddings model_base
    (mismo universo sobre el que se construyeron kfold5_large.json/loso_large.json —
    ver build_splits.py). SIEMPRE usar esta función (no pd.read_csv directo) antes de
    pasar un DataFrame a cualquier model_module.load_data en run_queue.py, para que
    los índices de los splits persistidos sigan alineados con las filas reales.
    """
    import pandas as pd
    csv_path = f"../dataset_{dataset}{'_balanced' if dataset == 'large' else ''}/demographics.csv"
    df = pd.read_csv(csv_path).reset_index(drop=True)
    if dataset != "large":
        return df
    emb_dir = os.path.join(_SPLITS_DIR, "..", "sleepFM", "sleepfm-clinical", "sleepfm",
                            "checkpoints", "model_base", "physionet2026_large_5min_agg")
    has_emb = df.apply(lambda r: os.path.exists(
        os.path.join(emb_dir, f"{r['BidsFolder']}_ses-{r['SessionID']}.hdf5")), axis=1)
    return df[has_emb].reset_index(drop=True)
