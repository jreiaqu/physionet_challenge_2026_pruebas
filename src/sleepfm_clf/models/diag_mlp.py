"""
diag_mlp.py
===========
Fase 1 del estudio de ablaciones: MLP sobre la representación de 128 dims extraída
de la cabeza congelada de `model_diagnosis` (spatial_pooling + BiLSTM + mean-pool,
ver `generate_diagnosis_embeddings.py`), + demografía (age, sex) — mismo protocolo
que el A0 de `mlp.py` sobre model_base, para comparación directa.

Requiere haber corrido antes:
  python generate_diagnosis_embeddings.py --dataset large --window-size 5min
"""
import os
import numpy as np

from .mlp import (  # reutiliza arquitectura y bucle de entrenamiento tal cual
    suggest_params, train_fold, index_data, concat_data, extract_ages,
)

CKPT_DIR = "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_diagnosis"


def load_data(df, window_size=None, dataset=None):
    from ..config import DEFAULT_WINDOW_SIZE, DEFAULT_DATASET
    window_size = window_size or DEFAULT_WINDOW_SIZE
    dataset = dataset or DEFAULT_DATASET
    npz_path = os.path.join(CKPT_DIR, f"diag_embed_{dataset}_{window_size}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Faltan embeddings de model_diagnosis en {npz_path}. Genéralos con: "
            f"python generate_diagnosis_embeddings.py --dataset {dataset} --window-size {window_size}"
        )
    store = np.load(npz_path)

    X_list, y_list = [], []
    for _, row in df.iterrows():
        key = f"{row['BidsFolder']}_ses-{row['SessionID']}"
        if key not in store:
            continue
        emb = store[key]                                    # (128,)
        age = float(row["Age"]) if row["Age"] == row["Age"] else 0.0
        sex = 1.0 if str(row["Sex"]).lower().startswith("f") else 0.0
        X_list.append(np.concatenate([emb, [age, sex]]))
        y_list.append(1 if str(row["Cognitive_Impairment"]).upper() == "TRUE" else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    print(f"\nDataset (diag_embed {window_size}): {len(y)} pacientes | "
          f"positivos: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    return X, y
