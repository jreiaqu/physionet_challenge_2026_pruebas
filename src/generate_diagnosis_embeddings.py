"""
generate_diagnosis_embeddings.py
=================================
Fase 1 del estudio de ablaciones (ver report.md, "Hallazgo Fase 1"):
`model_diagnosis` NO es un encoder alternativo de señal cruda — es una cabeza
(spatial_pooling + BiLSTM + demo_embedding + disease_heads) fine-tuneada sobre los
embeddings YA CALCULADOS de `model_base`. Por tanto no hace falta reprocesar señal:
basta con pasar nuestros HDF5 de `model_base` (BAS/EKG/RESP/EMG por ventana) por la
cabeza congelada y quedarnos con la representación agrupada antes de `disease_heads`
(spatial_pooling → BiLSTM → mean-pool sobre pasos válidos). Esa es la "embedding
model_diagnosis" de 128 dims.

No se usa la rama `demo_embedding` (se entrenó con una normalización de edad/sexo que
no tenemos localmente documentada) — la demografía se añade después, en el propio
MLP de evaluación, igual que en el baseline A0 sobre model_base.

Salida: un .npz por (dataset, window_size) con {patient_key: vector(128,)}, en
  sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_diagnosis/diag_embed_{dataset}_{window_size}.npz

Uso:
  python generate_diagnosis_embeddings.py --dataset large --window-size 5min
  python generate_diagnosis_embeddings.py --dataset large --window-size 5s      # Fase 1b, condicional
  python generate_diagnosis_embeddings.py --dataset small --window-size 5min
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
from einops import rearrange
from torch.nn.utils import rnn as rnn_utils
from tqdm import tqdm

sys.path.append("../sleepFM/sleepfm-clinical/sleepfm")
from models.models import DiagnosisFinetuneFullLSTMCOXPHWithDemo  # noqa: E402

MODALITIES = ["BAS", "EKG", "RESP", "EMG"]  # orden irrelevante: spatial_pooling es
                                             # invariante al orden de canal (ver report.md)
CKPT_DIR = "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_diagnosis"
BASE_EMB_DIRS = {
    "small": {
        "5s":   "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base/physionet2026_small",
        "5min": "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base/physionet2026_small_5min_agg",
    },
    "large": {
        "5s":   "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base/physionet2026_large",
        "5min": "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base/physionet2026_large_5min_agg",
    },
}


def load_frozen_head(device):
    with open(os.path.join(CKPT_DIR, "config.json")) as f:
        config = json.load(f)
    mp = config["model_params"]
    model = DiagnosisFinetuneFullLSTMCOXPHWithDemo(
        embed_dim=mp["embed_dim"], num_heads=mp["num_heads"], num_layers=mp["num_layers"],
        num_classes=mp["num_classes"], pooling_head=mp["pooling_head"], dropout=0.0,
        max_seq_length=mp["max_seq_length"],
    )
    sd = torch.load(os.path.join(CKPT_DIR, "best.pth"), map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model.to(device)


@torch.no_grad()
def pooled_embedding(model, x, device):
    """Replica DiagnosisFinetuneFullLSTMCOXPHWithDemo.forward hasta antes de
    demo_embedding/disease_heads. x: (S, C=4, 128) float32, sin padding (batch=1)."""
    S, C, E = x.shape
    x = x.to(device).unsqueeze(0)               # (1, S, C, E) -> reordenar a (B,C,S,E)
    x = x.permute(0, 2, 1, 3)                    # (1, C, S, E)
    B = 1
    xt = rearrange(x, 'b c s e -> (b s) c e')
    mask_spatial = torch.zeros(B, S, C, dtype=torch.bool, device=device)
    mask_spatial = rearrange(mask_spatial, 'b t c -> (b t) c')

    xt = model.spatial_pooling(xt, mask_spatial)
    xt = xt.view(B, S, E)

    lengths = torch.tensor([S])
    packed = rnn_utils.pack_padded_sequence(xt, lengths, batch_first=True, enforce_sorted=False)
    packed_out, _ = model.lstm(packed)
    out, _ = rnn_utils.pad_packed_sequence(packed_out, batch_first=True)
    pooled = out[0, :S].mean(dim=0)              # (embed_dim,)
    return pooled.cpu().numpy()


def read_patient_embedding(emb_dir, patient_key):
    path = os.path.join(emb_dir, f"{patient_key}.hdf5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        mods = [f[m][:] for m in MODALITIES if m in f]
        if len(mods) != len(MODALITIES):
            return None
        n_win = min(m.shape[0] for m in mods)
        stacked = np.stack([m[:n_win] for m in mods], axis=1)  # (S, C, 128)
    return stacked.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["small", "large"], required=True)
    ap.add_argument("--window-size", choices=["5s", "5min"], required=True)
    ap.add_argument("--max-seq-len", type=int, default=6480,
                     help="tope de ventanas por paciente (igual al max_seq_length del "
                          "checkpoint; solo relevante para 5s)")
    args = ap.parse_args()

    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_frozen_head(device)

    csv_path = f"../dataset_{args.dataset}{'_balanced' if args.dataset == 'large' else ''}/demographics.csv"
    df = pd.read_csv(csv_path)
    emb_dir = BASE_EMB_DIRS[args.dataset][args.window_size]

    out = {}
    n_missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"diag-embed {args.dataset}/{args.window_size}"):
        key = f"{row['BidsFolder']}_ses-{row['SessionID']}"
        x = read_patient_embedding(emb_dir, key)
        if x is None:
            n_missing += 1
            continue
        if x.shape[0] > args.max_seq_len:
            x = x[:args.max_seq_len]
        x_t = torch.tensor(x, dtype=torch.float32)
        out[key] = pooled_embedding(model, x_t, device)

    print(f"{len(out)} pacientes procesados, {n_missing} sin embeddings model_base.")
    out_path = os.path.join(CKPT_DIR, f"diag_embed_{args.dataset}_{args.window_size}.npz")
    np.savez(out_path, **out)
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
