"""
ablation.py
===========
Módulo genérico de ablaciones sobre embeddings SleepFM congelados. Cubre las
Familias A (pooling), B (pérdida), C (modalidades) y D (demografía) del brief de
ablaciones con una única arquitectura configurable, para no reimplementar el
protocolo Optuna/CV por cada variante — un `build(config)` produce un objeto con
el mismo protocolo duck-typed que `mlp.py`/`lstm.py`
(load_data, suggest_params, train_fold, index_data, concat_data, extract_ages),
así que `evaluate.run_optuna`/`run_cv` no necesitan tocarse.

config (dict, fijo por experimento — NO se optimiza con Optuna):
  pooling:      "stats" | "attention" | "hierarchical"   (Familia A)
  modalities:   subset de ["BAS","EKG","RESP","EMG"]     (Familia C1/C2)
  gating:       bool  — combina modalidades con pesos aprendidos en vez de concat (C3)
  modality_dropout: float en [0,1) — dropout de modalidad completa durante train (C4)
  demo_level:   "none" | "age_sex" | "full"              (Familia D)
  loss:         "bce" | "focal" | "paired"                (Familia B)
  block_size:   int, solo para pooling="hierarchical" (nº ventanas 5s por bloque; 60=5min)
  max_seq_len:  tope de ventanas por paciente (protección VRAM en 5s)
  caisr:        dict opcional {"mode": "pool_by_stage"|"clinical_indices"|"both",
                               "source": "caisr"|"human"} — Familias F/G

Nota Familia C4 (modality dropout) y F (fusión CAISR): la Regla Dura de marcas humanas
se respeta a nivel de `run_queue.py`/`queue.yaml`: `caisr.source="human"` SOLO se usa en
los experimentos de diagnóstico G (comparación profesor) y en el profesor de destilación H
— nunca en una config marcada como desplegable.
"""
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from ..config import DEVICE, EMB_DIRS
from ..metrics import age_conditioned_auroc

MODALITIES = ["BAS", "EKG", "RESP", "EMG"]
EMBED_DIM = 128
_WARMUP, _PATIENCE, _ROLL = 10, 20, 3
_MAX_EPOCHS = 60   # tope duro del protocolo de ablaciones (brief: "máx 60 épocas")

DEFAULTS = dict(
    hidden_dims=[256, 128], dropout=0.3, lr=1e-3, weight_decay=1e-4,
    batch_size=32, epochs=_MAX_EPOCHS,
    focal_gamma=2.0,       # solo si loss=="focal"
    pair_margin=0.2,       # solo si loss=="paired"
)


# ── Demografía ────────────────────────────────────────────────────────────────

_RACE_LEVELS = ["White", "Black", "Others", "Unavailable"]
_ETH_LEVELS = ["Not Hispanic", "Hispanic", "Unavailable"]


def _demo_vector(row, level):
    if level == "none":
        return np.zeros(0, dtype=np.float32)
    age = float(row["Age"]) if not pd_isna(row["Age"]) else 0.0
    sex = 1.0 if str(row["Sex"]).lower().startswith("f") else 0.0
    if level == "age_sex":
        return np.array([age, sex], dtype=np.float32)
    # "full": + BMI + one-hot race/ethnicity (agrupando categorías raras en "Others")
    bmi = float(row["BMI"]) if not pd_isna(row["BMI"]) else -1.0  # -1 = ausente (flag implícito)
    race = str(row.get("Race", "Unavailable"))
    race = race if race in _RACE_LEVELS else "Others"
    eth = str(row.get("Ethnicity", "Unavailable"))
    eth = eth if eth in _ETH_LEVELS else "Unavailable"
    race_oh = [1.0 if race == r else 0.0 for r in _RACE_LEVELS]
    eth_oh = [1.0 if eth == e else 0.0 for e in _ETH_LEVELS]
    return np.array([age, sex, bmi] + race_oh + eth_oh, dtype=np.float32)


def pd_isna(v):
    try:
        return v != v  # NaN != NaN
    except Exception:
        return False


def demo_dim(level):
    return {"none": 0, "age_sex": 2, "full": 3 + len(_RACE_LEVELS) + len(_ETH_LEVELS)}[level]


# ── Carga de datos ──────────────────────────────────────────────────────────

def _read_patient_modalities(emb_dir, key, modalities, max_seq_len=None):
    """max_seq_len: si se da, trunca AL LEER (slicing h5py, no tras cargar) — evita
    tener las secuencias completas sin truncar en RAM. Con 5s y sin tope, la mediana de
    ventanas por paciente es ~10860 (~15h de grabación); cargar eso para ~990 pacientes
    x 4 modalidades x 128 dims en float32 son ~24 GB solo en `data["seqs"]`, antes de
    tocar la GPU. Ver report.md / conversación sobre rendimiento."""
    import os, h5py
    path = os.path.join(emb_dir, f"{key}.hdf5")
    if not os.path.exists(path):
        return None
    out = {}
    with h5py.File(path, "r") as f:
        for m in modalities:
            if m in f:
                arr = f[m][:max_seq_len] if max_seq_len else f[m][:]
                out[m] = arr.astype(np.float32)
    return out if out else None


def load_data(df, window_size=None, dataset=None, config=None):
    """
    Devuelve (bundle, y). bundle = {
      "mode": config["pooling"],
      "seqs": list[dict[modality->np.ndarray(n_win,128)]]  (siempre; barato de mantener),
      "demo": np.ndarray (n, demo_dim),
      "ages": np.ndarray (n,),
      "caisr": np.ndarray (n, caisr_dim) | None,
      "keys": list[str],
    }
    """
    from .. import config as cfgmod
    window_size = window_size or cfgmod.DEFAULT_WINDOW_SIZE
    dataset = dataset or cfgmod.DEFAULT_DATASET
    modalities = config.get("modalities", MODALITIES)
    demo_level = config.get("demo_level", "age_sex")
    max_seq_len = config.get("max_seq_len")
    emb_dir = EMB_DIRS[dataset][window_size]

    seqs, demos, ages, keys, y_list = [], [], [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Cargando ({window_size}, ablation)"):
        key = f"{row['BidsFolder']}_ses-{row['SessionID']}"
        mods = _read_patient_modalities(emb_dir, key, modalities, max_seq_len)
        if mods is None:
            continue
        seqs.append(mods)
        demos.append(_demo_vector(row, demo_level))
        ages.append(float(row["Age"]))
        keys.append(key)
        y_list.append(1 if str(row["Cognitive_Impairment"]).upper() == "TRUE" else 0)

    caisr_feats = None
    caisr_cfg = config.get("caisr")
    if caisr_cfg and caisr_cfg.get("mode") in ("clinical_indices", "both"):
        from ..caisr_features import batch_clinical_indices
        caisr_feats = batch_clinical_indices(df, keys, dataset, source=caisr_cfg.get("source", "caisr"))

    stage_labels = None
    needs_stage = config.get("pooling") == "caisr_stage" or (caisr_cfg and caisr_cfg.get("mode") in ("pool_by_stage", "both"))
    if needs_stage:
        from .. import caisr_features as cfmod
        site_by_key = {f"{r['BidsFolder']}_ses-{r['SessionID']}": r["SiteID"] for _, r in df.iterrows()}
        window_seconds = 5.0 if window_size == "5s" else 300.0
        stage_source = (caisr_cfg or {}).get("source", "caisr")
        stage_labels = []
        for i, key in enumerate(keys):
            site = site_by_key.get(key)
            ep = cfmod.stage_epochs(dataset, site, key, stage_source) if site else None
            n_win = min(m.shape[0] for m in seqs[i].values())
            stage_labels.append(cfmod.window_stage_labels(ep, n_win, window_seconds))

    bundle = {
        "mode": config.get("pooling", "stats"),
        "config": config,
        "seqs": seqs,
        "stage_labels": stage_labels,
        "demo": np.array(demos, dtype=np.float32) if demo_level != "none" else np.zeros((len(seqs), 0), dtype=np.float32),
        "ages": np.array(ages, dtype=np.float32),
        "caisr": caisr_feats,
        "keys": keys,
        "window_size": window_size,
        "dataset": dataset,
    }
    y = np.array(y_list, dtype=np.float32)
    print(f"\nDataset: {len(y)} pacientes | positivos: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    return bundle, y


def index_data(data, indices):
    idx = list(indices)
    out = dict(data)
    out["seqs"] = [data["seqs"][i] for i in idx]
    out["demo"] = data["demo"][idx]
    out["ages"] = data["ages"][idx]
    out["keys"] = [data["keys"][i] for i in idx]
    if data["caisr"] is not None:
        out["caisr"] = data["caisr"][idx]
    if data.get("stage_labels") is not None:
        out["stage_labels"] = [data["stage_labels"][i] for i in idx]
    return out


def concat_data(data_a, data_b):
    out = dict(data_a)
    out["seqs"] = data_a["seqs"] + data_b["seqs"]
    out["demo"] = np.vstack([data_a["demo"], data_b["demo"]]) if data_a["demo"].shape[1] else data_a["demo"]
    out["ages"] = np.concatenate([data_a["ages"], data_b["ages"]])
    out["keys"] = data_a["keys"] + data_b["keys"]
    if data_a["caisr"] is not None:
        out["caisr"] = np.vstack([data_a["caisr"], data_b["caisr"]])
    if data_a.get("stage_labels") is not None:
        out["stage_labels"] = data_a["stage_labels"] + data_b["stage_labels"]
    return out


def extract_ages(data):
    return data["ages"]


# ── Pooling modules ───────────────────────────────────────────────────────────

class AttnPool(nn.Module):
    """Attention pooling O(n): una query aprendida atiende sobre toda la secuencia."""
    def __init__(self, dim=EMBED_DIM):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T) bool, True = posición válida
        scores = (x @ self.query) * self.scale             # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        w = torch.softmax(scores, dim=1).unsqueeze(-1)      # (B, T, 1)
        w = torch.nan_to_num(w, nan=0.0)
        return (x * w).sum(dim=1)                            # (B, D)


class HierPool(nn.Module):
    """Pooling jerárquico: bloques de `block_size` ventanas -> mean-pool -> AttnPool corto."""
    def __init__(self, dim=EMBED_DIM, block_size=60):
        super().__init__()
        self.block_size = block_size
        self.block_attn = AttnPool(dim)

    def forward(self, x, mask):
        B, T, D = x.shape
        bs = self.block_size
        n_blocks = (T + bs - 1) // bs
        pad_len = n_blocks * bs - T
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
            mask = F.pad(mask, (0, pad_len))
        x = x.view(B, n_blocks, bs, D)
        mask_b = mask.view(B, n_blocks, bs)
        cnt = mask_b.sum(dim=2, keepdim=True).clamp(min=1)
        block_means = (x * mask_b.unsqueeze(-1)).sum(dim=2) / cnt   # (B, n_blocks, D)
        block_valid = mask_b.any(dim=2)                              # (B, n_blocks)
        return self.block_attn(block_means, block_valid)


def _stats_pool_np(seq):
    """[mean,std,p25,p75] parameter-free — para pooling="stats" (paridad con A0/mlp.py)."""
    return np.concatenate([seq.mean(0), seq.std(0), np.percentile(seq, 25, 0), np.percentile(seq, 75, 0)])


def _stats_extended_pool_np(seq):
    """Familia A3: [mean,std,p25,p75,min,max,skew,kurtosis,delta] — 9*128=1152-dim/modalidad.
    delta = media(2ª mitad) - media(1ª mitad) de la secuencia (deriva temporal cruda)."""
    mean, std = seq.mean(0), seq.std(0)
    std_safe = np.where(std > 1e-8, std, 1e-8)
    z = (seq - mean) / std_safe
    skew = (z ** 3).mean(0)
    kurt = (z ** 4).mean(0) - 3.0
    half = max(seq.shape[0] // 2, 1)
    delta = seq[half:].mean(0) - seq[:half].mean(0) if seq.shape[0] > 1 else np.zeros_like(mean)
    return np.concatenate([mean, std, np.percentile(seq, 25, 0), np.percentile(seq, 75, 0),
                            seq.min(0), seq.max(0), skew, kurt, delta])


_STAGE_CODES = [1, 2, 3, 4, 5]  # N3, N2, N1, REM, Wake


def _stage_pool_np(seq, labels):
    """Mean-pool por cubo de fase CAISR/humana (Familia F1): 5 cubos × 128 = 640-dim.
    Cubo vacío (fase ausente en ese paciente) -> vector cero, es una señal informativa
    en sí misma (p.ej. "sin REM registrado")."""
    n = min(seq.shape[0], len(labels))
    out = []
    for code in _STAGE_CODES:
        mask = labels[:n] == code
        if mask.any():
            out.append(seq[:n][mask].mean(0))
        else:
            out.append(np.zeros(EMBED_DIM, dtype=np.float32))
    return np.concatenate(out)


# ── Pérdidas ──────────────────────────────────────────────────────────────────

def _focal_loss(logits, targets, pos_weight, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.where(targets == 1, p, 1 - p)
    w = torch.where(targets == 1, pos_weight, torch.ones_like(pos_weight))
    return (w * (1 - pt) ** gamma * ce).mean()


def _paired_same_age_loss(logits, targets, ages, margin=0.2, delta=2.0, max_pairs=4096):
    """Surrogate directo de age_conditioned_auroc: hinge sobre pares (pos,neg) con |Δedad|<=delta."""
    pos = (targets == 1).nonzero(as_tuple=True)[0]
    neg = (targets == 0).nonzero(as_tuple=True)[0]
    if len(pos) == 0 or len(neg) == 0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    age_diff = (ages[pos].unsqueeze(1) - ages[neg].unsqueeze(0)).abs()
    valid = (age_diff <= delta).nonzero(as_tuple=False)
    if valid.shape[0] == 0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    if valid.shape[0] > max_pairs:
        sel = torch.randperm(valid.shape[0], device=valid.device)[:max_pairs]
        valid = valid[sel]
    pi, ni = pos[valid[:, 0]], neg[valid[:, 1]]
    diff = logits[ni] - logits[pi] + margin
    return F.relu(diff).mean()


# ── Red ───────────────────────────────────────────────────────────────────────

class AblationNet(nn.Module):
    def __init__(self, modalities, pooling, per_mod_dim, demo_d, caisr_d, gating,
                 hidden_dims, dropout, block_size=60):
        super().__init__()
        self.modalities = modalities
        self.pooling = pooling
        self.per_mod_dim = per_mod_dim
        self.gating = gating
        if pooling == "attention":
            self.poolers = nn.ModuleDict({m: AttnPool(EMBED_DIM) for m in modalities})
        elif pooling == "hierarchical":
            self.poolers = nn.ModuleDict({m: HierPool(EMBED_DIM, block_size) for m in modalities})
        else:
            self.poolers = None  # "stats": ya viene pooleado desde numpy

        if gating:
            self.gate = nn.Linear(per_mod_dim * len(modalities), len(modalities))
            combined_dim = per_mod_dim
        else:
            combined_dim = per_mod_dim * len(modalities)

        in_dim = combined_dim + demo_d + caisr_d
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.head = nn.Sequential(*layers)

    def forward(self, mod_inputs, demo, caisr, modality_dropout_mask=None):
        """mod_inputs: dict modality -> tensor.
           - pooling=="stats": ya son vectores (B, per_mod_dim).
           - si no: (padded (B,T,128), mask (B,T))."""
        vecs = []
        for m in self.modalities:
            if self.pooling in ("stats", "stats_extended", "caisr_stage"):
                v = mod_inputs[m]
            else:
                x, mask = mod_inputs[m]
                v = self.poolers[m](x, mask)
            vecs.append(v)
        stacked = torch.stack(vecs, dim=1)   # (B, n_mod, per_mod_dim)
        if modality_dropout_mask is not None:
            stacked = stacked * modality_dropout_mask.unsqueeze(-1)

        if self.gating:
            gate_logits = self.gate(stacked.flatten(1))
            w = torch.softmax(gate_logits, dim=1).unsqueeze(-1)   # (B, n_mod, 1)
            combined = (stacked * w).sum(dim=1)
        else:
            combined = stacked.flatten(1)

        parts = [combined]
        if demo.shape[1] > 0:
            parts.append(demo)
        if caisr is not None and caisr.shape[1] > 0:
            parts.append(caisr)
        x = torch.cat(parts, dim=1)
        return self.head(x).squeeze(1)


class _SeqDataset(Dataset):
    def __init__(self, seqs, demo, caisr, ages, labels, modalities, max_seq_len):
        self.seqs, self.demo, self.caisr, self.ages, self.labels = seqs, demo, caisr, ages, labels
        self.modalities, self.max_seq_len = modalities, max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {}
        for m in self.modalities:
            arr = self.seqs[i].get(m, np.zeros((1, EMBED_DIM), dtype=np.float32))
            if self.max_seq_len:
                arr = arr[:self.max_seq_len]
            item[m] = torch.tensor(arr, dtype=torch.float32)
        demo = torch.tensor(self.demo[i], dtype=torch.float32)
        caisr = torch.tensor(self.caisr[i], dtype=torch.float32) if self.caisr is not None else torch.zeros(0)
        return item, demo, caisr, self.ages[i], self.labels[i]


def _collate(batch, modalities):
    items, demos, caisrs, ages, labels = zip(*batch)
    mod_out = {}
    for m in modalities:
        seqs = [it[m] for it in items]
        lengths = torch.tensor([s.shape[0] for s in seqs])
        padded = pad_sequence(seqs, batch_first=True)
        T = padded.shape[1]
        mask = torch.arange(T)[None, :] < lengths[:, None]
        mod_out[m] = (padded, mask)
    return (mod_out, torch.stack(demos), torch.stack(list(caisrs)),
            torch.tensor(ages, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32))


_HIDDEN_DIMS = {"128": [128], "256": [256], "512": [512], "256-128": [256, 128], "512-256": [512, 256]}


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
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def _apply_modality_dropout(n, n_mod, p, device):
    if p <= 0:
        return None
    keep = torch.ones(n, n_mod, device=device)
    if n_mod > 1:
        drop_mod = torch.randint(0, n_mod, (n,), device=device)
        drop_flag = (torch.rand(n, device=device) < p)
        keep[torch.arange(n, device=device)[drop_flag], drop_mod[drop_flag]] = 0.0
    return keep


def train_fold(data, y, tr_idx, val_idx, params=None):
    config = data["config"]
    modalities = config.get("modalities", MODALITIES)
    pooling = config.get("pooling", "stats")
    demo_level = config.get("demo_level", "age_sex")
    loss_type = config.get("loss", "bce")
    gating = config.get("gating", False)
    mod_dropout_p = config.get("modality_dropout", 0.0)
    block_size = config.get("block_size", 60)
    max_seq_len = config.get("max_seq_len", None)

    p = {**DEFAULTS, **(params or {})}
    p["epochs"] = min(p.get("epochs", _MAX_EPOCHS), _MAX_EPOCHS)
    y_tr, y_val = y[tr_idx], y[val_idx]
    ages_val = data["ages"][val_idx]

    # Familia H (destilación privilegiada, condicional): "profesor" = probabilidades OOF
    # de un modelo entrenado con marcas HUMANAS (nunca usadas como input aquí, solo como
    # etiqueta blanda destino). Solo soportado con pooling en _PRECOMPUTED_POOLINGS (F3).
    distill_alpha = float(config.get("distill_alpha", 0.0))
    teacher_probs_full = config.get("teacher_probs")  # np.ndarray alineado a data["seqs"], o None
    teacher_tr_t = None
    if distill_alpha > 0 and teacher_probs_full is not None:
        teacher_tr_t = torch.tensor(teacher_probs_full[tr_idx], dtype=torch.float32).to(DEVICE)

    demo_d = data["demo"].shape[1]
    caisr_d = data["caisr"].shape[1] if data["caisr"] is not None else 0
    _PRECOMPUTED_POOLINGS = ("stats", "stats_extended", "caisr_stage")
    _PER_MOD_DIMS = {"stats": 512, "stats_extended": 9 * EMBED_DIM, "caisr_stage": 640}
    per_mod_dim = _PER_MOD_DIMS.get(pooling, EMBED_DIM)

    # Escalado de demo/caisr (age/sex/bmi/etc, y clinical indices) — fit en train
    demo_scaler = StandardScaler().fit(data["demo"][tr_idx]) if demo_d > 0 else None
    caisr_scaler = StandardScaler().fit(data["caisr"][tr_idx]) if caisr_d > 0 else None

    def scale_demo(idx):
        return demo_scaler.transform(data["demo"][idx]) if demo_d > 0 else data["demo"][idx]

    def scale_caisr(idx):
        return caisr_scaler.transform(data["caisr"][idx]) if caisr_d > 0 else np.zeros((len(idx), 0))

    if pooling in _PRECOMPUTED_POOLINGS:
        # Vectores (n, per_mod_dim) por modalidad, precomputados en numpy (barato, sin grad).
        mod_arrays = {}
        for m in modalities:
            if pooling == "stats":
                arr = np.stack([
                    _stats_pool_np(data["seqs"][i].get(m, np.zeros((1, EMBED_DIM), dtype=np.float32)))
                    for i in range(len(data["seqs"]))
                ])
            elif pooling == "stats_extended":
                arr = np.stack([
                    _stats_extended_pool_np(data["seqs"][i].get(m, np.zeros((1, EMBED_DIM), dtype=np.float32)))
                    for i in range(len(data["seqs"]))
                ])
            else:  # "caisr_stage"
                arr = np.stack([
                    _stage_pool_np(data["seqs"][i].get(m, np.zeros((1, EMBED_DIM), dtype=np.float32)),
                                   data["stage_labels"][i])
                    for i in range(len(data["seqs"]))
                ])
            scaler = StandardScaler().fit(arr[tr_idx])
            mod_arrays[m] = scaler.transform(arr)

        def make_tensors(idx):
            mod_t = {m: torch.tensor(mod_arrays[m][idx], dtype=torch.float32) for m in modalities}
            return mod_t, torch.tensor(scale_demo(idx), dtype=torch.float32), \
                torch.tensor(scale_caisr(idx), dtype=torch.float32)

        mod_tr, demo_tr, caisr_tr = make_tensors(tr_idx)
        mod_val, demo_val, caisr_val = make_tensors(val_idx)
        mod_tr = {m: v.to(DEVICE) for m, v in mod_tr.items()}
        mod_val = {m: v.to(DEVICE) for m, v in mod_val.items()}
        demo_tr, demo_val = demo_tr.to(DEVICE), demo_val.to(DEVICE)
        caisr_tr, caisr_val = caisr_tr.to(DEVICE), caisr_val.to(DEVICE)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(DEVICE)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
        ages_tr_t = torch.tensor(data["ages"][tr_idx], dtype=torch.float32).to(DEVICE)

        model = AblationNet(modalities, pooling, per_mod_dim, demo_d, caisr_d, gating,
                             p["hidden_dims"], p["dropout"]).to(DEVICE)

        n_tr = len(tr_idx)
        batch_size = p["batch_size"]

        def iterate_batches():
            perm = torch.randperm(n_tr, device=DEVICE)
            n_full = (n_tr // batch_size) * batch_size
            for s in range(0, n_full, batch_size):
                yield perm[s:s + batch_size]
            rem = n_tr - n_full
            if rem >= 2:
                yield perm[n_full:]

        def forward_batch(idx_t, mod_src, demo_src, caisr_src):
            mdd_mask = _apply_modality_dropout(len(idx_t), len(modalities), mod_dropout_p, DEVICE) if mod_dropout_p > 0 else None
            mods = {m: mod_src[m][idx_t] for m in modalities}
            return model(mods, demo_src[idx_t], caisr_src[idx_t], mdd_mask)

    else:
        ds_tr = _SeqDataset([data["seqs"][i] for i in tr_idx], scale_demo(tr_idx),
                             scale_caisr(tr_idx) if caisr_d else None, data["ages"][tr_idx], y_tr,
                             modalities, max_seq_len)
        ds_val = _SeqDataset([data["seqs"][i] for i in val_idx], scale_demo(val_idx),
                              scale_caisr(val_idx) if caisr_d else None, data["ages"][val_idx], y_val,
                              modalities, max_seq_len)
        collate = lambda b: _collate(b, modalities)
        # Secuencias sin truncar (mediana ~10860 ventanas/paciente en 5s, ver queue.yaml):
        # el cuello de botella no es la GPU sino el padding/collation en CPU bloqueando
        # entre lotes. num_workers>0 + persistent_workers prepara el siguiente lote en
        # paralelo mientras la GPU procesa el actual; pin_memory acelera el H2D transfer.
        # No cambia ni un solo dato de entrada — es una descripción distinta de CÓMO se
        # entrega el mismo tensor a la GPU.
        _nw = min(4, os.cpu_count() or 1)
        dl_tr = DataLoader(ds_tr, batch_size=p["batch_size"], shuffle=True, collate_fn=collate,
                            drop_last=True, num_workers=_nw, pin_memory=(DEVICE.type == "cuda"),
                            persistent_workers=_nw > 0)
        dl_val = DataLoader(ds_val, batch_size=p["batch_size"], shuffle=False, collate_fn=collate,
                             num_workers=_nw, pin_memory=(DEVICE.type == "cuda"),
                             persistent_workers=_nw > 0)

        model = AblationNet(modalities, pooling, per_mod_dim, demo_d, caisr_d, gating,
                             p["hidden_dims"], p["dropout"], block_size).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=p["epochs"], eta_min=1e-6)
    pos_weight = torch.tensor([(y_tr == 0).sum() / max((y_tr == 1).sum(), 1)], device=DEVICE)
    # Precisión mixta (fp16 autocast + GradScaler): reduce a la mitad memoria/tiempo de
    # los matmuls en GPU sin tocar los datos de entrada (Tensor Cores en la RTX 3060).
    # También alivia la presión de VRAM que se veía con secuencias largas sin truncar.
    _amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=_amp)

    def compute_loss(logits, targets, ages_batch=None, teacher_batch=None):
        if loss_type == "focal":
            base = _focal_loss(logits, targets, pos_weight, p["focal_gamma"])
        elif loss_type == "paired":
            base = _paired_same_age_loss(logits, targets, ages_batch, p["pair_margin"])
        else:
            base = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        if teacher_batch is not None:
            soft = F.binary_cross_entropy_with_logits(logits, teacher_batch)
            return (1 - distill_alpha) * base + distill_alpha * soft
        return base

    auc_history, best_roll, best_state, no_improve = [], -1.0, None, 0
    pbar = tqdm(range(p["epochs"]), desc="  epoch", leave=False, dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        running_loss, n_batches = 0.0, 0

        if pooling in _PRECOMPUTED_POOLINGS:
            for idx_t in iterate_batches():
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=_amp):
                    logits = forward_batch(idx_t, mod_tr, demo_tr, caisr_tr)
                    teacher_batch = teacher_tr_t[idx_t] if teacher_tr_t is not None else None
                    loss = compute_loss(logits, y_tr_t[idx_t], ages_tr_t[idx_t], teacher_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item(); n_batches += 1
        else:
            for mods_b, demo_b, caisr_b, ages_b, labels_b in dl_tr:
                mods_b = {m: (x.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True))
                          for m, (x, mask) in mods_b.items()}
                demo_b, caisr_b = demo_b.to(DEVICE, non_blocking=True), caisr_b.to(DEVICE, non_blocking=True)
                ages_b, labels_b = ages_b.to(DEVICE, non_blocking=True), labels_b.to(DEVICE, non_blocking=True)
                mdd_mask = _apply_modality_dropout(labels_b.shape[0], len(modalities), mod_dropout_p, DEVICE) if mod_dropout_p > 0 else None
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=_amp):
                    logits = model(mods_b, demo_b, caisr_b, mdd_mask)
                    loss = compute_loss(logits, labels_b, ages_b)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item(); n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=_amp):
            if pooling in _PRECOMPUTED_POOLINGS:
                probs = torch.sigmoid(forward_batch(torch.arange(len(val_idx), device=DEVICE), mod_val, demo_val, caisr_val)).float().cpu().numpy()
            else:
                probs_list = []
                for mods_b, demo_b, caisr_b, ages_b, labels_b in dl_val:
                    mods_b = {m: (x.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True))
                              for m, (x, mask) in mods_b.items()}
                    logits = model(mods_b, demo_b.to(DEVICE, non_blocking=True), caisr_b.to(DEVICE, non_blocking=True))
                    probs_list.append(torch.sigmoid(logits).float().cpu().numpy())
                probs = np.concatenate(probs_list)

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
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=_amp):
        if pooling in _PRECOMPUTED_POOLINGS:
            best_probs = torch.sigmoid(forward_batch(torch.arange(len(val_idx), device=DEVICE), mod_val, demo_val, caisr_val)).float().cpu().numpy()
        else:
            probs_list = []
            for mods_b, demo_b, caisr_b, ages_b, labels_b in dl_val:
                mods_b = {m: (x.to(DEVICE, non_blocking=True), mask.to(DEVICE, non_blocking=True))
                          for m, (x, mask) in mods_b.items()}
                logits = model(mods_b, demo_b.to(DEVICE, non_blocking=True), caisr_b.to(DEVICE, non_blocking=True))
                probs_list.append(torch.sigmoid(logits).float().cpu().numpy())
            best_probs = np.concatenate(probs_list)
    best_auc = age_conditioned_auroc(y_val, best_probs, ages_val)
    if np.isnan(best_auc):
        best_auc = 0.5
    return best_auc, best_probs


def build(config):
    """Factory: devuelve un objeto con el protocolo model_module, con `config` ya
    'horneado' (fijo, no optimizado por Optuna) vía closures."""
    class _Module:
        pass
    m = _Module()
    m.load_data = lambda df, window_size=None, dataset=None: load_data(df, window_size, dataset, config)
    m.suggest_params = suggest_params
    m.resolve_params = resolve_params
    m.train_fold = train_fold
    m.index_data = index_data
    m.concat_data = concat_data
    m.extract_ages = extract_ages
    m.config = config
    return m
