import torch

_BASE = "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base"

# Directorios de embeddings por resolución temporal
EMB_DIRS = {
    "5s":   f"{_BASE}/physionet2026",
    "5min": f"{_BASE}/physionet2026_5min_agg",
}

# Resolución por defecto (cambiar aquí para alternar globalmente)
DEFAULT_WINDOW_SIZE = "5min"

# Alias de compatibilidad
EMB_DIR = EMB_DIRS[DEFAULT_WINDOW_SIZE]

CSV_PATH   = "../data/demographics_total.csv"
MODALITIES = ["BAS", "EKG", "RESP", "EMG"]
MOD_DIM    = 128   # embedding dim por modalidad por ventana

N_FOLDS  = 5
N_TRIALS = 50
SEED     = 42
def _resolve_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            print(f"[WARNING] GPU detectada pero no operativa ({e}). Usando CPU.")
    return torch.device("cpu")

DEVICE = _resolve_device()
