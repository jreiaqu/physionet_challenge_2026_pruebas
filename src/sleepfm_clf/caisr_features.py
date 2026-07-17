"""
caisr_features.py
==================
Extrae índices clínicos y etiquetas de fase por época a partir de las anotaciones
CAISR (algorítmicas) o humanas (expertas), para las Familias F (fusión temprana
test-safe) y G (diagnóstico del hueco humano↔CAISR) del brief de ablaciones.

Codebook verificado en disco con `edfio` (ver report.md, sección "Anotaciones CAISR
y humanas") — usar SIEMPRE edfio, no mne.io.read_raw_edf (que resamplea a una
frecuencia común y corrompe los canales de códigos discretos, al forzar todos los
canales -de frecuencias de muestreo distintas- a un único sfreq).

  arousal_{expert,caisr}: 2 Hz,  {0,1}
  limb_{expert,caisr}:    1 Hz,  {0,1,2}
  resp_{expert,caisr}:    1 Hz,  {0..5} (0 = sin evento)
  stage_{expert,caisr}:   1/30 Hz (época de 30s), {1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=relleno}

REGLA DURA: `source="human"` SOLO se usa para el profesor de comparación de la
Familia G / destilación H — nunca en una config marcada como desplegable (ver
queue.yaml, columna `deployable`).
"""
import os
import numpy as np
import edfio

_BASE_DIRS = {
    "large": "../dataset_large_balanced",
    "small": "../dataset_small",
}
_SOURCE_SUFFIX = {"caisr": ("algorithmic_annotations", "caisr"), "human": ("human_annotations", "expert")}

N_CLINICAL_FEATS = 10  # efficiency, waso_min, tst_min, pct_n1, pct_n2, pct_n3, pct_rem, arousal_idx, ahi, plmi


def _annotation_path(dataset, site, key, source):
    subdir, suffix = _SOURCE_SUFFIX[source]
    return os.path.join(_BASE_DIRS[dataset], subdir, site, f"{key}_{suffix}_annotations.edf")


def _read_channels(path):
    """Devuelve dict canal->(np.ndarray, sfreq) o None si el fichero no existe."""
    if not os.path.exists(path):
        return None
    edf = edfio.read_edf(path)
    return {sig.label: (sig.data, sig.sampling_frequency) for sig in edf.signals}


def _events_per_hour(sig, sfreq):
    """Cuenta flancos de subida (0->>0) de un canal de eventos discretos."""
    if sig is None or len(sig) == 0:
        return 0.0
    binary = (sig > 0).astype(int)
    diff = np.diff(binary, prepend=0)
    n_events = np.count_nonzero(diff == 1)
    hours = len(sig) / sfreq / 3600.0
    return n_events / hours if hours > 0 else 0.0


def clinical_indices(dataset, site, key, source="caisr"):
    """
    Devuelve np.ndarray(10,): [efficiency, waso_min, tst_min, pct_n1, pct_n2, pct_n3,
    pct_rem, arousal_idx, ahi, plmi]. Vector de ceros si falta el fichero (paciente
    sin anotación — el modelo debe tolerarlo, igual que en test real).
    """
    path = _annotation_path(dataset, site, key, source)
    chans = _read_channels(path)
    if chans is None:
        return np.zeros(N_CLINICAL_FEATS, dtype=np.float32)

    stage_key = "stage_caisr" if source == "caisr" else "stage_expert"
    arousal_key = "arousal_caisr" if source == "caisr" else "arousal_expert"
    resp_key = "resp_caisr" if source == "caisr" else "resp_expert"
    limb_key = "limb_caisr" if source == "caisr" else "limb_expert"

    stage, _ = chans.get(stage_key, (np.array([]), 1.0))
    valid = stage[stage < 9.0]
    if len(valid) > 0:
        n_total = len(valid)
        n_sleep = np.count_nonzero(valid < 5)           # N1,N2,N3,REM (Wake=5)
        efficiency = n_sleep / n_total
        tst_min = n_sleep * 30.0 / 60.0
        pct_n1 = np.mean(valid == 3)
        pct_n2 = np.mean(valid == 2)
        pct_n3 = np.mean(valid == 1)
        pct_rem = np.mean(valid == 4)
        sleep_idx = np.where(valid < 5)[0]
        onset = sleep_idx[0] if len(sleep_idx) > 0 else 0
        waso_epochs = np.count_nonzero(valid[onset:] == 5)
        waso_min = waso_epochs * 30.0 / 60.0
    else:
        efficiency = tst_min = pct_n1 = pct_n2 = pct_n3 = pct_rem = waso_min = 0.0

    arousal_sig, arousal_sf = chans.get(arousal_key, (None, 1.0))
    resp_sig, resp_sf = chans.get(resp_key, (None, 1.0))
    limb_sig, limb_sf = chans.get(limb_key, (None, 1.0))
    arousal_idx = _events_per_hour(arousal_sig, arousal_sf)
    ahi = _events_per_hour(resp_sig, resp_sf)
    plmi = _events_per_hour(limb_sig, limb_sf)

    return np.array([efficiency, waso_min, tst_min, pct_n1, pct_n2, pct_n3, pct_rem,
                      arousal_idx, ahi, plmi], dtype=np.float32)


def batch_clinical_indices(df, keys, dataset, source="caisr"):
    """df debe tener SiteID indexado por la misma posición que `keys` (mismo orden)."""
    site_by_key = {f"{r['BidsFolder']}_ses-{r['SessionID']}": r["SiteID"] for _, r in df.iterrows()}
    out = np.zeros((len(keys), N_CLINICAL_FEATS), dtype=np.float32)
    for i, key in enumerate(keys):
        site = site_by_key.get(key)
        if site is None:
            continue
        out[i] = clinical_indices(dataset, site, key, source)
    return out


def stage_epochs(dataset, site, key, source="caisr"):
    """np.ndarray de códigos de fase a resolución de época (30s), o None si falta el fichero."""
    path = _annotation_path(dataset, site, key, source)
    chans = _read_channels(path)
    if chans is None:
        return None
    stage_key = "stage_caisr" if source == "caisr" else "stage_expert"
    stage, _ = chans.get(stage_key, (np.array([]), 1.0))
    return stage


def window_stage_labels(stage_ep, n_windows, window_seconds):
    """
    Asigna a cada ventana de embedding (0..n_windows-1, de `window_seconds` cada una,
    empezando en t=0 de la grabación) el código de fase de la época de 30s en la que
    cae su punto medio. Devuelve np.ndarray(n_windows,) con códigos {1..5}; 0 = fase
    desconocida (fuera de rango / relleno 9 / fichero ausente).
    """
    out = np.zeros(n_windows, dtype=np.int64)
    if stage_ep is None or len(stage_ep) == 0:
        return out
    for w in range(n_windows):
        mid_t = (w + 0.5) * window_seconds
        ep_idx = int(mid_t // 30.0)
        if 0 <= ep_idx < len(stage_ep):
            code = stage_ep[ep_idx]
            out[w] = int(code) if code < 9.0 else 0
    return out
