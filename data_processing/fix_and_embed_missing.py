"""
fix_and_embed_missing.py
========================
Repara los 201 pacientes sin embedding:
  1. Normaliza nombres de canal a minúsculas en HDF5 no vacíos con case incorrecto
  2. Re-convierte a HDF5 los EDF que generaron ficheros vacíos
  3. Genera embeddings SleepFM solo para los pacientes reparados
  4. Verifica el resultado final

Uso:
    python fix_and_embed_missing.py
"""

import glob
import h5py
import json
import numpy as np
import os
import shutil
import subprocess
import sys
import tempfile
import time
from loguru import logger

# ── RUTAS ─────────────────────────────────────────────────────────────────────
BASE        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HDF5_DIR    = os.path.join(BASE, "dataset_small/physiological_data/hdf5")
EDF_DIR     = os.path.join(BASE, "dataset_small/physiological_data")
EMB_BASE    = os.path.join(BASE, "sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base")
EMB_5S      = os.path.join(EMB_BASE, "physionet2026_small")
EMB_5MIN    = os.path.join(EMB_BASE, "physionet2026_small_5min_agg")
MODEL_PATH  = EMB_BASE
CHANNEL_GRP = os.path.join(BASE, "src/configs/channel_groups.json")
SLEEPFM_PKG = os.path.join(BASE, "sleepFM/sleepfm-clinical/sleepfm")
PYTHON      = sys.executable
MISSING_TXT = os.path.join(os.path.dirname(__file__), "missing_patients.txt")
# ──────────────────────────────────────────────────────────────────────────────


def get_missing_patients():
    """Devuelve lista de nombres (sin .hdf5) de pacientes sin embedding."""
    have = {os.path.splitext(f)[0] for f in os.listdir(EMB_5S) if f.endswith(".hdf5")}
    all_in = {os.path.splitext(f)[0] for f in os.listdir(HDF5_DIR) if f.endswith(".hdf5")}
    return sorted(all_in - have)


def normalize_hdf5_channels(src_path, dst_path):
    """Copia el HDF5 renombrando todos los datasets a minúsculas."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        for key in src.keys():
            new_key = key.lower()
            src.copy(key, dst, name=new_key)


def reconvert_edf(name):
    """Re-convierte un EDF a HDF5 usando EDFToHDF5Converter."""
    site = None
    for s in ("I0002", "I0006", "S0001"):
        if s in name:
            site = s
            break
    if site is None:
        logger.warning(f"No se encontró site para {name}")
        return False

    edf_path = os.path.join(EDF_DIR, site, f"{name}.edf")
    if not os.path.exists(edf_path):
        logger.error(f"EDF no encontrado: {edf_path}")
        return False

    hdf5_path = os.path.join(HDF5_DIR, f"{name}.hdf5")
    try:
        sys.path.insert(0, os.path.join(BASE, "data_processing"))
        from preprocessing_raw_data import EDFToHDF5Converter
        converter = EDFToHDF5Converter(
            root_dir=os.path.dirname(edf_path),
            target_dir=HDF5_DIR,
            resample_rate=256,
            num_threads=1,
        )
        converter.convert(edf_path, hdf5_path)
        size = os.path.getsize(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            n_channels = len(f.keys())
        logger.info(f"Re-convertido {name}: {n_channels} canales, {size/1e6:.1f} MB")
        return n_channels > 0
    except Exception as e:
        logger.error(f"Error re-convirtiendo {name}: {e}")
        return False


def fix_hdf5_files(missing):
    """
    Para cada paciente faltante con HDF5 no vacío:
      - Si tiene canales con case incorrecto → normaliza a minúsculas
      - Si está vacío → re-convierte desde EDF
    """
    fixed = []
    skipped_no_ekg = []

    channel_groups = json.load(open(CHANNEL_GRP))
    all_valid_channels = set()
    for chans in channel_groups.values():
        all_valid_channels.update(chans)

    for name in missing:
        hdf5_path = os.path.join(HDF5_DIR, f"{name}.hdf5")

        with h5py.File(hdf5_path, "r") as f:
            keys = list(f.keys())

        if len(keys) == 0:
            logger.info(f"[VACÍO] {name} → re-convirtiendo desde EDF...")
            ok = reconvert_edf(name)
            if ok:
                # Normalizar a minúsculas también
                tmp = hdf5_path + ".tmp"
                normalize_hdf5_channels(hdf5_path, tmp)
                os.replace(tmp, hdf5_path)
                fixed.append(name)
            else:
                logger.warning(f"No se pudo re-convertir {name}")
        else:
            # Verificar si al normalizar a minúsculas tendrá todas las modalidades
            lower_keys = {k.lower() for k in keys}
            modalities_found = {}
            for mod, chans in channel_groups.items():
                found = [c for c in chans if c in lower_keys]
                modalities_found[mod] = found

            missing_mods = [m for m, c in modalities_found.items() if len(c) == 0]
            if missing_mods:
                logger.warning(
                    f"[SKIP] {name}: sin canales para {missing_mods} incluso tras normalizar"
                )
                skipped_no_ekg.append((name, missing_mods))
                continue

            logger.info(f"[FIX]  {name} → normalizando canales a minúsculas...")
            tmp = hdf5_path + ".tmp"
            normalize_hdf5_channels(hdf5_path, tmp)
            os.replace(tmp, hdf5_path)
            fixed.append(name)

    logger.info(f"\nFicheros reparados: {len(fixed)}")
    logger.info(f"Ficheros irrecuperables (sin modalidad): {len(skipped_no_ekg)}")
    for name, mods in skipped_no_ekg:
        logger.info(f"  {name}: sin {mods}")

    return fixed


def generate_embeddings_for(names):
    """Genera embeddings solo para la lista de nombres dada."""
    sys.path.insert(0, SLEEPFM_PKG)
    sys.path.insert(0, os.path.join(BASE, "src"))

    from utils import load_config, load_data, count_parameters
    from models.dataset import SetTransformerDataset, collate_fn
    from models.models import SetTransformer
    import torch

    config = load_config(os.path.join(MODEL_PATH, "config.json"))
    channel_groups = load_data(CHANNEL_GRP)
    embed_dim = config["embed_dim"]

    hdf5_paths = [os.path.join(HDF5_DIR, f"{n}.hdf5") for n in names
                  if os.path.exists(os.path.join(HDF5_DIR, f"{n}.hdf5"))]

    logger.info(f"Generando embeddings para {len(hdf5_paths)} ficheros...")

    dataset = SetTransformerDataset(config, channel_groups, hdf5_paths=hdf5_paths, split="test")
    if len(dataset) == 0:
        logger.warning("Dataset vacío tras indexación — ningún fichero tiene todas las modalidades")
        return 0

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=4, shuffle=False, collate_fn=collate_fn
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SetTransformer(
        config["in_channels"], config["patch_size"], embed_dim,
        config["num_heads"], config["num_layers"],
        pooling_head=config["pooling_head"], dropout=0.0
    )
    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)
    ckpt = torch.load(os.path.join(MODEL_PATH, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    os.makedirs(EMB_5S, exist_ok=True)
    os.makedirs(EMB_5MIN, exist_ok=True)

    processed = 0
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_data, mask_list, file_paths, dset_names_list, chunk_starts = batch
            bas, resp, ekg, emg = batch_data
            mask_bas, mask_resp, mask_ekg, mask_emg = mask_list

            embeddings = [
                model(bas.to(device, dtype=torch.float), mask_bas.to(device, dtype=torch.bool)),
                model(resp.to(device, dtype=torch.float), mask_resp.to(device, dtype=torch.bool)),
                model(ekg.to(device, dtype=torch.float), mask_ekg.to(device, dtype=torch.bool)),
                model(emg.to(device, dtype=torch.float), mask_emg.to(device, dtype=torch.bool)),
            ]

            # 5min_agg
            emb_new = [e[0].unsqueeze(1) for e in embeddings]
            for i in range(len(file_paths)):
                subject_id = os.path.basename(file_paths[i]).split(".")[0]
                out_path = os.path.join(EMB_5MIN, f"{subject_id}.hdf5")
                with h5py.File(out_path, "a") as hf:
                    for mi, mod in enumerate(config["modality_types"]):
                        emb_i = emb_new[mi][i]
                        suffix = tuple(int(s) for s in emb_i.shape[1:])
                        if mod in hf:
                            dset = hf[mod]
                            cc = chunk_starts[i] // (embed_dim * 5 * 60)
                            ce = cc + emb_i.shape[0]
                            if dset.shape[0] < ce:
                                dset.resize((ce,) + suffix)
                            dset[cc:ce] = emb_i.cpu().numpy()
                        else:
                            hf.create_dataset(mod, data=emb_i.cpu().numpy(),
                                              chunks=(int(embed_dim),) + suffix,
                                              maxshape=(None,) + suffix)

            # 5s
            emb_new = [e[1] for e in embeddings]
            for i in range(len(file_paths)):
                subject_id = os.path.basename(file_paths[i]).split(".")[0]
                out_path = os.path.join(EMB_5S, f"{subject_id}.hdf5")
                with h5py.File(out_path, "a") as hf:
                    for mi, mod in enumerate(config["modality_types"]):
                        emb_i = emb_new[mi][i]
                        suffix = tuple(int(s) for s in emb_i.shape[1:])
                        if mod in hf:
                            dset = hf[mod]
                            cc = chunk_starts[i] // (embed_dim * 5)
                            ce = cc + emb_i.shape[0]
                            if dset.shape[0] < ce:
                                dset.resize((ce,) + suffix)
                            dset[cc:ce] = emb_i.cpu().numpy()
                        else:
                            hf.create_dataset(mod, data=emb_i.cpu().numpy(),
                                              chunks=(int(embed_dim),) + suffix,
                                              maxshape=(None,) + suffix)

            processed += len(file_paths)
            if (batch_idx + 1) % 20 == 0:
                elapsed = time.time() - t0
                logger.info(f"  Batch {batch_idx+1}/{len(dataloader)} — {elapsed/60:.1f} min")

    logger.info(f"Embeddings generados en {(time.time()-t0)/60:.1f} min")
    return len(dataset)


def verify(original_count):
    """Verifica que el número final de embeddings es correcto."""
    n_5s   = len([f for f in os.listdir(EMB_5S)   if f.endswith(".hdf5")])
    n_5min = len([f for f in os.listdir(EMB_5MIN)  if f.endswith(".hdf5")])
    n_hdf5 = len([f for f in os.listdir(HDF5_DIR)  if f.endswith(".hdf5")])

    logger.info(f"\n{'='*55}")
    logger.info(f"HDF5 de entrada   : {n_hdf5}")
    logger.info(f"Embeddings 5s     : {n_5s}")
    logger.info(f"Embeddings 5min   : {n_5min}")

    still_missing = get_missing_patients()
    if still_missing:
        logger.warning(f"Pacientes aún sin embedding ({len(still_missing)}):")
        for n in still_missing:
            logger.warning(f"  {n}")
    else:
        logger.info("✅ Todos los pacientes tienen embedding")

    assert n_5s == n_5min, f"Desajuste 5s ({n_5s}) vs 5min ({n_5min})"
    return len(still_missing)


def main():
    logger.info("=== fix_and_embed_missing.py ===")

    missing = get_missing_patients()
    logger.info(f"Pacientes sin embedding: {len(missing)}")

    # Paso 1: reparar HDF5
    logger.info("\n--- PASO 1: reparar HDF5 ---")
    fixed = fix_hdf5_files(missing)

    if not fixed:
        logger.warning("No se reparó ningún fichero. Abortando.")
        sys.exit(1)

    # Paso 2: generar embeddings para los reparados
    logger.info(f"\n--- PASO 2: generar embeddings para {len(fixed)} pacientes ---")
    generate_embeddings_for(fixed)

    # Paso 3: verificación final
    logger.info("\n--- PASO 3: verificación final ---")
    n_remaining = verify(len(missing))

    sys.exit(0 if n_remaining == 0 else 1)


if __name__ == "__main__":
    main()
