"""
fix_and_embed_missing_large.py
================================
Adaptación de fix_and_embed_missing.py para dataset_large_balanced.

Diferencia clave con la versión de dataset_small: aquí NO hay hdf5
persistente (embed_large_in_batches.py convierte y borra por lotes para
no llenar el disco), así que "reparar" no es normalizar un hdf5 ya en
disco: hay que reconvertir cada paciente pendiente desde su EDF.

Se investigaron los 188 pacientes sin embedding tras la primera pasada
(2026-07-14) y aparecieron 3 causas:
  1. Case mixto en nombres de canal (I0006: "ChinA", "Left Leg"...) no
     coincide ni con las variantes en minúsculas ni mayúsculas de
     channel_groups.json → el paciente entero se descarta.
     FIX YA APLICADO en preprocessing_raw_data.py: guarda los canales
     siempre en minúsculas. Se resuelve solo al reconvertir.
  2. Bug determinista en save_to_hdf5: chunk fijo de 5 min más grande
     que alguna señal corta → "Chunk shape must not be greater than
     data shape". FIX YA APLICADO: chunk = min(chunk_fijo, len(señal)).
  3. Modalidad genuinamente ausente en la grabación (p.ej. sin canal
     EKG/ECG en absoluto) → IRRECUPERABLE, no hay nada que arreglar.

Este script:
  1. Reintenta todos los pacientes pendientes vía embed_large_in_batches
     (que ya se beneficia de los dos fixes anteriores sin cambios).
  2. Para los que sigan sin embedding, diagnostica cada uno de forma
     aislada y reporta la razón exacta (irrecuperable vs error puntual).

Uso:
    python fix_and_embed_missing_large.py
"""

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import embed_large_in_batches as elb
import h5py
from loguru import logger

DIAG_SCRATCH = os.path.join(elb.REPO_ROOT, "dataset_large_balanced", "_diag_scratch")
RETRY_BATCH_SIZE = 130  # mismo tamaño ya validado en embed_large_in_batches


def _diagnose_patient(edf_path: str):
    """Convierte un único paciente de forma aislada y clasifica el resultado.
    Devuelve (recoverable: bool, reason: str)."""
    subject_id = elb._subject_id(edf_path)
    edf_link_dir = os.path.join(DIAG_SCRATCH, "edf")
    hdf5_dir = os.path.join(DIAG_SCRATCH, "hdf5")
    os.makedirs(edf_link_dir, exist_ok=True)
    os.makedirs(hdf5_dir, exist_ok=True)

    link_path = os.path.join(edf_link_dir, os.path.basename(edf_path))
    if os.path.lexists(link_path):
        os.remove(link_path)
    os.symlink(edf_path, link_path)

    hdf5_path = os.path.join(hdf5_dir, f"{subject_id}.hdf5")
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    from preprocessing_raw_data import EDFToHDF5Converter
    converter = EDFToHDF5Converter(root_dir=edf_link_dir, target_dir=hdf5_dir,
                                    resample_rate=256, num_threads=1)
    try:
        converter.convert(edf_path, hdf5_path)
    except Exception as e:
        return False, f"error de conversión: {e}"

    if not os.path.exists(hdf5_path):
        return False, "conversión no generó fichero hdf5"

    with h5py.File(hdf5_path, "r") as hf:
        keys = list(hf.keys())

    if len(keys) == 0:
        return False, "hdf5 vacío tras conversión"

    channel_groups = json.load(open(elb.CHANNEL_GROUPS_PATH))
    missing_mods = [mod for mod, chans in channel_groups.items()
                    if not any(c in keys for c in chans)]
    if missing_mods:
        return False, f"sin canales para {missing_mods} (canales disponibles: {keys})"

    return True, "conversión aislada OK pero seguía sin embedding tras el lote"


def main() -> None:
    logger.info("=== fix_and_embed_missing_large.py ===")

    pending_before = [p for p in elb._list_edf_files() if not elb._already_embedded(elb._subject_id(p))]
    logger.info(f"Pacientes sin embedding al empezar: {len(pending_before)}")

    if not pending_before:
        logger.info("✅ Nada que reparar.")
        return

    logger.info("--- PASO 1: reintentar vía embed_large_in_batches (ya con los fixes) ---")
    os.makedirs(elb.SCRATCH_ROOT, exist_ok=True)
    batches = [pending_before[i:i + RETRY_BATCH_SIZE]
               for i in range(0, len(pending_before), RETRY_BATCH_SIZE)]
    for i, batch in enumerate(batches, 1):
        elb.process_batch(batch, i, len(batches), num_threads=16, resample_rate=256,
                           num_workers=16, batch_size_gen=128)
    shutil.rmtree(elb.SCRATCH_ROOT, ignore_errors=True)

    pending_after = [p for p in elb._list_edf_files() if not elb._already_embedded(elb._subject_id(p))]
    logger.info(f"Reparados en el paso 1: {len(pending_before) - len(pending_after)}")
    logger.info(f"Aún pendientes: {len(pending_after)}")

    if not pending_after:
        logger.info("✅ Todos los pacientes reparados.")
        return

    logger.info("--- PASO 2: diagnóstico individual de los que siguen sin embedding ---")
    irrecoverable = []
    unexplained = []
    for edf_path in pending_after:
        subject_id = elb._subject_id(edf_path)
        ok, reason = _diagnose_patient(edf_path)
        if ok:
            logger.warning(f"[SIN EXPLICAR] {subject_id}: {reason}")
            unexplained.append(subject_id)
        else:
            logger.warning(f"[IRRECUPERABLE] {subject_id}: {reason}")
            irrecoverable.append((subject_id, reason))

    shutil.rmtree(DIAG_SCRATCH, ignore_errors=True)

    total = len(elb._list_edf_files())
    embedded = total - len(pending_after)
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Total pacientes    : {total}")
    logger.info(f"Con embedding      : {embedded} ({embedded/total*100:.1f}%)")
    logger.info(f"Irrecuperables     : {len(irrecoverable)}")
    logger.info(f"Sin explicar       : {len(unexplained)} (reintenta manualmente)")
    for subject_id, reason in irrecoverable:
        logger.info(f"  [IRRECUPERABLE] {subject_id}: {reason}")
    for subject_id in unexplained:
        logger.info(f"  [SIN EXPLICAR]  {subject_id}")


if __name__ == "__main__":
    main()
