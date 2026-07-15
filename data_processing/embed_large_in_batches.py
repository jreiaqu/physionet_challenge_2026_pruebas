"""
embed_large_in_batches.py
==========================
Genera los embeddings SleepFM del dataset "large" (dataset_large_balanced)
en LOTES, borrando el hdf5 intermedio de cada lote antes de procesar el
siguiente. Necesario porque los ~996 pacientes generarían ~170 GB de hdf5
de golpe (misma proporción que dataset_small: 188 GB / 1103 pacientes ≈
175 MB/paciente), y el disco no tiene ese margen disponible.

Por cada lote:
  1. Comprueba espacio libre real (heurística: tamaño_lote × 175 MB × margen).
  2. Symlinks de los EDF del lote a un directorio temporal (no duplica datos).
  3. data_processing/preprocessing_raw_data.py  (EDF → hdf5, ese lote)
  4. src/generate_embeddings.py --dataset_name physionet2026_large
     (hdf5 → embeddings, acumula en el mismo directorio de salida entre lotes)
  5. Borra el hdf5 y los symlinks del lote.

Reanudable: antes de formar los lotes, se salta cualquier paciente cuyo
embedding ya exista en ambos directorios de salida (5s y 5min_agg).

Uso:
    python embed_large_in_batches.py
    python embed_large_in_batches.py --batch-size 250
    python embed_large_in_batches.py --dry-run
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EDF_ROOT      = os.path.join(REPO_ROOT, "dataset_large_balanced/physiological_data")
DATASET_NAME  = "physionet2026_large"
EMB_BASE      = os.path.join(REPO_ROOT, "sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base")
EMB_5S_DIR    = os.path.join(EMB_BASE, DATASET_NAME)
EMB_5MIN_DIR  = os.path.join(EMB_BASE, f"{DATASET_NAME}_5min_agg")
SRC_DIR       = os.path.join(REPO_ROOT, "src")
CHANNEL_GROUPS_PATH = os.path.join(SRC_DIR, "configs/channel_groups.json")
# generate_embeddings.py hace `from utils import *` / `from models.dataset import ...` de forma
# relativa al cwd (sys.path.append("../")), no a su propia ubicación en disco. Solo resuelve
# si se ejecuta con cwd = sleepFM/sleepfm-clinical/sleepfm/pipeline/ (donde "../" = sleepfm/,
# que sí contiene utils.py y models/). Verificado en vivo el 2026-07-14.
GENERATE_EMBEDDINGS_CWD = os.path.join(REPO_ROOT, "sleepFM/sleepfm-clinical/sleepfm/pipeline")
SCRATCH_ROOT  = os.path.join(REPO_ROOT, "dataset_large_balanced", "_batch_scratch")

MB_PER_PATIENT_HDF5 = 175   # medido en dataset_small: 188GB / 1103 pacientes
SAFETY_MARGIN        = 1.3  # margen sobre la estimación (variabilidad entre pacientes)

EXIT_OK       = 0
EXIT_PRECHECK = 1
EXIT_STEP     = 2
EXIT_DRYRUN   = 5


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, log_fh=None) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_fh:
        log_fh.write(line + "\n")
        log_fh.flush()


def _list_edf_files() -> list:
    return sorted(glob.glob(os.path.join(EDF_ROOT, "**/*.[eE][dD][fF]"), recursive=True))


def _subject_id(edf_path: str) -> str:
    return os.path.splitext(os.path.basename(edf_path))[0]


def _already_embedded(subject_id: str) -> bool:
    return (
        os.path.exists(os.path.join(EMB_5S_DIR, f"{subject_id}.hdf5"))
        and os.path.exists(os.path.join(EMB_5MIN_DIR, f"{subject_id}.hdf5"))
    )


def _check_disk_for_batch(batch: list, log_fh=None) -> None:
    needed = len(batch) * MB_PER_PATIENT_HDF5 * 1e6 * SAFETY_MARGIN
    free = shutil.disk_usage(REPO_ROOT).free
    _log(
        f"  Espacio estimado para el lote: {needed/1e9:.1f} GB  |  libre: {free/1e9:.1f} GB",
        log_fh
    )
    if free < needed:
        _log(
            f"❌ Espacio insuficiente para este lote (faltan {(needed-free)/1e9:.1f} GB). "
            f"Reduce --batch-size o libera espacio.", log_fh
        )
        sys.exit(EXIT_PRECHECK)


def _run(cmd: list, cwd: str, log_fh=None) -> None:
    _log(f"  $ {' '.join(cmd)}  (cwd={cwd})", log_fh)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        _log(f"❌ Comando falló con código {result.returncode}", log_fh)
        sys.exit(EXIT_STEP)


def process_batch(batch: list, batch_num: int, total_batches: int, num_threads: int,
                   resample_rate: int, num_workers: int, batch_size_gen: int, log_fh=None) -> None:
    _log(f"── Lote {batch_num}/{total_batches}: {len(batch)} pacientes ──", log_fh)
    _check_disk_for_batch(batch, log_fh)

    scratch_edf  = os.path.join(SCRATCH_ROOT, f"batch_{batch_num}_edf")
    scratch_hdf5 = os.path.join(SCRATCH_ROOT, f"batch_{batch_num}_hdf5")
    os.makedirs(scratch_edf, exist_ok=True)
    os.makedirs(scratch_hdf5, exist_ok=True)

    for edf_path in batch:
        link_path = os.path.join(scratch_edf, os.path.basename(edf_path))
        if not os.path.exists(link_path):
            os.symlink(edf_path, link_path)

    t0 = time.time()
    _run(
        [sys.executable, "preprocessing_raw_data.py",
         "--root_dir", scratch_edf,
         "--target_dir", scratch_hdf5,
         "--num_threads", str(num_threads),
         "--resample_rate", str(resample_rate)],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        log_fh=log_fh,
    )
    _log(f"  EDF→hdf5 del lote {batch_num} en {(time.time()-t0)/60:.1f} min", log_fh)

    t0 = time.time()
    _run(
        [sys.executable, os.path.join(SRC_DIR, "generate_embeddings.py"),
         "--model_path", EMB_BASE,
         "--dataset_name", DATASET_NAME,
         "--hdf5_dir", scratch_hdf5,
         "--channel_groups_path", CHANNEL_GROUPS_PATH,
         "--num_workers", str(num_workers),
         "--batch_size", str(batch_size_gen)],
        cwd=GENERATE_EMBEDDINGS_CWD,
        log_fh=log_fh,
    )
    _log(f"  Embeddings del lote {batch_num} en {(time.time()-t0)/60:.1f} min", log_fh)

    shutil.rmtree(scratch_hdf5, ignore_errors=True)
    shutil.rmtree(scratch_edf, ignore_errors=True)
    _log(f"  hdf5 y symlinks del lote {batch_num} borrados", log_fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera embeddings SleepFM del dataset large en lotes, sin acumular hdf5."
    )
    parser.add_argument("--batch-size", type=int, default=250,
                        help="Pacientes por lote (default: 250, ~44GB de hdf5 por lote)")
    parser.add_argument("--num-threads", type=int, default=16,
                        help="Hilos para preprocessing_raw_data.py (default: 16)")
    parser.add_argument("--resample-rate", type=int, default=256,
                        help="Frecuencia de remuestreo EDF→hdf5 (default: 256, igual que dataset_small)")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="num_workers de generate_embeddings.py (default: 16)")
    parser.add_argument("--gen-batch-size", type=int, default=128,
                        help="batch_size de generate_embeddings.py (default: 128)")
    parser.add_argument("--log", default="embed_large_pipeline.log",
                        help="Fichero de log (default: embed_large_pipeline.log)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra cuántos pacientes faltan y el nº de lotes, sin procesar")
    args = parser.parse_args()

    log_fh = open(args.log, "a", buffering=1)
    _log("=" * 70, log_fh)
    _log("Iniciando embed_large_in_batches.py", log_fh)

    all_edf = _list_edf_files()
    _log(f"EDF encontrados en {EDF_ROOT}: {len(all_edf)}", log_fh)

    pending = [p for p in all_edf if not _already_embedded(_subject_id(p))]
    _log(f"Pendientes de embedding: {len(pending)} (ya hechos: {len(all_edf) - len(pending)})", log_fh)

    if not pending:
        _log("✅ Nada pendiente. Todos los pacientes ya tienen embedding.", log_fh)
        log_fh.close()
        sys.exit(EXIT_OK)

    batches = [pending[i:i + args.batch_size] for i in range(0, len(pending), args.batch_size)]
    _log(f"Lotes a procesar: {len(batches)} de hasta {args.batch_size} pacientes", log_fh)

    if args.dry_run:
        _log("Dry-run: sin procesar.", log_fh)
        log_fh.close()
        sys.exit(EXIT_DRYRUN)

    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    t0 = time.time()
    for i, batch in enumerate(batches, 1):
        process_batch(
            batch, i, len(batches), args.num_threads, args.resample_rate,
            args.num_workers, args.gen_batch_size, log_fh
        )
    shutil.rmtree(SCRATCH_ROOT, ignore_errors=True)

    _log(f"✅ Todos los lotes procesados en {(time.time()-t0)/60:.1f} min", log_fh)
    log_fh.close()
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
