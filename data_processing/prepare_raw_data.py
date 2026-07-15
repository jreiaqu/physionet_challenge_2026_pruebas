"""
prepare_raw_data.py
===================
Descarga el dataset PhysioNet Challenge 2026 desde Kaggle usando curl con soporte
de reanudación (HTTP Range), con logging de progreso periódico a stdout y fichero.

Tamaños reales medidos (13 jul 2026):
  ZIP comprimido : 129.82 GB
  Extraído       : 230.02 GB  (3 292 ficheros)
  Pico de disco  : ~360 GB (zip + extraído durante la extracción)

Estructura del dataset:
  physiological_data/{I0002,I0006,S0001}/sub-XXXXX_ses-X.edf   (PSG, 50-900 MB c/u)
  algorithmic_annotations/{sitio}/sub-XXXXX_ses-X_caisr_*.edf  (~400-600 KB c/u)
  human_annotations/{sitio}/sub-XXXXX_ses-X_*.edf              (~50-200 KB c/u)
  demographics.csv, ICD_codes_CI.csv

NOTA: La descarga de ficheros individuales con `kaggle datasets download -f`
devuelve 404 para EDFs en subdirectorios. La única vía es el zip completo.

Uso:
    python prepare_raw_data.py                   # descarga + extrae
    python prepare_raw_data.py --dry-run         # solo comprobaciones previas
    python prepare_raw_data.py --verify-only     # verifica extracción existente
    python prepare_raw_data.py --no-extract      # solo descarga, no extrae
    python prepare_raw_data.py --log otra.log    # fichero de log alternativo
    python prepare_raw_data.py --interval 60     # log cada 60 s (default 30)

Códigos de salida:
    0  Éxito
    1  Error de pre-comprobación (credenciales, espacio, permisos)
    2  Error en la descarga
    3  Error en la extracción
    4  Error en la verificación de integridad
    5  Dry-run completado (sin acción)

Requisitos:
    curl  (disponible en el sistema)
    ~/.config/kaggle/kaggle.json o ~/.kaggle/kaggle.json con tu API key
    (descárgalo en https://www.kaggle.com/settings → API → Create token)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
API_URL   = "https://www.kaggle.com/api/v1/datasets/download/physionet/physionetchallenge2026data"
ZIP_NAME  = "physionetchallenge2026data.zip"
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset_small"))

ZIP_SIZE_BYTES   = 129_818_473_924   # medido con HEAD request
FILES_EXPECTED   = 3_292
BYTES_EXTRACTED  = 230_024_509_878   # suma real de todos los ficheros
DISK_PEAK_BYTES  = ZIP_SIZE_BYTES + BYTES_EXTRACTED  # pico durante extracción

EXIT_OK       = 0
EXIT_PRECHECK = 1
EXIT_DOWNLOAD = 2
EXIT_EXTRACT  = 3
EXIT_VERIFY   = 4
EXIT_DRYRUN   = 5
# ──────────────────────────────────────────────────────────────────────────────


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str, log_fh=None) -> None:
    line = f"[{_ts()}] {msg}"
    print(line, flush=True)
    if log_fh:
        log_fh.write(line + "\n")
        log_fh.flush()


def _find_kaggle_credentials() -> dict:
    candidates = [
        os.path.expanduser("~/.config/kaggle/kaggle.json"),
        os.path.expanduser("~/.kaggle/kaggle.json"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                creds = json.load(open(p))
                if "username" in creds and "key" in creds:
                    return creds
            except Exception:
                pass
    return {}


def _check_prequisites(dest_dir: str, dry_run: bool, log_fh=None) -> None:
    """Valida credenciales, espacio y permisos. Termina con EXIT_PRECHECK si falla."""
    errors = []

    # Credenciales
    creds = _find_kaggle_credentials()
    if not creds:
        errors.append(
            "kaggle.json no encontrado. Descárgalo en "
            "https://www.kaggle.com/settings → API → Create token"
        )
    else:
        _log(f"Credenciales Kaggle: OK (usuario={creds['username']})", log_fh)

    # curl disponible
    if not shutil.which("curl"):
        errors.append("curl no está disponible en PATH. Instálalo con: sudo apt install curl")
    else:
        _log("curl: OK", log_fh)

    # Espacio en disco
    os.makedirs(dest_dir, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    _log(
        f"Espacio libre en {dest_dir}: {free/1e9:.1f} GB "
        f"(necesario pico: {DISK_PEAK_BYTES/1e9:.1f} GB)", log_fh
    )
    if free < DISK_PEAK_BYTES:
        errors.append(
            f"Espacio insuficiente: {free/1e9:.1f} GB libres, "
            f"se necesitan {DISK_PEAK_BYTES/1e9:.1f} GB (zip + extraído simultáneos)"
        )

    # Escritura en destino
    test_file = os.path.join(dest_dir, ".write_test")
    try:
        open(test_file, "w").close()
        os.remove(test_file)
        _log(f"Destino escribible: {dest_dir}", log_fh)
    except OSError as e:
        errors.append(f"Destino no escribible ({dest_dir}): {e}")

    if errors:
        for err in errors:
            _log(f"❌ {err}", log_fh)
        sys.exit(EXIT_PRECHECK)

    if dry_run:
        _log("Dry-run: todas las comprobaciones OK. Sin acción.", log_fh)
        sys.exit(EXIT_DRYRUN)


def _monitor_progress(zip_path: str, total: int, interval: int, stop_event: threading.Event,
                      log_fh=None) -> None:
    """Hilo que loguea el progreso de descarga cada `interval` segundos."""
    t0 = time.time()
    last_size = 0
    last_time = t0

    while not stop_event.is_set():
        time.sleep(interval)
        if stop_event.is_set():
            break

        now = time.time()
        try:
            size = os.path.getsize(zip_path)
        except FileNotFoundError:
            continue

        elapsed = now - t0
        delta_t = now - last_time
        delta_b = size - last_size

        pct     = size / total * 100 if total else 0
        avg_mbs = size / elapsed / 1e6 if elapsed > 0 else 0
        inst_mbs = delta_b / delta_t / 1e6 if delta_t > 0 else 0
        eta_s   = (total - size) / (avg_mbs * 1e6) if avg_mbs > 0 else 0
        eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}min"

        _log(
            f"Descarga: {size/1e9:.2f}/{total/1e9:.2f} GB  "
            f"({pct:.1f}%)  inst={inst_mbs:.1f} MB/s  avg={avg_mbs:.1f} MB/s  ETA={eta_str}",
            log_fh
        )

        last_size = size
        last_time = now


def _download(zip_path: str, creds: dict, interval: int, log_fh=None) -> None:
    """Descarga el zip con curl -C - (reanudable). Termina con EXIT_DOWNLOAD si falla."""
    auth = f"{creds['username']}:{creds['key']}"

    existing = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
    if existing > 0:
        _log(f"Fichero parcial encontrado ({existing/1e9:.2f} GB). Reanudando...", log_fh)
    else:
        _log("Iniciando descarga desde cero...", log_fh)

    _log(f"ZIP destino   : {zip_path}", log_fh)
    _log(f"Tamaño total  : {ZIP_SIZE_BYTES/1e9:.2f} GB", log_fh)

    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_monitor_progress,
        args=(zip_path, ZIP_SIZE_BYTES, interval, stop_event, log_fh),
        daemon=True,
    )
    monitor.start()

    cmd = [
        "curl",
        "-L",                     # sigue redirecciones
        "-C", "-",                # reanuda si hay fichero parcial
        "--user", auth,
        "--user-agent", "python-kaggle/1.6.0",
        "--retry", "5",           # reintenta hasta 5 veces en error de red
        "--retry-delay", "10",
        "--retry-max-time", "0",  # sin límite de tiempo en reintentos
        "--connect-timeout", "30",
        "--silent",               # sin barra de progreso (va al hilo monitor)
        "--show-error",
        "-o", zip_path,
        API_URL,
    ]

    t0 = time.time()
    result = subprocess.run(cmd)
    stop_event.set()
    monitor.join(timeout=2)

    elapsed = time.time() - t0
    if result.returncode not in (0, 33):  # 33 = "Range not satisfiable" (ya completo)
        _log(f"❌ curl terminó con código {result.returncode}", log_fh)
        sys.exit(EXIT_DOWNLOAD)

    final_size = os.path.getsize(zip_path) if os.path.exists(zip_path) else 0
    avg_mbs = final_size / elapsed / 1e6 if elapsed > 0 else 0
    _log(
        f"✅ Descarga completada: {final_size/1e9:.2f} GB en {elapsed/60:.1f} min "
        f"({avg_mbs:.1f} MB/s avg)",
        log_fh
    )


def _extract(zip_path: str, dest_dir: str, log_fh=None) -> None:
    """Extrae el zip con progreso. Termina con EXIT_EXTRACT si falla."""
    _log(f"Iniciando extracción de {zip_path} → {dest_dir}", log_fh)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            entries = zf.infolist()
            total_files = len(entries)
            total_bytes = sum(e.file_size for e in entries)
            _log(f"Ficheros a extraer: {total_files}  ({total_bytes/1e9:.2f} GB)", log_fh)

            extracted_bytes = 0
            last_log_time = time.time()
            t0 = time.time()

            for i, entry in enumerate(entries, 1):
                zf.extract(entry, dest_dir)
                extracted_bytes += entry.file_size

                now = time.time()
                if now - last_log_time >= 30:
                    pct = extracted_bytes / total_bytes * 100 if total_bytes else 0
                    elapsed = now - t0
                    speed = extracted_bytes / elapsed / 1e6 if elapsed > 0 else 0
                    eta_s = (total_bytes - extracted_bytes) / (speed * 1e6) if speed > 0 else 0
                    eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}min"
                    _log(
                        f"Extracción: {i}/{total_files} ficheros  "
                        f"{extracted_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB  "
                        f"({pct:.1f}%)  {speed:.0f} MB/s  ETA={eta_str}",
                        log_fh
                    )
                    last_log_time = now

    except zipfile.BadZipFile as e:
        _log(f"❌ ZIP corrupto o incompleto: {e}", log_fh)
        sys.exit(EXIT_EXTRACT)
    except Exception as e:
        _log(f"❌ Error durante la extracción: {e}", log_fh)
        sys.exit(EXIT_EXTRACT)

    elapsed = time.time() - t0
    _log(f"✅ Extracción completada en {elapsed/60:.1f} min", log_fh)


def _verify(dest_dir: str, log_fh=None) -> None:
    """Cuenta ficheros y suma bytes. Termina con EXIT_VERIFY si no cuadra."""
    _log(f"Verificando integridad de {dest_dir}...", log_fh)

    found_files = 0
    found_bytes = 0
    for root, _, files in os.walk(dest_dir):
        for fname in files:
            path = os.path.join(root, fname)
            try:
                found_bytes += os.path.getsize(path)
                found_files += 1
            except OSError:
                pass

    _log(f"  Encontrados : {found_files} ficheros  ({found_bytes/1e9:.2f} GB)", log_fh)
    _log(f"  Esperados   : {FILES_EXPECTED} ficheros  ({BYTES_EXTRACTED/1e9:.2f} GB)", log_fh)

    ok = True
    if found_files != FILES_EXPECTED:
        _log(f"⚠️  Diferencia en nº de ficheros: {found_files} vs {FILES_EXPECTED}", log_fh)
        ok = False
    tol = 0.01  # 1% tolerancia en bytes (metadatos del SO)
    if abs(found_bytes - BYTES_EXTRACTED) / BYTES_EXTRACTED > tol:
        _log(
            f"⚠️  Diferencia en tamaño: {found_bytes/1e9:.2f} vs {BYTES_EXTRACTED/1e9:.2f} GB",
            log_fh
        )
        ok = False

    if not ok:
        sys.exit(EXIT_VERIFY)

    _log("✅ Verificación OK.", log_fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga PhysioNet Challenge 2026 desde Kaggle (reanudable)."
    )
    parser.add_argument("--dest", default=BASE_PATH,
                        help=f"Directorio de extracción (default: {BASE_PATH})")
    parser.add_argument("--log", default="descarga.log",
                        help="Fichero de log (default: descarga.log)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Segundos entre líneas de progreso (default: 30)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Solo descarga el ZIP, sin extraer")
    parser.add_argument("--keep-zip", action="store_true",
                        help="No borrar el ZIP tras extraer")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo comprobaciones previas, sin descargar")
    parser.add_argument("--verify-only", action="store_true",
                        help="Solo verifica la extracción existente")
    args = parser.parse_args()

    log_fh = open(args.log, "a", buffering=1)
    _log("=" * 60, log_fh)
    _log("Iniciando prepare_raw_data.py", log_fh)
    _log(f"Destino  : {args.dest}", log_fh)
    _log(f"Log      : {args.log}", log_fh)

    if args.verify_only:
        _verify(args.dest, log_fh)
        log_fh.close()
        sys.exit(EXIT_OK)

    creds = _find_kaggle_credentials()
    _check_prequisites(args.dest, args.dry_run, log_fh)

    zip_path = os.path.join(args.dest, ZIP_NAME)
    _download(zip_path, creds, args.interval, log_fh)

    if not args.no_extract:
        _extract(zip_path, args.dest, log_fh)
        if not args.keep_zip:
            _log(f"Borrando ZIP ({zip_path})...", log_fh)
            os.remove(zip_path)
            _log("ZIP borrado.", log_fh)
        _verify(args.dest, log_fh)

    log_fh.close()
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
