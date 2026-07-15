"""
download_large_stratified.py
=============================
Descarga una submuestra estratificada de la versión "large" del PhysioNet
Challenge 2026 (`physionet/physionetchallenge2026datalargeversion`) SIN
descargar el ZIP completo (774.9 GB comprimidos, medido con HEAD/Range el
13 jul 2026 — la cifra de "1.3 TB" corresponde al contenido descomprimido).

Hallazgos de la investigación previa (ver también prepare_raw_data.py):
  - `kaggle datasets download -f <ruta>` funciona para ficheros en la RAÍZ
    del dataset (demographics.csv, ICD_codes_CI.csv) pero devuelve 404 para
    cualquier fichero anidado en subcarpeta (physiological_data/,
    algorithmic_annotations/, human_annotations/). Es una limitación de la
    API de Kaggle, no del dataset "small": se reproduce igual en el "large".
    -> Descarga de ficheros individuales vía API de Kaggle: NO VIABLE.
  - El endpoint de descarga de Kaggle redirige (302) a una URL firmada de
    Google Cloud Storage que SÍ soporta HTTP Range requests (HTTP 206).
  - El ZIP usa formato ZIP64 (19 713 entradas totales) y su directorio
    central pesa ~2.6 MB: se puede leer con una única petición Range sin
    tocar el resto del archivo.
  - Conclusión: se puede extraer selectivamente, vía Range requests, solo
    las entradas de los pacientes que nos interesan, usando la librería
    `remotezip` sobre la URL de descarga autenticada. Verificado en vivo:
    listar el directorio central + extraer un fichero individual tardó
    < 2.5 s combinados, sin descargar los otros ~18 700 pacientes.

Por eso se descarta la alternativa de "Kaggle Notebook que filtra en sus
servidores": añadiría latencia de cola de ejecución remota y complejidad de
orquestación (push del notebook, poll de estado, pull del output) para
resolver un problema que Range requests ya resuelven de forma más simple,
rápida y depurable localmente.

Estrategia estadística:
  1. Descargar solo demographics.csv (fichero raíz, 787 KB) vía Range.
  2. Filtrar los 498 pacientes positivos (Cognitive_Impairment == True).
  3. Emparejar 498 negativos por EDAD EXACTA (fallback a edad±1, ±2... si
     no quedan negativos de esa edad exacta en el pool), sin reemplazo,
     con random_state=42 para reproducibilidad.
  4. Para cada uno de los 996 pacientes, extraer selectivamente sus
     ficheros de physiological_data/, algorithmic_annotations/ y
     human_annotations/ (si existen; no todos los pacientes tienen las
     3 modalidades) manteniendo la misma estructura de carpetas que
     dataset_small, para compatibilidad con el resto del pipeline.

Uso:
    python download_large_stratified.py --output_dir ./data
    python download_large_stratified.py --dry-run
    python download_large_stratified.py --verify-only --output_dir ./data
    python download_large_stratified.py --output_dir ./data --no-annotations
    # Reanudar tras un corte (se saltan los ficheros ya descargados):
    python download_large_stratified.py --output_dir ./data

NOTA sobre el endpoint de descarga completa de Kaggle: en una prueba real
(13-14 jul 2026) empezó a devolver 404 en TODAS las peticiones tras ~50
ficheros descargados con 4 hilos concurrentes, mientras que la descarga
individual de ficheros raíz (`-f`) del mismo dataset seguía funcionando.
Todo apunta a un límite de ráfaga/cuota específico de ese endpoint, no a
un problema de rutas. Por eso el default es `--workers 1` y `--delay 0.5`,
y hay un circuit breaker que aborta tras 8 fallos consecutivos en vez de
seguir golpeando un endpoint bloqueado (ver download_manifest). El script
es reanudable: si se aborta o lo cortas, vuelve a lanzar el mismo comando
y saltará los ficheros ya descargados (comprobando su tamaño).

Códigos de salida:
    0  Éxito
    1  Error de pre-comprobación (credenciales, dependencias, espacio)
    2  Error durante la descarga selectiva
    4  Error en la verificación de integridad
    5  Dry-run completado (sin acción)

Requisitos:
    pip install remotezip   (ya añadido a requirements.txt)
    ~/.config/kaggle/kaggle.json o ~/.kaggle/kaggle.json con tu API key
"""

import argparse
import io
import json
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from remotezip import RemoteZip
except ImportError:
    RemoteZip = None

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
DATASET      = "physionet/physionetchallenge2026datalargeversion"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"
BASE_PATH    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset_large_balanced"))

N_POSITIVE        = 498     # clase minoritaria completa (fijo, medido en demographics.csv)
MIN_FREE_GB_FLOOR = 50      # suelo de sanidad mínimo, además del cálculo dinámico real
SAFETY_MARGIN     = 1.05    # margen sobre el tamaño real calculado del manifiesto

EXIT_OK       = 0
EXIT_PRECHECK = 1
EXIT_DOWNLOAD = 2
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


def _check_prerequisites(dest_dir: str, dry_run: bool, log_fh=None) -> dict:
    """Valida credenciales, dependencias y escritura. Termina con EXIT_PRECHECK si falla."""
    errors = []

    creds = _find_kaggle_credentials()
    if not creds:
        errors.append(
            "kaggle.json no encontrado. Descárgalo en "
            "https://www.kaggle.com/settings → API → Create token"
        )
    else:
        _log(f"Credenciales Kaggle: OK (usuario={creds['username']})", log_fh)

    if RemoteZip is None:
        errors.append("Falta la librería 'remotezip'. Instálala con: pip install remotezip")
    else:
        _log("remotezip: OK", log_fh)

    os.makedirs(dest_dir, exist_ok=True)
    free = shutil.disk_usage(dest_dir).free
    _log(f"Espacio libre en {dest_dir}: {free/1e9:.1f} GB (suelo mínimo: {MIN_FREE_GB_FLOOR} GB)", log_fh)
    if free < MIN_FREE_GB_FLOOR * 1e9:
        errors.append(f"Espacio insuficiente: {free/1e9:.1f} GB libres, mínimo {MIN_FREE_GB_FLOOR} GB")

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

    return creds


def sample_age_matched_negatives(positives: pd.DataFrame, negatives_pool: pd.DataFrame,
                                  seed: int = 42, log_fh=None) -> pd.DataFrame:
    """Empareja 1:1 cada positivo con un negativo de la MISMA edad (sin reemplazo).

    Si no quedan negativos de la edad exacta en el pool, amplía la búsqueda a
    edad±1, ±2, ... hasta encontrar candidato. Determinista dado `seed`.
    """
    rng = np.random.RandomState(seed)
    pool = negatives_pool.copy()
    ages_needed = positives.sort_values("BDSPPatientID")["Age"].tolist()

    selected_idx = []
    fallback_count = 0
    for age in ages_needed:
        offset = 0
        candidates = []
        while not candidates:
            offset_ages = {age} if offset == 0 else {age - offset, age + offset}
            candidates = pool.index[pool["Age"].isin(offset_ages)].tolist()
            if offset > 0:
                fallback_count += 1
            offset += 1
            if offset > 100:
                raise RuntimeError(f"No se encontró negativo disponible cerca de la edad {age}")
        chosen = int(rng.choice(candidates))
        selected_idx.append(chosen)
        pool = pool.drop(index=chosen)

    if fallback_count:
        _log(f"⚠️  {fallback_count} negativos emparejados con edad±N (no exacta) "
             f"por agotamiento del pool en esa edad exacta", log_fh)

    return negatives_pool.loc[selected_idx]


def _print_age_stats(positives: pd.DataFrame, negatives: pd.DataFrame, log_fh=None) -> None:
    _log(
        f"Edad positivos : media={positives['Age'].mean():.2f}  std={positives['Age'].std():.2f}  "
        f"n={len(positives)}", log_fh
    )
    _log(
        f"Edad negativos : media={negatives['Age'].mean():.2f}  std={negatives['Age'].std():.2f}  "
        f"n={len(negatives)}", log_fh
    )
    _log(f"Diferencia de medias: {abs(positives['Age'].mean() - negatives['Age'].mean()):.3f} años", log_fh)


def build_manifest(demographics: pd.DataFrame, seed: int, log_fh=None) -> pd.DataFrame:
    positives = demographics[demographics["Cognitive_Impairment"] == True]  # noqa: E712
    negatives_pool = demographics[demographics["Cognitive_Impairment"] == False]  # noqa: E712

    if len(positives) != N_POSITIVE:
        _log(f"⚠️  Se esperaban {N_POSITIVE} positivos, se encontraron {len(positives)} "
             f"(el dataset pudo actualizarse; se usarán todos los encontrados)", log_fh)

    matched_negatives = sample_age_matched_negatives(positives, negatives_pool, seed=seed, log_fh=log_fh)
    _print_age_stats(positives, matched_negatives, log_fh)

    manifest = pd.concat([positives, matched_negatives]).sort_values(
        ["Cognitive_Impairment", "SiteID", "BDSPPatientID"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return manifest


@dataclass
class ZipTarget:
    zip_name: str
    dest_relpath: str
    expected_size: int = None


def build_zip_targets(row: pd.Series, namelist_set: set, include_annotations: bool) -> list:
    site, bids, ses = row["SiteID"], row["BidsFolder"], row["SessionID"]
    stem = f"{bids}_ses-{ses}"

    candidates = [f"physiological_data/{site}/{stem}.edf"]
    if include_annotations:
        candidates.append(f"algorithmic_annotations/{site}/{stem}_caisr_annotations.edf")
        candidates.append(f"human_annotations/{site}/{stem}_expert_annotations.edf")

    return [ZipTarget(c, c) for c in candidates if c in namelist_set]


def compute_required_bytes(manifest: pd.DataFrame, zf, namelist_set: set,
                            include_annotations: bool, log_fh=None):
    total_bytes = 0
    missing = 0
    all_targets = []
    for _, row in manifest.iterrows():
        targets = build_zip_targets(row, namelist_set, include_annotations)
        expected = 3 if include_annotations else 1
        missing += expected - len(targets)
        all_targets.extend(targets)

    for t in all_targets:
        t.expected_size = zf.getinfo(t.zip_name).file_size
        total_bytes += t.expected_size

    if missing:
        _log(f"ℹ️  {missing} ficheros esperados no existen en el dataset "
             f"(pacientes sin alguna modalidad) — se omitirán sin error", log_fh)

    return total_bytes, all_targets


def _check_disk_dynamic(dest_dir: str, required_bytes: int, log_fh=None) -> None:
    free = shutil.disk_usage(dest_dir).free
    needed = required_bytes * SAFETY_MARGIN
    _log(
        f"Espacio requerido real (996 pacientes): {required_bytes/1e9:.2f} GB  "
        f"(con margen {SAFETY_MARGIN}x: {needed/1e9:.2f} GB)  |  libre: {free/1e9:.2f} GB",
        log_fh
    )
    if free < needed:
        _log(
            f"❌ Espacio insuficiente para la submuestra real: "
            f"faltan {(needed - free)/1e9:.2f} GB", log_fh
        )
        sys.exit(EXIT_PRECHECK)


class _ProgressTracker:
    def __init__(self, total_files: int, total_bytes: int, interval: int, log_fh=None):
        self.total_files = total_files
        self.total_bytes = total_bytes
        self.interval = interval
        self.log_fh = log_fh
        self.done_files = 0
        self.done_bytes = 0
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.last_log = self.t0

    def update(self, nbytes: int) -> None:
        with self.lock:
            self.done_files += 1
            self.done_bytes += nbytes
            now = time.time()
            if now - self.last_log >= self.interval or self.done_files == self.total_files:
                elapsed = now - self.t0
                pct = self.done_files / self.total_files * 100 if self.total_files else 0
                mbs = self.done_bytes / elapsed / 1e6 if elapsed > 0 else 0
                eta_s = (self.total_bytes - self.done_bytes) / (mbs * 1e6) if mbs > 0 else 0
                eta_str = f"{eta_s/3600:.1f}h" if eta_s > 3600 else f"{eta_s/60:.0f}min"
                _log(
                    f"Progreso: {self.done_files}/{self.total_files} ficheros ({pct:.1f}%)  "
                    f"{self.done_bytes/1e9:.2f}/{self.total_bytes/1e9:.2f} GB  "
                    f"{mbs:.1f} MB/s  ETA={eta_str}",
                    self.log_fh
                )
                self.last_log = now


_thread_local = threading.local()


def _get_thread_zip(auth):
    if not hasattr(_thread_local, "zf"):
        _thread_local.zf = RemoteZip(DOWNLOAD_URL, auth=auth)
    return _thread_local.zf


class _CircuitBreaker:
    """Corta la descarga si se acumulan fallos consecutivos.

    Kaggle bloqueó (404) TODO el endpoint de descarga completa del dataset
    large tras ~50 ficheros en la primera prueba real, y el script siguió
    martilleando el mismo endpoint bloqueado durante los ~1500 intentos
    restantes. Este breaker corta el proceso a la primera racha sospechosa
    en vez de agotar los 996 pacientes contra un endpoint caído.
    """

    def __init__(self, threshold: int = 8):
        self.threshold = threshold
        self.consecutive = 0
        self.tripped = False
        self.lock = threading.Lock()

    def record(self, success: bool) -> None:
        with self.lock:
            if success:
                self.consecutive = 0
            else:
                self.consecutive += 1
                if self.consecutive >= self.threshold:
                    self.tripped = True

    def is_tripped(self) -> bool:
        with self.lock:
            return self.tripped


def _download_one(target: ZipTarget, dest_dir: str, auth, tracker: _ProgressTracker,
                   breaker: "_CircuitBreaker", delay: float, log_fh=None) -> bool:
    dest_path = os.path.join(dest_dir, target.dest_relpath)

    if breaker.is_tripped():
        return False

    if os.path.exists(dest_path) and os.path.getsize(dest_path) == target.expected_size:
        tracker.update(target.expected_size)
        breaker.record(True)
        return True

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        zf = _get_thread_zip(auth)
        data = zf.read(target.zip_name)
        with open(dest_path, "wb") as fh:
            fh.write(data)
        tracker.update(len(data))
        breaker.record(True)
        if delay:
            time.sleep(delay)
        return True
    except Exception as e:
        _log(f"❌ Error descargando {target.zip_name}: {e}", log_fh)
        breaker.record(False)
        return False


def download_manifest(all_targets: list, dest_dir: str, auth, total_bytes: int,
                       workers: int, interval: int, delay: float, log_fh=None):
    """Devuelve (failures, aborted). `aborted=True` si saltó el circuit breaker."""
    tracker = _ProgressTracker(len(all_targets), total_bytes, interval, log_fh)
    breaker = _CircuitBreaker(threshold=8)
    failures = 0

    if workers <= 1:
        zf = RemoteZip(DOWNLOAD_URL, auth=auth)
        _thread_local.zf = zf
        for t in all_targets:
            if breaker.is_tripped():
                break
            if not _download_one(t, dest_dir, auth, tracker, breaker, delay, log_fh):
                failures += 1
        zf.close()
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_download_one, t, dest_dir, auth, tracker, breaker, delay, log_fh)
                       for t in all_targets]
            for fut in as_completed(futures):
                if not fut.result():
                    failures += 1

    aborted = breaker.is_tripped()
    if aborted:
        _log(
            f"⛔ Circuit breaker activado: {breaker.threshold} fallos consecutivos. "
            f"Se detiene la descarga en vez de seguir golpeando un endpoint probablemente "
            f"bloqueado/con cuota agotada. Los ficheros ya descargados se conservan; "
            f"vuelve a lanzar el mismo comando más tarde para reanudar (se salta lo ya bajado).",
            log_fh
        )

    return failures, aborted


def _verify(dest_dir: str, all_targets: list, log_fh=None) -> None:
    _log(f"Verificando integridad de {dest_dir}...", log_fh)
    expected = len(all_targets)
    found = 0
    for t in all_targets:
        if os.path.exists(os.path.join(dest_dir, t.dest_relpath)):
            found += 1

    _log(f"  Encontrados : {found}/{expected} ficheros esperados", log_fh)
    if found != expected:
        _log(f"⚠️  Faltan {expected - found} ficheros", log_fh)
        sys.exit(EXIT_VERIFY)

    _log("✅ Verificación OK.", log_fh)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga una submuestra estratificada (498 pos + 498 neg emparejados "
                    "por edad) del dataset PhysioNet Challenge 2026 'large' sin bajar el ZIP completo."
    )
    parser.add_argument("--output_dir", default=BASE_PATH,
                        help=f"Directorio de salida (default: {BASE_PATH})")
    parser.add_argument("--log", default="download_pipeline.log",
                        help="Fichero de log (default: download_pipeline.log)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Segundos entre líneas de progreso (default: 30)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla para el emparejamiento por edad (default: 42)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Hilos concurrentes para la descarga selectiva (default: 1, "
                             "secuencial: el endpoint de descarga completa de Kaggle parece "
                             "tener un límite de ráfaga muy sensible, no lo satures)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Segundos de pausa entre ficheros descargados, para no disparar "
                             "el límite de peticiones de Kaggle (default: 0.5)")
    parser.add_argument("--no-annotations", action="store_true",
                        help="No descargar algorithmic_annotations/ ni human_annotations/, solo physiological_data/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo comprobaciones + muestreo estadístico, sin descargar nada")
    parser.add_argument("--verify-only", action="store_true",
                        help="Solo verifica una descarga ya existente")
    args = parser.parse_args()
    include_annotations = not args.no_annotations

    log_fh = open(args.log, "a", buffering=1)
    _log("=" * 70, log_fh)
    _log("Iniciando download_large_stratified.py", log_fh)
    _log(f"Destino  : {args.output_dir}", log_fh)
    _log(f"Log      : {args.log}", log_fh)

    creds = _check_prerequisites(args.output_dir, args.dry_run, log_fh)
    auth = (creds["username"], creds["key"])

    _log(f"Abriendo directorio central del ZIP remoto ({DOWNLOAD_URL})...", log_fh)
    t0 = time.time()
    try:
        zf_ctx = RemoteZip(DOWNLOAD_URL, auth=auth)
    except Exception as e:
        _log(
            f"❌ No se pudo abrir el ZIP remoto: {e}\n"
            f"    Si es un 404 en el endpoint de descarga completa, probablemente sigue "
            f"activo el límite de peticiones de Kaggle detectado el 13-14 jul 2026. "
            f"Espera un tiempo (se desconoce cuánto exactamente: no es cuestión de minutos) "
            f"y reintenta con --dry-run antes de lanzar la descarga real.",
            log_fh
        )
        log_fh.close()
        sys.exit(EXIT_PRECHECK)

    with zf_ctx as zf:
        _log(f"Directorio central leído en {time.time()-t0:.1f}s ({len(zf.namelist())} entradas)", log_fh)
        namelist_set = set(zf.namelist())

        demographics_bytes = zf.read("demographics.csv")
        demographics = pd.read_csv(io.BytesIO(demographics_bytes))
        _log(f"demographics.csv: {len(demographics)} pacientes, "
             f"{(demographics['Cognitive_Impairment'] == True).sum()} positivos", log_fh)  # noqa: E712

        manifest = build_manifest(demographics, seed=args.seed, log_fh=log_fh)
        _log(f"Manifiesto final: {len(manifest)} pacientes "
             f"({(manifest['Cognitive_Impairment'] == True).sum()} pos / "
             f"{(manifest['Cognitive_Impairment'] == False).sum()} neg)", log_fh)  # noqa: E712

        required_bytes, all_targets = compute_required_bytes(
            manifest, zf, namelist_set, include_annotations, log_fh
        )

        if args.verify_only:
            _verify(args.output_dir, all_targets, log_fh)
            log_fh.close()
            sys.exit(EXIT_OK)

        _check_disk_dynamic(args.output_dir, required_bytes, log_fh)

        if args.dry_run:
            _log("Dry-run: prechequeos + muestreo + cálculo de espacio OK. Sin descarga.", log_fh)
            log_fh.close()
            sys.exit(EXIT_DRYRUN)

        os.makedirs(args.output_dir, exist_ok=True)
        manifest.to_csv(os.path.join(args.output_dir, "demographics.csv"), index=False)
        icd_bytes = zf.read("ICD_codes_CI.csv") if "ICD_codes_CI.csv" in namelist_set else None
        if icd_bytes:
            with open(os.path.join(args.output_dir, "ICD_codes_CI.csv"), "wb") as fh:
                fh.write(icd_bytes)

    _log(f"Iniciando descarga selectiva de {len(all_targets)} ficheros "
         f"con {args.workers} hilo(s), delay={args.delay}s (se saltan los ya descargados)...", log_fh)
    t0 = time.time()
    failures, aborted = download_manifest(
        all_targets, args.output_dir, auth, required_bytes, args.workers, args.interval,
        args.delay, log_fh
    )
    elapsed = time.time() - t0
    _log(f"Descarga terminada en {elapsed/60:.1f} min ({failures} errores, aborted={aborted})", log_fh)

    if aborted:
        log_fh.close()
        sys.exit(EXIT_DOWNLOAD)

    if failures:
        _log(f"❌ {failures} ficheros fallaron. Revisa el log para detalles.", log_fh)
        log_fh.close()
        sys.exit(EXIT_DOWNLOAD)

    _verify(args.output_dir, all_targets, log_fh)
    log_fh.close()
    sys.exit(EXIT_OK)


if __name__ == "__main__":
    main()
