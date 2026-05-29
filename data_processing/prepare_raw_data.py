"""
download_physiological_data.py
================================
Descarga los archivos .edf de señales fisiológicas crudas del dataset
PhysioNet Challenge 2026 desde Kaggle.

Estructura de salida:
    ../data/physiological_data/
        I0002/
            sub-I000215000_ses-1_eeg.edf
            sub-I000215000_ses-1_ecg.edf
            ...
        I0006/
            ...
        S0001/
            ...

Uso:
    python download_physiological_data.py

Requisitos:
    pip install kaggle
    kaggle.json configurado en ~/.kaggle/kaggle.json
"""

import os
import subprocess
import re
import sys

# ── CONFIGURACIÓN ─────────────────────────────────────────────────────────────
DATASET   = "physionet/physionetchallenge2026data"
BASE_PATH = "../../data/physiological_data"
SITIOS    = ["I0002", "I0006", "S0001"]
PREFIJO   = "training_set/physiological_data"
# ──────────────────────────────────────────────────────────────────────────────


def paginar_archivos_kaggle() -> list[str]:
    """
    Recorre todas las páginas de 'kaggle datasets files' usando el truco
    del page-token y devuelve todas las líneas de salida concatenadas.
    """
    todas_las_lineas = []
    page_token = None
    pagina = 1

    print("── Paginando índice de Kaggle ──────────────────────────────────────")
    while True:
        cmd = ["kaggle", "datasets", "files", DATASET, "--page-size", "100"]
        if page_token:
            cmd.extend(["--page-token", page_token])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"[ERROR] Kaggle CLI devolvió error:\n{result.stderr}")
            sys.exit(1)

        output = result.stdout
        lineas = output.split("\n")
        todas_las_lineas.extend(lineas)
        print(f"  Página {pagina}: {len(lineas)} líneas leídas.")

        token_match = re.search(r"Next Page Token\s*=\s*(\S+)", output)
        if token_match:
            page_token = token_match.group(1)
            pagina += 1
        else:
            print("  → Fin de la paginación.\n")
            break

    return todas_las_lineas


def filtrar_edfs_fisiologicos(todas_las_lineas: list[str]) -> list[tuple[str, str]]:
    """
    Filtra las líneas para quedarse solo con los .edf de physiological_data
    de los sitios de interés.

    Devuelve lista de (sitio, ruta_remota).
    """
    targets = []
    vistos  = set()  # evitar duplicados

    for linea in todas_las_lineas:
        linea = linea.strip()
        if not linea:
            continue

        # Solo nos interesan archivos dentro de training_set/physiological_data y con .edf
        if PREFIJO not in linea:
            continue
        ruta_remota = linea.split()[0]
        if not ruta_remota.lower().endswith(".edf"):
            continue

        for sitio in SITIOS:
            if f"/{sitio}/" in ruta_remota or f"\\{sitio}\\" in ruta_remota:

                if ruta_remota not in vistos:
                    vistos.add(ruta_remota)
                    targets.append((sitio, ruta_remota))
                break

    return targets


def descargar_targets(targets: list[tuple[str, str]]) -> None:
    """
    Descarga cada archivo si no existe ya en local.
    Respeta la estructura de carpetas por sitio.
    """
    total     = len(targets)
    omitidos  = 0
    descargados = 0
    errores   = 0

    print(f"── Descargando {total} archivos EDF ───────────────────────────────")

    for i, (sitio, ruta_remota) in enumerate(targets, start=1):
        dest_dir       = os.path.join(BASE_PATH, sitio)
        os.makedirs(dest_dir, exist_ok=True)

        nombre_archivo = os.path.basename(ruta_remota)
        ruta_local     = os.path.join(dest_dir, nombre_archivo)

        # ── Bloque de seguridad: no re-descargar lo que ya existe ──────────
        if os.path.exists(ruta_local) and os.path.getsize(ruta_local) > 0:
            print(f"  [{i:04d}/{total}] ⏭  Ya existe: {sitio}/{nombre_archivo}")
            omitidos += 1
            continue

        print(f"  [{i:04d}/{total}] ⬇  {sitio}/{nombre_archivo}")

        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                DATASET,
                "-f", ruta_remota,
                "-p", dest_dir,
                "--force",          # sobreescribe descargas a medias
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"           ❌ Error: {result.stderr.strip()}")
            errores += 1
        else:
            descargados += 1

    # ── Resumen final ───────────────────────────────────────────────────────
    print("\n═══════════════════════════════════════════════════════════════════")
    print(f"  Total encontrados : {total}")
    print(f"  Descargados ahora : {descargados}")
    print(f"  Ya existían       : {omitidos}")
    print(f"  Errores           : {errores}")
    print("═══════════════════════════════════════════════════════════════════\n")

    if errores > 0:
        print("⚠  Hay errores. Vuelve a ejecutar el script: los archivos ya")
        print("   descargados se saltarán y solo reintentará los fallidos.\n")


def verificar_kaggle_cli() -> None:
    """Comprueba que kaggle CLI esté instalado y configurado."""
    result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    if result.returncode != 0:
        print("❌ kaggle CLI no encontrado. Instálalo con: pip install kaggle")
        print("   Y configura ~/.kaggle/kaggle.json con tu API key.")
        sys.exit(1)

    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        print("❌ No se encontró ~/.kaggle/kaggle.json")
        print("   Descárgalo desde https://www.kaggle.com/settings → API → Create token")
        sys.exit(1)


def main() -> None:
    print(f"\nDataset : {DATASET}")
    print(f"Destino : {os.path.abspath(BASE_PATH)}")
    print(f"Sitios  : {', '.join(SITIOS)}\n")

    todas_las_lineas = paginar_archivos_kaggle()

    targets = filtrar_edfs_fisiologicos(todas_las_lineas)
    print(f"Archivos EDF de señales crudas encontrados: {len(targets)}\n")

    if not targets:
        print("⚠  No se encontró ningún EDF en physiological_data.")
        sys.exit(0)

    descargar_targets(targets)
    print("✅ Descarga completada.")

if __name__ == "__main__":
    main()
