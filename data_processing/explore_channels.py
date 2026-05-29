"""
explore_channels.py
====================
Para cada sitio (I0002, I0006, S0001), lee todos los EDFs de
physiological_data y compara si todos los pacientes tienen
exactamente los mismos canales.

Salida por pantalla:
  - Lista de canales del primer paciente (referencia)
  - Pacientes con canales distintos (añadidos o faltantes)
  - Resumen final

Uso:
    python explore_channels.py
"""

import os
import glob
import mne
from collections import defaultdict

BASE_PATH = "../../data/physiological_data"
SITIOS    = ["I0002", "I0006", "S0001"]

mne.set_log_level("ERROR")  # silenciar output de mne


def get_channels(edf_path: str) -> list[str]:
    """Lee un EDF y devuelve la lista de canales en minúsculas y sin espacios extra."""
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    return [ch.lower().strip() for ch in raw.ch_names], raw.info["sfreq"]


def explorar_sitio(sitio: str) -> None:
    carpeta = os.path.join(BASE_PATH, sitio)
    edfs = sorted(glob.glob(os.path.join(carpeta, "*.edf")))

    if not edfs:
        print(f"\n{'='*60}")
        print(f"  {sitio}: no se encontraron EDFs en {carpeta}")
        return

    print(f"\n{'='*60}")
    print(f"  SITIO: {sitio}  ({len(edfs)} pacientes)")
    print(f"{'='*60}")

    # ── Referencia: primer paciente ──────────────────────────────
    ref_path = edfs[0]
    ref_nombre = os.path.basename(ref_path)
    ref_canales, ref_fs = get_channels(ref_path)
    ref_set = set(ref_canales)

    print(f"\n  Referencia → {ref_nombre}")
    print(f"  fs = {ref_fs} Hz  |  {len(ref_canales)} canales:")
    for ch in ref_canales:
        print(f"    · {ch}")

    # ── Comparación con el resto ─────────────────────────────────
    diferencias: dict[str, dict] = {}   # patient → {extra, missing}
    fs_distintas: list[tuple] = []

    for edf_path in edfs[1:]:
        nombre = os.path.basename(edf_path)
        try:
            canales, fs = get_channels(edf_path)
        except Exception as e:
            diferencias[nombre] = {"error": str(e)}
            continue

        actual_set = set(canales)
        extra   = sorted(actual_set - ref_set)
        missing = sorted(ref_set - actual_set)

        if extra or missing:
            diferencias[nombre] = {"extra": extra, "missing": missing}

        if fs != ref_fs:
            fs_distintas.append((nombre, fs))

    # ── Resultados ───────────────────────────────────────────────
    if not diferencias and not fs_distintas:
        print(f"\n  ✅ Todos los pacientes tienen exactamente los mismos canales y fs.\n")
    else:
        if diferencias:
            print(f"\n  ⚠  {len(diferencias)} pacientes con canales distintos:\n")
            for paciente, diff in diferencias.items():
                print(f"    [{paciente}]")
                if "error" in diff:
                    print(f"      ❌ Error al leer: {diff['error']}")
                else:
                    if diff.get("missing"):
                        print(f"      Faltan : {diff['missing']}")
                    if diff.get("extra"):
                        print(f"      De más : {diff['extra']}")

        if fs_distintas:
            print(f"\n  ⚠  {len(fs_distintas)} pacientes con fs distinta a {ref_fs} Hz:")
            for nombre, fs in fs_distintas:
                print(f"    [{nombre}]  fs = {fs} Hz")

    # ── Unión de todos los canales del sitio ─────────────────────
    print(f"\n  📋 Unión de TODOS los canales vistos en {sitio}:")
    todos = set(ref_canales)
    for edf_path in edfs[1:]:
        try:
            canales, _ = get_channels(edf_path)
            todos.update(canales)
        except Exception:
            pass
    for ch in sorted(todos):
        print(f"    · {ch}")


def main() -> None:
    print("\n🔍 Explorador de canales — physiological_data")
    print(f"   Base path: {os.path.abspath(BASE_PATH)}\n")

    for sitio in SITIOS:
        explorar_sitio(sitio)

    print(f"\n{'='*60}")
    print("  Fin del análisis.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()