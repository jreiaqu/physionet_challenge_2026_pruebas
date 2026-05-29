import os
import glob
import pandas as pd
from tqdm import tqdm

# Importamos tu función para leer los EDFs
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_utils import load_signal_data

# --- CONFIGURACIÓN ---
BASE_DATA_FOLDER = "../data"
CSV_PATH = os.path.join(BASE_DATA_FOLDER, "demographics_total.csv")

PATIENT_COL = "BidsFolder"
SITE_COL = "SiteID"
SESSION_COL = "SessionID"
TARGET_COL = "Cognitive_Impairment"

MIN_SIZE_BYTES = 2048 

# Las señales EXACTAS que tu CNN necesita para funcionar
EXPECTED_CHANNELS = [
    'caisr_prob_no-ar', 'caisr_prob_arous', 'limb_caisr', 
    'resp_caisr', 'caisr_prob_n3', 'caisr_prob_n2', 
    'caisr_prob_n1', 'caisr_prob_r', 'caisr_prob_w'
]

def analyze_dataset():
    print(f"\n📊 Analizando {CSV_PATH} y el contenido de los .edf...")

    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: No se encuentra el CSV en: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH)
    total_patients = len(df)
    
    stats = {
        "valid": 0,
        "missing": 0,
        "corrupt": 0,
        "missing_signals": 0, # NUEVA MÉTRICA
        "valid_pos": 0,
        "valid_neg": 0
    }

    bad_patients = []

    print("\n🔍 Escaneando disco duro y abriendo archivos .edf...")
    # Usamos tqdm para ver el progreso (abrir EDFs lleva un poquito más de tiempo)
    for _, row in tqdm(df.iterrows(), total=total_patients, desc="Comprobando Señales"):
        patient_id = row[PATIENT_COL]
        site_id = row[SITE_COL]
        session_id = row[SESSION_COL]
        target = row[TARGET_COL]

        search_pattern = os.path.join(BASE_DATA_FOLDER, "algorithmic_annotations", site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
        matching_files = glob.glob(search_pattern)

        # 1. Comprobar existencia física
        if not matching_files:
            stats["missing"] += 1
            bad_patients.append({"Patient": patient_id, "Site": site_id, "Issue": "Archivo NO existe"})
            continue

        file_path = matching_files[0]
        file_size = os.path.getsize(file_path)

        # 2. Comprobar tamaño (si pesa 0 bytes, load_signal_data crashearía)
        if file_size < MIN_SIZE_BYTES:
            stats["corrupt"] += 1
            bad_patients.append({"Patient": patient_id, "Site": site_id, "Issue": f"Corrupto (Pesa {file_size} bytes)"})
            continue

        # 3. Comprobar señales internas (El nuevo filtro)
        try:
            # Cargamos el archivo usando la librería del Challenge
            algo_data, _ = load_signal_data(file_path)
            
            # Buscamos qué señales de nuestra lista NO están en el diccionario
            missing_channels = [ch for ch in EXPECTED_CHANNELS if ch not in algo_data]
            
            if len(missing_channels) > 0:
                stats["missing_signals"] += 1
                ch_str = ", ".join(missing_channels)
                bad_patients.append({"Patient": patient_id, "Site": site_id, "Issue": f"Faltan señales: [{ch_str}]"})
                continue # Saltamos, no es un paciente válido
                
        except Exception as e:
            # Si load_signal_data falla por cualquier motivo (formato EDF roto internamente)
            stats["corrupt"] += 1
            bad_patients.append({"Patient": patient_id, "Site": site_id, "Issue": f"EDF Roto / Ilegible: {str(e)}"})
            continue

        # 4. Si pasa todos los filtros (existe, pesa, se lee bien y tiene los 9 canales)
        stats["valid"] += 1
        
        if target == True or target == 1 or str(target).lower() == 'true':
            stats["valid_pos"] += 1
        else:
            stats["valid_neg"] += 1

    # --- REPORTE FINAL ---
    print("\n" + "="*55)
    print("📋 REPORTE DE INTEGRIDAD DEL DATASET (NIVEL PROFUNDO)")
    print("="*55)
    print(f"Total esperado (CSV):      {total_patients}")
    print(f"Archivos Válidos (Sanos):  {stats['valid']} ✅")
    print(f"Archivos Faltantes:        {stats['missing']} ❌")
    print(f"Archivos Corruptos:        {stats['corrupt']} ⚠️")
    print(f"Señales Incompletas:       {stats['missing_signals']} 📉 (Tienen el archivo pero faltan canales)")
    print("-" * 55)
    print(f"Balance REAL para Entrenamiento (Solo sobre Válidos):")
    print(f" - Deterioro (Positivos):  {stats['valid_pos']}")
    print(f" - Sanos (Negativos):      {stats['valid_neg']}")
    
    if stats['valid'] > 0:
        prev = (stats['valid_pos'] / stats['valid']) * 100
        print(f" - Prevalencia Real:       {prev:.2f}%")
    print("="*55)

    if bad_patients:
        df_bad = pd.DataFrame(bad_patients)
        log_path = "dataset_errors_log.csv"
        df_bad.to_csv(log_path, index=False)
        print(f"\n📝 Log generado: '{log_path}'.")
        print("   Revisa este Excel para ver exactamente qué canal le falta a cada paciente.")

if __name__ == "__main__":
    analyze_dataset()