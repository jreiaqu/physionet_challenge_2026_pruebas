import os
import glob
import pandas as pd
import shutil

# --- CONFIGURACIÓN ---
BASE_DATA_FOLDER = "../data"
CSV_PATH = os.path.join(BASE_DATA_FOLDER, "demographics_total.csv")
CSV_BACKUP = os.path.join(BASE_DATA_FOLDER, "demographics_total_BACKUP.csv")
LOG_PATH = "dataset_errors_log.csv"

def fast_purge():
    print("🚀 Iniciando purga ultra-rápida basada en el log de errores...")

    # 1. Comprobar que existen los archivos necesarios
    if not os.path.exists(LOG_PATH):
        print(f"❌ No se encontró '{LOG_PATH}'. ¡No hay nada que purgar!")
        return
    if not os.path.exists(CSV_PATH):
        print(f"❌ No se encontró el CSV maestro en '{CSV_PATH}'.")
        return

    # 2. Copia de seguridad del CSV original
    if not os.path.exists(CSV_BACKUP):
        shutil.copy2(CSV_PATH, CSV_BACKUP)
        print(f"💾 Copia de seguridad creada: {CSV_BACKUP}")

    # 3. Leer los datos
    df_errores = pd.read_csv(LOG_PATH)
    df_total = pd.read_csv(CSV_PATH)
    
    lista_negra = df_errores['Patient'].tolist()
    print(f"\n🎯 Se han identificado {len(lista_negra)} pacientes para eliminar.")

    # 4. Eliminar los archivos físicos (.edf)
    archivos_borrados = 0
    for _, row in df_errores.iterrows():
        patient_id = row['Patient']
        site_id = row['Site']
        
        # Buscamos cualquier .edf de este paciente usando comodines (*)
        search_pattern = os.path.join(BASE_DATA_FOLDER, "algorithmic_annotations", site_id, f"{patient_id}_*.edf")
        archivos = glob.glob(search_pattern)
        
        for archivo in archivos:
            try:
                os.remove(archivo)
                archivos_borrados += 1
                print(f"  🗑️ Borrado físico: {os.path.basename(archivo)}")
            except Exception as e:
                print(f"  ⚠️ No se pudo borrar {archivo}: {e}")

    # 5. Eliminar a los pacientes del DataFrame
    # Nos quedamos con los pacientes cuyo ID NO está (~) en la lista negra
    df_limpio = df_total[~df_total['BidsFolder'].isin(lista_negra)]

    # 6. Sobrescribir el CSV maestro
    df_limpio.to_csv(CSV_PATH, index=False)

    print("\n" + "="*50)
    print("✨ PURGA COMPLETADA CON ÉXITO ✨")
    print("="*50)
    print(f"Archivos .edf eliminados: {archivos_borrados}")
    print(f"Pacientes en el CSV antes: {len(df_total)}")
    print(f"Pacientes en el CSV ahora: {len(df_limpio)}")
    print("="*50)
    print("Ya puedes borrar '../cv_splits', regenerarlos y lanzar Optuna.")

if __name__ == "__main__":
    fast_purge()