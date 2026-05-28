# Dividir en entrene y test/val (2 cjtos en total) (.edf y demographics.csv)

import os
import shutil
import random
import pandas as pd

def split_train_test(src_base, test_base, original_csv_path, test_csv_path, split_ratio=0.2):
    sites = ['I0002', 'I0006', 'S0001']
    columna_id = 'BidsFolder' 
    
    print("--- 1. Limpiando pacientes sin archivo EDF ---")
    # A. Recopilar todos los IDs de los archivos físicos que realmente existen
    archivos_reales_ids = []
    for site in sites:
        src_dir = os.path.join(src_base, site)
        if os.path.exists(src_dir):
            files = [f for f in os.listdir(src_dir) if f.endswith('.edf')]
            for f in files:
                archivos_reales_ids.append(f.split('_')[0])
                
    # B. Cargar el CSV y eliminar las filas fantasma
    df = pd.read_csv(original_csv_path)
    filas_originales = len(df)
    df = df[df[columna_id].isin(archivos_reales_ids)]
    print(f"  -> Eliminados {filas_originales - len(df)} pacientes del CSV que no tenían archivo .edf")
    print(f"  -> Total de pacientes reales a repartir: {len(df)}")

    print(f"\n--- 2. Iniciando división de datos ({split_ratio*100}% para Test) ---")
    test_patient_ids = []
    
    for site in sites:
        src_dir = os.path.join(src_base, site)
        test_dir = os.path.join(test_base, site)
        os.makedirs(test_dir, exist_ok=True)
        
        if not os.path.exists(src_dir):
            continue
            
        files = [f for f in os.listdir(src_dir) if f.endswith('.edf')]
        if not files:
            continue
            
        num_test = int(len(files) * split_ratio)
        random.seed(42) 
        test_files = random.sample(files, num_test)
        
        print(f"  -> Sitio {site}: Moviendo {num_test} de {len(files)} pacientes al test set.")
        
        for f in test_files:
            src_path = os.path.join(src_dir, f)
            dest_path = os.path.join(test_dir, f)
            shutil.move(src_path, dest_path)
            
            patient_id = f.split('_')[0]
            test_patient_ids.append(patient_id)
            
    print("\n--- 3. Dividiendo el demographics.csv ---")
    try:
        # AHORA usamos el 'df' que ya está limpio de fantasmas
        df_test = df[df[columna_id].isin(test_patient_ids)]
        df_train = df[~df[columna_id].isin(test_patient_ids)] 
        
        df_test.to_csv(test_csv_path, index=False)
        df_train.to_csv(original_csv_path, index=False)
        
        print(f"CSV dividido con éxito:")
        print(f"  - Train: {len(df_train)} pacientes (sobrescrito en {original_csv_path})")
        print(f"  - Test:  {len(df_test)} pacientes (guardado en {test_csv_path})")
        
    except Exception as e:
        print(f"\n[!] Error al procesar el CSV: {e}")

if __name__ == "__main__":
    SOURCE_DIR = "../training_data/algorithmic_annotations"
    TEST_DIR = "../val_data/algorithmic_annotations"
    TRAIN_CSV = "../training_data/demographics.csv"
    TEST_CSV = "../val_data/demographics.csv"
    
    split_train_test(SOURCE_DIR, TEST_DIR, TRAIN_CSV, TEST_CSV, split_ratio=0.20)