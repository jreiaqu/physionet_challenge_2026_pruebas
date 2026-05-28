import os
import glob
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

def create_kfold_splits_85_15(base_data_folder="../data", 
                              output_folder="../../../cv_splits_balanced", 
                              n_splits=5, 
                              test_ratio=0.15, 
                              target_col="Cognitive_Impairment",  
                              patient_col="BidsFolder", 
                              site_col="SiteID", 
                              session_col="SessionID",
                              seed=42):
    
    print(f"Iniciando generación robusta: {100 - int(test_ratio*100)}% CV / {int(test_ratio*100)}% Test Fijo con {n_splits} Folds...")
    
    # 1. Cargar y verificar integridad (igual que tu código original)
    demo_path = os.path.join(base_data_folder, "demographics_total.csv")
    df_raw = pd.read_csv(demo_path)
    
    print("Verificando integridad de los archivos .edf originales...")
    valid_rows = []
    for _, row in df_raw.iterrows():
        patient_id = row[patient_col]
        site_id = row[site_col]
        session_id = row[session_col]
        
        search_pattern = os.path.join(base_data_folder, "algorithmic_annotations", site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
        matching_files = glob.glob(search_pattern)
        
        if matching_files and os.path.getsize(matching_files[0]) > 1024:
            valid_rows.append(row)
            
    df = pd.DataFrame(valid_rows)
    print(f"Pacientes válidos tras el filtro: {len(df)} de {len(df_raw)}")

    # 2. SEPARAR LA CAJA FUERTE (Test Set Fijo)
    dev_df, test_df = train_test_split(
        df, 
        test_size=test_ratio, 
        stratify=df[target_col], 
        random_state=seed
    )
    
    print(f"\n[CAJA FUERTE] Test Set Global: {len(test_df)} pacientes (Aislados para siempre).")
    print(f"[CV SET] Disponibles para Train/Val: {len(dev_df)} pacientes.")

    # 3. CROSS-VALIDATION SOBRE EL 85% RESTANTE
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # dev_df.reset_index(drop=True) es importante para que skf.split funcione bien con los índices
    dev_df = dev_df.reset_index(drop=True) 
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(dev_df, dev_df[target_col])):
        
        train_df = dev_df.iloc[train_idx]
        val_df = dev_df.iloc[val_idx]
        
        # Estructura de carpetas: cv_splits/split_0/
        fold_folder = os.path.join(output_folder, f"split_{fold}")
        
        splits_to_save = {
            "training_data": train_df,
            "val_data": val_df,
            "test_data": test_df # ¡Mismo dataframe de Test para TODOS los folds!
        }
        
        print(f"\n -> Generando Fold {fold} (Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)})...")
        
        for split_name, split_df in splits_to_save.items():
            dest_folder = os.path.join(fold_folder, split_name)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Guardar CSV (el nombre coincide con lo que espera Optuna)
            csv_filename = f"demographics_split_{fold}_{split_name}.csv"
            split_df.to_csv(os.path.join(dest_folder, csv_filename), index=False)
            
            # Copiar archivos EDF
            for _, row in split_df.iterrows():
                patient_id = row[patient_col]
                site_id = row[site_col]
                session_id = row[session_col]

                search_pattern = os.path.join(base_data_folder, "algorithmic_annotations", site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
                matching_files = glob.glob(search_pattern) 
                
                if matching_files:
                    src_file = matching_files[0]
                    
                    # Recrear estructura interna: algorithmic_annotations/site_id/
                    dest_site_folder = os.path.join(dest_folder, "algorithmic_annotations", site_id)
                    os.makedirs(dest_site_folder, exist_ok=True)
                    
                    # Si el archivo ya existe (para no sobreescribir tontamente el test en cada iteración aunque no pasa nada)
                    dest_file_path = os.path.join(dest_site_folder, os.path.basename(src_file))
                    if not os.path.exists(dest_file_path):
                        shutil.copy2(src_file, dest_file_path)

    print("\n¡Generación de Splits blindada y completada con éxito! Ya puedes lanzar Optuna.")

# Si ejecutas este archivo directamente
if __name__ == "__main__":
    create_kfold_splits_85_15()