# generate_splits.py
import os
from CNN import create_stratified_splits

BASE_DATA_FOLDER = '../data' # Carpeta original con los 611 pacientes
CV_BASE_FOLDER = '../../../cv_splits'
PREVALENCIAS = [0.5, 0.5, 0.5, 0.5, 0.5]

if __name__ == "__main__":
    os.makedirs(CV_BASE_FOLDER, exist_ok=True)
    print("Iniciando generación de los 5 splits de validación cruzada...\n")

    for i, prev in enumerate(PREVALENCIAS):
        split_folder = os.path.join(CV_BASE_FOLDER, f"split_{i}")
        
        if not os.path.exists(split_folder):
            print(f" -> Creando Split {i} (Prevalencia: {prev*100}% | Seed: {42+i})")
            create_stratified_splits(
                base_data_folder=BASE_DATA_FOLDER,
                output_folder=split_folder,
                train_ratio=0.7,
                val_ratio=0.15,
                test_ratio=0.15,
                target_prevalence=prev,
                seed=42 + i  # La magia de las semillas distintas
            )
        else:
            print(f" -> Split {i} ya existe en {split_folder}. Saltando.")
            
    print("\n¡Todos los splits generados y listos para Optuna!")