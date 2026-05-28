# Ejecutar: python tune_optuna_stratified.py --models_dir ../models_ --model_type 0

import os
import time
import optuna
import numpy as np
import json
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from CNN import train_model, load_model, run_model
from helper_code import HEADERS, find_patients, load_diagnoses

parser = argparse.ArgumentParser(description="Optimización de hiperparámetros de CNN con Optuna.")
parser.add_argument(
    '--models_dir', 
    type=str, 
    default='../models_optuna_default',
    help='Ruta de la carpeta donde se guardarán los trials de Optuna.'
)
parser.add_argument(
    '--model_type', 
    type=int, 
    default=0,
    help='Arquitectura a entrenar: 0=CNN, 1=CNN+LSTM, 2=LSTM, 3=ResNet, 4=Transformer'
)
args = parser.parse_args()

CV_BASE_FOLDER = '/home/local/alumno.upv.es/jreiaqu/cv_splits_balanced'
OPTUNA_MODELS_FOLDER = args.models_dir
MODEL_TYPE = args.model_type
NUM_SPLITS = 5

os.makedirs(OPTUNA_MODELS_FOLDER, exist_ok=True)

def evaluate_model_auc(model, data_folder, csv_path):
    patient_metadata_list = find_patients(csv_path)
    
    y_true = []
    y_probs = []
    
    for record in patient_metadata_list:
        patient_id = record[HEADERS['bids_folder']]
        label = float(load_diagnoses(csv_path, patient_id))
        y_true.append(label)
        
        try:
            _, _, prob = run_model(model, record, data_folder, verbose=False)
            y_probs.append(prob)
        except Exception:
            y_probs.append(0.0)
            
    # Convertir probabilidades a 0 o 1 para Acc y F1 (umbral 0.5)
    y_preds = [1 if p >= 0.5 else 0 for p in y_probs]
            
    # Calcular las 3 métricas (con try-except por si colapsa)
    try: auc = roc_auc_score(y_true, y_probs)
    except ValueError: auc = 0.5
        
    try: acc = accuracy_score(y_true, y_preds)
    except ValueError: acc = 0.0
        
    try: f1 = f1_score(y_true, y_preds, zero_division=0)
    except ValueError: f1 = 0.0
        
    return auc, acc, f1

def objective(trial):
    
    # Parámetros compartidos por todos los modelos
    params = {
        'model_type': MODEL_TYPE,
        'lr': 5e-4,
        'bs': 32,
    }

    if MODEL_TYPE == 0:
        # ----- ESPACIO DE BÚSQUEDA: CNN PURA -----
        params['n_blocks'] = trial.suggest_int('n_blocks', 2, 5)
        params['initial_filters'] = trial.suggest_categorical('initial_filters', [16, 32, 64])
        # Usamos kernel_size impar y calculamos el padding para mantener dimensiones (padding 'same')
        params['kernel_size'] = trial.suggest_int('kernel_size', 3, 11, step=2) 
        params['padding'] = params['kernel_size'] // 2 
        params['pool_size'] = 2
        params['stride'] = 1
        params['adaptive_pool'] = trial.suggest_categorical('adaptive_pool', [50, 100, 250])
        params['dense_units'] = trial.suggest_int('dense_units', 64, 256, step=64)
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)
        params['max_pool'] = trial.suggest_categorical('max_pool', [0, 1])

    elif MODEL_TYPE == 1:
        # ----- ESPACIO DE BÚSQUEDA: CNN + LSTM -----
        # 1. Extracción de características espaciales/frecuenciales (CNN)
        params['n_blocks'] = trial.suggest_int('n_blocks', 2, 4)
        params['initial_filters'] = trial.suggest_categorical('initial_filters', [16, 32])
        params['kernel_size'] = trial.suggest_int('kernel_size', 3, 7, step=2)
        params['padding'] = params['kernel_size'] // 2
        params['pool_size'] = 2
        params['stride'] = 1
        
        # 2. Modelado secuencial temporal (LSTM)
        params['lstm_layers'] = trial.suggest_int('lstm_layers', 1, 2)
        params['lstm_hidden'] = trial.suggest_int('lstm_hidden', 64, 128, step=64)
        params['bidirectional'] = True
        
        # 3. Clasificador final
        params['dense_units'] = trial.suggest_int('dense_units', 64, 256, step=64)
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)
        # Fijamos max_pool a 0 (AvgPool) o 1 (MaxPool) según lo que prefieras para la salida de la CNN
        params['max_pool'] = 0 
        
    elif MODEL_TYPE == 2:
        # ----- ESPACIO DE BÚSQUEDA: LSTM ORIGINAL -----
        params['lstm_layers'] = trial.suggest_int('lstm_layers', 1, 3)
        params['lstm_hidden'] = trial.suggest_int('lstm_hidden', 64, 256, step=64)
        params['bidirectional'] = True
        params['dense_units'] = trial.suggest_int('dense_units', 64, 256, step=64)
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)

    elif MODEL_TYPE == 3:
        # ----- ESPACIO DE BÚSQUEDA: RESNET 1D -----
        params['n_blocks'] = trial.suggest_int('n_blocks', 2, 5)
        params['initial_filters'] = trial.suggest_categorical('initial_filters', [16, 32, 64])
        params['kernel_size'] = trial.suggest_categorical('kernel_size', [3, 5, 7, 9, 11])
        params['padding'] = params['kernel_size'] // 2
        params['dense_units'] = trial.suggest_categorical('dense_units', [64, 128, 256])
        params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)
        params['adaptive_pool'] = trial.suggest_categorical('adaptive_pool', [1])

    elif MODEL_TYPE == 4:
        # ----- ESPACIO DE BÚSQUEDA: TRANSFORMER -----
        # CRÍTICO: d_model debe ser divisible por nhead
        params['d_model'] = 32
        params['nhead'] = trial.suggest_categorical('nhead', [4, 8])
        params['num_layers'] = 2
        params['dim_feedforward'] = 64
        params['dropout_rate'] = 0.3

    else:
        raise ValueError("Aún no has configurado el espacio de búsqueda para este model_type.")
    
    trial_folder = os.path.join(OPTUNA_MODELS_FOLDER, f'trial_{trial.number}')
    
    print(f"\nTrial {trial.number} (Evaluando en {NUM_SPLITS} splits CV)...")    
    os.makedirs(trial_folder, exist_ok=True)

    with open(os.path.join(trial_folder, 'grid_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    val_aucs, test_aucs = [], []
    best_epochs = []

    for i in range(NUM_SPLITS):
        train_folder = os.path.join(CV_BASE_FOLDER, f"split_{i}", "training_data")
        val_folder   = os.path.join(CV_BASE_FOLDER, f"split_{i}", "val_data")
        test_folder  = os.path.join(CV_BASE_FOLDER, f"split_{i}", "test_data")

        csv_train = os.path.join(train_folder, f"demographics_split_{i}_training_data.csv")  
        csv_val = os.path.join(val_folder, f"demographics_split_{i}_val_data.csv")       
        csv_test = os.path.join(test_folder, f"demographics_split_{i}_test_data.csv")

        params['current_trial'] = trial.number
        params['current_split'] = i + 1

        params['val_csv_path'] = csv_val
        params['val_data_folder'] = val_folder

        start_time = time.time()

        try:
            best_epoch_split = train_model(data_folder=train_folder, model_folder=trial_folder,csv_path=csv_train, verbose=True, **params) # ** desempaqueta diccionario
                                                                                                                        # se traduce de 'clave':valor a clave=valor
            trained_model = load_model(trial_folder, verbose=False)
            val_auc, _, _ = evaluate_model_auc(trained_model, val_folder, csv_val)
            test_auc, _, _ = evaluate_model_auc(trained_model, test_folder, csv_test)
            
            val_aucs.append(val_auc)
            test_aucs.append(test_auc)
            best_epochs.append(best_epoch_split)
            
        except Exception as e:
            print(f"Error en Split {i}: {e}")
            return 0.0 # Castigo si falla en RAM/VRAM
        
        end_time = time.time()
        mins, secs = divmod(end_time - start_time, 60)
        print(f"Split {i+1} completado en {int(mins)}m {int(secs)}s | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    # Calcular medias
    mean_val_auc = np.mean(val_aucs)
    mean_test_auc = np.mean(test_aucs)
    mean_best_epoch = np.mean(best_epochs)
    
    trial.set_user_attr("mean_val_auc", mean_val_auc)
    trial.set_user_attr("mean_test_auc", mean_test_auc)
    trial.set_user_attr("mean_epoch", mean_best_epoch)
    trial.set_user_attr("epochs_por_split", str(best_epochs))
    
    print(f"Trial {trial.number} | Mean Val AUC: {mean_val_auc:.4f} | Mean Test AUC: {mean_test_auc:.4f}| Mean Epoch: {mean_best_epoch:.1f}")
    
    model_file_path = os.path.join(trial_folder, 'model.pth')
    if os.path.exists(model_file_path):
        os.remove(model_file_path)
        
    return mean_val_auc

def save_optuna_progress_callback(study, trial):
    # 1. Actualizar y guardar el CSV en cada iteración
    df_resultados = study.trials_dataframe()
    csv_path = os.path.join(OPTUNA_MODELS_FOLDER, 'resultados_cv_optuna.csv')
    df_resultados.to_csv(csv_path, index=False)
    
    # 2. Guardar el JSON del espacio de búsqueda SOLO en el Trial 0
    if trial.number == 0:
        # Obtenemos las distribuciones reales que Optuna acaba de construir
        distribuciones = trial.distributions
        search_space_auto = {param_name: str(dist) for param_name, dist in distribuciones.items()}
        
        # Añadimos las variables fijas para que consten en el log
        search_space_auto['model_type'] = 4
        # search_space_auto['n_blocks'] = 3
        # search_space_auto['kernel_size'] = 5
        #search_space_auto['pool_size_FIJO'] = 2
        #search_space_auto['stride_FIJO'] = 1
        #search_space_auto['padding_FIJO'] = 0
        #search_space_auto['lstm_layers_FIJO'] = 1
        #search_space_auto['bidirectional_FIJO'] = True
        #search_space_auto['torch.mean()'] = False
        # search_space_auto['dropout_FIJO'] = 0.25 # Asumiendo que has subido a 0.5
        # Descomental max_pool si 2 Hz
        # search_space_auto['max_pool_FIJO'] = 0
        # search_space_auto['lr_FIJO'] = 1e-3
        # search_space_auto['bs_FIJO'] = 32

        space_path = os.path.join(OPTUNA_MODELS_FOLDER, 'search_space.json')
        with open(space_path, 'w', encoding='utf-8') as f:
            json.dump(search_space_auto, f, indent=4, ensure_ascii=False)
            
        print(f"\n[Info] Espacio de búsqueda extraído y guardado en: {space_path}")

if __name__ == "__main__":
    study = optuna.create_study(
        study_name="CNN_CV_Prevalences", 
        direction="maximize", 
        sampler=optuna.samplers.TPESampler()
    ) 
    
    print(f"Iniciando optimización. Los resultados se actualizarán en vivo en {OPTUNA_MODELS_FOLDER}")
    
    # Le pasamos el callback. Optuna lo llamará en segundo plano cada vez que acabe un trial.
    study.optimize(objective, n_trials=30, callbacks=[save_optuna_progress_callback]) 
    
    print("\n¡Optimización de Optuna completada al 100%!")