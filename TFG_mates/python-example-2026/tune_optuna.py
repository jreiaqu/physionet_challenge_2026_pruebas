# Ejecutar: python tune_optuna.py --models_dir ../mi_nueva_carpeta

import os
import optuna
import numpy as np
import pandas as pd
import shutil
import json
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# Importamos TUS funciones desde tu script principal
# Cambia 'new_first_try' por el nombre real de tu archivo .py (sin el .py)
from CNN import train_model, load_model, run_model, DEMOGRAPHICS_FILE, load_diagnoses

# --- 2. Configuramos el lector de argumentos de la terminal ---
parser = argparse.ArgumentParser(description="Optimización de hiperparámetros de CNN con Optuna.")
parser.add_argument(
    '--models_dir', 
    type=str, 
    default='../models_optuna_default', # Valor por defecto si no pones nada
    help='Ruta de la carpeta donde se guardarán los trials de Optuna.'
)
args = parser.parse_args()

# Configuración de carpetas
TRAIN_DATA_FOLDER = '../training_data'  # La carpeta con tus 493 pacientes
VAL_DATA_FOLDER = '../val_data'        # IMPORTANTE: Necesitamos pacientes distintos para evaluar (si no tienes, coge unos pocos del train)
OPTUNA_MODELS_FOLDER = args.models_dir

os.makedirs(OPTUNA_MODELS_FOLDER, exist_ok=True)

def evaluate_model_auc(model, data_folder):
    """Evalúa el modelo y devuelve AUC, Accuracy y F1-Score"""
    import helper_code
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = helper_code.find_patients(patient_data_file)
    
    y_true = []
    y_probs = []
    
    for record in patient_metadata_list:
        patient_id = record[helper_code.HEADERS['bids_folder']]
        diagnosis_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
        label = float(helper_code.load_diagnoses(diagnosis_file, patient_id))
        y_true.append(label)
        
        try:
            _, prob = run_model(model, record, data_folder, verbose=False)
            y_probs.append(prob)
        except Exception:
            y_probs.append(0.5)
            
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
    """La función que Optuna intentará MAXIMIZAR"""
    
    # 1. OPTUNA ELIGE LOS HIPERPARÁMETROS (Definimos los rangos de búsqueda)
    # Le obligamos a mantener max_pool = 0 porque ya sabes que es mejor
    params = {
        'n_blocks': trial.suggest_int('n_blocks', 2, 3),
        'initial_filters': trial.suggest_int('initial_filters', 4, 16), 
        'kernel_size': trial.suggest_int('kernel_size', 11, 31, step=2), # step=2 fuerza números impares
        'pool_size': trial.suggest_int('pool_size', 2, 4),
        'stride': trial.suggest_int('stride', 1, 3),
        'padding': trial.suggest_int('padding', 1, 3),
        'adaptive_pool': trial.suggest_int('adaptive_pool', 50, 150),
        'dense_units': trial.suggest_int('dense_units', 64, 256),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.6),
        'max_pool': trial.suggest_int('max_pool', 0, 1), 
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'bs': trial.suggest_categorical('bs', [16, 32]), 
        'ne': 30  # FIJADO
    }

    # Carpeta temporal para guardar el modelo de esta prueba
    trial_folder = os.path.join(OPTUNA_MODELS_FOLDER, f'trial_{trial.number}')
    print(f"\n Iniciando Trial {trial.number} con parámetros:\n{params}")
    
    os.makedirs(trial_folder, exist_ok=True)

    with open(os.path.join(trial_folder, 'grid_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    # 2. ENTRENAR EL MODELO
    # Llamamos a tu función de entrenamiento desempaquetando el diccionario
    try:
        train_model(
            data_folder=TRAIN_DATA_FOLDER,
            model_folder=trial_folder,
            verbose=False,  # En falso para que no sature la terminal
            **params
        )
    except Exception as e:
        print(f"Error en el entrenamiento del trial {trial.number}: {e}")
        # Si un modelo es tan grande que da Out of Memory, lo penalizamos con un AUC de 0
        return 0.0
        
# 3. CARGAR Y EVALUAR EL MODELO (EN EL SET DE VALIDACIÓN)
    try:
        trained_model = load_model(trial_folder, verbose=False)
        # Recogemos las 3 notas
        auc_score, acc_score, f1_score = evaluate_model_auc(trained_model, VAL_DATA_FOLDER)
        
        # MAGIA: Guardamos Acc y F1 en la memoria de este Trial
        trial.set_user_attr("accuracy", acc_score)
        trial.set_user_attr("f1_score", f1_score)
        
        print(f"✅ Trial {trial.number} terminado. AUC: {auc_score:.4f} | Acc: {acc_score:.4f} | F1: {f1_score:.4f}")
    except Exception as e:
        print(f"❌ Error al evaluar trial {trial.number}: {e}")
        return 0.0
        
    # 4. TRUCO DE MAGIA: Borrar SOLO el peso pesado (model.pth) y dejar el JSON
    model_file_path = os.path.join(trial_folder, 'model.pth')
    try:
        if os.path.exists(model_file_path):
            os.remove(model_file_path)
    except Exception as e:
        print(f"⚠️ No se pudo borrar el archivo pesado del trial {trial.number}: {e}")
        
    # 5. DEVOLVER LA NOTA A OPTUNA
    return auc_score

if __name__ == "__main__":
    print("Iniciando búsqueda con Optuna...")
    
    # Crear un "estudio" que busque MAXIMIZAR el valor devuelto (el AUC-ROC)
    study = optuna.create_study(
        study_name="optimization_CNN_1D", 
        direction="maximize",
        # Usamos TPE (Tree-structured Parzen Estimator), el mejor para hiperparámetros
        sampler=optuna.samplers.TPESampler() 
    )
    
    '''# Dos combinaciones que probará primero
    # 1. El Monstruo del Contexto
    study.enqueue_trial({
        'n_blocks': 3, 'initial_filters': 32, 'kernel_size': 51,
        'pool_size': 4, 'stride': 2, 'padding': 2, 'adaptive_pool': 200,
        'dense_units': 256, 'dropout_rate': 0.45, 'max_pool': 0,
        'lr': 1e-4, 'bs': 32, 'ne': 30
    })

    # 2. El Bisturí Profundo
    study.enqueue_trial({
        'n_blocks': 5, 'initial_filters': 8, 'kernel_size': 15,
        'pool_size': 2, 'stride': 1, 'padding': 2, 'adaptive_pool': 400,
        'dense_units': 128, 'dropout_rate': 0.2, 'max_pool': 0,
        'lr': 5e-4, 'bs': 32, 'ne': 30
    })

    # 3. El Cuello de Botella (Compresión extrema)
    study.enqueue_trial({
        'n_blocks': 4, 'initial_filters': 24, 'kernel_size': 25,
        'pool_size': 6, 'stride': 3, 'padding': 2, 'adaptive_pool': 100,
        'dense_units': 64, 'dropout_rate': 0.3, 'max_pool': 0,
        'lr': 1e-4, 'bs': 32, 'ne': 30
    })'''

    # Lanzar la optimización. Le decimos que intente 30 combinaciones distintas (n_trials).
    # Puedes subir este número, o pararlo con Ctrl+C cuando quieras.
    study.optimize(objective, n_trials=30)
    
    print("\n🎉 ¡Optimización finalizada!")
    print("Mejores hiperparámetros encontrados:")
    print(study.best_params)
    print(f"Mejor AUC obtenido: {study.best_value:.4f}")

    # GUARDAR ABSOLUTAMENTE TODO EN UN EXCEL/CSV
    df_resultados = study.trials_dataframe()
    # Si quieres quitar columnas internas raras de Optuna para que quede más limpio:
    df_resultados = df_resultados.drop(['datetime_start', 'datetime_complete'], axis=1, errors='ignore')
    
    csv_path = os.path.join(OPTUNA_MODELS_FOLDER, 'resultados_completos_optuna.csv')
    df_resultados.to_csv(csv_path, index=False)
    
    # Si de verdad lo quieres en JSON también, es solo esta línea:
    # df_resultados.to_json(csv_path.replace('.csv', '.json'), orient='records', indent=4)
    
    print(f"\n📊 Archivo con todos los modelos y métricas guardado en: {csv_path}")