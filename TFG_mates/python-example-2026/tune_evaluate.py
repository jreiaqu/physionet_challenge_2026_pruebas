import os
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# Importamos tus funciones
from CNN import load_model, run_model
from helper_code import find_patients, DEMOGRAPHICS_FILE, HEADERS, load_diagnoses

def evaluate_all_models():
    print("⚖️ Iniciando Evaluación Masiva de Modelos...")

    # RUTAS (¡Ajusta a tu carpeta de validación local donde SÍ haya diagnósticos!)
    val_data_folder = '../cv_splits/split_3/test_data' # Idealmente debería ser una carpeta separada 'val_data'
    base_model_folder = '../best_model'
    
    # 1. Cargar la lista de pacientes a evaluar
    patient_data_file = os.path.join(val_data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    
    if not patient_metadata_list:
        print("❌ No se encontraron pacientes en la carpeta de validación.")
        return

    # Buscar todas las carpetas de modelos generadas por tune_mega.py
    model_folders = [f.path for f in os.scandir(base_model_folder) if f.is_dir()]
    print(f"Encontrados {len(model_folders)} modelos para evaluar.\n")

    resultados_finales = []

    # 2. Bucle principal: Evaluar modelo a modelo
    for m_folder in model_folders:
        nombre_modelo = os.path.basename(m_folder)
        print(f"Evaluando: {nombre_modelo}...")
        
        try:
            # Cargamos el modelo (tu función load_model ya lee los hiperparámetros sola)
            model = load_model(m_folder, verbose=False)
            
            y_true = []
            y_pred = []
            y_prob = []
            
            # 3. Evaluar paciente a paciente
            for record in patient_metadata_list:
                patient_id = record[HEADERS['bids_folder']]
                
                # Obtener la verdad absoluta (del médico)
                # Necesitamos que la etiqueta exista en el demographics.csv
                true_label = float(load_diagnoses(patient_data_file, patient_id))
                
                # Obtener la predicción de tu red
                bin_out, prob_out = run_model(model, record, val_data_folder, verbose=False)
                
                y_true.append(true_label)
                y_pred.append(float(bin_out))
                y_prob.append(prob_out)
                
            # 4. Calcular métricas matemáticas de rendimiento
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # El AUC (Área bajo la curva ROC) es la métrica más importante en medicina
            try:
                auc = roc_auc_score(y_true, y_prob)
            except ValueError:
                auc = 0.0 # Por si el modelo predice todo ceros y el AUC se rompe
                
            # Leer qué hiperparámetros tenía este modelo para apuntarlos en la tabla
            json_path = os.path.join(m_folder, 'grid_params.json')
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    params = json.load(f)
            else:
                params = {}

            # Guardar la fila de resultados
            fila = {
                'Modelo': nombre_modelo,
                'Accuracy': acc,
                'F1_Score': f1,
                'AUC_ROC': auc
            }
            # Añadir los hiperparámetros a las columnas de la tabla
            fila.update(params)
            resultados_finales.append(fila)

        except Exception as e:
            print(f"❌ Error al evaluar {nombre_modelo}: {e}")
            continue

    # 5. Crear una tabla (DataFrame) ordenando del mejor al peor según AUC
    if resultados_finales:
        df_resultados = pd.DataFrame(resultados_finales)
        df_resultados = df_resultados.sort_values(by='AUC_ROC', ascending=False)
        
        # Guardar en un CSV para que puedas abrirlo en Excel y ponerlo en la memoria del TFG
        output_csv = 'ranking_modelos.csv'
        df_resultados.to_csv(output_csv, index=False)
        
        print("\n=======================================================")
        print(f"✅ Evaluación terminada. Ranking guardado en '{output_csv}'")
        print("Top 3 mejores modelos:")
        print(df_resultados[['Modelo', 'AUC_ROC', 'F1_Score', 'Accuracy']].head(3).to_string(index=False))
        print("=======================================================")

if __name__ == '__main__':
    evaluate_all_models()