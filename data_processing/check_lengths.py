# Script para calcular percentil 95 duraciones archivos

import os
import numpy as np
# import matplotlib.pyplot as plt
from first_try import find_patients, load_signal_data, DEMOGRAPHICS_FILE, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, HEADERS

def analyze_record_lengths(data_folder):
    patient_data_file = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    patient_metadata_list = find_patients(patient_data_file)
    
    durations_hours = []
    
    print("Calculando duraciones...")
    for record in patient_metadata_list:
        patient_id = record[HEADERS['bids_folder']]
        site_id    = record[HEADERS['site_id']]
        session_id = record[HEADERS['session_id']]
        
        # Ruta al archivo algorítmico
        algo_file = os.path.join(data_folder, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, site_id, f"{patient_id}_ses-{session_id}_caisr_annotations.edf")
        
        if os.path.exists(algo_file):
            try:
                algo_data, _ = load_signal_data(algo_file)
                # Usamos stage_caisr porque sabemos que va a 1 época = 30s
                if 'stage_caisr' in algo_data:
                    n_epochs = len(algo_data['stage_caisr'])
                    hours = (n_epochs * 30) / 3600.0
                    durations_hours.append(hours)
            except Exception as e:
                print(f"Error leyendo {patient_id}: {e}")

    durations_hours = np.array(durations_hours)
    
    print("\n--- Estadísticas de Duración ---")
    print(f"Total archivos analizados: {len(durations_hours)}")
    print(f"Duración Mínima: {np.min(durations_hours):.2f} horas")
    print(f"Duración Media:  {np.mean(durations_hours):.2f} horas")
    print(f"Duración Máxima: {np.max(durations_hours):.2f} horas")
    print(f"Percentil 95:    {np.percentile(durations_hours, 95):.2f} horas")
    
    # Opcional: Mostrar un histograma rápido si lo ejecutas en un entorno gráfico o Jupyter
    # plt.hist(durations_hours, bins=30, edgecolor='black')
    # plt.title('Distribución de Duraciones de Sueño')
    # plt.xlabel('Horas')
    # plt.ylabel('Frecuencia')
    # plt.show()

if __name__ == '__main__':
    analyze_record_lengths('/training_data')