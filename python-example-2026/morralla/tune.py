import os
import itertools
import json
from new_first_try import train_model

def run_full_grid_search():
    print("🚀 Iniciando el Mega Grid Search...")

    # 1. Definir el espacio de búsqueda (Ajusta estos valores con cuidado)
    # ¡Empieza con poquitas opciones para probar que no explote la memoria!
    grid = {
    "n_blocks": [2],
    "initial_filters": [16],
    "kernel_size": [32],
    "pool_size": [3],
    "stride": [1],
    "adaptive_pool": [250],
    "dense_units": [256],
    "dropout_rate": [0.3],
    "max_pool": [0],
    "lr": [0.0001],
    "bs": [32],
    "ne": [30]
}
    

    # Extraer los nombres de los parámetros y sus listas de opciones
    keys = list(grid.keys())
    values = list(grid.values())

    # Generar TODAS las combinaciones posibles (Producto Cartesiano)
    combinaciones = list(itertools.product(*values))
    total_modelos = len(combinaciones)
    
    print(f"⚠️ Atención: Se van a entrenar {total_modelos} modelos distintos.")
    print("===================================================================")

    base_model_folder = '../models_grid_3'
    os.makedirs(base_model_folder, exist_ok=True)

    # 2. Bucle sobre cada combinación generada
    for idx, combinacion in enumerate(combinaciones):
        # Unir las claves con la combinación actual en un diccionario
        params = dict(zip(keys, combinacion))
        
        # Crear un nombre de carpeta manejable (ej: model_001_b2_lr0.001)
        folder_name = f"model_{idx:03d}_b{params['n_blocks']}_lr{params['lr']}"
        model_folder = os.path.join(base_model_folder, folder_name)
        os.makedirs(model_folder, exist_ok=True)

        # Guardar la "receta" exacta en un JSON dentro de su carpeta
        # Así siempre sabrás qué hiperparámetros tenía el model_042
        with open(os.path.join(model_folder, 'grid_params.json'), 'w') as f:
            json.dump(params, f, indent=4)

        print(f"\n[{idx+1}/{total_modelos}] Entrenando: {folder_name}")
        # Opcional: imprimir los parámetros actuales
        # print(f"Parametros: {params}")

        try:
            # 3. Llamar a tu función mágica.
            # El **params "desempaqueta" el diccionario y le pasa cada variable 
            # a train_model exactamente por su nombre. ¡Magia de Python!
            train_model(
                data_folder='../training_data', 
                model_folder=model_folder, 
                verbose=True,  # En False para no ensuciar la terminal con 2000 barras de carga
                **params
            )
        except Exception as e:
            print(f"❌ Error entrenando el modelo {folder_name}: {e}")
            continue

    print("\n✅ ¡Búsqueda en rejilla masiva completada!")

if __name__ == '__main__':
    run_full_grid_search()