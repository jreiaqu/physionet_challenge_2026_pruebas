# Para restaurar los labels sobreescritos en demographics.csv de test. 

import pandas as pd

def restaurar_etiquetas(ruta_original, ruta_test):
    print("Iniciando restauración de la columna 'Cognitive_Impairment'...")
    
    # 1. Cargar ambos CSVs
    df_original = pd.read_csv(ruta_original)
    df_test = pd.read_csv(ruta_test)
    
    # 2. Crear un "diccionario" a partir del CSV original
    # Esto asocia cada BidsFolder con su valor real de Cognitive_Impairment
    mapeo_real = df_original.set_index('BidsFolder')['Cognitive_Impairment']
    
    # 3. Sobrescribir la columna corrupta en el test usando el mapeo
    df_test['Cognitive_Impairment'] = df_test['BidsFolder'].map(mapeo_real)
    
    # 4. Guardar el archivo arreglado (sobrescribiendo el de test)
    df_test.to_csv(ruta_test, index=False)
    
    print("¡Restauración completada con éxito!")
    print(f"Se han corregido {len(df_test)} filas en el archivo de test.")

if __name__ == "__main__":
    # --- RUTAS ---
    # Pon aquí la ruta al CSV completo de +600 pacientes que te bajaste al principio
    CSV_ORIGINAL = "../training_data/demographics_total.csv" 
    
    # Pon aquí la ruta al CSV de test que se te ha corrompido
    CSV_TEST = "../output_data/demographics.csv"
    
    restaurar_etiquetas(CSV_ORIGINAL, CSV_TEST)