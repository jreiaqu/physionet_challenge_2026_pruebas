# Descargar marcas algorítmicas de Kaggle

import os
import subprocess
import re

def download_with_token_loop():
    dataset = "physionet/physionetchallenge2026data"
    base_path = "../data/algorithmic_annotations"
    sitios = ["I0002", "I0006", "S0001"]

    all_files_output = []
    page_token = None

    print("--- 1. Extrayendo páginas con el truco del Token ---")
    while True:
        # Preparamos el comando base
        cmd = ["kaggle", "datasets", "files", dataset, "--page-size", "100"]
        
        # Si tenemos un token de la página anterior, lo añadimos al comando
        if page_token:
            cmd.extend(["--page-token", page_token])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout
        all_files_output.extend(output.split('\n'))

        # Buscar el token de la siguiente página usando expresiones regulares
        token_match = re.search(r"Next Page Token\s*=\s*(\S+)", output)
        if token_match:
            page_token = token_match.group(1)
            print(" -> Página leída. Siguiente token encontrado, iterando...")
        else:
            print(" -> No hay más tokens. Fin de la paginación.")
            break
    
    print("\n--- 2. Filtrando matemáticamente (sin perder el S0001) ---")
    targets = []
    for line in all_files_output:
        # Verificamos que sea una marca, sin importar si es training o supplementary
        if "algorithmic_annotations" in line:
            for s in sitios:
                if f"/{s}/" in line or f"\\{s}\\" in line:
                    file_path = line.split()[0] # Cogemos la ruta
                    targets.append((s, file_path))
                    break
    
    print(f"Archivos encontrados: {len(targets)} (Deberían ser unos 616)")

    print("\n--- 3. Descargando a sus respectivas carpetas ---")
    for i, (sitio, remote_path) in enumerate(targets):
        dest_dir = os.path.join(base_path, sitio)
        os.makedirs(dest_dir, exist_ok=True)
        
        file_name = os.path.basename(remote_path)
        local_file_path = os.path.join(dest_dir, file_name) # Calculamos la ruta final
        
        # --- EL BLOQUE DE SEGURIDAD ---
        if os.path.exists(local_file_path):
            print(f"[{i+1}/{len(targets)}] Saltando {file_name} (Ya existe en {sitio}/)")
            continue # Aborta esta iteración y pasa al siguiente paciente
            
        print(f"[{i+1}/{len(targets)}] Descargando faltante en {sitio}/ -> {file_name}")
        
        # Si llega aquí, es porque el archivo NO existe localmente. 
        # Aquí sí es seguro (y recomendable) dejar el --force para evitar 
        # conflictos si Kaggle dejó alguna descarga a medias oculta.
        subprocess.run([
            "kaggle", "datasets", "download", 
            dataset, "-f", remote_path, 
            "-p", dest_dir, "--force"
        ], capture_output=True)

if __name__ == "__main__":
    download_with_token_loop()