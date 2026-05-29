# TFG — Predicción de Deterioro Cognitivo desde Señales de Sueño

Predicción de deterioro cognitivo a partir de señales fisiológicas de polisomnografía (PSG).
Integra arquitecturas propias de deep learning con el modelo preentrenado **SleepFM**.

## Estructura del proyecto

```
TFG_Mates/
├── src/                        Código fuente principal
│   ├── models_and_training.py  5 arquitecturas DL + entrenamiento (CNN, LSTM, ResNet, Transformer)
│   ├── data_utils.py           Carga EDF, demografía, métricas, estandarización de canales
│   ├── train_model.py          CLI: entrenar modelo desde cero
│   ├── run_model.py            CLI: inferencia sobre datos nuevos
│   ├── hyperparameter_search.py Búsqueda Optuna con 5-fold CV
│   ├── generate_cv_splits.py   Genera los splits de CV estratificados
│   ├── generate_embeddings.py  Genera embeddings SleepFM (ejecutar una vez)
│   ├── mlp_sleepfm.py          Clasificador MLP sobre embeddings SleepFM
│   ├── train_clf.py            CLI: entrenar MLP/LSTM sobre embeddings
│   ├── quick_split_local.py    Split local 80/20 para pruebas rápidas
│   ├── sleepfm_clf/            Paquete modular (MLP + LSTM sobre embeddings)
│   └── configs/                channel_groups.json, channel_table.csv
│
├── data_processing/            Scripts de preparación y validación de datos
├── reference/                  Código de referencia del challenge PhysioNet
├── models/best_model/          Mejor modelo encontrado (model.pth + grid_params.json)
├── results/figures/            Curvas ROC y otras visualizaciones
├── data/                       Datos crudos (191 GB, EDF + demographics_total.csv)
├── cv_splits_balanced/         5-fold CV estratificado (regenerable con generate_cv_splits.py)
├── sleepFM/                    Modelo preentrenado SleepFM (Nature Medicine)
└── requirements.txt            Dependencias Python 3.10
```

## Configuración del entorno

```bash
python -m venv mi_entorno_tfg
source mi_entorno_tfg/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## Flujo de trabajo

### Pipeline 1: Modelos propios (CNN/LSTM/ResNet/Transformer)

```bash
cd src/

# 1. Generar splits de CV (solo la primera vez)
python generate_cv_splits.py

# 2. Buscar hiperparámetros óptimos
python hyperparameter_search.py \
    --CV_data_dir ../cv_splits_balanced \
    --models_dir ../models/experimento_1 \
    --model_type 3   # 0=CNN, 1=CNN+LSTM, 2=LSTM, 3=ResNet, 4=Transformer

# 3. Entrenar modelo final
python train_model.py -d ../data -m ../models/best_model -v

# 4. Inferencia sobre test
python run_model.py -d ../data/test -m ../models/best_model -o ../results/predictions -v
```

### Pipeline 2: Clasificador sobre embeddings SleepFM

```bash
cd src/

# 1. Generar embeddings (solo la primera vez, requiere sleepFM/)
python generate_embeddings.py

# 2. Entrenar y evaluar clasificador MLP o LSTM
python train_clf.py --model mlp --window-size 5min
python train_clf.py --model lstm --window-size 5s
```

## Datos

- **291 pacientes** | 9 canales CAISR (probabilidades de fase de sueño, respiración, movimiento)
- **Etiqueta**: `Cognitive_Impairment` (boolean) — ratio 3:1 (deterioro:control)
- **Formato señales**: EDF → ventanas de 30 segundos
- **Demografía**: `data/demographics_total.csv`

## Modelo guardado

Ver [models/best_model/README.md](models/best_model/README.md) para detalles del mejor modelo.

## Dependencias clave

| Paquete | Versión | Uso |
|---------|---------|-----|
| torch | 2.10.0 | Arquitecturas DL |
| optuna | 4.8.0 | Búsqueda de hiperparámetros |
| edfio | 0.4.10 | Lectura de archivos EDF |
| h5py | 3.16.0 | Lectura de embeddings HDF5 |
| scikit-learn | 1.6.0 | Métricas y splits |
| mne | 1.12.1 | Procesado de señales EEG |
