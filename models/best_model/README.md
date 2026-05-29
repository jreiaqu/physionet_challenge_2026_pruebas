# Mejor modelo guardado

## Arquitectura

**Tipo:** SimpleCNN1D (model_type=0)

## Hiperparámetros (`grid_params.json`)

| Parámetro | Valor |
|-----------|-------|
| n_blocks | 2 |
| initial_filters | 16 |
| kernel_size | 32 |
| pool_size | 3 |
| stride | 1 |
| adaptive_pool | 250 |
| dense_units | 256 |
| dropout_rate | 0.3 |
| max_pool | 0 |
| lr | 0.0001 |
| batch_size | 32 |
| epochs | 30 |

## Datos de entrenamiento

- Dataset: `cv_splits_balanced/` (5-fold estratificado)
- Input: 9 canales CAISR × ventanas de 30s
- Etiqueta: `Cognitive_Impairment` (binaria)

## Origen

Guardado manualmente desde la familia de experimentos `models_30s_total/`.
El backup completo de experimentos está en `TFG_mates_backup_20260529.tar.gz`.

## Uso

```python
from src.models_and_training import load_model
model, hparams = load_model("models/best_model")
```
