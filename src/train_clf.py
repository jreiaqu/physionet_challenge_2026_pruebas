"""
train_clf.py
============
Punto de entrada para entrenar y evaluar clasificadores sobre embeddings SleepFM.

Uso:
  python train_clf.py --model mlp
  python train_clf.py --model lstm
  python train_clf.py --model lstm --trials 100
  python train_clf.py --model mlp --dataset large
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sleepfm_clf.config import (
    CSV_PATHS, SEED, N_TRIALS, DEVICE, EMB_DIRS, DEFAULT_WINDOW_SIZE, DEFAULT_DATASET
)
from sleepfm_clf import evaluate
from sleepfm_clf.metrics import (
    age_conditioned_auroc,
    prevalence_reward,
    find_threshold_youden,
)


def main():
    parser = argparse.ArgumentParser(description="SleepFM classifier con Optuna")
    parser.add_argument("--model",  choices=["mlp", "lstm"], required=True)
    parser.add_argument("--dataset", choices=list(EMB_DIRS.keys()), default=DEFAULT_DATASET,
                        help=f"Dataset a usar: small (291 pac.) o large (996 pac., "
                             f"498 pos + 498 neg emparejados por edad) (default: {DEFAULT_DATASET})")
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Número de trials Optuna (default: {N_TRIALS})")
    parser.add_argument("--window-size", choices=list(EMB_DIRS[DEFAULT_DATASET].keys()),
                        default=DEFAULT_WINDOW_SIZE,
                        help=f"Resolución temporal de embeddings (default: {DEFAULT_WINDOW_SIZE})")
    parser.add_argument("--plot", action="store_true",
                        help="Guardar curva ROC como PNG (por defecto desactivado)")
    args = parser.parse_args()
    window_size = args.window_size

    print(f"Device: {DEVICE}  |  Modelo: {args.model.upper()}  |  "
          f"Dataset: {args.dataset}  |  Ventanas: {window_size}")

    if args.model == "mlp":
        from sleepfm_clf.models import mlp as model_module
    else:
        from sleepfm_clf.models import lstm as model_module

    emb_dir = EMB_DIRS[args.dataset][window_size]
    if not os.path.isdir(emb_dir):
        raise FileNotFoundError(
            f"No existen embeddings para dataset='{args.dataset}' en {emb_dir}. "
            f"Genera antes los embeddings con: "
            f"1) data_processing/preprocessing_raw_data.py (EDF → hdf5), "
            f"2) src/generate_embeddings.py --dataset_name physionet2026_{args.dataset} "
            f"(hdf5 → embeddings SleepFM)."
        )

    df = pd.read_csv(CSV_PATHS[args.dataset])
    data, y = model_module.load_data(df, window_size=window_size, dataset=args.dataset)

    # Test oculto: separado antes de cualquier optimización
    all_idx = np.arange(len(y))
    tv_idx, test_idx = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=SEED)

    data_tv   = model_module.index_data(data, tv_idx)
    data_test = model_module.index_data(data, test_idx)
    y_tv, y_test = y[tv_idx], y[test_idx]
    print(f"Train+Val: {len(y_tv)}  |  Test oculto: {len(y_test)}")

    # Edades para métricas del challenge
    train_ages = model_module.extract_ages(data_tv)
    test_ages  = model_module.extract_ages(data_test)

    # ── Optuna sobre train+val únicamente ────────────────────────────────────
    print(f"\n── Optimización de hiperparámetros ({args.trials} trials Optuna) ──────")
    best_params = evaluate.run_optuna(model_module, data_tv, y_tv, n_trials=args.trials)

    # ── 5-fold CV con los mejores hiperparámetros ─────────────────────────────
    cv_mean, cv_std, best_fold = evaluate.run_cv(model_module, data_tv, y_tv, best_params)

    # Umbral óptimo (Youden's J) fijado sobre el mejor fold de validación
    if best_fold is not None:
        y_val_best, probs_val_best = best_fold
        threshold = find_threshold_youden(y_val_best, probs_val_best)
    else:
        threshold = 0.5
    print(f"  Umbral Youden (val): {threshold:.4f}")

    # ── Evaluación final: reentrenar en todo train+val, predecir en test oculto
    print(f"\n── Evaluación en test oculto ───────────────────────────────────────")
    data_all = model_module.concat_data(data_tv, data_test)
    y_all    = np.concatenate([y_tv, y_test])
    tr_full  = np.arange(len(y_tv))
    te_full  = np.arange(len(y_tv), len(y_all))

    test_auc_age_cond, test_probs = model_module.train_fold(
        data_all, y_all, tr_full, te_full, best_params
    )

    # AUROC estándar (solo reporting, no controla decisiones)
    try:
        test_auroc_std = float(roc_auc_score(y_test, test_probs))
    except ValueError:
        test_auroc_std = float("nan")

    # Prevalence reward con umbral Youden fijado desde validación
    y_pred_binary_test = (test_probs >= threshold).astype(int)
    test_reward, _ = prevalence_reward(
        y_test, y_pred_binary_test, test_ages, train_ages, y_tv
    )

    print(f"  Test AUROC (age-cond):  {test_auc_age_cond:.4f}")
    print(f"  Test AUROC (estándar):  {test_auroc_std:.4f}")
    print(f"  Test Reward:            {test_reward:.4f}")
    print(f"  Umbral Youden aplicado: {threshold:.4f}")

    # ── Curva ROC (solo si se pasa --plot) ───────────────────────────────────
    if args.plot:
        roc_fname = f"roc_{args.model}_{args.dataset}_{window_size}_sleepfm.png"
        evaluate.plot_roc(y_test, test_probs, test_auroc_std, roc_fname)

    # ── Guardar resultados en JSON ────────────────────────────────────────────
    results = {
        "timestamp":            datetime.datetime.now().isoformat(),
        "model":                args.model,
        "dataset":              args.dataset,
        "window_size":          window_size,
        "n_optuna_trials":      args.trials,
        "best_params":          best_params,
        "val_auroc_age_cond":   round(cv_mean, 6),
        "val_auroc_age_cond_std": round(cv_std, 6),
        "threshold_youden":     round(threshold, 6),
        "test_auroc_age_cond":  round(float(test_auc_age_cond), 6),
        "test_auroc_std":       round(float(test_auroc_std), 6),
        "test_reward":          round(float(test_reward), 6) if not np.isnan(test_reward) else None,
    }

    os.makedirs("../results", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = f"../results/test_results_{args.model}_{args.dataset}_{window_size}_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResultados guardados en {json_path}")


if __name__ == "__main__":
    main()
