"""
train_clf.py
============
Punto de entrada para entrenar y evaluar clasificadores sobre embeddings SleepFM.

Uso:
  python train_clf.py --model mlp
  python train_clf.py --model lstm
  python train_clf.py --model lstm --trials 100
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sleepfm_clf.config import CSV_PATH, SEED, N_TRIALS, DEVICE, EMB_DIRS, DEFAULT_WINDOW_SIZE
from sleepfm_clf import evaluate


def main():
    parser = argparse.ArgumentParser(description="SleepFM classifier con Optuna")
    parser.add_argument("--model",  choices=["mlp", "lstm"], required=True)
    parser.add_argument("--trials", type=int, default=N_TRIALS,
                        help=f"Número de trials Optuna (default: {N_TRIALS})")
    parser.add_argument("--window-size", choices=list(EMB_DIRS.keys()),
                        default=DEFAULT_WINDOW_SIZE,
                        help=f"Resolución temporal de embeddings (default: {DEFAULT_WINDOW_SIZE})")
    args = parser.parse_args()
    window_size = args.window_size

    print(f"Device: {DEVICE}  |  Modelo: {args.model.upper()}  |  Ventanas: {window_size}")

    if args.model == "mlp":
        from sleepfm_clf.models import mlp as model_module
    else:
        from sleepfm_clf.models import lstm as model_module

    df = pd.read_csv(CSV_PATH)
    data, y = model_module.load_data(df, window_size=window_size)

    # Test oculto: separado antes de cualquier optimización
    all_idx = np.arange(len(y))
    tv_idx, test_idx = train_test_split(all_idx, test_size=0.2, stratify=y, random_state=SEED)

    data_tv   = model_module.index_data(data, tv_idx)
    data_test = model_module.index_data(data, test_idx)
    y_tv, y_test = y[tv_idx], y[test_idx]
    print(f"Train+Val: {len(y_tv)}  |  Test oculto: {len(y_test)}")

    # Optuna sobre train+val únicamente
    print(f"\n── Optimización de hiperparámetros ({args.trials} trials Optuna) ──────")
    best_params = evaluate.run_optuna(model_module, data_tv, y_tv, n_trials=args.trials)

    # 5-fold CV con los mejores hiperparámetros
    evaluate.run_cv(model_module, data_tv, y_tv, best_params)

    # Evaluación final: reentrenar en todo train+val, predecir en test oculto
    print(f"\n── Evaluación en test oculto ───────────────────────────────────")
    data_all = model_module.concat_data(data_tv, data_test)
    y_all    = np.concatenate([y_tv, y_test])
    tr_full  = np.arange(len(y_tv))
    te_full  = np.arange(len(y_tv), len(y_all))

    test_auc, test_probs = model_module.train_fold(data_all, y_all, tr_full, te_full, best_params)
    print(f"  Test AUROC: {test_auc:.4f}")

    evaluate.plot_roc(y_test, test_probs, test_auc, f"roc_{args.model}_{window_size}_sleepfm.png")


if __name__ == "__main__":
    main()
