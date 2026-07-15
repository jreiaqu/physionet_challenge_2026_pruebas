"""
evaluate.py
===========
Utilidades de evaluación compartidas por MLP y LSTM:
  run_optuna() → búsqueda de hiperparámetros con Optuna (3-fold CV interno)
  run_cv()     → evaluación final con 5-fold CV
  plot_roc()   → genera curva ROC y la guarda como PNG
"""
import numpy as np
import matplotlib.pyplot as plt
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from .config import N_FOLDS, SEED


def _fmt(params):
    parts = []
    for k, v in params.items():
        parts.append(f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)


def run_optuna(model_module, data, y, n_trials):
    """Búsqueda de hiperparámetros con 3-fold CV interno. Devuelve best_params."""
    dummy = np.arange(len(y)).reshape(-1, 1)

    def objective(trial):
        params = model_module.suggest_params(trial)
        print(f"\nTrial {trial.number+1:3d}/{n_trials} | {_fmt(params)}", flush=True)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        fold_aucs = []
        for fi, (tr, val) in enumerate(skf.split(dummy, y), 1):
            auc, _ = model_module.train_fold(data, y, tr, val, params)
            fold_aucs.append(auc)
            print(f"  fold {fi}/3 → {auc:.4f}", flush=True)
        mean = float(np.mean(fold_aucs))
        print(f"  mean={mean:.4f}", flush=True)
        return mean

    def callback(study, trial):
        if trial.value == study.best_value:
            print(f"  ★ nuevo mejor: {study.best_value:.4f}", flush=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])
    print(f"\n  Mejor AUROC age-cond Optuna (3-fold): {study.best_value:.4f}")
    print(f"  Mejores hipers: {study.best_params}")
    return study.best_params


def run_cv(model_module, data, y, params):
    """5-fold CV estratificado con los params dados. Devuelve (mean, std, best_fold_data)."""
    dummy = np.arange(len(y)).reshape(-1, 1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aucs, best_auc, best_fold_data = [], -np.inf, None

    print(f"\n── 5-Fold CV ───────────────────────────────────────────────────")
    for fold, (tr, val) in enumerate(skf.split(dummy, y), 1):
        auc, probs = model_module.train_fold(data, y, tr, val, params)
        aucs.append(auc)
        print(f"  Fold {fold}: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_fold_data = (y[val], probs)

    mean, std = float(np.mean(aucs)), float(np.std(aucs))
    print(f"\n  CV AUROC age-cond: {mean:.4f} ± {std:.4f}  |  mejor fold: {best_auc:.4f}")
    return mean, std, best_fold_data


def plot_roc(y_true, y_prob, auc, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC — SleepFM embeddings")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nCurva ROC guardada en {path}")
