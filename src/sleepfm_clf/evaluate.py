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


def run_optuna(model_module, data, y, n_trials, folds=None):
    """
    Búsqueda de hiperparámetros. Devuelve best_params.

    folds: lista opcional de (train_idx, val_idx) ya calculados (p.ej. desde
    splits/kfold5_large.json vía sleepfm_clf.splits). Si es None (comportamiento
    original de train_clf.py, sin tocar), usa un StratifiedKFold 3-fold interno
    y efímero — NO uses folds=None en el harness de ablaciones: ahí SIEMPRE hay
    que pasar los folds persistidos para que todos los experimentos se comparen
    sobre los mismos pacientes.
    """
    dummy = np.arange(len(y)).reshape(-1, 1)
    _folds = folds if folds is not None else list(
        StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(dummy, y)
    )
    n_inner = len(_folds)

    def objective(trial):
        params = model_module.suggest_params(trial)
        print(f"\nTrial {trial.number+1:3d}/{n_trials} | {_fmt(params)}", flush=True)
        fold_aucs = []
        for fi, (tr, val) in enumerate(_folds, 1):
            auc, _ = model_module.train_fold(data, y, tr, val, params)
            fold_aucs.append(auc)
            print(f"  fold {fi}/{n_inner} → {auc:.4f}", flush=True)
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
    best_params = study.best_params
    # Algunos model_module codifican choices no-escalares (p.ej. hidden_dims) como claves
    # string en suggest_categorical (Optuna solo acepta None/bool/int/float/str como choices
    # persistibles) y las traducen de vuelta a su forma real en resolve_params — si no se
    # aplica aquí, study.best_params se queda con la clave string cruda.
    if hasattr(model_module, "resolve_params"):
        best_params = model_module.resolve_params(best_params)
    print(f"\n  Mejor AUROC age-cond Optuna ({n_inner}-fold): {study.best_value:.4f}")
    print(f"  Mejores hipers: {best_params}")
    return best_params


def run_cv(model_module, data, y, params, folds=None):
    """
    CV estratificado con los params dados. Devuelve (mean, std, best_fold_data, oof).

    folds: lista opcional de (train_idx, val_idx); si None usa un StratifiedKFold
    N_FOLDS-fold interno efímero (comportamiento original de train_clf.py).
    oof: dict {"idx": np.ndarray, "y": np.ndarray, "probs": np.ndarray} con las
    predicciones out-of-fold concatenadas de todos los folds (para calibración /
    umbral / ensemble posteriores — familias I y J).
    """
    dummy = np.arange(len(y)).reshape(-1, 1)
    _folds = folds if folds is not None else list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(dummy, y)
    )
    aucs, best_auc, best_fold_data = [], -np.inf, None
    oof_idx, oof_y, oof_probs = [], [], []

    print(f"\n── {len(_folds)}-Fold CV ───────────────────────────────────────────────────")
    for fold, (tr, val) in enumerate(_folds, 1):
        auc, probs = model_module.train_fold(data, y, tr, val, params)
        aucs.append(auc)
        print(f"  Fold {fold}: {auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_fold_data = (y[val], probs)
        oof_idx.append(np.asarray(val))
        oof_y.append(np.asarray(y[val]))
        oof_probs.append(np.asarray(probs))

    mean, std = float(np.mean(aucs)), float(np.std(aucs))
    print(f"\n  CV AUROC age-cond: {mean:.4f} ± {std:.4f}  |  mejor fold: {best_auc:.4f}")
    oof = {
        "idx": np.concatenate(oof_idx),
        "y": np.concatenate(oof_y),
        "probs": np.concatenate(oof_probs),
    }
    return mean, std, best_fold_data, oof


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
