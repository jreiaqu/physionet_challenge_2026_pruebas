"""
metrics.py
==========
Métricas del PhysioNet Challenge 2026:
  age_conditioned_auroc()  — AUROC restringido a parejas de edad similar
  prevalence_reward()      — reward ponderado por prevalencia local por edad
  find_threshold_youden()  — umbral óptimo por índice de Youden (val → test)
"""
import warnings
import numpy as np
from sklearn.metrics import roc_curve


def age_conditioned_auroc(y_true, y_score, ages, delta=2.0):
    """
    AUROC restringido a parejas (positivo i, negativo j) con |age_i - age_j| <= delta.

    Fórmula del challenge:
        s_C = Pr(z_i >= z_j | x_i=1, x_j=0, |age_i - age_j| <= delta)

    Retorna np.nan si no existe ninguna pareja válida (warning emitido).
    Complejidad O(n_pos × n_neg) vectorizado — válido para n < 2000.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    ages    = np.asarray(ages,    dtype=float)

    pos_mask = y_true == 1
    neg_mask = y_true == 0

    if pos_mask.sum() == 0 or neg_mask.sum() == 0:
        warnings.warn("age_conditioned_auroc: solo una clase presente. Devolviendo np.nan.")
        return np.nan

    pos_scores = y_score[pos_mask]   # (n_pos,)
    pos_ages   = ages[pos_mask]
    neg_scores = y_score[neg_mask]   # (n_neg,)
    neg_ages   = ages[neg_mask]

    # Matrices (n_pos, n_neg)
    age_diff = np.abs(pos_ages[:, None] - neg_ages[None, :])
    valid    = age_diff <= delta

    n_valid = int(valid.sum())
    if n_valid == 0:
        warnings.warn(
            f"age_conditioned_auroc: ninguna pareja con |Δedad| <= {delta} años. "
            "Devolviendo np.nan."
        )
        return np.nan

    wins = ((pos_scores[:, None] >= neg_scores[None, :]) & valid).sum()
    return float(wins) / n_valid


def prevalence_reward(y_true, y_pred_binary, ages, train_ages, train_labels, delta=2.0):
    """
    Reward del challenge ponderado por prevalencia local estimada en training.

    Para cada paciente k:
      p_a = prevalencia de positivos en training con |age - age_k| <= delta
      r_k:  TP → 1/p_a - 1  |  FP → -1  |  FN → -1  |  TN → 1/(1-p_a) - 1

    Casos extremos:
      - Sin pacientes de training en la ventana  → prevalencia global (warning).
      - p_a = 0 o p_a = 1                        → paciente excluido (warning).

    Retorna (mean_reward: float, per_patient_rewards: np.ndarray).
    """
    y_true        = np.asarray(y_true,        dtype=float)
    y_pred_binary = np.asarray(y_pred_binary, dtype=float)
    ages          = np.asarray(ages,          dtype=float)
    train_ages    = np.asarray(train_ages,    dtype=float)
    train_labels  = np.asarray(train_labels,  dtype=float)

    global_prev = float(train_labels.mean()) if len(train_labels) > 0 else 0.5

    rewards  = []
    excluded = 0

    for k in range(len(y_true)):
        age_k = ages[k]
        local = np.abs(train_ages - age_k) <= delta

        if local.sum() == 0:
            warnings.warn(
                f"prevalence_reward: sin pacientes de training con |age - {age_k:.1f}| "
                f"<= {delta}. Usando prevalencia global ({global_prev:.3f})."
            )
            p_a = global_prev
        else:
            p_a = float(train_labels[local].mean())

        if p_a == 0.0:
            warnings.warn(
                f"prevalence_reward: p_a=0 (sin positivos) para edad={age_k:.1f}. "
                "Paciente excluido del reward."
            )
            excluded += 1
            continue
        if p_a == 1.0:
            warnings.warn(
                f"prevalence_reward: p_a=1 (sin negativos) para edad={age_k:.1f}. "
                "Paciente excluido del reward."
            )
            excluded += 1
            continue

        x_k = y_true[k]
        y_k = y_pred_binary[k]

        if   x_k == 1 and y_k == 1:  r_k = 1.0 / p_a - 1.0          # TP
        elif x_k == 0 and y_k == 1:  r_k = -1.0                       # FP
        elif x_k == 1 and y_k == 0:  r_k = -1.0                       # FN
        else:                         r_k = 1.0 / (1.0 - p_a) - 1.0  # TN

        rewards.append(r_k)

    if excluded > 0:
        warnings.warn(f"prevalence_reward: {excluded} paciente(s) excluido(s) por p_a extrema.")

    if len(rewards) == 0:
        warnings.warn("prevalence_reward: ningún paciente contribuye al reward. Devolviendo np.nan.")
        return np.nan, np.array([])

    return float(np.mean(rewards)), np.array(rewards)


def find_threshold_youden(y_true, y_score):
    """
    Umbral que maximiza el índice de Youden (TPR - FPR) en los datos dados.
    Calcúlalo sobre validación y aplícalo fijo en test, sin recalcular.
    """
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true, dtype=float),
                                     np.asarray(y_score, dtype=float))
    best = int(np.argmax(tpr - fpr))
    return float(thresholds[best])


# ── Tests unitarios ───────────────────────────────────────────────────────────

def _test_age_conditioned_auroc():
    rng = np.random.default_rng(0)

    # Caso 1: AUROC perfecto — positivos siempre puntúan más
    y  = np.array([1, 1, 0, 0], dtype=float)
    sc = np.array([0.9, 0.8, 0.2, 0.1])
    ag = np.array([50., 51., 50., 52.])
    auc = age_conditioned_auroc(y, sc, ag, delta=5)
    assert auc == 1.0, f"Esperado 1.0, obtenido {auc}"

    # Caso 2: Scores aleatorios → AUROC ≈ 0.5
    n   = 300
    y   = np.array([1]*60 + [0]*240, dtype=float)
    sc  = rng.uniform(0, 1, n)
    ag  = rng.uniform(50, 80, n)
    auc = age_conditioned_auroc(y, sc, ag, delta=2)
    assert 0.3 < auc < 0.7, f"Esperado ~0.5, obtenido {auc}"

    # Caso 3: Sin parejas válidas (edades muy separadas) → np.nan + warning
    y  = np.array([1., 0.])
    sc = np.array([0.9, 0.1])
    ag = np.array([40., 80.])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        auc = age_conditioned_auroc(y, sc, ag, delta=2)
        assert np.isnan(auc), f"Esperado np.nan, obtenido {auc}"
        assert len(w) >= 1

    print("  age_conditioned_auroc: OK")


def _test_prevalence_reward():
    # 2 positivos, 2 negativos, misma franja de edad
    # training: 2 pos + 3 neg → p_a = 0.4
    y_true       = np.array([1., 1., 0., 0.])
    y_pred       = np.array([1., 0., 0., 1.])
    ages         = np.array([60., 60., 60., 60.])
    train_ages   = np.array([60., 60., 60., 60., 60.])
    train_labels = np.array([1.,  1.,  0.,  0.,  0.])

    p_a = 0.4
    expected = np.mean([1/p_a - 1, -1.0, 1/(1-p_a) - 1, -1.0])
    mean_r, _ = prevalence_reward(y_true, y_pred, ages, train_ages, train_labels, delta=2)
    assert abs(mean_r - expected) < 1e-6, f"Esperado {expected:.4f}, obtenido {mean_r:.4f}"

    # p_a = 0 → exclusión con warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mean_r, rewards = prevalence_reward(
            np.array([0.]), np.array([0.]),
            np.array([60.]), np.array([60.]), np.array([0.]), delta=2
        )
        assert np.isnan(mean_r)
        assert len(rewards) == 0

    print("  prevalence_reward: OK")


if __name__ == "__main__":
    print("Ejecutando tests de métricas del challenge...")
    _test_age_conditioned_auroc()
    _test_prevalence_reward()
    print("Todos los tests pasaron.")
