"""
run_queue.py
============
Runner autónomo del estudio de ablaciones (ver docs del brief / report.md Fase 0).
Lee `queue.yaml`, ejecuta cada experimento pendiente con el protocolo fijo:

  1. Optuna (n_trials, default 20) sobre 5-fold persistido (splits/kfold5_large.json),
     objetivo = age-conditioned AUROC.
  2. Confirmación LOSO (splits/loso_large.json) con los mejores hiperparámetros.
  3. Calibración/umbral (Youden) SOLO sobre predicciones OOF del 5-fold.
  4. Chequeo de realismo en dataset_small (prevalencia natural, excluyendo el solape
     con dataset_large_balanced — ver splits/small_eval_keys.json): refit en TODO
     large + eval en small, mismo patrón que train_clf.py para su test oculto.
  5. Append a results.csv y a report.md.

Resumible: si un exp_id ya está en results.csv, se salta. No requiere modelo/humano
en el bucle — las únicas decisiones automáticas son:
  - Familia G: calcula el hueco humano↔CAISR y decide si la Familia H se ejecuta
    (umbral 0.03, ver ablation_state.json).
  - Familia J (ensemble): se computa al final, sobre los OOF de los top-k experimentos
    ya completados (por LOSO age-cond AUROC).

Uso:
  python run_queue.py                      # corre toda la cola pendiente
  python run_queue.py --only A1 A2         # solo esos exp_id (para depuración)
  python run_queue.py --queue queue_smoke.yaml --results ../results/smoke_results.csv --report ../results/smoke_report.md
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import warnings

# Log limpio en segundo plano (nohup): las barras tqdm por época usan \r y, sobre un
# fichero (no una terminal), quedan como miles de líneas casi idénticas en vez de
# redibujarse in situ. Si stdout no es una tty, las desactivamos aquí ANTES de importar
# los model_module (que hacen `from tqdm import tqdm` a nivel de módulo) — el print()
# estructurado que ya existe (Trial N/n_trials, fold i/n -> auc, CV/LOSO...) sigue
# intacto y es lo que queda en el log. En una terminal interactiva (isatty) no se toca
# nada: las barras siguen viéndose en vivo como siempre.
if not sys.stdout.isatty():
    import functools
    import tqdm as _tqdm_pkg
    _tqdm_pkg.tqdm = functools.partial(_tqdm_pkg.tqdm, disable=True)

# El aviso de Optuna sobre choices no-escalares para persistencia en storage no aplica
# aquí (usamos el estudio en memoria, sin storage=), y ya no debería dispararse tras
# codificar hidden_dims como claves string — se deja este filtro como cinturón y
# tirantes por si algún trial futuro reintroduce una choice no-escalar.
warnings.filterwarnings(
    "ignore", category=UserWarning,
    message="Choices for a categorical distribution should be a tuple.*",
)

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from sleepfm_clf import evaluate
from sleepfm_clf.config import SEED
from sleepfm_clf.metrics import prevalence_reward, find_threshold_youden
from sleepfm_clf.splits import load_usable_df, load_kfold_indices, load_loso_indices, small_eval_mask
from sleepfm_clf.models import mlp as mlp_mod, lstm as lstm_mod, diag_mlp as diag_mlp_mod, coxph as coxph_mod
from sleepfm_clf.models import ablation as ablation_mod

FIELDS = [
    "exp_id", "familia", "base_embedding", "config", "semilla",
    "cv5_agecond_mean", "cv5_agecond_std", "loso_agecond_mean", "loso_agecond_std",
    "loso_per_site", "reward_balanced", "reward_small_natural", "std_auroc", "auprc",
    "f1", "threshold_youden", "n_optuna_trials", "wall_clock_s", "deployable", "notes",
]


def build_module(exp):
    model = exp["model"]
    if model == "mlp":
        return mlp_mod
    if model == "lstm":
        return lstm_mod
    if model == "diag_mlp":
        return diag_mlp_mod
    if model == "coxph":
        return coxph_mod.build(exp.get("config"))
    if model == "ablation":
        return ablation_mod.build(exp.get("config", {}))
    raise ValueError(f"modelo desconocido en queue.yaml: {model}")


def already_done(results_csv, exp_id):
    if not os.path.exists(results_csv):
        return False
    df = pd.read_csv(results_csv)
    return "exp_id" in df.columns and exp_id in df["exp_id"].values


def append_result(results_csv, row):
    write_header = not os.path.exists(results_csv)
    row = {k: row.get(k, "") for k in FIELDS}
    with open(results_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_report(report_md, text):
    with open(report_md, "a") as f:
        f.write(text)


def load_state(state_json):
    if os.path.exists(state_json):
        with open(state_json) as f:
            return json.load(f)
    return {}


def save_state(state_json, state):
    with open(state_json, "w") as f:
        json.dump(state, f, indent=2)


def _get_keys(data):
    if isinstance(data, dict) and "keys" in data:
        return data["keys"]
    return None


def small_realism_check(model_module, exp, data_large, y_large, best_params, threshold, ages_large):
    """Refit en TODO large + eval en dataset_small (excluyendo solape) — mismo patrón
    que train_clf.py usa para su test oculto (concat_data + train_fold train/val)."""
    dataset_small_csv = "../dataset_small/demographics.csv"
    df_small_full = pd.read_csv(dataset_small_csv).reset_index(drop=True)
    mask = small_eval_mask(df_small_full)
    df_small_eval = df_small_full[mask].reset_index(drop=True)
    if len(df_small_eval) == 0:
        return None

    window_size = exp.get("window_size", "5min")
    data_small, y_small = model_module.load_data(df_small_eval, window_size=window_size, dataset="small")
    if len(y_small) == 0 or y_small.sum() == 0 or y_small.sum() == len(y_small):
        return None  # sin ambas clases: age_conditioned_auroc/reward no definidos

    data_all = model_module.concat_data(data_large, data_small)
    y_all = np.concatenate([y_large, y_small])
    tr_idx = np.arange(len(y_large))
    val_idx = np.arange(len(y_large), len(y_all))
    _, probs_small = model_module.train_fold(data_all, y_all, tr_idx, val_idx, best_params)

    ages_small = model_module.extract_ages(data_small)
    reward_small, _ = prevalence_reward(
        y_small, (probs_small >= threshold).astype(int), ages_small, ages_large, y_large
    )
    return reward_small


def run_single(exp, args, state):
    exp_id = exp["exp_id"]
    family = exp.get("family", "?")
    dataset = exp.get("dataset", "large")
    window_size = exp.get("window_size", "5min")
    n_trials = exp.get("n_trials", 20)

    print(f"\n{'='*70}\n=== {exp_id} ({family}) — model={exp['model']} dataset={dataset}/{window_size}\n{'='*70}")
    t0 = time.time()

    # Familia H: gate + inyección de teacher_probs (solo tras Familia G, ver report.md)
    if family == "H":
        if not state.get("H_enabled", False):
            print(f"[SKIP] {exp_id}: Familia H deshabilitada (hueco G < umbral 0.03)")
            append_report(args.report, (
                f"\n### {exp_id} (Familia H) — OMITIDO\n\nEl hueco humano↔CAISR medido en la "
                f"Familia G no superó el umbral de materialidad (0.03). No se entrena "
                f"destilación (regla del brief: \"si el hueco de G es ~0, SALTA H\").\n"
            ))
            append_result(args.results, {"exp_id": exp_id, "familia": family, "notes": "skipped_gap_below_threshold"})
            return
        teacher_exp_id = exp["teacher_exp_id"]
        teacher_oof_path = os.path.join(args.oof_dir, f"{teacher_exp_id}.npz")
        if not os.path.exists(teacher_oof_path):
            print(f"[SKIP] {exp_id}: falta OOF del profesor {teacher_exp_id} (¿corrió ya G?)")
            return
        teacher_oof = np.load(teacher_oof_path, allow_pickle=True)
        teacher_probs_by_key = dict(zip(teacher_oof["keys"], teacher_oof["probs"]))

    df = load_usable_df(dataset) if dataset == "large" else pd.read_csv("../dataset_small/demographics.csv")
    model_module = build_module(exp)
    data, y = model_module.load_data(df, window_size=window_size, dataset=dataset)

    if family == "H":
        keys = _get_keys(data)
        exp["config"]["teacher_probs"] = np.array(
            [teacher_probs_by_key.get(k, 0.5) for k in keys], dtype=np.float32)

    folds = load_kfold_indices(df) if dataset == "large" else None
    best_params = evaluate.run_optuna(model_module, data, y, n_trials=n_trials, folds=folds)
    cv_mean, cv_std, best_fold, oof = evaluate.run_cv(model_module, data, y, best_params, folds=folds)

    keys = _get_keys(data)
    oof_keys = np.array(keys)[oof["idx"]] if keys is not None else oof["idx"].astype(str)
    os.makedirs(args.oof_dir, exist_ok=True)
    np.savez(os.path.join(args.oof_dir, f"{exp_id}.npz"),
             idx=oof["idx"], y=oof["y"], probs=oof["probs"], keys=oof_keys)

    # LOSO
    loso_aucs = {}
    if dataset == "large":
        for site, tr, val in load_loso_indices(df):
            auc, _ = model_module.train_fold(data, y, tr, val, best_params)
            loso_aucs[site] = auc
    loso_vals = list(loso_aucs.values())
    loso_mean = float(np.mean(loso_vals)) if loso_vals else float("nan")
    loso_std = float(np.std(loso_vals)) if loso_vals else float("nan")

    # Umbral + métricas estándar sobre OOF
    ages_all = model_module.extract_ages(data)
    threshold = find_threshold_youden(oof["y"], oof["probs"])
    reward_balanced, _ = prevalence_reward(
        oof["y"], (oof["probs"] >= threshold).astype(int), ages_all[oof["idx"]], ages_all, y
    )
    try:
        std_auroc = float(roc_auc_score(oof["y"], oof["probs"]))
        auprc = float(average_precision_score(oof["y"], oof["probs"]))
        f1 = float(f1_score(oof["y"], (oof["probs"] >= threshold).astype(int)))
    except Exception as e:
        std_auroc = auprc = f1 = float("nan")
        print(f"  [warn] métricas estándar fallaron: {e}")

    # Chequeo de realismo en dataset_small
    reward_small = None
    if dataset == "large" and exp.get("check_small", True):
        try:
            reward_small = small_realism_check(model_module, exp, data, y, best_params, threshold, ages_all)
        except Exception as e:
            print(f"  [warn] chequeo dataset_small falló: {e}")

    wall = time.time() - t0

    row = {
        "exp_id": exp_id, "familia": family,
        "base_embedding": exp.get("base_embedding", "model_base"),
        "config": json.dumps(exp.get("config", {}), default=str),
        "semilla": SEED,
        "cv5_agecond_mean": round(cv_mean, 4), "cv5_agecond_std": round(cv_std, 4),
        "loso_agecond_mean": round(loso_mean, 4) if loso_vals else "",
        "loso_agecond_std": round(loso_std, 4) if loso_vals else "",
        "loso_per_site": json.dumps({k: round(v, 4) for k, v in loso_aucs.items()}),
        "reward_balanced": round(reward_balanced, 4) if not np.isnan(reward_balanced) else "",
        "reward_small_natural": round(reward_small, 4) if reward_small is not None and not np.isnan(reward_small) else "",
        "std_auroc": round(std_auroc, 4), "auprc": round(auprc, 4), "f1": round(f1, 4),
        "threshold_youden": round(threshold, 4),
        "n_optuna_trials": n_trials, "wall_clock_s": round(wall, 1),
        "deployable": exp.get("deployable", True), "notes": exp.get("notes", ""),
    }
    append_result(args.results, row)

    flag = " ⚠ CV>LOSO (posible sobreajuste de sitio)" if (loso_vals and cv_mean - loso_mean > 0.03) else ""
    append_report(args.report, (
        f"\n### {exp_id} — {family}\n\n"
        f"- config: `{json.dumps(exp.get('config', {}), default=str)}`\n"
        f"- best_params (Optuna {n_trials} trials): `{json.dumps(best_params, default=str)}`\n"
        f"- 5-fold age-cond AUROC: **{cv_mean:.4f} ± {cv_std:.4f}**\n"
        f"- LOSO age-cond AUROC: **{loso_mean:.4f} ± {loso_std:.4f}** (por sitio: "
        f"{json.dumps({k: round(v,4) for k,v in loso_aucs.items()})}){flag}\n"
        f"- reward_balanced (OOF, large): {reward_balanced:.4f} | "
        f"reward_small_natural: {reward_small if reward_small is None else round(reward_small,4)}\n"
        f"- AUROC estándar: {std_auroc:.4f} | AUPRC: {auprc:.4f} | F1: {f1:.4f} | "
        f"umbral Youden: {threshold:.4f}\n"
        f"- wall clock: {wall:.1f}s\n"
        f"- veredicto: _[PENDIENTE DE INTERPRETACIÓN]_\n"
    ))

    # Familia G: decide el gate de la Familia H
    if family == "G":
        state.setdefault("G_runs", {})[exp_id] = loso_mean
        g_runs = state["G_runs"]
        if len(g_runs) >= 2 and "human" in exp_id.lower() or exp.get("annotation_source") == "human":
            pass  # el cálculo real del hueco se hace en maybe_resolve_G_gate tras ambas corridas
    print(f"  -> cv5={cv_mean:.4f} loso={loso_mean:.4f} reward_bal={reward_balanced:.4f} "
          f"reward_small={reward_small} wall={wall:.1f}s")


def maybe_resolve_G_gate(queue, state, args):
    """Tras correr ambas ramas de la Familia G (humano vs CAISR), calcula el hueco y
    decide si la Familia H se activa (umbral 0.03, ver brief)."""
    g_exps = [e for e in queue if e.get("family") == "G"]
    ids = [e["exp_id"] for e in g_exps]
    if len(ids) < 2 or not all(already_done(args.results, i) for i in ids):
        return
    if "H_enabled" in state:
        return
    df = pd.read_csv(args.results)
    df_g = df[df["exp_id"].isin(ids)]
    human_row = df_g[df_g["exp_id"].str.contains("human", case=False)]
    caisr_row = df_g[df_g["exp_id"].str.contains("caisr", case=False)]
    if human_row.empty or caisr_row.empty:
        print("[G] no se pudo identificar exp_id humano/caisr por nombre — revisa queue.yaml (usa "
              "'human'/'caisr' en el exp_id de la Familia G).")
        return
    gap = float(human_row["loso_agecond_mean"].iloc[0]) - float(caisr_row["loso_agecond_mean"].iloc[0])
    state["G_gap"] = gap
    state["H_enabled"] = gap >= 0.03
    save_state(args.state, state)
    append_report(args.report, (
        f"\n### Familia G — resolución del gate de la Familia H\n\n"
        f"- hueco humano↔CAISR (LOSO age-cond AUROC): **{gap:.4f}** "
        f"(humano={float(human_row['loso_agecond_mean'].iloc[0]):.4f}, "
        f"caisr={float(caisr_row['loso_agecond_mean'].iloc[0]):.4f})\n"
        f"- umbral de materialidad: 0.03 -> Familia H {'ACTIVADA' if state['H_enabled'] else 'OMITIDA'}\n"
    ))
    print(f"[G] hueco={gap:.4f} -> H_enabled={state['H_enabled']}")


def run_ensemble(exp, args):
    """Familia J: media de probabilidades OOF calibradas (isotónica) de las top-k
    configs por LOSO age-cond AUROC ya completadas."""
    from sklearn.isotonic import IsotonicRegression
    if not os.path.exists(args.results):
        print("[J] no hay resultados previos, se omite ensemble.")
        return
    df = pd.read_csv(args.results)
    df = df[pd.to_numeric(df["loso_agecond_mean"], errors="coerce").notna()]
    df = df[df["exp_id"] != exp["exp_id"]]
    k = exp.get("k", 3)
    top = df.sort_values("loso_agecond_mean", ascending=False).head(k)
    print(f"[J] ensemble de top-{k}: {top['exp_id'].tolist()}")

    probs_by_key = {}
    y_by_key = {}
    for exp_id in top["exp_id"]:
        path = os.path.join(args.oof_dir, f"{exp_id}.npz")
        if not os.path.exists(path):
            continue
        npz = np.load(path, allow_pickle=True)
        iso = IsotonicRegression(out_of_bounds="clip").fit(npz["probs"], npz["y"])
        calibrated = iso.predict(npz["probs"])
        for k_, p, yy in zip(npz["keys"], calibrated, npz["y"]):
            probs_by_key.setdefault(k_, []).append(p)
            y_by_key[k_] = yy

    common_keys = [k_ for k_, v in probs_by_key.items() if len(v) == len(top)]
    if not common_keys:
        print("[J] sin pacientes comunes entre los top-k OOF (folds distintos) — se omite.")
        append_report(args.report, "\n### Ensemble (Familia J) — OMITIDO: sin overlap de pacientes OOF entre folds.\n")
        return
    y_ens = np.array([y_by_key[k_] for k_ in common_keys])
    probs_ens = np.array([np.mean(probs_by_key[k_]) for k_ in common_keys])
    from sleepfm_clf.data import load_aggregated  # solo para extraer edades por key — usamos usable_df
    df_large = load_usable_df("large")
    age_by_key = {f"{r['BidsFolder']}_ses-{r['SessionID']}": r["Age"] for _, r in df_large.iterrows()}
    ages_ens = np.array([age_by_key.get(k_, np.nan) for k_ in common_keys])
    valid = ~np.isnan(ages_ens)
    from sleepfm_clf.metrics import age_conditioned_auroc
    auc = age_conditioned_auroc(y_ens[valid], probs_ens[valid], ages_ens[valid])
    print(f"[J] ensemble age-cond AUROC (OOF, n={valid.sum()}): {auc:.4f}")
    append_report(args.report, (
        f"\n### {exp['exp_id']} — Ensemble (Familia J)\n\n"
        f"- miembros: {top['exp_id'].tolist()}\n"
        f"- age-cond AUROC (OOF combinado, n={int(valid.sum())}): **{auc:.4f}**\n"
        f"- veredicto: _[PENDIENTE DE INTERPRETACIÓN]_\n"
    ))
    append_result(args.results, {
        "exp_id": exp["exp_id"], "familia": "J", "notes": f"members={top['exp_id'].tolist()}",
        "cv5_agecond_mean": round(float(auc), 4),
    })


def reset_state(args):
    """--force: mueve (NUNCA borra) resultados/estado previos a una carpeta de backup con
    timestamp, y reconstruye report.md quedándose solo con la parte estática (Fase 0),
    truncando en el marcador "## Resultados por experimento". Tras esto no queda ningún
    exp_id en results.csv, así que TODOS los experimentos de la cola se ejecutan desde
    cero, sin saltar ninguno."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(os.path.dirname(os.path.abspath(args.results)), f"ablation_backup_{ts}")
    os.makedirs(backup_dir, exist_ok=True)

    report_backup_path = None
    for p in [args.results, args.report, args.state]:
        if os.path.exists(p):
            dest = os.path.join(backup_dir, os.path.basename(p))
            shutil.move(p, dest)
            if p == args.report:
                report_backup_path = dest
    if os.path.isdir(args.oof_dir) and os.listdir(args.oof_dir):
        shutil.move(args.oof_dir, os.path.join(backup_dir, os.path.basename(args.oof_dir)))
    os.makedirs(args.oof_dir, exist_ok=True)

    marker = "## Resultados por experimento"
    placeholder = (
        f"{marker}\n\n_(reiniciado con --force el {ts}; resultados previos en "
        f"`{os.path.basename(backup_dir)}/`. Se añaden automáticamente vía `run_queue.py`.)_\n"
    )
    kept_prefix = ""
    if report_backup_path:
        with open(report_backup_path) as f:
            content = f.read()
        idx = content.find(marker)
        if idx != -1:
            kept_prefix = content[:idx]
    with open(args.report, "w") as f:
        f.write(kept_prefix + placeholder)

    print(f"[--force] resultados anteriores movidos a {backup_dir} (no se han borrado).")
    print(f"[--force] {args.results} / {args.report} / {args.state} / {args.oof_dir} reiniciados: "
          f"ningún exp_id se saltará.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default="queue.yaml")
    ap.add_argument("--results", default="../results/ablation_results.csv")
    ap.add_argument("--report", default="../results/ablation_report.md")
    ap.add_argument("--oof-dir", default="../results/ablation_oof")
    ap.add_argument("--state", default="../results/ablation_state.json")
    ap.add_argument("--only", nargs="*", default=None, help="ejecutar solo estos exp_id")
    ap.add_argument("--force", action="store_true",
                     help="ignora/backup de resultados previos y corre TODA la cola desde cero, "
                          "sin saltar ningún exp_id (ver reset_state)")
    args = ap.parse_args()

    if args.force:
        reset_state(args)

    with open(args.queue) as f:
        queue = yaml.safe_load(f)["experiments"]

    state = load_state(args.state)
    os.makedirs(args.oof_dir, exist_ok=True)

    for exp in queue:
        exp_id = exp["exp_id"]
        if args.only and exp_id not in args.only:
            continue
        if already_done(args.results, exp_id):
            print(f"[SKIP] {exp_id}: ya está en {args.results}")
            continue
        if exp.get("family") == "J":
            run_ensemble(exp, args)
            continue
        try:
            run_single(exp, args, state)
        except Exception as e:
            # IMPORTANTE: no se escribe fila en results.csv en caso de error. Si se
            # escribiera, already_done() la vería como "completado" y este exp_id se
            # saltaría para siempre en el próximo lanzamiento aunque nunca terminó bien.
            # Al no escribir nada, el próximo `python run_queue.py` lo reintenta solo.
            print(f"[ERROR] {exp_id} falló (se reintentará en el próximo lanzamiento): {e}")
            import traceback; traceback.print_exc()
            append_report(args.report, f"\n### {exp_id} — ERROR (no completado, se reintentará)\n\n```\n{e}\n```\n")
        maybe_resolve_G_gate(queue, state, args)

    print("\nCola completada (o pendiente de experimentos con --only).")


if __name__ == "__main__":
    main()
