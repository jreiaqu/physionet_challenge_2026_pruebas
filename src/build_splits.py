"""
build_splits.py
================
Genera y persiste, UNA SOLA VEZ, los splits reutilizados por todo el estudio de
ablaciones (ver docs/ablation_brief.md, Fase 0):
  - StratifiedKFold 5-fold (semilla 42) sobre dataset_large_balanced.
  - Leave-One-Site-Out (LOSO) por SiteID sobre dataset_large_balanced.

Los splits se guardan como claves de paciente ("{BidsFolder}_ses-{SessionID}"),
NO como índices posicionales — así son estables aunque cambie el orden de lectura
del CSV o el subconjunto de columnas cargado.

Uso: python build_splits.py   (idempotente; sobreescribe splits/*.json)
"""
import json
import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
CSV_LARGE = "../dataset_large_balanced/demographics.csv"
CSV_SMALL = "../dataset_small/demographics.csv"
OUT_DIR = "../splits"


def _key(row):
    return f"{row['BidsFolder']}_ses-{row['SessionID']}"


def _label(row):
    return 1 if str(row["Cognitive_Impairment"]).upper() == "TRUE" else 0


def build_kfold(df):
    y = df.apply(_label, axis=1).values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    dummy = df.index.values.reshape(-1, 1)
    for fold, (tr, val) in enumerate(skf.split(dummy, y)):
        folds.append({
            "fold": fold,
            "train": [_key(df.iloc[i]) for i in tr],
            "val":   [_key(df.iloc[i]) for i in val],
        })
        n_pos_tr = int(y[tr].sum())
        n_pos_val = int(y[val].sum())
        print(f"  fold {fold}: train={len(tr)} (pos={n_pos_tr}) | val={len(val)} (pos={n_pos_val})")
    return folds


def build_loso(df):
    sites = sorted(df["SiteID"].unique())
    loso = []
    for site in sites:
        val_df = df[df["SiteID"] == site]
        train_df = df[df["SiteID"] != site]
        n_pos_val = int(val_df.apply(_label, axis=1).sum())
        n_pos_tr = int(train_df.apply(_label, axis=1).sum())
        if n_pos_val < 5 or (len(val_df) - n_pos_val) < 5:
            print(f"  [SKIP] site={site}: muestra insuficiente "
                  f"(pos={n_pos_val}, neg={len(val_df) - n_pos_val}) para LOSO fiable.")
            continue
        loso.append({
            "site": site,
            "train": [_key(r) for _, r in train_df.iterrows()],
            "val":   [_key(r) for _, r in val_df.iterrows()],
        })
        print(f"  site={site}: train={len(train_df)} (pos={n_pos_tr}) | "
              f"val={len(val_df)} (pos={n_pos_val}, neg={len(val_df) - n_pos_val})")
    return loso


EMB_DIR_5MIN = "../sleepFM/sleepfm-clinical/sleepfm/checkpoints/model_base/physionet2026_large_5min_agg"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_LARGE)
    df = df.reset_index(drop=True)

    # IMPORTANTE: los splits deben construirse sobre el mismo universo de pacientes que
    # load_data() acabará devolviendo (todas las implementaciones descartan en silencio
    # a quien no tenga HDF5 de model_base). Si no filtramos aquí, los índices posicionales
    # de kfold5_large.json/loso_large.json quedarían desalineados frente a los arrays
    # (X, y) reales en cuanto un experimento cargue datos (3/996 pacientes sin embeddings
    # — mismos 3 para 5s y 5min, verificado). run_queue.py SIEMPRE debe pasar este mismo
    # demographics.csv filtrado (o su equivalente) a cada model_module.load_data.
    has_emb = df.apply(lambda r: os.path.exists(
        os.path.join(EMB_DIR_5MIN, f"{r['BidsFolder']}_ses-{r['SessionID']}.hdf5")), axis=1)
    n_dropped = int((~has_emb).sum())
    df = df[has_emb].reset_index(drop=True)
    if n_dropped:
        print(f"[filtro] {n_dropped} pacientes sin embeddings model_base excluidos de los splits.")

    print(f"dataset_large_balanced (usable): {len(df)} pacientes")
    print("SiteID counts:", df["SiteID"].value_counts().to_dict())

    print("\n-- StratifiedKFold 5-fold (seed=42) --")
    folds = build_kfold(df)
    with open(os.path.join(OUT_DIR, "kfold5_large.json"), "w") as f:
        json.dump({"seed": SEED, "n_folds": N_FOLDS, "folds": folds}, f, indent=2)

    print("\n-- Leave-One-Site-Out --")
    loso = build_loso(df)
    with open(os.path.join(OUT_DIR, "loso_large.json"), "w") as f:
        json.dump({"splits": loso}, f, indent=2)

    # dataset_small: usado solo como chequeo de realismo (prevalencia natural) sobre
    # modelos ya entrenados en large. Aquí solo dejamos constancia de qué pacientes de
    # small NO se solapan con large (evitar fuga: 163/1103 pacientes de small están
    # también en large_balanced — deben excluirse del eval de "realismo").
    df_small = pd.read_csv(CSV_SMALL)
    key_small = df_small.apply(_key, axis=1)
    key_large = set(df.apply(_key, axis=1))
    overlap = key_small.isin(key_large)
    print(f"\ndataset_small: {len(df_small)} pacientes | solapan con large: {overlap.sum()} "
          f"(se excluirán del chequeo de realismo)")
    non_overlap_keys = key_small[~overlap].tolist()
    with open(os.path.join(OUT_DIR, "small_eval_keys.json"), "w") as f:
        json.dump({
            "note": "Claves de dataset_small SIN solape con dataset_large_balanced. "
                    "Usar SOLO estas para el chequeo de realismo (prevalencia natural) "
                    "de configs entrenadas en large, para evitar fuga train/eval.",
            "n_total_small": len(df_small),
            "n_overlap_excluded": int(overlap.sum()),
            "keys": non_overlap_keys,
        }, f, indent=2)

    print(f"\nSplits guardados en {OUT_DIR}/: kfold5_large.json, loso_large.json, small_eval_keys.json")


if __name__ == "__main__":
    main()
