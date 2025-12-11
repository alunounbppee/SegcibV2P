#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usa o modelo treinado (RandomForest refinado) para identificar spoofing
diretamente no CSV de features: spoof_pedestre_features.csv.

Entrada:
  - features: data/carla/ml/spoof_pedestre_features.csv
  - modelo:   data/carla/ml/model_spoof_rf_refined/model.joblib + meta.json

Saída:
  - data/carla/ml/spoof_pedestre_predictions.csv

Colunas adicionadas:
  - proba_spoof   : probabilidade de spoof segundo o modelo
  - is_spoof_pred : 0/1 usando o threshold salvo em meta.json
"""

import argparse
import json
import pathlib as P

import numpy as np
import pandas as pd
import joblib

# Raiz do projeto: .../SegCib_VeiculosAutonomos
PROJECT_ROOT = P.Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "carla"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--features-csv",
        default=str(DATA_ROOT / "ml" / "spoof_pedestre_features.csv"),
        help="CSV de features (default: data/carla/ml/spoof_pedestre_features.csv).",
    )
    ap.add_argument(
        "--model-dir",
        default=str(DATA_ROOT / "ml" / "model_spoof_rf_refined"),
        help="Diretório do modelo treinado (default: data/carla/ml/model_spoof_rf_refined).",
    )
    ap.add_argument(
        "--out-csv",
        default=str(DATA_ROOT / "ml" / "spoof_pedestre_predictions.csv"),
        help="CSV de saída com predições (default: data/carla/ml/spoof_pedestre_predictions.csv).",
    )
    args = ap.parse_args()

    features_path = P.Path(args.features_csv)
    model_dir = P.Path(args.model_dir)
    out_csv = P.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 1) Carrega features
    # -------------------------------------------------
    df = pd.read_csv(features_path)

    # -------------------------------------------------
    # 2) Carrega modelo e metadados
    # -------------------------------------------------
    model_path = model_dir / "model.joblib"
    meta_path = model_dir / "meta.json"

    if not model_path.exists():
        raise FileNotFoundError("Modelo não encontrado em: {}".format(model_path))
    if not meta_path.exists():
        raise FileNotFoundError("Meta.json não encontrado em: {}".format(meta_path))

    model = joblib.load(model_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    feature_names = meta.get("feature_names")
    if feature_names is None:
        raise ValueError("Meta.json não contém 'feature_names'.")

    thr = float(meta.get("threshold", 0.5))

    # -------------------------------------------------
    # 3) Monta matriz X com as mesmas features do treino
    # -------------------------------------------------
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError("As seguintes features do modelo não estão no CSV: {}".format(missing))

    X = df[feature_names].to_numpy(dtype=float)
    X = np.nan_to_num(X)  # segurança contra NaN/inf

    # -------------------------------------------------
    # 4) Predição: probabilidade e classe 0/1
    # -------------------------------------------------
    proba_spoof = model.predict_proba(X)[:, 1]
    is_spoof_pred = (proba_spoof >= thr).astype(int)

    df["proba_spoof"] = proba_spoof
    df["is_spoof_pred"] = is_spoof_pred

    # -------------------------------------------------
    # 5) Salva resultado
    # -------------------------------------------------
    df.to_csv(out_csv, index=False)

    n_total = len(df)
    n_pred_spoof = int((df["is_spoof_pred"] == 1).sum())
    print("✔ Predições salvas em:", out_csv)
    print("  Frames totais      :", n_total)
    print("  Frames spoof (pred):", n_pred_spoof)
    if n_total > 0:
        print("  Proporção spoof    : {:.4f}".format(n_pred_spoof / float(n_total)))
    print("  Threshold usado    : {:.3f}".format(thr))


if __name__ == "__main__":
    main()
