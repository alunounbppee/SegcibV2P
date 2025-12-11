#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treina um modelo de ML para detecção de spoofing no radar,
usando rótulos refinados por frame (comparação clean x spoof).

Entrada:
  - data/carla/ml/spoof_pedestre_features.csv
  - data/carla/ml/spoof_frame_labels.csv

Saída:
  - data/carla/ml/model_spoof_rf_refined/model.joblib
  - data/carla/ml/model_spoof_rf_refined/meta.json
"""

import argparse
import json
import pathlib as P

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

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
        "--labels-csv",
        default=str(DATA_ROOT / "ml" / "spoof_frame_labels.csv"),
        help="CSV com is_spoof_frame por run/frame "
             "(default: data/carla/ml/spoof_frame_labels.csv).",
    )
    ap.add_argument(
        "--out-dir",
        default=str(DATA_ROOT / "ml" / "model_spoof_rf_refined"),
        help="Dir de saída do modelo (default: data/carla/ml/model_spoof_rf_refined).",
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    out_dir = P.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------
    # 1) Carrega features e labels refinados
    # -----------------------------------------
    df_feat = pd.read_csv(args.features_csv)
    df_lab = pd.read_csv(args.labels_csv)

    # Garante tipos
    if "frame" in df_lab.columns:
        df_lab["frame"] = df_lab["frame"].astype(int)
    if "frame" in df_feat.columns:
        df_feat["frame"] = df_feat["frame"].astype(int)

    # Mapa (run_spoof, frame) -> is_spoof_frame
    lab_map = {}
    for row in df_lab.itertuples(index=False):
        run_spoof = row.run_spoof
        frame = int(row.frame)
        is_spoof = int(row.is_spoof_frame)
        lab_map[(run_spoof, frame)] = is_spoof

    # Constrói novo vetor de labels por frame em df_feat
    refined_labels = []
    for row in df_feat.itertuples(index=False):
        run_id = row.run_id
        frame = int(row.frame)

        key = (run_id, frame)
        if key in lab_map:
            y = lab_map[key]      # 0 ou 1 conforme comparação clean x spoof
        else:
            y = 0                 # qualquer coisa fora de dataset_spoofed é real
        refined_labels.append(y)

    df_feat["label_refined"] = refined_labels

    # Stats de labels
    labels_arr = np.array(refined_labels, dtype=int)
    n_total = len(labels_arr)
    n_spoof = int((labels_arr == 1).sum())
    print("Total frames      :", n_total)
    print("Frames spoof (1)  :", n_spoof)
    print("Frames real  (0)  :", n_total - n_spoof)
    if n_total > 0:
        print("Proporção spoof  : {:.4f}".format(n_spoof / float(n_total)))

    # -----------------------------------------
    # 2) Monta X, y
    # -----------------------------------------
    ignore_cols = {"run_id", "frame", "label", "label_refined", "distance"}
    feature_cols = [c for c in df_feat.columns if c not in ignore_cols]

    X = df_feat[feature_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X)  # garante ausência de NaN/inf
    y = labels_arr

    # Separação treino/teste estratificada (spoof é raro)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if n_spoof > 0 else None,
    )

    # -----------------------------------------
    # 3) Pipeline: scaler + RandomForest
    # -----------------------------------------
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=1,
                class_weight="balanced",  # spoof é minoria
                random_state=args.random_state,
            )),
        ]
    )

    pipe.fit(X_train, y_train)

    # -----------------------------------------
    # 4) Avaliação
    # -----------------------------------------
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= 0.5).astype(int)

    print("\n=== Confusion matrix (thr=0.5) ===")
    print(confusion_matrix(y_test, y_pred_default))
    print("\n=== Classification report (thr=0.5) ===")
    print(classification_report(y_test, y_pred_default, digits=3))

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC-AUC: {roc_auc:.3f}")
    except ValueError:
        roc_auc = float("nan")
        print("ROC-AUC não pôde ser calculado.")

    # Threshold ótimo (Youden)
    try:
        fpr, tpr, thr = roc_curve(y_test, y_proba)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thr = float(thr[best_idx])
        print(f"\nThreshold sugerido (Youden): {best_thr:.3f}")
    except Exception:
        best_thr = 0.5
        print("\nNão foi possível calcular threshold ótimo, usando 0.5.")

    # -----------------------------------------
    # 5) Salva modelo + meta
    # -----------------------------------------
    model_path = out_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    meta = {
        "feature_names": feature_cols,
        "threshold": best_thr,
        "roc_auc": roc_auc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "label_source": "frame_diff_clean_spoof",
    }
    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✔ Modelo salvo em: {model_path}")
    print(f"✔ Metadados salvos em: {meta_path}")


if __name__ == "__main__":
    main()
