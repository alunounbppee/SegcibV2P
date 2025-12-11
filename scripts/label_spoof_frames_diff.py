#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compara dataset_clean x dataset_spoofed frame a frame e marca quais frames
foram realmente alterados (spoofados).

Saída:
  data/carla/ml/spoof_frame_labels.csv

Colunas:
  run_clean       - nome do run original (sem spoof)
  run_spoof       - nome do run spoofado
  frame           - número do frame (a partir do nome radar_XXXXXX.csv)
  is_spoof_frame  - 1 se houve diferença significativa, 0 se igual
  max_diff        - maior diferença absoluta entre os valores numéricos
  distance        - distância (se disponível em events/crosswalk.csv do spoof)
"""

import argparse
import pathlib as P

import numpy as np
import pandas as pd

PROJECT_ROOT = P.Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "carla"


def find_matching_clean_run(spoof_name, clean_runs_dict):
    """
    Tenta achar o run "clean" correspondente a um run spoof.

    Estratégia:
      - Se tiver "_spoof" no nome, pega tudo antes disso e tenta achar no clean.
      - Se não achar, tenta usar o nome inteiro.
    """
    if "_spoof" in spoof_name:
        base = spoof_name.split("_spoof", 1)[0]
        if base in clean_runs_dict:
            return base
    if spoof_name in clean_runs_dict:
        return spoof_name
    return None


def compare_frame(clean_csv, spoof_csv, tol=1e-3):
    """
    Compara dois arquivos radar_XXXXXX.csv.

    Retorna:
      (is_spoof_frame, max_diff)

    is_spoof_frame = True se max_diff > tol OU diferenças de tamanho/colunas.
    """
    df_c = pd.read_csv(clean_csv)
    df_s = pd.read_csv(spoof_csv)

    # Colunas em comum
    common_cols = [c for c in df_c.columns if c in df_s.columns]

    # Se não tiver colunas em comum, consideramos spoof
    if not common_cols:
        return True, float("inf")

    # Se quantidade de linhas diferente, consideramos spoof
    if len(df_c) != len(df_s):
        # ainda dá pra calcular alguma coisa, mas a diferença já é grande
        try:
            n = min(len(df_c), len(df_s))
            arr_c = df_c[common_cols].to_numpy()[:n]
            arr_s = df_s[common_cols].to_numpy()[:n]
            diff = np.abs(arr_c - arr_s)
            max_diff = float(diff.max())
        except Exception:
            max_diff = float("inf")
        return True, max_diff

    arr_c = df_c[common_cols].to_numpy()
    arr_s = df_s[common_cols].to_numpy()

    diff = np.abs(arr_c - arr_s)
    max_diff = float(diff.max())

    is_spoof = max_diff > tol
    return is_spoof, max_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--clean-root",
        default=str(DATA_ROOT / "dataset_clean"),
        help="Root dos runs REAIS (default: data/carla/dataset_clean).",
    )
    ap.add_argument(
        "--spoof-root",
        default=str(DATA_ROOT / "dataset_spoofed"),
        help="Root dos runs SPOOFADOS (default: data/carla/dataset_spoofed).",
    )
    ap.add_argument(
        "--out-csv",
        default=str(DATA_ROOT / "ml" / "spoof_frame_labels.csv"),
        help="CSV de saída (default: data/carla/ml/spoof_frame_labels.csv).",
    )
    ap.add_argument(
        "--tol",
        type=float,
        default=1e-3,
        help="Tolerância numérica para considerar diferenças como spoof (default: 1e-3).",
    )
    args = ap.parse_args()

    clean_root = P.Path(args.clean_root)
    spoof_root = P.Path(args.spoof_root)
    out_csv = P.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Mapeia runs clean pelo nome
    clean_runs = {}
    if clean_root.exists():
        for d in clean_root.iterdir():
            if d.is_dir():
                clean_runs[d.name] = d

    rows = []

    if not spoof_root.exists():
        print("⚠ spoof_root não existe:", spoof_root)
        return

    for spoof_dir in sorted(spoof_root.iterdir()):
        if not spoof_dir.is_dir():
            continue

        run_spoof = spoof_dir.name
        run_clean = find_matching_clean_run(run_spoof, clean_runs)

        if run_clean is None:
            print("⚠ Não encontrei run_clean correspondente para:", run_spoof)
            continue

        clean_dir = clean_runs[run_clean]

        radar_spoof_dir = spoof_dir / "radar"
        radar_clean_dir = clean_dir / "radar"

        if not radar_spoof_dir.exists() or not radar_clean_dir.exists():
            print("⚠ Faltando diretório radar em:", run_spoof, "ou", run_clean)
            continue

        # Carrega events só uma vez por run spoof
        ev_df = None
        events_path = spoof_dir / "events" / "crosswalk.csv"
        if events_path.exists():
            ev_df = pd.read_csv(events_path)

        print("▶ Comparando run spoof:", run_spoof, "vs clean:", run_clean)

        for spoof_csv in sorted(radar_spoof_dir.glob("radar_*.csv")):
            fname = spoof_csv.name
            try:
                frame = int(spoof_csv.stem.split("_")[-1])
            except ValueError:
                frame = -1

            clean_csv = radar_clean_dir / fname
            if not clean_csv.exists():
                # se não tiver o frame correspondente no clean, pula
                continue

            is_spoof, max_diff = compare_frame(clean_csv, spoof_csv, tol=args.tol)

            distance = np.nan
            if ev_df is not None and "frame" in ev_df.columns:
                match = ev_df[ev_df["frame"] == frame]
                if not match.empty and "distance" in match.columns:
                    distance = float(match["distance"].iloc[0])

            rows.append(
                {
                    "run_clean": run_clean,
                    "run_spoof": run_spoof,
                    "frame": frame,
                    "is_spoof_frame": int(is_spoof),
                    "max_diff": max_diff,
                    "distance": distance,
                }
            )

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)

    n_total = len(df_out)
    n_spoof = int((df_out["is_spoof_frame"] == 1).sum())
    print("✔ Labels por frame salvos em:", out_csv)
    print("  Frames comparados :", n_total)
    print("  Frames spoofados  :", n_spoof)
    if n_total > 0:
        print("  Proporção spoof   : {:.3f}".format(n_spoof / float(n_total)))


if __name__ == "__main__":
    main()
