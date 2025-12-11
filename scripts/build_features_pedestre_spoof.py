#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera dataset de features para detecção de spoofing no radar.

- Lê runs REAIS de data/carla/dataset_clean
- Lê runs SPOOFADOS de data/carla/dataset_spoofed
- Para cada frame (radar_XXXXXX.csv):
    - Extrai estatísticas de depth, velocity, azimuth, altitude
    - Faz join (opcional) com events/crosswalk.csv para pegar a distância
    - Cria um dataset tabular: 1 linha por frame

Saída (default):
  data/carla/ml/spoof_pedestre_features.csv
"""

import argparse
import pathlib as P

import numpy as np
import pandas as pd

# Raiz do projeto: .../SegCib_VeiculosAutonomos
PROJECT_ROOT = P.Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "carla"


def extract_features_from_radar(radar_path):
    """
    Lê um CSV de radar (azimuth, altitude, depth, velocity)
    e extrai features numéricas simples.
    """
    df = pd.read_csv(radar_path)

    # Ajuste aqui se os nomes de coluna forem diferentes
    az = df["azimuth"].to_numpy() if "azimuth" in df.columns else df.iloc[:, 0].to_numpy()
    alt = df["altitude"].to_numpy() if "altitude" in df.columns else df.iloc[:, 1].to_numpy()
    depth = df["depth"].to_numpy() if "depth" in df.columns else df.iloc[:, 2].to_numpy()
    vel = df["velocity"].to_numpy() if "velocity" in df.columns else df.iloc[:, 3].to_numpy()

    feats = {}
    n_points = len(df)
    feats["n_points"] = int(n_points)

    def _stats(prefix, arr):
        if len(arr) == 0:
            feats[f"{prefix}_mean"] = 0.0
            feats[f"{prefix}_std"] = 0.0
            feats[f"{prefix}_min"] = 0.0
            feats[f"{prefix}_max"] = 0.0
            feats[f"{prefix}_p25"] = 0.0
            feats[f"{prefix}_p50"] = 0.0
            feats[f"{prefix}_p75"] = 0.0
        else:
            feats[f"{prefix}_mean"] = float(arr.mean())
            feats[f"{prefix}_std"] = float(arr.std())
            feats[f"{prefix}_min"] = float(arr.min())
            feats[f"{prefix}_max"] = float(arr.max())
            feats[f"{prefix}_p25"] = float(np.percentile(arr, 25))
            feats[f"{prefix}_p50"] = float(np.percentile(arr, 50))
            feats[f"{prefix}_p75"] = float(np.percentile(arr, 75))

    _stats("depth", depth)
    _stats("vel", vel)

    # "Abertura" angular do feixe naquele frame
    for prefix, arr in (("az", az), ("alt", alt)):
        if len(arr) == 0:
            feats[f"{prefix}_spread"] = 0.0
        else:
            feats[f"{prefix}_spread"] = float(arr.max() - arr.min())

    return feats


def collect_runs(root, label, max_distance=None):
    """
    Varre todos os runs em root, lendo radar_*.csv e crosswalk.csv
    e adicionando a coluna 'label' (0 = real, 1 = spoof).
    """
    rows = []

    if not root.exists():
        print("⚠ Root não existe:", root)
        return rows

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_id = run_dir.name
        radar_dir = run_dir / "radar"
        events_path = run_dir / "events" / "crosswalk.csv"

        if not radar_dir.exists():
            continue

        ev_df = None
        if events_path.exists():
            ev_df = pd.read_csv(events_path)

        for radar_path in sorted(radar_dir.glob("radar_*.csv")):
            # frame a partir do nome do arquivo: radar_000123 -> 123
            try:
                frame = int(radar_path.stem.split("_")[-1])
            except ValueError:
                frame = -1  # fallback

            feats = extract_features_from_radar(radar_path)

            distance = np.nan
            if ev_df is not None and "frame" in ev_df.columns:
                ev_match = ev_df[ev_df["frame"] == frame]
                if not ev_match.empty and "distance" in ev_match.columns:
                    distance = float(ev_match["distance"].iloc[0])

            # se quiser focar só em interação pedestre-carro até certo range
            if (max_distance is not None) and (not np.isnan(distance)):
                if distance > max_distance:
                    continue

            feats.update(
                {
                    "run_id": run_id,
                    "frame": frame,
                    "distance": distance,
                    "label": int(label),
                }
            )
            rows.append(feats)

    return rows


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
        default=str(DATA_ROOT / "ml" / "spoof_pedestre_features.csv"),
        help="CSV de saída (default: data/carla/ml/spoof_pedestre_features.csv).",
    )
    ap.add_argument(
        "--max-distance",
        type=float,
        default=60.0,
        help="Máxima distância pedestre-carro (m). Use -1 para desativar.",
    )
    args = ap.parse_args()

    clean_root = P.Path(args.clean_root)
    spoof_root = P.Path(args.spoof_root)
    out_csv = P.Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # None = não aplica filtro por distância
    if args.max_distance is not None and args.max_distance < 0:
        max_dist = None
    else:
        max_dist = args.max_distance

    rows_real = collect_runs(clean_root, label=0, max_distance=max_dist)
    rows_spoof = collect_runs(spoof_root, label=1, max_distance=max_dist)

    df = pd.DataFrame(rows_real + rows_spoof)
    df.to_csv(out_csv, index=False)

    print(f"✔ Dataset salvo em: {out_csv}")
    print(f"  Frames reais : {len(rows_real)}")
    print(f"  Frames spoof : {len(rows_spoof)}")
    print(f"  Total        : {len(df)}")


if __name__ == "__main__":
    main()
