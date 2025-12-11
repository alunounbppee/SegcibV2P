#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera spoofing de radar para CENÁRIOS DE PEDESTRE NA FAIXA.

Estrutura esperada (a mesma do script de pedestre na faixa):

data/carla/dataset_clean/<run_id>/
  radar/
    radar_XXXXXX.csv
  events/
    crosswalk.csv

Saída:

data/carla/dataset_spoofed/<run_id>/
  radar/
    radar_XXXXXX.csv            (com ou sem ponto fantasma)
  events/                       (cópia da pasta events do clean)
  spoof_manifest_spoof.csv      (frame_file, spoofed)

Se rodar SEM argumentos (botão Run do PyCharm):
  - assume que este arquivo está em <projeto>/scripts
  - usa <projeto>/data/carla como raiz.

Também é possível passar:
  --root <projeto>/data/carla
  --root <projeto>/data/carla/dataset_clean
  --root <projeto>/data/carla/dataset_clean/AAAAmmdd_HHMMSS
"""

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Optional, Set


# ---------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------
def parse_args():
    script_dir = Path(__file__).resolve().parent
    # Raiz padrão: <projeto>/data/carla
    default_root = script_dir.parent / "data" / "carla"

    ap = argparse.ArgumentParser(
        description="Gera spoofing de radar em cenários de pedestre na faixa."
    )
    ap.add_argument(
        "--root",
        default=str(default_root),
        help=(
            "pasta raiz (ex: data/carla) OU dataset_clean "
            "OU um run específico em dataset_clean/AAAAmmdd_HHMMSS. "
            f"Default: {default_root}"
        ),
    )
    ap.add_argument(
        "--rate",
        type=float,
        default=0.1,
        help="proporção de frames a adulterar entre os candidatos (0-1)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="semente global para o sorteio dos frames",
    )
    ap.add_argument(
        "--depth",
        type=float,
        default=30.0,
        help="profundidade (m) do ponto fantasma",
    )
    ap.add_argument(
        "--az",
        type=float,
        default=0.03,
        help="azimute do ponto fantasma (rad)",
    )
    ap.add_argument(
        "--alt",
        type=float,
        default=-0.05,
        help="altitude do ponto fantasma (rad)",
    )
    ap.add_argument(
        "--vel",
        type=float,
        default=0.4,
        help="velocidade radial do ponto fantasma (m/s)",
    )
    ap.add_argument(
        "--only-crosswalk",
        action="store_true",
        help="se ativo, só faz spoof em frames com evento em events/crosswalk.csv",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def frame_from_name(p: Path) -> Optional[int]:
    """Extrai número do frame de 'radar_000123.csv' -> 123."""
    try:
        return int(p.stem.split("_")[-1])
    except ValueError:
        return None


def load_crosswalk_frames(events_dir: Path) -> Set[int]:
    """Carrega conjunto de frames a partir de events/crosswalk.csv (se existir)."""
    cw_file = events_dir / "crosswalk.csv"
    frames: Set[int] = set()
    if not cw_file.exists():
        return frames

    with cw_file.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                frames.add(int(row["frame"]))
            except (KeyError, ValueError):
                continue
    return frames


def discover_structure(root: Path):
    """
    Descobre carla_root, dataset_clean_dir e lista de runs a partir do --root.

    Casos suportados:
      - root = <projeto>/data/carla                (contém dataset_clean)
      - root = <projeto>/data/carla/dataset_clean
      - root = <projeto>/data/carla/dataset_clean/<run_id>

    Retorna:
        carla_root: Path          (ex: <projeto>/data/carla)
        dataset_clean_dir: Path   (ex: <projeto>/data/carla/dataset_clean)
        run_dirs: list[Path]      (cada um é um run_id)
    """
    root = root.resolve()

    # Caso 1: root é "data/carla" (tem dataset_clean dentro)
    if (root / "dataset_clean").exists():
        carla_root = root
        dataset_clean_dir = root / "dataset_clean"
        run_dirs = [d for d in sorted(dataset_clean_dir.iterdir()) if d.is_dir()]
        return carla_root, dataset_clean_dir, run_dirs

    # Caso 2: root é "dataset_clean"
    if root.name == "dataset_clean":
        carla_root = root.parent
        dataset_clean_dir = root
        run_dirs = [d for d in sorted(dataset_clean_dir.iterdir()) if d.is_dir()]
        return carla_root, dataset_clean_dir, run_dirs

    # Caso 3: root é um run específico dentro de dataset_clean
    if root.parent.name == "dataset_clean":
        dataset_clean_dir = root.parent
        carla_root = dataset_clean_dir.parent
        run_dirs = [root]
        return carla_root, dataset_clean_dir, run_dirs

    raise SystemExit(
        f"[ERRO] Estrutura inesperada para --root={root}. "
        "Esperado algo como 'data/carla', 'data/carla/dataset_clean' "
        "ou 'data/carla/dataset_clean/AAAAmmdd_HHMMSS'."
    )


# ---------------------------------------------------------------------
# Lógica principal por run
# ---------------------------------------------------------------------
def process_run(run_dir: Path, out_root: Path, args, seed_offset: int):
    """
    Gera spoof para um run específico:
      run_dir  = .../dataset_clean/AAAAmmdd_HHMMSS
      out_root = .../dataset_spoofed/AAAAmmdd_HHMMSS
    """
    rad_in_dir = run_dir / "radar"
    events_in = run_dir / "events"

    if not rad_in_dir.exists():
        print(f"[AVISO] Sem pasta radar/ em {run_dir}, pulando.")
        return

    # Precisa ter crosswalk para fazer sentido no estudo
    if not (events_in / "crosswalk.csv").exists():
        print(f"[AVISO] Sem crosswalk.csv em {run_dir}, pulando run.")
        return

    # Cria estrutura de saída
    rad_out_dir = out_root / "radar"
    rad_out_dir.mkdir(parents=True, exist_ok=True)

    # Copia events/ (mantém info de pedestre na faixa)
    if events_in.exists():
        events_out = out_root / "events"
        # Python desta env não suporta dirs_exist_ok em copytree
        if events_out.exists():
            shutil.rmtree(events_out)
        shutil.copytree(events_in, events_out)

    all_rad_files = sorted(rad_in_dir.glob("radar_*.csv"))
    if not all_rad_files:
        print(f"[AVISO] Nenhum radar_*.csv em {rad_in_dir}, pulando.")
        return

    # Frames candidatos para spoof
    candidates = all_rad_files
    if args.only_crosswalk:
        cross_frames = load_crosswalk_frames(events_in)
        if cross_frames:
            candidates = [
                p for p in all_rad_files
                if frame_from_name(p) in cross_frames
            ]
        else:
            print(
                f"[AVISO] --only-crosswalk, mas crosswalk vazio em {run_dir}. "
                "Usando todos os frames como candidatos."
            )
            candidates = all_rad_files

    if not candidates:
        print(f"[AVISO] Nenhum frame candidato em {run_dir}, nada será adulterado.")
        spoof_frames = set()
    else:
        # sorteio reprodutível por run
        random.seed(args.seed + seed_offset)
        num_spoof = int(len(candidates) * args.rate)
        if num_spoof > 0:
            num_spoof = min(num_spoof, len(candidates))
            spoof_frames = set(random.sample(candidates, num_spoof))
        else:
            spoof_frames = set()

    manifest = out_root / "spoof_manifest_spoof.csv"
    with manifest.open("w", newline="") as mf:
        mw = csv.writer(mf)
        mw.writerow(["frame_file", "spoofed"])

        for p in all_rad_files:
            dst = rad_out_dir / p.name

            # lê radar original
            with p.open() as f:
                r = csv.reader(f)
                _header = next(r, None)  # descarta cabeçalho original
                rows = [row for row in r]

            # injeta ponto fantasma, se for frame escolhido
            if p in spoof_frames:
                rows.append(
                    [
                        f"{args.az:.6f}",
                        f"{args.alt:.6f}",
                        f"{args.depth:.3f}",
                        f"{args.vel:.3f}",
                    ]
                )
                spoofed = 1
            else:
                spoofed = 0

            # grava CSV de saída
            with dst.open("w", newline="") as g:
                w = csv.writer(g)
                w.writerow(["azimuth", "altitude", "depth", "velocity"])
                w.writerows(rows)

            mw.writerow([p.name, spoofed])

    print(
        "  -> OK run {name}: {spoof} frames spoofados de {cand} candidatos.".format(
            name=run_dir.name,
            spoof=len(spoof_frames),
            cand=len(candidates),
        )
    )


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    args = parse_args()

    root = Path(args.root)
    carla_root, dataset_clean_dir, run_dirs = discover_structure(root)

    dataset_spoofed_dir = carla_root / "dataset_spoofed"
    dataset_spoofed_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] root informado    : {root}")
    print(f"[INFO] carla_root        : {carla_root}")
    print(f"[INFO] dataset_clean_dir : {dataset_clean_dir}")
    print(f"[INFO] runs encontrados  : {len(run_dirs)}")

    for idx, run_dir in enumerate(sorted(run_dirs)):
        out_root = dataset_spoofed_dir / run_dir.name
        print(f"\n[RUN] {run_dir.name}")
        process_run(run_dir, out_root, args, seed_offset=idx)

    print("\n[FINALIZADO] Dataset spoof gerado em:", dataset_spoofed_dir)


if __name__ == "__main__":
    main()
