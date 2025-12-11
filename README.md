# Detecção de Spoofing em Radar V2P (CARLA)

Este repositório contém os **scripts principais do pipeline** para detecção de spoofing em radar em cenários Veículo–para–Pedestre (V2P), usando o simulador **CARLA 0.9.15** (mapa Town03).

O cenário base é:

> Veículo em autopilot se aproximando de uma **faixa de pedestres** enquanto um pedestre atravessa, com um **sensor de radar** no veículo.  
> A partir desses dados reais, geramos versões **spoofadas** dos frames de radar, extraímos features tabulares e treinamos um modelo de Machine Learning para detectar **frames spoofados**.

---

## Estrutura (scripts incluídos neste repositório)

Os arquivos versionados aqui são apenas os scripts centrais do pipeline:

```text
SegcibV2P/
├─ .gitignore
└─ scripts/
   ├─ coleta_pedestres_faixa_0915.py
   ├─ spoofa_radar.py
   ├─ build_features_pedestre_spoof.py
   ├─ label_spoof_frames_diff.py
   ├─ train_spoof_detector_pedestre_refined.py
   └─ predict_spoof_pedestre_from_features.py
