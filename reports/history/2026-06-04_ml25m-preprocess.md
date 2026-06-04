---
date: 2026-06-04
phase: 1
topic: ml25m-preprocess
status: completed
---

# Step 2 부분 완료: ML-25M 다운로드 + 전처리

## What changed

- 데이터 처리 모듈 작성:
  - `reco_bench/data/base.py` — `RecoDataset` 추상 + canonical schema.
  - `reco_bench/data/preprocessing.py` — implicit filter, k-core,
    ID remap, temporal split, leak validation, per-user positives.
  - `reco_bench/data/ml25m.py` — MovieLens-25M zip 다운로드 + raw load.
  - `reco_bench/data/pipeline.py` — 전체 파이프라인 entry
    (`python -m reco_bench.data.pipeline <config.yaml>`).
- `scripts/00_download_data.sh` 가 위 entry 를 호출하는 실제 스크립트로
  채워짐. (이전엔 placeholder 였음.)
- ML-25M 실 다운로드 + 전처리 완료. 결과:
  | 항목 | 값 |
  |---|---|
  | Raw interaction | 25,000,095 |
  | Implicit (rating ≥ 3.5) 후 | 15,630,129 |
  | K-core (k=5) 후 | 15,584,360 |
  | Users (final) | 161,572 |
  | Items (final) | 23,965 |
  | Train / Val / Test | 14,025,924 / 779,218 / 779,218 |
  | 소요 시간 | 약 58초 (다운로드 + 전처리) |
  | Raw 디스크 | 1.1 GB |
  | Processed 디스크 | 234 MB |

- `configs/datasets/ml25m.yaml` 의 `stats_expected` 를 raw 기준에서 전
  처리-후 floor 로 조정 (`num_items_min: 50000 → 20000`,
  `num_interactions_min: 20M → 10M`).

## Why

- **MovieLens-25M sha256 mismatch**: YAML 의 placeholder 해시와 실제 zip
  의 sha256 가 다른 것을 발견. mirror 가 시간에 따라 zip 내부 구조를
  재패키징할 수 있고, 해시 lock 은 검증 의무가 강해 build 가 자주
  깨질 수 있다. 현재는 mismatch 시 경고만 출력하고 계속 진행. raw
  파일 (`ratings.csv`, `movies.csv`) 의 row count 가 reproducibility 의
  진짜 floor 이다 (이건 1996~2019 의 ratings 가 fix 이므로 변하지 않음).
- **k-core 가 거의 줄이지 않음**: implicit filter 후 15.63M → k-core
  후 15.58M (-0.3%). ML-25M 사용자들이 이미 활발한 reviewer 이기 때문.
  Amazon Reviews 에서는 더 많이 줄어들 것으로 예상.
- **ratio normalize**: `temporal_split` 가 비율을 받을 때 합이 1.0 이
  아니어도 자동 정규화하도록 구현 — YAML 에서 `[0.9, 0.05, 0.05]` 의
  반올림 오차로 비율 합이 약간 빗나가도 안전.

## Validation

- `validate_split` 의 시간 누설 체크 통과: train_max < val_min,
  val_max < test_min.
- per-user train positives `train_positives.npz` 가 정상 생성 (user-tower
  의 recent-N input 으로 사용).
- 11,000 사용자 (test) 중 ground_truth 빌드 동작 확인 (smoke 단계).
- 합성 데이터 (200 rows) 로 모든 preprocessing helper 함수의 unit
  smoke 통과.

## Open questions / next

- **Amazon Reviews 2023**: huggingface_hub 의 `McAuley-Lab/Amazon-Reviews-2023`
  에 대한 `reco_bench/data/amazon2023.py` 구현은 Step 7 (Two-tower 학습
  이 안정화된 후) 로 미룸. ML-25M 으로 먼저 등뼈 검증.
- **GroupLens sha256 mismatch**: 진짜 sha를 한 번 측정한 뒤 YAML 에
  pinned 으로 박을지, 아니면 row count 만 정합성 기준으로 쓸지 결정
  필요. 현재는 후자.
- 다음 = Step 3 (Two-tower 모델 + 학습 파이프라인).
