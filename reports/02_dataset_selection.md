# 02 · 데이터셋 선정 (Dataset Selection)

> 이 문서는 Phase 0+1 에서 사용하는 데이터셋의 **선정 근거**, **전처리
> 절차**, **라이센스/인용**, **재현 절차** 를 정의한다.

## 1. 선정 원칙

다음 다섯 가지 기준을 모두 만족하는 데이터셋만 채택한다:

1. **공개 가능** — raw 데이터가 공개되어 있어야 reader 가 재현 가능.
2. **추천 도메인 표준** — 학계/산업 컨퍼런스 (RecSys, KDD, WWW) 에서
   비교 기준으로 자주 쓰여야 결과가 외부와 정렬됨.
3. **Two-tower 적합** — user × item × timestamp interaction 구조가
   명확해야 함.
4. **스케일 다양성** — sanity check 용 작은 셋과 의미 있는 medium 셋이
   모두 있어야 함.
5. **라이센스 명확** — 연구용 사용 권리가 문서화되어 있어야 함.

## 2. 채택 데이터셋

### 2.1 MovieLens 25M

- **출처**: GroupLens Research, University of Minnesota.
- **URL**: https://grouplens.org/datasets/movielens/25m/
- **규모**:
  - Interaction: 25,000,095 ratings.
  - User: 162,541.
  - Item: 62,423 (movie title).
  - Period: 1995-01-09 ~ 2019-11-21.
- **Rating 분포**: 0.5 ~ 5.0 (0.5 단위, 10단계).
- **라이센스**: GroupLens 연구용 라이센스. 재배포 금지, 학술 사용 권장,
  citation 의무.
- **인용 (BibTeX)**:
  ```bibtex
  @article{harper2015movielens,
    title={The MovieLens Datasets: History and Context},
    author={Harper, F. Maxwell and Konstan, Joseph A.},
    journal={ACM Transactions on Interactive Intelligent Systems},
    volume={5}, number={4}, pages={1--19}, year={2015},
    publisher={ACM}, doi={10.1145/2827872}
  }
  ```

**왜 채택했는가:**
- 추천 시스템 분야의 de facto 표준 sanity check.
- 작은 규모 (6만 item) → 빠른 iteration, brute-force exact top-K 계산
  가능 (모델 오차와 ANN 오차 분리에 필수).
- Microsoft `recommenders`, RecBole, OpenP5 등 다수 라이브러리가 동일
  데이터셋에 대한 Recall@10 reference 값을 공개 → sanity check 의 ground.

**한계:**
- Item 수 6만으로는 VDPU 의 강점 (대규모 corpus 가속) 이 잘 드러나지
  않음. 따라서 **methodology 검증용** 으로 한정한다.

### 2.2 Amazon Reviews 2023

- **출처**: Julian McAuley Lab, UC San Diego.
- **URL**: https://amazon-reviews-2023.github.io/
- **HuggingFace**: `McAuley-Lab/Amazon-Reviews-2023` (per-category parquet).
- **규모** (전체):
  - Review: 약 5.7억 (2023년 9월 기준).
  - User: 5,470만+.
  - Item: 4,800만+.
  - Period: 1996-05-21 ~ 2023-09-30.
- **본 벤치마크에서 사용하는 카테고리**:
  | 카테고리 | Review | User | Item | 본 벤치마크에서의 용도 |
  |---|---|---|---|---|
  | Beauty_and_Personal_Care | 약 23M | 약 11M | 약 110k | 가장 작은 medium 셋, 1차 실험 |
  | Books | 약 30M | 약 10M | 약 700k | 가장 큰 medium 셋, scaling 효과 |
  | Electronics | 약 43M | 약 18M | 약 1.6M | item 수 1M+ 의 본격적 부하 |
- **라이센스**: 연구용 라이센스. McAuley UCSD 의 attribution 의무.
- **인용 (BibTeX)**:
  ```bibtex
  @article{hou2024bridging,
    title={Bridging Language and Items for Retrieval and Recommendation},
    author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and
            Chen, Xiusi and McAuley, Julian},
    journal={arXiv preprint arXiv:2403.03952},
    year={2024}
  }
  ```

**왜 채택했는가:**
- E-commerce 도메인의 사실상 표준. 영업 narrative ("쇼핑몰") 와 도메인
  일치.
- 카테고리별로 scale 이 자연스럽게 다양 (10만 ~ 100만 item) → corpus
  size sweep 가능.
- 텍스트 metadata (제목, 설명, 리뷰) 가 함께 제공 → text encoder 를 가진
  realistic two-tower 학습 가능.

**한계:**
- 사용자 ID 가 카테고리별로 분리되어 cross-category recommendation 은
  본 벤치마크에서 다루지 않음.
- 평점 5점 척도라 binary 변환 정책 필요 (§3 참조).

## 3. 전처리 절차

모든 전처리는 `reco_bench/data/` 에 구현되며 `scripts/00_download_data.sh`
한 번에 실행 가능.

### 3.1 공통 canonical schema

전처리 후 모든 데이터셋은 두 개의 parquet 으로 정규화된다:

**`interactions.parquet`**:
```
user_id      int64    # 0-indexed dense remap
item_id      int64    # 0-indexed dense remap
ts           int64    # unix epoch seconds
rating       float32  # original rating (nullable)
```

**`item_meta.parquet`**:
```
item_id      int64
title        string   # nullable for ML-25M only-title-no-text
category     string   # ML: genres pipe-joined, Amazon: category path
text         string   # Amazon: title + brand + features; ML: title only
```

### 3.2 ID remapping

- 원본 user/item ID 는 보통 sparse 또는 hash 형태. 메모리 효율과 학습
  속도를 위해 [0, N) 의 dense int 로 remap.
- Remap 테이블은 `data/processed/<dataset>/id_map.json` 에 보관 (재현용).

### 3.3 Rating 의 implicit 변환

Two-tower retrieval 은 implicit feedback 이 표준이다. 다음 정책으로
binary positive 변환:

- **ML-25M**: rating ≥ 3.5 → positive. (그 이하는 dislike 으로 보고 학습
  에서 제외.)
- **Amazon-2023**: rating ≥ 4.0 → positive. (e-commerce 의 4점 이상이
  "구매 만족" 임을 가정.)
- 이 cutoff 는 Microsoft `recommenders` 와 동일하다.

### 3.4 User/item filtering (k-core)

Cold-start 문제 회피와 학습 안정성을 위해 5-core filtering:

```
반복:
  - interaction 수 < 5 인 user 제거
  - interaction 수 < 5 인 item 제거
수렴 시 종료
```

이는 RecBole 의 default 와 동일.

### 3.5 Split policy

`01_metric_design.md §2.2` 와 동일:

- **Temporal global split (기본)**: timestamp 정렬 후 90/5/5.
- **Leave-last-N (Microsoft recommenders 호환)**: 별도 split 으로 옵션
  제공.

산출: `train.parquet`, `val.parquet`, `test.parquet`.

### 3.6 Text feature 처리 (Amazon 만)

- `title + ' ' + (features 의 처음 256자)` 를 단일 문자열로 합쳐
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim) 로 임베딩.
- 본 텍스트 임베딩은 학습 중 frozen.
- ML-25M 은 title 만 가지므로 text feature 없이 ID-only two-tower 사용.

### 3.7 Ground-truth 캐시

- `data/processed/<dataset>/exact_topk.npy`:
  학습 종료 후 item embedding 에 대해 brute-force top-K 계산해 cache.
  ANN-isolation Recall 계산에 사용.

## 4. 데이터 통계 검증

각 데이터셋의 전처리 결과는 다음 sanity check 를 통과해야 함
(`tests/test_data.py`):

- [ ] User 수 = `id_map.user_id` 의 unique count.
- [ ] Item 수 = `id_map.item_id` 의 unique count.
- [ ] Split: train ∩ test = ∅ (user-time tuple 기준).
- [ ] Train interaction 의 timestamp max < val interaction 의 timestamp
      min (temporal split 의 경우).
- [ ] Implicit 변환 후 positive 비율이 reasonable 범위 (e.g. 20-80%).
- [ ] k-core 후 모든 user/item 의 interaction 수 ≥ 5.

## 5. 라이센스 준수 가이드

본 저장소는 **raw 데이터를 포함하지 않는다.** 사용자는 `scripts/00_download_data.sh`
실행 시 GroupLens / HuggingFace 에서 직접 다운로드하며, 그 시점에 각
라이센스에 동의한다.

전처리된 parquet 도 `.gitignore` 에 의해 제외되므로 fork/PR 시에도
유출되지 않는다.

**공개 시:**
- README 에 라이센스 명시 (이미 반영).
- `reports/02_dataset_selection.md` (본 문서) 를 통해 citation 의무 안내.
- 만약 결과 발표 (블로그/논문) 시 두 데이터셋 모두의 citation 명시.

## 6. Phase 2 확장 후보

다음 데이터셋은 방법론이 정착한 후 추가 검토 대상:

| 후보 | 규모 | 매력 | 위험 |
|---|---|---|---|
| KuaiRec | 약 600만 interaction, 약 1만 item | 산업급 dense matrix | 작은 item space |
| Taobao User Behavior | 약 1억 interaction, 약 400만 item | item space 큼, 산업급 | 라이센스 복잡 |
| Criteo Display 1TB | 수억 행 | DLRM 표준 | CTR 용이라 two-tower 부적합 |
| Spotify MPD | 100만 playlist × 200만 track | 큰 item space | 평가 정의가 다름 |

## 7. 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-05-20 | 초안 작성 | Phase 0 초기 설계 |
