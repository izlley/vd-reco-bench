# 04 · VDPU 가치 명제 (Value Proposition) — 영업 narrative 의 수식화

> 이 문서는 디노티시아 VDPU 의 "CPU 대비 10×, GPU 대비 6~7× 가성비"
> 라는 공개 주장을 **iso-recall 비교 프레임 안에서 수치로 검증 가능한
> 형태로 다시 쓴 것** 이다. Phase 0 에서는 schema 와 수식만 정의하고,
> 실측 칼럼은 Phase 2 에서 채운다.
>
> 본 문서는 **공개 배포** 를 전제로 작성되므로 비공개 가격 협상,
> 미공개 고객사, 미공개 ASIC 사양은 포함하지 않는다. 공개 출처
> (hellot.net CDO 인터뷰 등) 만 인용한다.

---

## 1. 핵심 헤드라인 — 한 줄

> **추천 시스템의 ANN 검색 단계에서, Recall@10 = 0.95 라는 동일한 정확도
> 를 달성하는 데 드는 비용을 GPU(H100) baseline 과 VDPU 가 각각 얼마인지
> 비교하여, $/1M queries 의 비율을 보고한다.**

이 비율이 영업이 이야기하는 "**6~7×**" 의 수학적 형태이다.

---

## 2. 영업 narrative 를 수식으로 다시 쓰기

### 2.1 원래의 주장 (hellot.net, 2025-03-XX 인터뷰)

> "VDPU 는 CPU 대비 최대 10배, GPU 대비 6~7배의 가성비를 확보했다."

이 문장은 그 자체로는 검증 불가능하다. 다음을 명시해야 한다:

1. **무엇의** 가성비인가? — 처리량 단위 비용 ($/QPS) 인가, 단위 전력 당
   처리량 (QPS/W) 인가, 전체 TCO 인가?
2. **어떤 작업** 에서? — 임의의 벡터 dot product 인가, 추천 ANN 인가,
   RAG retrieval 인가, 벡터 DB 의 일부 연산인가?
3. **어떤 정확도** 에서? — recall 을 어디로 맞춰 놓고 비교하는가?
4. **어떤 baseline GPU** 인가? — A100, H100, H200, B100?

본 벤치마크는 이 네 가지를 다음과 같이 고정한다:

| 항목 | 본 벤치마크의 정의 |
|---|---|
| 비용 단위 | `$/1M queries` (cloud SKU 시간당 가격 기준) |
| 작업 | Two-tower 추천 검색의 ANN top-10 / top-100 |
| 정확도 | Recall@K vs exact $\geq$ 0.95 (primary), 0.90 (secondary) |
| Baseline GPU | H100 SXM5 80G + (FAISS-GPU IVF-PQ, cuVS CAGRA) 중 더 저렴한 쪽 |

### 2.2 핵심 수식 (재게재)

`01_metric_design.md §4.1` 의 정의:

$$
\text{IsoRecallCostRatio}_{R=r}
= \frac{\$/\text{QPS}\,|\, \text{baseline at Recall@K vs exact} = r}
       {\$/\text{QPS}\,|\, \text{VDPU at Recall@K vs exact} = r}
$$

목표:
- $r = 0.95$: **목표 6~7×**.
- $r = 0.90$: **참고 측정** (VDPU 가 낮은 recall 에서만 빠르다면 영업에
  치명적).

### 2.3 보조 수식

전력 효율:
$$
\text{IsoRecallEnergyRatio}_{R=r}
= \frac{(\bar{P}_{\text{net}}/\text{QPS})_{\text{baseline}}}
       {(\bar{P}_{\text{net}}/\text{QPS})_{\text{VDPU}}}
$$

데이터 센터 PUE 가 동일하다고 가정하면 이는 전력 비용 비율과 같다.

빌드 비용 (일회성):
$$
\text{IndexBuildRatio} = \frac{T_{\text{build, baseline}}}{T_{\text{build, VDPU}}}
$$

VDPU 가 build 마저 가속한다면 추가 영업 포인트.

---

## 3. TCO 모델 (3년 amortize)

영업 자료에서 자주 요구되는 "연간 절감액" 을 계산하기 위한 모델.

### 3.1 가정

| 변수 | 기본값 | 출처/근거 |
|---|---|---|
| Workload | 1B queries/day | 중대형 쇼핑몰 추천 호출 추정 (top 페이지 + 검색 + 상세 페이지) |
| 가동률 | 70% | peak/off-peak 평균 |
| 운영 기간 | 3년 | 일반 amortize 기간 |
| 전력 단가 | $0.10 /kWh | 한국 산업용 평균 (KEPCO 공개 자료) |
| PUE | 1.4 | tier-3 IDC 평균 |
| 인건/운영 OpEx | TCO 의 15% | 업계 통상 비율 (소거 가능) |

### 3.2 TCO 식

연간 비용 (한 가속기 종류 기준):

$$
\text{TCO}_{\text{annual}} = \frac{C_{\text{capex}}}{3}
+ \bar{P}_{\text{net}} \cdot \text{hours} \cdot \text{PUE} \cdot p_{\text{kWh}}
+ \alpha \cdot (\text{above})
$$

- $C_{\text{capex}}$: 하드웨어 구매가 (cloud 라면 시간당 가격을 amortize
  한 등가).
- $\bar{P}_{\text{net}}$: 평균 net 전력 (W) — `01_metric_design.md §3.4`.
- $\text{hours}$: 가동률 보정한 연간 가동 시간.
- $\alpha = 0.15$: 운영 OpEx 비율.

### 3.3 필요 가속기 수

$$
N_{\text{HW}} = \left\lceil \frac{\text{queries/day} / 86400}{\text{QPS}_{\text{single}} \cdot \text{utilization}} \right\rceil
$$

VDPU 가 6× 더 빠르면 $N_{\text{HW}}$ 도 6× 줄어 CapEx 가 6× 절감되고,
전력도 비례 절감.

### 3.4 절감액 = (baseline TCO – VDPU TCO)

연간 절감액 × 3년 = 영업 자료의 "3년간 X억 절감".

---

## 4. Sensitivity 분석 (Phase 2 산출)

영업 자료의 신뢰성은 **외부 변수에 따라 6~7× 가 어떻게 변하는가** 를
보여줄 때 비로소 생긴다. 다음 sweep 을 수행한다:

### 4.1 Corpus size 의 영향

| Item 수 | 예상 영향 |
|---|---|
| 60k (MovieLens) | VDPU 우위 작음 (GPU 도 빠르게 처리) |
| 1M (Amazon-Books) | VDPU 우위 명확 |
| 10M (synthetic, 추후 Phase 2) | VDPU 우위 극대화 |

### 4.2 Embedding dim 의 영향

dim ∈ {64, 128, 256, 512}.
VDPU 의 connectivity 와 SIMD 폭이 어느 dim 에서 최적인지 측정.

### 4.3 Recall target 의 영향

$r \in [0.80, 0.99]$ 의 sweep.
이론적으로 모든 ANN 가속기는 r → 1 에 가까워질수록 brute-force 에
수렴하므로 가속비가 떨어진다. 본 벤치마크는 그 떨어지는 지점이 어디인지
시각화한다.

### 4.4 Concurrency 의 영향

concurrency ∈ {1, 4, 16, 64}.
VDPU 의 batch sweet-spot 이 GPU 와 다르다면 sweep 그래프에 명확히 보여
야 한다.

### 4.5 데이터 분포의 영향

ML-25M 의 dense item embedding 과 Amazon-Reviews 의 long-tail item
distribution 은 ANN 가속기에 다른 부하를 준다. 두 데이터셋에서의
가속비 차이를 별도 보고.

---

## 5. 결과 보고 schema (Phase 2 채움)

### 5.1 헤드라인 표

```markdown
## Iso-recall cost ratio @ Recall@10 vs exact = 0.95

| Dataset       | Items | Baseline (GPU) | VDPU $/1M | Ratio |
|---------------|-------|----------------|-----------|-------|
| ML-25M        | 62k   | TBD            | TBD       | TBD ×  |
| Amazon-Beauty | 80k   | TBD            | TBD       | TBD ×  |
| Amazon-Books  | 800k  | TBD            | TBD       | TBD ×  |
| Amazon-Elec.  | 200k  | TBD            | TBD       | TBD ×  |
```

### 5.2 Sensitivity 그래프

- `reports/figures/cost_ratio_vs_corpus_size.png`
- `reports/figures/cost_ratio_vs_recall.png`
- `reports/figures/cost_ratio_vs_dim.png`
- `reports/figures/cost_ratio_vs_concurrency.png`

각 그래프의 x 축 위에 baseline 의 가격 스냅샷 날짜 명시.

### 5.3 TCO 시뮬레이션 표

```markdown
## 3년 TCO 비교 (1B queries/day, PUE 1.4, 전력 $0.10/kWh)

| 구성 | 필요 HW 수 | CapEx | 전력 OpEx (3yr) | 운영 OpEx (3yr) | Total |
|------|-----------|-------|----------------|-----------------|-------|
| H100 baseline | N        | $X    | $Y             | $Z              | $T₁   |
| VDPU          | N/6.5    | $X'   | $Y'            | $Z'             | $T₂   |
| 절감          |           |       |                |                  | $T₁-$T₂ |
```

---

## 6. Anti-narrative (반대 입장 미리 대응)

영업 narrative 가 강하려면 reviewer 의 의심을 먼저 대응해야 한다.

### 6.1 "VDPU 가 빠른 건 recall 이 낮아서 아닌가?"

→ 본 벤치마크는 **iso-recall** 비교. 같은 recall 에서 측정.

### 6.2 "Cherry-picked GPU baseline 아닌가?"

→ baseline 은 **현재 시점에서 가장 저렴한 GPU 구성** 이며 sweep 결과
   중 winner 를 사용. 즉 VDPU 에게 가장 불리한 baseline.

### 6.3 "데이터셋이 VDPU 에 유리하게 골라진 것 아닌가?"

→ MovieLens 와 Amazon Reviews 는 추천 시스템 학계의 표준이며 디노티시아
   가 제어하지 않는다. raw 데이터는 GroupLens, McAuley UCSD 가 배포.

### 6.4 "Cloud 가격이 변하면 의미 없는 것 아닌가?"

→ 가격 스냅샷 날짜를 명시하고, `configs/cost_model.yaml` 을 갱신하면
   누구나 재계산 가능. 결과의 절대값보다 비율이 영업 포인트.

### 6.5 "Power 측정이 noise 가 큰데?"

→ 측정 방법론 (샘플 레이트, 적분 윈도우, baseline idle) 을 두 가속기에
   동일하게 적용. 그 차이가 6× 이상이 되려면 측정 오차로 설명 불가능.

### 6.6 "VDPU 가 빠르지만 운영 도구 (Milvus/Qdrant/Pinecone) 와 통합 안
   되면 의미 없는 것 아닌가?"

→ 본 벤치마크는 가속기 단위 평가. 시스템 통합은 별도 작업으로 추적
   (Phase 2.5: Seahorse 와의 통합 데모).

---

## 7. 헤드라인 차트 (목업)

Phase 2 에서 자동 생성될 차트의 모양은 다음과 같다.

### 7.1 Recall-QPS Pareto (메인 차트)

```
       Recall@10 vs exact
            ▲
       1.0 ─┤
            │       ╱──────────── VDPU 목표 곡선 (예측)
            │      ╱
            │     ╱      ●●●●●● cuVS CAGRA H100
            │    ●●●●●●●●
       0.95 ╾────────────────────────  ← iso-recall 기준선 (0.95)
            │ ●●●          ●●●●●● FAISS-GPU IVF-PQ H100
            │
       0.90 ─┤●●          ●●●●●● FAISS-CPU HNSW EPYC
            │           ●●●●●● ScaNN CPU
            │
            └───────────────────────▶ QPS (log scale)
                  1k   10k   100k   1M
```

iso-recall 기준선 (0.95) 과 각 곡선이 만나는 점의 QPS 가 비교의 핵심.

### 7.2 비용 막대 그래프

```
   $/1M queries @ Recall@10=0.95
    ▲
    │ ████ FAISS-CPU HNSW        $0.075
    │ ███  ScaNN CPU             $0.063
    │ ██   FAISS-GPU IVF-PQ      $0.094
    │ █    cuVS CAGRA H100       $0.055  ← baseline 의 winner
    │ ▓    VDPU (target)         $0.008  ← 6.5× 저렴
    └─────────────────────────────────▶
```

(위 숫자는 schema 예시.)

---

## 8. 책임 분리

| 활동 | 본 벤치마크의 책임 | 영업/마케팅의 책임 |
|---|---|---|
| 측정 정확성 | O | – |
| 측정 방법론의 공개 | O | – |
| 측정 결과의 수치 | O | – |
| 결과를 영업 자료로 가공 | – | O |
| 미공개 가격으로 TCO 갱신 | – | O |
| 고객사별 customization | – | O |

본 벤치마크의 결과는 **공개 가능한 가격과 공개 데이터셋으로 재현 가능한
숫자** 만 다룬다. 그 외의 가공은 본 저장소 바깥에서.

---

## 9. Open questions

- VDPU FPGA 의 $C_{\text{hourly}}$ 를 산출할 방법론 정의 필요
  (양산 ASIC 가격이 미공개이므로, FPGA prototype 의 cloud-equivalent
  단가를 합리적으로 추정해야).
- 1B queries/day 가정의 출처를 더 단단히 (publicly verifiable e-commerce
  query volume 자료).
- "VDPU 가 GPU + reranker 까지 가속" 같은 확장 시나리오를 별도 도큐로
  분리할 가치 있는지.

---

## 10. 변경 이력

| 날짜 | 변경 | 근거 |
|---|---|---|
| 2026-05-20 | 초안 작성 | Phase 0 초기 설계 |
