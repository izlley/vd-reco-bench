# scripts/

`reco_bench` 의 end-to-end 파이프라인 진입점 shell 스크립트.

설계 근거: `reports/03_baseline_methodology.md` §6,
`reports/05_reproducibility.md` §5.

## 실행 순서

```
00_download_data.sh    # 데이터 다운로드 + 전처리
10_train_two_tower.sh  # Two-tower 학습 → checkpoints/
20_build_index.sh      # ANN 인덱스 빌드 → indexes/
30_run_benchmark.sh    # 평가 → results/
99_make_report.sh      # 결과 집계 → reports/baseline_results.md
```

## 상태

Phase 1 진입 전 placeholder. 본 스크립트들의 실제 구현은 Phase 1 의
다음 작업 단위에서 추가된다.
