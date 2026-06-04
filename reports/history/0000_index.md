# reco_bench — 진행 history 인덱스

이 파일은 `reports/history/` 의 모든 entry 를 시간순으로 나열한다. 각 entry
는 벤치마크 개발의 한 discrete 한 단계 — 새 문서 작성, 코드 마일스톤,
설계 방향 전환, 명시적 의사결정 — 를 기록한다.

목표는 향후 reader (공개 배포 후의 오픈소스 사용자 포함) 가 벤치마크가
**왜 지금의 모습인지** 를 재구성할 수 있도록 하는 것이다. 단순히 무엇을
하는지가 아니다.

## 규칙

- 의미 있는 단계마다 entry 하나. 입자도 (granularity) 기준: 컨퍼런스 논문의
  "implementation" 섹션에서 한 단락 분량이 될 만하면 entry 가치가 있다.
- 파일명: `YYYY-MM-DD_<short-slug>.md`. 같은 날짜에 여러 entry 가능;
  순서는 slug 알파벳 순.
- Frontmatter 필드: `date`, `phase` (0, 1, 2, ...), `topic`, `status`
  (`completed` / `in_progress` / `abandoned`).
- 각 entry 는 네 섹션: **What changed**, **Why**, **Validation**,
  **Open questions / next**.
- 다른 entry 참조는 상대 경로로, 예: `[init](2026-05-20_initial-planning.md)`.

## Entries

| 날짜 | Phase | Slug | 상태 | 한 줄 요약 |
|---|---|---|---|---|
| 2026-05-20 | 0 | [initial-planning](2026-05-20_initial-planning.md) | completed | 프로젝트 킥오프: Phase 0+1 을 GPU baseline 만으로 한정 (VDPU 미사용), MovieLens-25M + Amazon Reviews 2023 채택, `Retriever` 추상 클래스를 VDPU 플러그인 seam 으로 정의. |
| 2026-05-20 | 0 | [repo-scaffold](2026-05-20_repo-scaffold.md) | completed | 디렉토리 골격, 공개용 foundation 파일 (LICENSE/CITATION/README/.gitignore), Phase 0 문서 scaffold 생성. |
| 2026-05-20 | 0 | [phase0-docs-complete](2026-05-20_phase0-docs-complete.md) | completed | Phase 0 의 reports 문서 6종 + configs YAML + `Retriever`/`metrics` 코드 seam + 단위 테스트 스켈레톤 완성. Phase 1 코드 구현 진입 가능 상태. |
| 2026-06-04 | 1 | [env-setup](2026-06-04_env-setup.md) | completed | Phase 1 시작. 4× H100 / CUDA 12.8 / PyTorch 2.9 / cuVS 26.04 / faiss-cpu 1.14 환경 검증. **GPU retriever 를 FAISS-GPU 에서 cuVS IVF-PQ 로 변경** (cublas ABI 충돌 회피). `utils/{seed,io,gpu_info}` 추가. |
| 2026-06-04 | 1 | [ml25m-preprocess](2026-06-04_ml25m-preprocess.md) | completed | Step 2 부분. ML-25M 다운로드 + 전처리 파이프라인 (`data/{base,preprocessing,ml25m,pipeline}.py`) 작성 + 실 데이터 처리 완료. 14M train / 779k val / 779k test, 24k items, 162k users. |
| 2026-06-04 | 1 | [user-requirements-clarified](2026-06-04_user-requirements-clarified.md) | completed | 사용자가 vector DB 비교 + 시각화 + 단일 명령 + 속도 강조의 4가지 요구사항을 명시. retriever 카테고리 A/B/C 분리, `reports/00_overview.md` 와 `04_vdpu_value_proposition.md` 갱신. |
| 2026-06-04 | 1 | [phase1-full-pipeline](2026-06-04_phase1-full-pipeline.md) | completed | 6 retriever (ANN 라이브러리 4 + Vector DB 2) end-to-end. Milvus Lite, Qdrant Local, ScaNN 추가. `speedup ratio` metric, QUICKSTART.md 추가. v1 hyperparameter 실패 (logQ off → 더 떨어짐, v0 로 회귀). |
| 2026-06-04 | 1 | [final-eval](2026-06-04_final-eval.md) | completed | Phase 1 평가 완료. milvus/qdrant API 호환성 디버그, 측정 시간 최적화 (4-5배 단축), GPU keepalive 적용. 자동 리포트 5종 그래프 + speedup table 생성. |
| 2026-06-04 | 1 | [phase1-complete](2026-06-04_phase1-complete.md) | completed | **Phase 1 종료.** ML-25M + Amazon Beauty 두 dataset × 4-6 retriever 측정 완료. ML-25M 에서 FAISS-CPU HNSW × 74.33 vs cuVS CAGRA, Amazon Beauty 에서 ×19.65. baseline_results.md 자동 생성 (메인 표 + speedup + 5종 그래프). |
| 2026-06-04 | 1 | [readme-references](2026-06-04_readme-references.md) | completed | README 에 "참고한 오픈 벤치마크" 섹션 추가 (ann-benchmarks, MLPerf DLRM, VectorDBBench, recommenders, RecBole, Google Two-tower + 데이터셋). 각 항목에 링크 + 반영 위치 명시. |
| 2026-06-04 | 1 | [confluence-export](2026-06-04_confluence-export.md) | completed | reports 문서 21개를 Confluence Wiki Markup 으로 변환 (`scripts/md_to_confluence.py` → `reports/confluence/`). 원본 .md 는 유지. |
| 2026-06-04 | 1 | [html-export](2026-06-04_html-export.md) | completed | reports 문서 22개를 standalone HTML 로 변환 (`scripts/md_to_html.py` → `reports/html/`, MathJax 수식 렌더 + index.html + figures). 원본 .md 유지. |
