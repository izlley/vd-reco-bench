---
date: 2026-06-04
phase: 1
topic: confluence-export
status: completed
---

# reports 문서의 Confluence Wiki Markup 변환

## What changed

- `scripts/md_to_confluence.py` 추가 — markdown → Confluence Wiki
  Markup (legacy wiki markup) 변환기.
- `reports/` 의 모든 `.md` (21개) 를 `reports/confluence/` 에 동일한
  디렉토리 구조로 `.confluence` 파일로 변환.
- `reports/confluence/README.md` 추가 — 생성 방법, Confluence 업로드
  방법, 변환 규칙, 한계 설명.
- 원본 `.md` 는 그대로 유지 (source of truth).

지원하는 변환:
- 제목 `#`..`######` → `h1.`..`h6.`
- `**bold**` → `*bold*`, `` `code` `` → `{{code}}`
- 코드블록 → `{code:lang} ... {code}`
- markdown 표 → Confluence `||header||` / `|cell|`
- 링크 `[t](u)` → `[t|u]` (링크 텍스트 내 매크로 정리 포함)
- frontmatter → `{info:title=metadata}` 패널
- blockquote → `{quote}`, 수평선 → `----`
- 블록 수식 `$$..$$` → `{noformat}` (raw TeX 보존)

## Why

- 사용자가 reports 문서를 사내 Confluence Wiki 에 올릴 수 있도록
  Wiki Markup 형식의 별도 파일을 요청.
- **변환기 스크립트로 구현** 한 이유: 문서가 21개 + 향후 계속 늘어나며,
  원본 `.md` 가 바뀔 때마다 재변환이 필요하므로 수작업 변환은 drift
  위험. `scripts/99_make_report.sh` 처럼 재현 가능한 파이프라인으로 둠.
- 원본 `.md` 를 source of truth 로 유지하고 `.confluence` 는 생성물로
  취급 → 편집은 항상 `.md` 에서.

## Validation

- 21개 파일 모두 변환 성공.
- 표 변환 검증: `baseline_results.confluence` 의 메인 표가
  `||header||` + `|cell|` 형식으로 정상.
- frontmatter 검증: history 파일의 `--- yaml ---` 가 `{info}` 패널로.
- 코드블록 / LaTeX 블록: `{code}` / `{noformat}` 로 감싸지고 내부
  내용은 변환되지 않음 (보존).
- 링크 텍스트 내 매크로 버그 (`[{{x}}|url]`) 를 후처리로 `[x|url]` 로
  정리, 잔존 0건 확인.

## Open questions / next

- **LaTeX 렌더**: Confluence 가 기본 LaTeX 미지원이라 `{noformat}` 으로
  raw TeX 만 보임. 수식 렌더가 필요하면 Confluence LaTeX 매크로 플러그인
  설치 후 `{latex}` 로 교체 필요 (reports/confluence/README.md 에 명시).
- **이미지**: `figures/*.png` 는 wiki markup `!img!` 로 자동 변환 안 함.
  Confluence 에 그림 첨부 후 수동 삽입 권장.
- Confluence REST API 로 자동 업로드 (페이지 생성/갱신) 까지 확장
  가능하나, 현재는 복사-붙여넣기용 파일 생성까지만.
