# reports/confluence/

`reports/` 의 모든 markdown 문서를 **Confluence Wiki Markup** (legacy
wiki markup) 으로 변환한 사본이다. 사내 Confluence Wiki 에 붙여넣기
위한 것이며, **원본은 `reports/` 의 `.md` 파일**이다.

## 생성 방법

```bash
python scripts/md_to_confluence.py            # reports/ 전체 재변환
python scripts/md_to_confluence.py <file.md>  # 단일 파일
```

원본 `.md` 가 바뀌면 위 명령으로 재생성한다. 이 디렉토리의
`.confluence` 파일은 **직접 편집하지 않는다** (다음 변환 때 덮어쓰여짐).

## Confluence 에 올리는 법

각 `.confluence` 파일의 내용을 복사해서:

1. Confluence 페이지 편집 → `...` 메뉴 → **Insert Wiki Markup**
   (또는 편집창에서 `{` 입력 후 "Markup")
2. 파일 내용 붙여넣기 → Insert.

> Confluence Cloud 의 신규 에디터는 "Insert Wiki Markup" 이 기본 비활성
> 일 수 있다. 그 경우 페이지 생성 시 "..." → "Markup" 또는 Data Center
> 버전의 편집기를 사용. 변환 불가 시 markdown 원본을 Confluence 의
> Markdown 매크로로 직접 붙여도 된다.

## 변환 규칙 (요약)

| Markdown | Confluence Wiki |
|---|---|
| `# H1` ~ `###### H6` | `h1.` ~ `h6.` |
| `**bold**` | `*bold*` |
| `` `code` `` | `{{code}}` |
| ` ```lang ... ``` ` | `{code:lang} ... {code}` |
| 표 (`\| a \| b \|`) | `\|\|a\|\|b\|\|` / `\|1\|2\|` |
| `[text](url)` | `[text\|url]` |
| `> quote` | `{quote} ... {quote}` |
| frontmatter (`--- yaml ---`) | `{info:title=metadata}` 패널 |
| 블록 수식 `$$ ... $$` | `{noformat} ... {noformat}` |
| `---` (수평선) | `----` |

## 알려진 한계

- **LaTeX 수식**: Confluence 는 기본적으로 LaTeX 를 렌더하지 않는다.
  블록 수식 (`$$...$$`) 은 `{noformat}` 으로 감싸 raw TeX 소스를
  보여준다. 인라인 `$...$` 는 그대로 둔다. 수식 렌더가 필요하면
  Confluence 의 LaTeX 매크로 플러그인 (예: "LaTeX Math") 을 별도 설치
  후 `{latex}` 매크로로 교체해야 한다.
- **이미지** (`figures/*.png`): wiki markup 의 `!image.png!` 로 자동
  변환하지 않았다. Confluence 에 그림을 첨부한 뒤 수동으로 `!파일명!`
  추가 권장.
- 중첩 리스트의 들여쓰기 깊이는 2-space 기준으로 추정한다.
