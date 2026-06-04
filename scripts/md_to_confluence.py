"""Markdown → Confluence Wiki Markup 변환기.

reports/ 의 .md 문서를 Confluence Wiki Markup (legacy wiki markup) 으로
변환하여 reports/confluence/ 에 .confluence 파일로 저장한다. 원본 .md 는
유지한다.

사용법:
    python scripts/md_to_confluence.py            # reports/ 전체 변환
    python scripts/md_to_confluence.py <file.md>  # 단일 파일

지원 변환:
- 제목 #..###### → h1...h6.
- **bold** → *bold*,  `code` → {{code}}
- ```lang code``` → {code:lang} ... {code}
- 표 (| a | b | + |---|) → ||a||b|| / |1|2|
- 링크 [t](u) → [t|u]
- 인용 > → {quote}
- frontmatter (--- yaml ---) → {info} 패널
- 블록 수식 $$..$$ → {noformat}, 인라인 $..$ 는 보존
- 수평선 --- → ----
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPORTS = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR = REPORTS / "confluence"


def _convert_inline(text: str) -> str:
    """코드블록 바깥의 한 줄에 대한 inline 변환."""
    # inline code 를 placeholder 로 보호 (그 안의 ** 등 변환 방지)
    codes: list[str] = []

    def _stash(m: re.Match) -> str:
        codes.append(m.group(1))
        return f"\x00CODE{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)

    # 링크 [text](url) → [text|url]
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"[\1|\2]", text)

    # bold **x** 또는 __x__ → *x*
    text = re.sub(r"\*\*([^*]+)\*\*", r"*\1*", text)
    text = re.sub(r"__([^_]+)__", r"*\1*", text)

    # 보호한 inline code 복원 → {{code}}
    def _restore(m: re.Match) -> str:
        idx = int(m.group(1))
        return "{{" + codes[idx] + "}}"

    text = re.sub(r"\x00CODE(\d+)\x00", _restore, text)

    # 링크 텍스트 안의 매크로 정리: [{{x}}|url] → [x|url]
    # (Confluence 링크 텍스트에는 {{...}} 매크로가 렌더되지 않음)
    text = re.sub(r"\[\{\{([^}]+)\}\}\|", r"[\1|", text)
    return text


def _convert_table(block: list[str]) -> list[str]:
    """markdown 표 블록 → Confluence 표.

    block 의 각 줄은 '| .. | .. |' 형태. 두 번째 줄이 separator(|---|).
    """
    out: list[str] = []
    for i, line in enumerate(block):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if i == 1 and all(re.fullmatch(r":?-+:?", c) for c in cells):
            continue  # separator 행 제거
        cells = [_convert_inline(c) if c else " " for c in cells]
        if i == 0:
            out.append("||" + "||".join(cells) + "||")
        else:
            out.append("|" + "|".join(cells) + "|")
    return out


def convert(md: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    # 1. frontmatter (--- ... ---) → {info}
    if lines and lines[0].strip() == "---":
        fm: list[str] = []
        j = 1
        while j < n and lines[j].strip() != "---":
            fm.append(lines[j])
            j += 1
        if j < n:  # closing --- 발견
            out.append("{info:title=metadata}")
            out.extend(fm)
            out.append("{info}")
            out.append("")
            i = j + 1

    in_code = False
    code_buf: list[str] = []
    code_lang = ""

    while i < n:
        line = lines[i]

        # 코드블록 토글
        fence = re.match(r"^```(\w*)\s*$", line)
        if fence and not in_code:
            in_code = True
            code_lang = fence.group(1)
            code_buf = []
            i += 1
            continue
        if line.strip() == "```" and in_code:
            in_code = False
            macro = f"{{code:{code_lang}}}" if code_lang else "{code}"
            out.append(macro)
            out.extend(code_buf)
            out.append("{code}")
            i += 1
            continue
        if in_code:
            code_buf.append(line)
            i += 1
            continue

        # 블록 수식 $$ ... $$ → {noformat}
        if line.strip() == "$$":
            math_buf = []
            i += 1
            while i < n and lines[i].strip() != "$$":
                math_buf.append(lines[i])
                i += 1
            out.append("{noformat}")
            out.extend(math_buf)
            out.append("{noformat}")
            i += 1
            continue

        # 표 감지: 현재 줄과 다음 줄이 | 로 시작 + 다음이 separator
        if (
            line.strip().startswith("|")
            and i + 1 < n
            and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1])
        ):
            tbl: list[str] = []
            while i < n and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            out.extend(_convert_table(tbl))
            continue

        # 제목
        h = re.match(r"^(#{1,6})\s+(.*)$", line)
        if h:
            level = len(h.group(1))
            out.append(f"h{level}. {_convert_inline(h.group(2))}")
            i += 1
            continue

        # 수평선
        if re.match(r"^---+\s*$", line) or re.match(r"^\*\*\*+\s*$", line):
            out.append("----")
            i += 1
            continue

        # blockquote
        bq = re.match(r"^>\s?(.*)$", line)
        if bq:
            # 연속된 quote 줄 묶기
            q_buf = [bq.group(1)]
            i += 1
            while i < n:
                m2 = re.match(r"^>\s?(.*)$", lines[i])
                if not m2:
                    break
                q_buf.append(m2.group(1))
                i += 1
            out.append("{quote}")
            out.extend(_convert_inline(x) for x in q_buf)
            out.append("{quote}")
            continue

        # 리스트: markdown 의 -, * 불릿 → confluence * ; 1. → #
        lst = re.match(r"^(\s*)([-*])\s+(.*)$", line)
        if lst:
            depth = len(lst.group(1)) // 2 + 1
            out.append("*" * depth + " " + _convert_inline(lst.group(3)))
            i += 1
            continue
        num = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if num:
            depth = len(num.group(1)) // 2 + 1
            out.append("#" * depth + " " + _convert_inline(num.group(2)))
            i += 1
            continue

        # 일반 줄
        out.append(_convert_inline(line))
        i += 1

    return "\n".join(out)


def convert_file(src: Path, dst_root: Path) -> Path:
    rel = src.relative_to(REPORTS)
    dst = dst_root / rel.with_suffix(".confluence")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    return dst


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        srcs = [Path(argv[1])]
    else:
        srcs = sorted(p for p in REPORTS.rglob("*.md") if "confluence" not in p.parts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in srcs:
        dst = convert_file(src, OUT_DIR)
        print(f"  {src.relative_to(REPORTS)} → {dst.relative_to(REPORTS)}")
    print(f"\n{len(srcs)} files converted → {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv)
