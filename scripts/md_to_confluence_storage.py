"""Markdown → Confluence Storage Format (XHTML 기반) 변환기.

Confluence 의 storage format 은 REST API 의 `body.storage.value` 에 넣거나
페이지 import 에 쓰는 XHTML 기반 포맷이다. wiki markup (legacy) 과 달리
`<ac:structured-macro>` 등 Confluence 네임스페이스 태그를 사용한다.

전략: Python-Markdown 으로 XHTML body 를 만든 뒤, 다음을 storage format
매크로로 후처리한다.
- ```lang ...```  → <ac:structured-macro ac:name="code"> + CDATA
- frontmatter      → <ac:structured-macro ac:name="info"> 패널
- 블록 수식 $$..$$ → <ac:structured-macro ac:name="noformat"> (raw TeX)
- 인라인 수식 $..$ → <code> (Confluence 기본 LaTeX 미지원)
- 이미지           → <ac:image><ri:attachment .../></ac:image>
- 표 / 제목 / 리스트 / 강조 → 표준 XHTML (storage format 이 그대로 수용)

출력은 **body fragment** (html/head 래퍼 없음) — Confluence 페이지 본문에
그대로 들어가는 형태.

사용법:
    python scripts/md_to_confluence_storage.py            # reports/ 전체
    python scripts/md_to_confluence_storage.py <file.md>  # 단일 파일
"""

from __future__ import annotations

import html as _html
import re
import sys
from pathlib import Path

import markdown

REPORTS = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR = REPORTS / "confluence_storage"


# ---- 수식 보호 (md_to_html.py 와 동일 전략) ----
def _protect_math(md: str) -> tuple[str, list[tuple[str, str]]]:
    blocks: list[tuple[str, str]] = []

    def _stash_block(m: re.Match) -> str:
        blocks.append(("block", m.group(1).strip()))
        return f"\x00MATH{len(blocks) - 1}\x00"

    md = re.sub(r"\$\$(.+?)\$\$", _stash_block, md, flags=re.DOTALL)

    def _stash_inline(m: re.Match) -> str:
        blocks.append(("inline", m.group(1)))
        return f"\x00MATH{len(blocks) - 1}\x00"

    md = re.sub(r"\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)", _stash_inline, md)
    return md, blocks


def _restore_math_storage(xhtml: str, blocks: list[tuple[str, str]]) -> str:
    for i, (kind, expr) in enumerate(blocks):
        ph = f"\x00MATH{i}\x00"
        if kind == "block":
            macro = (
                '<ac:structured-macro ac:name="noformat">'
                "<ac:plain-text-body><![CDATA[" + expr + "]]></ac:plain-text-body>"
                "</ac:structured-macro>"
            )
            # 블록 수식은 <p>placeholder</p> 형태로 감싸졌을 수 있어 같이 치환
            xhtml = xhtml.replace(f"<p>{ph}</p>", macro)
            xhtml = xhtml.replace(ph, macro)
        else:
            xhtml = xhtml.replace(ph, "<code>" + _html.escape(expr) + "</code>")
    return xhtml


def _split_frontmatter(md: str) -> tuple[dict[str, str], str]:
    if not md.startswith("---\n"):
        return {}, md
    end = md.find("\n---", 4)
    if end == -1:
        return {}, md
    meta: dict[str, str] = {}
    for line in md[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, md[end + 4 :].lstrip("\n")


def _code_blocks_to_macro(xhtml: str) -> str:
    """<pre><code class="language-x">..escaped..</code></pre> → code 매크로."""

    def repl(m: re.Match) -> str:
        lang = m.group("lang") or ""
        body = _html.unescape(m.group("body"))
        lang_param = (
            f'<ac:parameter ac:name="language">{lang}</ac:parameter>' if lang else ""
        )
        return (
            '<ac:structured-macro ac:name="code">'
            + lang_param
            + "<ac:plain-text-body><![CDATA["
            + body
            + "]]></ac:plain-text-body></ac:structured-macro>"
        )

    pat = re.compile(
        r'<pre><code(?:\s+class="language-(?P<lang>[^"]*)")?>(?P<body>.*?)</code></pre>',
        re.DOTALL,
    )
    return pat.sub(repl, xhtml)


def _images_to_attachment(xhtml: str) -> str:
    """<img src="figures/x.png" .../> → <ac:image><ri:attachment .../></ac:image>.

    Confluence 에 해당 파일명을 페이지 첨부로 올리면 자동 표시된다.
    """

    def repl(m: re.Match) -> str:
        src = m.group(1)
        fname = src.rsplit("/", 1)[-1]
        return (
            '<ac:image><ri:attachment ri:filename="' + fname + '"/></ac:image>'
        )

    return re.sub(r'<img\s+[^>]*?src="([^"]+)"[^>]*/?>', repl, xhtml)


def _frontmatter_panel(meta: dict[str, str]) -> str:
    if not meta:
        return ""
    rows = "".join(
        f"<tr><th>{_html.escape(k)}</th><td>{_html.escape(v)}</td></tr>" for k, v in meta.items()
    )
    return (
        '<ac:structured-macro ac:name="info">'
        '<ac:parameter ac:name="title">metadata</ac:parameter>'
        "<ac:rich-text-body><table><tbody>" + rows + "</tbody></table></ac:rich-text-body>"
        "</ac:structured-macro>\n"
    )


def convert(md: str) -> str:
    meta, body = _split_frontmatter(md)
    body, math_blocks = _protect_math(body)

    md_engine = markdown.Markdown(
        extensions=["tables", "fenced_code", "attr_list", "sane_lists"],
        output_format="xhtml",  # self-closing 태그 → storage format 호환
    )
    xhtml = md_engine.convert(body)

    xhtml = _code_blocks_to_macro(xhtml)
    xhtml = _restore_math_storage(xhtml, math_blocks)
    xhtml = _images_to_attachment(xhtml)

    # 내부 .md 링크는 storage format 에서 페이지 제목 링크로 변환이 까다로워
    # (페이지 제목 의존) 일반 anchor 의 .html 로만 rewrite 해 둔다.
    xhtml = re.sub(
        r'href="([^"]+?)\.md(#[^"]*)?"',
        lambda m: f'href="{m.group(1)}.html{m.group(2) or ""}"',
        xhtml,
    )

    return _frontmatter_panel(meta) + xhtml + "\n"


def convert_file(src: Path) -> Path:
    rel = src.relative_to(REPORTS)
    dst = OUT_DIR / rel.with_suffix(".storage.xml")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(convert(src.read_text(encoding="utf-8")), encoding="utf-8")
    return dst


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        srcs = [Path(argv[1])]
    else:
        srcs = sorted(
            p
            for p in REPORTS.rglob("*.md")
            if not {"html", "confluence", "confluence_storage"} & set(p.parts)
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in srcs:
        dst = convert_file(src)
        print(f"  {src.relative_to(REPORTS)} → {dst.relative_to(REPORTS)}")
    print(f"\n{len(srcs)} files converted → {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv)
