"""Markdown → standalone HTML 변환기.

reports/ 의 .md 문서를 self-contained HTML 로 변환하여 reports/html/ 에
저장한다. 원본 .md 는 유지한다.

- Python-Markdown + extensions (tables, fenced_code, toc, attr_list).
- LaTeX 수식 ($...$, $$...$$) 은 MathJax CDN 으로 브라우저 렌더.
- frontmatter (--- yaml ---) 는 상단 메타 박스로 표시.
- 내부 .md 링크는 .html 로 rewrite (사이트처럼 상호 이동 가능).
- 임베디드 CSS (GitHub 풍) 로 단독 열람 가능.

사용법:
    python scripts/md_to_html.py            # reports/ 전체
    python scripts/md_to_html.py <file.md>  # 단일 파일
"""

from __future__ import annotations

import html as _html
import re
import sys
from pathlib import Path

import markdown

REPORTS = Path(__file__).resolve().parent.parent / "reports"
OUT_DIR = REPORTS / "html"

CSS = """
:root { --fg:#1f2328; --muted:#656d76; --border:#d0d7de; --bg:#fff;
        --code-bg:#f6f8fa; --link:#0969da; --accent:#0550ae; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans KR",
       Helvetica, Arial, sans-serif; line-height: 1.6; color: var(--fg);
       background: var(--bg); max-width: 980px; margin: 0 auto; padding: 32px 24px 80px; }
h1,h2,h3,h4 { font-weight: 600; line-height: 1.25; margin-top: 1.6em; margin-bottom: .6em; }
h1 { font-size: 1.9em; border-bottom: 2px solid var(--border); padding-bottom: .3em; }
h2 { font-size: 1.45em; border-bottom: 1px solid var(--border); padding-bottom: .25em; }
h3 { font-size: 1.2em; } h4 { font-size: 1.05em; color: var(--muted); }
a { color: var(--link); text-decoration: none; } a:hover { text-decoration: underline; }
code { background: var(--code-bg); padding: .15em .4em; border-radius: 6px;
       font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; font-size: .88em; }
pre { background: var(--code-bg); padding: 14px 16px; border-radius: 8px; overflow-x: auto;
      border: 1px solid var(--border); }
pre code { background: none; padding: 0; font-size: .85em; }
table { border-collapse: collapse; margin: 1em 0; width: 100%; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--border); padding: 6px 13px; text-align: left; }
th { background: var(--code-bg); font-weight: 600; }
tr:nth-child(2n) td { background: #fafbfc; }
blockquote { border-left: 4px solid var(--border); margin: 1em 0; padding: .2em 1em;
             color: var(--muted); background: #f6f8fa55; }
hr { border: none; border-top: 1px solid var(--border); margin: 2em 0; }
.metabox { background: var(--code-bg); border: 1px solid var(--border); border-radius: 8px;
           padding: 10px 16px; margin-bottom: 24px; font-size: .9em; color: var(--muted); }
.metabox b { color: var(--fg); }
.docnav { font-size: .85em; color: var(--muted); margin-bottom: 8px; }
.toc { background: #f6f8fa55; border: 1px solid var(--border); border-radius: 8px;
       padding: 8px 16px 8px 0; margin: 1em 0; font-size: .92em; }
.toc > ul { margin: .3em 0; }
img { max-width: 100%; }
""".strip()

# MathJax: 달러($) 기반 구분자는 끄고 \(..\) / \[..\] 만 인식.
# 이렇게 하면 "$0.0984" 같은 가격 표기가 수식으로 오인되지 않는다.
# 수식 자체는 pymdownx.arithmatex (generic mode) 가 markdown 처리 전에
# 추출해 \(..\) / \[..\] 로 감싸므로 subscript 의 '_' 도 깨지지 않는다.
MATHJAX = """
<script>
window.MathJax = { tex: { inlineMath: [['\\\\(','\\\\)']], displayMath: [['\\\\[','\\\\]']] },
                   options: { skipHtmlTags: ['script','noscript','style','textarea','pre','code'] } };
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
""".strip()


def _split_frontmatter(md: str) -> tuple[dict[str, str], str]:
    if not md.startswith("---\n"):
        return {}, md
    end = md.find("\n---", 4)
    if end == -1:
        return {}, md
    fm_block = md[4:end]
    body = md[end + 4 :].lstrip("\n")
    meta: dict[str, str] = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, body


def _protect_math(md: str) -> tuple[str, list[tuple[str, str]]]:
    """markdown 처리 전에 $$..$$ / $..$ 수식을 placeholder 로 보호.

    arithmatex 의 인라인/블록 경계 매칭 버그를 피하기 위해 직접 추출한다.
    가격 표기 '$0.0984' (닫는 $ 없음) 는 쌍이 아니라서 잡히지 않는다.
    """
    blocks: list[tuple[str, str]] = []

    def _stash_block(m: re.Match) -> str:
        blocks.append(("block", m.group(1).strip()))
        return f"\x00MATH{len(blocks) - 1}\x00"

    md = re.sub(r"\$\$(.+?)\$\$", _stash_block, md, flags=re.DOTALL)

    def _stash_inline(m: re.Match) -> str:
        blocks.append(("inline", m.group(1)))
        return f"\x00MATH{len(blocks) - 1}\x00"

    # 인라인: $...$ 같은 줄, 여는 $ 뒤 공백 아님, 닫는 $ 뒤 숫자 아님(가격 제외)
    md = re.sub(r"\$(?!\s)([^$\n]+?)(?<!\s)\$(?!\d)", _stash_inline, md)
    return md, blocks


def _restore_math(html_body: str, blocks: list[tuple[str, str]]) -> str:
    for i, (kind, expr) in enumerate(blocks):
        ph = f"\x00MATH{i}\x00"
        repl = f"\\[{expr}\\]" if kind == "block" else f"\\({expr}\\)"
        html_body = html_body.replace(ph, repl)
    return html_body


def _rewrite_md_links(html_body: str) -> str:
    """href="...md" / "...md#anchor" → ".html" (내부 상호 이동)."""
    def repl(m: re.Match) -> str:
        url = m.group(1)
        if url.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        new = re.sub(r"\.md(#|$)", r".html\1", url)
        return f'href="{new}"'

    return re.sub(r'href="([^"]+)"', repl, html_body)


def convert(md: str, title: str, depth: int) -> str:
    meta, body = _split_frontmatter(md)

    body, math_blocks = _protect_math(body)
    md_engine = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "attr_list", "sane_lists"],
        output_format="html5",
    )
    body_html = md_engine.convert(body)
    body_html = _restore_math(body_html, math_blocks)
    body_html = _rewrite_md_links(body_html)

    meta_html = ""
    if meta:
        items = " &nbsp;·&nbsp; ".join(
            f"<b>{_html.escape(k)}</b>: {_html.escape(v)}" for k, v in meta.items()
        )
        meta_html = f'<div class="metabox">{items}</div>'

    # 상위 reports 인덱스로 가는 nav (html/ 기준 상대 경로)
    up = "../" * depth
    nav = f'<div class="docnav"><a href="{up}00_overview.html">↑ 개요</a> · ' \
          f'<a href="{up}history/0000_index.html">history</a></div>'

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)} · reco_bench</title>
<style>{CSS}</style>
{MATHJAX}
</head>
<body>
{nav}
{meta_html}
{body_html}
</body>
</html>
"""


def convert_file(src: Path) -> Path:
    rel = src.relative_to(REPORTS)
    depth = len(rel.parts) - 1  # html/ 기준 하위 디렉토리 깊이
    dst = OUT_DIR / rel.with_suffix(".html")
    dst.parent.mkdir(parents=True, exist_ok=True)
    title = rel.stem
    dst.write_text(convert(src.read_text(encoding="utf-8"), title, depth), encoding="utf-8")
    return dst


def _copy_figures() -> int:
    """reports/figures/*.png 를 html/figures/ 로 복사 (baseline_results.html 이 참조)."""
    import shutil

    src_dir = REPORTS / "figures"
    if not src_dir.is_dir():
        return 0
    dst_dir = OUT_DIR / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for png in src_dir.glob("*.png"):
        shutil.copy(png, dst_dir / png.name)
        n += 1
    return n


def _write_index() -> None:
    """html/index.html — 변환된 모든 문서로의 링크 모음."""
    core = [
        ("00_overview.html", "00 · 개요 (Overview)"),
        ("01_metric_design.html", "01 · 메트릭 설계"),
        ("02_dataset_selection.html", "02 · 데이터셋 선정"),
        ("03_baseline_methodology.html", "03 · Baseline 방법론"),
        ("04_vdpu_value_proposition.html", "04 · VDPU 가치 명제"),
        ("05_reproducibility.html", "05 · 재현성"),
        ("06_phase1_findings.html", "06 · Phase 1 발견"),
        ("baseline_results.html", "baseline_results (자동 생성 결과)"),
    ]
    core = [(h, t) for h, t in core if (OUT_DIR / h).exists()]
    hist = sorted((OUT_DIR / "history").glob("*.html"))
    plan = sorted((OUT_DIR / "planning").glob("*.html"))

    def ul(rows: list[str]) -> str:
        return "\n".join(f"      <li>{r}</li>" for r in rows)

    core_li = [f'<a href="{h}">{t}</a>' for h, t in core]
    hist_li = [f'<a href="history/{p.name}">{p.stem}</a>' for p in hist]
    plan_li = [f'<a href="planning/{p.name}">{p.stem}</a>' for p in plan]

    html_doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>reco_bench · reports</title>
<style>{CSS}</style>
</head><body>
<h1>reco_bench — reports (HTML)</h1>
<p>Vector-DB 가속기(VDPU)용 Two-tower 추천 검색 벤치마크 문서.
원본은 <code>reports/*.md</code>, 본 HTML 은 <code>scripts/md_to_html.py</code> 로 생성.</p>
<h2>핵심 문서</h2>
<ul>
{ul(core_li)}
</ul>
<h2>진행 history</h2>
<ul>
{ul(hist_li)}
</ul>
<h2>planning</h2>
<ul>
{ul(plan_li)}
</ul>
</body></html>
"""
    (OUT_DIR / "index.html").write_text(html_doc, encoding="utf-8")


def main(argv: list[str]) -> None:
    if len(argv) > 1:
        srcs = [Path(argv[1])]
    else:
        srcs = sorted(p for p in REPORTS.rglob("*.md") if "html" not in p.parts and "confluence" not in p.parts)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for src in srcs:
        dst = convert_file(src)
        print(f"  {src.relative_to(REPORTS)} → {dst.relative_to(REPORTS)}")
    n_fig = _copy_figures()
    _write_index()
    print(f"\n{len(srcs)} files converted → {OUT_DIR}")
    print(f"  + {n_fig} figures copied, index.html written")


if __name__ == "__main__":
    main(sys.argv)
