"""
split_chapters.py — 新世界より OCR全文を3話分に分割する

出力:
  trpg/source/session1_ch1.md      (1章  p.010-176)
  trpg/source/session2_ch2_3.md    (2-3章 p.178-445)
  trpg/source/session3_ch4_5_6.md  (4-6章 p.448-953)

使用法:
  python split_chapters.py
"""

import re
from pathlib import Path

INPUT = Path(__file__).parent / "scan/output/prod/shinsekaiyori.md"
OUT_DIR = Path(__file__).parent / "trpg/source"

SESSIONS = [
    ("session1_ch1.md",      "第1話「偽りの箱庭」1章",            10,  176),
    ("session2_ch2_3.md",    "第2話「欠けゆく輪」2-3章",         178,  445),
    ("session3a_ch4_5.md",   "第3話前半「業火の獣たち」4-5章",   448,  797),
    ("session3b_ch6.md",     "第3話後半「業火の獣たち」6章",     800,  953),
]

PAGE_RE = re.compile(r"<!-- page: shinsekaiyori_(\d+) -->")

def split():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    text = INPUT.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    # 各行の「現在のページ番号」を求める
    current_page = 0
    page_of_line = []
    for line in lines:
        m = PAGE_RE.search(line)
        if m:
            current_page = int(m.group(1))
        page_of_line.append(current_page)

    for filename, label, start, end in SESSIONS:
        out_path = OUT_DIR / filename
        selected = [
            line for line, page in zip(lines, page_of_line)
            if start <= page <= end
        ]
        out_path.write_text("".join(selected), encoding="utf-8")
        size_kb = out_path.stat().st_size // 1024
        print(f"[OK] {filename}  ({size_kb} KB, {len(selected):,} lines)  {label}")

if __name__ == "__main__":
    if not INPUT.exists():
        print(f"[ERROR] 入力ファイルが見つかりません: {INPUT}")
        raise SystemExit(1)
    split()
    print("\n出力先:", OUT_DIR)
