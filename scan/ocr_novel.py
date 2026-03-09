"""
日本語小説スキャン OCR スクリプト（manga-ocr + 動的列検出 版）
=============================================================
対象レイアウト:
  - 縦書き2段組（上段・下段）
  - 各段内は右から左へ列が並ぶ
  - 各列は上から下へ読む

【前提】
  - uv 管理の .venv（Python 3.11）内で実行すること
  - manga-ocr, numpy, pillow がインストール済みであること
  - 列検出失敗時はプレースホルダー出力（手動確認用）

【使い方】
  # テスト（1枚、ground_truth と比較）
  .venv/Scripts/python ocr_novel.py --input "(一般小説) ..." --test 1 --offset 7 --ext png

  # 全体処理
  .venv/Scripts/python ocr_novel.py --input "(一般小説) ..." --ext png

  オプション:
    --input   DIR   画像フォルダ               ※デフォルト: ./images
    --output  FILE  出力ファイル名             ※デフォルト: 自動（test/prod 振り分け）
    --ext           jpg,png,jpeg,webp          ※デフォルト: jpg,jpeg,png
    --sort          name | number              ※デフォルト: number
    --test    N     最初のN枚だけ処理し output/test/ に保存
    --offset  N     最初のN枚をスキップして処理開始
    --ground-truth  FILE  比較用正解ファイル  ※デフォルト: output/test/ground_truth_010.txt
"""

import argparse
import difflib
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# manga-ocr の遅延ロード（起動時のみ1回）
# ---------------------------------------------------------------------------
_mocr = None


def get_mocr():
    global _mocr
    if _mocr is None:
        import torch
        from manga_ocr import MangaOcr
        print("manga-ocr モデルをロード中...", end=" ", flush=True)
        _mocr = MangaOcr()
        device = _mocr.model.device
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(device.index or 0)
            print(f"完了  [GPU: {gpu_name} / {device}]")
        else:
            print(f"完了  [WARNING: CPU モード（処理が非常に遅くなります）]")
    return _mocr


# ---------------------------------------------------------------------------
# 段分割・列検出ロジック
# ---------------------------------------------------------------------------
def detect_block_split(ink):
    """上段と下段の境界行を動的に検出する。ink: bool 2D array"""
    h = ink.shape[0]
    row_ink = ink.sum(axis=1).astype(float)

    center = h // 2
    margin = h // 6
    s, e = max(0, center - margin), min(h, center + margin)

    k = max(3, h // 200)
    smoothed = np.convolve(row_ink[s:e], np.ones(k) / k, mode="same")
    max_val = smoothed.max()
    if max_val == 0:
        return h // 2

    gap_indices = np.where(smoothed < max_val * 0.05)[0]
    if len(gap_indices) >= 3:
        return int(gap_indices[len(gap_indices) // 2]) + s

    return h // 2


def find_column_peaks(block_ink, image_width, min_ink_ratio=0.12, min_dist=25):
    """縦書きブロック内の各列の中心 x 座標を返す（左→右順）。"""
    col_ink = block_ink.sum(axis=0).astype(float)
    k = max(3, image_width // 150)
    smoothed = np.convolve(col_ink, np.ones(k) / k, mode="same")

    max_val = smoothed.max()
    if max_val == 0:
        return []

    threshold = max_val * min_ink_ratio
    peaks = []

    for i in range(1, len(smoothed) - 1):
        if (smoothed[i] >= threshold
                and smoothed[i] >= smoothed[i - 1]
                and smoothed[i] >= smoothed[i + 1]):
            if not peaks or i - peaks[-1] >= min_dist:
                peaks.append(i)
            elif smoothed[i] > smoothed[peaks[-1]]:
                peaks[-1] = i

    if len(peaks) < 3:
        return peaks

    # 孤立したピーク（章番号・ページ番号等）を除去
    # ※隣接ピークが全て median×1.5 より離れている場合のみ除去
    # （内部の列グループ間ギャップには影響しない）
    spacings = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    median_sp = sorted(spacings)[len(spacings) // 2]
    cutoff = median_sp * 1.5

    filtered = []
    for i, p in enumerate(peaks):
        neighbor_dists = []
        if i > 0:
            neighbor_dists.append(peaks[i] - peaks[i - 1])
        if i < len(peaks) - 1:
            neighbor_dists.append(peaks[i + 1] - peaks[i])
        # 少なくとも1つの隣接ピークが閾値内にあればキープ
        if any(d <= cutoff for d in neighbor_dists):
            filtered.append(p)

    return filtered


def crop_columns(image, peaks, y_start, y_end, half_width):
    """ピーク位置を中心に各列をクロップして返す（右→左の読み順）。"""
    w = image.width
    crops = []
    for p in reversed(peaks):
        x0 = max(0, p - half_width)
        x1 = min(w, p + half_width)
        crops.append(image.crop((x0, y_start, x1, y_end)))
    return crops


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
def ocr_batch(mocr, images):
    """PIL画像のリストをバッチ推論してテキストのリストを返す。"""
    if not images:
        return []
    import torch
    from manga_ocr.ocr import post_process

    MAX_BATCH_SIZE = 64

    tensors = []
    for img in images:
        img_rgb = img.convert("L").convert("RGB")
        pv = mocr.processor(img_rgb, return_tensors="pt").pixel_values
        tensors.append(pv.squeeze(0))

    results = []
    for start in range(0, len(tensors), MAX_BATCH_SIZE):
        sub = tensors[start:start + MAX_BATCH_SIZE]
        batch = torch.stack(sub).to(mocr.model.device)
        with torch.no_grad():
            out = mocr.model.generate(batch, max_length=300)
        for tokens in out:
            text = mocr.tokenizer.decode(tokens.cpu(), skip_special_tokens=True)
            results.append(post_process(text))
        del batch, out

    return results


def ocr_page_manga(image_path):
    """manga-ocr + 動的列検出 + バッチ推論でページを OCR する。"""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    ink = np.array(img.convert("L")) < 200

    split_y = detect_block_split(ink)
    upper_peaks = find_column_peaks(ink[:split_y], w)
    lower_peaks = find_column_peaks(ink[split_y:], w)

    if len(upper_peaks) < 3 and len(lower_peaks) < 3:
        return "", "fallback_needed"

    def half_w(peaks):
        if len(peaks) < 2:
            return 30
        sp = sorted([peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)])
        return max(20, sp[len(sp) // 2] // 2 + 4)

    CHUNK_H = 300
    BOUNDARY_TOLERANCE = 25
    MIN_INK_RATIO = 0.01

    def find_char_boundary(col_ink_arr, target_y, tol=BOUNDARY_TOLERANCE):
        """target_y 付近でインク密度が最小の行（文字間ギャップ）を探す。"""
        col_h = col_ink_arr.shape[0]
        s = max(0, target_y - tol)
        e = min(col_h, target_y + tol)
        row_ink = col_ink_arr[s:e].sum(axis=1)
        return s + int(row_ink.argmin())

    # Phase 1: 全チャンクを収集（GPU未使用）
    all_chunks = []
    col_chunk_ranges = []  # 各列に対応する all_chunks のスライス範囲
    col_errors = []

    for peaks, y0, y1 in [(upper_peaks, 0, split_y), (lower_peaks, split_y, h)]:
        if not peaks:
            continue
        for col_crop in crop_columns(img, peaks, y0, y1, half_w(peaks)):
            try:
                col_h = col_crop.height
                col_ink_arr = np.array(col_crop.convert("L")) < 200
                start = len(all_chunks)
                if col_h <= CHUNK_H:
                    all_chunks.append(col_crop)
                else:
                    boundaries = [0]
                    cy = CHUNK_H
                    while cy < col_h:
                        boundary = find_char_boundary(col_ink_arr, cy)
                        boundaries.append(boundary)
                        cy = boundary + CHUNK_H
                    boundaries.append(col_h)

                    for i in range(len(boundaries) - 1):
                        s, e = boundaries[i], boundaries[i + 1]
                        chunk = col_crop.crop((0, s, col_crop.width, e))
                        if col_ink_arr[s:e].mean() >= MIN_INK_RATIO:
                            all_chunks.append(chunk)

                col_chunk_ranges.append((start, len(all_chunks)))
                col_errors.append(None)
            except Exception as e:
                col_chunk_ranges.append(None)
                col_errors.append(f"[列OCRエラー: {e}]")

    # Phase 2: 1回のバッチ推論
    mocr = get_mocr()
    all_texts = ocr_batch(mocr, all_chunks)

    # Phase 3: 結果を列ごとに再構成
    lines = []
    for col_range, error in zip(col_chunk_ranges, col_errors):
        if error is not None:
            lines.append(error)
        elif col_range is None:
            lines.append("")
        else:
            start, end = col_range
            lines.append("".join(all_texts[start:end]))

    return "\n".join(lines), "ok"


# ---------------------------------------------------------------------------
# ファイル収集・ソート
# ---------------------------------------------------------------------------
def collect_images(input_dir, extensions):
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*.{ext}"))
        files.extend(input_dir.glob(f"*.{ext.upper()}"))
    return sorted(set(files))


def sort_by_number(files):
    def key(p):
        nums = re.findall(r"\d+", p.stem)
        return int(nums[0]) if nums else 0
    return sorted(files, key=key)


# ---------------------------------------------------------------------------
# Markdown フォーマット
# ---------------------------------------------------------------------------
def format_as_markdown(page_stem, text):
    stripped = text.strip()
    lines = [l for l in stripped.splitlines() if l.strip()]
    if len(stripped) < 40 and len(lines) <= 3:
        return f"<!-- page: {page_stem} -->\n\n## {' '.join(lines)}\n\n---\n"
    return f"<!-- page: {page_stem} -->\n\n{stripped}\n\n---\n"


# ---------------------------------------------------------------------------
# チェックポイント（中断・再開）
# ---------------------------------------------------------------------------
def load_checkpoint(checkpoint_path):
    """処理済みファイル名のセットを返す。"""
    if not checkpoint_path.exists():
        return set()
    return set(line for line in checkpoint_path.read_text(encoding="utf-8").splitlines() if line)


def save_checkpoint(checkpoint_path, filename):
    """処理済みファイル名を追記する。"""
    with open(checkpoint_path, "a", encoding="utf-8") as f:
        f.write(filename + "\n")


def format_eta(seconds):
    """秒数を「Xh Ym」または「Ym」形式に変換する。"""
    m = int(seconds) // 60
    h = m // 60
    if h > 0:
        return f"{h}h {m % 60:02d}m"
    return f"{m}m"


# ---------------------------------------------------------------------------
# ground_truth 比較
# ---------------------------------------------------------------------------
def normalize_text(text):
    return "".join(
        l.strip() for l in text.splitlines()
        if l.strip() and not l.strip().startswith("#")
    )


def compare_with_ground_truth(ocr_text, gt_path):
    if not gt_path.exists():
        print(f"  [比較スキップ] ground_truth ファイルが見つかりません: {gt_path}")
        return

    gt_raw = gt_path.read_text(encoding="utf-8")
    if not normalize_text(gt_raw):
        print(f"  [比較スキップ] ground_truth ファイルが空です: {gt_path}")
        return

    ocr_clean = normalize_text(ocr_text)
    gt_clean = normalize_text(gt_raw)
    ratio = difflib.SequenceMatcher(None, gt_clean, ocr_clean).ratio()

    sep = "─" * 50
    print(f"\n{sep}")
    print(f"  ground_truth 比較: {gt_path.name}")
    print(f"  文字一致率: {ratio*100:.1f}%  (正解 {len(gt_clean)} 文字 / OCR {len(ocr_clean)} 文字)")

    diff = list(difflib.unified_diff(
        gt_raw.splitlines(), ocr_text.strip().splitlines(),
        fromfile="ground_truth", tofile="OCR結果", lineterm="", n=1,
    ))
    if diff:
        print("\n  --- diff（抜粋）---")
        for line in diff[:40]:
            sys.stdout.buffer.write(("  " + line + "\n").encode("utf-8", errors="replace"))
            sys.stdout.buffer.flush()
        if len(diff) > 40:
            print(f"  ... ({len(diff)-40} 行省略)")
    else:
        print("  差異なし（完全一致）")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="日本語小説スキャン OCR（manga-ocr + 動的列検出）")
    parser.add_argument("--input",        default="./images")
    parser.add_argument("--output",       default="")
    parser.add_argument("--ext",          default="jpg,jpeg,png")
    parser.add_argument("--sort",         default="number", choices=["name", "number"])
    parser.add_argument("--test",         type=int, default=0, metavar="N")
    parser.add_argument("--offset",       type=int, default=0, metavar="N")
    parser.add_argument("--ground-truth", default="output/test/ground_truth_010.txt",
                        dest="ground_truth")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"[ERROR] フォルダが見つかりません: {input_dir}")
        sys.exit(1)

    extensions = [e.strip().lstrip(".") for e in args.ext.split(",")]
    files = collect_images(input_dir, extensions)
    if not files:
        print(f"[ERROR] 画像が見つかりません: {input_dir}  (拡張子: {extensions})")
        sys.exit(1)

    if args.sort == "number":
        files = sort_by_number(files)
    if args.offset > 0:
        files = files[args.offset:]

    test_mode = args.test > 0
    if test_mode:
        files = files[:args.test]

    out_dir = Path("output/test") if test_mode else Path("output/prod")
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else out_dir / "shinsekaiyori.md"

    # チェックポイント（テストモードでは使用しない）
    checkpoint_path = output_path.with_suffix(".checkpoint") if not test_mode else None
    done = load_checkpoint(checkpoint_path) if checkpoint_path else set()
    remaining = [f for f in files if f.name not in done]

    resuming = bool(done)
    mode_label = "（テストモード）" if test_mode else ("（再開）" if resuming else "")
    skipped = len(files) - len(remaining)

    print(f"OCRエンジン: manga-ocr（列検出失敗時はプレースホルダー出力）")
    print(f"画像枚数   : {len(remaining)} 枚{mode_label}" + (f"  ※{skipped}枚スキップ（処理済）" if skipped else ""))
    print(f"出力先     : {output_path}")
    if checkpoint_path:
        print(f"チェックポイント: {checkpoint_path}")
    print("─" * 40)

    errors, fallbacks = [], []
    last_text = ""  # ground_truth 比較用（テストモード1枚時のみ使用）
    start = time.time()
    total_pages = len(remaining)

    file_mode = "a" if resuming else "w"
    with open(output_path, file_mode, encoding="utf-8") as out:
        for i, img_path in enumerate(remaining, 1):
            try:
                text, status = ocr_page_manga(img_path)
                if status == "fallback_needed":
                    fallbacks.append(img_path.name)
                    text = "[列検出失敗: 要手動確認]"
                    flag = " [列検出失敗]"
                else:
                    flag = ""
                last_text = text
                out.write(format_as_markdown(img_path.stem, text))
                out.write("\n")
                out.flush()
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, img_path.name)
                elapsed = time.time() - start
                eta = elapsed / i * (total_pages - i)
                pct = i / total_pages * 100
                sec_per_page = elapsed / i
                print(f"  [{i:>4}/{total_pages}] {pct:4.1f}%  {img_path.name}  ({len(text)}文字){flag}"
                      f"  残り約{format_eta(eta)}  ({sec_per_page:.1f}秒/枚)")
            except Exception as e:
                print(f"  [{i:>4}/{total_pages}] 警告: {img_path.name} -> {e}")
                errors.append((img_path.name, str(e)))
                out.write(format_as_markdown(img_path.stem, f"[OCRエラー: {img_path.name}]"))
                out.write("\n")
                out.flush()
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, img_path.name)

    print("─" * 40)
    total = time.time() - start
    print(f"完了！  {total_pages}枚  {int(total//60)}分{int(total%60)}秒")
    if fallbacks:
        print(f"  フォールバック({len(fallbacks)}件): {', '.join(fallbacks[:5])}")
    if errors:
        for name, msg in errors:
            print(f"  エラー: {name}: {msg}")

    # 全ページ完了したらチェックポイントを削除
    if checkpoint_path and checkpoint_path.exists() and not errors:
        checkpoint_path.unlink()
        print("チェックポイントを削除しました（処理完了）")

    print(f"出力完了: {output_path}")

    if test_mode and total_pages == 1:
        compare_with_ground_truth(last_text, Path(args.ground_truth))


if __name__ == "__main__":
    main()
