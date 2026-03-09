# CLAUDE.md - ScanOcrProject

このファイルは新しいチャットセッション開始時に必ず読み込むこと。
共通ワークフロー・原則は親ディレクトリの `~/.claude/CLAUDE.md` に定義されており、自動適用される。

---

## プロジェクト固有のルール

### OCR スクリプト (`scan/ocr_novel.py`)
- 実行環境: `scan/.venv/Scripts/python ocr_novel.py`（uv 管理の Python 3.11）
- OCR エンジン: manga-ocr（バッチ推論）。列検出失敗時はプレースホルダー出力。
- GPU: RTX 4070 Ti (CUDA)。1枚約2.5秒（バッチ推論）。
- レイアウト: 縦書き2段組、右→左列順、上→下読み
- チェックポイント: 処理済みファイルを `.checkpoint` ファイルに追記。中断後の再実行で続きから再開。

### よく使うコマンド
```bash
cd C:/Dev/projects/ScanOcrProject/scan

# テスト（1枚 + ground_truth 比較）
.venv/Scripts/python ocr_novel.py \
  --input "(一般小説) [貴志祐介] 新世界より (講談社ノベルス・新書版)" \
  --test 1 --offset 7 --ext png

# 全体処理（958枚、中断・再開可能）
.venv/Scripts/python ocr_novel.py \
  --input "(一般小説) [貴志祐介] 新世界より (講談社ノベルス・新書版)" \
  --ext png
```

---

## セッション開始時のチェックリスト（省略不可）

- [ ] このファイルを読む
- [ ] `C:/Dev/claude/docs/lessons-global.md`（汎用教訓・全PJ共通）を読む
- [ ] `tasks/lessons.md`（このPJ固有の教訓）を読む

---

## タスク管理ファイル

| ファイル | 用途 |
|---|---|
| `tasks/todo.md` | 実装チェックリスト |
| `tasks/lessons.md` | PJ固有の教訓 |
