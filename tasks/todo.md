# tasks/todo.md - 新世界よりTRPG OCR プロジェクト

## 現在のタスク

- [x] OCR エンジン選定（manga-ocr 採用）
- [x] 動的列検出ロジック実装
- [x] チャンク分割ロジック実装（文字間ギャップ検出）
- [x] テスト画像 010.png で 94.2% 達成
- [x] .gitignore 修正・プロジェクト規約整備
- [x] チェックポイント/再開機能実装
- [ ] 全 958 ページの一括 OCR 実行
- [ ] 出力テキストの品質確認・後処理（必要に応じて）

## 参考

- スキャン画像: `scan/(一般小説) [貴志祐介] 新世界より (講談社ノベルス・新書版)/`
- 出力先: `scan/output/prod/shinsekaiyori.md`
- ground_truth: `scan/output/test/ground_truth_010.txt`
