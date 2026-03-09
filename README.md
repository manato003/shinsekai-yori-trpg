# TRPGセッション — Phase 1

オンラインマルチプレイTRPGセッションツール。
Cocofolia / Udonarium / Roll20 を参考に再構築したコアプラットフォーム。

## Phase 1 実装済み機能

- ルーム作成 / パスワード保護 / URL共有
- リアルタイムチャット (Supabase Realtime / WebSocket)
- BCDice対応ダイスロール (汎用フォールバック付き)
- Presence によるユーザー一覧表示
- ゲームシステム選択 (BCDice対応全システム)
- ダークテーマ UI

## セットアップ

### 1. Supabase プロジェクト作成

1. [supabase.com](https://supabase.com) でプロジェクト作成
2. SQL Editor で `supabase/schema.sql` を実行
3. Settings > API から URL と anon key を取得

### 2. 環境変数設定

`.env.local` を編集:

```
NEXT_PUBLIC_SUPABASE_URL=https://xxxxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJhbGci...
NEXT_PUBLIC_BCDICE_API_URL=https://bcdice.trpg.net
```

### 3. 開発サーバー起動

```bash
npm run dev
```

`http://localhost:3000` を開く。

## ダイスコマンド

| コマンド | 説明 |
|---|---|
| `2d6` | 2d6（汎用） |
| `1d100` | 1d100 |
| `3d6+2` | 修正値付き |
| `CC<=70` | CoC 7th 技能判定 (システム: Cthulhu7th) |

チャット入力でダイスコマンドを入力すると自動判定。

## Phase 2 予定

- [ ] キャンバス + トークン (2Dマップ)
- [ ] キャラクターシート
- [ ] BGM/SE プレイヤー
- [ ] ハンドアウト (秘匿情報)
- [ ] カード/デッキ機能
- [ ] ユーザー認証 + GM/PL権限分離
