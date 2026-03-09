# trpg-session — プロジェクト概要

オンラインマルチプレイTRPGセッションプラットフォーム。
Cocofolia / Udonarium / Roll20 を参考に再構築。

## 技術スタック

- **フレームワーク**: Next.js 15 (App Router) + TypeScript
- **DB/Realtime**: Supabase (PostgreSQL + WebSocket Realtime + Presence)
- **UI**: Tailwind CSS v4 + shadcn/ui
- **ダイス**: BCDice API (`https://bcdice.trpg.net/v2/`) + 汎用フォールバック

## ディレクトリ構成

```
src/
├── app/
│   ├── page.tsx                  # ランディング（ルーム作成/入室）
│   └── room/[roomId]/page.tsx    # セッションページ
├── components/session/
│   ├── SessionRoom.tsx           # メインコンテナ（Realtime購読）
│   ├── ChatLog.tsx               # チャットログ表示
│   ├── ChatInput.tsx             # 入力欄 + ダイスショートカット
│   ├── RoomHeader.tsx            # ヘッダー（システム選択/ID共有）
│   └── UserPanel.tsx             # ユーザー一覧（Presence）
└── lib/
    ├── supabase.ts               # Supabaseクライアント
    ├── database.types.ts         # 型定義（rooms/messages）
    └── bcdice.ts                 # BCDice APIクライアント
```

## 環境変数 (`.env.local`)

```
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
NEXT_PUBLIC_BCDICE_API_URL=https://bcdice.trpg.net
```

## データモデル

- `rooms`: id (uuid), name, password (nullable), created_at
- `messages`: id, room_id, user_name, content, is_dice, dice_result (jsonb), secret, created_at

## 現在の作業状況

`tasks/todo.md` を参照。
