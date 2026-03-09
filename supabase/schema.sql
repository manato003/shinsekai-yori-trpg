-- TRPG Session Platform — Supabase Schema
-- Run this in Supabase SQL Editor

-- ============================================================
-- rooms
-- ============================================================
create table if not exists rooms (
  id          uuid      default gen_random_uuid() primary key,
  name        text      not null,
  password    text      default null,  -- null = no password
  created_at  timestamptz default now()
);

-- ============================================================
-- messages
-- ============================================================
create table if not exists messages (
  id          uuid      default gen_random_uuid() primary key,
  room_id     uuid      references rooms(id) on delete cascade not null,
  user_name   text      not null,
  content     text      not null,
  is_dice     boolean   default false,
  dice_result jsonb     default null,
  -- { command: "2d6", result: [3,4], total: 7, text: "2D6 → 3+4 = 7" }
  secret      boolean   default false,   -- GMのみ表示
  created_at  timestamptz default now()
);

-- index for efficient room-based queries
create index if not exists messages_room_id_idx on messages (room_id, created_at asc);

-- ============================================================
-- Realtime: enable for messages
-- ============================================================
alter publication supabase_realtime add table messages;

-- ============================================================
-- RLS policies (open for now — add auth later)
-- ============================================================
alter table rooms    enable row level security;
alter table messages enable row level security;

-- Anyone can read/write rooms and messages (for MVP, no auth)
create policy "allow all rooms"    on rooms    for all using (true) with check (true);
create policy "allow all messages" on messages for all using (true) with check (true);
