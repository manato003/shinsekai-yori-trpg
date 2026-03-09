'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { Handout } from '@/lib/database.types'

interface Props {
  userName: string
  users: string[]
  handouts: Handout[]
  isGM: boolean
  onSend: (title: string, content: string, targetUser: string | null) => void
}

export function HandoutPanel({ userName, users, handouts, isGM, onSend }: Props) {
  const [title, setTitle] = useState('')
  const [content, setContent] = useState('')
  const [target, setTarget] = useState<string>('__all__')
  const [selected, setSelected] = useState<Handout | null>(null)

  function handleSend() {
    if (!title.trim() || !content.trim()) return
    onSend(title.trim(), content.trim(), target === '__all__' ? null : target)
    setTitle('')
    setContent('')
    setTarget('__all__')
  }

  const others = users.filter((u) => u !== userName)

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {selected ? (
        /* ハンドアウト詳細表示 */
        <div className="flex flex-col flex-1 overflow-hidden p-4 space-y-3">
          <button
            onClick={() => setSelected(null)}
            className="text-xs text-muted-foreground hover:text-foreground self-start"
          >
            ← 一覧に戻る
          </button>
          <h2 className="font-semibold">{selected.title}</h2>
          <p className="text-xs text-muted-foreground">
            配信者: {selected.from_user} → {selected.target_user ?? '全員'}
          </p>
          <div className="flex-1 overflow-y-auto bg-secondary/30 rounded p-3 text-sm whitespace-pre-wrap">
            {selected.content}
          </div>
        </div>
      ) : (
        <div className="flex flex-col flex-1 overflow-hidden">
          {/* ハンドアウト一覧 */}
          <div className="flex-1 overflow-y-auto p-3 space-y-1">
            {handouts.length === 0 && (
              <p className="text-xs text-muted-foreground py-4 text-center">
                ハンドアウトはありません
              </p>
            )}
            {handouts.map((ho) => (
              <button
                key={ho.id}
                onClick={() => setSelected(ho)}
                className="w-full text-left px-3 py-2 rounded hover:bg-secondary/50 transition-colors"
              >
                <p className="text-xs font-medium truncate">{ho.title}</p>
                <p className="text-[10px] text-muted-foreground">
                  {ho.from_user} → {ho.target_user ?? '全員'}
                </p>
              </button>
            ))}
          </div>

          {/* 配信フォーム（GMのみ） */}
          {isGM && <div className="border-t border-border p-3 space-y-2 shrink-0">
            <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
              ハンドアウト配信
            </p>
            <Input
              placeholder="タイトル"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              className="h-7 text-xs"
            />
            <textarea
              placeholder="内容"
              value={content}
              onChange={(e) => setContent(e.target.value)}
              rows={3}
              className="w-full text-xs bg-input border border-border rounded px-2 py-1 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring resize-none"
            />
            <div className="flex items-center gap-2">
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="flex-1 text-xs bg-input border border-border rounded px-2 py-1 text-foreground focus:outline-none"
              >
                <option value="__all__">全員に公開</option>
                {others.map((u) => (
                  <option key={u} value={u}>{u} に秘匿</option>
                ))}
              </select>
              <Button
                size="sm"
                onClick={handleSend}
                disabled={!title.trim() || !content.trim()}
                className="h-7 text-xs"
              >
                配信
              </Button>
            </div>
          </div>}
        </div>
      )}
    </div>
  )
}
