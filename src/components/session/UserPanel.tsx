'use client'

import { useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { BgmPlayer } from './BgmPlayer'
import type { BgmState } from '@/lib/database.types'

interface Props {
  users: string[]
  roomId: string
  isGM: boolean
  bgmState: BgmState | null
  onBgmChange: (updates: Partial<BgmState>) => void
}

export function UserPanel({ users, roomId, isGM, bgmState, onBgmChange }: Props) {
  const [collapsed, setCollapsed] = useState(false)

  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="w-8 border-l border-border bg-card flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors"
        title="パネルを開く"
      >
        <span className="text-xs rotate-90">▶</span>
      </button>
    )
  }

  return (
    <aside className="w-48 border-l border-border bg-card flex flex-col shrink-0">
      <div className="flex items-center justify-between px-3 py-2 border-b border-border">
        <span className="text-xs font-medium text-muted-foreground">参加者</span>
        <div className="flex items-center gap-1">
          <Badge variant="secondary" className="text-xs h-4 px-1">{users.length}</Badge>
          <button
            onClick={() => setCollapsed(true)}
            className="text-muted-foreground hover:text-foreground transition-colors ml-1"
            title="パネルを閉じる"
          >
            <span className="text-xs">✕</span>
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-2 space-y-1">
        {users.length === 0 ? (
          <p className="text-xs text-muted-foreground p-2">接続中...</p>
        ) : (
          users.map((name) => (
            <div key={name} className="flex items-center gap-2 px-2 py-1 rounded hover:bg-secondary/50">
              <UserDot name={name} />
              <span className="text-xs truncate">{name}</span>
            </div>
          ))
        )}
      </div>

      <Separator />

      {/* BGMプレイヤー */}
      <BgmPlayer bgmState={bgmState} isGM={isGM} onStateChange={onBgmChange} />

      {/* ルームID表示 */}
      <div className="p-3 border-t border-border">
        <p className="text-[10px] text-muted-foreground mb-1">ルームID</p>
        <code className="text-[9px] text-muted-foreground break-all leading-tight block">
          {roomId}
        </code>
      </div>
    </aside>
  )
}

function UserDot({ name }: { name: string }) {
  const hue = (name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) * 47) % 360
  return (
    <div
      className="w-2 h-2 rounded-full shrink-0"
      style={{ background: `hsl(${hue} 60% 55%)` }}
    />
  )
}
