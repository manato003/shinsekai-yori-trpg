'use client'

import { useState, useEffect } from 'react'
import type { Room } from '@/lib/database.types'
import { fetchGameSystems } from '@/lib/bcdice'
import { Badge } from '@/components/ui/badge'

interface GameSystem {
  id: string
  name: string
  sortKey: string
}

interface Props {
  room: Room
  userName: string
  gameSystem: string
  onGameSystemChange: (id: string) => void
}

export function RoomHeader({ room, userName, gameSystem, onGameSystemChange }: Props) {
  const [systems, setSystems] = useState<GameSystem[]>([])
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    fetchGameSystems().then((list) => {
      // ソートして先頭にDiceBotを持ってくる
      const sorted = list.sort((a, b) => a.sortKey.localeCompare(b.sortKey, 'ja'))
      setSystems(sorted)
    })
  }, [])

  function copyRoomId() {
    navigator.clipboard.writeText(room.id)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <header className="flex items-center gap-3 px-4 py-2 border-b border-border bg-card shrink-0">
      <div className="flex items-center gap-2 flex-1 min-w-0">
        <span className="text-primary font-bold text-lg">⚔</span>
        <h1 className="font-semibold truncate">{room.name}</h1>
        <Badge variant="outline" className="text-xs shrink-0">{userName}</Badge>
      </div>

      {/* ゲームシステム選択 */}
      <div className="flex items-center gap-2 shrink-0">
        <label className="text-xs text-muted-foreground whitespace-nowrap">システム:</label>
        <select
          value={gameSystem}
          onChange={(e) => onGameSystemChange(e.target.value)}
          className="text-xs bg-input border border-border rounded px-2 py-1 text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
        >
          <option value="DiceBot">DiceBot（汎用）</option>
          {systems.map((s) => (
            <option key={s.id} value={s.id}>{s.name}</option>
          ))}
        </select>
      </div>

      {/* ルームID共有 */}
      <button
        onClick={copyRoomId}
        className="text-xs text-muted-foreground hover:text-foreground transition-colors shrink-0 border border-border rounded px-2 py-1"
        title="ルームIDをコピー"
      >
        {copied ? '✓ コピー済み' : '🔗 ID共有'}
      </button>
    </header>
  )
}
