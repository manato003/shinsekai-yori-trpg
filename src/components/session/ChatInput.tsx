'use client'

import { useState, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { isDiceCommand } from '@/lib/bcdice'

interface Props {
  onSend: (content: string) => Promise<void>
  gameSystem: string
}

const DICE_SHORTCUTS = ['2d6', '1d100', '3d6', '1d6', '2d8']

export function ChatInput({ onSend, gameSystem }: Props) {
  const [value, setValue] = useState('')
  const [sending, setSending] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const isDice = isDiceCommand(value)

  const handleSubmit = useCallback(
    async (e?: React.FormEvent) => {
      e?.preventDefault()
      const trimmed = value.trim()
      if (!trimmed || sending) return
      setSending(true)
      setValue('')
      try {
        await onSend(trimmed)
      } finally {
        setSending(false)
        inputRef.current?.focus()
      }
    },
    [value, sending, onSend]
  )

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  function insertShortcut(cmd: string) {
    setValue(cmd)
    inputRef.current?.focus()
  }

  return (
    <div className="border-t border-border bg-card px-4 py-3 shrink-0">
      {/* ショートカットダイス */}
      <div className="flex gap-1.5 mb-2 flex-wrap">
        {DICE_SHORTCUTS.map((d) => (
          <button
            key={d}
            onClick={() => insertShortcut(d)}
            className="text-xs px-2 py-0.5 rounded border border-border text-muted-foreground hover:text-primary hover:border-primary transition-colors"
          >
            {d}
          </button>
        ))}
        <span className="text-xs text-muted-foreground/40 ml-1 self-center">
          または自由入力 ({gameSystem})
        </span>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-2">
        <div className="relative flex-1">
          <Input
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={`メッセージ or ダイスコマンド (例: 2d6, ${gameSystem === 'DiceBot' ? '3d6' : 'コマンド'})`}
            className={isDice ? 'pr-16 border-primary/50 focus-visible:ring-primary/50' : ''}
            disabled={sending}
            autoComplete="off"
          />
          {isDice && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[10px] text-primary font-medium pointer-events-none">
              🎲 ダイス
            </span>
          )}
        </div>
        <Button type="submit" disabled={!value.trim() || sending} className="shrink-0">
          {sending ? '...' : isDice ? '🎲 ロール' : '送信'}
        </Button>
      </form>
    </div>
  )
}
