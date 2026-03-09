'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

interface Props {
  userName: string
  users: string[]
  sheets: Record<string, Record<string, string>>  // user_name -> {key: value}
  onSheetChange: (data: Record<string, string>) => void
}

export function CharacterSheet({ userName, users, sheets, onSheetChange }: Props) {
  const [viewUser, setViewUser] = useState(userName)
  const [newKey, setNewKey] = useState('')

  useEffect(() => {
    if (userName) setViewUser(userName)
  }, [userName])
  const [newVal, setNewVal] = useState('')

  const isOwn = viewUser === userName
  const data = sheets[viewUser] ?? {}

  function handleAddRow() {
    if (!newKey.trim()) return
    onSheetChange({ ...sheets[userName] ?? {}, [newKey.trim()]: newVal })
    setNewKey('')
    setNewVal('')
  }

  function handleEditValue(key: string, value: string) {
    onSheetChange({ ...sheets[userName] ?? {}, [key]: value })
  }

  function handleDeleteRow(key: string) {
    const next = { ...sheets[userName] ?? {} }
    delete next[key]
    onSheetChange(next)
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* ユーザー選択タブ */}
      {users.length > 1 && (
        <div className="flex gap-1 px-3 pt-2 flex-wrap shrink-0">
          {users.map((u) => (
            <button
              key={u}
              onClick={() => setViewUser(u)}
              className={`text-xs px-2 py-0.5 rounded transition-colors ${
                viewUser === u
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
              }`}
            >
              {u}
            </button>
          ))}
        </div>
      )}

      {/* KV表 */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {Object.keys(data).length === 0 && (
          <p className="text-xs text-muted-foreground py-4 text-center">
            {isOwn ? '下のフォームで項目を追加してください' : 'キャラクターシートは空です'}
          </p>
        )}
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground w-28 shrink-0 truncate">{key}</span>
            {isOwn ? (
              <>
                <Input
                  value={value}
                  onChange={(e) => handleEditValue(key, e.target.value)}
                  className="h-6 text-xs flex-1"
                />
                <button
                  onClick={() => handleDeleteRow(key)}
                  className="text-muted-foreground hover:text-destructive transition-colors text-xs shrink-0"
                  title="削除"
                >
                  ✕
                </button>
              </>
            ) : (
              <span className="text-xs flex-1">{value}</span>
            )}
          </div>
        ))}
      </div>

      {/* 追加フォーム（自分のシートのみ） */}
      {isOwn && (
        <div className="border-t border-border p-3 flex gap-2 shrink-0">
          <Input
            placeholder="項目名"
            value={newKey}
            onChange={(e) => setNewKey(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAddRow()}
            className="h-7 text-xs w-28"
          />
          <Input
            placeholder="値"
            value={newVal}
            onChange={(e) => setNewVal(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleAddRow()}
            className="h-7 text-xs flex-1"
          />
          <Button size="sm" onClick={handleAddRow} className="h-7 text-xs px-2">
            追加
          </Button>
        </div>
      )}
    </div>
  )
}
