'use client'

import { useState } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

const COLORS = [
  '#6366f1', // indigo
  '#ef4444', // red
  '#22c55e', // green
  '#eab308', // yellow
  '#3b82f6', // blue
  '#ec4899', // pink
  '#f97316', // orange
  '#ffffff',  // white
]

interface Props {
  onAdd: (label: string, color: string) => void
  onClose: () => void
}

export function AddTokenDialog({ onAdd, onClose }: Props) {
  const [label, setLabel] = useState('')
  const [color, setColor] = useState(COLORS[0])

  function handleAdd() {
    if (label.trim()) onAdd(label.trim(), color)
  }

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-xs">
        <DialogHeader>
          <DialogTitle>トークン追加</DialogTitle>
        </DialogHeader>
        <div className="space-y-3">
          <Input
            placeholder="名前（例: 勇者、ゴブリン）"
            value={label}
            onChange={(e) => setLabel(e.target.value)}
            autoFocus
            onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
          />
          <div className="flex gap-2 flex-wrap">
            {COLORS.map((c) => (
              <button
                key={c}
                type="button"
                className={`w-7 h-7 rounded-full border-2 transition-transform ${
                  color === c ? 'border-white scale-110' : 'border-transparent opacity-70'
                }`}
                style={{ background: c }}
                onClick={() => setColor(c)}
              />
            ))}
          </div>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={onClose}>キャンセル</Button>
            <Button size="sm" disabled={!label.trim()} onClick={handleAdd}>追加</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
