'use client'

import { useState, useRef, useEffect } from 'react'
import { Stage, Layer, Rect, Circle, Text, Line, Group } from 'react-konva'
import type { Token } from '@/lib/database.types'
import { AddTokenDialog } from './AddTokenDialog'

const GRID = 40

interface Props {
  tokens: Token[]
  isGM: boolean
  onTokenMove: (id: string, x: number, y: number) => void
  onTokenAdd: (label: string, color: string, x: number, y: number) => void
  onTokenDelete: (id: string) => void
}

export function CanvasPanel({ tokens, isGM, onTokenMove, onTokenAdd, onTokenDelete }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ w: 800, h: 600 })
  const [dialogPos, setDialogPos] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    const update = () => {
      if (containerRef.current) {
        setSize({ w: containerRef.current.offsetWidth, h: containerRef.current.offsetHeight })
      }
    }
    update()
    const ro = new ResizeObserver(update)
    if (containerRef.current) ro.observe(containerRef.current)
    return () => ro.disconnect()
  }, [])

  const cols = Math.ceil(size.w / GRID) + 1
  const rows = Math.ceil(size.h / GRID) + 1

  return (
    <div ref={containerRef} className="relative flex-1 overflow-hidden bg-[#12121e]">
      <Stage
        width={size.w}
        height={size.h}
        onDblClick={(e) => {
          if (!isGM) return
          if (e.target === e.target.getStage()) {
            const pos = e.target.getStage()?.getPointerPosition()
            if (pos) setDialogPos(pos)
          }
        }}
      >
        <Layer>
          {/* グリッド */}
          {Array.from({ length: cols }, (_, i) => (
            <Line key={`v${i}`} points={[i * GRID, 0, i * GRID, size.h]} stroke="#1e1e30" strokeWidth={1} />
          ))}
          {Array.from({ length: rows }, (_, i) => (
            <Line key={`h${i}`} points={[0, i * GRID, size.w, i * GRID]} stroke="#1e1e30" strokeWidth={1} />
          ))}

          {/* トークン */}
          {tokens.map((token) => (
            <Group
              key={token.id}
              x={token.x}
              y={token.y}
              draggable
              onDragEnd={(e) => onTokenMove(token.id, e.target.x(), e.target.y())}
              onContextMenu={(e) => {
                e.evt.preventDefault()
                if (isGM) onTokenDelete(token.id)
              }}
            >
              <Circle radius={token.size / 2} fill={token.color} opacity={0.9} />
              <Text
                text={token.label}
                fontSize={11}
                fill="white"
                align="center"
                width={80}
                x={-40}
                y={token.size / 2 + 3}
              />
            </Group>
          ))}
        </Layer>
      </Stage>

      {dialogPos && (
        <AddTokenDialog
          onAdd={(label, color) => {
            onTokenAdd(label, color, dialogPos.x, dialogPos.y)
            setDialogPos(null)
          }}
          onClose={() => setDialogPos(null)}
        />
      )}

      {isGM && (
        <p className="absolute bottom-2 right-2 text-xs text-muted-foreground/40 pointer-events-none">
          ダブルクリックで追加 / 右クリックで削除
        </p>
      )}
    </div>
  )
}
