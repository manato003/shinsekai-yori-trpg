'use client'

import { useState, useEffect, useRef } from 'react'
import type { BgmState } from '@/lib/database.types'

interface Props {
  bgmState: BgmState | null
  isGM: boolean
  onStateChange: (updates: Partial<BgmState>) => void
}

export function BgmPlayer({ bgmState, isGM, onStateChange }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [urlInput, setUrlInput] = useState(bgmState?.url ?? '')
  const [unlocked, setUnlocked] = useState(false)

  // 外部からURLが変わったときに入力欄も更新
  useEffect(() => {
    setUrlInput(bgmState?.url ?? '')
  }, [bgmState?.url])

  // ブラウザのAutoplay Policyを回避するため、ユーザー操作で一度アンロックする
  function unlock() {
    if (!audioRef.current) {
      audioRef.current = new Audio()
      audioRef.current.loop = true
    }
    audioRef.current.play().then(() => {
      audioRef.current?.pause()
      setUnlocked(true)
    }).catch(() => {
      setUnlocked(true)
    })
  }

  // bgmStateの変化に合わせてHTML5 Audioを制御（アンロック後のみ）
  useEffect(() => {
    if (!bgmState || !unlocked) return

    if (!audioRef.current) {
      audioRef.current = new Audio()
      audioRef.current.loop = true
    }

    const audio = audioRef.current
    audio.volume = bgmState.volume

    if (bgmState.url && audio.src !== bgmState.url) {
      audio.src = bgmState.url
      audio.load()
    }

    if (bgmState.is_playing && bgmState.url) {
      audio.play().catch(() => {})
    } else {
      audio.pause()
    }
  }, [bgmState, unlocked])

  // アンマウント時に停止
  useEffect(() => {
    return () => {
      audioRef.current?.pause()
      audioRef.current = null
    }
  }, [])

  function handlePlay() {
    const url = urlInput.trim()
    if (!url) return
    onStateChange({ url, is_playing: true })
  }

  function handlePause() {
    onStateChange({ is_playing: false })
  }

  return (
    <div className="p-3 space-y-2 border-t border-border">
      <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-wider">BGM</p>

      {!unlocked ? (
        <button
          onClick={unlock}
          className="w-full text-[10px] px-2 py-1 rounded border border-border hover:bg-secondary transition-colors text-muted-foreground"
        >
          🔊 音声を有効にする
        </button>
      ) : (
        <>
          {isGM && (
            <input
              type="url"
              placeholder="音声URL（.mp3 / .ogg）"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handlePlay()}
              className="w-full text-[10px] bg-input border border-border rounded px-2 py-1 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-ring"
            />
          )}

          <div className="flex items-center gap-2">
            {isGM && (
              <button
                onClick={bgmState?.is_playing ? handlePause : handlePlay}
                className="text-xs px-2 py-0.5 rounded border border-border hover:bg-secondary transition-colors"
                title={bgmState?.is_playing ? '停止' : '再生'}
              >
                {bgmState?.is_playing ? '⏸' : '▶'}
              </button>
            )}
            {!isGM && (
              <span className="text-[10px] text-muted-foreground">
                {bgmState?.is_playing ? '▶ 再生中' : '停止中'}
              </span>
            )}

            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={bgmState?.volume ?? 0.7}
              onChange={(e) => onStateChange({ volume: parseFloat(e.target.value) })}
              className="flex-1 accent-primary"
              title="音量（自分の端末のみ）"
            />
            <span className="text-[10px] text-muted-foreground w-6 text-right">
              {Math.round((bgmState?.volume ?? 0.7) * 100)}
            </span>
          </div>
        </>
      )}
    </div>
  )
}
