'use client'

import { useEffect, useRef } from 'react'
import type { Message } from '@/lib/database.types'
import { cn } from '@/lib/utils'

interface Props {
  messages: Message[]
  currentUser: string
}

export function ChatLog({ messages, currentUser }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // 新メッセージで自動スクロール
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex-1 overflow-y-auto px-4 py-3 space-y-1.5">
      {messages.length === 0 && (
        <p className="text-muted-foreground text-sm text-center mt-10">
          チャットログがここに表示されます。<br />
          ダイスコマンド例: <code className="text-primary">2d6</code> <code className="text-primary">1d100</code>
        </p>
      )}

      {messages.map((msg) => (
        <MessageRow key={msg.id} msg={msg} isOwn={msg.user_name === currentUser} />
      ))}

      <div ref={bottomRef} />
    </div>
  )
}

function MessageRow({ msg, isOwn }: { msg: Message; isOwn: boolean }) {
  const time = new Date(msg.created_at).toLocaleTimeString('ja-JP', {
    hour: '2-digit', minute: '2-digit',
  })

  if (msg.is_dice) {
    return <DiceRow msg={msg} isOwn={isOwn} time={time} />
  }

  return (
    <div className={cn('flex gap-2 items-start group', isOwn && 'flex-row-reverse')}>
      <Avatar name={msg.user_name} />
      <div className={cn('max-w-[70%]', isOwn && 'items-end flex flex-col')}>
        <div className="flex items-baseline gap-1.5 mb-0.5">
          {!isOwn && <span className="text-xs font-medium text-muted-foreground">{msg.user_name}</span>}
          <span className="text-[10px] text-muted-foreground/60 opacity-0 group-hover:opacity-100 transition-opacity">{time}</span>
        </div>
        <div
          className={cn(
            'px-3 py-1.5 rounded-lg text-sm leading-relaxed break-words',
            isOwn
              ? 'bg-primary text-primary-foreground rounded-tr-none'
              : 'bg-secondary text-foreground rounded-tl-none'
          )}
        >
          {msg.content}
        </div>
      </div>
    </div>
  )
}

function DiceRow({ msg, isOwn, time }: { msg: Message; isOwn: boolean; time: string }) {
  const dr = msg.dice_result

  // BCDiceのテキストを解析してsuccessかfailureか判定
  const text = msg.content
  const isSuccess = /成功|クリティカル|ファンブル以外.*成功/.test(text) && !/失敗/.test(text)
  const isFailure = /失敗|ファンブル/.test(text)
  const isCritical = /クリティカル/.test(text)
  const isFumble = /ファンブル/.test(text)

  const accentColor = isCritical
    ? 'text-yellow-400 border-yellow-400/30 bg-yellow-400/10'
    : isFumble
    ? 'text-red-400 border-red-400/30 bg-red-400/10'
    : isSuccess
    ? 'text-green-400 border-green-400/30 bg-green-400/10'
    : isFailure
    ? 'text-red-400 border-red-400/30 bg-red-400/10'
    : 'text-primary border-primary/30 bg-primary/10'

  return (
    <div className="my-2 flex justify-center">
      <div className={cn('border rounded-lg px-4 py-2 max-w-md w-full text-center', accentColor)}>
        <div className="text-[10px] text-muted-foreground mb-1">
          {msg.user_name} · {time}
          {dr?.systemId && dr.systemId !== 'DiceBot' && (
            <span className="ml-1 opacity-60">[{dr.systemId}]</span>
          )}
        </div>
        <div className="font-mono text-sm font-semibold whitespace-pre-wrap break-words">
          {msg.content}
        </div>
        {dr && (
          <div className="flex justify-center gap-1 mt-1.5 flex-wrap">
            {dr.result.map((v, i) => (
              <span
                key={i}
                className="inline-flex items-center justify-center w-7 h-7 rounded border border-current/40 text-xs font-bold bg-background/50"
              >
                {v}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Avatar({ name }: { name: string }) {
  const initials = name.slice(0, 2)
  // ハッシュベースで色を決める
  const hue = (name.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0) * 47) % 360
  return (
    <div
      className="w-7 h-7 rounded-full flex items-center justify-center text-[10px] font-bold text-white shrink-0 mt-0.5"
      style={{ background: `hsl(${hue} 60% 45%)` }}
    >
      {initials}
    </div>
  )
}
