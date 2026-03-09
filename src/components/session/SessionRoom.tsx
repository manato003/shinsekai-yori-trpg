'use client'

import { useEffect, useState, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import { supabase } from '@/lib/supabase'
import { isDiceCommand, rollDice } from '@/lib/bcdice'
import type { Message, Room, Token, BgmState, CharacterSheet, Handout } from '@/lib/database.types'
import { ChatLog } from './ChatLog'
import { ChatInput } from './ChatInput'
import { RoomHeader } from './RoomHeader'
import { UserPanel } from './UserPanel'
import { CharacterSheet as CharacterSheetPanel } from './CharacterSheet'
import { HandoutPanel } from './HandoutPanel'

// react-konva はブラウザ専用のためSSR無効でロード
const CanvasPanel = dynamic(
  () => import('./CanvasPanel').then((m) => ({ default: m.CanvasPanel })),
  { ssr: false }
)

type Tab = 'chat' | 'map' | 'sheet' | 'handout'

interface Props {
  roomId: string
}

export function SessionRoom({ roomId }: Props) {
  const [room, setRoom] = useState<Room | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [tokens, setTokens] = useState<Token[]>([])
  const [users, setUsers] = useState<string[]>([])
  const [userName, setUserName] = useState('')
  const [gameSystem, setGameSystem] = useState('DiceBot')
  const [bgmState, setBgmState] = useState<BgmState | null>(null)
  // sheets: user_name -> {key: value}
  const [sheets, setSheets] = useState<Record<string, Record<string, string>>>({})
  const [handouts, setHandouts] = useState<Handout[]>([])
  const [activeTab, setActiveTab] = useState<Tab>('chat')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const channelRef = useRef<ReturnType<typeof supabase.channel> | null>(null)

  useEffect(() => {
    const name = sessionStorage.getItem('trpg_user_name') ?? 'ゲスト'
    setUserName(name)
  }, [])

  // ルーム情報 + 過去メッセージ + トークン + BGM + キャラシート取得
  useEffect(() => {
    async function init() {
      setLoading(true)
      const [roomRes, msgRes, tokenRes, bgmRes, sheetRes, handoutRes] = await Promise.all([
        supabase.from('rooms').select('*').eq('id', roomId).single(),
        supabase
          .from('messages')
          .select('*')
          .eq('room_id', roomId)
          .order('created_at', { ascending: true })
          .limit(200),
        supabase.from('tokens').select('*').eq('room_id', roomId),
        supabase.from('bgm_state').select('*').eq('room_id', roomId).maybeSingle(),
        supabase.from('character_sheets').select('*').eq('room_id', roomId),
        supabase.from('handouts').select('*').eq('room_id', roomId).order('created_at', { ascending: true }),
      ])

      if (roomRes.error || !roomRes.data) {
        setError('ルームが見つかりません。URLを確認してください。')
        setLoading(false)
        return
      }

      setRoom(roomRes.data)
      setMessages(msgRes.data ?? [])
      setTokens(tokenRes.data ?? [])
      setBgmState(bgmRes.data ?? null)

      const sheetMap: Record<string, Record<string, string>> = {}
      for (const s of (sheetRes.data ?? []) as CharacterSheet[]) {
        sheetMap[s.user_name] = s.data
      }
      setSheets(sheetMap)

      setHandouts((handoutRes.data ?? []) as Handout[])

      setLoading(false)
    }
    init()
  }, [roomId])

  // Realtime: messages + Presence
  useEffect(() => {
    if (!userName) return

    const channel = supabase.channel(`room:${roomId}`, {
      config: { presence: { key: userName } },
    })

    channelRef.current = channel

    channel.on(
      'postgres_changes',
      { event: 'INSERT', schema: 'public', table: 'messages', filter: `room_id=eq.${roomId}` },
      (payload) => {
        setMessages((prev) => [...prev, payload.new as Message])
      }
    )

    channel.on('presence', { event: 'sync' }, () => {
      const state = channel.presenceState<{ name: string }>()
      const names = Object.values(state)
        .flat()
        .map((u) => u.name)
        .filter((v, i, a) => a.indexOf(v) === i)
      setUsers(names)
    })

    channel.subscribe(async (status) => {
      if (status === 'SUBSCRIBED') {
        await channel.track({ name: userName })
      }
    })

    return () => {
      channel.untrack()
      supabase.removeChannel(channel)
    }
  }, [roomId, userName])

  // Realtime: bgm_state
  useEffect(() => {
    const ch = supabase.channel(`bgm:${roomId}`)
    ch.on(
      'postgres_changes',
      { event: '*', schema: 'public', table: 'bgm_state', filter: `room_id=eq.${roomId}` },
      (payload) => {
        if (payload.eventType === 'INSERT' || payload.eventType === 'UPDATE') {
          setBgmState(payload.new as BgmState)
        }
      }
    ).subscribe()
    return () => { supabase.removeChannel(ch) }
  }, [roomId])

  // Realtime: tokens
  useEffect(() => {
    const ch = supabase.channel(`tokens:${roomId}`)
    ch.on(
      'postgres_changes',
      { event: '*', schema: 'public', table: 'tokens', filter: `room_id=eq.${roomId}` },
      (payload) => {
        if (payload.eventType === 'INSERT') {
          setTokens((prev) => [...prev, payload.new as Token])
        } else if (payload.eventType === 'UPDATE') {
          setTokens((prev) =>
            prev.map((t) => (t.id === (payload.new as Token).id ? (payload.new as Token) : t))
          )
        } else if (payload.eventType === 'DELETE') {
          const deletedId = (payload.old as Partial<Token>).id
          if (deletedId) setTokens((prev) => prev.filter((t) => t.id !== deletedId))
        }
      }
    ).subscribe()
    return () => { supabase.removeChannel(ch) }
  }, [roomId])

  // Realtime: handouts
  useEffect(() => {
    const ch = supabase.channel(`handouts:${roomId}`)
    ch.on(
      'postgres_changes',
      { event: 'INSERT', schema: 'public', table: 'handouts', filter: `room_id=eq.${roomId}` },
      (payload) => {
        setHandouts((prev) => [...prev, payload.new as Handout])
      }
    ).subscribe()
    return () => { supabase.removeChannel(ch) }
  }, [roomId])

  // Realtime: character_sheets
  useEffect(() => {
    const ch = supabase.channel(`sheets:${roomId}`)
    ch.on(
      'postgres_changes',
      { event: '*', schema: 'public', table: 'character_sheets', filter: `room_id=eq.${roomId}` },
      (payload) => {
        if (payload.eventType === 'INSERT' || payload.eventType === 'UPDATE') {
          const s = payload.new as CharacterSheet
          setSheets((prev) => ({ ...prev, [s.user_name]: s.data }))
        }
      }
    ).subscribe()
    return () => { supabase.removeChannel(ch) }
  }, [roomId])

  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || !userName) return

      const isDice = isDiceCommand(content)

      if (isDice) {
        const result = await rollDice(content.trim(), gameSystem)
        if (!result) {
          await supabase.from('messages').insert({
            room_id: roomId,
            user_name: userName,
            content,
            is_dice: false,
          })
          return
        }

        await supabase.from('messages').insert({
          room_id: roomId,
          user_name: userName,
          content: result.text,
          is_dice: true,
          dice_result: {
            command: result.command,
            result: result.rands.map((r) => r.value),
            total: result.total,
            text: result.text,
            systemId: gameSystem,
          },
          secret: result.secret,
        })
      } else {
        await supabase.from('messages').insert({
          room_id: roomId,
          user_name: userName,
          content,
          is_dice: false,
        })
      }
    },
    [roomId, userName, gameSystem]
  )

  const handleTokenAdd = useCallback(
    async (label: string, color: string, x: number, y: number) => {
      await supabase.from('tokens').insert({ room_id: roomId, label, color, x, y })
    },
    [roomId]
  )

  const handleTokenMove = useCallback(async (id: string, x: number, y: number) => {
    setTokens((prev) => prev.map((t) => (t.id === id ? { ...t, x, y } : t)))
    await supabase.from('tokens').update({ x, y }).eq('id', id)
  }, [])

  const handleTokenDelete = useCallback(async (id: string) => {
    await supabase.from('tokens').delete().eq('id', id)
  }, [])

  const handleBgmChange = useCallback(
    async (updates: Partial<BgmState>) => {
      const next = {
        ...(bgmState ?? { url: null, is_playing: false, volume: 0.7 }),
        ...updates,
        room_id: roomId,
        updated_at: new Date().toISOString(),
      }
      setBgmState(next as BgmState)
      await supabase.from('bgm_state').upsert(next)
    },
    [bgmState, roomId]
  )

  const handleHandoutSend = useCallback(
    async (title: string, content: string, targetUser: string | null) => {
      await supabase.from('handouts').insert({
        room_id: roomId,
        title,
        content,
        from_user: userName,
        target_user: targetUser,
      })
    },
    [roomId, userName]
  )

  const handleSheetChange = useCallback(
    async (data: Record<string, string>) => {
      setSheets((prev) => ({ ...prev, [userName]: data }))
      await supabase.from('character_sheets').upsert({
        room_id: roomId,
        user_name: userName,
        data,
        updated_at: new Date().toISOString(),
      })
    },
    [roomId, userName]
  )

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-muted-foreground">
        読み込み中...
      </div>
    )
  }

  if (error || !room) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-destructive">{error || 'エラーが発生しました'}</p>
      </div>
    )
  }

  const isGM = !!room.gm_name && room.gm_name === userName

  const TABS = [
    { id: 'chat',    label: '💬 チャット' },
    { id: 'map',     label: '🗺 マップ' },
    { id: 'sheet',   label: '📋 キャラ' },
    { id: 'handout', label: '📜 HO' },
  ] as const

  return (
    <div className="h-screen flex flex-col overflow-hidden">
      <RoomHeader room={room} userName={userName} gameSystem={gameSystem} onGameSystemChange={setGameSystem} />

      {/* タブ */}
      <div className="flex gap-1 px-3 py-1.5 border-b border-border bg-card shrink-0">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`text-xs px-3 py-1 rounded transition-colors ${
              activeTab === tab.id
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div className="flex flex-1 overflow-hidden">
        {activeTab === 'chat' && (
          <div className="flex flex-col flex-1 overflow-hidden">
            <ChatLog messages={messages} currentUser={userName} />
            <ChatInput onSend={sendMessage} gameSystem={gameSystem} />
          </div>
        )}
        {activeTab === 'map' && (
          <CanvasPanel
            tokens={tokens}
            isGM={isGM}
            onTokenAdd={handleTokenAdd}
            onTokenMove={handleTokenMove}
            onTokenDelete={handleTokenDelete}
          />
        )}
        {activeTab === 'sheet' && (
          <CharacterSheetPanel
            userName={userName}
            users={users}
            sheets={sheets}
            onSheetChange={handleSheetChange}
          />
        )}
        {activeTab === 'handout' && (
          <HandoutPanel
            userName={userName}
            users={users}
            isGM={isGM}
            handouts={handouts.filter(
              (h) => h.target_user === null || h.target_user === userName || h.from_user === userName
            )}
            onSend={handleHandoutSend}
          />
        )}

        <UserPanel users={users} roomId={roomId} isGM={isGM} bgmState={bgmState} onBgmChange={handleBgmChange} />
      </div>
    </div>
  )
}
